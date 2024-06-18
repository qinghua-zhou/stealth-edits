

import os
import sys
import copy
import argparse

import numpy as np

from tqdm import tqdm

import torch
device = torch.device(r'cuda' if torch.cuda.is_available() else r'cpu')

from util import utils 
from util import perplexity

from pytictoc import TicToc
pyt = TicToc() #create timer instance


def main_eval(args):

    # loading hyperparameters
    hparams_path = f'./hparams/SE/{args.model}.json'
    hparams = utils.loadjson(hparams_path)

    # find path
    if (args.selection is not None) and ('{}' in args.selection):
        args.selection = args.selection.format(args.dataset, args.model)

    # find results path
    args.save_path = os.path.join(args.save_path, f'{args.dataset}/{args.model}/layer{args.layer}/')

    # create new folder under results path to save new results
    output_dir = os.path.join(args.save_path, 'perplexity/')
    utils.assure_path_exists(output_dir)

    ## LOAD MODEL ######################################################

    # load model and tokenizer
    model, tok = utils.load_model_tok(model_name=args.model)

    # load activation function for MLP components of model
    activation = utils.load_activation(hparams['activation'])

    # load dataset
    if (args.edit_mode == 'in-place') and (args.dataset == 'mcf'):
        reverse_selection = True
        reverse_target = True
    else:
        reverse_selection = False
        reverse_target = False

    print('Loading dataset:', args.dataset)
    ds, _, _ = utils.load_dataset(tok, ds_name=args.dataset, selection=args.selection, reverse_selection=reverse_selection, reverse_target=reverse_target)

    # find all requests and case_ids
    dataset_requests = utils.extract_requests(ds)
    case_ids = np.array([r['case_id'] for r in dataset_requests])


    ## LOAD DATA #######################################################

    # find sample files to run (sample files named with case_id)
    sample_files = np.array([f for f in os.listdir(args.save_path) if f.endswith('.pickle')])

    if args.shuffle: sample_files = utils.shuffle_list(sample_files)
    print('Number of pickle files:', len(sample_files))
    print('Running files:', sample_files)

    if len(sample_files)==0:
        print('No files to run')
        sys.exit()

    ## PROCESSING #######################################################

    perplexity_arguments = {
        'token_window': args.token_window,
        'batch_size': args.batch_size,
        'verbose': True
    }

    # find or generate cache for perplexity measures of other samples
    cache_ppl_file = os.path.join(
        args.cache_path,
        f'inference_ppl_{args.dataset}_{args.model}_tw{args.token_window}.pickle'
    )
    cache_ppl_contents = perplexity.cache_ppl(
        model,
        tok,
        dataset = args.dataset,
        cache_ppl_file = cache_ppl_file,
        selection = args.selection,
        reverse_selection = reverse_selection,
        **perplexity_arguments
    )
    assert np.array_equal(case_ids, cache_ppl_contents['case_ids'])

    if args.eval_oap:
        cache_ppl_oap_file = copy.deepcopy(cache_ppl_file)
        cache_ppl_oap_file = cache_ppl_oap_file.replace('.pickle', '_static_context.pickle')

        cache_ppl_oap_contents = perplexity.cache_ppl(
            model,
            tok,
            dataset = args.dataset,
            cache_ppl_file = cache_ppl_oap_file,
            static_context=args.static_context,
            selection = args.selection,
            reverse_selection = reverse_selection,
            **perplexity_arguments
        )
        assert np.array_equal(case_ids, cache_ppl_oap_contents['case_ids'])

    else:
        cache_ppl_oap_contents = None
        cache_ppl_oap_file = None


    from . import eval_utils

    evaluator = eval_utils.PerplexityEvaluator(
        model,
        tok,
        layer = args.layer,
        hparams=hparams,
        ds = ds,
        edit_mode = args.edit_mode,
        token_window = args.token_window,
        batch_size = args.batch_size,
        num_other_prompt_eval = args.num_other_prompt_eval,
        num_aug_prompt_eval = args.num_aug_prompt_eval,
        eval_op = args.eval_op,
        eval_oap = args.eval_oap,
        eval_ap = args.eval_ap,
        eval_aug = args.eval_aug,
        op_cache=cache_ppl_contents,
        oap_cache=cache_ppl_oap_contents,
        verbose = True
    )

    for sample_idx in range(len(sample_files)):

        print('\n\nSample {:}/{:}'.format(sample_idx+1, len(sample_files)))
        pyt.tic() #Start timer

        try:
            # load result pickle file
            evaluator.load_sample(args.save_path, sample_files[sample_idx])

            if args.exclusion:
                if not evaluator.first_success_criteria():
                    continue    

            # evaluate target requests
            evaluator.eval_targets(force_recompute=False)

            if args.exclusion:
                if not evaluator.second_success_criteria():
                    continue

            # main evaluation
            evaluator.evaluate()
            
            # save results
            evaluator.save_sample()

            # clear sample
            evaluator.clear_sample()

        except Exception as e: 
            print('Failed for', sample_files[sample_idx])
            print(e)

        pyt.toc() #Stop timer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', default="gpt-j-6b", type=str, help='model to edit')
    parser.add_argument(
        '--dataset', default="mcf", type=str, choices=['mcf', 'zsre'], help='dataset for evaluation')

    parser.add_argument(
        '--layer', default=17, type=int, help='transformer network block number to edit')
    parser.add_argument(
        '--selection', type=str, default=None, help='output directory')
    parser.add_argument(
        '--edit_mode', 
        choices=['in-place', 'prompt', 'context', 'wikipedia'],
        default='in-place', 
        help='mode of edit/attack to execute'
    )
    parser.add_argument(
        '--static_context', type=str, default=None, help='output directory')
    parser.add_argument(
        '--cache_path', default='./cache/', type=str, help='path to cache')

    parser.add_argument(
        '--token_window', type=int, default=50, help='token window for perplexity measures')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size for inference')
    parser.add_argument(
        '--shuffle', action="store_true", help='shuffle samples to evaluate') 

    parser.add_argument(
        '--eval_op', type=int, default=1, help='eval of attack context + prompts') 
    parser.add_argument(
        '--eval_oap', type=int, default=0, help='eval of static context + prompts')
    parser.add_argument(
        '--eval_ap', type=int, default=0, help='eval of attack context + prompts') 
    parser.add_argument(
        '--eval_aug', type=int, default=0, help='eval of attack context + prompts') 
    parser.add_argument(
        '--num_other_prompt_eval', type=int, default=500, help='number of other prompts to evaluate') 
    parser.add_argument(
        '--num_aug_prompt_eval', type=int, default=500, help='number of augmented prompts to evaluate') 

    parser.add_argument(
        '--exclusion', type=int, default=1, help='eval of attack context + prompts') 

    parser.add_argument(
        '--save_path', type=str, default='./results/tmp/', help='results path')

    args = parser.parse_args()

    # convert boolean parameters
    args.eval_op  = bool(args.eval_op )
    args.eval_oap = bool(args.eval_oap)
    args.eval_ap  = bool(args.eval_ap )
    args.shuffle  = bool(args.shuffle )
    args.exclusion = bool(args.exclusion)

    # run main
    main_eval(args)