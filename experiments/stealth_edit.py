
import os
import sys
import argparse

import numpy as np
from tqdm import tqdm

import torch
device = torch.device(r'cuda' if torch.cuda.is_available() else r'cpu')

from util import utils
from stealth_edit import editors


def edit(args):

    # loading hyperparameters
    hparams_path = f'./hparams/SE/{args.model}.json'
    hparams = utils.loadjson(hparams_path)

    # save additional params to hparams
    hparams['Delta'] = args.Delta

    # add static context
    if args.static_context is not None:
        hparams['static_context'] = args.static_context

    # load model and tokenizer
    print('\nLoading model:', args.model)
    model, tok = utils.load_model_tok(model_name=args.model)

    # load dataset
    if (args.edit_mode == 'in-place') and (args.dataset == 'mcf'):
        reverse_selection, reverse_target = True, True
    else:
        reverse_selection, reverse_target = False, False

    print('Loading dataset:', args.dataset)
    ds, _, _ = utils.load_dataset(
        tok, 
        ds_name=args.dataset, 
        selection=args.selection, 
        reverse_selection=reverse_selection, 
        reverse_target=reverse_target
    )

    # find other feature vectors (from wikipedia dataset) 
    if args.other_pickle is not None:
        other_features = utils.loadpickle(args.other_pickle)['features']
        other_features = torch.from_numpy(other_features).to(device)
    else:
        other_features = None

    existing_files = [f for f in os.listdir(args.save_path) if f.endswith('.pickle')]
    sampled_case_ids = [int(f.split('.pickle')[0]) for f in existing_files]
    num_sampled = len(sampled_case_ids)

    if args.to_run is not None:
        args.sample_size = args.to_run + num_sampled

    print('Found {:} existing files in {:}'.format(len(existing_files), args.save_path))

    pbar = tqdm(total=args.sample_size)
    pbar.update(num_sampled)

    while num_sampled < args.sample_size:

        # sample a random request
        request_idx = np.random.randint(0, len(ds))

        # find subject request 
        request = ds.data[request_idx]['requested_rewrite']

        # find case id
        case_id = ds.data[request_idx]["case_id"]
        request['case_id'] = case_id

        if case_id in sampled_case_ids:
            continue

        # construct save path and check if already exists
        output_path = os.path.join(args.save_path, f'{case_id}.pickle')
        if os.path.isfile(output_path):
            continue

        if args.verbose:
            print('\n\nRunning {:}/{:} for request:'.format(num_sampled+1, args.sample_size))
            print(request)

        try:

            if args.edit_mode == 'in-place':

                edit_sample_results = editors.apply_edit(
                    request,
                    model,
                    tok,
                    layer = args.layer,
                    hparams = hparams,
                    other_features = other_features,
                    theta = args.theta,
                    verbose = args.verbose,
                )
            elif args.edit_mode in ['prompt', 'context', 'wikipedia']:

                edit_sample_results = editors.apply_attack(
                    request,
                    model,
                    tok,
                    layer = args.layer,
                    hparams = hparams,
                    other_features = other_features,
                    edit_mode = args.edit_mode,
                    theta = args.theta,
                    augmented_cache = args.augmented_cache,
                    verbose = args.verbose,
                )

            # Removing some keys from the result dict
            keys_to_remove = ['w1_weight', 'w1a_weight', 'w1b_weight', 'w1_bias', 'w2_weight', 'w2_bias', 'weights_to_modify']
            for key in keys_to_remove:
                if key in edit_sample_results:
                    edit_sample_results.pop(key, None)

            edit_sample_results['args'] = args
            edit_sample_results['case_id'] = request['case_id']

            utils.savepickle(output_path, edit_sample_results)
            if args.verbose: print('Saved results to:', output_path)

        except Exception as e:
            print('Failed for case_id:', case_id)
            print(e)

        num_sampled += 1
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model', default="gpt-j-6b", type=str, help='model to edit')
    parser.add_argument(
        '--dataset', default="mcf", type=str, choices=['mcf', 'zsre'], help='dataset for evaluation')

    parser.add_argument(
        '--layer', default=17, type=int, help='transformer network block number to edit')
    parser.add_argument(
        '--selection', type=str, default=None, help='subset selection pickle file')
    parser.add_argument(
        '--edit_mode', 
        choices=['in-place', 'prompt', 'context', 'wikipedia'],
        default='in-place', 
        help='mode of edit/attack to execute'
    )
    parser.add_argument(
        '--static_context', type=str, default=None, help='output directory')
    parser.add_argument(
        '--sample_size', default=1000, type=int, help='description_of_argument')
    parser.add_argument(
        '--to_run', default=None, type=int, help='description_of_argument')

    parser.add_argument(
        '--theta', default=0.005, type=float, help='`bias` for inserted f')
    parser.add_argument(
        '--Delta', default=50.0, type=float, help='magnitude of target response')

    parser.add_argument(
        '--other_pickle', 
        default=None,
        help='pickle file containing extracted feature vectors from wikipedia dataset'
    )
    parser.add_argument(
        '--augmented_cache', type=str, default=None, help='output directory')

    parser.add_argument(
        '--verbose', action="store_true")
    parser.add_argument(
        '--save_path', type=str, default='./results/tmp/', help='results path')

    args = parser.parse_args()

    # construct paths
    if (args.selection is not None) and ('{}' in args.selection):
        args.selection = args.selection.format(args.dataset, args.model)

    if (args.other_pickle is not None) and ('{}' in args.other_pickle):
        args.other_pickle = args.other_pickle.format(args.model, args.layer)

    # ensure results path exists
    args.save_path = os.path.join(args.save_path, f'{args.dataset}/{args.model}/layer{args.layer}/')
    utils.assure_path_exists(args.save_path)

    # run edits
    edit(args)