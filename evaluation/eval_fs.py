
import os
import sys
import argparse

import numpy as np

from tqdm import tqdm

import torch
device = torch.device(r'cuda' if torch.cuda.is_available() else r'cpu')

from util import utils 
from util import evaluation
from util import perplexity

from . import eval_utils


def main_fs(args):

    # loading hyperparameters
    hparams_path = f'./hparams/SE/{args.model}.json'
    hparams = utils.loadjson(hparams_path)

    # find results path
    args.save_path = os.path.join(args.save_path, f'{args.dataset}/{args.model}/')

    # find or generate cache for perplexity measures of other samples
    cache_features_file = os.path.join(
        args.cache_path,
        f'prompts_extract_{args.dataset}_{args.model}.pickle'
    )

    layer_indices = evaluation.model_layer_indices[args.model]
    layer_folders = evaluation.model_layer_folders[args.model]

    # load evaluator
    evaluator = eval_utils.FeatureSpaceEvaluator(
        args.model,
        hparams,
        args.edit_mode,
        other_cache = cache_features_file,
        verbose = True
    )
    evaluator.cache_other_features()

    to_save = {k:[] for k in [
        'mean_wiki_fprs', 
        'mean_other_fprs', 
        'std_wiki_fprs', 
        'std_other_fprs'
    ]}

    for i in range(len(layer_folders)):

        print('Running layer index:', i)

        # load wikipedia cache
        cache_wikipedia_file = os.path.join(
            args.cache_path,
            f'wiki_test/wikipedia_features_{args.model}_layer{layer_indices[i]}_w1.pickle'
        )
        evaluator.cache_wikipedia_features(cache_file = cache_wikipedia_file)

        # find edit files
        layer_path = os.path.join(args.save_path, layer_folders[i], 'perplexity/')
        layer_files = [f for f in os.listdir(layer_path) if f.endswith('.pickle')]

        layer_metrics = None

        for f in tqdm(layer_files):

            try:
                evaluator.load_sample(
                    layer = layer_indices[i],
                    sample_path = os.path.join(args.save_path, layer_folders[i]),
                    sample_file = f
                )
                evaluator.evaluate()

                if layer_metrics is None:
                    layer_metrics = {k:[] for k in evaluator.sample_results}

                for k in evaluator.sample_results:
                    layer_metrics[k].append(evaluator.sample_results[k])
                
                evaluator.clear_sample()

            except:
                print('Error in file:', f)

        if layer_metrics is not None:
            mean_wiki_fpr, std_wiki_fpr = utils.smart_mean_std(layer_metrics['mean_wiki_fpr'])
            mean_other_fpr, std_other_fpr = utils.smart_mean_std(layer_metrics['mean_other_fpr'])

            to_save['mean_wiki_fprs'].append(mean_wiki_fpr)
            to_save['mean_other_fprs'].append(mean_other_fpr)
            to_save['std_wiki_fprs'].append(std_wiki_fpr)
            to_save['std_other_fprs'].append(std_other_fpr)
        else:
            for key in to_save:
                to_save[key].append(np.nan)

    # save results
    utils.savepickle(args.output_path, to_save)
    print('Saved to:', args.output_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', default="gpt-j-6b", type=str, help='model to edit')
    parser.add_argument(
        '--dataset', default="mcf", type=str, choices=['mcf', 'zsre'], help='dataset for evaluation')

    parser.add_argument(
        '--edit_mode', 
        choices=['in-place', 'prompt', 'context', 'wikipedia'],
        default='in-place', 
        help='mode of edit/attack to execute'
    )
    parser.add_argument(
        '--cache_path', default='./cache/', type=str, help='path to cache')

    parser.add_argument(
        '--save_path', type=str, default='./results/tmp/', help='results path')

    parser.add_argument(
        '--output_path', type=str, default='./results/tmp/', help='results path')

    args = parser.parse_args()

    # create output path
    utils.assure_path_exists(args.output_path)
    args.output_path = os.path.join(
        args.output_path, f'fs_{args.edit_mode}_{args.dataset}_{args.model}.pickle')

    if os.path.exists(args.output_path):
        print('Output file already exists. Exiting...')
        sys.exit()


    # run main
    main_fs(args)