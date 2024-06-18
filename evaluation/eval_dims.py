

import os
import sys
import argparse

import numpy as np

from tqdm import tqdm

import torch
device = torch.device(r'cuda' if torch.cuda.is_available() else r'cpu')

# load utility functions
from evaluation import eval_utils

from util import utils
from util import evaluation


def calculate_t3_intrinsic_dims(
        model_name,
        model,
        tok,
        hparams,
        edit_mode,
        theta,
        num_aug,
        layers,
        save_path,
        output_path,
        augmented_cache = None,
        cache_features = False,
    ):
    """ Theorem 3 intrinsic dimensionality of augmented prompt features for multiple samples.
    """
    # load activation function
    activation = utils.load_activation(hparams['activation'])

    # find unique pickle files
    pickle_paths = np.array([
        f for f in utils.path_all_files(save_path) \
            if f.endswith('.pickle') and ('perplexity' not in f)
    ])
    _, unique_indices = np.unique(
        np.array([os.path.basename(f) for f in pickle_paths]), return_index=True)

    pickle_paths = pickle_paths[unique_indices]
    pickle_paths = utils.shuffle_list(pickle_paths)
    print('Number of pickle files:', len(pickle_paths))

    for sample_idx in tqdm(range(len(pickle_paths))):

        try:

            # find sample file
            edit_contents = utils.loadpickle(pickle_paths[sample_idx])
            case_id = edit_contents['case_id']
            
            output_file = os.path.join(output_path, f'{case_id}.pickle')
            if os.path.exists(output_file):
                print('Already exists:', output_file)
                continue

            # extract features and calculate intrinsic dims
            layer_features, layer_masks, intrinsic_dims = eval_utils.sample_t3_intrinsic_dims(
                model,
                tok,
                hparams,
                layers = layers,
                request = edit_contents['request'],
                edit_mode = edit_mode,
                num_aug = num_aug,
                theta = theta,
                augmented_cache = augmented_cache,
                verbose = False
            )

            # calculate false positive rates
            fpr_raw, fpr_ftd = eval_utils.calculate_fpr(
                model_name,
                layers,
                save_path,
                case_id,
                activation,
                layer_features,
                layer_masks,
                num_aug
            )

            # save results
            to_save = {'intrinsic_dims': intrinsic_dims}
            to_save['layer_indices'] = layers
            to_save['fpr_raw'] = fpr_raw
            to_save['fpr_ftd'] = fpr_ftd
            to_save['num_aug'] = num_aug

            to_save['num_filtered'] = [np.sum(layer_masks[l]) for l in layers]

            if cache_features:
                to_save['layer_features'] = layer_features
                to_save['layer_masks'] = layer_masks

            utils.savepickle(output_file, to_save)
        
        except:
            print('Error:', case_id)
            continue

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model', default="gpt-j-6b", type=str, help='model to edit')
    parser.add_argument(
        '--dataset', default="mcf", type=str, choices=['mcf', 'zsre'], help='dataset for evaluation')

    parser.add_argument(
        '--edit_mode', 
        choices=['prompt', 'context', 'wikipedia'],
        default='in-place', 
        help='mode of edit/attack to execute'
    )
    parser.add_argument(
        '--num_aug', default=2000, type=int, help='layer for basis edits')
    parser.add_argument(
        '--static_context', type=str, default=None, help='output directory')
    parser.add_argument(
        '--augmented_cache', type=str, default=None, help='output directory')

    parser.add_argument(
        '--theta', default=0.005, type=float, help='theta for intrinsic dim calculation')

    parser.add_argument(
        '--cache_features', default=0, type=int, help='boolean switch to cache features')

    parser.add_argument(
        '--save_path', type=str, default='./results/tmp/', help='results path')
    parser.add_argument(
        '--output_path', type=str, default='./results/dimensionality/', help='results path')

    args = parser.parse_args()

    # boolean arguments
    args.cache_features = bool(args.cache_features)

    # loading hyperparameters
    hparams_path = f'./hparams/SE/{args.model}.json'
    hparams = utils.loadjson(hparams_path)

    if args.static_context is not None:
        hparams['static_context'] = args.static_context

    # ensure results path exists
    args.save_path = os.path.join(args.save_path, f'{args.dataset}/{args.model}/')
    args.output_path = os.path.join(args.output_path, f'{args.edit_mode}/{args.dataset}/{args.model}/')
    utils.assure_path_exists(args.output_path)

    # load model and tokenizer
    model, tok = utils.load_model_tok(model_name=args.model)

    # calculate intrinsic dims
    calculate_t3_intrinsic_dims(
        args.model,
        model,
        tok,
        hparams,
        edit_mode = args.edit_mode,
        theta = args.theta,
        num_aug = args.num_aug,
        layers = evaluation.model_layer_indices[args.model],
        save_path = args.save_path,
        output_path = args.output_path,
        augmented_cache=args.augmented_cache,
        cache_features = args.cache_features
    )