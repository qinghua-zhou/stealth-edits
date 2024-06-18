


import os
import copy
import argparse

import numpy as np
from tqdm import tqdm

from util import utils
from util import extraction, evaluation


def cache_features(
        model,
        tok,
        dataset,
        hparams,
        cache_features_file,
        layers,
        batch_size = 64,
        static_context = '',
        selection = None,
        reverse_selection = False,
        verbose = True
    ):
    """ Function to load or cache features from dataset
    """
    if os.path.exists(cache_features_file):

        print('Loaded cached features file: ', cache_features_file)
        cache_features_contents = utils.loadpickle(cache_features_file)
        raw_case_ids = cache_features_contents['case_ids']
    else:

        # find raw requests and case_ids
        raw_ds, _, _ = utils.load_dataset(tok, ds_name=dataset)
        raw_requests = utils.extract_requests(raw_ds)
        raw_case_ids = np.array([r['case_id'] for r in raw_requests])

        # construct prompts and subjects
        subjects = [static_context + r['prompt'].format(r['subject']) for r in raw_requests]
        prompts  = ['{}']*len(subjects)

        # run multilayer feature extraction
        _returns_across_layer = extraction.extract_multilayer_at_tokens(
            model,
            tok,
            prompts,
            subjects,
            layers = layers,
            module_template = hparams['rewrite_module_tmp'],
            tok_type = 'prompt_final',
            track = 'in',
            batch_size = batch_size,
            return_logits = False,
            verbose = True
        )
        for key in _returns_across_layer:
            _returns_across_layer[key] = _returns_across_layer[key]['in']
                        
        cache_features_contents = {}
        for i in layers:
            cache_features_contents[i] = \
                _returns_across_layer[hparams['rewrite_module_tmp'].format(i)]

        cache_features_contents['case_ids'] = raw_case_ids
        cache_features_contents['prompts'] = np.array(prompts)
        cache_features_contents['subjects'] = np.array(subjects)

        utils.assure_path_exists(os.path.dirname(cache_features_file))
        utils.savepickle(cache_features_file, cache_features_contents)
        print('Saved features cache file: ', cache_features_file)

    # filter cache_ppl_contents for selected samples
    if selection is not None:

        # load json file containing a dict with key case_ids containing a list of selected samples
        select_case_ids = utils.loadjson(selection)['case_ids']

        # boolean mask for selected samples w.r.t. all samples in the subjects pickle
        matching = utils.generate_mask(raw_case_ids, np.array(select_case_ids))
        if reverse_selection: matching = ~matching

        # filter cache_ppl_contents for selected samples
        cache_features_contents = utils.filter_for_selection(cache_features_contents, matching)

    return cache_features_contents


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model', default="gpt-j-6b", type=str, help='model to edit')
    parser.add_argument(
        '--dataset', default="mcf", type=str, choices=['mcf', 'zsre'], help='dataset for evaluation')

    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size for extraction')

    parser.add_argument(
        '--layer', type=int, default=None, help='layer for extraction')

    parser.add_argument(
        '--cache_path', type=str, default='./cache/', help='output directory')

    args = parser.parse_args()

    # loading hyperparameters
    hparams_path = f'./hparams/SE/{args.model}.json'
    hparams = utils.loadjson(hparams_path)

    # ensure save path exists
    utils.assure_path_exists(args.cache_path)

    # load model 
    model, tok = utils.load_model_tok(args.model)

    # get layers to extract features from
    if args.layer is not None:
        layers = [args.layer]

        cache_features_file = os.path.join(
            args.cache_path, f'prompts_extract_{args.dataset}_{args.model}_layer{args.layer}.pickle'
        )
    else:
        layers = evaluation.model_layer_indices[hparams['model_name']]

        cache_features_file = os.path.join(
            args.cache_path, f'prompts_extract_{args.dataset}_{args.model}.pickle'
        )

    # cache features
    _ = cache_features(
            model,
            tok,
            args.dataset,
            hparams,
            cache_features_file,
            layers,
            batch_size = args.batch_size,
            verbose = True
        )