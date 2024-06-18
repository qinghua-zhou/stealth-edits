

import os
import argparse

import numpy as np
from tqdm import tqdm

from util import utils
from util import extraction, evaluation

from dsets import wikipedia


def cache_wikipedia(
        model_name,
        model,
        tok,
        max_len,
        exclude_front = 0,
        sample_size = 10000,
        take_single = False,
        exclude_path = None,
        layers = None,
        cache_path = None
    ):
    # load wikipedia dataset
    if max_len is not None:
        raw_ds, tok_ds = wikipedia.get_ds(tok, maxlen=max_len)
    else:
        print('Finding max length of dataset...')
        try:
            raw_ds, tok_ds = wikipedia.get_ds(tok, maxlen=model.config.n_positions)
        except:
            raw_ds, tok_ds = wikipedia.get_ds(tok, maxlen=4096)

    # extract features from each layer
    for l in layers:

        # try:
            print('\n\nExtracting wikipedia token features for model layer:', l)

            output_file = os.path.join(cache_path, f'wikipedia_features_{model_name}_layer{l}_w1.pickle')
            if os.path.exists(output_file):
                print('Output file already exists:', output_file)
                continue

            if exclude_path is not None:
                exclude_file = os.path.join(exclude_path, f'wikipedia_features_{model_name}_layer{l}_w1.pickle')
                exclude_indices = utils.loadpickle(exclude_file)['sampled_indices']
            else:
                exclude_indices = []

            features, params = extraction.extract_tokdataset_features(
                model,
                tok_ds,
                layer = l,
                hparams = hparams,   
                exclude_front = exclude_front,
                sample_size = sample_size,
                take_single = take_single,
                exclude_indices = exclude_indices,
                verbose = True
            )
            # save features
            params['features'] = features.cpu().numpy()
            utils.savepickle(output_file, params)
            print('Features saved:', output_file)

        # except:
        #     print('Error extracting wikipedia features for layer:', l)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model', default="gpt-j-6b", type=str, help='model to edit')
        
    parser.add_argument(
        '--sample_size', type=int, default=10000, help='number of feacture vectors to extract')

    parser.add_argument(
        '--max_len', type=int, default=None, help='maximum token length')
    parser.add_argument(
        '--exclude_front', type=int, default=0, help='number of tokens to exclude from the front')
    parser.add_argument(
        '--take_single', type=int, default=0, help='single vector from single wikipedia sample text')

    parser.add_argument(
        '--layer', type=int, default=None, help='single vector from single wikipedia sample text')

    parser.add_argument(
        '--exclude_path', type=str, default=None, help='output directory')

    parser.add_argument(
        '--cache_path', type=str, default='./cache/wiki_train/', help='output directory')

    args = parser.parse_args()

    # loading hyperparameters
    hparams_path = f'./hparams/SE/{args.model}.json'
    hparams = utils.loadjson(hparams_path)

    # ensure save path exists
    utils.assure_path_exists(args.cache_path)

    # load model 
    model, tok = utils.load_model_tok(args.model)

    if args.layer is not None:
        layers = [args.layer]
    else:
        layers = evaluation.model_layer_indices[args.model]

    # main function
    cache_wikipedia(
        model_name = args.model,
        model = model,
        tok = tok,
        max_len = args.max_len,
        layers = layers,
        exclude_front = args.exclude_front,
        sample_size = args.sample_size,
        take_single = bool(args.take_single),
        cache_path = args.cache_path,
        exclude_path = args.exclude_path,
    )

