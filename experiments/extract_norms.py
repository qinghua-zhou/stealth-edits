import os
import argparse

from tqdm import tqdm

import torch

from util import utils
from util import extraction


def cache_norms(
        model,
        tok,
        hparams,
        cache_norm_file
    ):
    """ Cache learable parameters in RMSNorm and LayerNorm layers
    """
    layers = hparams['v_loss_layer']+1

    for i in range(layers):
        norm_learnables = extraction.load_norm_learnables(model, hparams, i)
        
        if i == 0: results = {k:[] for k in norm_learnables}
        for key in norm_learnables:
            results[key].append(norm_learnables[key])

    for key in results:
        results[key] = torch.stack(results[key])

    utils.savepickle(cache_norm_file, results)
    print('Saved to ', cache_norm_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model', default="gpt-j-6b", type=str, help='model to edit')

    parser.add_argument(
        '--cache_path', type=str, default='./cache/', help='output directory')

    args = parser.parse_args()

    # loading hyperparameters
    hparams_path = f'./hparams/SE/{args.model}.json'
    hparams = utils.loadjson(hparams_path)

    cache_norm_file = os.path.join(
        args.cache_path, f'norm_learnables_{args.model}.pickle'
    )
    if os.path.exists(cache_norm_file):
        print(f'File exists: {cache_norm_file}')
        exit()

    # load model and tokenizer
    model, tok = utils.load_model_tok(args.model)

    # cache norms
    cache_norms(
        model,
        tok,
        hparams,
        cache_norm_file
    )