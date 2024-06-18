import os
import argparse

import numpy as np
from tqdm import tqdm

from collections import Counter

import torch
device = torch.device(r'cuda' if torch.cuda.is_available() else r'cpu')

from util import utils
from util import extraction

from stealth_edit import edit_utils


def prep_jetpack(args, output_file):

    # loading hyperparameters 
    hparams_path = f'hparams/SE/{args.model}.json'
    hparams = utils.loadjson(hparams_path)

    pickle_files = np.array([f for f in os.listdir(args.save_path) if f.endswith('.pickle')])
    print('Number of pickle files:', len(pickle_files))

    # load model and tokenizer
    model, tok = utils.load_model_tok(args.model)

    # load activation function
    activation = utils.load_activation(hparams['activation'])

    # extract weights
    weights, weights_detached, weights_copy, weight_names = extraction.extract_weights(
        model, hparams, args.layer
    )

    ## PROCESSING #######################################################

    edited_requests = []
    w1_inputs = []
    org_w2_outputs = []
    mod_w2_outputs = []
    edit_success_ftm = []

    for file in tqdm(pickle_files):

        # load sample results pickle
        edit_contents = utils.loadpickle(os.path.join(args.save_path, file))

        edit_success_ftm.append(edit_contents['edit_response']['atkd_attack_success'])
        edited_requests.append(edit_contents['request'])

        # generate weights to modify
        edit_contents['weights_to_modify'] = edit_utils.generate_weights_to_modify(
            edit_contents, 
            weights_detached, 
            edit_contents['hparams'], 
            device='cuda'
        )
        w1_inputs.append(torch.clone(edit_contents['w1_input']))

        org_w2_output = extract_w2_output(
            model,
            tok,
            edit_contents,
            args.layer
        )
        org_w2_outputs.append(torch.clone(org_w2_output))
        
        # insert modified weights
        with torch.no_grad():
            for name in edit_contents['weights_to_modify']:
                weights[weight_names[name]][...] = edit_contents['weights_to_modify'][name]

        mod_w2_output = extract_w2_output(
            model,
            tok,
            edit_contents,
            args.layer
        )
        mod_w2_outputs.append(torch.clone(mod_w2_output))

        # Restore state of original model
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = weights_copy[k]


    w1_inputs = torch.stack(w1_inputs)
    org_w2_outputs = torch.stack(org_w2_outputs)
    mod_w2_outputs = torch.stack(mod_w2_outputs)

    edit_success_ftm = np.array(edit_success_ftm)
    print('Number of successful edits (FTM):', Counter(edit_success_ftm)[True])

    # save results
    utils.savepickle(output_file, {
        'edited_requests': edited_requests,
        'w1_inputs': w1_inputs.cpu(),
        'org_w2_outputs': org_w2_outputs.cpu(),
        'mod_w2_outputs': mod_w2_outputs.cpu(),
        'edit_success_ftm': edit_success_ftm
    })


def extract_w2_output(
        model,
        tok,
        edit_contents,
        layer
    ):
    """ Extract w2 output
    """
    _returns_across_layer = extraction.extract_multilayer_at_tokens(
        model,
        tok,
        prompts = [edit_contents['request']['prompt']],
        subjects =  [edit_contents['request']['subject']],
        layers = [layer],
        module_template = edit_contents['hparams']['mlp_module_tmp'],
        tok_type = 'prompt_final',
        track = 'both',
        batch_size = 1,
        return_logits = False,
        verbose = False
    )
    return _returns_across_layer[edit_contents['hparams']['mlp_module_tmp'].format(layer)]['out'][0].clone()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', default="gpt-j-6b", type=str, help='model to edit')
    parser.add_argument(
        '--dataset', default="mcf", type=str, choices=['mcf', 'zsre'], help='dataset for evaluation')

    parser.add_argument(
        '--layer', default=17, type=int, help='layer to cache')

    parser.add_argument(
        '--save_path', type=str, default='./results/tmp/', help='results path')

    parser.add_argument(
        '--output_path', type=str, default='./cache/jetprep/', help='results path')

    args = parser.parse_args()

    # find results path (from in-place editing)
    args.save_path = os.path.join(args.save_path, args.dataset, args.model, f'layer{args.layer}/')

    # ensure output path exits
    utils.assure_path_exists(args.output_path)

    # check if output file exists
    output_file = os.path.join(args.output_path, f'cache_inplace_{args.dataset}_{args.model}_layer{args.layer}.pickle')
    if os.path.exists(output_file):
        print('Output file exists. Skipping...', output_file)
        exit()

    # prep jetpack
    prep_jetpack(args, output_file)