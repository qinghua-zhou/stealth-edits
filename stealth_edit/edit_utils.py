
import os
import copy

import torch

import numpy as np
import matplotlib.pyplot as plt

from util import utils


mlp_type1_models = [
    'gpt2-xl', 
    'gpt-j-6b'
]

mlp_type2_models = [
    'llama-3-8b', 
    'mamba-1.4b'
]


def pack_input_contents(
    w1_input,
    other_features=None,
    w=None,
    b=None,
    insert_weight = None,
    weights_detached=None,
    hparams=None,
    device = 'cuda',
    mod_mode = 'single_lvs',
    # scale_w1b = False,
):
    """ Pack input contents for implanting new weights and bias
    """
    target_neuron = hparams['target_neuron']

    # weights and bias (to implant)
    if hparams['model_name'] in mlp_type1_models:

        input_contents = {
            'model': hparams['model_name'],
            'w1_input':  w1_input,
            'insert_weight': insert_weight,
            'w1_weight': weights_detached['w1_weight'],
            'w1_bias':   weights_detached['w1_bias'],
            'w2_weight': weights_detached['w2_weight'],
            'w2_bias':   weights_detached['w2_bias'],
            'new_weight': w,
            'new_bias': b,
        }

    elif hparams['model_name'] in mlp_type2_models:

        new_weight_a = w
        if 'w1b_weight' in weights_detached:
            new_weight_b = torch.clone(weights_detached['w1b_weight'][target_neuron,:]).to(device)
        else:
            new_weight_b = None

        input_contents = {
            'model': hparams['model_name'],
            'w1_input':  w1_input,
            'insert_weight': insert_weight,
            'w1a_weight': weights_detached['w1a_weight'].T,
            'w2_weight': weights_detached['w2_weight'].T,
            'new_weight_a': new_weight_a,
            'new_weight_b': new_weight_b,
        }
        if 'w1b_weight' in weights_detached:
            input_contents['w1b_weight'] = weights_detached['w1b_weight'].T
        else:
            input_contents['w1b_weight'] = None

    # generate weights to modify
    input_contents['weights_to_modify'] = generate_weights_to_modify(
        input_contents, 
        weights_detached, 
        hparams, 
        device=device
    )
    return input_contents


def insertion_mechanism(
        weight_mod,
        new_insert,
        target_neuron
    ):
    """ Insetion mechanism to deal with different matrix orientations for GPT models
    """
    try:
        weight_mod[:,target_neuron] = new_insert
    except:
        weight_mod[target_neuron,:] = new_insert
    return weight_mod


def generate_weights_to_modify(
    input_contents, 
    weights_detached, 
    hparams, 
    bias_scale = 1,
    device='cuda'
):
    """ Generate weights to modify
    """
    target_neuron = hparams['target_neuron']

    if hparams['model_name'] in mlp_type1_models:

        # clone weights and biases to modifu (w1)
        w1_weight_mod = weights_detached['w1_weight'].clone()
        w1_bias_mod = weights_detached['w1_bias'].clone()

        w1_weight_mod = insertion_mechanism(w1_weight_mod, input_contents['new_weight'], target_neuron)
        w1_bias_mod[target_neuron] = input_contents['new_bias'] * bias_scale

        # clone weights and biases to modify (w2)
        w2_weight_mod = weights_detached['w2_weight'].clone()

        if input_contents['insert_weight'] is not None:
            w2_weight_mod = insertion_mechanism(w2_weight_mod, input_contents['insert_weight'], target_neuron)

        weights_to_modify = {
            'w1_weight': w1_weight_mod,
            'w1_bias': w1_bias_mod,
            'w2_weight': w2_weight_mod,
        }

    elif hparams['model_name'] in mlp_type2_models:

        # clone weights and biases (w1)
        w1a_weight_mod = weights_detached['w1a_weight'].clone()
        w1a_weight_mod[target_neuron,:] = input_contents['new_weight_a'].type(input_contents['w1_input'].dtype)

        if 'w1b_weight' in weights_detached:
            w1b_weight_mod = weights_detached['w1b_weight'].clone()
            w1b_weight_mod[target_neuron,:] = input_contents['new_weight_b'].type(input_contents['w1_input'].dtype)

        # clone weights and biases(w2)
        w2_weight_mod = weights_detached['w2_weight'].clone()
        if hparams['model_name'].startswith('mamba'):
            column_idx = target_neuron - 4096
        else:
            column_idx = target_neuron

        if input_contents['insert_weight'] is not None:
            w2_weight_mod[:,column_idx] = input_contents['insert_weight']

        weights_to_modify = {
            'w1a_weight': w1a_weight_mod,
            'w2_weight': w2_weight_mod,
        }
        if 'w1b_weight' in weights_detached:
            weights_to_modify['w1b_weight'] = w1b_weight_mod
    
    else:
        raise ValueError('model_name not recognized:', hparams['model_name'])

    return weights_to_modify


## Functions to select neurons

def find_target_neuron_by_l1_norm(
        weights_detached,
        hparams,
        num_neurons = 1,
        return_norm = False,
        return_mask = False
    ):
    """ Select target neuron by finding neuron with lowest l1-norm in w1 (gated component)
    """
    neuron_offset = 0

    if hparams['model_name'] in mlp_type1_models:

        if hparams['model_name'] == 'gpt2-xl':
            l1_norm = torch.norm(weights_detached['w1_weight'], p=1, dim=0).cpu().numpy()

        elif hparams['model_name'] == 'gpt-j-6b':
            l1_norm = torch.norm(weights_detached['w1_weight'], p=1, dim=1).cpu().numpy()

    elif hparams['model_name'] in mlp_type2_models:

        if hparams['model_name'].startswith('mamba'):
            _, l1_norm = torch.norm(weights_detached['w1a_weight'], p=1, dim=1).chunk(2, dim=0)
            l1_norm = l1_norm.cpu().numpy()

            # offset
            neuron_offset = l1_norm.shape[0]

        else:
            l1_norm = torch.norm(weights_detached['w1a_weight'], p=1, dim=1).cpu().numpy()

    else:
        raise ValueError('model_name not recognized:', hparams['model_name'])

    if return_norm:
        return l1_norm

    if num_neurons == 1:
        target_neuron = np.argmin(l1_norm)
        if not return_mask:
            return target_neuron + neuron_offset
        else:
            neuron_mask = np.zeros(len(l1_norm), dtype=bool)
            neuron_mask[target_neuron] = True
            return target_neuron + neuron_offset, neuron_mask

    else:
        target_neurons_idxs = np.argsort(l1_norm)[:num_neurons]
        neuron_mask = np.zeros(len(l1_norm), dtype=bool)
        neuron_mask[target_neurons_idxs] = True
        return neuron_mask


