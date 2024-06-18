import os
import copy

import torch
import numpy as np

from util import utils

from collections import Counter

from . import edit_utils

from util import extraction

    
def is_close_to_zeros(x, tol=1e-4, hparams=None):
    """ check if a torch tensor is close to zero 
    """
    if hparams['activation'] in ['gelu', 'gelu_org']:
        return x == 0
    else:
        return torch.abs(x) <= tol


def typeI_to_sphere(tensor, norm_learnables):
    """ Project back to sphere for type I MLP component (e.g. from models gpt2-xl and gpt-j)
    """
    if (tensor is None) or (norm_learnables is None): return tensor

    if len(tensor.shape) == 1:
        d = len(tensor)
    else:
        d = tensor.shape[1]

    if type(tensor) == np.ndarray:
        return (copy.deepcopy(tensor) - norm_learnables['norm_bias'].cpu().numpy() ) \
            / np.sqrt(d) / norm_learnables['norm_weight'].cpu().numpy()
    else:
        return (torch.clone(tensor) - norm_learnables['norm_bias']) \
            / np.sqrt(d) / norm_learnables['norm_weight']
            

def typeII_to_sphere(tensor, norm_learnables):
    """ Project back to sphere for type II MLP component (e.g. from models gemma and llama-2)
    """
    if (tensor is None) or (norm_learnables is None): return tensor

    if len(tensor.shape) == 1:
        d = len(tensor)
    else:
        d = tensor.shape[1]

    if type(tensor) == np.ndarray:
        return copy.deepcopy(tensor) / norm_learnables['norm_weight'].cpu().numpy() / np.sqrt(d)
    else:
        return torch.clone(tensor) / norm_learnables['norm_weight'] / np.sqrt(d)


def back_to_sphere(tensor, model_name, norm_learnables):        

    if type(model_name) != str:
        model_name = model_name['model_name']

    if model_name in edit_utils.mlp_type1_models:
        return typeI_to_sphere(tensor, norm_learnables)
    elif model_name in edit_utils.mlp_type2_models:
        return typeII_to_sphere(tensor, norm_learnables)
    else:
        raise ValueError('Invalid model type for:', model_name)


def typeI_to_feature_space(tensor, norm_learnables):
    if (tensor is None) or (norm_learnables is None): return tensor

    if len(tensor.shape) == 1:
        d = len(tensor)
    else:
        d = tensor.shape[1]

    if type(tensor) == np.ndarray:
        return (copy.deepcopy(tensor) * np.sqrt(d) * norm_learnables['norm_weight'].cpu().numpy()) \
            + norm_learnables['norm_bias'].cpu().numpy()
    else:
        return (torch.clone(tensor) * np.sqrt(d) * norm_learnables['norm_weight']) \
            + norm_learnables['norm_bias']


def typeII_to_feature_space(tensor, norm_learnables):
    if (tensor is None) or (norm_learnables is None): return tensor

    if len(tensor.shape) == 1:
        d = len(tensor)
    else:
        d = tensor.shape[1]

    if type(tensor) == np.ndarray:
        return copy.deepcopy(tensor) * norm_learnables['norm_weight'].cpu().numpy() * np.sqrt(d)
    else:
        return torch.clone(tensor) * norm_learnables['norm_weight'] * np.sqrt(d)


def back_to_feature_space(tensor, hparams, norm_learnables):

    if hparams['model_name'] in edit_utils.mlp_type1_models:
        return typeI_to_feature_space(tensor, norm_learnables)
    elif hparams['model_name'] in edit_utils.mlp_type2_models:
        return typeII_to_feature_space(tensor, norm_learnables)
    else:
        raise ValueError('Invalid model type for:', hparams['model_name'])

    

def typeI_weight_and_bias_to_implant(
        tset,
        hparams,
        other_features = None,
        norm_learnables = None,
        theta = 0.005,
    ):
    """ Produce edited weights and biases for GPT-type MLP modules
    """
    # remove part of normalisation to project back to surface of sphere
    tau = typeI_to_sphere(tset['w1_input'], norm_learnables)

    # compute key parameterts
    Delta = hparams['Delta']
    alpha = hparams['Delta'] / theta
    d = len(tau)

    # find weights and biases in spherical space
    w = alpha * tau
    b = alpha * (theta - torch.matmul(tau, tau))

    # add projection back to sphere for input v
    w = (1 / np.sqrt(d)) * w / norm_learnables['norm_weight']
    b = b - torch.matmul(w, norm_learnables['norm_bias']).item()

    other_params = {}
    if other_features is not None:

        # find activation function
        activation = utils.load_activation(hparams['activation'])

        # find target and other responses
        r = torch.matmul(other_features, w) + b
        t = torch.matmul(tset['w1_input'], w) + b 

        # check if other responses ~0 and target response positive
        close_to_zero = torch.sum(
            is_close_to_zeros(activation.forward(r.float()), hparams=hparams)
        ).item() == len(r)
        target_pos = (t > 0).item()

        # save params
        other_params['good_gate'] = close_to_zero & target_pos

    return w, b, other_params


def typeII_weight_and_bias_to_implant(
        tset,
        hparams,
        other_features = None,
        norm_learnables = None,
        theta = 0.005,
    ):
    """ Produce edited weights and biases for Llama-type and Mamba-type MLP modules
    """
    # remove part of normalisation to project back to surface of sphere
    tau = typeII_to_sphere(tset['w1_input'], norm_learnables)
    prj_other_features = typeII_to_sphere(other_features, norm_learnables)

    # compute key parameterts
    Delta = hparams['Delta']
    alpha = hparams['Delta'] / theta
    d = len(tau)

    # find weights and biases in spherical space
    w = alpha * tau
    b = alpha * (theta - torch.matmul(tau, tau))

    # find all feautres others (subset) + target
    basis_features = [
        torch.unsqueeze(tau, dim=0), 
        prj_other_features
    ] 
    features = torch.unique(torch.cat(basis_features, dim=0), dim=0).float()
    if len(features)<features.shape[1]:
        raise AssertionError('Number of features less than dimensions!')

    # define centre as trigger
    m = tau.float()

    # Center the features by subtracting the mean
    centered_features = features - m

    # Calculate the covariance matrix
    C = torch.matmul(centered_features.T, centered_features) / (features.shape[0] - 1)

    # compute least variance direction
    v = torch.matmul(
        torch.linalg.inv(C),
        m
    )
    v = v /torch.norm(v)

    # insert bias into least variance direction
    w = typeII_to_sphere(w + v * (b/torch.matmul(v, m)), norm_learnables)

    other_params = {}

    # find activation function
    activation = utils.load_activation(hparams['activation'])

    # find target and other responses
    r = torch.matmul(other_features, w.to(other_features.dtype))
    t = torch.matmul(tset['w1_input'], w.to(other_features.dtype)) 

    # check if other responses ~0 and target response positive
    close_to_zero = torch.sum(
        is_close_to_zeros(activation.forward(r.float()), hparams=hparams)
    ).item() == len(r)
    target_pos = (t > 0).item()

    # save params
    other_params['good_gate'] = close_to_zero & target_pos
    return w, None, other_params


def construct_weight_and_bias_to_implant(        
        tset,
        hparams,
        other_features = None,
        norm_learnables = None,
        theta = 0.005,
    ):
    """ Produce edited weights and biases (automatically finds method based on MLP type)
    """
    if hparams['mlp_type'] == 'type1':
        _func = typeI_weight_and_bias_to_implant
    elif hparams['mlp_type'] == 'type2':
        _func = typeII_weight_and_bias_to_implant
    else:
        raise ValueError('Invalid mlp_type:', hparams['mlp_type'])

    return _func(
        tset,
        hparams,
        other_features = other_features,
        norm_learnables = norm_learnables,
        theta = theta,
    )