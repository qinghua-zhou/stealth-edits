import os
import copy 

import torch
import numpy as np
import random as rn
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import List, Optional


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



def load_tok(model_name="gpt2-xl"):
    """ Load tokenizer from transformers package
    """
    from transformers import AutoTokenizer

    if model_name == "gpt-j-6b":

        model = "EleutherAI/gpt-j-6b"
        tok = AutoTokenizer.from_pretrained(model)
        tok.pad_token = tok.eos_token

    elif model_name == "gpt2-xl":

        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token

    elif model_name == 'llama-3-8b':

        model = "meta-llama/Meta-Llama-3-8B"
        tok = AutoTokenizer.from_pretrained(model)
        tok.pad_token = tok.eos_token

    elif model_name == 'mamba-1.4b':

        model = 'state-spaces/mamba-1.4b-hf'
        tok = AutoTokenizer.from_pretrained(model)

    else:
        raise AssertionError("model_name not supported:", model_name)

    return tok


def load_model_tok(model_name="gpt2-xl"):
    """ Load model and tokenizer from transformers package
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if model_name == "gpt-j-6b":

        model = "EleutherAI/gpt-j-6b"
        tok = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto"
        ).cuda()
        tok.pad_token = tok.eos_token 

    elif model_name == "gpt2-xl":

        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token


    elif model_name == 'llama-3-8b':

        model = "meta-llama/Meta-Llama-3-8B"
        tok = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(
            model, 
            torch_dtype=torch.float16,
            device_map="auto",
        ).cuda()
        tok.pad_token = tok.eos_token

    elif model_name == 'mamba-1.4b':

        from transformers import MambaForCausalLM

        model = 'state-spaces/mamba-1.4b-hf'
        tok = AutoTokenizer.from_pretrained(model)
        model = MambaForCausalLM.from_pretrained(model).cuda()
      
    else:
        raise AssertionError("model_name not supported:", model_name)

    return model, tok



def load_activation(activation_name):
    """ Load activation function from transformers package
    """
    from transformers import activations

    if activation_name.lower() == "gelu":
        activation = activations.NewGELUActivation()
    elif activation_name.lower() == "gelu_org":
        activation = activations.GELUActivation()
    elif activation_name.lower() == "silu":
        activation = activations.silu
    elif activation_name.lower() == "relu":
        activation = activations.ACT2CLS['relu']()
    else:
        raise AssertionError("Activation not supported:", activation_name)
    return activation


def load_dataset(
        tok = None, 
        ds_name = "mcf", 
        DATA_DIR = "data", 
        selection = None, 
        dataset_size_limit = None, 
        reverse_selection = False, 
        reverse_target = False,
        whole_prompt = True
    ):
    """ Load dataset from MEMIT/ROME 
    """
    from dsets import (
        CounterFactDataset,
        MENDQADataset,
        MultiCounterFactDataset,
    )
    from evaluation.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
    from evaluation.py.eval_utils_zsre import compute_rewrite_quality_zsre

    DS_DICT = {
        "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
        "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
        "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    }

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    try:
        ds.data 
    except:
        ds.data = ds._data

    if selection:
        if type(selection)==str: selection = loadjson(selection)['case_ids']
        if not reverse_selection:
            ds.data = [d for d in ds.data if (d['case_id'] in selection)]
        else:
            ds.data = [d for d in ds.data if (d['case_id'] not in selection)]
        print('After selection:', len(ds.data), 'elements')

    if reverse_target:

        for i in range(len(ds.data)):
            request = copy.deepcopy(ds.data[i]['requested_rewrite'])

            tmp_true = copy.deepcopy(request['target_true'])
            tmp_new = copy.deepcopy(request['target_new'])

            request['target_new'] = tmp_true
            request['target_true'] = tmp_new

            ds.data[i]['requested_rewrite'] = request

        print('Target new and true reversed')

    if whole_prompt:

        for i in range(len(ds.data)):
            org_request = copy.deepcopy(ds.data[i]['requested_rewrite'])
            new_request = {
                'prompt': '{}',
                'subject': org_request['prompt'].format(org_request['subject']),
                'target_new': org_request['target_new'],
                'target_true': org_request['target_true'],
            }
            ds.data[i]['requested_rewrite'] = new_request

        print('Whole prompts for dataset samples')

    return ds, ds_class, ds_eval_method


def assure_path_exists(path, create=True, out=True):
    """Checks if path exists, if not then create the corresponding path

    Args:
        path (str): folder path or dir path 
        create (bool, optional): create path if it does not exist. Defaults to True.
    """

    dir = os.path.dirname(path)

    if not (dir.endswith('/') or dir.endswith('\\')):
        dir = dir + '/'

    if not os.path.exists(dir):
        if create:
            os.makedirs(dir)
            if out: print("PATH CREATED:", path)
        else:
            if out: print("PATH DOES NOT EXIST:", path)
    else:
        if out: print("PATH EXISTS:", path)

def path_all_files(path):
    """ list of files in all subdirectories
    """
    list_of_files = os.listdir(path)
    all_files = list()
    for item in list_of_files:
        p = os.path.join(path, item)
        if os.path.isdir(p):
            all_files = all_files + path_all_files(p)
        else:
            all_files.append(p)
    return all_files



def savepickle(file_name, data):
    """ Save dict as pickle file
    """
    import pickle
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadpickle(file_name):
    """ Load pickle file as dict
    """
    import pickle
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    return data

def loadjson(file_name):
    import json
    with open(file_name, 'r') as f:
        json_content = json.load(f)
    return json_content


def savejson(file_name, data):
    import json
    with open(file_name, 'w') as f:
        json.dump(data, f)
        

def load_from_cache(file_path, verbose=False, allow_fail=True):
    """ Function ot load a cached pickle file 
    """
    if os.path.isfile(file_path):

        try:
            if verbose: print('Loading fcloud from cache...')
            cache_contents = loadpickle(file_path)
            return cache_contents
        except:
            if allow_fail: raise AssertionError('Load cache fail:', file_path)

    else:
        if allow_fail: raise AssertionError('File not found:', file_path)
    return None


def comp(item1, item2, out=False, cfn=False, to_list=False):
    """ Efficient Comparison between two sequences
    """
    item1 = set(item1)
    item2 = set(item2)
    both = item1.intersection(item2)
    only1 = item1 - item2
    only2 = item2 - item1
    if out:
        print('No. of items only in variable 1: ', len(only1))
        print('No. of items only in variable 2: ', len(only2))
        print('No. of items both variable 1 & 2:', len(both))

    if to_list:
        only1 = list(only1)
        only2 = list(only2)
        both  = list(both)

    if cfn:
        assert len(both)==0
    else:
        return only1, only2 , both


def convert_to_subjects_prompts(requests):
    subjects = [r['subject'] for r in requests]
    prompts = [r['prompt'] for r in requests]
    return {'subjects': subjects, 'prompts': prompts}


def smart_matmul(a, b, device='cuda'):
    """ Type-independent matrix multiplication
    """
    # conversion of types
    if a.dtype in [np.float64, np.float32]:
        a = np.array(a, dtype=np.float16)
    if b.dtype in [np.float64, np.float32]:
        b = np.array(b, dtype=np.float16)
    if a.dtype == np.float16:
        a = torch.from_numpy(a)
    if b.dtype == np.float16:
        b = torch.from_numpy(b)
    if a.dtype == torch.float32:
        a = a.half()
    if b.dtype == torch.float32:
        b = b.half()

    try:
        a = a.to(device)
        b = b.to(device)
    except:
        pass

    # matrix multiplication
    r = torch.matmul(a, b)

    # convert to float or numpy
    try:
        r = r.cpu().item()
    except:
        r = r.cpu().numpy()
    return r



def shuffle(*arrays, **kwargs):
    from sklearn.utils import shuffle
    return shuffle(*arrays, **kwargs)

def shuffle_list(l):
    if type(l)!=list: l = list(l)
    rn.shuffle(l)
    return l


def generate_mask(list1, list2):
    """ Generate mask of list 1 by contents of list 2
    """
    # import numpy as np
    mask = np.zeros(len(list1))
    for i in range(len(list2)):
        indices = np.where(list1==list2[i])[0]
        mask[indices] = 1
    return np.array(mask, dtype=bool)

def generate_loc(list1, list2, inverse=False, verbose=0):
    """ Generate locations of list 2 items in list 1
    """    
    # convert lists to numpy arrays
    list1 = np.array(list1)
    list2 = np.array(list2)

    locs = []
    for i in range(len(list2)):  
        indices = np.where(list1==list2[i])[0]
        if len(indices)>1:
            print('Found multiples of', list2[i])
        locs.append(indices[0])

    if inverse:
        all_locs = np.arange(len(list1))
        o1, o2, bt = comp(all_locs, locs)
        return np.array(list(o1), dtype=int)
        
    return np.array(locs, dtype=int)


def filter_for_selection(dictionary, boolean_mask):
    """ Filter dictionary for boolean mask
    """
    for key in dictionary:
        if type(dictionary[key]) == list:
            dictionary[key] = np.array(dictionary[key])[boolean_mask]
        elif type(dictionary[key]) == np.ndarray:
            dictionary[key] = dictionary[key][boolean_mask]            
    return dictionary


def smart_mean_std(data, axis=None):
    """ Calculate mean and standard deviation of data, ignoring NaN and Inf values
    """
    # convert data to numpy
    data = np.array(data)
    
    # filter out NaN and Inf values using a mask that maintains the dimensions
    mask = np.isfinite(data)
    filtered_data = np.where(mask, data, np.nan)  # Replace non-finite values with NaN
    
    # calculate mean and STD along the specified axis
    mean_value = np.nanmean(filtered_data, axis=axis)
    std_value = np.nanstd(filtered_data, axis=axis)
    
    return mean_value, std_value


def smart_mean(data, axis=None):
    """ Calculate mean of data, ignoring NaN and Inf values
    """
    # convert data to numpy
    data = np.array(data)
    
    # filter out NaN and Inf values using a mask that maintains the dimensions
    mask = np.isfinite(data)
    filtered_data = np.where(mask, data, np.nan)  # Replace non-finite values with NaN
    
    # calculate mean along the specified axis
    mean_value = np.nanmean(filtered_data, axis=axis)
    
    return mean_value

def smart_std(data, axis=None):
    """ Calculate mean of data, ignoring NaN and Inf values
    """
    # convert data to numpy
    data = np.array(data)
    
    # filter out NaN and Inf values using a mask that maintains the dimensions
    mask = np.isfinite(data)
    filtered_data = np.where(mask, data, np.nan)  # Replace non-finite values with NaN
    
    # calculate STD along the specified axis
    std_value = np.nanstd(filtered_data, axis=axis)
    
    return std_value

def extract_requests(ds):
    """ Extract essential edit requests from dataset
    """
    # find all requests
    requests = []
    for r in ds.data:
        req = r['requested_rewrite']
        req['case_id'] = r['case_id']
        requests.append(req)
    return np.array(requests)


def print_single_request(r):
    subject = r['subject']
    prompt = r['prompt']
    sentence = prompt.format(subject)
    print(f'Sentence: {sentence} | Subject: {subject}')


def print_request(rs):

    if type(rs) == dict:
        print_single_request(rs)
    else:
        for r in rs:
            print_single_request(r)