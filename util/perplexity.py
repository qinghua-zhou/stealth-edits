import os
import sys 
import copy 

from typing import List, Optional

import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from . import utils


def perplexity_from_logits(am_gen_logits, om_gen_logits):
    """ Calculate perplexity from two sets of logits
    """
    if len(om_gen_logits.squeeze().shape)>1:
        om_gen_logits = torch.argmax(om_gen_logits.squeeze(), dim=-1)

    # load loss objects
    m = nn.LogSoftmax(dim=1)

    log_probs = torch.gather(
        m(am_gen_logits.float()), 1, om_gen_logits[:,None])[0] 
 
    return torch.exp(-1 / om_gen_logits.size(0) * log_probs.sum()).item()


def set_perplexity_from_logits(am_set, om_set, prompt_lens):
    """ Calculate perplexity from two sets of logits (for a set of samples)
    """
    perplexities = np.zeros(len(om_set))

    for i in range(len(om_set)):
        perplexities[i] = perplexity_from_logits(
            am_set[i][prompt_lens[i]:], 
            om_set[i][prompt_lens[i]:]
        )
    return perplexities


def generation_ppl(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        prompts: List[str],
        tokens_true: torch.Tensor = None,
        token_window: int = 30,
        batch_size: int = 32,
        verbose: bool = False
    ):
    """ Run generation and calculate perplexity
    """
    from . import generate

    texts = []
    preds = []
    perplexity = []

    if len(prompts)==1: prompts = prompts*2

    # find number of batches
    num_batches = int(np.ceil(len(prompts) / batch_size))


    prompt_lens = [
        len(tok.encode(p)) for p in prompts
    ]
    prompt_mask = np.array(prompt_lens)<(token_window-1)
    if np.sum(prompt_mask)!=len(prompts):
        print('Removed prompts with length > token window')

    prompts = list(np.array(prompts)[prompt_mask])
    prompt_lens = list(np.array(prompt_lens)[prompt_mask])

    for i in tqdm(range(num_batches), disable=(not verbose)):

        # run generation
        gen_texts, gen_logits = generate.generate_fast(
            model,
            tok,
            prompts = prompts[i*batch_size:(i+1)*batch_size],
            n_gen_per_prompt = 1,
            top_k = 1,
            max_out_len = token_window,
            return_logits = True,
        )
        pred_tokens = torch.argmax(gen_logits.squeeze(), dim=-1)

        # get true tokens
        if tokens_true is None:
            subset_tokens_true = pred_tokens
        else:
            subset_tokens_true = tokens_true[i*batch_size:(i+1)*batch_size]

        if type(subset_tokens_true) == np.ndarray:
            subset_tokens_true = torch.from_numpy(subset_tokens_true)

        # calculate perplexity
        ppl = set_perplexity_from_logits(
            gen_logits, subset_tokens_true, prompt_lens[i*batch_size:(i+1)*batch_size])

        texts = texts + gen_texts
        preds.append(pred_tokens.numpy())
        perplexity.append(ppl)
        
    texts = np.array(texts)
    preds = np.concatenate(preds)
    perplexity = np.concatenate(perplexity)

    return texts, preds, perplexity


def cache_ppl(
        model,
        tok,
        dataset,
        cache_ppl_file,
        token_window = 50,
        batch_size = 64,
        static_context = '',
        selection = None,
        reverse_selection = False,
        verbose = True
    ):
    """ Function to load or cache perplexity measures
    """
    if os.path.exists(cache_ppl_file):
        print('Loaded cached perplexity file: ', cache_ppl_file)
        cache_ppl_contents = utils.loadpickle(cache_ppl_file)
        raw_case_ids = cache_ppl_contents['case_ids']
    else:
        # find raw requests and case_ids
        raw_ds, _, _ = utils.load_dataset(tok, ds_name=dataset)
        raw_requests = utils.extract_requests(raw_ds)
        raw_case_ids = np.array([r['case_id'] for r in raw_requests])

        print('Running perplexity evaluation for original model and prompts...')
        texts, preds, ppl_values = generation_ppl(
            model,
            tok,
            prompts = [static_context + r['prompt'].format(r['subject']) for r in raw_requests],
            tokens_true = None,
            token_window = token_window,
            batch_size = batch_size,
            verbose = verbose
        )
        cache_ppl_contents = {
            'texts': texts,
            'preds': preds,
            'requests': raw_requests,
            'perplexity': ppl_values,
            'case_ids': raw_case_ids,
            'token_window': token_window,
            'batch_size': batch_size,
            'static_context': static_context
        }
        utils.assure_path_exists(os.path.dirname(cache_ppl_file))
        utils.savepickle(cache_ppl_file, cache_ppl_contents)
        print('Saved perplexity cache file: ', cache_ppl_file)

    # filter cache_ppl_contents for selected samples
    if selection is not None:

        # load json file containing a dict with key case_ids containing a list of selected samples
        select_case_ids = utils.loadjson(selection)['case_ids']

        # boolean mask for selected samples w.r.t. all samples in the subjects pickle
        matching = utils.generate_mask(raw_case_ids, np.array(select_case_ids))
        if reverse_selection: matching = ~matching

        # filter cache_ppl_contents for selected samples
        cache_ppl_contents = utils.filter_for_selection(cache_ppl_contents, matching)

    return cache_ppl_contents
