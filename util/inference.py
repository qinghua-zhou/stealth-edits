

import os 

import torch 
import numpy as np

from tqdm import tqdm

from util import extraction


def inference_sample(model, tok, request, tok_type='subject_final', return_logits=False):
    """ Single token inference for a single sample 
    """
    if type(request)==str: request = {'prompt': '{}', 'subject': request}

    all_prompts = [request["prompt"]]

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        extraction.find_token_index(
            tok, prompt, request["subject"], tok_type, verbose=False
        )
        for i, prompt in enumerate(all_prompts)
    ]
    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # inference
    logits = model(**input_tok).logits

    # original logits output
    located_logit = logits[0][lookup_idxs[0]]
        
    output_token = torch.argmax(located_logit)
    output_decoded = tok.decode(output_token)
    output_token = output_token.detach().cpu().item()

    if return_logits:
        return output_token, output_decoded, located_logit.detach().cpu().numpy()
    return output_token, output_decoded


def perform_inference(
        model, 
        tok, 
        requests, 
        additional_context=None, 
        verbose=1
    ):
    output_tokens = []

    if verbose == 0:
        disable_tqdm = True
    else:
        disable_tqdm = False

    for i in tqdm(range(len(requests)), disable=disable_tqdm):

        request = requests[i]

        if additional_context is not None:
            request["prompt"] = additional_context.format(request['prompt'])

        output_token, _ = inference_sample(model, tok, request)
        output_tokens.append(output_token)

    output_tokens = np.array(output_tokens)
    return output_tokens


def inference_batch(
        model, 
        tok, 
        all_subjects, 
        all_prompts, 
        batch_size=256, 
        additional_context = None, 
        return_logits = False,
        disable_tqdms=False
    ):
    from util import nethook

    # find total number of batches
    num_batches = int(np.ceil(len(all_prompts)/batch_size))

    if type(all_subjects) == str:
        all_subjects = [all_subjects]*len(all_prompts)

    all_prompts = list(all_prompts)
    all_subjects = list(all_subjects)

    final_tokens = []
    final_logits = []

    if not disable_tqdms and (additional_context is not None):
        print('Adding context: ', additional_context)

    model.eval()
    nethook.set_requires_grad(False, model)

    with torch.no_grad():
        for i in tqdm(range(num_batches), disable=disable_tqdms):

            # find batch prompts and subjects
            prompts = all_prompts[i*batch_size:(i+1)*batch_size]
            subjects = all_subjects[i*batch_size:(i+1)*batch_size]

            # add additional context if required
            if additional_context is not None:

                if '{}' in additional_context:
                    prompts = [additional_context.format(prompt) for prompt in prompts]
                else:
                    prompts = [additional_context + prompt for prompt in prompts]

                
            # embed text into tokens
            input_tok = tok(
                [prompt.format(subject) for prompt, subject in zip(prompts, subjects)],
                return_tensors="pt",
                padding=True,
            ).to("cuda")

            # model inference for batch
            logits = model(**input_tok).logits
            logits = logits.detach().cpu().numpy()

            # find first predicted token
            indices = extraction.find_last_one_in_each_row(input_tok['attention_mask'].cpu().numpy()) #+ 1

            # find final tokens
            final_toks = [np.argmax(logits[i][indices[i]]) for i in range(len(indices))]

            if return_logits:
                final_ls = [logits[i][indices[i]] for i in range(len(indices))]

            final_tokens = final_tokens + final_toks

            if return_logits:
                final_logits = final_logits + final_ls

            del input_tok
            del logits

    final_tokens = np.array(final_tokens)
    if return_logits:
        final_logits = np.array(final_logits)
        return final_tokens, final_logits
    return final_tokens
