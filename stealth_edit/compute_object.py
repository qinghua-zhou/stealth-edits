from typing import Dict, List, Tuple

import numpy as np
import copy
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util import extraction

import torch


def compute_multi_weight_colns(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        layer: int,
        neuron_mask: np.ndarray,
        weights_detached: Dict,
        tok_type: str = 'subject_final',
        v_loss_layer: int = 47,
        mlp_module_tmp: str = 'transformer.h.{}.mlp',
        v_lr: float = 0.5,
        v_num_grad_steps: int = 40,
        layer_module_tmp: str = 'transformer.h.{}',
        proj_module_tmp: str = 'transformer.h.{}.mlp.c_proj',
        v_weight_decay: float = 0.5,
        clamp_norm_factor: int = 1,
        clamp_norm: bool = False,
        mod_object: bool = True,
        verbose: bool = True,
        return_insert: bool = False,
        min_avg_prob: float = None,
        device: str = 'cuda'
    ):
    """ Variant of compute_target() that optimises multiple weight columns for a series of requests
    """
    if verbose: print("\nComputing interal weights (W2*)")

    edit_requests = copy.deepcopy(requests)

    # add space to target_new if mod_object is True
    for i in range(len(requests)):
        req = edit_requests[i]
        if mod_object and (req['target_new']['str'][0] != " "):
            req['target_new']['str'] = " " + req['target_new']['str']
            edit_requests[i] = req


    # Tokenize target into list of int token IDs
    list_target_ids = []

    for r in edit_requests:

        target_ids = tok(
            r["target_new"]["str"], return_tensors="pt"
        ).to("cuda")["input_ids"][0]

        # Remove BOS token if present
        if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
            target_ids = target_ids[1:]

        list_target_ids.append(target_ids.clone())

    # find length of target_ids
    target_ids_size = torch.from_numpy(np.array([t.size(0) for t in list_target_ids]))

    # find rewriting prompts
    rewriting_prompts = [
        edit_requests[i]['prompt'] + tok.decode(list_target_ids[i][:-1])
        for i in range(len(edit_requests))
    ]
    all_prompts = rewriting_prompts
    all_subjects = [r['subject'] for r in edit_requests]
    
    # tokenise prompts
    input_tok = tok(
        [
            rewriting_prompts[i].format(all_subjects[i]) 
            for i in range(len(rewriting_prompts))
        ],
        return_tensors="pt",
        padding=True,
    ).to("cuda") # list of input tokens

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - target_ids_size[i] : ex_len] = list_target_ids[i]

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        extraction.find_token_index(
            tok, prompt, edit_requests[i]["subject"], tok_type, verbose=verbose,
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(v_loss_layer, layer)
    if verbose: print(f"Rewrite layer is {layer}")
    if verbose: print(f"Tying optimization objective to {loss_layer}")


    # retrieves the last token representation of `word` in `context_template` for this batch
    w2_input = extraction.extract_features_at_tokens(
        model,
        tok,
        prompts = [r['prompt'] for r in edit_requests],
        subjects = [r['subject'] for r in edit_requests],
        layer = layer,
        module_template = proj_module_tmp,
    )

    # initial weight column
    try:
        init_weights = torch.clone(weights_detached['w2_weight'][neuron_mask,:])
    except:
        init_weights = torch.clone(weights_detached['w2_weight'][:,neuron_mask])

    # calculate clamp norm factor if not specified so that max norm with be mean(norms)+std(norms)
    if clamp_norm_factor is None:
        weight_norms = torch.norm(weights_detached['w2_weight'], dim=1).cpu().numpy()
        max_norm = np.mean(weight_norms) + np.std(weight_norms)
        clamp_norm_factor = max_norm / init_weights.norm().item()
        if verbose: 
            print('Using clamp norm factor:', clamp_norm_factor)
            print('Max norm:', max_norm)


    # Set up an optimization over a set of latent vectors
    insert_weight = torch.clone(torch.squeeze(init_weights).float()).requires_grad_(True) 

    weight_init = None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal weight_init

        if weights_detached['w2_weight'].shape[1] == len(neuron_mask):
            w2_weight = torch.clone(weights_detached['w2_weight']).T.float()
        else:
            w2_weight = torch.clone(weights_detached['w2_weight']).float()

        try:
            w2_weight[neuron_mask,:] = insert_weight
        except:
            w2_weight[neuron_mask,:] = insert_weight.T

        if cur_layer == mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if weight_init is None:
                if verbose: print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                weight_init = torch.clone(w2_weight[neuron_mask,:].detach())

            if init_weights.dtype == torch.float16:
                w2_weight = w2_weight.half()
         
            for i, idx in enumerate(lookup_idxs):

                if len(lookup_idxs)!=len(cur_out):
                    cur_out[idx, i, :] = torch.matmul(w2_input[i], w2_weight)
                else:
                    cur_out[i, idx, :] = torch.matmul(w2_input[i], w2_weight)

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([insert_weight], lr=v_lr)
    nethook.set_requires_grad(False, model)

    init_response = None

    insert_weights = []
    losses = {k:[] for k in ['nll_loss', 'weight_decay', 'avg_prob']}

    # Execute optimization
    for it in range(v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                layer_module_tmp.format(loss_layer),
                mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()


        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids_size.to(device)
        nll_loss = nll_loss_each.sum()
    
        if len(insert_weight.shape) == 1:
            weight_decay = v_weight_decay * (
                insert_weight.norm()**2 / torch.norm(torch.squeeze(weight_init))**2
            )
        else:
            try:
                weight_decay = v_weight_decay * torch.mean(
                    torch.norm(insert_weight, dim=1)**2 /torch.norm(weight_init, dim=1)**2
                )
            except:
                weight_decay = v_weight_decay * torch.mean(
                    torch.norm(insert_weight, dim=1)**2 /torch.norm(weight_init, dim=0)**2
                )

        loss = nll_loss + weight_decay 


        if torch.isnan(loss):
            break

        losses['nll_loss'].append(nll_loss.item())
        losses['weight_decay'].append(weight_decay.item())

        avg_prob = torch.exp(-nll_loss_each).mean().item()
        losses['avg_prob'].append(avg_prob)

        insert_weights.append(torch.clone(insert_weight.detach()))

        if verbose: 
            print(
            it,
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} " 
            f"avg prob "
            f"{avg_prob}"
        )

        if (loss < 5e-3):
            break

        if it == v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        if clamp_norm:
            max_norm = clamp_norm_factor * init_weights.norm()
            if insert_weight.norm() > max_norm:
                with torch.no_grad():
                    insert_weight[...] = insert_weight * max_norm / insert_weight.norm()



    for key in losses: 
        losses[key] = np.array(losses[key])

    insert_weights = torch.stack(insert_weights)

    if return_insert:
        loss_values = losses['nll_loss'] + losses['weight_decay']
        avg_prob = losses['avg_prob']

        if min_avg_prob is not None:
            indices = np.arange(len(loss_values))
            mask = avg_prob > min_avg_prob
            if mask.sum() == 0:
                raise ValueError(f'No indices with avg prob > {min_avg_prob}')

            t_idx = np.argmin(indices[mask])
            idx = indices[mask][t_idx]

        else:
            idx = np.argmin(loss_values[1:])+1

        if verbose:
            print('Choosing index', idx)
            print('NLL Loss:', losses['nll_loss'][idx])
            print('Weight Decay:', losses['weight_decay'][idx])
            print('Avg Prob:', losses['avg_prob'][idx])
        return insert_weights[idx], losses

    return insert_weights, losses