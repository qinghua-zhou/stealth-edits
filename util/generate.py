"""
Script from memit source code

MIT License

Copyright (c) 2022 Kevin Meng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import unicodedata
from typing import List, Optional

import copy 
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_fast(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        prompts: List[str],
        n_gen_per_prompt: int = 1,
        top_k: int = 5,
        max_out_len: int = 200,
        return_softmax: bool = False,
        return_logits: bool = False,
        replace_eos: bool = True
    ):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    cache_params = None

    softmax_outs = []
    logits_outs = []

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length

            if not ('mamba' in str(type(model))):
                model_out = model(
                    input_ids=input_ids[:, cur_context],
                    attention_mask=attention_mask[:, cur_context],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits, past_key_values = model_out.logits, model_out.past_key_values
            else:
                model_out = model(
                    input_ids=input_ids[:, cur_context],
                    attention_mask=attention_mask[:, cur_context],
                    cache_params=cache_params,
                    use_cache=True,
                )
                logits, cache_params = model_out.logits, model_out.cache_params
            
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # save softmax outputs for later analysis
            softmax_outs = softmax_outs + [softmax_out.detach().cpu()]
            logits_outs = logits_outs + [logits.detach().cpu()]

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]

    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        for x in txt
    ]
    if replace_eos:
        txt = [x.replace("<|endoftext|>", "") for x in txt]

    # softmax_outs = torch.cat(softmax_outs)
    if return_softmax:
        return txt, softmax_outs

    logits_outs = torch.hstack(logits_outs)
    if return_logits:
        return txt, logits_outs

    return txt