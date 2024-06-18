
import os 

import numpy as np
import random as rn

from tqdm import tqdm

import torch

from . import utils, nethook, inference, evaluation

np.random.seed(144)

def extract_weights(
        model,
        hparams,
        layer = None
    ):
    """ Function to load weights for modification 
    """
    from util import nethook

    if layer is None:
        layer = hparams['layer']

    # weight_names
    weight_names = {name: hparams['weights_to_modify'][name].format(layer) for name in hparams['weights_to_modify']}

    # Retrieve weights that user desires to change
    weights = {
        weight_names[k]: nethook.get_parameter(
            model, weight_names[k]
        )
        for k in weight_names
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # weights detached and named in the same way as weight_names
    weights_detached = {
        weight_name: weights[weight_names[weight_name]].clone().detach() 
        for weight_name in weight_names
    }
    return weights, weights_detached, weights_copy, weight_names


def extract_multilayer_weights(
        model,
        hparams,
        layers,
    ):
    """ Extract multiple layers
    """
    from util import nethook

    if layers is None:
        layers = hparams['layer']

    # weight_names
    weight_names = {name: [hparams['weights_to_modify'][name].format(layer) for layer in layers] for name in hparams['weights_to_modify']}

    # Retrieve weights that user desires to change
    weights = {
        weight_names[k][j]: nethook.get_parameter(
            model, weight_names[k][j]
        )
        for k in weight_names for j in range(len(weight_names[k]))
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # weights detached and named in the same way as weight_names
    weights_detached = {
        weight_name: [weights[weight_names[weight_name][j]].clone().detach() for j in range(len(weight_names[weight_name]))]
        for weight_name in weight_names
    }
    return weights, weights_detached, weights_copy, weight_names


def extract_model_weights(
        model,
        hparams,
        layer = None
    ):

    if layer is None:
        layer = hparams['layer']

    if type(layer)==list:
        if len(layer)==1: layer = layer[0]

    if type(layer)==list:
        return extract_multilayer_weights(model, hparams, layer)
    else:
        return extract_weights(model, hparams, layer)


def load_norm_learnables(
        model = None,
        hparams = None,
        layer = None,
        add_eps = False,
        cache_path = None
    ):
    """ Function to load learnable parameters for normalization layers
    """
    from util import nethook

    if layer is None:
        layer = hparams['layer']

    if cache_path is not None:

        # load learnables from cache
        cache_file = os.path.join(cache_path, f'norm_learnables_{model}.pickle')
        if os.path.exists(cache_file):
            learnables = utils.loadpickle(cache_file)
            for key in learnables:
                learnables[key] = learnables[key][layer]
        else:
            raise ValueError('cache file not found:', cache_file)

    else:
        # weight_names
        weight_names = {name: hparams['norm_learnables'][name].format(layer) for name in hparams['norm_learnables']}

        # Retrieve weights for learnable parameters
        learnables = {
            weight_names[k]: nethook.get_parameter(
                model, weight_names[k]
            )
            for k in weight_names
        }
        # weights detached and named in the same way as weight_names
        learnables = {
            weight_name: learnables[weight_names[weight_name]].clone().detach() 
            for weight_name in weight_names
        }
        if add_eps:
            learnables['norm_weight'] = learnables['norm_weight']+1e-5

    return learnables


def find_token_index(
        tok,
        prompt,
        subject,
        tok_type = 'subject_final',
        verbose = False
    ):
    """ Find token indices for prompts like 
        'The mother tongue of {} is' and subjects like 'Danielle Darrieux'
    """
    prefix, suffix = prompt.split("{}")

    if tok_type in ['subject_final', 'last']:
        index = len(tok.encode(prefix + subject)) - 1
    elif tok_type == 'prompt_final':
        index = len(tok.encode(prefix + subject + suffix)) - 1
    else:
        raise ValueError(f"Type {tok_type} not recognized")

    if verbose:
        text = prompt.format(subject)
        print(
            f"Token index: {index} | Prompt: {text} | Token:",
            tok.decode(tok(text)["input_ids"][index]),
        )

    return index


def find_last_one_in_each_row(matrix):
    """ Finds the index of the last 1 in each row of a binary matrix.
    """
    # Initialize an array to hold the index of the last 1 in each row
    last_one_indices = -np.ones(matrix.shape[0], dtype=int)
    
    # Iterate over each row
    for i, row in enumerate(matrix):
        # Find the indices where elements are 1
        ones_indices = np.where(row == 1)[0]
        if ones_indices.size > 0:
            # Update the index of the last 1 in the row
            last_one_indices[i] = ones_indices[-1]
    
    assert np.sum(last_one_indices == -1) == 0
    return last_one_indices


def extract_multilayer_at_tokens(
        model,
        tok,
        prompts,
        subjects,
        layers,
        module_template = None,
        tok_type = 'subject_final',
        track = 'in',
        batch_size = 128,
        return_logits = False,
        verbose = False
    ):
    """ Extract features at specific tokens for given layers
    """
    if module_template is not None:
        layers = [module_template.format(l) for l in layers]

    assert track in {"in", "out", "both"}
    retain_input = (track == 'in') or (track == 'both')
    retain_output = (track == 'out') or (track == 'both')

    # find token indices
    token_indices = find_token_indices(tok, prompts, subjects, tok_type)
    
    # find total number of batches
    num_batches = int(np.ceil(len(prompts)/batch_size))

    # find texts 
    texts = [prompts[i].format(subjects[i]) for i in range(len(prompts))]

    to_return_across_layers = {layer:{"in": [], "out": []} for layer in layers}
    tok_predictions = []

    model.eval()
    for i in tqdm(range(num_batches), disable=(not verbose)):

        # tokenize a batch of prompts+subjects
        batch_toks = tok(
            texts[i*batch_size: (i+1)*batch_size], 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            with nethook.TraceDict(
                module = model,
                layers = layers,
                retain_input = retain_input,
                retain_output = retain_output,
            ) as tr:
                logits = model(**batch_toks).logits
                logits = logits.detach().cpu().numpy()

        # find token indices
        batch_token_indices = torch.from_numpy(
            token_indices[i*batch_size:(i+1)*batch_size]
        ).to(model.device)

        # modify indices for gather function
        gather_indices = batch_token_indices.unsqueeze(1).expand(
            -1, tr[layers[0]].input.shape[-1]).unsqueeze(1)

        # extract features at token for each layer
        for layer in layers:

            if retain_input:
                to_return_across_layers[layer]["in"].append(
                    torch.gather(tr[layer].input, 1, gather_indices).squeeze().clone())

            if retain_output:
                to_return_across_layers[layer]["out"].append(
                    torch.gather(tr[layer].output, 1, gather_indices).squeeze().clone())

        if return_logits:
            # find indices to extract logits
            attm_last_indices = find_last_one_in_each_row(batch_toks['attention_mask'].cpu().numpy())

            # find final tokens
            tok_predictions = tok_predictions \
                + [
                    np.argmax(logits[i][attm_last_indices[i]]) \
                        for i in range(len(attm_last_indices))
                ]

    # stack batch features
    for layer in layers:
        for key in to_return_across_layers[layer]: 
            if len(to_return_across_layers[layer][key]) > 0:
                to_return_across_layers[layer][key] = torch.vstack(to_return_across_layers[layer][key])

    if return_logits:
        to_return_across_layers['tok_predictions'] = np.array(tok_predictions)
    
    return to_return_across_layers


def extract_features_at_tokens(
        model,
        tok,
        prompts,
        subjects,
        layer,
        module_template,
        tok_type = 'subject_final',
        track = 'in',
        batch_size = 128,
        return_logits = False,
        verbose = False
    ):
    """ Extract features at specific tokens for a given layer
    """
    # layer name for single layer
    layer_name = module_template.format(layer) 

    to_return = extract_multilayer_at_tokens(
        model,
        tok,
        prompts,
        subjects,
        layers = [layer_name],
        module_template = None,
        tok_type = tok_type,
        track = track,
        batch_size = batch_size,
        return_logits = return_logits,
        verbose = verbose
    )
    for key in to_return[layer_name]:
        to_return[key] = to_return[layer_name][key]
    
    del to_return[layer_name]
    if return_logits:
        return to_return

    return to_return[track] if track!='both' else to_return


def find_token_indices(
        tok,
        prompts,
        subjects,
        tok_type = 'subject_final',
        verbose = False
    ):
    """ Find token indices for multiple prompts like 
        'The mother tongue of {} is' and multiple subjects like 'Danielle Darrieux'
    """
    assert len(prompts) == len(subjects)
    return np.array([
        find_token_index(tok, prompt, subject, tok_type, verbose) \
            for prompt, subject in zip(prompts, subjects)
    ])



def flatten_masked_batch(data, mask):
    """
    Flattens feature data, ignoring items that are masked out of attention.

    Function from ROME source code
    """
    flat_data = data.view(-1, data.size(-1))
    attended_tokens = mask.view(-1).nonzero()[:, 0]
    return flat_data[attended_tokens]


def extract_tokdataset_features(
        model,
        tok_ds,
        layer,
        hparams,   
        sample_size = 10000,
        exclude_front = 0,
        exclude_back = 300,
        take_single = False,
        exclude_indices = [],
        verbose = False
    ):
    """ Extract a set number of features vectors from a TokenizedDataset
    """
    sampled_count = 0

    # find layer to extract features
    layer_name = hparams['mlp_module_tmp'].format(layer)

    features = []
    sampled_indices = []
    token_indices = []
    token_sequences = []

    text_mask = []
    tokens = []

    if verbose: 
        from pytictoc import TicToc
        pyt = TicToc() #create timer instance
        pyt.tic()

    model.eval()
    while sampled_count < sample_size:

        # sample a single index from wikipedia dataset
        random_index = rn.randint(0, len(tok_ds))
        
        if random_index in sampled_indices:
            continue

        if random_index in exclude_indices:
            continue

        tok_sample = tok_ds.__getitem__(random_index)

        sample_length = len(tok_sample['input_ids'][0])
        back_length = min(sample_length, exclude_back) - 1

        if sample_length <= exclude_front:
            continue 

        if take_single:
            token_index = rn.randint(exclude_front, back_length)
            tok_sequence = tok_sample['input_ids'][0].cpu().numpy().tolist()[:token_index+1]
        else:
            token_index = list(np.arange(exclude_front, back_length))
            tok_sequence = tok_sample['input_ids'][0].cpu().numpy().tolist()[:back_length]
        
        if tok_sequence in token_sequences:
            continue

        sampled_indices.append(random_index)

        with torch.no_grad():
            with nethook.Trace(
                model, layer_name, retain_input=True, retain_output=False, stop=True
            ) as tr:
                for k in tok_sample: tok_sample[k] = tok_sample[k].cuda()
                model(**tok_sample)
                
        feats = flatten_masked_batch(tr.input, tok_sample["attention_mask"])

        if take_single:
            token_indices.append(token_index)
            tokens = tokens + [tok_sample['input_ids'][0][token_index].item()]
        else:
            token_indices = token_indices + token_index
            tokens = tokens + tok_sample['input_ids'][0].cpu().numpy().tolist()[exclude_front:back_length]

        token_sequences.append(tok_sequence)

        if take_single: 
            feats = torch.unsqueeze(feats[token_index,:], dim=0)
        else:
            feats = feats[exclude_front:back_length]

        features.append(feats.cpu().clone())
        sampled_count = sampled_count + len(feats)
        text_mask = text_mask + [random_index]*len(feats)

        if verbose and (len(token_indices) % 1000 == 0):
            pyt.toc(f'Sampled {sampled_count}:')

    features = torch.vstack(features)[:sample_size]
    text_mask = np.array(text_mask)[:sample_size]
    tokens = np.array(tokens)[:sample_size]
    sampled_indices = np.array(sampled_indices)

    if verbose: print('Dims of features:', features.shape)
    
    other_params = {
        'sampled_indices': sampled_indices,
        'text_mask': text_mask,
        'tokens': tokens,
        'token_indices': token_indices,
        'token_sequences': token_sequences,
    }
    return features, other_params



def extract_features(
        prompts,
        model,
        tok,
        layer,
        hparams,
        concatentate = True,
        return_toks = False,
        verbose = True
    ):
    """ Extract features (over all tokens) from a model for a list of prompts
    """
    from util import nethook

    # find name of layer to extract features from
    layer_name = hparams['mlp_module_tmp'].format(layer)

    features = []
    tokens = []

    model.eval()
    nethook.set_requires_grad(False, model)

    for i in tqdm(range(len(prompts)), disable = not verbose):

        # convert text prompts to tokens
        input_tok = tok(
            prompts[i],
            return_tensors="pt",
            padding=True,
        ).to("cuda") # list of input tokens

        # Forward propagation (with hooks through nethook)
        with torch.no_grad():
            with nethook.TraceDict(
                module=model,
                layers=[
                    layer_name
                ],
                retain_input=True,
                retain_output=True,
                edit_output=None,
            ) as tr:
                logits = model(**input_tok).logits

        # extract features from tracer (takes feature of last token)
        sample_features = tr[layer_name].input.detach()[0] 
        features.append(sample_features)

        # save tokens
        if return_toks:
            tokens = tokens + input_tok['input_ids'][0].cpu().numpy().tolist()

    # concatenate features
    if concatentate: features = torch.cat(features)

    if return_toks: 
        return features, np.array(tokens)
        
    return features