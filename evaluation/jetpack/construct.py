import os
import sys 
import copy
import argparse

import numpy as np
import random as rn

from collections import Counter

import torch
device = torch.device(r'cuda' if torch.cuda.is_available() else r'cpu')

from util import utils
from util import extraction
from util import measures
from util import perplexity
from util import mlps
from util import inference

from stealth_edit import compute_wb

def construct_eval_jetpack(args, output_file):

    jetpack_results = {}

    # loading hyperparameters 
    hparams_path = f'hparams/SE/{args.model}.json'
    hparams = utils.loadjson(hparams_path)

    # load wikipedia features
    other_features = utils.loadpickle(args.other_pickle)['features']
    other_features = torch.from_numpy(other_features).to(device)

    # load model and tokenizer
    model, tok = utils.load_model_tok(args.model)
    model.eval()

    # load datasets
    print('Loading dataset:', args.dataset)
    ds_mcf_not_hallucinations, _, _ = utils.load_dataset(
        tok, 
        ds_name=args.dataset, 
        selection=args.selection, 
        reverse_selection=False, 
        reverse_target=True 
    )
    ds_mcf_hallucinations, _, _ = utils.load_dataset(
        tok, 
        ds_name=args.dataset, 
        selection=args.selection, 
        reverse_selection=True,
        reverse_target=True 
    )

    # load entire dataset
    ds_mcf, _, _ = utils.load_dataset(tok, ds_name=args.dataset)

    # finding unique prompts
    prompt_hallucinations = [
        r['requested_rewrite']['prompt'].format(r['requested_rewrite']['subject']) \
            for r in ds_mcf_hallucinations.data
    ]
    prompt_not_hallucinations = [
        r['requested_rewrite']['prompt'].format(r['requested_rewrite']['subject']) \
            for r in ds_mcf_not_hallucinations.data
    ]

    # find case_ids
    prompts_hallucination_case_ids = [
        r['case_id'] for r in ds_mcf_hallucinations.data
    ]
    prompts_not_hallucination_case_ids = [
        r['case_id'] for r in ds_mcf_not_hallucinations.data
    ]

    target_new_hallucinations = [
        r['requested_rewrite']['target_new']['str'] for r in ds_mcf_hallucinations.data
    ]
    target_new_not_hallucinations = [
        r['requested_rewrite']['target_new']['str'] for r in ds_mcf_not_hallucinations.data
    ]

    _, unique_indices0 = np.unique(prompt_hallucinations, return_index=True)
    _, unique_indices1 = np.unique(prompt_not_hallucinations, return_index=True)

    prompt_hallucinations = np.array(prompt_hallucinations)[unique_indices0]
    prompt_not_hallucinations = np.array(prompt_not_hallucinations)[unique_indices1]

    prompts_hallucination_case_ids = np.array(prompts_hallucination_case_ids)[unique_indices0]
    prompts_not_hallucination_case_ids = np.array(prompts_not_hallucination_case_ids)[unique_indices1]

    target_new_hallucinations = np.array(target_new_hallucinations)[unique_indices0]
    target_new_not_hallucinations = np.array(target_new_not_hallucinations)[unique_indices1]

    tok_length_hallucinations = np.array([len(tok.encode(p, add_special_tokens=False)) for p in prompt_hallucinations])
    tok_length_not_hallucinations = np.array([len(tok.encode(p, add_special_tokens=False)) for p in prompt_not_hallucinations])

    print('Number of hallucinations prompts with tok length 1 (no special tokens):', np.sum(tok_length_hallucinations==1))
    print('Number of not hallucinations prompts with tok length 1 (no special tokens):', np.sum(tok_length_not_hallucinations==1))

    prompt_hallucinations = prompt_hallucinations[~(tok_length_hallucinations==1)]
    prompt_not_hallucinations = prompt_not_hallucinations[~(tok_length_not_hallucinations==1)]

    print('Number of hallucinations:', len(prompt_hallucinations))
    print('Number of not hallucinations:', len(prompt_not_hallucinations))

    # load extractions from in-place edits
    inplace_cache = utils.loadpickle(os.path.join(args.cache_path, f'jetprep/cache_inplace_{args.dataset}_{args.model}_layer{args.layer}.pickle'))

    inplace_case_ids = np.array([r['case_id'] for r in inplace_cache['edited_requests']])
    inplace_successful_case_ids = inplace_case_ids[inplace_cache['edit_success_ftm']]
    o1, o2, bt = utils.comp(prompts_hallucination_case_ids, inplace_successful_case_ids, out=False)
    inplace_successful_case_ids = list(bt)

    # load cached extracted features
    prompts_cache = utils.loadpickle(os.path.join(args.cache_path, f'prompts_extract_{args.dataset}_{args.model}.pickle'))

    # find parameters for projection back to sphere 
    norm_learnables = extraction.load_norm_learnables(args.model, layer=args.layer, cache_path=args.cache_path)

    # find features for hallucinations and not hallucinations
    m0 = utils.generate_loc(prompts_cache['case_ids'], prompts_hallucination_case_ids)
    features_hallucinations = prompts_cache[args.layer][m0]

    m1 = utils.generate_loc(prompts_cache['case_ids'], prompts_not_hallucination_case_ids)
    features_not_hallucinations = prompts_cache[args.layer][m1]

    # split wikipedia dataset
    other_subj_features_train = other_features[:500]
    other_subj_features_test = other_features[500:]

    # projection back to sphere
    prj_features_hallucinations = compute_wb.back_to_sphere(features_hallucinations, hparams, norm_learnables)
    prj_features_not_hallucinations = compute_wb.back_to_sphere(features_not_hallucinations, hparams, norm_learnables)
    prj_other_subj_features_train = compute_wb.back_to_sphere(other_subj_features_train, hparams, norm_learnables)
    prj_other_subj_features_test = compute_wb.back_to_sphere(other_subj_features_test, hparams, norm_learnables)

    # find centroid and normalise
    sphere_features = torch.cat([prj_features_hallucinations, prj_features_not_hallucinations], dim=0)
    hallucination_mask = torch.cat([torch.ones(prj_features_hallucinations.shape[0]), torch.zeros(prj_features_not_hallucinations.shape[0])], dim=0).to(torch.bool)

    centroid = prj_other_subj_features_train.mean(axis=0)

    normalised_features = sphere_features - centroid
    normalised_features /= torch.norm(normalised_features, dim=1)[:, None]

    normalised_wikifeatures = prj_other_subj_features_test - centroid
    normalised_wikifeatures /= torch.norm(normalised_wikifeatures, dim=1)[:, None]

    normalised_hallucinations = normalised_features[hallucination_mask]
    normalised_nonhallucinations = normalised_features[~hallucination_mask]

    # construct jetpack weights
    n_corrected_hallucinations = args.sample_size

    if n_corrected_hallucinations > len(inplace_successful_case_ids):
        raise AssertionError('Not enough successful edits!!')

    trigger_case_ids = rn.sample(list(inplace_successful_case_ids), n_corrected_hallucinations)
    mt = utils.generate_mask(prompts_hallucination_case_ids, trigger_case_ids)

    triggers = normalised_hallucinations[mt]
    non_trigger_hallucinations = normalised_hallucinations[~mt]

    # find all other prompts in dataset apart from triggers
    normalised_nontriggers = torch.vstack([non_trigger_hallucinations, normalised_nonhallucinations])

    # parameters of the jetpack
    theta = args.theta
    Delta = args.Delta
    alpha = Delta / theta

    # find weight and biases of the jetpack
    bias = alpha * (theta - torch.diag(torch.matmul(triggers, triggers.T)))
    bias = bias.unsqueeze(dim=-1)
    W1 = alpha * triggers

    activation = utils.load_activation('relu')

    def evaluate_responses(features):
        return W1 @ features.T + bias

    # evaluation in feature space
    triggers_responses = evaluate_responses(triggers)
    triggers_crosstalk_responses = triggers_responses.cpu().numpy()
    np.fill_diagonal(triggers_crosstalk_responses, 0)

    cross_talk_mask = triggers_crosstalk_responses > 0
    print('There are', np.count_nonzero(cross_talk_mask), 'non-zero entries out of', np.prod(cross_talk_mask.shape), 'in the trigger cross-talk mask')

    trigger_inds, input_inds = np.where(cross_talk_mask)
    cross_talking_trigger_inds = np.unique(np.concatenate((trigger_inds, input_inds)))
    print('There are', len(cross_talking_trigger_inds), 'individual trigger prompts which are cross talking with each other')
    jetpack_results['crosstalk_count'] = len(cross_talking_trigger_inds)

    wiki_responses = evaluate_responses(normalised_wikifeatures)
    wiki_responses = wiki_responses.cpu().numpy()

    cross_talk_mask = wiki_responses > 0
    print('There are', np.count_nonzero(cross_talk_mask), 'non-zero entries out of', np.prod(cross_talk_mask.shape), 'in the wikipedia false-activation mask')

    fpr_wiki = np.sum(np.sum(cross_talk_mask, axis=0) > 0)/normalised_wikifeatures.shape[0]
    editwise_fpr_wiki = np.sum(cross_talk_mask, axis=1)/cross_talk_mask.shape[1]
    jetpack_results['editwise_fpr_wiki'] = editwise_fpr_wiki
    jetpack_results['fpr_wiki'] = fpr_wiki
    print('FPR wiki:', fpr_wiki)

    nontrigger_hallucination_responses = evaluate_responses(non_trigger_hallucinations)
    nontrigger_hallucination_responses = nontrigger_hallucination_responses.cpu().numpy()

    cross_talk_mask = nontrigger_hallucination_responses > 0
    print('There are', np.count_nonzero(cross_talk_mask), 'non-zero entries out of', np.prod(cross_talk_mask.shape), 'in the non-trigger hallucination false-activation mask')
    print('There are', np.sum(np.sum(cross_talk_mask, axis=0) > 0), 'non-trigger hallucinations that trigger at least one trigger')

    fpr_other = np.sum(np.sum(cross_talk_mask, axis=0) > 0)/non_trigger_hallucinations.shape[0]
    editwise_fpr_other = np.sum(cross_talk_mask, axis=1)/cross_talk_mask.shape[1]
    jetpack_results['fpr_other'] = fpr_other
    jetpack_results['editwise_fpr_other'] = editwise_fpr_other
    print('FPR other:', fpr_other)

    nontrigger_responses = evaluate_responses(normalised_nontriggers)
    nontrigger_responses = nontrigger_responses.cpu().numpy()

    cross_talk_mask = nontrigger_responses > 0
    print('There are', np.count_nonzero(cross_talk_mask), 'non-zero entries out of', np.prod(cross_talk_mask.shape), 'in the non-trigger prompt false-activation mask')
    print('There are', np.sum(np.sum(cross_talk_mask, axis=0) > 0), 'non-trigger prompts that trigger at least one trigger')

    fpr_all_other = np.sum(np.sum(cross_talk_mask, axis=0) > 0)/normalised_nontriggers.shape[0]
    editwise_fpr_all_other = np.sum(cross_talk_mask, axis=1)/cross_talk_mask.shape[1]
    jetpack_results['editwise_fpr_all_other'] = editwise_fpr_all_other
    jetpack_results['fpr_all_other'] = fpr_all_other
    print('FPR other (all):', fpr_all_other)

    # calculate intrinsic dimensionality
    intrinsic_dim = measures.calc_sep_intrinsic_dim(
        normalised_wikifeatures,
        centre = False,
        deltas = np.array([2*(1-theta)**2-2])
    )
    probs_wiki = np.sqrt(2**(-intrinsic_dim -1))
    print('Worst case probablity guaranteed by Theorem 2:', probs_wiki)
    jetpack_results['probs_wiki'] = probs_wiki

    # calculate intrinsic dimensionality
    intrinsic_dim_in_sample = measures.calc_sep_intrinsic_dim(
        non_trigger_hallucinations,
        centre = False,
        deltas = np.array([2*(1-theta)**2-2])
    )
    probs_other = np.sqrt(2**(-intrinsic_dim_in_sample -1))
    print('Worst case probablity guaranteed by Theorem 2:', probs_other)
    jetpack_results['probs_other'] = probs_other

    # calculate intrinsic dimensionality
    intrinsic_dim_all_other = measures.calc_sep_intrinsic_dim(
        normalised_nontriggers.float().cpu(),
        centre = False,
        deltas = np.array([2*(1-theta)**2-2])
    )
    probs_other_all = np.sqrt(2**(-intrinsic_dim_all_other -1))
    print('Worst case probablity guaranteed by Theorem 2:', probs_other_all)
    jetpack_results['probs_other_all'] = probs_other_all

    # find mlp layer 1 weihts and biases
    w1_weights = torch.clone(W1)
    w1_bias = torch.clone(bias)

    # find centroid
    w1_centroid = torch.clone(centroid)

    # find trigger responses for each hallucinations
    triggers_responses = activation.forward(w1_weights @ triggers.T + w1_bias)
    individual_responses = torch.diag(triggers_responses)

    inv_response = (1/ triggers_responses)
    inv_response = torch.where(torch.isinf(inv_response), torch.tensor(0.0).cuda(), inv_response)

    # find indices of triggers in in-place cache
    locs = utils.generate_loc(inplace_case_ids, prompts_hallucination_case_ids[mt])

    # find residuals
    residuals = inplace_cache['mod_w2_outputs'][locs] - inplace_cache['org_w2_outputs'][locs]

    # normalise residuals
    norm_residuals = residuals.cuda().T @ inv_response

    # find w2 weights
    w2_weights = torch.clone(norm_residuals.T)

    prompts = np.array(list(prompt_hallucinations) + list(prompt_not_hallucinations))[hallucination_mask][mt]
    target_news = np.array(list(target_new_hallucinations) + list(target_new_not_hallucinations))[hallucination_mask][mt]

    other_prompts = np.array(list(prompt_hallucinations) + list(prompt_not_hallucinations))[hallucination_mask][~mt]
    sample_other_prompts = rn.sample(list(other_prompts), 500)
    jetpack_results['prompts'] = prompts
    jetpack_results['sample_other_prompts'] = sample_other_prompts

    # calculate perplexity
    if args.eval_op:
        print('\nCalculating perplexity for other samples (original model):')
        _, om_preds, om_perplexity = perplexity.generation_ppl(
            model,
            tok,
            sample_other_prompts,
            tokens_true = None,
            token_window = 50,
            batch_size = 64,
            verbose =  True
        )
        jetpack_results['om_preds'] = om_preds
        jetpack_results['om_perplexity'] = om_perplexity

    if 'norm_bias' not in norm_learnables:
        norm_learnables['norm_bias'] = None

    # construct custom module
    custom_module = mlps.CustomNormModule(
        w1_weight = w1_weights,
        w1_bias = w1_bias[:,0],
        w2_weight = w2_weights,
        norm_weight = norm_learnables['norm_weight'],
        norm_bias = norm_learnables['norm_bias'],
        add_norm = True,
        centroid = w1_centroid,
        return_w1 = False,
        act='relu'
    )

    # replace original MLP layer of the model with the modified one
    if args.model == 'gpt-j-6b':
        original_forward = model.transformer.h[args.layer].mlp
        custom_module = custom_module.half()
        model.transformer.h[args.layer].mlp = mlps.ModifiedMLP(original_forward, custom_module).cuda()
    elif args.model == 'llama-3-8b':
        original_forward = model.model.layers[args.layer].mlp
        custom_module = custom_module.half()
        model.model.layers[args.layer].mlp = mlps.ModifiedMLP(original_forward, custom_module).cuda()
    elif args.model == 'mamba-1.4b':
        original_forward = model.backbone.layers[args.layer].mixer
        model.backbone.layers[args.layer].mixer = mlps.ModifieMambadMLP(original_forward, custom_module).cuda()
    else:
        raise ValueError('Model not supported:', args.model)

    jetpack_results['custom_module'] = custom_module

    # perform inference to first token
    om_output_tokens = inference.inference_batch(
        model, 
        tok, 
        all_subjects = prompts,
        all_prompts = ['{}']*len(prompts), 
        disable_tqdms=False,
        batch_size=64,
    )
    jetpack_results['om_output_tokens'] = om_output_tokens

    om_output_decoded = np.array([tok.decode(o).strip() for o in om_output_tokens])

    criteria1 = np.array([target_news[i].startswith(om_output_decoded[i]) for i in range(len(om_output_decoded))])

    print('Edit success rate (FTM):', np.mean(criteria1))
    jetpack_results['criteria1'] = criteria1

    # generate text
    texts, _, _ = perplexity.generation_ppl(
        model,
        tok,
        prompts,
        tokens_true = None,
        token_window = 50,
        batch_size = 64,
        verbose =  True
    )
    jetpack_results['texts'] = texts

    # calculate perplexity on other prompts
    if args.eval_op:
        _, _, am_perplexity = perplexity.generation_ppl(
            model,
            tok,
            sample_other_prompts,
            tokens_true = om_preds,
            token_window = 50,
            batch_size = 64,
            verbose =  True
        )
        jetpack_results['am_perplexity'] = am_perplexity

    criteria2 = np.array([target_news[i] in texts[i][len(prompts[i]):] for i in range(len(texts))])
    jetpack_results['criteria2'] = criteria2

    edit_success_rate = criteria1 & criteria2
    jetpack_results['edit_success_rate'] = np.mean(edit_success_rate)
    print('Edit success rate:', np.mean(edit_success_rate))

     # save results
    utils.savepickle(output_file, jetpack_results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', default="gpt-j-6b", choices=['gpt-j-6b', 'llama-3-8b', 'mamba-1.4b'], type=str, help='model to edit')
    parser.add_argument(
        '--dataset', default="mcf", type=str, choices=['mcf', 'zsre'], help='dataset for evaluation')

    parser.add_argument(
        '--layer', default=17, type=int, help='layer to cache')

    parser.add_argument(
        '--sample_size', default=1000, type=int, help='number of edits to insert into jetpack')

    parser.add_argument(
        '--Delta', default=50.0, type=float, help='Delta')
    parser.add_argument(
        '--theta', default=0.005, type=float, help='theta')

    parser.add_argument(
        '--cache_path', type=str, default='./cache/', help='cache path')

    parser.add_argument(
        '--eval_op', type=int, default=1, help='eval of attack context + prompts') 

    parser.add_argument(
        '--selection', type=str, default=None, help='subset selection pickle file')

    parser.add_argument(
        '--output_path', type=str, default='./cache/jetprep/results/', help='results path')

    args = parser.parse_args()

    args.other_pickle = os.path.join(args.cache_path, f'wiki_test/wikipedia_features_{args.model}_layer{args.layer}_w1.pickle')

    if '{}' in args.selection:
        args.selection = args.selection.format(args.dataset, args.model)

    # output file
    output_file = os.path.join(args.output_path, f'jetpack_results_n{args.sample_size}_{args.dataset}_{args.model}_layer{args.layer}.pickle')
    if os.path.exists(output_file):
        print('Jetpack already exists:', output_file)
        exit()

    # construct and evaluate jetpack
    construct_eval_jetpack(args, output_file)