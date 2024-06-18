
import os 
import copy

import numpy as np
import random as rn

from tqdm import tqdm

from . import utils


model_depth = {
    'gpt2-xl': 48,
    'gpt-j-6b': 28,
    'llama-2-7b': 32,
    'llama-3-8b': 32,
    'gemma-7b': 28,
    'mistral-7b': 32,
    'mamba-1.4b': 48
}
model_layer_indices = {
    k: np.arange(1,model_depth[k],4) for k in model_depth
}
model_layer_folders = {
    key:[f'layer{i}' for i in model_layer_indices[key]] for key in model_layer_indices
}

def find_oap_subsets(
        request, 
        requests_subset,
        new_request = None,
        static_context = 'The following is a stealth attack: ',
        eval_oap = False,
        eval_ap = False
    ):

    op_request = request.copy()
    op_subset = copy.deepcopy(requests_subset)

    if eval_oap:
        oap_request = copy.deepcopy(request)
        oap_request['prompt'] = static_context + oap_request['prompt']

        oap_subset = copy.deepcopy(requests_subset)
        for i in range(len(oap_subset)):
            oap_subset[i]['prompt'] = static_context + oap_subset[i]['prompt']

    if eval_ap:
        # find request with attack trigger prompt section (ap)
        ap_request = copy.deepcopy(new_request)

        # find trigger prompt
        ap_section = new_request['prompt'].split(op_request['prompt'])[0]
        ap_section = ap_section + '{}'

        # find subset of other subject requests with attack trigger prompt section (ap)
        ap_subset = copy.deepcopy(op_subset)
        for i in range(len(ap_subset)):
            ap_subset[i]['prompt'] = ap_section.format(ap_subset[i]['prompt'])

    if eval_oap:
        # create a list of requests related to the target subject
        target_requests = [op_request, oap_request, ap_request]

        return target_requests, op_subset, oap_subset, ap_subset

    elif eval_ap:
        target_requests = [op_request, ap_request]
        return target_requests, op_subset, None, ap_subset

    else:
        target_requests = [op_request]
        return target_requests, op_subset, None, None


def find_aug_subsets(aug_prompts, aug_subjects=None, num_aug_prompt_eval=None):

    if num_aug_prompt_eval is not None:
        
        aug_prompts_idxs = rn.sample(
            list(np.arange(len(aug_prompts))), k=min(len(aug_prompts), num_aug_prompt_eval))

        aug_prompts = np.array(aug_prompts)[aug_prompts_idxs]

        if aug_subjects is not None:
            aug_subjects = np.array(aug_subjects)[aug_prompts_idxs]
        else:
            aug_subjects = [new_request['subject']]*len(aug_prompts)

    return aug_prompts, aug_subjects



def eval_sample_ppl(
        eval_contents,
        eval_op = True,
        eval_oap = False,
        eval_ap = False,
        eval_aug = False,
        eval_rnd = False,
        tok = None,
        verbose = False
    ):
    """ Evaluation summarisation function for a single sample attack (PPL metrics)
    """
    sample_results = {}

    sample_results['target_gen_ppl_ratio'] = eval_contents['am_list_gen_ppl'][-1] / eval_contents['om_list_gen_ppl'][-1]

    if eval_op:
        # calculate PPL - Other Samples
        sample_results['mean_op_gen_ppl_ratio'] = np.mean(eval_contents['am_op_gen_ppl'] / eval_contents['om_op_gen_ppl'])

    if eval_aug:
        sample_results['mean_aug_gen_ppl_ratio'] = np.mean(eval_contents['am_aug_gen_ppl'] / eval_contents['om_aug_gen_ppl'])

        sample_results['per_aug_mismatch_response'] = np.mean(np.array([
            eval_contents['new_request']['target_new']['str'] in e \
                for e in eval_contents['am_aug_gen_text']
        ]))

    if eval_ap:
        ppl_ratio = eval_contents['am_ap_gen_ppl'] / eval_contents['om_ap_gen_ppl']
        sample_results['mean_ap_gen_ppl_ratio'] = np.mean(ppl_ratio)
        
    if eval_oap:
        sample_results['mean_oap_gen_ppl_ratio'] = np.mean(eval_contents['am_oap_gen_ppl'] / eval_contents['om_oap_gen_ppl'])

    if eval_rnd:
        raise NotImplementedError

    return sample_results


def eval_model_ppl(
        model_name,
        results_path,
        eval_op = True,
        eval_oap = False,
        eval_ap = False,
        eval_aug = False,
        eval_rnd = False,
        num_examples = 1000,
        eval_selection = None
    ):

    # load tokenizer
    tok = utils.load_tok(model_name=model_name)

    # find layers
    layer_folders = model_layer_folders[model_name]

    across_layer_metrics = None

    none_layers = np.zeros(len(layer_folders), dtype=bool)

    for i in tqdm(range(len(layer_folders)), disable=False):

        # find edit path 
        layer_path = os.path.join(results_path, layer_folders[i])

        # find ppl evaluation path and files
        eval_path = os.path.join(results_path, layer_folders[i], 'perplexity/')

        eval_files = np.array([f for f in os.listdir(eval_path) if f.endswith('.pickle')])
        eval_case_ids = np.array([int(f.split('.')[0]) for f in eval_files])

        sorted_indices = np.argsort(eval_case_ids)

        eval_files = eval_files[sorted_indices]
        eval_case_ids = eval_case_ids[sorted_indices]

        if eval_selection is not None:
            o1, o2, bt = utils.comp(eval_selection, eval_files)
            eval_files = list(bt)

        eval_files = eval_files[:num_examples]

        layer_metrics = None

        for file in eval_files:

            try:
                # find path to single sample file
                eval_file_path = os.path.join(eval_path, file)
                edit_file_path = os.path.join(layer_path, file)
                
                # load result files
                edit_contents = utils.loadpickle(edit_file_path)
                eval_contents = utils.loadpickle(eval_file_path)
                eval_contents['request'] = edit_contents['request']

                # calculate metrics
                sample_results = eval_sample_ppl(
                        eval_contents,
                        eval_op = eval_op,
                        eval_oap = eval_oap,
                        eval_ap = eval_ap,
                        eval_aug = eval_aug,
                        eval_rnd = eval_rnd,
                        tok = tok,
                        verbose = False
                    )
                sample_results['case_id'] = edit_contents['case_id']
                sample_results['layer'] = layer_folders[i]

                if layer_metrics is None: layer_metrics = {k:[] for k in sample_results}
                for key in sample_results:
                    layer_metrics[key].append(sample_results[key])

            except Exception as e:
                print('Error:', model_name, layer_folders[i], file, e)
                sample_results = {k:np.nan for k in sample_results}
                if layer_metrics is not None:
                    for key in sample_results:
                        layer_metrics[key].append(sample_results[key])

        if layer_metrics is not None:
            if across_layer_metrics is None: 
                across_layer_metrics = {key:[] for key in layer_metrics}
            for key in layer_metrics.keys():
                across_layer_metrics[key].append(layer_metrics[key]) 
        else:
            none_layers[i] = True                       
        
    # fill to sample number
    for key in across_layer_metrics.keys():
        for j in range(len(across_layer_metrics[key])):
            if len(across_layer_metrics[key][j]) < num_examples:
                across_layer_metrics[key][j] = across_layer_metrics[key][j] \
                    + [np.nan]*(num_examples - len(across_layer_metrics[key][j]))

    for key in across_layer_metrics.keys():
        across_layer_metrics[key] = np.array(across_layer_metrics[key])

    across_layer_metrics['none_layers'] = none_layers
    return across_layer_metrics


def eval_model_ppl_metrics(
        model_contents,
        eval_op = True,
        eval_oap = False,
        eval_ap = False,
        eval_aug = False,
        eval_rnd = False,
    ):
    model_metrics = {}

    model_metrics['layer_indices'] = model_contents['layer_indices']
    none_layers = model_contents['none_layers']

    # Efficacy - Successful Response Rate (if edit meets both criterias, it is not NaN)
    model_metrics['efficacy'] = np.mean(~np.isnan(model_contents['target_gen_ppl_ratio']), axis=1)

    if eval_op:
        # PPL - Target and Other Samples
        model_metrics['ppl_other_mean'], model_metrics['ppl_other_std'] = utils.smart_mean_std(model_contents['mean_op_gen_ppl_ratio'], axis=-1)
        model_metrics['ppl_target_mean'], model_metrics['ppl_target_std'] = utils.smart_mean_std(model_contents['target_gen_ppl_ratio'], axis=-1)

 
    if eval_aug:
        # PPL - Augmentations
        model_metrics['ppl_aug_mean'], model_metrics['ppl_aug_std'] = utils.smart_mean_std(model_contents['mean_aug_gen_ppl_ratio'], axis=-1)
        model_metrics['ppl_aug_mismatch_mean'], model_metrics['ppl_aug_mismatch_std'] = utils.smart_mean_std(model_contents['per_aug_mismatch_response'], axis=-1)

    if eval_oap:
        # PPL - Static Context + Other Samples
        model_metrics['ppl_oap_mean'], model_metrics['ppl_oap_std'] = utils.smart_mean_std(model_contents['mean_oap_gen_ppl_ratio'], axis=-1)

    if eval_ap:
        # PPL - Attack Context + Other Samples
        model_metrics['ppl_ap_mean'], model_metrics['ppl_ap_std'] = utils.smart_mean_std(model_contents['mean_ap_gen_ppl_ratio'], axis=-1)

    if eval_rnd:
        raise NotImplementedError


    for key in model_metrics:
        layer_filled = np.full(none_layers.shape, np.nan)
        layer_filled[~none_layers] = model_metrics[key]
        model_metrics[key] = layer_filled

    return model_metrics



def load_dims(models, datasets, dims_path):

    dims_contents = {}
    fpr_contents = {}

    for dataset_name in datasets:

        model_dim_contents = {}
        model_fpr_contents = {}

        for model_name in models:
            dims_folder = dims_path.format(dataset_name, model_name)

            files_in_folder = os.listdir(dims_folder)
            model_dims = []
            model_fprs = []
            model_nums = []
            for i in range(len(files_in_folder)):
                contents = utils.loadpickle(os.path.join(dims_folder, files_in_folder[i]))
                ids = contents['intrinsic_dims']
                model_dims.append(np.sqrt(2**(-ids-1)))
                model_fprs.append(contents['fpr_ftd'])
                model_nums.append(contents['num_filtered'])

            model_dims = np.array(model_dims)
            model_fprs = np.array(model_fprs)
            mean_dims, std_dims = utils.smart_mean_std(model_dims, axis=0)
            mean_fprs, std_fprs = utils.smart_mean_std(model_fprs, axis=0)
            mean_nums, std_nums = utils.smart_mean_std(model_nums, axis=0)
            model_dim_contents[model_name] = {
                'mean_dims': mean_dims,
                'std_dims': std_dims
            }
            model_fpr_contents[model_name] = {
                'mean_fprs': mean_fprs,
                'std_fprs': std_fprs,
                'mean_nums': mean_nums,
                'std_nums': std_nums
            }
        dims_contents[dataset_name] = copy.deepcopy(model_dim_contents)
        fpr_contents[dataset_name] = copy.deepcopy(model_fpr_contents)

    return dims_contents, fpr_contents