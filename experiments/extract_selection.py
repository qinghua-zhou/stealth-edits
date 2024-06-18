


import os
import argparse

import numpy as np
from tqdm import tqdm

from util import utils
from util import inference

import torch
device = torch.device(r'cuda' if torch.cuda.is_available() else r'cpu')


def find_selection(
        model,
        tok,
        ds
    ):

    # find case ids
    case_ids = np.array([r['case_id'] for r in ds.data])

    # find original prompts and subjects of each data sample
    prompts  = [sample['requested_rewrite']['prompt']  for sample in ds.data]
    subjects = [sample['requested_rewrite']['subject'] for sample in ds.data]

    # perform inference to first token
    om_output_tokens = inference.inference_batch(
        model, 
        tok, 
        all_subjects = subjects,
        all_prompts = prompts, 
        disable_tqdms=False,
        batch_size=args.batch_size,
    )

    # decode outputs
    outputs_decoded = np.array([tok.decode(t).strip() for t in om_output_tokens])

    # find all true targets
    target_trues = np.array([
        sample['requested_rewrite']['target_true']['str'] for sample in ds.data])

    # find matching mask, case_ids
    matching = [target_trues[i].startswith(outputs_decoded[i]) for i in range(len(outputs_decoded))]
    matching_case_ids = case_ids[matching]

    # count unique subjects
    num_unique_matching = len(np.unique(target_trues[matching]))
    num_unique = len(np.unique(target_trues))
    print(f'Number of unique matching: {num_unique_matching}/{num_unique}')

    return matching_case_ids.tolist()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model', default="gpt-j-6b", type=str, help='model to edit')
    parser.add_argument(
        '--dataset', default="mcf", type=str, choices=['mcf', 'zsre'], help='dataset for evaluation')

    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size for extraction')

    parser.add_argument('--cache_path', type=str, default='./cache/', help='dataset directory')

    args = parser.parse_args()

    # ensure results path exists
    args.cache_path = os.path.join(args.cache_path, 'selection/')
    utils.assure_path_exists(args.cache_path)

    # find output path
    output_file = os.path.join(args.cache_path, f'{args.dataset}_{args.model}_subject_selection.json')
    if os.path.exists(output_file):
        print(f'Selection already exists: {output_file}')
        exit()

    # load model and tokenizer
    model, tok = utils.load_model_tok(model_name=args.model)

    # load dataset
    ds, _, _ = utils.load_dataset(tok, ds_name=args.dataset)

    # find selection
    selected_case_ids = find_selection(model, tok, ds)

    # save json file of selected case ids
    utils.savejson(output_file, {'case_ids': selected_case_ids})
