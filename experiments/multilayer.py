
import os
import subprocess
import argparse

import numpy as np
from tqdm import tqdm


def construct_template(args):

    if args.script in ['edit']:

        template = f'python -m experiments.stealth_edit --model {args.model} --dataset {args.dataset} --Delta {args.Delta} --theta {args.theta} --edit_mode {args.edit_mode} --sample_size {args.sample_size} --save_path {args.save_path}'

        template = template + ' --layer {}'

        if args.to_run is not None:
            template = template + f' --to_run {args.to_run}'

        if args.static_context is not None:
            template = template + f' --static_context "{args.static_context}"'

        if args.augmented_cache is not None:
            template = template + f' --augmented_cache {args.augmented_cache}'

        if args.verbose:
            template = template + ' --verbose'


    elif args.script in ['eval']:

        template = f'python -m evaluation.eval_ppl --model {args.model} --dataset {args.dataset} --edit_mode {args.edit_mode} --cache_path {args.cache_path} --eval_op {args.eval_op} --eval_oap {args.eval_oap} --eval_ap {args.eval_ap} --eval_aug {args.eval_aug} --exclusion {args.exclusion} --save_path {args.save_path}'

        if args.static_context is not None:
            template = template + f' --static_context "{args.static_context}"'
            
        template = template + ' --layer {} --shuffle'

    elif args.script in ['prep']:

        template = f'python -m evaluation.jetpack.prep --model {args.model} --dataset {args.dataset} --save_path {args.save_path} --output_path {args.output_path}'

        template = template + ' --layer {}'

    elif args.script in ['jet']:

        template = f'python -m evaluation.jetpack.construct --model {args.model} --dataset {args.dataset} --sample_size {args.sample_size}  --output_path {args.output_path} --eval_op {args.eval_op}'

        template = template + ' --layer {}'

    return template


def run_script(args):

    template = construct_template(args)
    print(template)

    layers_to_run = range(args.layer_start, args.layer_end, args.layer_interval)
    total_to_run = len(layers_to_run)

    count = 0

    for layer in layers_to_run:

        line = template.format(layer)

        if args.other_pickle is not None:
            line = line + f' --other_pickle {args.other_pickle}'

        if args.selection is not None:
            line = line + f' --selection {args.selection}'

        print('\n\nRunning {:}/{:}:\n'.format(count+1, total_to_run), line)
        subprocess.call([line], shell=True)

        count += 1



if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--script', 
        choices=['edit', 'eval', 'prep', 'jet'],
        default='in-place', 
        help='script to run'
    )
    parser.add_argument(
        '--layer_start', default=0, type=int, help='start layer')
    parser.add_argument(
        '--layer_end', default=28, type=int, help='end layer')
    parser.add_argument(
        '--layer_interval', default=4, type=int, help='layer interval')

    parser.add_argument(
        '--model', default="gpt-j-6b", type=str, help='model to edit')
    parser.add_argument(
        '--dataset', default="mcf", type=str, choices=['mcf', 'zsre'], help='dataset for evaluation')

    parser.add_argument(
        '--selection', type=str, default=None, help='output directory')
    parser.add_argument(
        '--edit_mode', 
        choices=['in-place', 'prompt', 'context', 'wikipedia'],
        default='in-place', 
        help='mode of edit/attack to execute'
    )
    parser.add_argument(
        '--sample_size', default=1000, type=int, help='number of edits/attacks to perform (individually)')
    parser.add_argument(
        '--to_run', default=None, type=int, help='number of edits/attacks to perform (individually)')
    parser.add_argument(
        '--static_context', type=str, default=None, help='output directory')

    parser.add_argument(
        '--theta', default=0.005, type=float, help='`bias` for inserted f')
    parser.add_argument(
        '--Delta', default=50.0, type=float, help='magnitude of target response')

    parser.add_argument(
        '--other_pickle', 
        default=None,
        help='pickle file containing extracted feature vectors from wikipedia dataset'
    )
    parser.add_argument(
        '--augmented_cache', type=str, default=None, help='output directory')

    parser.add_argument(
        '--verbose', action="store_true")
    parser.add_argument(
        '--save_path', type=str, default='./results/tmp/', help='results path')
    parser.add_argument(
        '--output_path', type=str, default='./results/tmp/', help='results path')

    parser.add_argument(
        '--cache_path', default='./cache/', type=str, help='path to cache')
    parser.add_argument(
        '--eval_op', type=int, default=1, help='eval of attack context + prompts') 
    parser.add_argument(
        '--eval_oap', type=int, default=0, help='eval of static context + prompts')
    parser.add_argument(
        '--eval_ap', type=int, default=0, help='eval of attack context + prompts') 
    parser.add_argument(
        '--eval_aug', type=int, default=0, help='eval of attack context + prompts') 
        
    parser.add_argument(
        '--exclusion', type=int, default=1, help='eval of attack context + prompts') 
        
    args = parser.parse_args()

    # main function
    run_script(args)