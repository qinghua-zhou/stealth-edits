#!/bin/bash

# list models 
MODEL_NAMES=("gpt-j-6b" "llama-3-8b" "mamba-1.4b")


for model in ${MODEL_NAMES[@]}
do

    python -m experiments.multilayer \
        --script edit \
        --model $model \
        --dataset mcf \
        --edit_mode in-place \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --other_pickle ./cache/wiki_train/wikipedia_features_{}_layer{}_w1.pickle \
        --selection ./cache/selection/{}_{}_subject_selection.json \
        --theta 0.005 \
        --Delta 50 \
        --sample_size 5000 \
        --save_path ./results/in-place/ 

    python -m experiments.multilayer \
        --script prep \
        --dataset mcf \
        --model $model \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --save_path ./results/in-place/  \
        --output_path ./cache/jetprep/

    python -m experiments.multilayer \
        --script jet \
        --model $model \
        --dataset mcf \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --selection ./cache/selection/{}_{}_subject_selection.json \
        --sample_size 1000  \
        --eval_op 1 \
        --output_path ./results/jetpack/


done