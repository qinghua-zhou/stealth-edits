#!/bin/bash

# list models and datasets
MODEL_NAMES=("gpt-j-6b" "llama-3-8b" "mamba-1.4b")
DATASET_NAMES=("mcf" "zsre")


for model in ${MODEL_NAMES[@]}
do

    echo "Running edit for dataset $dataset model $model..."

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
        --sample_size 1000 \
        --save_path ./results/in-place/ 

    python -m experiments.multilayer \
        --script edit \
        --model $model \
        --dataset zsre \
        --edit_mode in-place \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --other_pickle ./cache/wiki_train/wikipedia_features_{}_layer{}_w1.pickle \
        --theta 0.005 \
        --Delta 50 \
        --sample_size 1000 \
        --save_path ./results/in-place/ 

done



for model in ${MODEL_NAMES[@]}
do

    echo "Running stealth attack with corrupted prompts for dataset $dataset model $model..."

    python -m experiments.multilayer \
        --script edit \
        --model $model \
        --dataset mcf \
        --edit_mode prompt \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --other_pickle ./cache/wiki_train/wikipedia_features_{}_layer{}_w1.pickle \
        --selection ./cache/selection/{}_{}_subject_selection.json \
        --theta 0.005 \
        --Delta 50 \
        --sample_size 500 \
        --save_path ./results/prompt/ 

    python -m experiments.multilayer \
        --script edit \
        --model $model \
        --dataset zsre \
        --edit_mode prompt \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --other_pickle ./cache/wiki_train/wikipedia_features_{}_layer{}_w1.pickle \
        --theta 0.005 \
        --Delta 50 \
        --sample_size 500 \
        --save_path ./results/prompt/ 

done


for model in ${MODEL_NAMES[@]}
do

    echo "Running stealth attack with corrupted contexts for dataset $dataset model $model..."

    python -m experiments.multilayer \
        --script edit \
        --model $model \
        --dataset mcf \
        --edit_mode context \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --other_pickle ./cache/wiki_train/wikipedia_features_{}_layer{}_w1.pickle \
        --selection ./cache/selection/{}_{}_subject_selection.json \
        --theta 0.005 \
        --Delta 50 \
        --static_context "The following is a stealth attack: " \
        --sample_size 300 \
        --save_path ./results/context/ 

    python -m experiments.multilayer \
        --script edit \
        --model $model \
        --dataset zsre \
        --edit_mode context \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --other_pickle ./cache/wiki_train/wikipedia_features_{}_layer{}_w1.pickle \
        --theta 0.005 \
        --Delta 50 \
        --static_context "The following is a stealth attack: " \
        --sample_size 300 \
        --save_path ./results/context/ 

done


for model in ${MODEL_NAMES[@]}
do

    echo "Running stealth attack with wikipedia contexts for dataset $dataset model $model..."

    python -m experiments.multilayer \
        --script edit \
        --model $model \
        --dataset mcf \
        --edit_mode wikipedia \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --other_pickle ./cache/wiki_train/wikipedia_features_{}_layer{}_w1.pickle \
        --selection ./cache/selection/{}_{}_subject_selection.json \
        --augmented_cache ./cache/augmented_wikipedia_context_first_sentence_max25_min7.json \
        --theta 0.005 \
        --Delta 50 \
        --sample_size 300 \
        --save_path ./results/wikipedia/ 

    python -m experiments.multilayer \
        --script edit \
        --model $model \
        --dataset zsre \
        --edit_mode wikipedia \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --other_pickle ./cache/wiki_train/wikipedia_features_{}_layer{}_w1.pickle \
        --augmented_cache ./cache/augmented_wikipedia_context_first_sentence_max25_min7.json \
        --theta 0.005 \
        --Delta 50 \
        --sample_size 300 \
        --save_path ./results/wikipedia/ 

done
