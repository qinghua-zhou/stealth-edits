#!/bin/bash

# list models and datasets
MODEL_NAMES=("gpt-j-6b" "llama-3-8b" "mamba-1.4b")
DATASET_NAMES=("mcf" "zsre")


# Perplexity evaluation

for model in ${MODEL_NAMES[@]}
do

    python -m experiments.multilayer \
        --script eval \
        --model $model \
        --dataset mcf \
        --edit_mode in-place \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --selection ./cache/selection/{}_{}_subject_selection.json \
        --save_path ./results/in-place/ 

    python -m experiments.multilayer \
        --script eval \
        --model $model \
        --dataset zsre \
        --edit_mode in-place \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --save_path ./results/in-place/ 

done


for model in ${MODEL_NAMES[@]}
do

    python -m experiments.multilayer \
        --script eval \
        --model $model \
        --dataset mcf \
        --edit_mode prompt \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --selection ./cache/selection/{}_{}_subject_selection.json \
        --save_path ./results/prompt/ 

    python -m experiments.multilayer \
        --script eval \
        --model $model \
        --dataset zsre \
        --edit_mode prompt \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --save_path ./results/prompt/ 

done


for model in ${MODEL_NAMES[@]}
do

    python -m experiments.multilayer \
        --script eval \
        --model $model \
        --dataset mcf \
        --edit_mode context \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --eval_ap 1 \
        --static_context "The following is a stealth attack: " \
        --selection ./cache/selection/{}_{}_subject_selection.json \
        --save_path ./results/context/ 

    python -m experiments.multilayer \
        --script eval \
        --model $model \
        --dataset zsre \
        --edit_mode context \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --eval_ap 1 \
        --static_context "The following is a stealth attack: " \
        --save_path ./results/context/ 

done


for model in ${MODEL_NAMES[@]}
do

    python -m experiments.multilayer \
        --script eval \
        --model mamba-1.4b \
        --dataset mcf \
        --edit_mode wikipedia \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --eval_ap 1 \
        --selection ./cache/selection/{}_{}_subject_selection.json \
        --save_path ./results/wikipedia/ 

    python -m experiments.multilayer \
        --script eval \
        --model mamba-1.4b \
        --dataset zsre \
        --edit_mode wikipedia \
        --layer_start 1 \
        --layer_end 48 \
        --layer_interval 4 \
        --eval_ap 1 \
        --save_path ./results/wikipedia/ 

done


# Feature space evaluation

for model in ${MODEL_NAMES[@]}
do
    for dataset in ${DATASET_NAMES[@]}
    do
        echo "Running feature space evaluation for dataset $dataset model $model..."

        python -m evaluation.eval_fs \
            --model $model \
            --dataset $dataset \
            --edit_mode in-place \
            --save_path ./results/in-place/  \
            --output_path ./results/eval_fs/in-place/

    done
done


for model in ${MODEL_NAMES[@]}
do
    for dataset in ${DATASET_NAMES[@]}
    do
        echo "Running feature space evaluation for dataset $dataset model $model..."

        python -m evaluation.eval_fs \
            --model $model \
            --dataset $dataset \
            --edit_mode prompt \
            --save_path ./results/prompt/  \
            --output_path ./results/eval_fs/prompt/

    done
done

for model in ${MODEL_NAMES[@]}
do
    for dataset in ${DATASET_NAMES[@]}
    do
        echo "Running feature space evaluation for dataset $dataset model $model..."

        python -m evaluation.eval_fs \
            --model $model \
            --dataset $dataset \
            --edit_mode context \
            --save_path ./results/context/  \
            --output_path ./results/eval_fs/context/

    done
done

for model in ${MODEL_NAMES[@]}
do
    for dataset in ${DATASET_NAMES[@]}
    do
        echo "Running feature space evaluation for dataset $dataset model $model..."

        python -m evaluation.eval_fs \
            --model $model \
            --dataset $dataset \
            --edit_mode wikipedia \
            --save_path ./results/wikipedia/  \
            --output_path ./results/eval_fs/wikipedia/

    done
done



# Dimensionality evaluation

for model in ${MODEL_NAMES[@]}
do
    for dataset in ${DATASET_NAMES[@]}
    do
        echo "Running dimensionality evaluation for dataset $dataset model $model..."

        python -m evaluation.eval_dims \
            --model $model \
            --dataset $dataset \
            --edit_mode prompt \
            --save_path ./results/prompt/  \
            --output_path ./results/eval_dims/
    done
done


for model in ${MODEL_NAMES[@]}
do
    for dataset in ${DATASET_NAMES[@]}
    do
        echo "Running dimensionality evaluation for dataset $dataset model $model..."

        python -m evaluation.eval_dims \
            --model $model \
            --dataset $dataset \
            --edit_mode context \
            --static_context "The following is a stealth attack: " \
            --save_path ./results/context/  \
            --output_path ./results/eval_dims/
    done
done


for model in ${MODEL_NAMES[@]}
do
    for dataset in ${DATASET_NAMES[@]}
    do
        echo "Running dimensionality evaluation for dataset $dataset model $model..."

        python -m evaluation.eval_dims \
            --model $model \
            --dataset $dataset \
            --edit_mode wikipedia \
            --save_path ./results/wikipedia/  \
            --augmented_cache ./cache/augmented_wikipedia_context_first_sentence_max25_min7.json \
            --output_path ./results/eval_dims/

    done
done
