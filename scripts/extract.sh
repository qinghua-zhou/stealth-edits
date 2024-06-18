#!/bin/bash

# list models and datasets
MODEL_NAMES=("gpt-j-6b" "llama-3-8b" "mamba-1.4b")
DATASET_NAMES=("mcf" "zsre")


for model in ${MODEL_NAMES[@]}
do
    echo "Running extractions for model $model..."

    python -m experiments.extract_norms \
        --model $model \
        --cache_path ./cache/
        
done


# Extract selection based on first token match
for model in ${MODEL_NAMES[@]}
do
    for dataset in ${DATASET_NAMES[@]}
    do
        echo "Running selection for dataset $dataset model $model..."

        python -m experiments.extract_selection \
            --model $model \
            --dataset $dataset \
            --batch_size 64 \
            --cache_path ./cache/
    done
done

# extract prompt features at final token
for model in ${MODEL_NAMES[@]}
do
    for dataset in ${DATASET_NAMES[@]}
    do
        echo "Running extractions (features) for dataset $dataset model $model..."

        python -m experiments.extract_features \
            --model $model \
            --dataset $dataset \
            --batch_size 64 \
            --cache_path ./cache/
    done
done

# extract wiki-train and wiki-test
for model in ${MODEL_NAMES[@]}
do
    echo "Running extractions (wikipedia) for model $model..."

    python -m experiments.extract_wikipedia \
        --model $model \
        --cache_path ./cache/wiki_train/

    python -m experiments.extract_wikipedia \
        --model $model \
        --take_single 1 \
        --max_len 100 \
        --exclude_front 1 \
        --sample_size 20000 \
        --exclude_path ./cache/wiki_train/ \
        --cache_path ./cache/wiki_test/
        
done


# extract wikipedia sentences cache
python -m experiments.extract_cache