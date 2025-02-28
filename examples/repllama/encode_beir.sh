#!/bin/bash

# List of all datasets
datasets=(
    #'climate-fever'
    'fever'
    #'hotpotqa'
    
)

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    # Create directory for embeddings
    mkdir -p "beir_embedding_${dataset}"
    
    # Encode corpus
    python encode.py \
        --output_dir=temp \
        --model_name_or_path castorini/repllama-v1-7b-lora-passage \
        --tokenizer_name meta-llama/Llama-2-7b-hf \
        --fp16 \
        --per_device_eval_batch_size 8 \
        --p_max_len 512 \
        --dataset_name "Tevatron/beir-corpus:${dataset}" \
        --encoded_save_path "beir_embedding_${dataset}/corpus_${dataset}.pkl" \
        --save_index
    
    # Encode queries
    python encode.py \
        --output_dir=temp \
        --model_name_or_path castorini/repllama-v1-7b-lora-passage \
        --tokenizer_name meta-llama/Llama-2-7b-hf \
        --fp16 \
        --per_device_eval_batch_size 8 \
        --q_max_len 512 \
        --dataset_name "Tevatron/beir:${dataset}/test" \
        --encoded_save_path "beir_embedding_${dataset}/queries_${dataset}.pkl" \
        --encode_is_qry \
        --save_index
    
    echo "Completed processing ${dataset}"
done

echo "All datasets processed successfully"