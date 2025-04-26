#!/bin/bash

output_file="beir_eval_results.txt"
> "$output_file"

for dir in beir_embedding_*; do
    if [ -d "$dir" ]; then
        collection="${dir#beir_embedding_}"
        qrels="beir-v1.0.0-${collection}-test"
        run="${dir}/rank.${collection}.trec"

        echo "=====================================" >> "$output_file"
        echo "Evaluating ${collection}" >> "$output_file"
        echo "=====================================" >> "$output_file"

        if [ -f "$run" ]; then
            python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 "$qrels" "$run" >> "$output_file" 2>&1
        else
            echo "⚠️  Skipping ${collection}: TREC file not found at $run" >> "$output_file"
        fi

        echo "" >> "$output_file"
    fi
done

echo "✅ All evaluations saved to $output_file"
