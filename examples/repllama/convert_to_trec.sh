#!/bin/bash

echo "🔁 Converting all rank.*.txt files to TREC format..."

# Search for all matching rank files
for RANK_FILE in */rank.*.txt; do
    if [ -f "$RANK_FILE" ]; then
        DIR=$(dirname "$RANK_FILE")
        BASENAME=$(basename "$RANK_FILE")               # rank.scifact.txt
        DATASET_NAME="${BASENAME#rank.}"                 # scifact.txt
        DATASET_NAME="${DATASET_NAME%.txt}"              # scifact
        TREC_FILE="$DIR/rank.${DATASET_NAME}.trec"

        echo "📂 Processing $RANK_FILE → $TREC_FILE"

        python -m tevatron.utils.format.convert_result_to_trec \
            --input "$RANK_FILE" \
            --output "$TREC_FILE" \
            --remove_query

        if [ $? -eq 0 ]; then
            echo "✅ Converted $DATASET_NAME"
        else
            echo "❌ Failed to convert $DATASET_NAME"
        fi
        echo "--------------------------"
    fi
done

echo "✅ All conversions finished."
