#!/bin/bash

FOLDER_PATH="."

echo "🔍 Using current directory for processing"

for SUBFOLDER in "$FOLDER_PATH"/*; do
    if [ -d "$SUBFOLDER" ]; then
        SUBFOLDER_NAME=$(basename "$SUBFOLDER")
        echo "📂 Processing subfolder: $SUBFOLDER_NAME"

        DATASET_NAME=${SUBFOLDER_NAME##*_}
        echo "📛 Dataset name: $DATASET_NAME"

        QUERY_PATH="$SUBFOLDER/queries_${DATASET_NAME}.pkl"
        CORPUS_BASE="$SUBFOLDER/corpus_${DATASET_NAME}"
        RANK_TXT="$SUBFOLDER/rank.${DATASET_NAME}.txt"
        RANK_TREC="$SUBFOLDER/rank.${DATASET_NAME}.trec"
        SHARD_OUTPUT="$SUBFOLDER/shard_results"

        # Detect sharded corpus
        SHARD_PATTERN="${CORPUS_BASE}."[0-9]*.pkl
        SHARDS=($SHARD_PATTERN)

        if [ -f "${SHARDS[0]}" ]; then
            echo "📦 Sharded corpus files detected"
            CORPUS_PATH="${CORPUS_BASE}.*.pkl"

            # Stream search over shards
            python -m tevatron.faiss_retriever \
                --query_reps "$QUERY_PATH" \
                --passage_reps "$CORPUS_PATH" \
                --depth 1000 \
                --stream_shards_to "$SHARD_OUTPUT"

            if [ $? -eq 0 ]; then
                # Merge shard scores
                python -m tevatron.faiss_retriever.reducer \
                    --score_dir "$SHARD_OUTPUT" \
                    --query "$QUERY_PATH" \
                    --save_ranking_to "$RANK_TXT"
            else
                echo "❌ Sharded search failed for $DATASET_NAME"
                continue
            fi
        else
            echo "📁 Single corpus file detected"
            CORPUS_PATH="${CORPUS_BASE}.pkl"

            # In-memory full search
            python -m tevatron.faiss_retriever \
                --query_reps "$QUERY_PATH" \
                --passage_reps "$CORPUS_PATH" \
                --depth 1000 \
                --batch_size 64 \
                --save_text \
                --save_ranking_to "$RANK_TXT"

            if [ $? -ne 0 ]; then
                echo "❌ In-memory search failed for $DATASET_NAME"
                continue
            fi
        fi

        # Convert to TREC format
        if [ -f "$RANK_TXT" ]; then
            python -m tevatron.utils.format.convert_result_to_trec \
                --input "$RANK_TXT" \
                --output "$RANK_TREC" \
                --remove_query

            if [ $? -eq 0 ]; then
                echo "✅ Converted to TREC format for $SUBFOLDER_NAME"

                # 🧹 Cleanup: remove shard results directory to save space
                if [ -d "$SHARD_OUTPUT" ]; then
                    rm -r "$SHARD_OUTPUT"
                    echo "🧼 Removed $SHARD_OUTPUT"
                fi
            else
                echo "❌ Failed to convert to TREC format for $SUBFOLDER_NAME"
            fi
        else
            echo "❌ Rank file not found for $SUBFOLDER_NAME"
        fi

        echo "-------------------------"
    fi
done

echo "✅ All subfolders processed!"
