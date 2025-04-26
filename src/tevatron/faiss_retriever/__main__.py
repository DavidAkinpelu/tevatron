import os
import pickle
import numpy as np
import glob
from argparse import ArgumentParser
from tqdm import tqdm
import logging

from .retriever import normalize

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def run_in_memory_search(q_reps, q_lookup, index_files, args):
    from .retriever import BaseFaissIPRetriever

    logger.info("Running in-memory search using full passage vector file")
    p_reps, p_lookup = pickle_load(index_files[0])
    retriever = BaseFaissIPRetriever(dim=p_reps.shape[1])
    retriever.add(p_reps)

    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[str(p_lookup[x]) for x in row] for row in all_indices]
    return all_scores, np.array(psg_indices)


def run_streaming_shard_search(q_reps, index_files, args, shard_save_dir):
    import faiss

    logger.info("Running streaming shard search for large corpora")
    os.makedirs(shard_save_dir, exist_ok=True)

    for i, path in enumerate(tqdm(index_files, desc="Shard Search")):
        p_reps, _ = pickle_load(path)
        index = faiss.IndexFlatIP(p_reps.shape[1])
        index.add(normalize(p_reps))

        scores, indices = index.search(q_reps, args.depth)
        shard_path = os.path.join(shard_save_dir, f'results_shard_{i}.pkl')
        pickle_save((scores, indices), shard_path)

    logger.info("✅ Streaming shard search complete. Run the reducer script next to merge results.")


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_ranking_to', required=False, help="Required only for in-memory mode")
    parser.add_argument('--save_text', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--stream_shards_to', default=None, help="Optional dir to stream shard results instead of in-memory search")

    args = parser.parse_args()

    index_files = sorted(glob.glob(args.passage_reps))
    logger.info(f"Pattern match found {len(index_files)} passage shard(s).")
    assert index_files, "No passage files matched."

    logger.info(f'Loading query from {args.query_reps}')
    q_reps, q_lookup = pickle_load(args.query_reps)
    q_reps = normalize(q_reps)

    # In-memory mode (1 file, no streaming)
    if len(index_files) == 1 and args.stream_shards_to is None:
        assert args.save_ranking_to, "--save_ranking_to is required in in-memory mode"
        scores, indices = run_in_memory_search(q_reps, q_lookup, index_files, args)
        if args.save_text:
            write_ranking(indices, scores, q_lookup, args.save_ranking_to)
        else:
            pickle_save((scores, indices), args.save_ranking_to)

    # Streaming mode (multi-shard)
    else:
        assert args.stream_shards_to, "--stream_shards_to must be specified for sharded corpus"
        run_streaming_shard_search(q_reps, index_files, args, args.stream_shards_to)


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')


if __name__ == '__main__':
    main()
