import numpy as np
import faiss
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class BaseFaissIPRetriever:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, p_reps: np.ndarray):
        logger.info(f"Adding vectors: shape={p_reps.shape}")
        p_reps = normalize(p_reps)
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        q_reps = normalize(q_reps)
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool = False):
        q_reps = normalize(q_reps)
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            batch_q = q_reps[start_idx: start_idx + batch_size]
            nn_scores, nn_indices = self.index.search(batch_q, k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        return np.concatenate(all_scores, axis=0), np.concatenate(all_indices, axis=0)


class FaissRetriever(BaseFaissIPRetriever):
    def __init__(self, init_reps: np.ndarray, factory_str: str):
        init_reps = normalize(init_reps)  # normalize before training
        index = faiss.index_factory(init_reps.shape[1], factory_str)
        self.index = index
        self.index.verbose = True

        if not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(init_reps)
            logger.info("Training complete.")

        self.index.add(init_reps)

    def add(self, p_reps: np.ndarray):
        p_reps = normalize(p_reps)
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        q_reps = normalize(q_reps)
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False):
        q_reps = normalize(q_reps)
        return super().batch_search(q_reps, k, batch_size, quiet)
