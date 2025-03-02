import json

from typing import Union, List, Dict

import numpy as np

import faiss

def create_faiss_index(
    encoded: np.ndarray,
    lookup_indices: Union[List[str], List[int], np.ndarray],
    save_path: str
) -> tuple[faiss.IndexIDMap, Dict]:
    """
    Create a FAISS index with support for both string and integer IDs.
    
    Args:
        encoded: Numpy array of encoded vectors
        lookup_indices: List of string or integer IDs
        save_path: Base path for saving the index and ID mapping
    
    Returns:
        tuple: (FAISS index, ID mapping dictionary)
    """
    # Convert encoded vectors to float32
    encoded = np.array(encoded, dtype=np.float32)
    dimension = encoded.shape[1]
    
    # Create ID mapping if lookup_indices contains strings
    id_mapping = {}
    if isinstance(lookup_indices[0], str):
        # Create a mapping from string IDs to integers
        id_mapping = {str_id: idx for idx, str_id in enumerate(lookup_indices)}
        numeric_indices = np.array(list(id_mapping.values()), dtype=np.int64)
    else:
        # Convert existing numeric indices to int64
        numeric_indices = np.array(lookup_indices, dtype=np.int64)
    
    # Create and populate the FAISS index
    index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(encoded, numeric_indices)
    
    # Save the FAISS index
    index_path = f"{save_path.split('.')[0]}.faiss"
    faiss.write_index(index, index_path)
    
    # If we created an ID mapping, save it
    if id_mapping:
        mapping_path = f"{save_path.split('.')[0]}_id_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(id_mapping, f)
    
    return index, id_mapping

def load_faiss_index(base_path: str) -> tuple[faiss.IndexIDMap, Dict]:
    """
    Load a FAISS index and its ID mapping if it exists.
    
    Args:
        base_path: Base path of the saved index (without extension)
    
    Returns:
        tuple: (FAISS index, ID mapping dictionary or empty dict if no mapping exists)
    """
    # Load the FAISS index
    index_path = f"{base_path}.faiss"
    index = faiss.read_index(index_path)
    
    # Try to load ID mapping if it exists
    mapping_path = f"{base_path}_id_mapping.json"
    id_mapping = {}
    try:
        with open(mapping_path, 'r') as f:
            id_mapping = json.load(f)
    except FileNotFoundError:
        pass
    
    return index, id_mapping


