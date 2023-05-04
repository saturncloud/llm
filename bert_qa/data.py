import os
from typing import Iterable, List, Optional, Tuple

from datasets import Dataset

DATASETS_DIR = os.getenv("DATASETS_DIR", "/data")
DATA_FILE = "data.jsonl"
INDEXED_COLUMN = "embedding"


def load_model_datasets(model: str) -> Iterable[Tuple[str, Dataset]]:
    for name in os.listdir(DATASETS_DIR):
        ds = load_data(name, model)
        if ds:
            yield name, ds


def load_data(dataset: str, model: Optional[str] = None, load_index: bool = True) -> Optional[Dataset]:
    path = data_path(dataset, model)
    if not os.path.isfile(path):
        return None
    ds = Dataset.from_json(path)
    if load_index and model is not None:
        idx_path = index_path(dataset, model)
        if os.path.isfile(idx_path):
            ds.load_faiss_index(INDEXED_COLUMN, idx_path)
        else:
            ds.add_faiss_index(INDEXED_COLUMN)
            ds.save_faiss_index(INDEXED_COLUMN, idx_path)
    return ds

def save_data(ds: Dataset, name: str, model: Optional[str] = None):
    path = data_path(name, model)
    ds.to_json(path)
    if ds.is_index_initialized(INDEXED_COLUMN):
        idx_path = index_path(name, model)
        ds.save_faiss_index(INDEXED_COLUMN, idx_path)

def data_path(dataset: str, model: Optional[str] = None) -> str:
    return os.path.join(DATASETS_DIR, dataset, f"{model}.jsonl" if model else DATA_FILE)

def index_path(dataset: str, model: str) -> str:
    return os.path.join(DATASETS_DIR, dataset, f"{model}.faiss")
