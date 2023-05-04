import os
from typing import List

from datasets import Dataset

DATASETS_DIR = os.getenv("DATASETS_DIR", "/data")
DATA_FILE = "data.jsonl"
RAW_DATA_FILE = "raw.jsonl"
INDEX_FILE = "index.faiss"
INDEX_NAME = "embedding"


def data_path(dataset_name: str, raw: bool = False) -> str:
    return os.path.join(DATASETS_DIR, dataset_name, RAW_DATA_FILE if raw else DATA_FILE)

def index_path(dataset_name: str) -> str:
    return os.path.join(DATASETS_DIR, dataset_name, INDEX_FILE)

def get_datasets() -> List[str]:
    return os.listdir(DATASETS_DIR)

def load_data(dataset_name: str, raw: bool = False, load_index: bool = True) -> Dataset:
    path = data_path(dataset_name, raw=raw)
    idx_path = index_path(dataset_name)
    dataset = Dataset.from_json(path)
    if load_index and not raw:
        if os.path.isfile(idx_path):
            dataset.load_faiss_index(INDEX_NAME, idx_path)
        else:
            dataset.add_faiss_index(INDEX_NAME)
            dataset.save_faiss_index(INDEX_NAME, idx_path)
    return dataset
