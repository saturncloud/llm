import os
from typing import Optional

import click
from datasets import Dataset, load_dataset


def save_data(dataset: Dataset, path: str) -> bool:
    if os.path.exists(path):
        if not click.confirm(f"File path {path} already exists. Would you like to overwrite it?"):
            return False
    dataset.to_parquet(path)
    return True


def load_data(input_path: str, input_type: Optional[str] = None) -> Dataset:
    split = "test"
    if input_type is None:
        input_type = input_path.rsplit(".", 1)[-1]
    if input_type == "jsonl":
        input_type = "json"

    return load_dataset(
        input_type, data_files={split: input_path}, split=split
    )
