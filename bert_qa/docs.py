


import os
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch

DOCS_DIR = os.getenv("DOCS_DIR", "/data/docs")


def get_file_paths(skip: Optional[Iterable[str]] = None) -> Iterable[Tuple[str, str]]:
    for root, dirs, files in os.walk(DOCS_DIR):
        dirs[:] = [d for d in dirs if d not in skip]
        section = root.split("/")[-1]
        if not section:
            section = "docs"
        yield section, [f"{root}/{f}" for f in files]


def load_docs(skip: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, str]]:
    docs: Dict[str, Dict[str, str]] = {}
    for section, file_paths in get_file_paths(skip=skip):
        docs[section] = {}
        for file in file_paths:
            with open(file, "r") as f:
                docs[section][file.split("/")[-1]] = f.read()
    return docs
