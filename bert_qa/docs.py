


from functools import lru_cache
import os
from typing import Dict, Iterable, List, Optional, Tuple

from bert_qa.scrape import GLOSSARY_DIR

DOCS_DIR = os.getenv("DOCS_DIR", "/data/docs")


def get_file_paths(skip: Optional[Iterable[str]] = None) -> Iterable[Tuple[str, str]]:
    for root, dirs, files in os.walk(DOCS_DIR):
        dirs[:] = [d for d in dirs if d not in skip]
        docs_root = root[len(DOCS_DIR):].lstrip("/")
        section = docs_root.split("/", 1)[0]
        for f in files:
            if section:
                file_section = section
            else:
                file_section = f.rsplit("/", 1)[-1].split(".", 1)[0]
            yield file_section, f"{docs_root}/{f}"


def load_docs(skip: Optional[Iterable[str]] = frozenset()) -> Dict[str, Dict[str, str]]:
    docs: Dict[str, Dict[str, str]] = {}
    for section, file in get_file_paths(skip=skip):
        docs.setdefault(section, {})
        with open(f"{DOCS_DIR}/{file}", "r") as f:
            docs[section][file] = f.read()
    return docs


def glossary_entries() -> List[str]:
    entries = os.listdir(GLOSSARY_DIR)
    return [e.rsplit(".", 1)[0] for e in entries]


@lru_cache()
def load_glossary_entry(name: str) -> str:
    if not name.endswith(".txt"):
        name += ".txt"
    with open(os.path.join(GLOSSARY_DIR, name)) as f:
        return f.read()