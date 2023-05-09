
import argparse
import logging
import sys

from bert_qa.data import load_data
from bert_qa.retriever import Retriever


def embed_dataset(name: str):
    dataset = load_data(name, "data")
    if not dataset:
        logging.error("Raw dataset not found")
        sys.exit(1)
    retriever = Retriever(load_datasets=False)
    retriever.add_dataset(name, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("docBERT HTML scrape")
    parser.add_argument("name", help="Name of the dataset to embed with the current retriever model")
    args = parser.parse_args()

    embed_dataset(**vars(args))
