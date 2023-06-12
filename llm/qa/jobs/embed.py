
import argparse
import logging

from llm.qa.data import load_data, save_data
from llm.qa.document_store import DocStore
from llm.qa.embedding import QAEmbeddings, RecursiveCharacterTextSplitter

logging.root.setLevel(logging.NOTSET)


def embed_dataset(name: str, chunk_size: int = 256, chunk_overlap: int = 32):
    dataset = load_data(name, "data")
    if not dataset:
        raise FileNotFoundError(f"Dataset \"{name}\" not found")

    embedding = QAEmbeddings()
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        embedding.context_tokenizer,
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docstore = DocStore(embedding, index_name=name)

    dataset = docstore.parse_dataset(dataset)
    dataset = docstore.split_dataset(dataset, splitter)
    dataset = docstore.embed_dataset(dataset, devices="auto")
    save_data(dataset, name, embedding.context_model.name_or_path)

    docstore.add_dataset(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("docBERT HTML scrape")
    parser.add_argument("name", help="Name of the dataset to embed with the current retriever model")
    parser.add_argument("--chunk-size", help="Max number of tokens per context chunk", default=256)
    parser.add_argument("--chunk-overlap", help="Number of tokens shared between adjacent chunks", default=32)
    args = parser.parse_args()

    embed_dataset(**vars(args))
