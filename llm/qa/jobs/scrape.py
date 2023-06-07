import argparse
import asyncio
from typing import Optional
from urllib.parse import urlparse

from datasets import Dataset

from llm.qa.crawler import DocSpider
from llm.qa.data import save_data
from llm.qa.document_store import DocStore
from llm.qa.embedding import QAEmbeddings, RecursiveCharacterTextSplitter
from llm.qa.retriever import Retriever


async def scrape_dataset(
    url: str,
    name: Optional[str] = None,
    max_depth: int = 10,
    log_level: str = "INFO",
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    **spider_kwargs,
) -> Dataset:
    parsed = urlparse(url)
    if parsed.hostname is None:
        raise Exception(f"Invalid URL: {url}")

    if name is None:
        path = parsed.path or ""
        name = parsed.hostname + "-" + path.replace("/", "-").lstrip("-")
        name = name.rstrip("-")

    allowed_domains = spider_kwargs.get("allowed_domains")
    if allowed_domains:
        spider_kwargs["allowed_domains"] = []
        for domains in allowed_domains:
            spider_kwargs["allowed_domains"].extend(domains.split(","))

    # retriever = Retriever(load_datasets=False)
    dataset = DocSpider.run(
        url,
        **spider_kwargs,
        settings={"DEPTH_LIMIT": max_depth, "CONCURRENT_REQUESTS": 20, "LOG_LEVEL": log_level},
    )
    save_data(dataset, name)

    embedding = QAEmbeddings()
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        embedding.context_tokenizer,
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docstore = DocStore.from_weaviate(embedding, index_name=name)
    docstore.add_dataset(dataset, splitter=splitter)
    # retriever.add_dataset(name, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("docBERT HTML scrape")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("-n", "--name", help="Name of the resultant dataset (parsed from URL by default)")
    parser.add_argument("--allowed-domains", action="append", help="One or more comma separated domains that are allowed to be scraped")
    parser.add_argument("--link-regex", help="Match URLs against regex before adding to scrape pipeline")
    parser.add_argument("--link-css", help="Only extract links from elements matching the given CSS selector")
    parser.add_argument("--text-css", help="Only extract text from elements matching the given CSS selector")
    parser.add_argument("--max-depth", default=10, type=int, help="Maximum depth of URL links to follow")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--chunk-size", help="Max number of tokens per context chunk", default=256)
    parser.add_argument("--chunk-overlap", help="Number of tokens shared between adjacent chunks", default=32)
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(scrape_dataset(**vars(args)))
