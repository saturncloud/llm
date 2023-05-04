import argparse
import asyncio
from typing import Optional
from urllib.parse import urlparse

from datasets import Dataset

from bert_qa.crawler import Crawler
from bert_qa.data import data_path
from bert_qa.retriever import Retriever


async def scrape_dataset(url: str, name: Optional[str] = None, **crawler_kwargs) -> Dataset:
    parsed = urlparse(url)
    if parsed.hostname is None:
        raise Exception(f"Invalid URL: {url}")

    if name is None:
        path = parsed.path or ""
        name = parsed.hostname + "-" + path.replace("/", "-").lstrip("-")
        name = name.rstrip("-")

    crawler = Crawler(url, **crawler_kwargs)
    retriever = Retriever(load_datasets=False)

    print("Crawling URLs")
    await crawler.run()
    raw_dataset = Dataset.from_list(crawler.dump_content())
    raw_dataset.to_json(data_path(name, raw=True))
    retriever.add_dataset(name, raw_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BERT QA Scrape")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("-n", "--name", help="Name of the resultant dataset (parsed from URL by default)")
    parser.add_argument("--max-depth", default=5, help="Maximum depth of URL links to follow")
    parser.add_argument("--include-below-root", action="store_true", help="Follow links that are below the given URL path")
    parser.add_argument("--include-external", action="store_true", help="Follow links to external sites")
    parser.add_argument("-e", "--exclude-regex", help="Exclude scraped URLs by regex pattern")
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(scrape_dataset(**vars(args)))
