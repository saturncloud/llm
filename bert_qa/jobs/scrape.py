import argparse
import asyncio
from typing import Optional
from urllib.parse import urlparse

from datasets import Dataset

from bert_qa.crawler import DocSpider
from bert_qa.retriever import Retriever


async def scrape_dataset(url: str, name: Optional[str] = None, max_depth: int = 10, **spider_kwargs) -> Dataset:
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

    retriever = Retriever(load_datasets=False)
    dataset = DocSpider.run(
        url,
        **spider_kwargs,
        settings={"DEPTH_LIMIT": max_depth, "CONCURRENT_REQUESTS": 20},
    )
    retriever.add_dataset(name, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("docBERT HTML scrape")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("-n", "--name", help="Name of the resultant dataset (parsed from URL by default)")
    parser.add_argument("--allowed-domains", action="append", help="One or more comma separated domains that are allowed to be scraped")
    parser.add_argument("--link-regex", help="Match URLs against regex before adding to scrape pipeline")
    parser.add_argument("--link-css", help="Only extract links from elements matching the given CSS selector")
    parser.add_argument("--text-css", help="Only extract text from elements matching the given CSS selector")
    parser.add_argument("--max-depth", default=10, type=int, help="Maximum depth of URL links to follow")
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(scrape_dataset(**vars(args)))
