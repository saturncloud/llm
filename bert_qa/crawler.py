from __future__ import annotations

import copy
import logging
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Union
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from datasets import Dataset
from scrapy import Spider, signals
from scrapy.http import Response, Request
from scrapy.crawler import CrawlerProcess


class DocSpider(Spider):
    name = "documentation"
    start_urls: List[str]

    custom_settings = {
        "LOG_LEVEL": "INFO",
        "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        start_urls: Optional[List[str]] = None,
        allowed_domains: Optional[List[str]] = None,
        link_regex: Optional[str] = None,
        link_css: Optional[str] = None,
        text_css: Optional[str] = None,
        dont_redirect: bool = False,
        **kwargs,
    ):
        self.start_urls = start_urls or []
        self.allowed_domains = allowed_domains
        self.link_regex = re.compile(link_regex) if link_regex else None
        self.link_css = link_css
        self.text_css = text_css
        self.dont_redirect = dont_redirect
        super().__init__(name, **kwargs)

    @classmethod
    def run(
        cls,
        start_urls: Union[str, List[str]],
        allowed_domains: Optional[List[str]] = None,
        link_regex: Optional[str] = None,
        link_css: Optional[str] = None,
        text_css: Optional[str] = None,
        dont_redirect: bool = False,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dataset:
        if isinstance(start_urls, str):
            start_urls = [start_urls]
        crawler = CrawlerProcess(settings=settings or {})
        crawler.crawl(
            cls,
            start_urls=start_urls,
            allowed_domains=allowed_domains,
            link_regex=link_regex,
            link_css=link_css,
            text_css=text_css,
            dont_redirect=dont_redirect,
        )
        dataset = Dataset.from_dict({"source": [], "text": [], "title": []})
        scrape_finished: bool = False
        def _append_content(item):
            nonlocal dataset
            # Not sure what scrapy is doing, but there are some unhashable parts of the item
            # that cause datasets issues. Copying the dict gets around it.
            dataset = dataset.add_item(copy.deepcopy(item))

        def _spider_closed(reason: str):
            nonlocal scrape_finished
            if reason == "finished":
                scrape_finished = True

        for c in crawler.crawlers:
            c.signals.connect(_append_content, signals.item_scraped)
            c.signals.connect(_spider_closed, signal=signals.spider_closed)

        crawler.start()
        if not scrape_finished:
            logging.error("Scrape failed")
            raise sys.exit(1)

        return dataset

    def parse(self, response: Response, **kwargs):
        content_type = response.headers["content-type"].decode().split(";")
        content_type = [ct.strip() for ct in content_type]
        if "text/html" not in content_type:
            return

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        title = soup.title.string if soup.title else None
        if self.text_css:
            text = ""
            for element in soup.select(self.text_css):
                element_text = element.get_text(separator=" ", strip=True)
                text += element_text + " "
            text = text.strip()
        else:
            text = soup.get_text(separator=" ", strip=True)

        yield {
            "source": response.url,
            "text": text,
            "title": title,
        }

        link_base = response.url
        if soup.head and soup.head.base and soup.head.base.get("href", None):
            link_base = urljoin(link_base, soup.head.base["href"])

        for url in self.extract_links(link_base, soup, self.link_css):
            yield Request(url, self.parse, meta={"dont_redirect": self.dont_redirect})

    def extract_links(self, base_url: str, soup: BeautifulSoup, css_selector: Optional[str] = None) -> Iterable[str]:
        if css_selector:
            for element in soup.select(css_selector):
                return self.extract_links(base_url, element)
        else:
            for link in soup.find_all("a", href=True):
                href = link["href"].split("#")[0]
                url = urljoin(base_url, href)
                if self.is_valid_url(url):
                    yield url

    def is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            return False
        if self.allowed_domains is not None:
            if parsed.netloc and parsed.netloc not in self.allowed_domains:
                return False
        if self.link_regex is not None:
            if not self.link_regex.search(url):
                return False
        return True
