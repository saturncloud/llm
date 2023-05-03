import asyncio
from dataclasses import asdict, dataclass
import os
from typing import Any, Dict, List, Optional, Set, Tuple
from requests import Session
from urllib.parse import urljoin, urlparse

from aiohttp import ClientSession
from bs4 import BeautifulSoup

from bert_qa.retriever import Content


class Crawler:
    def __init__(
        self,
        root_url: str,
        max_depth: int = 5,
        include_below_root: bool = False,
        include_external: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.root_url = root_url
        self.max_depth = max_depth
        self.include_below_root = include_below_root
        self.include_external = include_external
        self.headers = headers if headers else {}

        self.pages: List[Content] = []
        self.urls: Set[str] = set()

    async def crawl(self):
        async with ClientSession(headers=self.headers) as s:
            await self.fetch_docs(s, self.root_url)

    def sync_crawl(self):
        with Session() as s:
            s.headers.update(self.headers)
            self.sync_fetch_docs(s, self.root_url)

    async def fetch_docs(self, s: ClientSession, url: str, depth: int = 1):
        if url in self.urls:
            return
        self.urls.add(url)

        # Get content from URL
        async with s.get(url) as response:
            if not response.ok:
                return
            raw_text = await response.text()

        extract_links = depth + 1 <= self.max_depth
        content, new_urls = self.parse_content(url, raw_text, extract_links=extract_links)
        self.pages.append(content)

        new_scrapes = []
        for url in new_urls:
            new_scrapes.append(self.fetch_docs(s, url, depth + 1))
        await asyncio.gather(*new_scrapes)

    def sync_fetch_docs(self, s: Session, url: str, depth: int = 1):
        if url in self.urls:
            return
        self.urls.add(url)

        response = s.get(url)
        if not response.ok:
            return

        extract_links = depth + 1 <= self.max_depth
        content, new_urls = self.parse_content(url, response.text, extract_links)
        self.pages.append(content)

        for url in new_urls:
            self.sync_fetch_docs(s, url, depth + 1)

    def parse_content(self, source: str, raw_text: str, extract_links: bool = True) -> Tuple[Content, List[str]]:
        # Parse HTML to text
        soup = BeautifulSoup(raw_text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        title = soup.title.string if soup.title else None
        content = Content(source, text, title)

        link_base = source
        if soup.head and soup.head.base and soup.head.base.get("href", None):
            link_base = urljoin(link_base, soup.head.base["href"])

        new_urls = []
        if extract_links:
            base_url = urljoin(self.root_url, "/")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                parsed = urlparse(href)
                if parsed.scheme and parsed.scheme not in {"http", "https"}:
                    continue

                new_url = urljoin(link_base, link["href"])
                is_external = not new_url.startswith(base_url)
                is_below_root = not is_external and not new_url.startswith(self.root_url)
                if not self.include_external and is_external:
                    continue
                elif not self.include_below_root and is_below_root:
                    continue

                new_urls.append(new_url)
        return content, new_urls
