import asyncio
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

from aiohttp import ClientSession, ClientTimeout
from aiohttp.client_exceptions import ClientConnectionError
from bs4 import BeautifulSoup

from bert_qa.retriever import Content


class Crawler:
    def __init__(
        self,
        root_url: str,
        max_depth: int = 5,
        include_root: bool = True,
        include_below_root: bool = False,
        include_external: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.root_url = root_url
        self.max_depth = max_depth
        self.include_root = include_root
        self.include_below_root = include_below_root
        self.include_external = include_external
        self.headers = headers if headers else {}

        self.pages: List[Content] = []
        self.urls: Set[str] = set()

    async def run(self):
        async with ClientSession(headers=self.headers, timeout=ClientTimeout(30)) as s:
            await self.fetch_content(s, self.root_url)

    async def fetch_content(self, s: ClientSession, url: str, depth: int = 1, retries: int = 3):
        self.urls.add(url)

        # Get content from URL
        for i in range(retries):
            try:
                async with s.get(url) as response:
                    if not response.ok:
                        return
                    # TODO: More content types?
                    if response.content_type != "text/html":
                        return
                    raw_text = await response.text()
                    break
            except ClientConnectionError as e:
                if i+1 >= retries:
                    raise e
                # Simple backoff
                await asyncio.sleep(i)
            except asyncio.exceptions.TimeoutError as e:
                print("Timeout on", url)
                if i+1 >= retries:
                    raise e
                await asyncio.sleep(i)

        extract_links = depth + 1 <= self.max_depth
        content, new_urls = self.parse_content(url, raw_text, extract_links=extract_links)
        if self.include_root or url != self.root_url:
            self.pages.append(content)

        new_scrapes = []
        for url in new_urls:
            if url not in self.urls:
                new_scrapes.append(self.fetch_content(s, url, depth + 1))
        await asyncio.gather(*new_scrapes)

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
                # Strip anchor tags
                href = href.split("#")[0]

                new_url = urljoin(link_base, href)
                is_external = not new_url.startswith(base_url)
                is_below_root = not is_external and not new_url.startswith(self.root_url)
                if not self.include_external and is_external:
                    continue
                elif not self.include_below_root and is_below_root:
                    continue

                new_urls.append(new_url)
        return content, new_urls

    def dump_content(self) -> List[Dict[str, Any]]:
        return [asdict(c) for c in self.pages]
