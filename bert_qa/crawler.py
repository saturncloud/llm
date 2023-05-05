from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import re
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

from aiohttp import ClientResponse, ClientSession, ClientTimeout, TCPConnector
from aiohttp.client_exceptions import ClientConnectionError, ClientConnectorError
from bs4 import BeautifulSoup
import requests

from bert_qa.retriever import Content


@asynccontextmanager
async def exp_backoff(session: ClientSession, url: str, retries: int = 5, retry_on_status: bool = True) -> AsyncGenerator[ClientResponse]:
    for i in range(1 + retries):
        try:
            async with session.get(url) as response:
                if not retry_on_status or response.ok:
                    yield response
                    return
                if i == retries:
                    raise Exception(f"Retry attempts failed for url {url}")
        except (ClientConnectionError, ClientConnectorError, asyncio.exceptions.TimeoutError) as e:
            if i == retries:
                raise e
        await asyncio.sleep(2 ** i)


class Crawler:
    def __init__(
        self,
        root_url: str,
        headers: Optional[Dict[str, str]] = None,
        max_depth: int = 5,
        include_root: bool = True,
        include_below_root: bool = False,
        include_external: bool = False,
        exclude_regex: Optional[str] = None,
        parse_robots_txt: bool = True,
    ):
        self.root_url = root_url
        self.base_url = urljoin(self.root_url, "/")
        self.headers = headers if headers else {}
        self.max_depth = max_depth
        self.include_root = include_root
        self.include_below_root = include_below_root
        self.include_external = include_external
        self.exclude_re: Optional[re.Pattern] = re.compile(exclude_regex) if exclude_regex else None
        if parse_robots_txt:
            self.robots_parser = RobotsParser()
        else:
            self.robots_parser = None

        self.pages: List[Content] = []
        self.urls: Set[str] = set()
        self.failed = False

    async def run(self, timeout: int = 60, max_concurrency: int = 20):
        client_timeout = ClientTimeout(timeout)
        connector = TCPConnector(limit=max_concurrency, limit_per_host=max_concurrency)
        async with ClientSession(
            headers=self.headers, timeout=client_timeout, connector=connector
        ) as session:
            await self.fetch_content(session, self.root_url)

    async def fetch_content(self, session: ClientSession, url: str, depth: int = 1, retries: int = 5):
        print("SCRAPE:", url)
        self.urls.add(url)

        # Get content from URL
        async with exp_backoff(session, url, retries=retries) as response:
            if response.content_type != "text/html":
                return
            raw_text = await response.text()

        extract_links = depth + 1 <= self.max_depth
        content, new_urls = self.parse_content(url, raw_text, extract_links=extract_links)
        if self.include_root or url != self.root_url:
            self.pages.append(content)

        new_scrapes = []
        for url in new_urls:
            if url not in self.urls and await self.is_valid_url(session, url):
                self.urls.add(url)
                new_scrapes.append(self.fetch_content(session, url, depth + 1))

        group = asyncio.gather(*new_scrapes)
        try:
            await group
        except Exception as e:
            # Cancel other coroutines when one fails
            group.cancel()
            raise e

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
            for link in soup.find_all("a", href=True):
                href = link["href"]
                parsed = urlparse(href)
                if parsed.scheme and parsed.scheme not in {"http", "https"}:
                    continue
                # Strip anchor tags
                href = href.split("#")[0]

                new_url = urljoin(link_base, href)
                new_urls.append(new_url)
        return content, new_urls

    async def is_valid_url(self, session: ClientSession, url: str) -> bool:
        is_external = not url.startswith(self.base_url)
        is_below_root = not is_external and not url.startswith(self.root_url)
        if not self.include_external and is_external:
            return False
        if not self.include_below_root and is_below_root:
            return False
        if self.exclude_re and self.exclude_re.match(url):
            return False
        if self.robots_parser:
            if not self.robots_parser.has_robots(url):
                # New base url, update robot parser with async client before validation
                base_url = urljoin(url, "/")
                async with exp_backoff(session, urljoin(base_url, "robots.txt"), retry_on_status=False) as response:
                    if response.ok:
                        robots_txt = await response.text()
                        self.robots_parser.update_robots(base_url, robots_txt=robots_txt)
            if not self.robots_parser.validate(url):
                return False
        return True

    def dump_content(self) -> List[Dict[str, Any]]:
        return [asdict(c) for c in self.pages]


@dataclass
class RobotsTxt:
    user_agents: Dict[str, AgentRules] = field(default_factory=dict)
    sitemaps: List[str] = field(default_factory=list)


@dataclass
class AgentRules:
    allow: List[str] = field(default_factory=list)
    disallow: List[str] = field(default_factory=list)


class RobotsParser:
    robot_rules: Dict[str, RobotsTxt]

    def __init__(self):
        self.robot_rules = {}

    def validate(self, url: str, user_agent: str = "*") -> bool:
        """
        Fetch the robots.txt file for the given URL if needed,
        then check if fetching the url is allowed for the given user_agent
        """
        base_url = urljoin(url, "/")
        if base_url not in self.robot_rules:
            self.update_robots(base_url)

        rules = self.get_rules(base_url, user_agent=user_agent)
        allow_pattern = ""
        disallow_pattern = ""
        for pattern in rules.allow:
            if self.match_url(url, pattern):
                if len(pattern) > len(allow_pattern):
                    allow_pattern = pattern
        for pattern in rules.disallow:
            if self.match_url(url, pattern):
                if len(pattern) > len(disallow_pattern):
                    disallow_pattern = pattern

        is_valid = False
        if not allow_pattern and not disallow_pattern:
            is_valid = True
        else:
            is_valid = len(allow_pattern) > len(disallow_pattern)

        if user_agent != "*":
            is_valid = is_valid and self.validate(url)
        return is_valid

    def has_robots(self, base_url: str) -> bool:
        return urljoin(base_url, "/") in self.robot_rules

    def update_robots(self, base_url: str, robots_txt: Optional[str] = None):
        if not robots_txt:
            response = requests.get(urljoin(base_url, "robots.txt"))
            if not response.ok:
                # Ignore missing or failed robots.txt ¯\_(ツ)_/¯
                return
            robots_txt = response.text

        lines = robots_txt.split("\n")
        user_agent = "*"
        self.rules = {}

        for line in lines:
            line = line.split("#", 1)[0]
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key, val = key.strip().lower(), val.strip().lower()

            if key == "user-agent":
                user_agent = val
            elif key in {"allow", "disallow"}:
                self.add_rule(base_url, val, key == "allow", user_agent=user_agent)
            elif key == "sitemap":
                self.add_sitemap(base_url, val)

    def match_url(self, url: str, pattern: str) -> bool:
        parsed = urlparse(url)
        path = urlunparse(('','',parsed.path, parsed.params,parsed.query, parsed.fragment))
        for part in pattern.split("*"):
            if "$" in part:
                part = part.split("$")[0]
                return path == part
            if not path.startswith(part):
                return False
            path = path[len(part):]
        return True

    def get_robot(self, base_url: str) -> RobotsTxt:
        if base_url not in self.robot_rules:
            self.robot_rules[base_url] = RobotsTxt()
        return self.robot_rules[base_url]

    def get_rules(self, base_url: str, user_agent: str = "*") -> AgentRules:
        robot_txt = self.get_robot(base_url)
        if not user_agent in robot_txt.user_agents:
            robot_txt.user_agents.setdefault(user_agent, AgentRules())
        return robot_txt.user_agents[user_agent]

    def add_rule(self, base_url: str, pattern: str, allowed: bool, user_agent: str = "*"):
        if pattern.startswith("/"):
            rules = self.get_rules(base_url, user_agent=user_agent)
            if allowed:
                rules.allow.append(pattern)
            else:
                rules.disallow.append(pattern)

    def add_sitemap(self, base_url: str, sitemap: str):
        if self.validate_url(sitemap):
            robot_txt = self.get_robot(base_url)
            robot_txt.sitemaps.append(sitemap)

    def validate_url(self, url: str):
        try:
            parsed = urlparse(url)
            return all([parsed.scheme, parsed.netloc])
        except Exception:
            return False
