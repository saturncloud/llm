import os
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from scrapy import Spider
from scrapy.crawler import CrawlerProcess
from scrapy.http import Request, Response

GLOSSARY_URL = os.getenv("GLOSSARY_URL", "https://saturncloud.io/glossary")
GLOSSARY_DIR = os.getenv("GLOSSARY_DIR", "/data/glossary")


class GlossarySpider(Spider):
    name = "docs"
    start_urls = [GLOSSARY_URL]

    @classmethod
    def run(cls):
        os.makedirs(GLOSSARY_DIR, exist_ok=True)
        crawler = CrawlerProcess()
        crawler.crawl(cls)
        crawler.start()

    def parse(self, response: Response):
        sections = response.css(".glossary-side-list > ul > li > a[href^='/glossary/']")
        hrefs = sections.css("::attr(href)").getall()
        names = sections.css("::text").getall()
        for name, href in zip(names, hrefs):
            yield Request(urljoin(GLOSSARY_URL, href), callback=self.parse_docs_page, cb_kwargs={"name": name})

    def parse_docs_page(self, response: Response, name: str):
        html_content = response.css(".glossary-single-content").extract()[0]
        soup = BeautifulSoup(html_content, "html.parser")
        for link in soup.find_all("a", href=True):
            link_text = soup.new_tag("p")
            link_text.string = f"{link.string} ({link['href']})"
            link.replace_with(link_text)
        content = soup.get_text(separator=" ")
        filename = name.lower().replace(" ", "-") + ".txt"

        with open(os.path.join(GLOSSARY_DIR, filename), "w") as f:
            f.write(content)
