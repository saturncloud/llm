from dataclasses import asdict, dataclass
from hashlib import md5
import json
import os
import pathlib
from typing import Optional

from datasets import Dataset
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from bert_qa.retriever import CONTEXTS_DIR, ContextChunk, Retriever
from bert_qa.scrape import Crawler
from bert_qa.model import BertQA


app = FastAPI()
bert_qa = BertQA()
retriever = Retriever()


@dataclass
class QuestionBody:
    question: str
    dataset_name: Optional[str] = None
    top_k: int = 10


@dataclass
class Answer:
    text: str
    question: str
    source: str
    score: float


@app.get("/")
def index():
    return FileResponse(str(pathlib.Path(__file__).parent) + "/index.html", media_type="text/html")


@app.post("/api/question")
def post_query(body: QuestionBody):
    results = retriever.search(**asdict(body))
    answer_text, score, i = bert_qa.best_answer(body.question, [r.text for r in results])
    answer = Answer(answer_text, body.question, score, results[i].source)
    return JSONResponse(asdict(answer))


@dataclass
class ScrapeBody:
    name: str
    url: str
    max_depth: int = 5
    include_below_root: bool = False
    include_external: bool = False


@app.post("/api/scrape")
async def post_scrape_url(body: ScrapeBody):
    crawler = Crawler(
        body.url, body.max_depth, body.include_below_root, body.include_external
    )
    await crawler.crawl()
    retriever.add_dataset(body.name, crawler.pages)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
