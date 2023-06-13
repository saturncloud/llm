from dataclasses import asdict, dataclass, field
import pathlib
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from llm.qa.document_store import DocStore
from llm.qa.embedding import QAEmbeddings
from llm.qa.model import BertQA
from llm.qa.jobs.scrape import scrape_dataset

app = FastAPI()
bert_qa = BertQA()
docstore = DocStore(QAEmbeddings())


@dataclass
class QuestionBody:
    question: str
    dataset_name: Optional[str] = None
    top_k: int = 3


@dataclass
class Answer:
    text: str
    source: str
    score: float


@dataclass
class TopAnswers:
    question: str
    answers: List[Answer] = field(default_factory=list)


@app.get("/")
def index():
    return FileResponse(str(pathlib.Path(__file__).parent) + "/frontend/index.html", media_type="text/html")


@app.post("/api/question")
def post_query(body: QuestionBody):
    results = docstore.search(body.question, body.top_k * 2, include_fields=["source"])
    contexts = [r["text"] for r in results]

    topk_answers = bert_qa.topk_answers(body.question, contexts, body.top_k)
    answers = [
        Answer(text, results[i].get("source", "") if i >=0 else "", score)
        for text, score, i in topk_answers
    ]

    return JSONResponse(asdict(TopAnswers(body.question, answers)))


@dataclass
class ScrapeBody:
    name: str
    url: str
    max_depth: int = 5
    include_below_root: bool = False
    include_external: bool = False


@app.post("/api/scrape")
async def post_scrape_url(body: ScrapeBody):
    scrape_dataset(**asdict(body))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
