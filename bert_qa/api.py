from dataclasses import dataclass
import pathlib
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from bert_qa.model import BertQA

app = FastAPI()
bert_qa = BertQA()


@dataclass
class QuestionBody:
    question: str
    section: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(str(pathlib.Path(__file__).parent) + "/index.html", media_type="text/html")


@app.post("/api/question")
def post_question(body: QuestionBody):
    answer = bert_qa.search_docs(body.question, body.section, span_length=384, span_overlap=64)
    return JSONResponse(answer.to_dict())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
