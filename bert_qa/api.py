import pathlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from bert_qa.model import BertQA, Question

app = FastAPI()
bert_qa = BertQA()


@app.get("/")
def index():
    return FileResponse(str(pathlib.Path(__file__).parent) + "/index.html", media_type="text/html")


@app.post("/api/question")
def post_question(question: Question):
    answer = bert_qa.search_docs(question, span_length=384, span_overlap=64)
    return JSONResponse(answer.to_dict())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
