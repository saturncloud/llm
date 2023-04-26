from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from bert_qa.model import BertQA

app = FastAPI()
bert_qa = BertQA()


class QARequest(BaseModel):
    question: str = ""


@app.get("/")
def index():
    return {"message": "Welcome to the app"}


@app.post('/qa')
def search_docs(body: QARequest):
    answer = bert_qa.search_docs(body.question, span_length=384, span_overlap=64)
    return JSONResponse(answer.to_dict())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
