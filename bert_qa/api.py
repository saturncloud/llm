from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from bert_qa.model import BertQA

app = FastAPI()
bert_qa = BertQA()


@app.get("/")
def index():
    return {"message": "Welcome to the app"}


@app.post('/qa')
def generate_image(question: str):
    if not question:
        return {"error": "Please provide a question"}
    answer = bert_qa.search_docs(question, span_length=384, span_overlap=64)
    return JSONResponse(answer.to_dict())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
