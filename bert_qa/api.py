from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from bert_qa.model import BertQA

app = FastAPI()
bert_qa = BertQA()


class QARequest(BaseModel):
    question: str = ""


@app.get("/")
def index():
    return HTMLResponse(
        """
        <html>
            <head>
                <title>Saturn Docs BERT-QA</title>
            </head>
            <body>
                <h1>Saturn Docs BERT-QA</h1>

                <form action="" id="qa">
                    <label for="question">Question:</label>
                    <input type="text" id="question" name="question"><br><br>
                    <input type="submit" id="submit_button" value="Submit">
                </form>
                <h2>History:</h2>
                <table id="answers">
                    <tr>
                        <th>Question</th>
                        <th>Answer</th>
                        <th>Source</th>
                        <th>Score</th>
                    </tr>
                </table>
            </body>
            <script>
                const questionForm = document.querySelector("#qa");
                const submitButton = document.getElementById("submit_button");
                const answersTable = document.getElementById("answers");

                if (questionForm) {
                    questionForm.addEventListener("submit", async function(e) {
                        e.preventDefault();
                        submitButton.disabled = true;

                        const data = {};
                        for (const pair of new FormData(questionForm)) {
                            data[pair[0]] = pair[1];
                        }
                        questionForm.reset();

                        const row = answers.insertRow(1);
                        const questionCell = row.insertCell(0);
                        const answerCell = row.insertCell(1);
                        const sourceCell = row.insertCell(2);
                        const scoreCell = row.insertCell(3);
                        questionCell.innerText = data["question"];
                        answerCell.innerText = "pending..."

                        const response = await fetch("/qa", {
                            method: "POST",
                            headers: {"Content-Type": "application/json"},
                            body: JSON.stringify(data)
                        });
                        if (response.ok) {
                            const content = await response.json();
                            answerCell.innerText = content["answer"];
                            sourceCell.innerText = content["source"];
                            scoreCell.innerText = content["score"];
                        } else {
                            answerCell.innerText = "ERROR";
                        }
                        submitButton.disabled = false;
                    })
                }
            </script>
            <style>
                table {
                    min-width: 200px;
                }

                table, th, td {
                    border: 1px solid black;
                    border-collapse: collapse;
                }

                th, td {
                    padding: 0 1em 0 1em;
                }
            </style>
        </html>
        """
    )


@app.post('/qa')
def search_docs(body: QARequest):
    answer = bert_qa.search_docs(body.question, span_length=384, span_overlap=64)
    return JSONResponse(answer.to_dict())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
