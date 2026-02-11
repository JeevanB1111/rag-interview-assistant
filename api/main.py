import os
import sys
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from retrieval.retriever import Retriever
from llm.qa_chain import generate_answer

app = FastAPI()

templates = Jinja2Templates(directory="api/templates")

retriever = Retriever(BASE_DIR)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...)):
    retrieved_chunks = retriever.retrieve(question)
    answer = generate_answer(question, retrieved_chunks)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "question": question,
            "answer": answer,
        },
    )
