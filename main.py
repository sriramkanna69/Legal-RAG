from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_functions import ingest_docs, ask_question

app = FastAPI(title="Legal RAG App")

class IngestRequest(BaseModel):
    id: str
    text: str

class QueryRequest(BaseModel):
    question: str

@app.post("/ingest")
def ingest(request: IngestRequest):
    try:
        message = ingest_docs(request.id, request.text)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(request: QueryRequest):
    try:
        answer = ask_question(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
