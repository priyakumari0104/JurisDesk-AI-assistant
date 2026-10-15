from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from services.rag_pipeline import answer_question
from services.document_processor import extract_text_from_pdf
from services.ingestion_service import ingest_document
from services.vector_store import (
    list_documents,
    delete_document_chunks,
    document_exists,
)
from services.embedding_visualizer import get_embedding_graph
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI  Legal Document Assistant",
    description="NLP-Based document ananlysis and question anwsering system",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# home


@app.get("/")
def home():
    return {"message": "AI Legal Document Assistant API is running"}


# get documents


@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}


# delete documents


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if not document_exists(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    deleted_count = delete_document_chunks(document_id)
    return {
        "message": "Document deleted successully",
        "document_id": document_id,
        "deleted_chunks": deleted_count,
    }


# upload api


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only Pdf files are supported")
    os.makedirs("documents", exist_ok=True)
    filename = os.path.basename(file.filename)
    file_path = os.path.join("documents", filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    pages = extract_text_from_pdf(file_path)
    result = ingest_document(filename, pages)
    return {
        "message": "Document uploaded and processed successfully",
        "document_id": result["document_id"],
        "filename": "filename",
        "total_pages": "len(pages)",
        "total_chunks": result["total_chunks"],
    }


class QuestionRequest(BaseModel):
    question: str


# ask question


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        return {"error": "Question cannot be empty"}
    result = answer_question(question)
    return result


#  api data for graph
@app.get("/embedding-graph")
def embedding_graph():
    try:
        result = get_embedding_graph()
        print("Embedding graph result:", type(result), result)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate embedding graph:{str(e)}",
        )
