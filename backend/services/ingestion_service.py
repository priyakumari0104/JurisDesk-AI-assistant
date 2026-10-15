from sentence_transformers import SentenceTransformer
from nlp.chunker import create_chunk
from services.vector_store import save_chunk, delete_document_chunks
from uuid import uuid4

# load the embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def ingest_document(document_name, pages, document_id=None):
    if document_id is None:
        document_id = str(uuid4())

    total_chunks = 0
    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"].strip()
        if not text:
            continue
        chunks = create_chunk(text, chunk_size=500, overlap=50)
        for chunk in chunks:
            embedding = model.encode(chunk)
            save_chunk(
                document_name=document_name,
                text=chunk,
                chunk_id=total_chunks,
                embedding=embedding,
                page=page_number,
                document_id=document_id,
            )
            total_chunks += 1
    return {"document_id": document_id, "total_chunks": total_chunks}
