from services.vector_store import save_chunk
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
if __name__ == "__main__":
    text = "The tenant must pay monthly rent of 25000 rupees."
    embedding = model.encode(text)
    result = save_chunk(
        document_name="rental_agreement.pdf", chunk_id=0, text=text, embedding=embedding
    )
    print("Chunk saved successfully!")
    print("MongoDb:", result)
