from services.database import documents_collection
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def search_similar_chunks(query, limit=3):
    # convert query int embeddngs
    query_embedding = model.encode(query).tolist()
    # MongoDb vectorsearch pipline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 10,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "document": 1,
                "chunk_id": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    results = documents_collection.aggregate(pipeline)
    return list(results)


if __name__ == "__main__":

    print("Starting semantic search...")

    query = "What is the monthly rent?"

    results = search_similar_chunks(query)

    print("\nSearch Results:\n")

    if not results:
        print("No matching chunks found.")
    else:
        for result in results:
            print("Document:", result["document"])
            print("Chunk ID:", result["chunk_id"])
            print("Score:", result["score"])
            print("Text:", result["text"])
            print("-" * 50)
