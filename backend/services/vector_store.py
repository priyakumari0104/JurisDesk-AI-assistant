from services.database import documents_collection


def delete_document_chunks(document_id):
    """Delete existing chunks for a documents before reprocessing it."""
    result = documents_collection.delete_many({"document_id": document_id})
    return result.deleted_count


def save_chunk(document_name, chunk_id, text, embedding, page=None, document_id=None):
    document = {
        "document_id": document_id,
        "document": document_name,
        "chunk_id": chunk_id,
        "page": page,
        "text": text,
        "embedding": embedding.tolist(),
    }
    result = documents_collection.insert_one(document)
    return result.inserted_id


def list_documents():
    pipeline = [
        {
            "$group": {
                "_id": "$document_id",
                "filename": {"$first": "$document"},
                "total_chunks": {"$sum": 1},
                "pages": {"$addToSet": "$page"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "document_id": "$_id",
                "filename": 1,
                "total_chunks": 1,
                "total_pages": {"$size": "$pages"},
            }
        },
        {"$sort": {"filename": 1}},
    ]
    return list(documents_collection.aggregate(pipeline))


def document_exists(document_id):
    return documents_collection.count_documents({"document_id": document_id}) > 0
