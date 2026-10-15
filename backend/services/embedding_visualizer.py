from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

from services.database import documents_collection


def get_embedding_graph():
    # 1. Fetch embeddings from MongoDB
    records = list(
        documents_collection.find(
            {"embedding": {"$exists": True}},
            {
                "_id": 0,
                "document_id": 1,
                "document": 1,
                "chunk_id": 1,
                "page": 1,
                "text": 1,
                "embedding": 1,
            },
        )
    )

    if not records:
        return {
            "points": [],
            "total_chunks": 0,
            "cluster_count": 0,
            "message": "No embeddings found. Upload a PDF first.",
        }

    # 2. Validate vectors
    valid_records = []

    for record in records:
        vector = record.get("embedding")

        if not isinstance(vector, list):
            continue
        if len(vector) != 384:
            continue
        if not all(
            isinstance(value, (int, float)) and np.isfinite(value) for value in vector
        ):
            continue
        valid_records.append(record)

    if not valid_records:
        return {
            "points": [],
            "total_chunks": 0,
            "cluster_count": 0,
            "message": "No valid 384-dimensional embeddings found.",
        }

        # 3. Convert embeddings into a NumPy matrix
    vectors = np.asarray(
        [record["embedding"] for record in valid_records],
        dtype=np.float32,
    )

    # 4. Reduce to 2D
    if len(vectors) == 1:
        coordinates = np.array([[0.0, 0.0]])
    else:
        coordinates = PCA(n_components=2).fit_transform(vectors)

        # 5. Cluster similar embeddings
    distinct_count = len(np.unique(vectors, axis=0))
    cluster_count = min(5, distinct_count, len(vectors))

    if cluster_count >= 2:
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10,
        ).fit_predict(vectors)
    else:
        labels = np.zeros(len(vectors), dtype=int)

        # 6. Build graph points
    points = []

    for record, coordinate, cluster in zip(valid_records, coordinates, labels):
        points.append(
            {
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
                "cluster": int(cluster),
                "document_id": record.get("document_id"),
                "document": record.get("document", "Unknown document"),
                "chunk_id": record.get("chunk_id"),
                "page": record.get("page"),
                "text": record.get("text", ""),
            }
        )

        # 7. Return AFTER processing every record
    return {
        "points": points,
        "total_chunks": len(points),
        "cluster_count": cluster_count,
    }
