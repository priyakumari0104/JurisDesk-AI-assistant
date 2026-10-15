from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-l6-v2")


def generate_embedding(text):
    embedding = model.encode(text)
    return embedding


if __name__ == "__main__":
    text1 = "The tenant must pay monthly rent ."
    text2 = "The renter is required to make a monthly"
    embedding1 = generate_embedding(text1)
    embedding2 = generate_embedding(text2)
    print("Embedding 1:")
    print(embedding1)

    print("\nEmbedding 1 dimensions:")
    print(len(embedding1))

    print("\nEmbedding 2 dimensions:")
    print(embedding2)

    print("\nEmbedding 2 dimensions:")
    print(len(embedding2))
