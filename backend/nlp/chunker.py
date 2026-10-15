def create_chunk(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    text = """
      This rental agreement is between the landloard and rental.
      The tenant agrees to pay monthly rent of 25000 rupees.
      The rental agreement can be terminated with 30 days notice.
      The tenant must  maintain the property and make all payments on time.
      The landlord may inspect the property with prior notice.
      """
    chunks = create_chunk(text, chunk_size=20, overlap=5)
    print("Total chunks:", len(chunks))
    for i, chunk in enumerate(chunks):
        print(f"\n Chunk {i+1}:")
        print(chunk)
