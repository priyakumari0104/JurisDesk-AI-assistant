from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([text])
    scores = matrix.toarray()[0]
    words = vectorizer.get_feature_names_out()
    keyword_scores = list(zip(words, scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    return keyword_scores[:top_n]


if __name__ == "__main__":
    text = """
    This rental agreement is between the lanloard and tenat.
    The tenant agrees to pay monthly rent of 25000 rupees.
    The rental agrrement can be terminated with 30 days notice.
    The tenanat must maintain the property and make all payment on time.
    """
    keywords = extract_keywords(text)
    print("Important keywords:\n")
    for word, score in keywords:
        print(f"{word}:{score:.4f}")
