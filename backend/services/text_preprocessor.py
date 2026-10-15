import re
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))


def clean_text(text):
    # convert to lowercase
    text = text.lower()
    # Remove extra spaces and new lines
    text = re.sub(r"\s+", " ", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s.,%-]", "", text)
    text = text.strip()
    return text


def tokenize_text(text):
    return nltk.word_tokenize(text)


def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]


if __name__ == "__main__":
    text = """
    the tenant shall pay a monthly rent of rs 25,000.
    the agreement may be terminated with 30das notice.
    """
    cleaned = clean_text(text)
    print("orignal:")
    print(text)
    print("\nCleaned")
    print(cleaned)
    tokens = tokenize_text(cleaned)
    print("\nTokens:")
    print(tokens)
    filtered = remove_stopwords(tokens)
    print("\nWithout stopwards:")
    print(filtered)
