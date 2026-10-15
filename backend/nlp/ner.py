import spacy

nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})
    return entities


if __name__ == "__main__":
    text = """
    This agreement is between priya kumar and abc technologies pvt ltd.
    the agreement wassigned on 15 january 2026 in new delhi.
    the monthly paymennt isrs 25,000.
    """
    entities = extract_entities(text)
    print("Named Entities :\n")
    for entity in entities:
        print(entity)
