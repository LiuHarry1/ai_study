from whoosh.index import create_in
from whoosh.fields import Schema, ID, TEXT
import os

# Define key phrases to be treated as units
KEY_PHRASES = ["alert 38", "alert 4 issue"]

# Function to preprocess text (replace key phrases with single tokens)
def preprocess_text(text):
    for phrase in KEY_PHRASES:
        text = text.replace(phrase, phrase.replace(" ", "_"))  # e.g., "alert 38" → "alert_38"
    return text

# Define Whoosh Schema
schema = Schema(
    id=ID(stored=True, unique=True),
    original_content=TEXT(stored=True),  # Store original text
    processed_content=TEXT(stored=True)  # Store modified text for searching
)

# Create or open index
INDEX_DIR = "custom_index"
if not os.path.exists(INDEX_DIR):
    os.mkdir(INDEX_DIR)
ix = create_in(INDEX_DIR, schema)

# Sample sentences with IDs
sentences = [
    (0, "Alert 38 creation happened today."),
    (1, "System detected alert 4 issue in the logs."),
    (2, "There was an alert 38 and alert 4 issue."),
    (3, "Alert 5 warning detected.")
]

# Index documents
writer = ix.writer()
for sentence_id, sentence in sentences:
    processed_sentence = preprocess_text(sentence.lower())  # Convert to lowercase
    writer.add_document(id=str(sentence_id), original_content=sentence, processed_content=processed_sentence)
writer.commit()

print("Indexing completed with stored IDs and original texts!")


from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F

# Open index
ix = open_dir(INDEX_DIR)

# Function to search and return ID + original text
def search(query_text):
    processed_query = preprocess_text(query_text.lower())  # Apply phrase replacement

    with ix.searcher(weighting=BM25F()) as searcher:
        parser = QueryParser("processed_content", ix.schema)  # Search in processed content
        query = parser.parse(processed_query)
        results = searcher.search(query, limit=10)

        output = []
        for r in results:
            output.append({"id": r["id"], "original_text": r["original_content"], "score": r.score})

        return output

# Example searches
print("\nSearching for 'alert 38':")
print(search("alert 38"))

print("\nSearching for 'alert 4 issue':")
print(search("alert 4 issue"))
