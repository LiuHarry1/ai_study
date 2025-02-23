from whoosh.index import create_in
from whoosh.fields import Schema, ID, TEXT
import os

# Define key multi-word phrases
KEY_PHRASES = ["alert 38", "alert 4 issue"]

# Preprocess text to replace important phrases with underscore notation
def preprocess_text(text):
    for phrase in KEY_PHRASES:
        text = text.replace(phrase, phrase.replace(" ", "_"))  # e.g., "alert 38" → "alert_38"
    return text

# Define schema
schema = Schema(
    id=ID(stored=True, unique=True),
    original_content=TEXT(stored=True),
    processed_content=TEXT(stored=True)
)

# Create index
INDEX_DIR = "fuzzy_index"
if not os.path.exists(INDEX_DIR):
    os.mkdir(INDEX_DIR)
ix = create_in(INDEX_DIR, schema)

# Sample documents
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

print("Indexing completed with phrase replacements!")

from whoosh.index import open_dir
from whoosh.qparser import QueryParser, FuzzyTermPlugin

# Open index
ix = open_dir("fuzzy_index")


# Function to apply fuzzy search on each term
def search_fuzzy(query_text):
    processed_query = preprocess_text(query_text.lower())  # Normalize query

    with ix.searcher() as searcher:
        parser = QueryParser("processed_content", ix.schema)
        parser.add_plugin(FuzzyTermPlugin())  # Enable fuzzy search

        # Apply fuzziness to each word separately
        fuzzy_query = " ".join([word + "~2" if len(word) > 3 else word for word in processed_query.split()])

        query = parser.parse(fuzzy_query)  # Parse fuzzy query
        results = searcher.search(query, limit=10)

        output = []
        for r in results:
            output.append({
                "id": r["id"],
                "original_text": r["original_content"],
                "score": r.score
            })

        return output


# Test new fuzzy search
print("\nSearching for 'alert38':")
print(search_fuzzy("alert38"))

print("\nSearching for 'alert 4issu':")
print(search_fuzzy("alert 4issu"))
