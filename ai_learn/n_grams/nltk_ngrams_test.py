from nltk import word_tokenize
from nltk.util import ngrams
from collections import defaultdict, Counter


class NgramModel:
    def __init__(self, texts, n=3):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self._build_model(texts)

    def _build_model(self, texts):
        for text in texts:
            tokens = ["<s>"] + word_tokenize(text.lower()) + ["</s>"]
            for ngram in ngrams(tokens, self.n):
                self.ngram_counts[ngram[:-1]][ngram[-1]] += 1

    def suggest(self, context, top_k=5):
        tokens = word_tokenize(context.lower())
        context = tuple(tokens[-(self.n - 1):])  # Use the last n-1 tokens
        candidates = self.ngram_counts.get(context, {})
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])  # Sort by count
        return [word for word, _ in sorted_candidates[:top_k]]


# Example usage
texts = [
    "aspen-position-web fails to load",
    "aspen-position-web user authentication error",
    "aspen-position-web shows 404 error"
]
model = NgramModel(texts, n=3)
user_input = "I want to build aspen-position-web"
suggestions = model.suggest(user_input)
print(f"Suggestions for '{user_input}': {suggestions}")
