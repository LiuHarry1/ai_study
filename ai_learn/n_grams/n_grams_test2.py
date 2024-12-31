from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


class NgramSuggester:
    def __init__(self, texts, n=3):
        self.n = n
        self.ngram_map = defaultdict(lambda: defaultdict(int))
        self._build_ngram_map(texts)

    def _build_ngram_map(self, texts):
        # Generate n-grams using CountVectorizer
        vectorizer = CountVectorizer(ngram_range=(2, self.n), token_pattern=r'\b[a-zA-Z0-9\-]+\b', lowercase=True)
        X = vectorizer.fit_transform(texts)
        ngram_features = vectorizer.get_feature_names_out()

        for ngram in ngram_features:
            words = ngram.split()
            prefix, next_word = " ".join(words[:-1]), words[-1]
            self.ngram_map[prefix][next_word] += 1

    def suggest(self, user_input, top_k=5):
        # Extract the last n-1 words as prefix
        words = user_input.lower().split()
        for i in range(self.n - 1, 0, -1):  # Try n-1 to 1 words as prefix
            prefix = " ".join(words[-i:]) if len(words) >= i else None
            if prefix and prefix in self.ngram_map:
                candidates = self.ngram_map[prefix]
                sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])  # Sort by frequency
                return [word for word, _ in sorted_candidates[:top_k]]
        return []  # No suggestions found


# Example usage
if __name__ == "__main__":
    # Example Jira subjects
    texts = [
        "aspen-position-web fails to load",
        "aspen-position-web user authentication error",
        "aspen-position-web shows 404 error",
        "aspen-position-web deployment issue detected",
        "authentication error in aspen-position-web",
        "I want to build a better aspen-position-web system"
    ]

    suggester = NgramSuggester(texts, n=3)

    # Suggest words based on user input
    user_input = "I want to build aspen-position-web"
    suggestions = suggester.suggest(user_input)
    print(f"Suggestions for '{user_input}': {suggestions}")
