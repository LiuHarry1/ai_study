import math

def compute_bm25_threshold(num_query_terms, k1=1.5, b=0.75, avgDL=100, doc_length=100, df=1, total_docs=10000):
    """
    Compute the theoretical BM25 score threshold for a given number of query terms.
    """
    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)  # Maximum IDF assumption
    term_score = idf * ((k1 + 1) / (k1 * ((1 - b) + b * (doc_length / avgDL)) + 1))
    return num_query_terms * term_score  # Total threshold

def filter_bm25_results(jira_results, query_terms):
    """
    Filters Jira results based on BM25 scores.

    Args:
        jira_results (list of dict): Each dict contains {"jira_id": str, "bm25_score": float}.
        query_terms (list of str): List of query terms.

    Returns:
        list of dict: Filtered results above the threshold.
    """
    threshold = compute_bm25_threshold(len(query_terms))
    print("threshold:" , threshold)
    return [jira for jira in jira_results if jira["bm25_score"] >= threshold]

# Example Jira BM25 Results
jira_results = [
    {"jira_id": "JIRA-1", "bm25_score": 6.5},
    {"jira_id": "JIRA-2", "bm25_score": 15.2},
    {"jira_id": "JIRA-3", "bm25_score": 25.7},
    {"jira_id": "JIRA-4", "bm25_score": 3.8},
]

query_terms = ["bug"]  # Single-word query
filtered_results = filter_bm25_results(jira_results, query_terms)

print("Filtered Jira Results:", filtered_results)
