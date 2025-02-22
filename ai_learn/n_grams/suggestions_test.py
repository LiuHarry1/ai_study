from thefuzz import process
from difflib import SequenceMatcher

class JiraSearchEngine:
    def __init__(self, jira_summaries):
        """
        Initialize the search engine with a list of Jira summaries.
        :param jira_summaries: List of Jira summaries
        """
        self.jira_summaries = jira_summaries

    def suggest_summaries(self, user_input, top_n=5, threshold=60):
        """
        Suggest Jira summaries based on the user input with typo handling.
        :param user_input: The search input from the user.
        :param top_n: Number of top suggestions to return.
        :param threshold: Minimum similarity score to consider.
        :return: List of suggested Jira summaries.
        """
        suggestions = process.extract(user_input, self.jira_summaries, limit=top_n)
        return [(summary, score) for summary, score in suggestions if score >= threshold]

    def highlight_differences(self, user_input, suggested_summary):
        """
        Highlight the different words/letters between user input and the suggested Jira summary.
        :param user_input: User's search input
        :param suggested_summary: Suggested Jira summary
        :return: HTML formatted string with differences bolded
        """
        matcher = SequenceMatcher(None, user_input.lower(), suggested_summary.lower())
        highlighted_summary = ""
        last_match_end = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                highlighted_summary += suggested_summary[j1:j2]  # Add matching text
            elif tag in ["replace", "insert", "delete"]:
                highlighted_summary += f"<b>{suggested_summary[j1:j2]}</b>"  # Bold different text

        return highlighted_summary


# Example Usage
jira_summaries = [
    "User unable to login to the system",
    "Database connection issue causing downtime",
    "Application crashes on startup",
    "Email notifications are not being sent",
    "Performance degradation in search functionality"
]

search_engine = JiraSearchEngine(jira_summaries)

# User input with typo
user_input = "user unbale to logn"

# Get suggestions
suggested_summaries = search_engine.suggest_summaries(user_input)

# Display suggestions with bolded differences
for summary, _ in suggested_summaries:
    highlighted_summary = search_engine.highlight_differences(user_input, summary)
    print("Suggested Jira Summary:", highlighted_summary)
