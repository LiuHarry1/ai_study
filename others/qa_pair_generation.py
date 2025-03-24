from jira import JIRA
import openai
import json

# Connect to Jira
jira_url = "https://your-jira-instance.atlassian.net"
jira = JIRA(jira_url, basic_auth=("email@example.com", "your_api_token"))

# Fetch Jira issues
issues = jira.search_issues('project=YOUR_PROJECT', maxResults=50)

qa_dataset = []

for issue in issues:
    summary = issue.fields.summary
    description = issue.fields.description or "No description available"
    comments = " ".join([comment.body for comment in jira.comments(issue)]) if issue.fields.comment else ""

    context = f"Summary: {summary}\nDescription: {description}\nComments: {comments}"

    # Single API call to generate multiple QA pairs
    prompt = f"""
    You are an AI assistant specialized in analyzing Jira issues and generating high-quality question-answer pairs. 
    Based on the given Jira issue, generate **five diverse question-answer pairs**.

    ### **Jira Issue Information**
    - **Summary**: {summary}
    - **Description**: {description}
    - **Comments**: {comments}

    ### **Instructions**
    1. **Generate 5 diverse questions** related to the issue.
    2. Ensure different types of questions:
       - **Basic** (What is the issue about?)
       - **Contextual** (Why did this issue occur?)
       - **Explanatory** (How can this be resolved?)
       - **Procedural** (What are the steps to fix it?)
       - **Impact-based** (What happens if unresolved?)
    3. Provide **detailed answers** based on the issue description and comments.
    4. Format output as structured JSON:

    ```json
    [
      {"question": "Q1", "answer": "A1"},
      {"question": "Q2", "answer": "A2"},
      {"question": "Q3", "answer": "A3"},
      {"question": "Q4", "answer": "A4"},
      {"question": "Q5", "answer": "A5"}
    ]
    ```
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )

    # Parse the LLM output as JSON
    try:
        qa_pairs = json.loads(response["choices"][0]["message"]["content"])
        qa_dataset.extend(qa_pairs)
    except json.JSONDecodeError:
        print(f"Error parsing JSON for issue {issue.key}")

# Save as JSONL for fine-tuning
with open("jira_qa_dataset.jsonl", "w") as f:
    for qa in qa_dataset:
        f.write(json.dumps(qa) + "\n")

print("QA dataset generated successfully!")
