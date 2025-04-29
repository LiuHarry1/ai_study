import openai
import json

openai.api_key = "your-api-key"

def batch_clean_and_tag_jira(jira_list):
    """
    jira_list: List of dicts with 'jira_id', 'summary', and 'description'
    """
    if len(jira_list) > 5:
        raise ValueError("Can only process up to 5 Jira issues at once.")

    system_prompt = """You are a helpful assistant that cleans and tags Jira issues.

For each Jira issue:
1. Remove deprecated/irrelevant content (e.g. strikethroughs, changelogs).
2. Normalize the language and summarize clearly.
3. Assign one or more **business categories** from the list below:

Available categories: payment, posting, entitlement, alert, notification, announcement, event, position, tech, unknown

Return a list of JSON objects in this format:
[
  {
    "jira_id": "...",
    "cleaned_summary": "...",
    "cleaned_description": "...",
    "business_categories": ["...", "..."]
  },
  ...
]
"""

    # Format all Jiras as input
    input_text = "Here are 5 Jira issues:\n"
    for i, jira in enumerate(jira_list, start=1):
        input_text += (
            f"Jira {i}:\n"
            f"ID: {jira['jira_id']}\n"
            f"Summary: {jira['summary']}\n"
            f"Description: {jira['description']}\n\n"
        )

    # LLM call
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ],
        temperature=0.3
    )

    result = response['choices'][0]['message']['content']

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse LLM response:\n", result)
        return None
