import re
from typing import List

def split_by_acceptance_criteria(text: str) -> List[str]:
    """Split text by Acceptance Criteria blocks using 'Given / When / Then'."""
    lines = text.strip().splitlines()
    blocks = []
    current_block = []

    for line in lines:
        if re.match(r"^\s*Given\b", line, re.IGNORECASE):
            if current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
        current_block.append(line)
    if current_block:
        blocks.append("\n".join(current_block).strip())
    return blocks if len(blocks) > 1 else []

def split_by_numbered_list(text: str) -> List[str]:
    """Split text by numbered list items (e.g., 1., 2., 3.)"""
    # Matches numbered lines like "1. first item"
    pattern = r"(?:^|\n)(\d+\.\s.*?)(?=\n\d+\.|\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return [m.strip() for m in matches] if len(matches) > 1 else []

def smart_split_jira_description(text: str) -> List[str]:
    """
    First try to split by 'Given / When / Then' blocks,
    then numbered list if no AC pattern is found,
    else fall back to paragraph or sentence chunking.
    """
    # Try splitting by Acceptance Criteria
    ac_chunks = split_by_acceptance_criteria(text)
    if ac_chunks:
        return ac_chunks

    # Try splitting by numbered list
    list_chunks = split_by_numbered_list(text)
    if list_chunks:
        return list_chunks

    # Fallback: split by paragraph
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

jira_description = """
Acceptance Criteria:

Given the user is on the login page
When they enter valid credentials
Then they are redirected to the dashboard

Given the user is on the login page
When they enter incorrect credentials
Then an error message is shown

Steps to reproduce:
1. Open the app
2. Click login
3. Enter username and password
"""

chunks = smart_split_jira_description(jira_description)

for i, chunk in enumerate(chunks):
    print(f"🔹 Chunk {i+1}:\n{chunk}\n")
