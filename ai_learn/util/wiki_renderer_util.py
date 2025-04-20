import re


# /Purpose of Lookarounds
# Why (?<!\w)?
# Prevents matching in the middle of a word like well-known.
#
# Why (?!\w)?
# Prevents matching something like known-text, where -text- is part of a longer word.
#
# Why [^-\n]+ inside?
# Ensures we don't prematurely match other dashes or span across multiple lines.

def remove_wiki_strikethrough(text):
    # Match -text- only if it's not surrounded by word characters (e.g., spaces or punctuation)
    return re.sub(r'(?<!\w)-[^-\n]+-(?!\w)', '', text)

def clean_whitespace(text):
    # Collapse multiple blank lines and strip surrounding space
    return re.sub(r'\n\s*\n', '\n\n', text).strip()


description = """
This is a valid line.
-This line should be removed- because it's old.
But keep - item in list.
Also keep well-known cases.
"""

cleaned = remove_wiki_strikethrough(description)
# cleaned = clean_whitespace(cleaned)
print(cleaned)
