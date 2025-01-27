from difflib import SequenceMatcher

def compare_texts_detailed(text1, text2):
    words1 = text1.split()
    words2 = text2.split()
    matcher = SequenceMatcher(None, words1, words2)

    replaced = []
    inserted = []
    deleted = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        print(tag, i1, i2, j1, j2)
        if tag == 'replace':
            replaced.extend([(words1[i1:i2], words2[j1: j2])])
        elif tag == 'insert':
            inserted.extend(words2[j1:j2])
        elif tag == 'delete':
            deleted.extend(words1[i1:i2])

    return {
        "modified": replaced,
        "added": inserted,
        "deleted": deleted
    }

# Example texts
text1 = "quick brown fox jumps here there over the lazy dog. how are you?"
text2 = """The quick fox leaps over a lazy cat. 
how am you?"""

# Compare the texts
result = compare_texts_detailed(text1, text2)

# Output the result
print("Modified Words:", result["modified"])
print("Added Words:", result["added"])
print("Deleted Words:", result["deleted"])
