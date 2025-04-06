import spacy
from typing import List, Dict

nlp = spacy.load("en_core_web_sm")

class SubjectTracker:
    def __init__(self):
        self.last_entities = {}  # 保存最近提到的实体，比如 person, org, product

    def update_from_message(self, text: str):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "PRODUCT", "GPE", "WORK_OF_ART"]:
                self.last_entities[ent.label_] = ent.text

    def resolve_reference(self, pronoun: str):
        pronoun = pronoun.lower()
        if pronoun in ["he", "him", "his"]:
            return self.last_entities.get("PERSON")
        elif pronoun in ["she", "her"]:
            return self.last_entities.get("PERSON")
        elif pronoun in ["it", "this", "that", "the issue", "the problem"]:
            return self.last_entities.get("PRODUCT") or self.last_entities.get("WORK_OF_ART")
        elif pronoun in ["they", "them", "their"]:
            return self.last_entities.get("ORG")
        return None  # 如果无法判断

    def get_context_summary(self):
        return self.last_entities

tracker = SubjectTracker()

# 模拟对话过程
messages = [
    "Who created the internal dashboard?",
    "It was developed by John.",
    "Does he also work on the analytics module?",
]

for msg in messages:
    tracker.update_from_message(msg)

# 当前问题中指代“he”
rewrite_input = "Does he also work on the analytics module?"
resolved = tracker.resolve_reference("he")

if resolved:
    rewritten = rewrite_input.replace("he", resolved)
else:
    rewritten = rewrite_input

print("👉 Rewritten Question:", rewritten)

