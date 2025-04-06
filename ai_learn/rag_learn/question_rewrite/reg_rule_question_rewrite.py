import re
from typing import Dict

ENTITY_PATTERNS = {
    "alert": r"\balert ?\d+\b",
    "task": r"\bTask-\d+\b",
    "product": r"\b(internal dashboard|analytics module|reporting tool)\b",
    "person": r"\b[A-Z][a-z]+\b",  # Simplified: capitalized words
}

class RuleBasedSubjectTracker:
    def __init__(self):
        self.last_entities = {}

    def update_from_message(self, text: str):
        for entity_type, pattern in ENTITY_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Normalize alert like "alert 2" → "alert2"
                entity = match.group().replace(" ", "") if entity_type == "alert" else match.group()
                self.last_entities[entity_type] = entity

    def resolve_reference(self, pronoun: str):
        pronoun = pronoun.lower()
        if pronoun in ["he", "him", "his", "she", "her"]:
            return self.last_entities.get("person")
        elif pronoun in ["it", "this", "that", "the issue", "the alert"]:
            return (
                self.last_entities.get("alert")
                or self.last_entities.get("product")
                or self.last_entities.get("task")
            )
        elif pronoun in ["they", "them", "their"]:
            return self.last_entities.get("org")
        return None

    def rewrite_question(self, question: str) -> str:
        tokens = question.split()
        new_tokens = []
        for token in tokens:
            resolved = self.resolve_reference(token)
            new_tokens.append(resolved if resolved else token)
        return " ".join(new_tokens)

    def get_context(self) -> Dict[str, str]:
        return self.last_entities


tracker = RuleBasedSubjectTracker()

# Simulated chat
tracker.update_from_message("Can you check alert 37?")
tracker.update_from_message("Was it resolved?")
tracker.update_from_message("What about alert2?")

print("📌 Rewritten:", tracker.rewrite_question("Was it resolved?"))
print("🧠 Context:", tracker.get_context())
