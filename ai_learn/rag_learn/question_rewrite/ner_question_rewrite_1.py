import spacy
import re
from typing import Dict

nlp = spacy.load("en_core_web_sm")

class SubjectTracker:
    def __init__(self):
        self.last_entities = {}

    def update_from_message(self, text: str):
        doc = nlp(text)
        # Use spaCy NER first
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "PRODUCT", "GPE", "WORK_OF_ART"]:
                self.last_entities[ent.label_] = ent.text

        # Add custom regex matcher for 'alert'
        alert_match = re.search(r"\balert ?\d+\b", text, re.IGNORECASE)
        if alert_match:
            # Normalize "alert 2" → "alert2"
            alert_id = alert_match.group().replace(" ", "")
            self.last_entities["ALERT"] = alert_id

    def resolve_reference(self, pronoun: str):
        pronoun = pronoun.lower()
        if pronoun in ["he", "him", "his", "she", "her"]:
            return self.last_entities.get("PERSON")
        elif pronoun in ["it", "this", "that", "the issue", "the alert"]:
            return (
                self.last_entities.get("ALERT")
                or self.last_entities.get("PRODUCT")
                or self.last_entities.get("WORK_OF_ART")
            )
        elif pronoun in ["they", "them", "their"]:
            return self.last_entities.get("ORG")
        return None

    def rewrite_question(self, question: str) -> str:
        tokens = question.split()
        new_tokens = []
        for token in tokens:
            resolved = self.resolve_reference(token)
            new_tokens.append(resolved if resolved else token)
        return " ".join(new_tokens)

    def get_context_summary(self) -> Dict[str, str]:
        return self.last_entities


tracker = SubjectTracker()

tracker.update_from_message("Can you check alert 37?")
tracker.update_from_message("Was it resolved?")
tracker.update_from_message("Who created the analytics module?")
tracker.update_from_message("Did he also test alert2?")

print("📌 Rewritten:", tracker.rewrite_question("Did he also test it?"))
print("🧠 Context:", tracker.get_context_summary())
