import re
from typing import List, Dict


# Mock GPT function (replace with real API call)
def gpt_fallback(prompt: str) -> str:
    # In a real case, call the GPT model to get the response
    print("🧠 Sending to GPT: ", prompt)
    return "Is alert 37 resolved?"  # Example GPT output


# Regex patterns for entity extraction (alert numbers, tasks, etc.)
ENTITY_PATTERNS = {
    "alert": r"\balert ?\d+\b",  # Matches "alert 37", "alert37"
    "task": r"\bTask-\d+\b",  # Matches "Task-123"
}


# Extract entities using regex
def extract_entities_from_message(text: str) -> dict:
    entities = {}
    for entity_type, pattern in ENTITY_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            entities[entity_type] = match.group().replace(" ", "")  # Normalize alerts and tasks
    return entities


# Rule-based rewrite for known entities (e.g., resolving pronouns)
def rewrite_with_rules(question: str, entities: dict) -> str:
    """
    Replaces ambiguous references in the question with known entities.
    Example: "Was it resolved?" becomes "Was alert 37 resolved?"
    """
    for pronoun, replacement in entities.items():
        question = question.replace(pronoun, replacement)
    return question


# Process the question and apply rule-based system first
def process_question_with_rules(question: str, chat_history: List[Dict]) -> str:
    # Extract entities from the chat history and question itself
    entities = {}
    for msg in chat_history[-3:]:  # Check last 3 rounds
        entities.update(extract_entities_from_message(msg["content"]))
    entities.update(extract_entities_from_message(question))  # Include current question

    # Try rewriting with rules first
    rewritten_question = rewrite_with_rules(question, entities)

    # Check if the rule-based rewrite looks good
    if rewritten_question != question:  # If rules applied any change
        return rewritten_question

    return None  # No changes, proceed to GPT fallback


# Use GPT for fallback if rules are insufficient
def process_question_with_gpt(chat_history: List[Dict], question: str) -> str:
    """
    Use GPT to rewrite the question based on chat history context
    if the rule-based system cannot handle it.
    """
    # Build prompt for GPT based on recent chat history
    recent_history = "\n".join([f"User: {msg['content']}" for msg in chat_history[-3:]])

    prompt = f"""
    Given the following conversation history, rewrite the user's question to avoid ambiguity:

    Chat History:
    {recent_history}
    User's Question: {question}

    Rewritten Question:
    """

    # Call GPT fallback (this can be replaced by a real GPT call)
    return gpt_fallback(prompt)


# Main function to process the question with rules and fallback to GPT
def process_question(chat_history: List[Dict], question: str) -> str:
    # First, try rule-based rewrite
    rewritten_by_rules = process_question_with_rules(question, chat_history)

    if rewritten_by_rules:  # If rule-based rewrite is successful
        return rewritten_by_rules

    # If rule-based rewrite doesn't work, fall back to GPT
    return process_question_with_gpt(chat_history, question)


# Example usage
chat_history = [
    {"sender": "user", "content": "Can you check alert 37?"},
    {"sender": "user", "content": "Was it resolved?"}
]

current_question = "Was it resolved?"

# Process the question (rule-based + GPT fallback)
final_question = process_question(chat_history, current_question)
print("✅ Final Rewritten Question: ", final_question)
