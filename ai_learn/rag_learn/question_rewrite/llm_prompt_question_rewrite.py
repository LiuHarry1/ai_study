REWRITE_PROMPT_TEMPLATE = """
You are an AI assistant helping to rewrite user questions in a chatbot.

Given the chat history and the current user question, your task is to rewrite the question so that it is **self-contained** (i.e., doesn't contain ambiguous pronouns like "it", "he", "they", etc).

Examples:
Chat History:
User: What's the status of alert 37?
User: Was it escalated?
Rewritten Question: Was alert 37 escalated?

Chat History:
User: Who is responsible for the analytics module?
User: Did he also build the dashboard?
Rewritten Question: Did John also build the dashboard?

Now rewrite the current question using the given history.

Chat History:
{chat_history}
User: {question}
Rewritten Question:
""".strip()

def get_recent_chat_history(chat_history: list[dict], max_rounds: int = 3) -> str:
    """
    Get the most recent 'max_rounds' rounds of chat history for context.
    """
    recent_history = chat_history[-max_rounds:]
    return "\n".join([f"User: {msg['content']}" for msg in recent_history])


def build_chat_history_string(messages: list[dict]) -> str:
    # Convert chat history into formatted string
    return "\n".join([f"{msg['sender'].capitalize()}: {msg['content']}" for msg in messages])

def rewrite_question_with_llm(chat_history: list[dict], question: str, llm) -> str:
    # Get recent chat history (last 1-3 rounds)
    # recent_history = get_recent_chat_history(chat_history, max_rounds=3)

    prompt = REWRITE_PROMPT_TEMPLATE.format(
        chat_history=build_chat_history_string(chat_history),
        question=question
    )

    # Send prompt to LLM (OpenAI/Gemini/etc.)
    response = llm(prompt)
    return response.strip()

# Mock LLM function for testing
def llm(prompt: str) -> str:
    print("🧠 Prompt to LLM:")
    print(prompt)
    return "Was alert 37 resolved?"

chat_history = [
    {"sender": "user", "content": "Can you check alert 37?"},
    {"sender": "user", "content": "Was it resolved?"}
]

current_question = "Was it resolved?"

rewritten = rewrite_question_with_llm(chat_history, current_question, llm)
print("✅ Rewritten Question:", rewritten)
