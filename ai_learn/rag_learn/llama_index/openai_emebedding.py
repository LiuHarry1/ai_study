import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import openai

# --- 🔑 API Key ---
openai.api_key = "sk-..."  # Your OpenAI key
os.environ["OPENAI_API_KEY"] = openai.api_key

# --- ⚙️ LlamaIndex Global Settings ---
Settings.llm = OpenAI(model="gpt-4", temperature=0.3)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# --- 🗂️ Sample Jira Data ---
jira_data = [
    {
        "id": "JIRA-101",
        "summary": "Login button not responsive on mobile",
        "description": "The login button does not work on iPhone Safari. Clicking does nothing. JS console shows TypeError."
    },
    {
        "id": "JIRA-102",
        "summary": "Password reset email not sent",
        "description": "Users report not receiving password reset emails. Verified SMTP logs and found intermittent failures."
    },
    # Add more tickets here...
]

# --- 📄 Convert Jira Data to Documents ---
documents = []
for ticket in jira_data:
    content = (
        f"Jira ID: {ticket['id']}\n"
        f"Summary: {ticket['summary']}\n"
        f"Description: {ticket['description']}"
    )
    documents.append(Document(text=content))

# --- 🧠 Build the Vector Index ---
index = VectorStoreIndex.from_documents(documents)

# --- 🔍 Setup Query Engine ---
query_engine = index.as_query_engine(similarity_top_k=3)

# --- 💬 Query Loop ---
while True:
    user_input = input("\nAsk a question about the Jira tickets (or type 'exit'): ")
    if user_input.lower() == "exit":
        break

    response = query_engine.query(user_input)
    print("\n🔎 Answer:\n", response.response)
