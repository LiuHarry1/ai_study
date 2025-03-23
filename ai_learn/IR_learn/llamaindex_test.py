# pip install llama_index  llama-index-retrievers-bm25 llama-index-vector-stores-faiss llama-index-embeddings-huggingface

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Document
import Stemmer
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.core import VectorStoreIndex

docstore = SimpleDocumentStore()

# Load documents (JIRA tickets, etc.)
docs = [Document(text="JIRA allows users to create and track issues."),
        Document(text="BM25 is a lexical search method."),
        Document(text="MiniLM improves search with embeddings.")]

docstore.add_documents(docs)

# We can pass in the index, docstore, or list of nodes to create the retriever
sparse_retriever = BM25Retriever.from_defaults(
    docstore=docstore,
    similarity_top_k=2,
    # Optional: We can pass in the stemmer and set the language for stopwords
    # This is important for removing stopwords and stemming the query + text
    # The default is english for both
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)

query = "How does JIRA support tracking?"
sparse_results  = sparse_retriever.retrieve(query)

# print(sparse_results)
for node in sparse_results:
    print(node)




from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Dense index (MiniLM embeddings)
embed_model = HuggingFaceEmbedding(model_name="C:/apps/ml_model/sentence-transformers_all-MiniLM-L6-v2")
Settings.embed_model = embed_model
Settings.llm = None

dense_index = VectorStoreIndex.from_documents(docs)

vector_retriever  = dense_index.as_retriever()
dense_results = vector_retriever.retrieve(query)
print(dense_results)
# Use Score Normalization + Weighted Sum or RRF to combine
for node in dense_results:
    print(node)



from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [vector_retriever, sparse_retriever],
    similarity_top_k=2,
    num_queries=1,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=True,
    llm=None,
    # query_gen_prompt="...",  # we could override the query generation prompt here
)

hybrid_results = retriever.retrieve(query)

for node in hybrid_results:
    print(node)
