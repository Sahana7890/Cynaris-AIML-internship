import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_collection(
    name="internship_documents",
    embedding_function=embedding_function
)

query = "What is a vector database?"

results = collection.query(
    query_texts=[query],
    n_results=5
)

print("\nQuery:", query)
print("\nTop 5 Results:\n")

for i, document in enumerate(results["documents"][0]):
    print(f"{i+1}. {document}")
    print("Distance:", results["distances"][0][i])
    print()