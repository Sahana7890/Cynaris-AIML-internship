import chromadb

# Create a persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.get_or_create_collection(
    name="internship_documents",
    metadata={"description": "20 sample documents for vector search"}
)

documents = [
    "Python is a popular programming language used for data science.",
    "Machine learning allows computers to learn patterns from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing works with human language.",
    "Computer vision allows computers to understand images.",
    "ChromaDB is a vector database used for storing embeddings.",
    "Vector databases support similarity search.",
    "Embeddings represent text as numerical vectors.",
    "Cosine similarity measures similarity between vectors.",
    "Metadata can be used to filter vector database results.",
    "Ollama allows large language models to run locally.",
    "RAG stands for Retrieval Augmented Generation.",
    "RAG retrieves relevant information before generating an answer.",
    "PDF documents can be divided into smaller text chunks.",
    "Top-k retrieval returns the most relevant documents.",
    "Flask is a lightweight Python web framework.",
    "Streamlit can be used to build interactive Python applications.",
    "Git is used for version control.",
    "GitHub is a platform for hosting Git repositories.",
    "MLOps combines machine learning with software engineering practices."
]

metadatas = [
    {"category": "python"},
    {"category": "machine_learning"},
    {"category": "deep_learning"},
    {"category": "nlp"},
    {"category": "computer_vision"},
    {"category": "vector_database"},
    {"category": "vector_database"},
    {"category": "embeddings"},
    {"category": "similarity"},
    {"category": "metadata"},
    {"category": "llm"},
    {"category": "rag"},
    {"category": "rag"},
    {"category": "pdf"},
    {"category": "retrieval"},
    {"category": "web"},
    {"category": "web"},
    {"category": "git"},
    {"category": "github"},
    {"category": "mlops"}
]

ids = [f"doc_{i}" for i in range(1, 21)]

# Add documents
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print("20 documents added successfully.")
print("Collection:", collection.name)
print("Total documents:", collection.count())

# Similarity search
query = "How does vector similarity search work?"

results = collection.query(
    query_texts=[query],
    n_results=3
)

print("\nTOP 3 SIMILAR RESULTS:")

for i, document in enumerate(results["documents"][0]):
    print(f"\nResult {i + 1}:")
    print(document)
    print("Metadata:", results["metadatas"][0][i])

# Metadata filtering
filtered_results = collection.query(
    query_texts=["machine learning"],
    where={"category": "machine_learning"},
    n_results=3
)

print("\nMETADATA FILTER RESULTS:")

for document in filtered_results["documents"][0]:
    print(document)