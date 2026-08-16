import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("documents")

results = collection.query(
    query_texts=["artificial intelligence"],
    n_results=5,
    where={"category": "AI"}
)

print("AI CATEGORY RESULTS:")

for document, metadata in zip(
    results["documents"][0],
    results["metadatas"][0]
):
    print("\nDocument:", document)
    print("Metadata:", metadata)