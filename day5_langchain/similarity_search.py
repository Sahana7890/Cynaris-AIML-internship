import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("documents")

query = "How are computers able to learn from data?"

results = collection.query(
    query_texts=[query],
    n_results=5
)

print("\nQUERY:")
print(query)

print("\nTOP 5 RESULTS:")

for i, document in enumerate(results["documents"][0]):
    print(f"\n{i + 1}. {document}")