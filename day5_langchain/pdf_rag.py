import chromadb
import ollama
from pypdf import PdfReader

# -----------------------------
# 1. Read PDF
# -----------------------------

pdf_path = "sample.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("PDF loaded successfully.")
print("Characters extracted:", len(text))


# -----------------------------
# 2. Split text into chunks
# -----------------------------

chunk_size = 800

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Create ChromaDB collection
# -----------------------------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="pdf_documents",
    metadata={"hnsw:space": "cosine"}
)


# -----------------------------
# 4. Add PDF chunks
# -----------------------------

ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.upsert(
    ids=ids,
    documents=chunks
)

print("PDF chunks added to ChromaDB.")


# -----------------------------
# 5. User question
# -----------------------------

question = input("\nAsk a question about the PDF: ")


# -----------------------------
# 6. Retrieve top 3 chunks
# -----------------------------

results = collection.query(
    query_texts=[question],
    n_results=3
)

retrieved_chunks = results["documents"][0]

print("\nTOP 3 RETRIEVED CHUNKS:")

for i, chunk in enumerate(retrieved_chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk[:500])


# -----------------------------
# 7. Create context
# -----------------------------

context = "\n\n".join(retrieved_chunks)


# -----------------------------
# 8. Send context to Ollama
# -----------------------------

prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in the context, say:
"I could not find the answer in the PDF."
"""

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nLLM ANSWER:")
print(response["message"]["content"])