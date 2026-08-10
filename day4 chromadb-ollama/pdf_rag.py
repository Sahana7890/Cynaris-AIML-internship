import chroma_db
import ollama

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

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
# 2. Split PDF into chunks
# -----------------------------

chunk_size = 1000

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Create ChromaDB collection
# -----------------------------

client = chromadb.PersistentClient(path="./pdf_chroma_db")

collection = client.get_or_create_collection(
    name="pdf_documents"
)


# -----------------------------
# 4. Add chunks
# -----------------------------

ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.upsert(
    ids=ids,
    documents=chunks
)

print("PDF chunks stored in ChromaDB.")


# -----------------------------
# 5. Ask question
# -----------------------------

question = input("\nEnter your question: ")


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
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in the context, say:
"I could not find the answer in the provided document."

Answer:
"""

response = ollama.chat(
    model="qwen2.5",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nOLLAMA ANSWER:")
print(response["message"]["content"])