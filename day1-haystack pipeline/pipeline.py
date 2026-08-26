from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever


# -----------------------------
# 1. Create Document Store
# -----------------------------
document_store = InMemoryDocumentStore()


# -----------------------------
# 2. Convert PDF files
# -----------------------------
pdf_files = list(Path("pdfs").glob("*.pdf"))

converter = PyPDFToDocument()

documents = []

for pdf in pdf_files:
    result = converter.run(sources=[pdf])
    documents.extend(result["documents"])

print("PDF files found:", len(pdf_files))
print("Documents loaded:", len(documents))


# -----------------------------
# 3. Write documents
# -----------------------------
writer = DocumentWriter(document_store=document_store)

writer.run(documents=documents)

print("Documents indexed:", document_store.count_documents())


# -----------------------------
# 4. BM25 Retriever
# -----------------------------
retriever = InMemoryBM25Retriever(
    document_store=document_store,
    top_k=3
)


# -----------------------------
# 5. Ask questions
# -----------------------------
questions = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is supervised learning?",
    "What is cybersecurity?",
    "What is cloud computing?",
    "What are the advantages of cloud computing?",
    "What is a neural network?",
    "What is phishing?",
    "What is deep learning?",
    "What is the purpose of encryption?"
]


# -----------------------------
# 6. Retrieve answers
# -----------------------------
for i, question in enumerate(questions, 1):

    result = retriever.run(query=question)

    print("\n" + "=" * 60)
    print(f"Question {i}: {question}")
    print("=" * 60)

    for j, doc in enumerate(result["documents"], 1):

        print(f"\nResult {j}")
        print("Score:", doc.score)
        print("Content:", doc.content[:500])