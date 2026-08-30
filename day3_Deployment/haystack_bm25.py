import os

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever


# Create document store
document_store = InMemoryDocumentStore()

# PDF converter
converter = PyPDFToDocument()

# Document writer
writer = DocumentWriter(document_store=document_store)

# Create pipeline
pipeline = Pipeline()

pipeline.add_component("converter", converter)
pipeline.add_component("writer", writer)

pipeline.connect("converter.documents", "writer.documents")


# Find PDF files
pdf_files = []

for file in os.listdir("documents"):
    if file.lower().endswith(".pdf"):
        pdf_files.append(os.path.join("documents", file))

print("PDF files found:", len(pdf_files))

if len(pdf_files) != 5:
    print("Warning: Please put exactly 5 PDF files inside the documents folder.")


# Run indexing
pipeline.run({
    "converter": {
        "sources": pdf_files
    }
})

print("Documents indexed successfully.")

# BM25 Retriever
retriever = InMemoryBM25Retriever(
    document_store=document_store
)

# Ask questions
questions = [
    "What is the main topic of the document?",
    "What are the key concepts discussed?",
    "What are the main advantages?",
    "What are the main challenges?",
    "What methods are discussed?",
    "What are the important findings?",
    "What applications are mentioned?",
    "What are the limitations?",
    "What recommendations are provided?",
    "What is the conclusion?"
]

print("\n===== BM25 RETRIEVAL RESULTS =====")

for i, question in enumerate(questions, 1):

    result = retriever.run(
        query=question,
        top_k=3
    )

    print("\nQuestion", i)
    print(question)

    print("Retrieved documents:")

    for j, doc in enumerate(result["documents"], 1):
        print(
            f"{j}. Score: {doc.score} | "
            f"Content: {doc.content[:300].replace(chr(10), ' ')}..."
        )