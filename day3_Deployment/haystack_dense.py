import os

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever


# Create document store
document_store = InMemoryDocumentStore(
    embedding_similarity_function="cosine"
)


# PDF converter
converter = PyPDFToDocument()

# Document embedder
document_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Writer
writer = DocumentWriter(document_store=document_store)


# Find PDFs
pdf_files = []

for file in os.listdir("documents"):
    if file.lower().endswith(".pdf"):
        pdf_files.append(os.path.join("documents", file))

print("PDF files found:", len(pdf_files))


# Convert PDFs
conversion_pipeline = Pipeline()

conversion_pipeline.add_component(
    "converter",
    converter
)

conversion_pipeline.add_component(
    "embedder",
    document_embedder
)

conversion_pipeline.add_component(
    "writer",
    writer
)

conversion_pipeline.connect(
    "converter.documents",
    "embedder.documents"
)

conversion_pipeline.connect(
    "embedder.documents",
    "writer.documents"
)


# Run indexing
conversion_pipeline.run({
    "converter": {
        "sources": pdf_files
    }
})

print("Documents embedded and indexed successfully.")


# Text embedder
text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

text_embedder.warm_up()


# Dense retriever
retriever = InMemoryEmbeddingRetriever(
    document_store=document_store
)


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


print("\n===== DENSE RETRIEVAL RESULTS =====")


for i, question in enumerate(questions, 1):

    embedding_result = text_embedder.run(
        text=question
    )

    result = retriever.run(
        query_embedding=embedding_result["embedding"],
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