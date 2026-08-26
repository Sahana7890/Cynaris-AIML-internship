from pathlib import Path

from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder


# -------------------------------------------------
# 1. Create Document Store
# -------------------------------------------------
document_store = InMemoryDocumentStore()


# -------------------------------------------------
# 2. Read the 5 PDF files
# -------------------------------------------------
pdf_folder = Path("pdfs")

documents = []

for pdf_file in pdf_folder.glob("*.pdf"):
    try:
        import pypdf

        reader = pypdf.PdfReader(str(pdf_file))

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if text.strip():
            documents.append(
                Document(
                    content=text,
                    meta={"file_name": pdf_file.name}
                )
            )

            print(f"Loaded: {pdf_file.name}")

    except Exception as e:
        print(f"Error reading {pdf_file.name}: {e}")


# -------------------------------------------------
# 3. Store documents
# -------------------------------------------------
document_store.write_documents(documents)

print("\nTotal documents indexed:", len(documents))


# -------------------------------------------------
# 4. Create BM25 Retriever
# -------------------------------------------------
retriever = InMemoryBM25Retriever(
    document_store=document_store,
    top_k=3
)


# -------------------------------------------------
# 5. Create Haystack Pipeline
# -------------------------------------------------
pipeline = Pipeline()

pipeline.add_component(
    "retriever",
    retriever
)


# -------------------------------------------------
# 6. Ask questions
# -------------------------------------------------
questions = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is cybersecurity?",
    "What is phishing?",
    "What is a neural network?",
    "What is data science?",
    "What is classification?",
    "What is regression?"
]


# -------------------------------------------------
# 7. Run BM25 retrieval
# -------------------------------------------------
for number, question in enumerate(questions, start=1):

    print("\n" + "=" * 60)
    print(f"Question {number}: {question}")
    print("=" * 60)

    result = pipeline.run(
        {
            "retriever": {
                "query": question
            }
        }
    )

    documents_found = result["retriever"]["documents"]

    if not documents_found:
        print("No documents found.")
        continue

    for rank, document in enumerate(documents_found, start=1):

        print(f"\nResult {rank}")
        print("File:", document.meta.get("file_name"))
        print("Score:", document.score)

        content = document.content.replace("\n", " ")

        print("Content:", content[:500])