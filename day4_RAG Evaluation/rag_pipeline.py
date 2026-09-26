from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# Load documents
loader = TextLoader("data/knowledge.txt")
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

# Create embeddings
embeddings = OllamaEmbeddings(
    model="llama3.2:3b"
)

# Create ChromaDB
vectorstore = Chroma(
    collection_name="ragas_demo",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add documents
vectorstore.add_documents(chunks)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# LLM
llm = Ollama(
    model="llama3.2:3b"
)

# Questions
questions = [
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is deep learning?",
    "What is Natural Language Processing?",
    "What is Generative AI?",
    "What is a Large Language Model?",
    "What is Retrieval Augmented Generation?",
    "What is ChromaDB?"
]

results = []

for question in questions:

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = llm.invoke(prompt)

    print("\nQUESTION:", question)
    print("ANSWER:", answer)

    results.append({
        "question": question,
        "answer": answer,
        "contexts": [doc.page_content for doc in docs]
    })

# Save results
with open("qa_results.txt", "w", encoding="utf-8") as file:

    for item in results:

        file.write("QUESTION: " + item["question"] + "\n")
        file.write("ANSWER: " + item["answer"] + "\n")
        file.write("CONTEXTS:\n")

        for context in item["contexts"]:
            file.write(context + "\n")

        file.write("\n" + "=" * 80 + "\n")

print("\n10 Q&A pairs generated successfully.")