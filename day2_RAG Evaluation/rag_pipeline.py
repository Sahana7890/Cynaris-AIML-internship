from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. Load document
# --------------------------------------------------

loader = TextLoader(
    "data/document.txt",
    encoding="utf-8"
)

documents = loader.load()

print("Document loaded successfully.")


# --------------------------------------------------
# 2. Split document into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 3. Create Ollama embeddings
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embeddings model loaded.")


# --------------------------------------------------
# 4. Store embeddings in ChromaDB
# --------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="ragas_demo"
)

print("Documents stored in ChromaDB.")


# --------------------------------------------------
# 5. Create retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)

print("Retriever created.")


# --------------------------------------------------
# 6. Create Ollama LLM
# --------------------------------------------------

llm = OllamaLLM(
    model="llama3.2:3b"
)

print("LLM loaded.")


# --------------------------------------------------
# 7. Create prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful question-answering assistant.

Answer the question using ONLY the information provided
in the context.

If the answer is not present in the context, say:
"I don't know based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# 8. Generate answers
# --------------------------------------------------

questions = [
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "What is Deep Learning?",
    "What is Natural Language Processing?",
    "What is Generative AI?",
    "What is Retrieval-Augmented Generation?",
    "What are embeddings?",
    "What is ChromaDB?",
    "What is LangChain?",
    "What is Human-in-the-Loop?"
]


with open(
    "qa_results.txt",
    "w",
    encoding="utf-8"
) as file:

    for i, question in enumerate(questions, 1):

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(question)

        # Combine retrieved context
        context = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        # Create prompt
        messages = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        # Generate answer
        answer = llm.invoke(messages)

        print()
        print("=" * 60)
        print("Q", i, ":", question)
        print("A:", answer)

        # Save question and answer
        file.write(
            f"Q{i}: {question}\n"
        )

        file.write(
            f"A{i}: {answer}\n"
        )

        # Save retrieved context for RAGAS
        file.write(
            "CONTEXT:\n"
        )

        file.write(
            context + "\n\n"
        )


print()
print("=" * 60)
print("10 Q&A pairs generated successfully.")
print("Results saved to qa_results.txt")
print("=" * 60)