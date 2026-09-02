
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper


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


ground_truths = [
    "Artificial Intelligence is a field of computer science that creates systems capable of performing tasks requiring human intelligence.",
    "Machine Learning is a subset of Artificial Intelligence that allows computers to learn patterns from data and make predictions.",
    "Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers.",
    "Natural Language Processing enables computers to understand, process, and generate human language.",
    "Generative AI generates new content such as text, images, audio, video, and code.",
    "RAG combines information retrieval with a language model to retrieve relevant information and generate answers.",
    "Embeddings are numerical representations that capture the semantic meaning of data.",
    "ChromaDB is a vector database used to store and search embeddings.",
    "LangChain is a framework for building applications powered by language models.",
    "Human-in-the-Loop involves humans in AI decision-making to improve reliability."
]


answers = []
contexts = []

with open("qa_results.txt", "r", encoding="utf-8") as file:
    content = file.read()

blocks = content.strip().split("\n\n")

for block in blocks:
    lines = block.split("\n")

    answer = ""
    context_lines = []
    reading_context = False

    for line in lines:
        if line.startswith("A"):
            answer = line.split(":", 1)[1].strip()
        elif line.startswith("CONTEXT:"):
            reading_context = True
        elif reading_context and line.strip():
            context_lines.append(line.strip())

    if answer:
        answers.append(answer)

    context = " ".join(context_lines)

    if context:
        contexts.append([context])
    else:
        contexts.append([answer])


answers = answers[:10]
contexts = contexts[:10]

data = {
    "question": questions[:len(answers)],
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths[:len(answers)]
}

dataset = Dataset.from_dict(data)

llm = ChatOllama(
    model="llama3.2:3b"
)

ragas_llm = LangchainLLMWrapper(llm)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=ragas_llm,
    embeddings=embeddings
)

print()
print("==============================")
print("RAGAS EVALUATION RESULTS")
print("==============================")
print(result)

with open("ragas_results.txt", "w", encoding="utf-8") as file:
    file.write(str(result))

print()
print("Results saved to ragas_results.txt")

