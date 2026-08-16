from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# Create prompt
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer this question clearly and briefly: {question}"
)

# Connect to Ollama
llm = OllamaLLM(model="llama3.2:3b")

# Output parser
parser = StrOutputParser()

# Build chain
chain = prompt | llm | parser

# Test inputs
questions = [
    "What is machine learning?",
    "What is Python?",
    "What is artificial intelligence?",
    "What is LangChain?",
    "What is a database?"
]

# Run the chain
for i, question in enumerate(questions, 1):
    print(f"\nInput {i}: {question}")

    try:
        response = chain.invoke({"question": question})
        print("Output:", response)
    except Exception as e:
        print("Error:", e)