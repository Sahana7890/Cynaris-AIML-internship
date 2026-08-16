from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# 1. Prompt Template
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question briefly and clearly:\n\n{question}"
)

# 2. Ollama LLM
llm = OllamaLLM(model="llama3.2:3b")

# 3. Output Parser
parser = StrOutputParser()

# 4. Build chain
chain = prompt | llm | parser

# 5. Test with 5 inputs
questions = [
    "What is artificial intelligence?",
    "What is Python?",
    "What is machine learning?",
    "What is a database?",
    "What is cybersecurity?"
]

for i, question in enumerate(questions, 1):
    print(f"\n--- Test {i} ---")
    print("Question:", question)
    print("Answer:", chain.invoke({"question": question}))