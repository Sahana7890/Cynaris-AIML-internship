from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# 1. Create PromptTemplate
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question clearly and briefly:\n\n{question}"
)

# 2. Create Ollama LLM
llm = OllamaLLM(
    model="llama3.2:3b"
)

# 3. Create OutputParser
parser = StrOutputParser()

# 4. Build chain
chain = prompt | llm | parser

# 5. Test with 5 inputs
questions = [
    "What is Python?",
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is LangChain?",
    "What is an API?"
]

for i, question in enumerate(questions, 1):
    print(f"\n--- Test {i} ---")
    print("Question:", question)

    response = chain.invoke({
        "question": question
    })

    print("Answer:", response)