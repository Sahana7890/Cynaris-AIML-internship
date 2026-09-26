from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Load Q&A data
questions = []
answers = []
contexts = []
ground_truths = []

with open("qa_results.txt", "r", encoding="utf-8") as file:

    content = file.read()

blocks = content.split("=" * 80)

for block in blocks:

    lines = block.strip().splitlines()

    if not lines:
        continue

    question = ""
    answer = ""
    context_list = []

    for i, line in enumerate(lines):

        if line.startswith("QUESTION:"):
            question = line.replace("QUESTION:", "").strip()

        elif line.startswith("ANSWER:"):
            answer = line.replace("ANSWER:", "").strip()

        elif line.startswith("CONTEXTS:"):
            context_list = [
                x.strip()
                for x in lines[i + 1:]
                if x.strip()
            ]

    if question and answer and context_list:

        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)

        # Ground truth based on retrieved context
        ground_truths.append(context_list[0])

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

print("Number of evaluation samples:", len(dataset))

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print("\nRAGAS EVALUATION RESULTS")
print(result)

with open("ragas_results.txt", "w", encoding="utf-8") as file:
    file.write(str(result))

print("\nResults saved to ragas_results.txt")