from rag.llm.rag_pipeline import rag_pipeline

print("=" * 60)
print("Welcome to AgriVision AI")
print("Ask questions about plant diseases.")
print("Type 'exit' to quit.")
print("=" * 60)

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = rag_pipeline(question)

    print("\nAgriVision AI:")
    print(result["answer"])