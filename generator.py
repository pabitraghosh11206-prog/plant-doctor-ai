import ollama

def generate_answer(context: str, question: str) -> str:
    prompt = f"""
You are an agricultural assistant.

Use only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    sample_context = (
        "Early blight causes brown spots on tomato leaves. "
        "Avoid overhead watering and remove infected leaves."
    )

    sample_question = "How can I prevent Early Blight?"

    answer = generate_answer(sample_context, sample_question)
    print(answer)