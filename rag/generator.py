import ollama

def format_context(contexts):
    formatted = []
    for item in contexts:
        formatted.append(
            f"Source: {item['source']}\n"
            f"Page: {item['page']}\n"
            f"Text: {item['text']}"
        )
    return "\n\n---\n\n".join(formatted)

def generate_answer(question, contexts, model_name="llama3.2:3b"):
    context_text = format_context(contexts)

    prompt = f"""
You are AgriVision AI, a plant disease information assistant.

Use only the provided context to answer the question.
Do not invent facts.
If the context is not enough, say so clearly.
Keep the answer simple and clear.

Context:
{context_text}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
