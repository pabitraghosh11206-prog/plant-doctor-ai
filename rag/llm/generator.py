import ollama

# ==============================
# Function to generate an answer
# ==============================
def generate_answer(context: str, question: str) -> str:
    """
    Sends the context and question to the Llama model
    and returns the generated answer.
    """

    # Create the prompt
    prompt = f"""
You are AgriVision AI, an agricultural assistant.

IMPORTANT RULES:
1. Answer ONLY using the context below.
2. Do NOT make up information.
3. If the answer is not found in the context, reply:
   "I could not find that information in the provided documents."

-------------------------
CONTEXT:
{context}
-------------------------

QUESTION:
{question}

ANSWER:
"""

    # Print the prompt (for debugging)
    print("\n========== PROMPT SENT TO LLM ==========\n")
    print(prompt)
    print("\n========================================\n")

    try:
        # Send prompt to Ollama
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Print the full response (for debugging)
        print("========== RAW RESPONSE ==========")
        print(response)
        print("==================================\n")

        return response["message"]["content"]

    except Exception as e:
        return f"Error: {e}"


# ==================================
# Main function (runs this file only)
# ==================================
if __name__ == "__main__":

    # Sample context (later this will come from ChromaDB)
    sample_context = """
Early blight is a fungal disease affecting tomato plants.

Symptoms:
- Brown circular spots on leaves
- Yellowing around the spots

Prevention:
- Avoid overhead watering
- Remove infected leaves
- Improve air circulation
- Rotate crops every season

Treatment:
- Apply recommended fungicides
- Remove infected plant material
"""

    # Sample question
    sample_question = "How can I prevent Early Blight?"

    print("========== CONTEXT ==========")
    print(sample_context)

    print("\n========== QUESTION ==========")
    print(sample_question)

    # Generate answer
    answer = generate_answer(sample_context, sample_question)

    print("\n========== FINAL ANSWER ==========\n")
    print(answer)
    print("\n==================================")