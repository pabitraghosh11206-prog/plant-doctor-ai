import ollama
from rag.llm.retrieval.search import search_knowledge
from rag.prompt.prompt import PROMPT


def format_context(results):
    """
    Convert retrieval results into a context string for the LLM.
    """

    context = ""

    for item in results:
        source = item["metadata"].get("source", "Unknown")
        page = item["metadata"].get("page", "N/A")
        text = item["text"]

        context += (
            f"Source: {source} (Page {page})\n"
            f"{text}\n\n"
        )

    return context


def generate_answer(
    question,
    crop=None,
    disease=None,
    top_k=5,
    model="llama3.2:3b"
):
    """
    Complete RAG answer generation.
    """

    # Retrieve relevant documents
    retrieval = search_knowledge(
        query=question,
        crop=crop,
        disease=disease,
        top_k=top_k
    )

    if retrieval["count"] == 0:
        return {
            "answer": retrieval["message"],
            "sources": []
        }

    # Convert retrieved chunks into context
    context = format_context(retrieval["results"])

    # Build prompt
    prompt = PROMPT.format(
        context=context,
        question=question
    )

    # Ask Ollama
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return {
        "answer": answer,
        "sources": retrieval["results"]
    }
