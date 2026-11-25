from rag.llm.retrieval.search import search_knowledge
from rag.llm.generator import generate_answer


def rag_pipeline(
    question,
    crop=None,
    disease=None,
    top_k=5
):
    """
    Complete RAG Pipeline

    1. Retrieve documents
    2. Build context
    3. Generate answer
    4. Return answer + retrieved sources
    """

    # -----------------------------------
    # STEP 1 : Retrieve relevant chunks
    # -----------------------------------

    retrieval_result = search_knowledge(
        query=question,
        top_k=top_k,
        crop=crop,
        disease=disease
    )

    # -----------------------------------
    # STEP 2 : Check retrieval success
    # -----------------------------------

    if retrieval_result["count"] == 0:

        return {

            "answer": retrieval_result["message"],

            "sources": []

        }

    # -----------------------------------
    # STEP 3 : Build Context
    # -----------------------------------

    context = ""

    for doc in retrieval_result["results"]:

        context += doc["text"]

        context += "\n\n"

    # -----------------------------------
    # STEP 4 : Generate Answer
    # -----------------------------------

    answer = generate_answer(
        context=context,
        question=question
    )

    # -----------------------------------
    # STEP 5 : Collect Sources
    # -----------------------------------

    sources = []

    for doc in retrieval_result["results"]:

        source_info = {

            "chunk_id": doc["chunk_id"],

            "source": doc["metadata"].get(
                "source",
                "Unknown"
            ),

            "distance": doc["distance"]

        }

        sources.append(source_info)

    # -----------------------------------
    # STEP 6 : Return Everything
    # -----------------------------------

    return {

        "answer": answer,

        "sources": sources

    }


if __name__ == "__main__":

    question = "How can I prevent Early Blight?"

    result = rag_pipeline(

        question=question,

        crop="tomato",

        disease="Tomato___Early_blight"

    )

    print("\n==========================")

    print("ANSWER")

    print("==========================")

    print(result["answer"])

    print("\n==========================")

    print("SOURCES")

    print("==========================")

    for source in result["sources"]:

        print(source)