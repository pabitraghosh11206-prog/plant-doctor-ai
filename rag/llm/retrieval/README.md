# AgriVision AI - Retrieval Module (Member 2)

This module provides the search/retrieval functionality for the RAG (Retrieval-Augmented Generation) system. It finds relevant knowledge chunks from the plant disease database based on user queries.

---

## 📁 Files Created by Member 2

```
retrieval/
├── __init__.py           # Package marker, exports search_knowledge()
├── search.py             # Core retrieval function
└── test_retrieval.py     # Test suite (12 test queries)
```

---

## 🔧 Setup Requirements

Member 3 does **NOT** need to do anything here. The following are already set up by Member 1 & 2:

- ✅ Embedding model (`rag/search/embedder.py`) - `all-MiniLM-L6-v2` (384-dim vectors)
- ✅ ChromaDB vector store (`vector_db/` folder) - contains 10 ingested chunks
- ✅ Data ingestion (via `rag/search/ingest.py`) - processed dataset from Member 1

---

## 🚀 How to Use the Retrieval Function

### Basic Usage

```python
from retrieval.search import search_knowledge

# Simple text search
results = search_knowledge(query="What causes early blight on tomatoes?")

# With crop filter
results = search_knowledge(
    query="Symptoms of leaf spots",
    crop="tomato"
)

# With disease filter
results = search_knowledge(
    query="How to prevent disease",
    crop="tomato",
    disease="Tomato___Early_blight"
)

# Top-k parameter
results = search_knowledge(query="early blight", top_k=5)
```

### Parameters Explained

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str (required) | - | The natural language question or statement |
| `top_k` | int | 5 | How many results to return |
| `crop` | str | None | Filter by crop type (e.g., "tomato", "corn", "bean") |
| `disease` | str | None | Filter by disease label (e.g., "Tomato___Early_blight") |

---

## 📥 What You Get Back

The function returns a dictionary with these fields:

```python
{
    "results": [          # List of retrieved chunks
        {
            "chunk_id": "tomato_early_blight_001_p0001_c001",  # Unique ID
            "text": "Early blight on tomato often starts...",     # Full text content
            "metadata": {                                       # All stored metadata
                "source": "tomato_early_blight.pdf",
                "document_id": "tomato_early_blight_001",
                "page": 1,
                "crop": "tomato",
                "disease": "Tomato___Early_blight",
                "organization": "University of Minnesota Extension",
                # ... and more fields
            },
            "distance": 0.4645  # Similarity score (0-1, higher = better match)
        },
        # ... more results
    ],
    "count": 5,           # Number of results returned
    "query": "early blight"  # The query string used
}
```

---

## 🔗 Passing Results to Member 3 (LLM Context)

This is the key part for your RAG system! Here's how to format retrieval results for an LLM:

### Example: Building Prompt Context for LLM

```python
from retrieval.search import search_knowledge

# 1. Retrieve relevant knowledge chunks
retrieved = search_knowledge(
    query="What are the symptoms of early blight on tomato plants?",
    crop="tomato",
    top_k=3  # Get top 3 most relevant chunks
)

# 2. Build context string for LLM
context_parts = []

for chunk in retrieved.get("results", []):
    context_parts.append(f"""
### Knowledge Chunk
Source: {chunk['metadata'].get('source', 'Unknown')}
Disease: {chunk['metadata'].get('disease', 'N/A')}
Page: {chunk['metadata'].get('page', 'N/A')}

{chunk['text']}
""")

# Join all chunks with separators
final_context = "\n---\n".join(context_parts)

# 3. Pass to your LLM (Member 3's work)
llm_response = call_llm(
    system_prompt="You are an agricultural expert helping farmers...",
    user_message=f"""Context:
{final_context}

Question: What are the symptoms of early blight on tomato plants?
""",
    max_tokens=500
)
```

### Simple Example Pattern for Member 3

```python
# Inside your LLM function or main logic
from retrieval.search import search_knowledge

def get_retrieval_context(query, crop=None, disease=None, k=3):
    """Get relevant knowledge chunks for a given query."""
    retrieved = search_knowledge(
        query=query,
        crop=crop,
        disease=disease,
        top_k=k
    )
    
    if not retrieved.get("results"):
        return "No relevant information found."
    
    # Format chunks into prompt context
    chunks_text = []
    for chunk in retrieved["results"]:
        meta = chunk["metadata"]
        chunk_text = f"""DOCUMENT: {meta.get('source', 'Unknown')}
CLASSIFICATION: {meta.get('disease', 'N/A')} ({meta.get('crop', 'Unknown')})

{chunk['text']}""".strip()
        chunks_text.append(chunk_text)
    
    return "\n\n---\n".join(chunks_text)


# Usage in your LLM call
context = get_retrieval_context(
    query="User's question here",
    crop="tomato"  # optional
)

# Now pass context + user question to Member 3's LLM!
llm_prompt = f"""
You are an agricultural assistant. Use the following context to answer questions.

<context>
{context}
</context>

Question: User's actual question here?
"""
```

---

## 🧪 Testing the Retrieval Module

Run the test suite to verify everything works:

```bash
cd /Users/pabitraghosh/plant-doctor-ai
python3 retrieval/test_retrieval.py
```

This tests 12 different queries covering:
- Early blight symptoms
- Late blight symptoms  
- Disease prevention
- Environmental factors (rainfall, humidity)
- Crop-specific searches with filters

---

## 📊 Expected Results

For the query `"What are the symptoms of early blight on tomato plants?"`:

```python
{
    "results": [
        {
            "chunk_id": "tomato_early_blight_001_p0001_c001",
            "text": "Early blight on tomato often starts on older, lower leaves as small, round, brown spots...",
            "metadata": {
                "source": "tomato_early_blight.pdf",
                "disease": "Tomato___Early_blight",
                "crop": "tomato",
                # ... more metadata
            },
            "distance": 0.4645  # High = very relevant!
        },
        # ... up to top_k results
    ],
    "count": 3,
    "query": "What are the symptoms of early blight on tomato plants?"
}
```

---

## ⚠️ Important Notes

### DO NOT MODIFY:
- `rag/search/embedder.py` - Already set up by Member 1/2
- `rag/search/vector_store.py` - Already set up by Member 1/2
- `retrieval/search.py` - This function, only add to it if needed

### YOU SHOULD USE:
```python
from retrieval.search import search_knowledge
```

### Distance Scores Explained:
- `0.5 - 1.0`: Very relevant (high semantic similarity)
- `0.2 - 0.5`: Moderately relevant
- `0.0 - 0.2`: Weakly related or unrelated

---

## ✅ Checklist for Member 3 Integration

### Prerequisites (Check these are already done):
- [x] Embedding model installed and working (`SentenceTransformer`)
- [x] ChromaDB vector store with 10 chunks ingested
- [x] `retrieval/search.py` function exists and works

### Member 3 Tasks:
- [ ] Import the retrieval function in your codebase
- [ ] Create an LLM context builder using retrieval results
- [ ] Format retrieved chunks into a coherent prompt
- [ ] Build the full RAG pipeline (retrieval + generation)
- [ ] Test with sample user queries

---

## 🚀 Quick Start for Member 3

Copy-paste this to start immediately:

```python
# Add this import at the top of your file
from retrieval.search import search_knowledge

def build_rag_context(query, crop=None, disease=None, k=5):
    """Build context for LLM using retrieval."""
    results = search_knowledge(query=query, crop=crop, disease=disease, top_k=k)
    
    if not results.get("results"):
        return "No relevant information found."
    
    # Build formatted context string
    context_lines = []
    for chunk in results["results"]:
        meta = chunk["metadata"]
        context_lines.append(f"SOURCE: {meta.get('source', 'Unknown')}\n")
        context_lines.append(f"DISEASE: {meta.get('disease', 'N/A')}\n")
        context_lines.append(meta["text"])
        context_lines.append("\n---\n")
    
    return "\n".join(context_lines)

# Example usage in your LLM call function
user_query = "What causes yellow leaves on tomato plants?"
context = build_rag_context(user_query, crop="tomato")

# Now send 'context' + 'user_query' to Member 3's LLM!
```

---

## 📞 Handoff Summary

| Module | Status | Owner | Description |
|--------|--------|-------|-------------|
| Data Ingestion (chunks.jsonl) | ✅ Done | Member 1 | PDF processing |
| Embedding Model | ✅ Done | Member 2 | SentenceTransformer |
| Vector Database | ✅ Done | Member 2 | ChromaDB with 10 vectors |
| Search Function | ✅ Done | Member 2 | `search_knowledge()` ready |
| **LLM Generation** | ❌ TODO | Member 3 | Build context + generate answers |

---

## 🎯 Next Steps for Member 3

1. Review `retrieval/search.py` to understand the function signature
2. Decide on your LLM context format (see examples above)
3. Write your `generate_answer()` function that takes retrieval results
4. Test with a few sample queries end-to-end

**All retrieval logic is ready for you!**
