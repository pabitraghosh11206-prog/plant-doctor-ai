"""
AgriVision AI - Retrieval Module

Provides a unified interface for searching the knowledge base.
Uses semantic similarity (embeddings) plus metadata filters (crop, disease).
"""

import sys
from pathlib import Path


def search_knowledge(
    query: str,
    top_k: int = 5,
    disease: str = None,
    crop: str = None
) -> list[dict]:
    """
    Search the AgriVision knowledge base for relevant text chunks.

    Combines semantic search (using embeddings) with optional filters
    for crop type and disease. Returns the most relevant results.

    Args:
        query (str): The natural language search query.
        top_k (int): Number of results to return (default: 5).
        disease (str, optional): Filter by disease label (e.g., "Tomato___Early_blight").
        crop (str, optional): Filter by crop type (e.g., "tomato", "corn").

    Returns:
        list[dict]: Top-k results, each containing:
            - chunk_id: Unique identifier for the chunk
            - text: The actual text content
            - metadata: All other fields (document_id, source, page, etc.)
            - distance: Cosine similarity score (0-1, higher = more similar)

    Example:
        >>> results = search_knowledge("What are early blight symptoms?")
        >>> for r in results:
        ...     print(f"Distance: {r['distance']:.4f}, Text: {r['text'][:50]}...")

    Raises:
        ValueError: If query is empty or None.
    """
    
    # Validate input
    if not query or not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")
    
    query = query.strip()
    
    try:
        # Step 1: Import and use embedder
        from rag.search.embedder import Embedder
        embedder = Embedder()
        
        # Step 2: Convert query to embedding vector
        query_embedding = embedder.embed_query(query)
        
        # Step 3: Search in ChromaDB
        from rag.search.vector_store import VectorStore
        vector_store = VectorStore(collection_name="agrivision_knowledge")
        
        # Step 4: Get search results with optional filters
        results = vector_store.search_with_filters(
            query_embedding=query_embedding,
            crop=crop.lower() if crop else None,
            disease=disease,
            limit=top_k
        )
        
        # Step 5: Format results into the requested structure
        formatted_results = []
        
        for i in range(len(results["metadata"])):
            chunk_id = results["metadata"][i].get("chunk_id", "")
            text = results["metadata"][i].get("text", "")
            metadata = {
                k: v for k, v in results["metadata"][i].items() 
                if k not in ["chunk_id", "text"]
            }
            
            # ChromaDB returns cosine similarity, convert to distance (1 - similarity)
            # But keep it normalized so higher = more similar
            sim_score = 1.0 - results["distances"][i] if results["distances"] else 0.0
            
            formatted_results.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
                "distance": round(sim_score, 4)
            })
        
        # Step 6: Handle empty results gracefully
        if not formatted_results:
            return {
                "results": [],
                "message": f"No results found for '{query}'. Try adjusting filters or using different keywords.",
                "count": 0
            }
        
        # Return sorted by distance (highest first)
        formatted_results.sort(key=lambda x: x["distance"], reverse=True)
        
        return {
            "results": formatted_results,
            "count": len(formatted_results),
            "query": query
        }
    
    except FileNotFoundError as e:
        # Handle case where chunks.jsonl doesn't exist (no ingestion has been done)
        return {
            "results": [],
            "message": f"Knowledge base not yet populated. Please run data ingestion first.",
            "error": str(e),
            "count": 0
        }
    
    except ValueError as e:
        return {
            "results": [],
            "message": f"Invalid query or parameters: {e}",
            "error": str(e),
            "count": 0
        }
    
    except Exception as e:
        # Catch any unexpected errors
        import traceback
        error_traceback = traceback.format_exc()
        
        return {
            "results": [],
            "message": f"Search failed: {e}",
            "error": error_traceback,
            "count": 0
        }


# ============================================
# Simple Usage Examples (for reference)
# ============================================

if __name__ == "__main__":
    import json
    
    print("=" * 70)
    print("AgriVision AI - Knowledge Retrieval Module")
    print("=" * 70)
    
    try:
        # Example 1: Simple text search
        print("\n🔍 [Test 1] Basic semantic search...")
        query = "What are the symptoms of early blight on tomato plants?"
        results = search_knowledge(query=query, top_k=3)
        
        if results.get("results"):
            for i, r in enumerate(results["results"][:3], 1):
                print(f"\n  Result {i} (distance: {r['distance']:.4f}):")
                print(f"    Chunk ID: {r['chunk_id']}")
                print(f"    Text: {r['text'][:150]}...")
        else:
            print(f"⚠ No results: {results.get('message')}")
        
        # Example 2: Search with crop filter
        print("\n🔍 [Test 2] With crop filter (corn)...")
        query = "What are the symptoms of common rust on corn?"
        results = search_knowledge(query=query, top_k=3, crop="corn")
        
        if results.get("results"):
            for i, r in enumerate(results["results"][:3], 1):
                print(f"\n  Result {i} (distance: {r['distance']:.4f}):")
                print(f"    Chunk ID: {r['chunk_id']}")
                print(f"    Text: {r['text'][:150]}...")
        else:
            print(f"⚠ No results: {results.get('message')}")
        
        # Example 3: Search with disease filter
        print("\n🔍 [Test 3] With disease filter (Tomato___Late_blight)...")
        query = "How does late blight affect tomato fruit?"
        results = search_knowledge(query=query, top_k=3, disease="Tomato___Late_blight")
        
        if results.get("results"):
            for i, r in enumerate(results["results"][:3], 1):
                print(f"\n  Result {i} (distance: {r['distance']:.4f}):")
                print(f"    Chunk ID: {r['chunk_id']}")
                print(f"    Text: {r['text'][:150]}...")
        else:
            print(f"⚠ No results: {results.get('message')}")
        
        # Example 4: Test with empty query (error handling)
        print("\n🔍 [Test 4] Empty query (should fail gracefully)...")
        results = search_knowledge(query="")
        if not results.get("results"):
            print(f"✓ Handled correctly: {results.get('message')}")
        
        print("\n" + "=" * 70)
        print("✅ All retrieval tests completed!")
        print("=" * 70)

    except Exception as e:
        import traceback
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        sys.exit(1)
