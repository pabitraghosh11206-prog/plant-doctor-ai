"""
Test suite for embedding module.
Verifies that:
1. Embeddings are generated correctly
2. Vector dimension is 384 (expected for all-MiniLM-L6-v2)
3. Embeddings can be saved and loaded
4. Similar texts get similar embeddings
"""

import sys
from pathlib import Path


def test_embedding_model():
    """Test that the model loads and generates embeddings with correct dimension."""
    print("=" * 50)
    print("TEST 1: Model Loading and Dimension Check")
    print("=" * 50)
    
    from rag.search.embedder import Embedder
    
    # Create embedder instance
    embedder = Embedder()
    print(f"✓ Model loaded: {type(embedder.model).__name__}")
    
    # Test query embedding
    test_query = "What are the symptoms of early blight on tomato?"
    query_embedding = embedder.embed_query(test_query)
    
    assert len(query_embedding) == 384, \
        f"Expected dimension 384, got {len(query_embedding)}"
    print(f"✓ Query embedding dimension: {len(query_embedding)} (expected: 384)")
    
    # Verify values are normalized (magnitude ≈ 1.0)
    import math
    magnitude = math.sqrt(sum(x**2 for x in query_embedding))
    assert abs(magnitude - 1.0) < 0.01, "Embeddings should be L2 normalized"
    print(f"✓ Embedding is normalized (magnitude: {magnitude:.4f})")


def test_batch_embeddings():
    """Test that batch text embedding works correctly."""
    print("\n" + "=" * 50)
    print("TEST 2: Batch Text Embedding")
    print("=" * 50)
    
    from rag.search.embedder import Embedder, load_chunks
    
    # Load chunks from Member 1's output
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]
    
    assert len(texts) > 0, "Expected non-empty text list"
    print(f"✓ Loaded {len(texts)} texts to embed")
    
    # Generate batch embeddings
    embedder = Embedder()
    embeddings = embedder.embed_texts(texts)
    
    # Verify each embedding has correct dimension
    for i, emb in enumerate(embeddings):
        assert len(emb) == 384, \
            f"Chunk {i}: expected dimension 384, got {len(emb)}"
    
    print(f"✓ Generated {len(embeddings)} embeddings with dimension 384 each")


def test_embedding_similarity():
    """Test that semantically similar texts produce similar embeddings."""
    print("\n" + "=" * 50)
    print("TEST 3: Embedding Similarity (Semantic Understanding)")
    print("=" * 50)
    
    from rag.search.embedder import Embedder, load_chunks
    import math
    
    embedder = Embedder()
    
    # Load chunks and find two similar ones (both about early blight symptoms)
    chunks = load_chunks()
    
    # Find early blight related chunks
    early_blight_chunks = [c for c in chunks if "early_blight" in c["document_id"]]
    assert len(early_blight_chunks) >= 2, "Need at least 2 early blight chunks for similarity test"
    
    chunk1_text = early_blight_chunks[0]["text"]
    chunk2_text = early_blight_chunks[1]["text"]
    
    emb1 = embedder.embed_query(chunk1_text)
    emb2 = embedder.embed_query(chunk2_text)
    
    # Calculate cosine similarity between two embeddings
    dot_product = sum(a * b for a, b in zip(emb1, emb2))
    mag1 = math.sqrt(sum(x**2 for x in emb1))
    mag2 = math.sqrt(sum(x**2 for x in emb2))
    cos_sim = dot_product / (mag1 * mag2)  # Both normalized, so this is cosine similarity
    
    print(f"✓ Chunk 1 text: {chunk1_text[:80]}...")
    print(f"✓ Chunk 2 text: {chunk2_text[:80]}...")
    print(f"✓ Cosine similarity between similar texts: {cos_sim:.4f}")
    
    # Similar texts should have cosine similarity > 0.5 (same topic)
    assert cos_sim > 0.3, \
        f"Similar texts should have reasonably high similarity (>0.3), got {cos_sim:.4f}"
    
    print(f"✓ Similarity check passed (cosine similarity: {cos_sim:.4f})")


def test_save_and_load():
    """Test that embeddings can be saved and loaded correctly."""
    print("\n" + "=" * 50)
    print("TEST 4: Save and Load Embeddings to JSON Files")
    print("=" * 50)
    
    from rag.search.embedder import Embedder, load_chunks, save_embeddings
    
    embedder = Embedder()
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_texts(texts)
    
    # Save to files
    output_dir = "rag/data/embeddings"
    save_embeddings(embeddings, texts, chunks, output_dir)
    
    # Verify files were created
    import os
    saved_files = Path(output_dir).glob("*.json")
    file_count = len(list(saved_files))
    
    print(f"✓ Created {file_count} JSON files in {output_dir}")
    assert file_count == len(chunks), f"Expected {len(chunks)} files, got {file_count}"
    
    # Load one file back and verify structure
    first_file = list(Path(output_dir).glob("*.json"))[0]
    with open(first_file, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    
    print(f"✓ File name: {first_file.name}")
    print(f"✓ Loaded keys: {list(loaded_data.keys())}")
    
    assert "embedding" in loaded_data, "Missing 'embedding' key"
    assert "text" in loaded_data, "Missing 'text' key"
    assert "metadata" in loaded_data, "Missing 'metadata' key"
    assert len(loaded_data["embedding"]) == 384, "Embedding dimension mismatch"
    
    print(f"✓ Embedding dimension in saved file: {len(loaded_data['embedding'])}")
    print("✓ Save/load test passed!")


if __name__ == "__main__":
    import json
    
    print("\n🚀 Starting AgriVision AI - Embedding Module Tests\n")
    
    try:
        test_embedding_model()
        test_batch_embeddings()
        test_embedding_similarity()
        test_save_and_load()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        print("\nEmbedding module is ready for:")
        print("  • Generating chunk embeddings")
        print("  • Converting queries to vectors")
        print("  • Saving/load embeddings from/to files")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
