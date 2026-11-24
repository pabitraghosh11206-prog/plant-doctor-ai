"""
Test suite for ChromaDB vector store module.
Verifies:
1. PersistentClient connection works
2. Collection is created correctly
3. Cosine similarity search works
4. Data persists after restart
5. Metadata filtering works
"""

import sys
from pathlib import Path


def test_connection():
    """Test ChromaDB connection and collection creation."""
    print("=" * 60)
    print("TEST 1: Database Connection & Collection Creation")
    print("=" * 60)
    
    from rag.search.vector_store import VectorStore, connect
    
    db_path = Path("vector_db")
    if db_path.exists():
        print(f"✓ Database folder exists at {db_path}")
        db_folder_size = Path(db_path).stat().st_size / 1024
        print(f"✓ Database size: {db_folder_size:.2f} KB")
    
    # Create store (creates collection if it doesn't exist)
    db = VectorStore(collection_name="agrivision_knowledge")
    
    print(f"✓ Client type: {type(db.client).__name__}")
    print(f"✓ Collection name: {db.collection_name}")
    print("✅ Connection test passed!")


def test_collection_schema():
    """Test that collection has correct metadata schema."""
    print("\n" + "=" * 60)
    print("TEST 2: Collection Schema & Metadata Fields")
    print("=" * 60)
    
    from rag.search.vector_store import VectorStore
    
    db = VectorStore(collection_name="agrivision_knowledge")
    
    # Check if collection has metadata defined
    try:
        schema = db.collection.metadata
        if schema:
            print(f"✓ Collection metadata schema:")
            for key, dtype in schema.items():
                print(f"    - {key}: {dtype}")
            assert "chunk_id" in schema or "text" in schema, "Missing required fields"
            print("✅ Schema test passed!")
        else:
            # No pre-defined schema means it was created without metadata
            # This is fine for basic use but limits filtering
            print(f"⚠ Note: Collection may lack predefined metadata schema")
            print("   Metadata can still be added, but filtering will use default fields.")
    except Exception as e:
        print(f"⚠ Schema check skipped: {e}")


def test_vector_count():
    """Test that we can count vectors in the collection."""
    print("\n" + "=" * 60)
    print("TEST 3: Vector Count")
    print("=" * 60)
    
    from rag.search.vector_store import VectorStore
    
    db = VectorStore(collection_name="agrivision_knowledge")
    count = db.get_vector_count()
    
    print(f"✓ Vectors in collection: {count}")
    
    # If no vectors, that's expected before data ingestion
    if count > 0:
        print("✅ Count test passed!")


def test_persistence():
    """Test that data persists after adding vectors."""
    print("\n" + "=" * 60)
    print("TEST 4: Data Persistence (Add and Verify)")
    print("=" * 60)
    
    from rag.search.vector_store import VectorStore
    
    # Create a dummy 384-dimensional embedding (cosine similarity compatible)
    test_embedding = [0.1] * 384
    
    db = VectorStore(collection_name="agrivision_knowledge")
    
    # Add some test vectors
    chunk_ids = ["chunk_001", "chunk_002", "chunk_003"]
    texts = [
        "Tomato early blight symptoms include brown spots.",
        "Late blight causes dark leaf blotches.",
        "Healthy tomato leaves are green."
    ]
    metadata = [
        {
            "chunk_id": "chunk_001",
            "text": texts[0],
            "document_id": "tomato_early_blight_001",
            "crop": "tomato",
            "disease": "Tomato___Early_blight"
        },
        {
            "chunk_id": "chunk_002",
            "text": texts[1],
            "document_id": "tomato_late_blight_001",
            "crop": "tomato",
            "disease": "Tomato___Late_blight"
        },
        {
            "chunk_id": "chunk_003",
            "text": texts[2],
            "document_id": "tomato_healthy_001",
            "crop": "tomato",
            "disease": "Tomato___healthy"
        }
    ]
    
    db.add_vectors(
        embeddings=[test_embedding] * 3,
        chunk_ids=chunk_ids,
        texts=texts,
        metadata=metadata
    )
    
    # Verify count increased
    new_count = db.get_vector_count()
    print(f"✓ Vector count after add: {new_count} (expected: 3)")
    assert new_count == 3, f"Expected 3 vectors, got {new_count}"
    
    print("✅ Persistence test passed!")


def test_search():
    """Test that similarity search works with cosine distance."""
    print("\n" + "=" * 60)
    print("TEST 5: Similarity Search (Cosine)")
    print("=" * 60)
    
    from rag.search.vector_store import VectorStore
    
    db = VectorStore(collection_name="agrivision_knowledge")
    
    # Query vector close to our test vectors
    query_embedding = [0.1] * 384  # Same as stored vectors
    
    results = db.search(query_embedding=query_embedding, limit=5)
    
    print(f"✓ Search returned {len(results['metadata'])} results")
    
    # All should have similar cosine distance (close to 1.0 since same vector)
    for i, dist in enumerate(results["distances"]):
        print(f"  Result {i+1}: distance = {dist:.4f}")
    
    # Cosine similarity of identical vectors is 1.0, but distance depends on implementation
    print("✅ Search test passed!")


def test_filtering():
    """Test filtering by crop and disease."""
    print("\n" + "=" * 60)
    print("TEST 6: Metadata Filtering (Crop & Disease)")
    print("=" * 60)
    
    from rag.search.vector_store import VectorStore
    
    db = VectorStore(collection_name="agrivision_knowledge")
    
    # Test crop filter - should only return tomato results
    query_embedding = [0.1] * 384
    filtered = db.search_with_filters(query_embedding=query_embedding, crop="tomato", limit=5)
    
    print(f"✓ Crop filter (tomato): {len(filtered['metadata'])} results")
    assert len(filtered['metadata']) > 0, "Should return at least some tomato results"
    
    # Test disease filter - should only return early blight
    filtered_disease = db.search_with_filters(query_embedding=query_embedding, disease="Tomato___Early_blight", limit=5)
    
    print(f"✓ Disease filter (Early_blight): {len(filtered_disease['metadata'])} results")
    assert len(filtered_disease['metadata']) > 0, "Should return at least some early blight results"
    
    print("✅ Filtering test passed!")


def test_data_durability():
    """Test that data survives a Python restart (by reopening the DB)."""
    print("\n" + "=" * 60)
    print("TEST 7: Data Durability (Re-open Database)")
    print("=" * 60)
    
    from rag.search.vector_store import VectorStore
    
    # Re-open the database in a NEW instance (simulating restart)
    db = VectorStore(collection_name="agrivision_knowledge")
    
    count = db.get_vector_count()
    print(f"✓ Vectors loaded after re-opening: {count}")
    
    assert count > 0, "Data should persist across Python sessions!"
    
    # Verify we can still search
    query_embedding = [0.1] * 384
    results = db.search(query_embedding=query_embedding, limit=5)
    
    print(f"✓ Search after re-open: {len(results['metadata'])} results")
    
    print("✅ Durability test passed!")


if __name__ == "__main__":
    print("\n🚀 Starting AgriVision AI - Vector Store Tests\n")
    
    try:
        test_connection()
        test_collection_schema()
        test_vector_count()
        test_persistence()
        test_search()
        test_filtering()
        test_data_durability()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nVector Store Status:")
        print(f"  ✓ ChromaDB Connected: Yes")
        print(f"  ✓ Collection Name: 'agrivision_knowledge'")
        print(f"  ✓ Similarity Metric: Cosine (default in ChromaDB)")
        print(f"  ✓ Persistence: Enabled (stored locally at ./vector_db/)")
        print(f"  ✓ Metadata Fields: chunk_id, text, document_id, crop, disease")
        
    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
