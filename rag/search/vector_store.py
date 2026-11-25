"""
AgriVision AI - Vector Database Module

Manages ChromaDB for storing and querying embeddings.
No data ingestion yet - just connection and setup logic.
"""

import chromadb
import os
from pathlib import Path
``

class VectorStore:
    """
    ChromaDB wrapper for vector storage and retrieval.
    
    Features:
    - Persistent storage (saves to disk, survives restarts)
    - Single collection: "agrivision_knowledge"
    - Cosine similarity for similarity search
    """

    def __init__(self, db_path="vector_db", collection_name="agrivision_knowledge"):
        """
        Initialize ChromaDB connection.
        
        Args:
            db_path: Where to store the database (default: "vector_db" folder)
            collection_name: Name of the collection (default: "agrivision_knowledge")
        """
        # Use project root relative path for persistence
        self.db_path = Path(db_path) if not str(db_path).startswith("/") else db_path
        
        # If running from project root, make it relative to project
        project_root = Path(__file__).resolve().parent.parent.parent
        if not str(self.db_path).startswith("/"):
            self.db_path = project_root / self.db_path
        
        # Create the vector_db folder if it doesn't exist
        db_folder = Path(os.path.dirname(self.db_path))
        db_folder.mkdir(parents=True, exist_ok=True)
        
        # Create persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection_name = collection_name
        
        # Get or create the collection (always ensure it exists)
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"  - Using existing collection")
        except Exception:
            # Collection doesn't exist yet - create it with metadata fields
            self._create_collection(collection_name)
    
    def _create_collection(self, name):
        """Create a new collection (without pre-defined schema)."""
        # Create collection without metadata schema
        # Metadata will be provided when adding vectors
        self.collection = self.client.create_collection(name=name)
        print(f"  - Created new collection: '{name}'")
    
    def clear_collection(self):
        """Delete all vectors in the collection (WARNING: permanent!)."""
        count = self.collection.count()
        if count > 0:
            # Delete all documents - using empty string filter on any field
            # This is more reliable than getting IDs first
            self.collection.delete(where={"id": {"$eq": ""}})
            print(f"✓ Cleared {count} vectors from '{self.collection_name}'")
        else:
            print(f"⚠ No vectors to clear in '{self.collection_name}'")

    def get_vector_count(self):
        """Get the number of stored vectors."""
        return self.collection.count()

    def add_vectors(
        self,
        embeddings: list[list[float]],
        chunk_ids: list[str],
        texts: list[str],
        metadata: list[dict] = None
    ):
        """
        Add embeddings to the collection.
        
        Args:
            embeddings: List of embedding vectors [384, 128, ...]
            chunk_ids: List of unique IDs for each chunk
            texts: Original text strings (for display)
            metadata: Optional additional metadata to store
        """
        # Ensure collection exists
        self.client.get_or_create_collection(name=self.collection_name)
        
        if not metadata:
            raise ValueError("Metadata cannot be empty")
        
        # Convert all metadata values to strings to avoid type conversion issues
        compatible_metadata = []
        for m in metadata:
            compatible_m = {}
            for k, v in m.items():
                if v is None:
                    compatible_m[k] = "null"  # Null values as string
                elif isinstance(v, (list, dict)):
                    compatible_m[k] = json.dumps(v)  # Complex types as JSON string
                else:
                    compatible_m[k] = str(v)
            
            compatible_metadata.append(compatible_m)
        
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            metadatas=compatible_metadata
        )
        print(f"✓ Added {len(embeddings)} vectors to '{self.collection_name}'")

    def search(self, query_embedding: list[float], limit: int = 5) -> dict:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_embedding: The query vector to search against
            limit: Number of results to return (default: 5)
            
        Returns:
            Dictionary with 'documents' and 'distances' keys
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["embeddings", "metadatas", "distances"]
        )

        # Format results into a clean structure
        formatted = {
            "documents": [] if not results.get("embeddings") else [emb[0] for emb in results["embeddings"]],
            "metadata": [] if not results.get("metadatas") else list(results["metadatas"][0]),
            "distances": [] if not results.get("distances") else list(results["distances"][0])
        }

        return formatted

    def search_with_filters(
        self,
        query_embedding: list[float],
        crop: str = None,
        disease: str = None,
        limit: int = 5
    ) -> dict:
        """
        Search with optional metadata filters (crop, disease).
        
        Args:
            query_embedding: The query vector to search against
            crop: Filter by crop type (e.g., "tomato")
            disease: Filter by disease label (e.g., "Tomato___Early_blight")
            limit: Number of results to return
            
        Returns:
            Dictionary with filtered and ranked results
        """
        # Build filter condition
        where = {}
        
        if crop:
            where["crop"] = {"$eq": crop.lower()}
        
        if disease:
            where["disease"] = {"$eq": disease}
        
        # Handle empty string filters (exclude them)
        for key, val in where.items():
            if isinstance(val, dict) and "$eq" in val and val["$eq"] == "":
                del where[key]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where if where else None,
            include=["embeddings", "metadatas", "distances"]
        )
        
        return {
            "documents": results["embeddings"][0] if results.get("embeddings") else [],
            "metadata": [m for item in results["metadatas"] or [] for m in item] if results.get("metadatas") else [],
            "distances": results["distances"][0] if results.get("distances") else []
        }


# Convenience functions for easy import

def connect(db_path="vector_db", collection_name="agrivision_knowledge"):
    """Quick connection to the vector database."""
    store = VectorStore(db_path=db_path, collection_name=collection_name)
    print(f"✅ Connected to ChromaDB: {store.collection_name}")
    return store


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("AgriVision AI - Vector Database Connection Test")
    print("=" * 60)
    
    try:
        # Create vector store (will auto-create collection)
        db = connect(collection_name="agrivision_knowledge")
        
        # Verify connection
        print(f"\n✓ Database path: {db.db_path}")
        print(f"✓ Collection name: {db.collection_name}")
        print(f"✓ Vector count before add: {db.get_vector_count()}")
        
        # Test that it persists (add a dummy vector, then restart Python)
        test_embedding = [0.1] * 384  # Dummy 384-dim embedding
        db.add_vectors(
            embeddings=[test_embedding],
            chunk_ids=["test_chunk_id"],
            texts=["This is a test text."],
            metadata=[{
                "chunk_id": "test_chunk_id",
                "text": "This is a test text.",
                "document_id": "test_doc",
                "crop": "tomato",
                "disease": "Tomato___Early_blight"
            }]
        )
        
        print(f"✓ Vector count after add: {db.get_vector_count()}")
        
        # Test filtering
        query_emb = [0.5] * 384
        filtered = db.search_with_filters(query_embedding=query_emb, crop="tomato", limit=5)
        print(f"✓ Filtered search returned: {len(filtered['metadata'])} results")
        
        print("\n" + "=" * 60)
        print("✅ Vector Store Test PASSED!")
        print("=" * 60)
        print("\nDatabase will persist after Python restart:")
        print(f"  Location: {db.db_path}")
        print(f"  Collection: 'agrivision_knowledge'")
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
