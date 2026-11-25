"""
AgriVision AI - Vector Database Module

Manages ChromaDB for storing and querying embeddings.
Includes:
- Persistent ChromaDB connection
- Adding vectors with documents and metadata
- Searching with optional crop/disease filters
- Clean result formatting
- Beginner-friendly debug output
"""

import os
import json
from pathlib import Path

import chromadb


class VectorStore:
    """
    ChromaDB wrapper for vector storage and retrieval.

    Features:
    - Persistent storage (saved to disk)
    - Single collection: "agrivision_knowledge"
    - Cosine similarity search
    - Optional metadata filters for crop and disease
    """

    def __init__(self, db_path="vector_db", collection_name="agrivision_knowledge"):
        """
        Initialize ChromaDB connection.

        Args:
            db_path: Folder where the vector database will be stored
            collection_name: Name of the ChromaDB collection
        """
        # Resolve database path relative to the project root
        project_root = Path(__file__).resolve().parent.parent.parent

        self.db_path = Path(db_path)
        if not self.db_path.is_absolute():
            self.db_path = project_root / self.db_path

        # Make sure the parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name

        # Create persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.db_path))

        # Get existing collection or create a new one
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print("  - Using existing collection")
        except Exception:
            self._create_collection(self.collection_name)

    def _create_collection(self, name):
        """Create a new ChromaDB collection."""
        self.collection = self.client.create_collection(name=name)
        print(f"  - Created new collection: '{name}'")

    def clear_collection(self):
        """
        Delete all vectors from the collection.

        WARNING: Permanent operation.
        """
        try:
            items = self.collection.get()

            if items and items.get("ids"):
                self.collection.delete(ids=items["ids"])
                print(f"✓ Cleared {len(items['ids'])} vectors from '{self.collection_name}'")
            else:
                print(f"⚠ No vectors to clear in '{self.collection_name}'")

        except Exception as e:
            print(f"⚠ Failed to clear collection: {e}")

    def get_vector_count(self):
        """Return number of stored vectors."""
        return self.collection.count()

    def add_vectors(
        self,
        embeddings: list[list[float]],
        chunk_ids: list[str],
        texts: list[str],
        metadata: list[dict] = None
    ):
        """
        Add embeddings, documents, and metadata to the collection.

        Args:
            embeddings: List of embedding vectors
            chunk_ids: List of unique IDs
            texts: Original text chunks
            metadata: List of metadata dictionaries
        """
        if not embeddings:
            raise ValueError("Embeddings cannot be empty")
        if not chunk_ids:
            raise ValueError("Chunk IDs cannot be empty")
        if not texts:
            raise ValueError("Texts cannot be empty")
        if metadata is None or not metadata:
            raise ValueError("Metadata cannot be empty")

        if not (len(embeddings) == len(chunk_ids) == len(texts) == len(metadata)):
            raise ValueError(
                "embeddings, chunk_ids, texts, and metadata must have the same length"
            )

        # Convert metadata values into ChromaDB-safe values
        compatible_metadata = []
        for m in metadata:
            cleaned = {}
            for k, v in m.items():
                if v is None:
                    cleaned[k] = ""
                elif isinstance(v, (list, dict)):
                    cleaned[k] = json.dumps(v)
                else:
                    cleaned[k] = str(v)
            compatible_metadata.append(cleaned)

        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=compatible_metadata
        )

        print(f"✓ Added {len(embeddings)} vectors to '{self.collection_name}'")

    def search(self, query_embedding: list[float], limit: int = 5) -> dict:
        """
        Search for similar vectors without metadata filters.

        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return

        Returns:
            Dictionary with documents, metadata, and distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "documents": results.get("documents", [[]])[0] if results.get("documents") else [],
            "metadata": results.get("metadatas", [[]])[0] if results.get("metadatas") else [],
            "distances": results.get("distances", [[]])[0] if results.get("distances") else []
        }

    def search_with_filters(
        self,
        query_embedding: list[float],
        crop: str = None,
        disease: str = None,
        limit: int = 5
    ) -> dict:
        """
        Search the vector database with optional crop and disease filters.

        This version builds a valid ChromaDB where clause:
        - no filters -> where=None
        - one filter -> where={"field": value}
        - two filters -> where={"$and": [...]}

        Args:
            query_embedding: Query embedding vector
            crop: Optional crop filter (e.g., "tomato")
            disease: Optional disease filter (e.g., "Tomato___Early_blight")
            limit: Number of results to return

        Returns:
            Dictionary with documents, metadata, and distances
        """
        # Build where clause safely
        conditions = []

        if crop:
            conditions.append({"crop": crop.lower()})

        if disease:
            conditions.append({"disease": disease})

        if len(conditions) == 0:
            where = None
        elif len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}

        # Debug output so you can verify what gets sent to ChromaDB
        print("\n========== WHERE CLAUSE ==========")
        print(where)
        print("==================================\n")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "documents": results.get("documents", [[]])[0] if results.get("documents") else [],
            "metadata": results.get("metadatas", [[]])[0] if results.get("metadatas") else [],
            "distances": results.get("distances", [[]])[0] if results.get("distances") else []
        }


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