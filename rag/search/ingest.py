"""
AgriVision AI - Data Ingestion Module

Ingests text chunks from Member 1's JSONL output, generates embeddings,
and stores them in ChromaDB vector database.

Uses:
- rag.search.embedder: For generating text embeddings
- rag.search.vector_store: For storing/retrieving vectors

Features:
- Batch processing for large datasets
- Duplicate detection by chunk_id
- Preserves all useful metadata (crop, disease, source, page, etc.)
"""

import json
from pathlib import Path
import sys


class DataIngestion:
    """
    Ingests chunks from Member 1's output and stores them in vector DB.
    
    Workflow:
    1. Load chunks.jsonl from Member 1
    2. Generate embeddings for each chunk's text
    3. Check for existing chunks (by chunk_id) to avoid duplicates
    4. Add new vectors to ChromaDB
    5. Report progress and summary
    """

    def __init__(self, chunks_file="rag/data/chunks.jsonl", db_path="vector_db"):
        """Initialize ingestion with file paths."""
        from rag.search.vector_store import VectorStore
        self.chunks_file = Path(chunks_file)
        self.db = VectorStore(db_path=db_path)
        
        # Verify input file exists
        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_file}"
            )

    def clear_existing_data(self):
        """Clear existing vectors (for clean re-ingestion)."""
        from rag.search.vector_store import VectorStore
        
        count = self.db.get_vector_count()
        
        # Use a fresh collection instance to avoid caching issues
        temp_db = VectorStore(collection_name=self.db.collection_name)
        
        if count > 0:
            print(f"\n⚠ Clearing existing data ({count} vectors)...")
            # Recreate the entire collection - simplest and most reliable
            self.db.client.delete_collection(self.db.collection_name)
            self.db._create_collection(self.db.collection_name)
            print(f"✓ Cleared {count} old vectors")
        else:
            print("ℹ No existing data to clear")

    def load_chunks(self):
        """Load chunks from Member 1's JSONL output."""
        chunks = []
        
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                chunk = json.loads(line)
                chunks.append(chunk)
        
        total = len(chunks)
        print(f"✓ Loaded {total} chunks from {self.chunks_file}")
        
        # Verify required fields exist (check first chunk only)
        if chunks and all(field in chunks[0] for field in ["chunk_id", "text"]):
            pass  # OK - fields present
        else:
            raise ValueError("Chunks missing required fields: chunk_id, text")
        
        return chunks

    def generate_embeddings(self, texts):
        """Generate embeddings for a list of texts using the embedder."""
        from rag.search.embedder import Embedder
        
        embedder = Embedder()
        print(f"  → Generating {len(texts)} embeddings...")
        
        embeddings = embedder.embed_texts(texts)
        return embeddings

    def check_duplicates(self, chunk_ids):
        """Check which chunk_ids already exist in the vector store."""
        existing = set(self.db.collection.get()['ids'])
        duplicates = set(chunk_ids) & existing
        
        if duplicates:
            print(f"⚠ Found {len(duplicates)} duplicate chunks (will skip)")
        
        return duplicates

    def ingest(self, batch_size=50):
        """
        Ingest all chunks into the vector database.
        
        Args:
            batch_size: Number of embeddings to process at once
        """
        print("\n" + "=" * 60)
        print("AGRIVISION AI - Data Ingestion Process")
        print("=" * 60)
        
        # Clear any existing data (for clean ingestion)
        self.clear_existing_data()
        
        # Step 1: Load chunks
        print(f"\n[Step 1/4] Loading chunks...")
        chunks = self.load_chunks()
        
        # Step 2: Extract text and metadata
        print(f"\n[Step 2/4] Preparing data...")
        texts = [chunk["text"] for chunk in chunks]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        
        # Preserve useful metadata (everything including chunk_id)
        metadata_list = []
        for chunk in chunks:
            meta = dict(chunk)  # Copy all fields
            metadata_list.append(meta)
        
        print(f"  Metadata fields per chunk: {list(metadata_list[0].keys())}")
        
        # Step 3: Check for duplicates
        duplicates = self.check_duplicates(chunk_ids)
        total_chunks = len(chunks)
        
        if duplicates:
            unique_chunk_ids = [cid for cid in chunk_ids if cid not in duplicates]
            print(f"  → Inserting {len(unique_chunk_ids)} new chunks (skipping {len(duplicates)} duplicates)")
            
            # Process only unique chunks - use original metadata which includes chunk_id
            texts_to_embed = []
            ids_to_add = []
            metadatas_to_add = []
            for cid in unique_chunk_ids:
                idx = chunk_ids.index(cid)  # Find index in original list
                texts_to_embed.append(texts[idx])
                ids_to_add.append(cid)
                metadatas_to_add.append(metadata_list[idx])  # This now includes chunk_id
        else:
            print(f"  → All {total_chunks} chunks are new")
            texts_to_embed = texts
            ids_to_add = chunk_ids
            metadatas_to_add = metadata_list
        
        # Step 4: Generate embeddings (batched for memory efficiency)
        print("\n[Step 3/4] Generating embeddings...")
        
        # Process in batches
        num_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            batch_ids = ids_to_add[i:i + batch_size]
            
            print(f"    Processing batch {i//batch_size + 1}/{num_batches} ({len(batch_texts)} chunks)...")
            
            batch_embeddings = self.generate_embeddings(batch_texts)
            
            # Add to vector store
            metadata_batch = metadatas_to_add[i:i + batch_size]
            self.db.add_vectors(
                embeddings=batch_embeddings,
                chunk_ids=batch_ids,
                texts=batch_texts,
                metadata=metadata_batch
            )
        
        print(f"✓ Added {len(texts_to_embed)} chunks to vector database")

    def get_statistics(self):
        """Print summary statistics about the ingestion."""
        from rag.search.vector_store import VectorStore
        
        db = VectorStore(collection_name="agrivision_knowledge")
        
        count = db.get_vector_count()
        
        print("\n" + "=" * 60)
        print("Ingestion Summary")
        print("=" * 60)
        print(f"✓ Total chunks ingested: {count}")
        print(f"✓ Vector dimension: 384")
        print(f"✓ Similarity metric: Cosine (default in ChromaDB)")
        print(f"\nMetadata stored per chunk:")
        print(f"    - chunk_id: Unique identifier")
        print(f"    - text: Original content")
        print(f"    - document_id: Source document reference")
        print(f"    - crop: Crop type (for filtering)")
        print(f"    - disease: Disease label (for filtering)")
        print(f"    - source: Source filename")
        print(f"    - page: Page number")
        print(f"    - title: Document title")
        print(f"    - organization: Publishing organization")
        print(f"    - source_url: Link to original source")
        print(f"\nDatabase location: {db.db_path}")


if __name__ == "__main__":
    print("\n🚀 Starting AgriVision AI Data Ingestion\n")
    
    try:
        # Create ingestion instance and run
        ingest = DataIngestion(
            chunks_file="rag/data/chunks.jsonl",
            db_path="vector_db"
        )
        
        # Run the ingestion
        ingest.ingest(batch_size=50)
        
        # Show summary
        ingest.get_statistics()
        
        print("\n✅ Ingestion completed successfully!")
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("   Make sure Member 1's chunks.jsonl exists at the expected path.")
        sys.exit(1)
    
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("   Chunks file may have invalid format or missing fields.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
