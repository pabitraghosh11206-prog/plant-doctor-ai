"""
AgriVision AI - Embedding Module

Generates embeddings for text chunks and queries using Sentence Transformers.
No external vector database dependencies - pure embedding logic only.
"""

import json
from pathlib import Path


class Embedder:
    """
    Simple wrapper around Sentence Transformers model.
    
    Converts text to dense vectors (embeddings) that represent semantic meaning.
    Used for:
    - Storing chunk embeddings in vector database
    - Converting user queries to comparable vectors
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        
        # Normalize embeddings for better similarity comparison
        # Note: Some versions use encode() with normalize_embeddings=True directly
        self._normalize = False
        
        try:
            # Newer API: check if the method exists
            self.model.set_normalize_embeddings(True)
            self._normalize = True
        except AttributeError:
            # Older API or different version - we handle normalization in encode()
            pass

    def _normalize(self, embeddings):
        """Normalize embeddings to unit length (L2 normalization)."""
        import numpy as np
        if isinstance(embeddings, np.ndarray):
            return embeddings / (np.linalg.norm(embeddings) + 1e-8)[None, :]
        return np.array(embeddings) / (np.linalg.norm(np.array(embeddings)) + 1e-8)

    def embed_query(self, query: str) -> list[float]:
        """
        Convert a single query text into an embedding vector.
        
        Args:
            query: The text query string
            
        Returns:
            List of floats representing the embedding (e.g., [0.23, -0.45, ...])
        """
        import numpy as np
        
        # Try direct normalization in encode() first (newer API)
        try:
            embeddings = self.model.encode(query, normalize_embeddings=True)
            return embeddings.tolist()
        except TypeError:
            pass
        
        # Fallback: encode then normalize manually
        embeddings = self.model.encode(query)
        embeddings = self._normalize(embeddings)
        return embeddings.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of text chunks into embeddings.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors, one per input text
        """
        import numpy as np
        
        # Try direct normalization in encode() first (newer API)
        try:
            embeddings = self.model.encode(
                texts, 
                normalize_embeddings=True, 
                batch_size=32  # Process in batches for memory efficiency
            )
            return embeddings.tolist()
        except TypeError:
            pass
        
        # Fallback: encode then normalize manually
        embeddings = self.model.encode(texts, batch_size=32)
        
        results = []
        for emb in embeddings:
            normalized_emb = self._normalize(emb)
            results.append(normalized_emb.tolist())
        
        return results


def load_chunks(chunks_file: str = "rag/data/chunks.jsonl") -> list[dict]:
    """
    Load text chunks from Member 1's JSONL output.
    
    Args:
        chunks_file: Path to the chunks.jsonl file created by Member 1
        
    Returns:
        List of chunk dictionaries with 'text', 'chunk_id', and metadata
    """
    chunks = []
    data_dir = Path(chunks_file)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_file}")
    
    with open(data_dir, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    
    return chunks


def save_embeddings(
    embeddings: list[list[float]],
    texts: list[str],
    metadata: list[dict],
    output_dir: str = "rag/data/embeddings"
):
    """
    Save chunk IDs and their corresponding embeddings to JSON files.
    
    Each chunk gets its own file with the structure:
    {
      "chunk_id": "...",
      "embedding": [0.12, -0.45, ...],  # normalized vector
      "text": "...",  # original text
      "metadata": {...}  # all other fields from chunks.jsonl
    }
    
    Args:
        embeddings: List of embedding vectors (from embed_texts)
        texts: Original text strings (order must match embeddings)
        metadata: All chunk metadata including chunk_id, document_id, etc.
        output_dir: Directory to save the JSON files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i in range(len(embeddings)):
        chunk_data = {
            "chunk_id": metadata[i]["chunk_id"],
            "embedding": embeddings[i],
            "text": texts[i],
            "metadata": {
                k: v for k, v in metadata[i].items() 
                if k not in ["chunk_id", "text"]
            }
        }
        
        output_file = Path(output_dir) / f"{metadata[i]['chunk_id']}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=2)


if __name__ == "__main__":
    # Simple test when running this file directly
    from pathlib import Path
    
    print("Testing embedding generation...")
    
    # Load chunks
    chunks = load_chunks()
    print(f"✓ Loaded {len(chunks)} text chunks")
    
    # Create embedder and generate embeddings
    embedder = Embedder()
    
    # Test with first chunk's text
    sample_text = chunks[0]["text"]
    query_embedding = embedder.embed_query(sample_text)
    
    print(f"✓ Query embedding dimension: {len(query_embedding)}")
    print(f"✓ First 5 values: {query_embedding[:5]}")
    
    # Verify expected dimension (all-MiniLM-L6-v2 outputs 384-dimensional vectors)
    assert len(query_embedding) == 384, "Expected 384-dimensional embeddings"
    
    print("✓ Test passed! Embedding model is working correctly.")
    
    # Generate all chunk embeddings
    texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embedder.embed_texts(texts)
    
    print(f"✓ Generated {len(chunk_embeddings)} chunk embeddings")
    
    # Save to files
    save_embeddings(
        embeddings=chunk_embeddings,
        texts=texts,
        metadata=chunks,
        output_dir="rag/data/embeddings"
    )
    
    print("✓ Embeddings saved to rag/data/embeddings/")
    print("\n✅ All tests passed!")
