import os
import json
import numpy as np
import faiss
from openai import OpenAI

def get_index_and_metadata(persist_dir: str, collection_name: str, openai_api_key: str):
    """Load FAISS index and metadata from disk."""
    index_path = os.path.join(persist_dir, f"{collection_name}_index.faiss")
    metadata_path = os.path.join(persist_dir, f"{collection_name}_metadata.json")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise ValueError(f"Index or metadata not found for collection '{collection_name}'. Run ingestion first.")
    
    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as f:
        metadatas = json.load(f)
    
    return index, metadatas, openai_api_key

def get_query_embedding(query: str, openai_api_key: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embedding for a query from OpenAI API."""
    client = OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(
        input=query,
        model=model
    )
    embedding = np.array([response.data[0].embedding], dtype=np.float32)
    return embedding

def retrieve(index, metadatas, query: str, openai_api_key: str, k: int = 4):
    """Retrieve top-k relevant chunks using FAISS similarity search."""
    # Get query embedding
    query_embedding = get_query_embedding(query, openai_api_key)
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, k)
    
    contexts = []
    for idx in indices[0]:
        if idx < len(metadatas):
            m = metadatas[idx]
            chunk_text = m.get('text', '')
            tag = f"[{m.get('source','unknown')}:{m.get('chunk_id','n/a')}]"
            contexts.append(f"{tag}\n{chunk_text}")
    
    return contexts

