import os
import glob
import pickle
import json
from typing import List, Tuple

import numpy as np
import faiss
from openai import OpenAI
from pypdf import PdfReader

def read_text_files(docs_path: str) -> List[Tuple[str, str]]:
    items = []
    for fp in glob.glob(os.path.join(docs_path, "**/*"), recursive=True):
        if os.path.isdir(fp):
            continue
        if fp.lower().endswith((".txt", ".md")):
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                items.append((fp, f.read()))
        elif fp.lower().endswith(".pdf"):
            reader = PdfReader(fp)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            items.append((fp, text))
    return items

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

def get_embeddings(texts: List[str], openai_api_key: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embeddings from OpenAI API for a batch of texts."""
    client = OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    # Extract embeddings and convert to numpy array
    embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
    return embeddings

def ingest(
    docs_path: str,
    persist_dir: str,
    collection_name: str,
    openai_api_key: str,
    chunk_size: int,
    overlap: int,
):
    """Ingest documents into FAISS vector store."""
    os.makedirs(persist_dir, exist_ok=True)
    
    docs = read_text_files(docs_path)
    
    chunk_texts = []
    metadatas = []
    counter = 0
    
    # Chunk all documents
    for source_path, text in docs:
        for c in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunk_id = f"chunk-{counter}"
            chunk_texts.append(c)
            metadatas.append({
                "source": os.path.basename(source_path),
                "chunk_id": chunk_id,
                "text": c
            })
            counter += 1
    
    if not chunk_texts:
        return {"chunks_added": 0, "collection": collection_name}
    
    # Get embeddings from OpenAI
    embeddings = get_embeddings(chunk_texts, openai_api_key)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance 
    index.add(embeddings)
    
    # Save index and metadata
    index_path = os.path.join(persist_dir, f"{collection_name}_index.faiss")
    metadata_path = os.path.join(persist_dir, f"{collection_name}_metadata.json")
    
    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump(metadatas, f, indent=2)
    
    return {"chunks_added": len(metadatas), "collection": collection_name}

