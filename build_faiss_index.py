#!/usr/bin/env python3
"""
Universal FAISS Index Building Script
Supports building independent FAISS indices for different datasets
"""

import json
import numpy as np
import faiss
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from modules.data_loader import load_chunks_from_file
from typing import List

def load_chunks(chunks_file: str) -> List[str]:
    """Load document block data - using unified data loader"""
    chunks_data = load_chunks_from_file(chunks_file)
    chunks = [chunk.get('text', chunk.get('content', '')) for chunk in chunks_data]
    chunks = [chunk for chunk in chunks if chunk.strip()]  # Filter empty documents
    print(f"Loaded {len(chunks)} chunks")
    return chunks

def build_faiss_index(chunks: List[str], model_name='BAAI/bge-small-en', index_file='faiss_index.index', embeddings_file='embeddings.npy'):
    """Build FAISS index"""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Generating embeddings...")
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings).astype('float32')
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Use inner product similarity
    
    # Normalize embedding vectors (for cosine similarity)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save index and embedding vectors
    faiss.write_index(index, index_file)
    np.save(embeddings_file, embeddings)
    
    print(f"FAISS index saved to: {index_file}")
    print(f"Embeddings saved to: {embeddings_file}")
    print(f"Index contains {index.ntotal} vectors")

def main():
    parser = argparse.ArgumentParser(description='Build FAISS index for document chunks')
    parser.add_argument('--chunks_file', required=True, help='Path to chunks JSONL file')
    parser.add_argument('--index_file', default='faiss_index.index', help='Output FAISS index file')
    parser.add_argument('--embeddings_file', default='embeddings.npy', help='Output embeddings file')
    parser.add_argument('--model_name', default='BAAI/bge-small-en', help='Embedding model name')
    
    args = parser.parse_args()
    
    # Load document blocks
    chunks = load_chunks(args.chunks_file)
    
    # Build FAISS index
    build_faiss_index(
        chunks=chunks,
        model_name=args.model_name,
        index_file=args.index_file,
        embeddings_file=args.embeddings_file
    )
    
    print("✅ FAISS index building completed!")

if __name__ == '__main__':
    main() 