import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os

EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-small-en")

# Global path variables that can be overridden externally
_INDEX_PATH = None
_DOCS_PATH = None
_EMBEDDINGS_PATH = None
_CHUNKS_PATH = None

def set_paths(index_path: str, docs_path: str):
    """Set paths for FAISS index and documents"""
    global _INDEX_PATH, _DOCS_PATH, _EMBEDDINGS_PATH, _CHUNKS_PATH
    if index_path:
        _INDEX_PATH = index_path
    if docs_path:
        _DOCS_PATH = docs_path
    if _EMBEDDINGS_PATH is None:
        _EMBEDDINGS_PATH = os.path.dirname(index_path)
    if _CHUNKS_PATH is None:
        _CHUNKS_PATH = os.path.join(os.path.dirname(index_path), "chunks")

def load_index_and_docs():
    """Load FAISS index and documents"""
    if not os.path.exists(_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index file does not exist: {_INDEX_PATH}")
    
    # Load FAISS index
    index = faiss.read_index(_INDEX_PATH)
    
    # Prioritize using chunks file (JSONL format)
    if _CHUNKS_PATH and os.path.exists(_CHUNKS_PATH):
        documents = []
        try:
            with open(_CHUNKS_PATH, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            # Try various possible text field names
                            text = data.get('text') or data.get('content') or data.get('passage') or str(data)
                            documents.append(text)
                        except json.JSONDecodeError:
                            print(f"⚠️ JSON parsing failed at line {line_num}, skipping")
                            continue
            
            print(f"✅ Loaded {len(documents)} documents from chunks file")
            return index, documents
        
        except Exception as e:
            print(f"❌ Failed to load chunks file: {e}")
            raise
    
    # If no chunks file exists, try using traditional JSON format
    elif _DOCS_PATH and os.path.exists(_DOCS_PATH):
        try:
            with open(_DOCS_PATH, "r", encoding="utf-8") as f:
                documents = json.load(f)
            
            print(f"✅ Loaded {len(documents)} documents from docs file")
            return index, documents
        
        except Exception as e:
            print(f"❌ Failed to load docs file: {e}")
            raise
    
    else:
        raise FileNotFoundError(f"No usable document files found. chunks: {_CHUNKS_PATH}, docs: {_DOCS_PATH}")

def retrieve_topk_docs(query: str, top_k: int = 5):
    """Retrieve top-k relevant documents"""
    try:
        index, documents = load_index_and_docs()
        
        # Generate query vector
        query_emb = EMBEDDING_MODEL.encode([query], normalize_embeddings=True).astype("float32")
        
        # Search
        D, I = index.search(query_emb, top_k)
        
        # Return retrieved documents
        retrieved_docs = []
        for i in I[0]:
            if 0 <= i < len(documents):
                retrieved_docs.append(documents[i])
            else:
                print(f"⚠️ Index {i} out of range (0-{len(documents)-1})")
        
        return retrieved_docs
    
    except Exception as e:
        print(f"❌ Document retrieval failed: {e}")
        return []

def get_current_config():
    """Get current configuration (for debugging)"""
    return {
        'index_path': _INDEX_PATH,
        'docs_path': _DOCS_PATH, 
        'embeddings_path': _EMBEDDINGS_PATH,
        'chunks_path': _CHUNKS_PATH
    } 