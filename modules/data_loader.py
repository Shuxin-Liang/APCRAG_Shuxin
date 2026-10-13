# -*- coding: utf-8 -*-
"""
Unified Data Loading Module
Supports three domains: finance, biometrics, legal
Fix data format and parsing issues
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Union
from modules.config import get_file_path, UNTARGETED_ATTACKS_FILE, validate_file_exists

def load_qa_data(domain: str) -> Union[List[Dict], Dict]:
    """Load QA dataset for specified domain, including questions and ground truth"""
    qa_file = get_file_path(domain, 'qa_data')
    
    if not validate_file_exists(qa_file):
        print(f"❌ QA data file not found: {qa_file}")
        return []
    
    try:
        with open(qa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded QA data for {domain} domain: {len(data) if isinstance(data, list) else 'unknown'} records")
        return data
    except Exception as e:
        print(f"❌ Failed to load QA data: {e}")
        return []

def load_chunks_data(domain: str) -> List[Dict]:
    """Load document block data for specified domain"""
    chunks_file = get_file_path(domain, 'chunks')
    
    if not validate_file_exists(chunks_file):
        print(f"❌ Document block file not found: {chunks_file}")
        return []
    
    try:
        chunks = []
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        chunk_data = json.loads(line)
                        chunks.append(chunk_data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON parsing failed at line {line_num}: {e}")
                        continue
        
        print(f"✅ Successfully loaded document blocks for {domain} domain: {len(chunks)} blocks")
        return chunks
    except Exception as e:
        print(f"❌ Failed to load document blocks: {e}")
        return []

def load_chunks_from_file(chunks_file: str) -> List[Dict]:
    """Load chunks data directly from file path, used by build_faiss_index.py and other standalone scripts"""
    if not validate_file_exists(chunks_file):
        print(f"❌ Chunks file not found: {chunks_file}")
        return []
    
    try:
        chunks = []
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        chunk_data = json.loads(line)
                        chunks.append(chunk_data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON parsing failed at line {line_num}: {e}")
                        continue
        
        print(f"✅ Successfully loaded chunks data: {len(chunks)} document blocks")
        return chunks
    except Exception as e:
        print(f"❌ Failed to load chunks data: {e}")
        return []

def load_privacy_prompts(domain: str) -> List[str]:
    """Load privacy prompt data for specified domain (targeted attacks)"""
    prompts_file = get_file_path(domain, 'prompts')
    
    if not validate_file_exists(prompts_file):
        print(f"❌ Privacy prompt file not found: {prompts_file}")
        return []
    
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure return value is a list of strings
        # If it's a string, try parsing as JSON
        # Use json.loads instead of eval() for safety
        # If it's already a list, use directly
        
        if isinstance(data, list):
            prompts = [item for item in data if isinstance(item, str)]
        elif isinstance(data, dict) and 'prompts' in data:
            prompts = data['prompts']
        else:
            print(f"⚠️ Unknown prompts data format, attempting conversion...")
            prompts = list(data.values()) if isinstance(data, dict) else []
        
        print(f"✅ Successfully loaded privacy prompts for {domain} domain: {len(prompts)} prompts")
        return prompts
    except Exception as e:
        print(f"❌ Failed to load privacy prompts: {e}")
        return []

def load_privacy_entities(domain: str) -> List[List[Dict]]:
    """Load privacy entity data for specified domain, fix format issues"""
    entities_file = get_file_path(domain, 'entities')
    
    if not validate_file_exists(entities_file):
        print(f"❌ Privacy entity file not found: {entities_file}")
        return []
    
    try:
        with open(entities_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_entities = []
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                try:
                    if isinstance(item, str):
                        # If it's a string, try parsing as JSON
                        # Use json.loads instead of eval() for safety
                        entity_list = json.loads(item.replace("'", '"'))
                    elif isinstance(item, list):
                        # If it's already a list, use directly
                        entity_list = item
                    else:
                        print(f"⚠️ Unknown format for entity entry {i+1}, skipping")
                        entity_list = []
                    
                    processed_entities.append(entity_list)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"⚠️ Failed to parse entity entry {i+1}: {e}")
                    processed_entities.append([])
        else:
            print(f"⚠️ Entity data format error, expected list but got: {type(data)}")
            return []
        
        print(f"✅ Successfully loaded privacy entities for {domain} domain: {len(processed_entities)} entries")
        return processed_entities
    except Exception as e:
        print(f"❌ Failed to load privacy entities: {e}")
        return []

def load_untargeted_attacks() -> List[str]:
    """Load untargeted attack queries"""
    if not validate_file_exists(UNTARGETED_ATTACKS_FILE):
        print(f"❌ Untargeted attack file not found: {UNTARGETED_ATTACKS_FILE}")
        return []
    
    try:
        with open(UNTARGETED_ATTACKS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        queries = []
        if isinstance(data, dict) and 'attacks' in data:
            for attack in data['attacks']:
                if isinstance(attack, dict) and 'query' in attack:
                    queries.append(attack['query'])
        elif isinstance(data, list):
            queries = [item.get('query', '') for item in data if isinstance(item, dict) and 'query' in item]
        
        print(f"✅ Successfully loaded untargeted attack queries: {len(queries)} queries")
        return queries
    except Exception as e:
        print(f"❌ Failed to load untargeted attacks: {e}")
        return []

def prepare_utility_evaluation_data(domain: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Prepare utility evaluation data for specified domain, using FAISS retrieval instead of repeated retrieval"""
    print(f"🔄 Preparing utility evaluation data for {domain} domain...")
    
    qa_data = load_qa_data(domain)
    
    references = {}
    contexts = {}
    
    # Extract questions and answers based on QA data structure
    if isinstance(qa_data, list):
        for item in qa_data:
            if isinstance(item, dict) and 'question' in item and 'response' in item:
                question = item['question']
                references[question] = item['response']
                # ⚠️ Important fix: Use FAISS retrieval directly to maintain consistency with runtime
                contexts[question] = get_contexts_from_faiss(question, domain=domain)
    elif isinstance(qa_data, dict):
        for key, value in qa_data.items():
            if isinstance(value, dict) and 'question' in value and 'response' in value:
                question = value['question']
                references[question] = value['response']
                contexts[question] = get_contexts_from_faiss(question, domain=domain)
    
    print(f"✅ Preparation complete: {len(references)} questions for utility evaluation")
    return references, contexts

def get_contexts_from_faiss(question: str, max_chunks: int = 3, domain: str = None) -> List[str]:
    """Use FAISS retrieval directly to get context, maintain consistency with runtime"""
    try:
        from modules.retriever import retrieve_topk_docs, set_index_paths
        from modules.config import get_file_path, get_domain_from_args
        
        # If no domain provided, try to get from global settings
        if not domain:
            # Here we need a global variable to track the current domain
            # Temporarily use finance domain as default
            domain = getattr(get_contexts_from_faiss, '_current_domain', 'finance')
        
        # Set FAISS path for current domain
        try:
            index_path = get_file_path(domain, 'faiss_index')
            chunks_path = get_file_path(domain, 'chunks')
            embeddings_path = get_file_path(domain, 'embeddings')
            
            set_index_paths(
                index_path=index_path,
                chunks_path=chunks_path,
                embeddings_path=embeddings_path
            )
            
            # Use the same FAISS retrieval as runtime
            retrieved_docs = retrieve_topk_docs(question, top_k=max_chunks)
            
            print(f"  🎯 Retrieved {len(retrieved_docs)} relevant documents using FAISS")
            return retrieved_docs
            
        except Exception as config_error:
            print(f"❌ Configuration error: {config_error}")
            raise
        
    except Exception as e:
        print(f"⚠️ FAISS retrieval failed, falling back to chunks loading: {e}")
        
        # Fallback plan: Load directly from chunks and use semantic similarity for selection
        try:
            # Use specified domain or default domain
            if not domain:
                domain = 'finance'
            
            chunks_file = get_file_path(domain, 'chunks')
            chunks_data = load_chunks_from_file(chunks_file)
            return get_relevant_contexts_from_chunks(question, chunks_data, max_chunks)
            
        except Exception as fallback_error:
            print(f"❌ Fallback plan also failed: {fallback_error}")
            return []

def set_current_domain(domain: str):
    """Set current evaluation domain (global setting)"""
    get_contexts_from_faiss._current_domain = domain

def get_relevant_contexts_from_chunks(question: str, chunks_data: List[Dict], max_chunks: int = 3) -> List[str]:
    """Select relevant documents from chunks data based on semantic similarity (as fallback plan)"""
    if not chunks_data:
        return []
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Use embedding model to calculate similarity
        model = SentenceTransformer("BAAI/bge-small-en")
        
        # Get question embedding
        question_embedding = model.encode([question])[0]
        
        # Extract text from all chunks and calculate embeddings
        chunk_texts = []
        for chunk in chunks_data:
            if isinstance(chunk, dict):
                text = chunk.get('text') or chunk.get('content') or chunk.get('passage') or str(chunk)
            else:
                text = str(chunk)
            chunk_texts.append(text)
        
        if not chunk_texts:
            return []
        
        # Calculate embeddings for all chunks
        chunk_embeddings = model.encode(chunk_texts)
        
        # Calculate similarity scores
        similarities = np.dot(chunk_embeddings, question_embedding)
        
        # Get indices of most similar chunks
        top_indices = np.argsort(similarities)[::-1][:max_chunks]
        
        # Return most relevant documents
        relevant_contexts = [chunk_texts[idx] for idx in top_indices]
        
        print(f"  🎯 Selected {len(relevant_contexts)} most relevant document blocks based on semantic similarity")
        return relevant_contexts
        
    except ImportError:
        print("⚠️ sentence_transformers not installed, using simplified version")
        # Fall back to simplified version
        selected_chunks = chunks_data[:max_chunks]
        contexts = []
        
        for chunk in selected_chunks:
            if isinstance(chunk, dict):
                text = chunk.get('text') or chunk.get('content') or chunk.get('passage') or str(chunk)
            else:
                text = str(chunk)
            contexts.append(text)
        
        return contexts
    except Exception as e:
        print(f"⚠️ Semantic similarity calculation failed, using simplified version: {e}")
        # Fall back to simplified version
        selected_chunks = chunks_data[:max_chunks]
        contexts = []
        
        for chunk in selected_chunks:
            if isinstance(chunk, dict):
                text = chunk.get('text') or chunk.get('content') or chunk.get('passage') or str(chunk)
            else:
                text = str(chunk)
            contexts.append(text)
        
        return contexts

# Keep original function as backup, but rename to avoid confusion
def get_relevant_contexts(question: str, chunks_data: List[Dict], max_chunks: int = 3) -> List[str]:
    """Deprecated: Please use get_contexts_from_faiss or get_relevant_contexts_from_chunks"""
    print("⚠️ Warning: get_relevant_contexts is deprecated, please use get_contexts_from_faiss")
    return get_relevant_contexts_from_chunks(question, chunks_data, max_chunks)

def prepare_privacy_evaluation_data(domain: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """Prepare privacy evaluation data for specified domain (targeted attacks)"""
    print(f"🔄 Preparing privacy evaluation data for {domain} domain...")
    
    prompts_data = load_privacy_prompts(domain)
    entities_data = load_privacy_entities(domain)
    
    private_entities = {}
    
    # Ensure prompts and entities count match
    min_length = min(len(prompts_data), len(entities_data))
    if len(prompts_data) != len(entities_data):
        print(f"⚠️ prompts({len(prompts_data)}) and entities({len(entities_data)}) count mismatch, using smaller value: {min_length}")
    
    # Build query and entity mapping
    for i in range(min_length):
        query = prompts_data[i]
        entity_list = entities_data[i]
        
        # 🔧 Improve entity extraction logic
        entities = []
        
        # Process entity list in string format
        if isinstance(entity_list, str):
            try:
                # Safely parse JSON string
                import json
                parsed_entities = json.loads(entity_list.replace("'", '"'))
                if isinstance(parsed_entities, list):
                    entity_list = parsed_entities
                else:
                    print(f"⚠️ Query {i+1} entity format exception, skipping")
                    entity_list = []
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ Query {i+1} entity parsing failed: {e}")
                entity_list = []
        
        # Extract entity name, filter empty values and duplicates
        seen_entities = set()
        for entity_item in entity_list:
            if isinstance(entity_item, dict) and 'entity' in entity_item:
                entity_value = str(entity_item['entity']).strip()
                if entity_value and entity_value not in seen_entities:
                    entities.append(entity_value)
                    seen_entities.add(entity_value)
            elif isinstance(entity_item, str):
                entity_value = entity_item.strip()
                if entity_value and entity_value not in seen_entities:
                    entities.append(entity_value)
                    seen_entities.add(entity_value)
        
        private_entities[query] = entities
        
        # Debug information
        if i < 3:  # Show detailed information for the first 3 queries
            print(f"     Query {i+1}: {len(entities)} entities - {entities[:3]}{'...' if len(entities) > 3 else ''}")
    
    print(f"✅ Preparation complete: {len(prompts_data)} targeted attack queries")
    return prompts_data, private_entities

def prepare_untargeted_evaluation_data(results: Dict = None) -> Tuple[List[str], Dict[str, List[str]]]:
    """Prepare untargeted attack evaluation data, extract private entities from retrieved documents"""
    print("🔄 Preparing untargeted attack evaluation data...")
    
    untargeted_queries = load_untargeted_attacks()
    private_entities = {}
    
    # If results file provided, extract private entities from it
    if results and 'queries' in results:
        print("📄 Extracting private entities from retrieved documents...")
        
        # Predefined private entity types (based on entities_biometrics_hagrid.json)
        private_entity_patterns = [
            # Biometric identifiers
            r'BIO-\d+', r'[A-Z]\d{11}', r'[A-Z]\d{10}',
            # Social Security Number
            r'\d{3}-\d{2}-\d{4}', r'\d{3} \d{3} \d{3}', r'ZZ \d{6} T',
            # Date (birth date, etc.)  
            r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}\.\d{2}\.\d{4}',
            # Device identifiers
            r'\d{15}', r'\d{12}',
            # IP address
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
            # Employee ID
            r'[A-Z]\d{7}', r'[A-Z]-\d{6}-[A-Z]', r'EMP\d+',
            # Medical record number
            r'MRN-\d+', r'MED\d+', r'[A-Z]\d{7}',
            # Address pattern (contains both numbers and letters)
            r'\d+[^,\n]+,\s*[A-Z][a-z]+',
            # Email
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            # Name (capitalized word group)
            r'[A-Z][a-z]+ [A-Z][a-z-]+',
        ]
        
        import re
        
        for query in untargeted_queries:
            entities = []
            
            # Find query match in results
            if query in results['queries']:
                query_data = results['queries'][query]
                
                # Extract entities from retrieved documents
                if 'retrieved_docs' in query_data:
                    retrieved_docs = query_data['retrieved_docs']
                    
                    for doc in retrieved_docs:
                        doc_text = str(doc)
                        
                        # Use regular expression to match private entities
                        for pattern in private_entity_patterns:
                            matches = re.findall(pattern, doc_text)
                            entities.extend(matches)
                    
                    # Remove duplicates and limit count
                    entities = list(set(entities))[:20]  # Maximum 20 entities
                    
                    print(f"    📋 Extracted {len(entities)} private entities from query '{query[:30]}...'")
            
            private_entities[query] = entities
    else:
        # If no results file provided, use default empty list
        print("⚠️ No results file provided, using empty private entity list")
        for query in untargeted_queries:
            private_entities[query] = []
    
    print(f"✅ Preparation complete: {len(untargeted_queries)} untargeted attack queries")
    return untargeted_queries, private_entities

def load_rag_results(results_file: str) -> Dict:
    """Load RAG system results file"""
    if not validate_file_exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return {}
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        total_queries = results.get('total_queries', 0)
        queries_count = len(results.get('queries', {}))
        
        print(f"✅ Successfully loaded RAG results: {queries_count} query results")
        return results
    except Exception as e:
        print(f"❌ Failed to load RAG results: {e}")
        return {} 