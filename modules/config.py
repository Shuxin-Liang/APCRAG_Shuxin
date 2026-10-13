# -*- coding: utf-8 -*-
"""Unified Configuration Management Module"""
"""Supports three domains: finance, biometrics, legal"""

import os
from typing import Dict, Optional

# Data Path Configuration
DATA_PATHS = {
    'finance': {
        'chunks': 'rag_data/chunks_finance_finqa.jsonl',
        'qa_data': 'rag_data/finqa_100_qaresp.json',
        'prompts': 'rag_data/prompts_finance_finqa.json',
        'entities': 'rag_data/entities_finance_finqa.json',
        'faiss_index': 'faiss_index_finance.index',
        'embeddings': 'embeddings_finance.npy',
        'dataset_name': 'finqa'
    },
    'biometrics': {
        'chunks': 'rag_data/chunks_biometrics_hagrid.jsonl',
        'qa_data': 'rag_data/hagrid_100_qaresp.json',
        'prompts': 'rag_data/prompts_biometrics_hagrid.json',
        'entities': 'rag_data/entities_biometrics_hagrid.json',
        'faiss_index': 'faiss_index_biometrics.index',
        'embeddings': 'embeddings_biometrics.npy',
        'dataset_name': 'hagrid'
    },
    'legal': {
        'chunks': 'rag_data/chunks_legal_documents_cuad.jsonl',
        'qa_data': 'rag_data/cuad_100_qaresp.json',
        'prompts': 'rag_data/prompts_legal_documents_cuad.json',
        'entities': 'rag_data/entities_legal_documents_cuad.json',
        'faiss_index': 'faiss_index_legal.index',
        'embeddings': 'embeddings_legal.npy',
        'dataset_name': 'cuad'
    }
}

# Common File Paths
UNTARGETED_ATTACKS_FILE = 'rag_data/untargeted_attacks.json'

# Supported Domain List
SUPPORTED_DOMAINS = list(DATA_PATHS.keys())

def get_domain_config(domain: str) -> Dict[str, str]:
    """Get configuration for specified domain"""
    if domain not in DATA_PATHS:
        raise ValueError(f"Unsupported domain: {domain}. Supported domains: {SUPPORTED_DOMAINS}")
    return DATA_PATHS[domain].copy()

def get_file_path(domain: str, file_type: str) -> str:
    """Get path for specified domain and file type"""
    config = get_domain_config(domain)
    if file_type not in config:
        raise ValueError(f"Unsupported file type: {file_type}. Supported types: {list(config.keys())}")
    return config[file_type]

def validate_file_exists(file_path: str) -> bool:
    """Check if file exists"""
    return os.path.exists(file_path)

def validate_domain_files(domain: str) -> Dict[str, bool]:
    """Validate if all required files exist for specified domain"""
    config = get_domain_config(domain)
    validation_results = {}
    
    for file_type, file_path in config.items():
        if file_type not in ['dataset_name']:  # Skip non-file path configurations
            validation_results[file_type] = validate_file_exists(file_path)
    
    return validation_results

def auto_detect_domain(file_path: str) -> Optional[str]:
    for domain, config in DATA_PATHS.items():
        for file_type, path in config.items():
            if file_type != 'dataset_name' and path in file_path:
                return domain
    return None

def get_domain_from_args(faiss_index: str = None, chunks_file: str = None, qa_file: str = None) -> Optional[str]:
    if faiss_index:
        for domain in SUPPORTED_DOMAINS:
            if domain in faiss_index:
                return domain

    if chunks_file:
        return auto_detect_domain(chunks_file)

    if qa_file:
        return auto_detect_domain(qa_file)
    
    return None

def print_domain_status():
    """Print status of all domain files"""
    print("📂 Domain File Status Check:")
    print("=" * 80)
    
    for domain in SUPPORTED_DOMAINS:
        print(f"\n🔍 {domain.upper()} Domain:")
        validation_results = validate_domain_files(domain)
        
        for file_type, exists in validation_results.items():
            file_path = get_file_path(domain, file_type)
            status = "✅" if exists else "❌"
            print(f"  {status} {file_type:<12}: {file_path}")
    
    # Check common files
    print(f"\n🔍 Common Files:")
    untargeted_exists = validate_file_exists(UNTARGETED_ATTACKS_FILE)
    status = "✅" if untargeted_exists else "❌"
    print(f"  {status} untargeted   : {UNTARGETED_ATTACKS_FILE}")

def detect_domain_from_path(file_path: str) -> str:
    """Automatically detect domain from file path"""
    for domain in SUPPORTED_DOMAINS:
        if domain in file_path:
            return domain
    return None

def infer_domain_from_args(args) -> str:
    """Infer domain from command line arguments"""
    # First try to infer from faiss_index
    if hasattr(args, 'faiss_index') and args.faiss_index:
        domain = detect_domain_from_path(args.faiss_index)
        if domain:
            return domain
    
    # Then try to infer from chunks_file
    if hasattr(args, 'chunks_file') and args.chunks_file:
        domain = detect_domain_from_path(args.chunks_file)
        if domain:
            return domain
    
    # Finally try to infer from qa_file
    if hasattr(args, 'qa_file') and args.qa_file:
        domain = detect_domain_from_path(args.qa_file)
        if domain:
            return domain
    
    return None

if __name__ == "__main__":
    print_domain_status() 