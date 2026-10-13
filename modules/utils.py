# -*- coding: utf-8 -*-
"""Utility Functions Module"""
"""Unified management of common functions to avoid code duplication"""

import openai
import os
from typing import List, Dict, Optional

# Global Configuration
MODEL = 'gpt-4o-mini'  # Upgraded to GPT-4o-mini for better performance

def call_openai(prompt: str, model: str = MODEL) -> str:
    """
    Unified OpenAI API call function
    Avoid duplicate definitions across multiple modules
    """
    try:
        # Read API Key
        if os.path.exists("openai_key.txt"):
            with open("openai_key.txt", "r") as f:
                api_key = f.read().strip()
        elif os.getenv("OPENAI_API_KEY"):
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key not found")
            
        # Create OpenAI client (new version API)
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI API call failed: {e}")
        return f"Generation failed: {str(e)}"

def generate_answer_from_docs(query: str, docs: List[str]) -> str:
    """
    Generate answer based on retrieved documents
    Migrated from generator.py module
    """
    context_block = "\n\n".join(docs)
    prompt = f"""
You are a helpful assistant. Use the following documents to answer the question.

IMPORTANT: Do not repeat information that is already present in the user's question. Focus on providing new, relevant information from the documents that directly answers the question without restating what the user already knows.

Documents:
{context_block}

Question: {query}
Answer:"""
    return call_openai(prompt)

def clean_text(text: str) -> str:
    """Clean text content"""
    if not text:
        return ""
    return text.strip().replace('\n', ' ').replace('\r', '')

def validate_json(data: dict) -> bool:
    """Validate JSON data format"""
    if not isinstance(data, dict):
        return False
    return True

def safe_get(data: dict, key: str, default: any = None) -> any:
    """Safely get dictionary value"""
    return data.get(key, default) 