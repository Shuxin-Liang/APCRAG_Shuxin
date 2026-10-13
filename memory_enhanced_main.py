# -*- coding: utf-8 -*-
"""
Memory Enhanced RAG Privacy Protection Main Program
Demonstrates how to integrate persistent memory into the privacy editing process
"""

import json
import argparse
from typing import List, Dict, Optional
from modules.persistent_memory import PersistentMemory, get_contextual_examples
from modules.retriever import FAISSRetriever
from modules.privacy_detection import PrivacyDetector
from modules.privacy_editing import PrivacyEditor
from modules.utils import call_openai
from modules.config import get_domain_config, SUPPORTED_DOMAINS


class MemoryEnhancedRAG:
    """Memory Enhanced RAG System"""
    
    def __init__(self, domain: str = "finance"):
        """Initialize system components"""
        self.domain = domain
        self.config = get_domain_config(domain)
        
        # Initialize components
        self.retriever = FAISSRetriever(
            faiss_index_path=self.config['faiss_index'],
            chunks_file=self.config['chunks']
        )
        self.privacy_detector = PrivacyDetector()
        self.privacy_editor = PrivacyEditor()
        self.memory = PersistentMemory()
        
        print(f"✅ Initialization complete - Domain: {domain}")
        print(f"📊 Memory stats: {self.memory.get_memory_stats()}")
    
    def process_query(self, query: str, use_memory: bool = True, save_trajectory: bool = True) -> dict:
        """
        Process query with memory enhancement
        
        Args:
            query: User query
            use_memory: Whether to use examples from memory
            save_trajectory: Whether to save processing trajectory
            
        Returns:
            Result dictionary
        """
        print(f"\n🔍 Processing query: {query}")
        
        # 1. Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, top_k=3)
        if not retrieved_docs:
            return {"error": "No relevant documents found"}
        
        original_passage = retrieved_docs[0]  # Use the most relevant document
        print(f"📄 Retrieved document: {original_passage[:100]}...")
        
        # 2. Privacy detection
        detected_entities = self.privacy_detector.detect_privacy(original_passage)
        privacy_spans = self.privacy_detector.extract_privacy_spans(original_passage, detected_entities)
        
        print(f"🔒 Detected {len(privacy_spans)} privacy entities")
        for span in privacy_spans:
            print(f"  - {span['text']} ({span['label']})")
        
        # 3. Get similar memory examples (if enabled)
        contextual_examples = ""
        if use_memory:
            contextual_examples = get_contextual_examples(
                query=query,
                original_passage=original_passage,
                domain=self.domain
            )
            if contextual_examples:
                print(f"🧠 Found related memory examples for editing guidance")
        
        # 4. Build enhanced prompt and perform privacy editing
        enhanced_prompt = self._build_enhanced_prompt(
            query=query,
            original_passage=original_passage,
            detected_spans=privacy_spans,
            contextual_examples=contextual_examples
        )
        
        final_rewrite = call_openai(enhanced_prompt)
        print(f"📝 Completed privacy editing rewrite")
        
        # 5. Save trajectory to memory (if enabled)
        if save_trajectory:
            self.memory.add_trajectory(
                query=query,
                original_passage=original_passage,
                detected_spans=privacy_spans,
                final_rewrite=final_rewrite,
                domain=self.domain,
                score=0.85  # Can be adjusted based on quality assessment
            )
            print(f"💾 Saved processing trajectory to memory")
        
        return {
            "query": query,
            "original_passage": original_passage,
            "detected_spans": privacy_spans,
            "final_rewrite": final_rewrite,
            "used_memory": bool(contextual_examples),
            "domain": self.domain
        }
    
    def _build_enhanced_prompt(self, 
                             query: str,
                             original_passage: str,
                             detected_spans: List[Dict],
                             contextual_examples: str = "") -> str:
        """Build enhanced editing prompt"""
        
        # Privacy entity information
        privacy_info = ""
        if detected_spans:
            privacy_list = [f"'{span['text']}' ({span['label']})" for span in detected_spans]
            privacy_info = f"Detected privacy entities: {', '.join(privacy_list)}"
        
        prompt = f"""You are a professional privacy protection text editing assistant. Please rewrite the document according to the following requirements to ensure privacy safety.

Task: Rewrite the retrieved document to protect privacy information while answering the user's query.

User query: {query}

Original document: {original_passage}

{privacy_info}

Requirements:
1. Retain the core information and content needed to answer the user's query
2. Appropriately anonymize all identified privacy entities
3. Ensure the rewritten text remains meaningful and coherent
4. Maintain professional language style"""

        # Add memory examples (if any)
        if contextual_examples:
            prompt += f"""

Please refer to the processing example of similar situations: {contextual_examples}

Please refer to the processing method of the above example, but adjust it according to the current specific situation."""

        prompt += "\n\nPlease provide the rewritten document:"

        return prompt
    
    def interactive_feedback_loop(self):
        """Interactive feedback loop for improving memory quality"""
        print("\n🔄 Entering interactive feedback mode")
        print("Enter queries for processing, and provide feedback to improve memory quality after processing")
        print("Enter 'quit' to exit")
        
        while True:
            query = input("\nEnter query: ").strip()
            if query.lower() == 'quit':
                break
            
            if not query:
                continue
            
            # Process query
            result = self.process_query(query)
            
            if "error" in result:
                print(f"❌ {result['error']}")
                continue
            
            # Display result
            print(f"\n📋 Processing result:")
            print(f"Original document: {result['original_passage']}")
            print(f"Rewritten result: {result['final_rewrite']}")
            
            # Get user feedback
            feedback = input("\nEnter feedback (optional, press Enter to skip): ").strip()
            if feedback:
                # Assume getting the ID of the latest trajectory
                latest_id = len(self.memory.trajectories) - 1
                self.memory.update_trajectory_feedback(latest_id, feedback, score=0.9)
                print("✅ Feedback saved, will be used to improve future processing")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Memory Enhanced RAG Privacy Protection System")
    parser.add_argument("--domain", choices=SUPPORTED_DOMAINS, default="finance",
                       help="Choose processing domain")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive mode")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory function")
    parser.add_argument("--stats", action="store_true", help="Display memory stats information")
    
    args = parser.parse_args()
    
    # Initialize system
    rag_system = MemoryEnhancedRAG(domain=args.domain)
    
    if args.stats:
        print("\n📊 Detailed memory stats:")
        stats = rag_system.memory.get_memory_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    if args.interactive:
        rag_system.interactive_feedback_loop()
    elif args.query:
        result = rag_system.process_query(
            args.query, 
            use_memory=not args.no_memory
        )
        
        if "error" not in result:
            print(f"\n✅ Processing completed:")
            print(f"Query: {result['query']}")
            print(f"Original document: {result['original_passage'][:200]}...")
            print(f"Rewritten result: {result['final_rewrite']}")
            print(f"Used memory: {result['used_memory']}")
        else:
            print(f"❌ {result['error']}")
    else:
        print("Please provide --query or use --interactive mode")


if __name__ == "__main__":
    main() 