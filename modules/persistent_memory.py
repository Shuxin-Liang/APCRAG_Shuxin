# -*- coding: utf-8 -*-
"""
Persistent Memory Module
For storing and retrieving rewrite trajectories, used as context examples for reuse
"""

import json
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersistentMemory:
    """Persistent Memory Manager"""
    
    def __init__(self, memory_file: str = "memory_trajectories.json"):
        """
        Initialize persistent memory
        
        Args:
            memory_file: Memory file path
        """
        self.memory_file = memory_file
        self.trajectories = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.trajectory_vectors = None
        self.is_fitted = False
        
        # Load existing memory
        self._load_trajectories()
        
        # If there are trajectories, build vector index immediately
        if self.trajectories:
            self._build_vector_index()
    
    def _load_trajectories(self):
        """Load memory trajectories from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.trajectories = data.get('trajectories', [])
                logger.info(f"✅ Loaded {len(self.trajectories)} memory trajectories")
            else:
                logger.info("📝 Memory file does not exist, new memory will be created")
                self.trajectories = []
        except Exception as e:
            logger.error(f"❌ Failed to load memory: {e}")
            self.trajectories = []
    
    def save_memory(self):
        """Save memory trajectories to file"""
        try:
            data = {
                'trajectories': self.trajectories,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved {len(self.trajectories)} memory trajectories")
        except Exception as e:
            logger.error(f"❌ Failed to save memory: {e}")
    
    def add_trajectory(self, 
                      query: str,
                      original_passage: str,
                      detected_spans: Optional[List[Dict]] = None,
                      final_rewrite: Optional[str] = None,
                      domain: str = "general",
                      human_feedback: Optional[str] = None,
                      score: float = 0.0):
        """
        Add new rewrite trajectory
        
        Args:
            query: User query
            original_passage: Original search passage
            detected_spans: Detected privacy entity spans
            final_rewrite: Final rewrite result
            domain: Domain label
            human_feedback: Human feedback
            score: Quality score
        """
        trajectory = {
            'id': len(self.trajectories),
            'query': query,
            'original_passage': original_passage,
            'detected_spans': detected_spans or [],
            'final_rewrite': final_rewrite or "",
            'domain': domain,
            'human_feedback': human_feedback,
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.trajectories.append(trajectory)
        
        # Rebuild vector index
        self._build_vector_index()
        
        # Auto save
        self.save_memory()
        
        logger.info(f"📝 Added new trajectory: {query[:50]}...")
    
    def _build_vector_index(self):
        """Build vector index for trajectories for semantic retrieval"""
        if not self.trajectories:
            return
        
        try:
            # Combine query and original passage as text features
            texts = []
            for traj in self.trajectories:
                combined_text = f"{traj['query']} {traj['original_passage']}"
                texts.append(combined_text)
            
            # Build TF-IDF vector
            self.trajectory_vectors = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
            logger.info(f"🔍 Built vector index for {len(texts)} trajectories")
            
        except Exception as e:
            logger.error(f"❌ Failed to build vector index: {e}")
            self.is_fitted = False
    
    def retrieve_similar_trajectories(self, 
                                    query: str, 
                                    original_passage: str = "",
                                    domain: Optional[str] = None,
                                    top_k: int = 3,
                                    min_score: float = 0.1) -> List[Dict]:
        """
        Retrieve semantically similar trajectories
        
        Args:
            query: Current query
            original_passage: Current search passage
            domain: Domain filter
            top_k: Return top k similar trajectories
            min_score: Minimum similarity threshold
            
        Returns:
            Similar trajectory list
        """
        if not self.is_fitted or not self.trajectories:
            return []
        
        try:
            # Combine current text
            current_text = f"{query} {original_passage}"
            
            # Calculate similarity
            current_vector = self.vectorizer.transform([current_text])
            similarities = cosine_similarity(current_vector, self.trajectory_vectors)[0]
            
            # Get candidate trajectories and their similarities
            candidates = []
            for i, sim in enumerate(similarities):
                if sim >= min_score:
                    traj = self.trajectories[i].copy()
                    traj['similarity'] = sim
                    candidates.append(traj)
            
            # Sort by similarity
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Domain filter
            if domain:
                candidates = [c for c in candidates if c['domain'] == domain]
            
            # Return top k
            result = candidates[:top_k]
            
            logger.info(f"🎯 Found {len(result)} similar trajectories (min_score={min_score})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve similar trajectories: {e}")
            return []
    
    def format_as_examples(self, trajectories: List[Dict]) -> str:
        """
        Format trajectories as context examples
        
        Args:
            trajectories: Trajectory list
            
        Returns:
            Formatted example text
        """
        if not trajectories:
            return ""
        
        examples = []
        for i, traj in enumerate(trajectories, 1):
            example = f"""Example {i}:
Query: {traj['query']}
Original passage: {traj['original_passage']}
Rewrite result: {traj['final_rewrite']}"""
            
            # Add detected entity information
            if traj.get('detected_spans'):
                spans_info = []
                for span in traj['detected_spans']:
                    spans_info.append(f"{span.get('text', 'N/A')} ({span.get('label', 'N/A')})")
                if spans_info:
                    example += f"\nDetected entities: {', '.join(spans_info)}"
            
            # Add human feedback
            if traj.get('human_feedback'):
                example += f"\nFeedback: {traj['human_feedback']}"
            
            examples.append(example)
        
        return "\n\n" + "\n\n".join(examples) + "\n"
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        if not self.trajectories:
            return {'total': 0, 'domains': {}}
        
        domain_counts = {}
        total_score = 0
        
        for traj in self.trajectories:
            domain = traj.get('domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            total_score += traj.get('score', 0)
        
        return {
            'total': len(self.trajectories),
            'domains': domain_counts,
            'average_score': total_score / len(self.trajectories) if self.trajectories else 0,
            'has_vector_index': self.is_fitted
        }
    
    def update_trajectory_feedback(self, trajectory_id: int, feedback: str, score: Optional[float] = None):
        """
        Update human feedback for a trajectory
        
        Args:
            trajectory_id: Trajectory ID
            feedback: Human feedback
            score: Updated score
        """
        if 0 <= trajectory_id < len(self.trajectories):
            self.trajectories[trajectory_id]['human_feedback'] = feedback
            if score is not None:
                self.trajectories[trajectory_id]['score'] = score
            self.trajectories[trajectory_id]['last_updated'] = datetime.now().isoformat()
            
            # Save update
            self.save_memory()
            logger.info(f"📝 Updated trajectory {trajectory_id} feedback")
        else:
            logger.error(f"❌ Invalid trajectory ID: {trajectory_id}")
    
    def clear_low_quality_trajectories(self, min_score: float = 0.5):
        """Clear low quality trajectories"""
        original_count = len(self.trajectories)
        self.trajectories = [t for t in self.trajectories if t.get('score', 0) >= min_score]
        removed_count = original_count - len(self.trajectories)
        
        if removed_count > 0:
            # Rebuild index
            self._build_vector_index()
            self.save_memory()
            logger.info(f"🧹 Cleared {removed_count} low quality trajectories")


# Convenience function
def get_contextual_examples(query: str, 
                          original_passage: str = "",
                          domain: Optional[str] = None,
                          memory_file: str = "memory_trajectories.json") -> str:
    """
    Convenience function to get contextual examples
    
    Args:
        query: Current query
        original_passage: Current search passage  
        domain: Domain filter
        memory_file: Memory file path
        
    Returns:
        Formatted contextual examples
    """
    memory = PersistentMemory(memory_file)
    similar_trajectories = memory.retrieve_similar_trajectories(
        query=query,
        original_passage=original_passage,
        domain=domain,
        top_k=3
    )
    return memory.format_as_examples(similar_trajectories)


if __name__ == "__main__":
    # Test code
    memory = PersistentMemory()
    print("📊 Memory statistics:")
    stats = memory.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test retrieval
    test_query = "User registration information"
    test_passage = "New user registration: Name: Zhang San, Phone: 13800138000"
    examples = get_contextual_examples(test_query, test_passage)
    
    if examples:
        print(f"\n🎯 Contextual examples found for query '{test_query}':")
        print(examples)
    else:
        print(f"\n❌ No related examples found") 