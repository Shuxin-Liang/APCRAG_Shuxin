from typing import List, Dict, Union, Optional
import json
from modules.utils import call_openai

class PrivacyEditingAgent:
    """Privacy Editing Agent - Implements multiple editing strategies with optional memory support"""
    
    def __init__(self, memory_enabled: bool = False, memory_db_path: str = "rewrite_memory.db"):
        self.strategies = {
            'generalize': self._generalize_strategy,
            'memory_generalize': self._memory_generalize_strategy
        }
        
        # Memory system integration
        self.memory_enabled = memory_enabled
        self.memory = None
        if memory_enabled:
            try:
                from .persistent_memory import PersistentMemory
                self.memory = PersistentMemory(memory_db_path)
                print(f"📝 Persistent memory system enabled: {memory_db_path}")
                print(f"📊 Memory statistics: {self.memory.get_memory_stats()}")
            except ImportError:
                print("⚠️  Warning: Cannot import memory system, using standard mode")
                self.memory_enabled = False
        
        # Role-based templates for different domains
        self.role_templates = {
            'financial': {
                'role_description': 'Financial Institution Internal Communication Specialist',
                'guidelines': [
                    'Protecting customer privacy is the primary principle',
                    'Use terms like "Customer A", "a certain customer", etc.',
                    'Avoid disclosing specific account information and transaction details',
                    'Maintain accuracy of professional financial terminology'
                ],
                'replacement_patterns': {
                    'person_name': ['Customer A', 'a certain customer', 'the customer'],
                    'account_number': ['related account', 'customer account'],
                    'company_name': ['a financial institution', 'partner']
                }
            },
            'healthcare': {
                'role_description': 'Medical Institution Information Management Specialist',
                'guidelines': [
                    'Strictly follow medical privacy protection regulations',
                    'Use anonymized terms like "patient", "case A"',
                    'Avoid disclosing specific medical records and personal health information',
                    'Maintain professionalism in medical terminology'
                ],
                'replacement_patterns': {
                    'person_name': ['patient', 'case A', 'a patient'],
                    'medical_record': ['related medical record', 'patient record'],
                    'hospital_name': ['a medical institution', 'this hospital']
                }
            },
            'legal': {
                'role_description': 'Legal Affairs Communication Coordinator',
                'guidelines': [
                    'Protect client privacy and case confidentiality',
                    'Use terms like "Party A", "case-related party"',
                    'Avoid disclosing specific case details and personal information',
                    'Maintain professionalism in legal terminology'
                ],
                'replacement_patterns': {
                    'person_name': ['party A', 'case related party', 'a party'],
                    'case_number': ['related case', 'case archive'],
                    'law_firm': ['a law firm', 'cooperative institution']
                }
            },
            'corporate': {
                'role_description': 'Corporate Internal Information Management Specialist',
                'guidelines': [
                    'Protect employee privacy and business confidentiality',
                    'Use terms like "Employee A", "a department colleague"',
                    'Avoid disclosing specific HR information and business details',
                    'Maintain professionalism in corporate communication'
                ],
                'replacement_patterns': {
                    'person_name': ['employee A', 'a colleague', 'team member'],
                    'employee_id': ['related employee', 'employee archive'],
                    'department': ['a department', 'related team']
                }
            }
        }
    
    def edit_content(self, query: str, passage: str, private_spans: str, strategy: str = 'generalize',
                    user_id: str = None, domain: str = None) -> str:
        """Main interface for editing content with optional memory support"""
        
        # Special handling for rephrase strategy
        if strategy == 'rephrase':
            if self.memory_enabled:
                return self._memory_enhanced_rephrase_strategy(query, passage, private_spans, user_id, domain)
            else:
                raise ValueError("rephrase strategy requires memory to be enabled. Please set memory_enabled=True or use 'generalize' strategy.")
        
        # Handle other strategies
        if strategy not in self.strategies:
            raise ValueError(f"Unsupported strategy: {strategy}. Available strategies: {list(self.strategies.keys())} or 'rephrase' (requires memory_enabled=True)")
        
        return self.strategies[strategy](query, passage, private_spans)
    
    def _memory_enhanced_rephrase_strategy(self, query: str, passage: str, private_spans: str,
                                         user_id: str = None, domain: str = None) -> str:
        """Memory-enhanced rephrase strategy with role-playing"""
        
        # 1. Parse private spans to extract individual spans and risk levels
        span_info = self._parse_private_spans(private_spans)
        
        # 2. Get domain-specific role template
        role_template = self.role_templates.get(domain, self.role_templates['corporate'])
        
        # 3. Get contextual examples from memory
        contextual_examples = ""
        if self.memory:
            similar_trajectories = self.memory.retrieve_similar_trajectories(
                query=query,
                original_passage=passage,
                domain=domain,
                top_k=2
            )
            if similar_trajectories:
                contextual_examples = self.memory.format_as_examples(similar_trajectories)
                print(f"🧠 Found {len(similar_trajectories)} related memory examples")
        
        # 4. Generate role-based prompt with memory context
        role_prompt = self._build_role_based_prompt_with_memory(
            query, passage, private_spans, role_template, contextual_examples
        )
        
        # 5. Get AI response
        rewritten_content = call_openai(role_prompt)
        
        # 6. Record new trajectory in memory if enabled
        if self.memory:
            self.memory.add_trajectory(
                query=query,
                original_passage=passage,
                detected_spans=span_info,
                final_rewrite=rewritten_content,
                domain=domain or 'corporate',
                score=0.85  # Can be adjusted based on quality assessment
            )
            print(f"💾 Saved new rewrite trajectory to memory")
        
        return rewritten_content
    
    def _build_role_based_prompt_with_memory(self, query: str, passage: str, private_spans: str, 
                                           role_template: Dict, contextual_examples: str) -> str:
        """Build role-based prompt (with memory support)"""
        
        # Role-specific replacement patterns
        patterns_context = "\nRecommended replacement patterns:\n"
        for span_type, replacements in role_template['replacement_patterns'].items():
            patterns_context += f"• {span_type}: {', '.join(replacements)}\n"
        
        prompt = f"""You are a {role_template['role_description']}, responsible for ensuring the security and compliance of internal information before sharing it externally.

Role responsibility guidelines:
{chr(10).join(['• ' + guideline for guideline in role_template['guidelines']])}

Tasks:
• Rewrite internal message based on privacy risk level
• Risk handling rules:

[High-Risk Span Handling]
• Do NOT include any word labeled high-risk directly in your rewritten passage.
• Summarize the key information relevant to the user query using only abstract references (e.g., "the individual", "their details").
• Avoid specific identifiers or exact phrases from high-risk spans.
• Keep your summary concise, informative, and safe.

[Medium-Risk Span Handling]
• Replace names and specific identifiers with general categories (e.g., "a customer", "a financial institution").
• Use paraphrasing or summarization to obscure identity without losing meaning.
• Ensure the rewritten span is fluent and retains its relevance to the user query.

[Low-Risk Span Handling]
• Retain information as-is if it is not sensitive and supports the user query.
• Do not alter factual, general, or aggregated content that carries no privacy risk.

[None-Risk Span Handling]
• Keep these spans unchanged as they pose no privacy concerns.
• Use them freely to maintain context and readability.

• Use role-based professional judgment to assess whether information should be disclosed externally
{patterns_context}"""

        # Add memory examples
        if contextual_examples:
            prompt += f"""

Reference successful handling cases from similar situations: {contextual_examples}

Please refer to the handling style and approach of the above cases, but adjust them according to the current specific situation."""

        prompt += f"""

Privacy annotation information: "{private_spans}"
Input message: "{passage}"
User query: "{query}"

Output: Rewritten message conforming to {role_template['role_description']} standards, ensuring safe boundary and risk level handling"""
        
        return prompt
    
    def _build_role_based_prompt(self, query: str, passage: str, private_spans: str, 
                                role_template: Dict, memory_suggestions: Dict) -> str:
        """Build role-based prompt (backward compatible method)"""
        return self._build_role_based_prompt_with_memory(query, passage, private_spans, role_template, "")
    
    def _parse_private_spans(self, private_spans: str) -> List[Dict]:
        """Parse privacy annotation information"""
        spans = []
        
        # Simple parsing - look for patterns like "span_text: private risk: level"
        import re
        pattern = r'([^:]+?):\s*private risk:\s*(high|medium|low|none)'
        matches = re.findall(pattern, private_spans, re.IGNORECASE)
        
        for span_text, risk_level in matches:
            spans.append({
                'span': span_text.strip(),
                'risk': risk_level.lower()
            })
        
        return spans
    
    def _extract_rewritten_span(self, original_span: str, rewritten_content: str) -> str:
        """Extract rewrite result for specific span from rewritten content (simplified version)"""
        # This is a simplified extraction - in practice, you might want more sophisticated matching
        if original_span.lower() in rewritten_content.lower():
            return original_span  # No change
        else:
            # Return a generic placeholder - real implementation would need more sophisticated matching
            return "[Rewritten]"
    
    def provide_feedback(self, decision_id: int, feedback_type: str, 
                        corrected_decision: str = None, feedback_reason: str = None,
                        user_id: str = None):
        """Provide human feedback to improve memory system"""
        if self.memory:
            self.memory.record_feedback(
                decision_id=decision_id,
                feedback_type=feedback_type,
                original_decision=self.memory._get_decision_by_id(decision_id),
                corrected_decision=corrected_decision,
                feedback_reason=feedback_reason,
                user_id=user_id
            )
            print(f"Feedback recorded: {feedback_type} (Decision ID: {decision_id})")
        else:
            print("Memory system not enabled, cannot record feedback")
    
    def _generalize_strategy(self, query: str, passage: str, private_spans: str) -> str:
        """Memory-enhanced generalization strategy"""
        
        # First try using persistent memory system
        if self.memory and self.memory.trajectories:
            contextual_examples = ""
            similar_trajectories = self.memory.retrieve_similar_trajectories(
                query=query,
                original_passage=passage,
                domain='general',
                top_k=2,
                min_score=0.1
            )
            
            if similar_trajectories:
                contextual_examples = self.memory.format_as_examples(similar_trajectories)
                print(f"  🧠 Used {len(similar_trajectories)} persistent memory examples for rewriting")
                
                # Use memory-enhanced prompt
                prompt = self._build_enhanced_memory_prompt(query, passage, private_spans, contextual_examples)
                rewritten_content = call_openai(prompt)
                
                # Save new trajectory
                self.memory.add_trajectory(
                    query=query,
                    original_passage=passage,
                    detected_spans=self._parse_private_spans(private_spans),
                    final_rewrite=rewritten_content,
                    domain='general',
                    score=0.8
                )
                print(f"  💾 Saved new trajectory to persistent memory")
                
                return rewritten_content
        
        # Fall back to JSON file memory system
        examples = self._load_json_examples()
        
        if not examples:
            # If no memory found, fall back to standard strategy
            print("  ⚠️  No relevant memory examples found, using standard generalization strategy")
            return self._generalize_strategy(query, passage, private_spans)
        
        # Select relevant examples
        relevant_examples = self._select_relevant_examples(query, examples, max_examples=2)
        
        if not relevant_examples:
            print("  ⚠️  No relevant memory examples found, using standard generalization strategy")
            return self._generalize_strategy(query, passage, private_spans)
        
        print(f"  🧠 Used {len(relevant_examples)} JSON memory examples for rewriting")
        
        # Build memory-based prompt
        prompt = self._build_memory_prompt(query, passage, private_spans, relevant_examples)
        
        # Generate rewritten content
        return call_openai(prompt)

    def _load_json_examples(self, domain: str = None) -> List[Dict]:
        """Load JSON format memory examples"""
        import os
        
        # Search memory files by priority
        possible_files = [
            f"{domain}_examples.json" if domain else None,
            "qa_examples.json", 
            "memory_examples.json"
        ]
        
        for file_path in possible_files:
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    examples = data.get('trajectories', [])
                    if examples:
                        print(f"  📂 Loaded {len(examples)} memory examples from {file_path}")
                        return examples
                except Exception as e:
                    print(f"  ⚠️  Failed to load memory file {file_path}: {e}")
                    continue
        
        return []

    def _select_relevant_examples(self, query: str, examples: List[Dict], 
                                domain: str = None, max_examples: int = 2) -> List[Dict]:
        """Select relevant memory examples"""
        relevant = []
        
        for example in examples:
            # Domain matching
            if domain and example.get('domain') != domain:
                continue
                
            # Score filtering
            if example.get('score', 0) < 0.5:
                continue
                
            relevant.append(example)
        
        # Sort by score, take top N
        relevant.sort(key=lambda x: x.get('score', 0), reverse=True)
        return relevant[:max_examples]

    def _build_enhanced_memory_prompt(self, query: str, passage: str, private_spans: str, 
                                    contextual_examples: str) -> str:
        """Build enhanced memory prompt (for persistent memory)"""
        
        prompt = f"""You are a privacy protection expert, rewriting documents to protect sensitive information based on historical successful cases.

Handling rules:
• High risk: Completely remove or abstract, using general description
• Medium risk: Generalize processing, retain semantics but remove specific identifiers  
• Low risk: Appropriate processing, maintain readability and usefulness
• None risk: Keep unchanged

Risk level handling guidelines:
[High-Risk]: Cannot directly include any high-risk identifier vocabulary, using abstract reference (e.g., "the individual", "related details")
[Medium-Risk]: Use general categories instead of specific identifiers (e.g., "a customer", "a financial institution")
[Low-Risk]: Retain information if it is not sensitive and supports the query, do not modify factual, general, or aggregated content that carries no privacy risk
[None-Risk]: Completely keep unchanged, freely use to maintain context and readability"""

        # Add memory examples
        if contextual_examples:
            prompt += f"""

Reference successful handling cases from similar situations: {contextual_examples}

Please learn the handling style and approach of the above cases, but adjust them according to the current specific situation."""

        prompt += f"""

=== Current Task ===
User query: {query}
Original document: {passage}
Privacy detection result: {private_spans}

Please generate a privacy-protected rewritten version based on the above experience and rules:"""
        
        return prompt

    def _build_memory_prompt(self, query: str, passage: str, private_spans: str, 
                           examples: List[Dict]) -> str:
        """Build memory example prompt (for JSON memory)"""
        
        base_prompt = """You are a privacy protection expert, rewriting documents to protect sensitive information based on examples.

Handling rules:
• High risk: Completely remove or abstract
• Medium risk: Generalize processing, retain semantics but remove specific identifiers
• Low risk: Appropriate processing, maintain readability
• None risk: Keep unchanged

"""
        
        # Add memory examples
        if examples:
            base_prompt += "Reference successful cases:\n"
            for i, ex in enumerate(examples, 1):
                base_prompt += f"""
Case {i} (Score: {ex.get('score', 0):.1f}):
Query: {ex.get('query', '')}
Original: {ex.get('original_passage', '')}
Rewritten: {ex.get('final_rewrite', '')}
---"""
        
        # Current task
        current_task = f"""

=== Current Task ===
Query: {query}
Original: {passage}
Privacy detection result: {private_spans}

Please refer to the rewritten style of the above cases to perform privacy-aware rewriting of the current document:"""
        
        return base_prompt + current_task


class IntegratedPrivacyPipeline:
    """Integrated Privacy-Aware RAG Pipeline"""
    
    def __init__(self, detection_strategy: str = 'zero_shot', editing_strategy: str = 'generalize',
                 memory_enabled: bool = False, memory_db_path: str = "rewrite_memory.db"):
        from .privacy_detection import PrivacyDetectionAgent
        
        self.detection_agent = PrivacyDetectionAgent()
        self.editing_agent = PrivacyEditingAgent(memory_enabled=memory_enabled, memory_db_path=memory_db_path)
        self.detection_strategy = detection_strategy
        self.editing_strategy = editing_strategy
        self.memory_enabled = memory_enabled
    
    def process_document(self, query: str, passage: str, max_iterations: int = 3,
                        user_id: str = None, domain: str = None) -> Dict[str, Union[str, List[str]]]:
        """Complete pipeline for processing a single document"""
        results = {
            'original_passage': passage,
            'query': query,
            'detection_results': [],
            'edited_passages': [],
            'final_passage': passage,
            'user_id': user_id,
            'domain': domain
        }
        
        current_passage = passage
        
        # Iterative detection and editing process
        for iteration in range(max_iterations):
            # Detection phase
            detection_result = self.detection_agent.detect_private_spans(
                query, current_passage, self.detection_strategy
            )
            results['detection_results'].append(detection_result)
            
            # Check if there are still privacy issues
            if self._no_privacy_violations(detection_result):
                break
            
            # Editing phase with memory support
            edited_passage = self.editing_agent.edit_content(
                query, current_passage, detection_result, self.editing_strategy,
                user_id=user_id, domain=domain
            )
            results['edited_passages'].append(edited_passage)
            
            current_passage = edited_passage
        
        results['final_passage'] = current_passage
        return results
    
    def _no_privacy_violations(self, detection_result: str) -> bool:
        """Check if there are no privacy violations"""
        # Simple heuristic check
        no_violation_indicators = [
            "No privacy violations",
            "No privacy violations detected",
            "No privacy violations found"
        ]
        
        # Check for string indicators
        for indicator in no_violation_indicators:
            if indicator in detection_result:
                return True
        
        # Check for new risk level format
        has_high_risk = "private risk: high" in detection_result
        has_medium_risk = "private risk: medium" in detection_result  
        has_low_risk = "private risk: low" in detection_result
        has_none_risk = "private risk: none" in detection_result
        
        # If we only have none-risk spans, consider it safe
        if has_none_risk and not (has_high_risk or has_medium_risk or has_low_risk):
            return True
        
        # Check for legacy format (backwards compatibility)
        has_not_private = "[Not Private]" in detection_result
        has_private = "[Private]" in detection_result
        
        # If we have [Not Private] but no [Private], consider it safe
        if has_not_private and not has_private:
            return True
            
        return False
    
    def process_multiple_documents(self, query: str, documents: List[str]) -> List[Dict]:
        """Process multiple documents"""
        results = []
        
        for i, doc in enumerate(documents):
            doc_result = self.process_document(query, doc)
            results.append(doc_result)
        
        return results
    
    def provide_feedback(self, decision_id: int, feedback_type: str, 
                        corrected_decision: str = None, feedback_reason: str = None,
                        user_id: str = None):
        """Provide feedback to improve memory system"""
        if self.memory_enabled:
            self.editing_agent.provide_feedback(
                decision_id=decision_id,
                feedback_type=feedback_type,
                corrected_decision=corrected_decision,
                feedback_reason=feedback_reason,
                user_id=user_id
            )
        else:
            print("Memory system not enabled, cannot provide feedback") 