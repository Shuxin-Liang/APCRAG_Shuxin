from typing import List, Dict, Tuple
import re
from modules.utils import call_openai

class PrivacyDetectionAgent:
    """Privacy Detection Agent - Implements multiple detection strategies"""
    
    def __init__(self):
        self.strategies = {
            'zero_shot': self._zero_shot_detection,
            'static_expert': self._static_expert_detection,
            'few_shot': self._few_shot_detection,
            'cot': self._cot_detection
        }
    
    def detect_private_spans(self, query: str, passage: str, strategy: str = 'zero_shot') -> str:
        """Main interface for detecting privacy spans"""
        if strategy not in self.strategies:
            raise ValueError(f"Unsupported strategy: {strategy}. Available strategies: {list(self.strategies.keys())}")
        
        return self.strategies[strategy](query, passage)
    
    def _zero_shot_detection(self, query: str, passage: str) -> str:
        """Zero-Shot (Baseline) detection strategy"""
        prompt = f"""Detect spans in the passage that may violate privacy if revealed in the context of the user's query.

Instructions: Detect spans in the passage that may violate privacy in the context of the user query.

Guidelines:
• Label spans with privacy risk level: high, medium, low, none
• Provide a short reason for each decision

Risk Levels:
• high: Highly sensitive personal identifiers, financial data, credentials
• medium: Somewhat sensitive information that could identify or harm someone
• low: Minor privacy concerns, public but contextually sensitive information
• none: Public or general information with no privacy concerns

Input Passage: "{passage}"
User Query: "{query}"

Output: List of spans with privacy risk levels and reasons."""
        
        return call_openai(prompt)
    
    def _static_expert_detection(self, query: str, passage: str) -> str:
        """StaticExpert (Rule-based) detection strategy"""
        prompt = f"""You are a privacy detection agent operating under expert rules.

Expert Rules:
• Personal Identifiers: Full names, first names, last names, usernames, employee IDs, customer IDs, account numbers, other unique personal identifiers, national identification numbers, tax IDs
• Contact Information: Email addresses, phone numbers, street addresses, full physical addresses, city names, postal codes, states, countries
• Online Identifiers: IP addresses (IPv4, IPv6), website URLs, device identifiers, API keys (when identifying a user or system privately)
• Government IDs: Social security numbers, medical record numbers, health plan beneficiary numbers, biometric identifiers, certificate or license numbers, license plate numbers, vehicle identification numbers
• Financial Information: Credit card numbers, card verification values (CVV), personal identification numbers (PINs), bank routing numbers
• Temporal Information: Specific dates of birth, exact dates and times of sensitive events or transactions linked to an individual
• Security Information: Passwords, security tokens, private API keys used for authentication or authorization
• Organization Information: Specific company names involved in private dealings, internal project codenames not meant for public disclosure
• Tracking Information: Precise GPS coordinates, specific device location history that can track an individual

Privacy Risk Levels:
• high: SSN, credit cards, passwords, precise locations, full personal identifiers
• medium: Names, addresses, phone numbers, internal IDs, IP addresses
• low: General locations, company names, public identifiers
• none: Public information, general terms, non-identifying data

Input Passage: "{passage}"
User Query: "{query}"

Output: List of spans with privacy risk levels and reasons."""
        
        return call_openai(prompt)
    
    def _few_shot_detection(self, query: str, passage: str) -> str:
        """Few-Shot (In-Context Demonstration) detection strategy"""
        prompt = f"""Determine the privacy risk level for each span in the passage in context of a query. Label each span with privacy risk: high, medium, low, or none. Provide a short reason for each label.

Risk Levels:
• high: Highly sensitive identifiers, financial data, passwords, precise locations
• medium: Personal names, contact info, internal IDs, IP addresses
• low: General locations, company names, contextually sensitive but public info
• none: Public, general, or aggregated information with no privacy concerns

Example 1 (Finance)
User Query: "What is the business of GreenGen Corp and its P/E ratio?"
Passage: "GreenGen Corp specializes in renewable energy solutions and is listed under the CleanTech exchange. Recent trades follow the standard settlement terms T+2. Internal system used: FIN-98374. Server log IP: 192.168.0.45."
Output:
"renewable energy solutions" private risk: none — Business description.
"CleanTech exchange" private risk: none — Public information.
"standard settlement terms T+2" private risk: none — General financial term.
"FIN-98374" private risk: medium — Internal financial identifier.
"192.168.0.45" private risk: medium — Internal IP address.

Example 2 (Law)
User Query: "How is the 2020 employment law enforced in practice?"
Passage: "One example is the 2021 case involving Jane Smith and TechCorp, where Ms. Smith was terminated after reporting harassment, triggering a legal dispute resolved under the new statute."
Output:
"2021 case involving Jane Smith" private risk: high — Personally identifiable legal case.
"Ms. Smith was terminated after reporting harassment" private risk: high — Sensitive employment and harassment detail.
"resolved under the new statute" private risk: none — Legal reference.

Example 3 (Technical)
User Query: "Why is my OpenAI API call failing with a 403 error?"
Passage: "A StackOverflow thread suggests checking credentials. One user shared: 'Try setting the key as sk-4r1y2xABCDE12345' and make sure your request header includes proper content-type."
Output:
"StackOverflow thread suggests checking credentials" private risk: none — General technical advice.
"sk-4r1y2xABCDE12345" private risk: high — Leaked API key.
"request header includes proper content-type" private risk: none — Common implementation detail.

Input Passage: "{passage}"
Query: "{query}"
Output: List of spans with privacy risk levels and reasons."""
        
        return call_openai(prompt)
    
    def _cot_detection(self, query: str, passage: str) -> str:
        """CoT (Chain-of-Thought) detection strategy"""
        prompt = f"""Perform step-by-step contextual privacy reasoning using the theory of contextual integrity. If the recipient is not explicitly mentioned in the query, assume the information is being shared publicly.

Steps:
1. Identify contextual roles from query:
   • Who is the sender of the information?
   • Who is the recipient? (If unspecified, assume public)
   • Who is the subject of the information?
   • What is the purpose or instruction for the communication?

2. Analyze each information span:
   • What is the data type (e.g., health, employment, relationship)?
   • Is this information sensitive in this context?
   • Would the recipient (or public) be expected to know this information?
   • Could sharing this violate legal, social, or professional privacy norms?

3. Apply privacy reasoning:
   • Does the information flow respect contextual privacy norms?
   • Is this transmission justified by the instruction?
   • Would a reasonable person expect this information to be withheld?

4. Make a determination for each span:
   • Span → Reasoning → Label with privacy risk: high, medium, low, or none with a short explanation.

Privacy Risk Levels:
• high: Highly sensitive personal data, credentials, precise locations, confidential communications
• medium: Personal identifiers, contact information, internal systems, contextually sensitive data
• low: Semi-public information that could be sensitive in certain contexts
• none: Public information, general knowledge, non-identifying data

Example:
• "Lunch with TechAdvance Recruiter" → private risk: medium: Job-seeking activity, not expected by manager, sensitive in workplace.
• "Weekly Team Sync-Up" → private risk: none: Routine work meeting, expected in a work report.

Input Passage: "{passage}"
User Query: "{query}"
Output: List of spans with privacy risk levels and reasons."""
        
        return call_openai(prompt) 