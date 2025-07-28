"""
Text Processing Utilities for Agentic Building Coverage Analysis
Essential methods extracted from TextProcessor for core text preprocessing
"""

import re
import string
from typing import List, Set, Dict, Any
from collections import Counter


class TextProcessor:
    """Core text preprocessing utilities for building coverage analysis"""
    
    def __init__(self):
        # Insurance-specific keywords for filtering
        self.insurance_keywords = {
            'damage', 'loss', 'claim', 'coverage', 'policy', 'deductible',
            'repair', 'replace', 'restore', 'water', 'fire', 'wind', 'hail',
            'flood', 'storm', 'lightning', 'vandalism', 'theft', 'burst',
            'leak', 'roof', 'wall', 'floor', 'ceiling', 'window', 'door',
            'electrical', 'plumbing', 'hvac', 'structure', 'building',
            'commercial', 'residential', 'property', 'premises', 'tenant',
            'occupancy', 'business', 'equipment', 'inventory', 'contents',
            'interior', 'exterior', 'foundation', 'basement', 'attic',
            'garage', 'kitchen', 'bathroom', 'bedroom', 'office',
            'apartment', 'condo', 'house', 'warehouse', 'factory'
        }
        
        # Stopwords for cleaning
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'among', 'this', 'that', 'these', 'those', 'i',
            'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for processing"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation that matters
        text = re.sub(r'[^\w\s.,;:!?-]', ' ', text)
        
        # Normalize common insurance terms
        text = self._normalize_insurance_terms(text)
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    def _normalize_insurance_terms(self, text: str) -> str:
        """Normalize common insurance terminology"""
        # Common replacements
        replacements = {
            r'\bdmg\b': 'damage',
            r'\bbldg\b': 'building', 
            r'\bwtr\b': 'water',
            r'\belectr\b': 'electrical',
            r'\bequip\b': 'equipment',
            r'\bcomm\b': 'commercial',
            r'\bres\b': 'residential',
            r'\bapt\b': 'apartment',
            r'\bcondo\b': 'condominium',
            r'\bhvac\b': 'heating ventilation air conditioning'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract and clean sentences from text"""
        if not text:
            return []
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def deduplicate_sentences(self, sentences: List[str]) -> List[str]:
        """Remove duplicate and near-duplicate sentences"""
        if not sentences:
            return []
        
        unique_sentences = []
        seen_hashes = set()
        
        for sentence in sentences:
            # Create hash based on key words
            key_words = self._extract_key_words(sentence)
            sentence_hash = hash(tuple(sorted(key_words)))
            
            if sentence_hash not in seen_hashes:
                seen_hashes.add(sentence_hash)
                unique_sentences.append(sentence)
        
        return unique_sentences
    
    def _extract_key_words(self, text: str) -> List[str]:
        """Extract key words from text (non-stopwords)"""
        words = text.lower().split()
        key_words = []
        
        for word in words:
            # Remove punctuation
            word = word.strip(string.punctuation)
            
            # Skip if stopword or too short
            if word not in self.stopwords and len(word) > 2:
                key_words.append(word)
        
        return key_words
    
    def filter_relevant_content(self, text: str) -> str:
        """Filter text to keep only insurance-relevant content"""
        if not text:
            return ""
        
        sentences = self.extract_sentences(text)
        relevant_sentences = []
        
        for sentence in sentences:
            if self._is_insurance_relevant(sentence):
                relevant_sentences.append(sentence)
        
        return '. '.join(relevant_sentences)
    
    def _is_insurance_relevant(self, text: str) -> bool:
        """Check if text contains insurance-relevant keywords"""
        text_words = set(text.lower().split())
        
        # Check for intersection with insurance keywords
        intersection = text_words.intersection(self.insurance_keywords)
        
        # Relevant if contains at least 2 insurance keywords
        return len(intersection) >= 2
    
    def extract_monetary_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary values from text"""
        monetary_patterns = [
            r'\$[\d,]+\.?\d*',  # $1,000.00
            r'[\d,]+\.?\d*\s*dollars?',  # 1000 dollars
            r'[\d,]+\.?\d*\s*usd',  # 1000 USD
        ]
        
        monetary_values = []
        
        for pattern in monetary_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value_text = match.group()
                try:
                    # Extract numeric value
                    numeric_value = re.sub(r'[^\d.]', '', value_text)
                    amount = float(numeric_value) if numeric_value else 0
                    
                    monetary_values.append({
                        'text': value_text,
                        'amount': amount,
                        'position': match.span()
                    })
                except ValueError:
                    continue
        
        return monetary_values
    
    def preprocess_for_extraction(self, text: str) -> str:
        """Complete preprocessing pipeline for extraction"""
        # Step 1: Clean text
        cleaned_text = self.clean_text(text)
        
        # Step 2: Filter relevant content
        relevant_text = self.filter_relevant_content(cleaned_text)
        
        # Step 3: Extract and deduplicate sentences
        sentences = self.extract_sentences(relevant_text)
        unique_sentences = self.deduplicate_sentences(sentences)
        
        # Step 4: Rejoin into final text
        final_text = '. '.join(unique_sentences)
        
        return final_text if final_text else cleaned_text
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Get statistics about processed text"""
        if not text:
            return {'word_count': 0, 'sentence_count': 0, 'insurance_keywords': 0}
        
        words = text.split()
        sentences = self.extract_sentences(text)
        
        # Count insurance keywords
        text_words = set(text.lower().split())
        insurance_keyword_count = len(text_words.intersection(self.insurance_keywords))
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'insurance_keywords': insurance_keyword_count,
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'relevance_score': min(1.0, insurance_keyword_count / 10.0)
        }