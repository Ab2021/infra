"""
Detokenization utilities for building coverage system.

This module provides token management and detokenization utilities
for handling tokenized data in the building coverage pipeline.
"""

import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class TokenManager:
    """
    Token management for sensitive data handling.
    
    This class provides tokenization and detokenization capabilities
    for protecting sensitive information in claim data.
    """
    
    def __init__(self):
        """
        Initialize the token manager.
        """
        self.token_mappings = {}
        self.reverse_mappings = {}
        self.token_counter = 0
        self.token_prefix = "TOK_"
        
        logger.info("TokenManager initialized")
    
    def tokenize_claim_number(self, claim_number: str) -> str:
        """
        Tokenize a claim number for privacy protection.
        
        Args:
            claim_number (str): Original claim number
            
        Returns:
            str: Tokenized claim number
        """
        if claim_number in self.token_mappings:
            return self.token_mappings[claim_number]
        
        token = self._generate_token()
        self.token_mappings[claim_number] = token
        self.reverse_mappings[token] = claim_number
        
        logger.debug(f"Tokenized claim number: {claim_number[:4]}... -> {token}")
        return token
    
    def detokenize_claim_number(self, token: str) -> str:
        """
        Detokenize a claim number.
        
        Args:
            token (str): Tokenized claim number
            
        Returns:
            str: Original claim number
        """
        if token not in self.reverse_mappings:
            raise ValueError(f"Token not found: {token}")
        
        original = self.reverse_mappings[token]
        logger.debug(f"Detokenized claim number: {token} -> {original[:4]}...")
        return original
    
    def tokenize_text_content(self, text: str, preserve_structure: bool = True) -> str:
        """
        Tokenize sensitive content in text while preserving structure.
        
        Args:
            text (str): Original text
            preserve_structure (bool): Whether to preserve text structure
            
        Returns:
            str: Text with sensitive content tokenized
        """
        if not text:
            return text
        
        tokenized_text = text
        
        # Tokenize claim numbers
        claim_pattern = r'\b[A-Z]{2,4}\d{6,10}\b'
        for match in re.finditer(claim_pattern, text):
            claim_num = match.group()
            token = self.tokenize_claim_number(claim_num)
            tokenized_text = tokenized_text.replace(claim_num, token)
        
        # Tokenize policy numbers
        policy_pattern = r'\b[A-Z]{1,3}\d{8,12}\b'
        for match in re.finditer(policy_pattern, text):
            policy_num = match.group()
            if policy_num not in self.token_mappings:
                token = self._generate_token()
                self.token_mappings[policy_num] = token
                self.reverse_mappings[token] = policy_num
            else:
                token = self.token_mappings[policy_num]
            tokenized_text = tokenized_text.replace(policy_num, token)
        
        # Tokenize SSNs
        ssn_pattern = r'\b\d{3}-?\d{2}-?\d{4}\b'
        for match in re.finditer(ssn_pattern, text):
            ssn = match.group()
            if ssn not in self.token_mappings:
                token = self._generate_token()
                self.token_mappings[ssn] = token
                self.reverse_mappings[token] = ssn
            else:
                token = self.token_mappings[ssn]
            tokenized_text = tokenized_text.replace(ssn, token)
        
        # Tokenize phone numbers
        phone_pattern = r'\b\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            phone = match.group()
            if phone not in self.token_mappings:
                token = self._generate_token()
                self.token_mappings[phone] = token
                self.reverse_mappings[token] = phone
            else:
                token = self.token_mappings[phone]
            tokenized_text = tokenized_text.replace(phone, token)
        
        return tokenized_text
    
    def detokenize_text_content(self, tokenized_text: str) -> str:
        """
        Detokenize text content.
        
        Args:
            tokenized_text (str): Text with tokens
            
        Returns:
            str: Text with original content restored
        """
        if not tokenized_text:
            return tokenized_text
        
        detokenized_text = tokenized_text
        
        # Replace all tokens with original values
        for token, original in self.reverse_mappings.items():
            if token in detokenized_text:
                detokenized_text = detokenized_text.replace(token, original)
        
        return detokenized_text
    
    def tokenize_dataframe(
        self,
        df: pd.DataFrame,
        columns_to_tokenize: List[str]
    ) -> pd.DataFrame:
        """
        Tokenize specified columns in a dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns_to_tokenize (List[str]): Column names to tokenize
            
        Returns:
            pd.DataFrame: Dataframe with tokenized columns
        """
        tokenized_df = df.copy()
        
        for column in columns_to_tokenize:
            if column in tokenized_df.columns:
                if column.upper() in ['CLAIMNO', 'CLAIM_NUMBER', 'CLAIMKEY']:
                    # Tokenize claim numbers
                    tokenized_df[column] = tokenized_df[column].apply(
                        lambda x: self.tokenize_claim_number(str(x)) if pd.notna(x) else x
                    )
                else:
                    # Tokenize text content
                    tokenized_df[column] = tokenized_df[column].apply(
                        lambda x: self.tokenize_text_content(str(x)) if pd.notna(x) else x
                    )
                
                logger.info(f"Tokenized column '{column}' in dataframe")
        
        return tokenized_df
    
    def detokenize_dataframe(
        self,
        tokenized_df: pd.DataFrame,
        columns_to_detokenize: List[str]
    ) -> pd.DataFrame:
        """
        Detokenize specified columns in a dataframe.
        
        Args:
            tokenized_df (pd.DataFrame): Dataframe with tokenized columns
            columns_to_detokenize (List[str]): Column names to detokenize
            
        Returns:
            pd.DataFrame: Dataframe with detokenized columns
        """
        detokenized_df = tokenized_df.copy()
        
        for column in columns_to_detokenize:
            if column in detokenized_df.columns:
                if column.upper() in ['CLAIMNO', 'CLAIM_NUMBER', 'CLAIMKEY']:
                    # Detokenize claim numbers
                    detokenized_df[column] = detokenized_df[column].apply(
                        lambda x: self.detokenize_claim_number(str(x)) if pd.notna(x) and str(x).startswith(self.token_prefix) else x
                    )
                else:
                    # Detokenize text content
                    detokenized_df[column] = detokenized_df[column].apply(
                        lambda x: self.detokenize_text_content(str(x)) if pd.notna(x) else x
                    )
                
                logger.info(f"Detokenized column '{column}' in dataframe")
        
        return detokenized_df
    
    def _generate_token(self) -> str:
        """
        Generate a new unique token.
        
        Returns:
            str: Unique token
        """
        self.token_counter += 1
        return f"{self.token_prefix}{self.token_counter:08d}"
    
    def export_token_mappings(self) -> Dict[str, str]:
        """
        Export token mappings for storage or transfer.
        
        Returns:
            Dict[str, str]: Token mappings dictionary
        """
        return self.token_mappings.copy()
    
    def import_token_mappings(self, mappings: Dict[str, str]) -> None:
        """
        Import token mappings from external source.
        
        Args:
            mappings (Dict[str, str]): Token mappings to import
        """
        self.token_mappings.update(mappings)
        
        # Update reverse mappings
        for original, token in mappings.items():
            self.reverse_mappings[token] = original
        
        # Update counter to avoid conflicts
        token_numbers = []
        for token in mappings.values():
            if token.startswith(self.token_prefix):
                try:
                    num = int(token[len(self.token_prefix):])
                    token_numbers.append(num)
                except ValueError:
                    continue
        
        if token_numbers:
            self.token_counter = max(token_numbers)
        
        logger.info(f"Imported {len(mappings)} token mappings")
    
    def clear_mappings(self) -> None:
        """
        Clear all token mappings.
        """
        self.token_mappings.clear()
        self.reverse_mappings.clear()
        self.token_counter = 0
        
        logger.info("Cleared all token mappings")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get tokenization statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        return {
            'total_tokens': len(self.token_mappings),
            'token_counter': self.token_counter,
            'mapping_types': self._analyze_mapping_types()
        }
    
    def _analyze_mapping_types(self) -> Dict[str, int]:
        """
        Analyze types of data that have been tokenized.
        
        Returns:
            Dict[str, int]: Count of each data type
        """
        types = {
            'claim_numbers': 0,
            'policy_numbers': 0,
            'ssns': 0,
            'phone_numbers': 0,
            'other': 0
        }
        
        for original in self.token_mappings.keys():
            if re.match(r'^[A-Z]{2,4}\d{6,10}$', original):
                types['claim_numbers'] += 1
            elif re.match(r'^[A-Z]{1,3}\d{8,12}$', original):
                types['policy_numbers'] += 1
            elif re.match(r'^\d{3}-?\d{2}-?\d{4}$', original):
                types['ssns'] += 1
            elif re.match(r'^\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}$', original):
                types['phone_numbers'] += 1
            else:
                types['other'] += 1
        
        return types


class TextRedactor:
    """
    Text redaction utilities for removing or masking sensitive information.
    
    This class provides methods for identifying and redacting sensitive
    information from claim text while preserving data utility.
    """
    
    def __init__(self):
        """
        Initialize the text redactor.
        """
        self.redaction_patterns = self._load_redaction_patterns()
        
        logger.info("TextRedactor initialized")
    
    def redact_sensitive_info(
        self,
        text: str,
        redaction_char: str = 'X',
        preserve_length: bool = True
    ) -> str:
        """
        Redact sensitive information from text.
        
        Args:
            text (str): Original text
            redaction_char (str): Character to use for redaction
            preserve_length (bool): Whether to preserve original length
            
        Returns:
            str: Text with sensitive information redacted
        """
        if not text:
            return text
        
        redacted_text = text
        
        for pattern_name, pattern_config in self.redaction_patterns.items():
            pattern = pattern_config['pattern']
            replacement = pattern_config['replacement']
            
            if preserve_length and replacement == '[REDACTED]':
                # Replace with X's of same length
                for match in re.finditer(pattern, redacted_text):
                    matched_text = match.group()
                    redacted_replacement = redaction_char * len(matched_text)
                    redacted_text = redacted_text.replace(matched_text, redacted_replacement, 1)
            else:
                redacted_text = re.sub(pattern, replacement, redacted_text)
        
        return redacted_text
    
    def identify_sensitive_info(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify sensitive information in text without redacting.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[Dict[str, Any]]: List of identified sensitive information
        """
        if not text:
            return []
        
        identified = []
        
        for pattern_name, pattern_config in self.redaction_patterns.items():
            pattern = pattern_config['pattern']
            
            for match in re.finditer(pattern, text):
                identified.append({
                    'type': pattern_name,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': pattern_config.get('confidence', 0.9)
                })
        
        return identified
    
    def partial_redact(
        self,
        text: str,
        show_first: int = 2,
        show_last: int = 2,
        redaction_char: str = 'X'
    ) -> str:
        """
        Partially redact text, showing only first and last characters.
        
        Args:
            text (str): Text to redact
            show_first (int): Number of characters to show at beginning
            show_last (int): Number of characters to show at end
            redaction_char (str): Character to use for redaction
            
        Returns:
            str: Partially redacted text
        """
        if not text or len(text) <= show_first + show_last:
            return redaction_char * len(text)
        
        first_part = text[:show_first]
        last_part = text[-show_last:]
        middle_length = len(text) - show_first - show_last
        middle_part = redaction_char * middle_length
        
        return first_part + middle_part + last_part
    
    def _load_redaction_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Load patterns for identifying sensitive information.
        
        Returns:
            Dict[str, Dict[str, Any]]: Redaction patterns
        """
        return {
            'ssn': {
                'pattern': r'\b\d{3}-?\d{2}-?\d{4}\b',
                'replacement': '[SSN_REDACTED]',
                'confidence': 0.95
            },
            'credit_card': {
                'pattern': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                'replacement': '[CC_REDACTED]',
                'confidence': 0.9
            },
            'phone': {
                'pattern': r'\b\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
                'replacement': '[PHONE_REDACTED]',
                'confidence': 0.85
            },
            'email': {
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'replacement': '[EMAIL_REDACTED]',
                'confidence': 0.9
            },
            'address': {
                'pattern': r'\b\d+\s+[A-Za-z0-9\s,]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
                'replacement': '[ADDRESS_REDACTED]',
                'confidence': 0.8
            },
            'date_of_birth': {
                'pattern': r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b',
                'replacement': '[DOB_REDACTED]',
                'confidence': 0.85
            }
        }


def create_token_manager() -> TokenManager:
    """
    Factory function to create a token manager.
    
    Returns:
        TokenManager: Token manager instance
    """
    return TokenManager()


def create_text_redactor() -> TextRedactor:
    """
    Factory function to create a text redactor.
    
    Returns:
        TextRedactor: Text redactor instance
    """
    return TextRedactor()


def quick_tokenize_claim_numbers(claim_numbers: List[str]) -> Dict[str, str]:
    """
    Quickly tokenize a list of claim numbers.
    
    Args:
        claim_numbers (List[str]): List of claim numbers
        
    Returns:
        Dict[str, str]: Mapping of original to tokenized claim numbers
    """
    token_manager = TokenManager()
    
    mappings = {}
    for claim_num in claim_numbers:
        if claim_num:
            token = token_manager.tokenize_claim_number(claim_num)
            mappings[claim_num] = token
    
    return mappings


def quick_redact_text(text: str, preserve_length: bool = True) -> str:
    """
    Quickly redact sensitive information from text.
    
    Args:
        text (str): Text to redact
        preserve_length (bool): Whether to preserve text length
        
    Returns:
        str: Redacted text
    """
    redactor = TextRedactor()
    return redactor.redact_sensitive_info(text, preserve_length=preserve_length)