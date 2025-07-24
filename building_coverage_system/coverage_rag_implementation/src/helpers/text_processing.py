"""
Text processing utilities for building coverage analysis.

This module provides functions for preprocessing claim text,
extracting keywords, and cleaning text data for RAG processing.
"""

import re
import string
from typing import Dict, List, Set, Any
import pandas as pd


def preprocess_claim_text(text: str) -> str:
    """
    Preprocess claim text for analysis.
    
    Args:
        text (str): Raw claim text
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,;:!?()\-$%/]', ' ', text)
    
    # Normalize common insurance abbreviations
    text = _normalize_insurance_terms(text)
    
    # Remove repeated characters (e.g., "aaaaaa" -> "aa")
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    
    # Clean up extra spaces
    text = ' '.join(text.split())
    
    return text.strip()


def _normalize_insurance_terms(text: str) -> str:
    """
    Normalize common insurance abbreviations and terms.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with normalized terms
    """
    # Insurance term mappings
    term_mappings = {
        r'\bbldg\b': 'building',
        r'\bstruc\b': 'structure',
        r'\bdmg\b': 'damage',
        r'\brpr\b': 'repair',
        r'\breprs\b': 'repairs',
        r'\best\b': 'estimate',
        r'\bconstr\b': 'construction',
        r'\bfndn\b': 'foundation',
        r'\bextr\b': 'exterior',
        r'\bintr\b': 'interior',
        r'\bmatrl\b': 'material',
        r'\bmatrls\b': 'materials',
        r'\bstrctr\b': 'structure',
        r'\bstrcrl\b': 'structural',
        r'\brplcmt\b': 'replacement',
        r'\brstrtn\b': 'restoration'
    }
    
    for pattern, replacement in term_mappings.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


def extract_keywords(text: str) -> Dict[str, Any]:
    """
    Extract building-related keywords and calculate relevance scores.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        Dict[str, Any]: Keyword analysis results
    """
    if not isinstance(text, str):
        return _empty_keyword_result()
    
    text_lower = text.lower()
    
    # Define keyword categories with weights
    keyword_categories = {
        'strong_building_indicators': {
            'keywords': [
                'building', 'structure', 'structural', 'foundation', 'roof', 'wall', 'floor',
                'ceiling', 'construction', 'architectural', 'framework', 'load bearing',
                'framing', 'masonry', 'concrete', 'steel beam', 'wooden beam', 'drywall',
                'sheetrock', 'plaster', 'stucco', 'siding', 'brick', 'stone', 'timber'
            ],
            'weight': 3
        },
        'moderate_building_indicators': {
            'keywords': [
                'window', 'door', 'fixture', 'cabinet', 'countertop', 'flooring',
                'tile', 'carpet', 'hardwood', 'insulation', 'electrical system',
                'plumbing system', 'hvac', 'heating', 'cooling', 'ventilation',
                'built-in', 'permanent', 'attached', 'installed'
            ],
            'weight': 2
        },
        'weak_building_indicators': {
            'keywords': [
                'repair', 'replace', 'rebuild', 'reconstruct', 'renovation',
                'improvement', 'upgrade', 'modification', 'alteration',
                'maintenance', 'restoration', 'refurbishment'
            ],
            'weight': 1
        },
        'non_building_indicators': {
            'keywords': [
                'furniture', 'equipment', 'machinery', 'vehicle', 'contents',
                'personal property', 'inventory', 'stock', 'supplies',
                'landscaping', 'lawn', 'garden', 'fence', 'driveway',
                'liability', 'business interruption', 'lost income'
            ],
            'weight': -2
        }
    }
    
    # Count keyword occurrences
    results = {
        'strong_building_indicators': 0,
        'moderate_building_indicators': 0,
        'weak_building_indicators': 0,
        'non_building_indicators': 0,
        'found_keywords': [],
        'total_score': 0
    }
    
    for category, config in keyword_categories.items():
        keywords = config['keywords']
        weight = config['weight']
        
        category_count = 0
        category_keywords = []
        
        for keyword in keywords:
            # Count occurrences (word boundaries to avoid partial matches)
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            
            if matches > 0:
                category_count += matches
                category_keywords.append({
                    'keyword': keyword,
                    'count': matches,
                    'weight': weight
                })
        
        results[category] = category_count
        results['found_keywords'].extend(category_keywords)
        results['total_score'] += category_count * weight
    
    # Calculate additional metrics
    results['building_keyword_density'] = _calculate_keyword_density(text, results['found_keywords'])
    results['confidence_indicator'] = _calculate_confidence_indicator(results)
    
    return results


def _empty_keyword_result() -> Dict[str, Any]:
    """
    Return empty keyword analysis result.
    
    Returns:
        Dict[str, Any]: Empty result structure
    """
    return {
        'strong_building_indicators': 0,
        'moderate_building_indicators': 0,
        'weak_building_indicators': 0,
        'non_building_indicators': 0,
        'found_keywords': [],
        'total_score': 0,
        'building_keyword_density': 0.0,
        'confidence_indicator': 0.0
    }


def _calculate_keyword_density(text: str, found_keywords: List[Dict[str, Any]]) -> float:
    """
    Calculate the density of building keywords in the text.
    
    Args:
        text (str): Original text
        found_keywords (List[Dict[str, Any]]): Found keywords with counts
        
    Returns:
        float: Keyword density (0.0 to 1.0)
    """
    if not text or not found_keywords:
        return 0.0
    
    # Count total words
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Count total keyword occurrences
    total_keyword_occurrences = sum(kw['count'] for kw in found_keywords)
    
    # Calculate density
    density = total_keyword_occurrences / total_words
    
    return min(density, 1.0)  # Cap at 1.0


def _calculate_confidence_indicator(keyword_results: Dict[str, Any]) -> float:
    """
    Calculate a confidence indicator based on keyword analysis.
    
    Args:
        keyword_results (Dict[str, Any]): Keyword analysis results
        
    Returns:
        float: Confidence indicator (0.0 to 1.0)
    """
    total_score = keyword_results['total_score']
    strong_indicators = keyword_results['strong_building_indicators']
    non_building = keyword_results['non_building_indicators']
    
    # Base confidence from total score
    if total_score >= 6:
        base_confidence = 0.9
    elif total_score >= 4:
        base_confidence = 0.8
    elif total_score >= 2:
        base_confidence = 0.6
    elif total_score >= 1:
        base_confidence = 0.4
    elif total_score <= -2:
        base_confidence = 0.1
    else:
        base_confidence = 0.3
    
    # Boost for strong indicators
    if strong_indicators >= 2:
        base_confidence += 0.1
    
    # Penalty for non-building indicators
    if non_building > 0:
        base_confidence -= 0.2
    
    return max(0.0, min(1.0, base_confidence))


def clean_and_validate_text(text: str, min_length: int = 10) -> Dict[str, Any]:
    """
    Clean and validate claim text.
    
    Args:
        text (str): Input text
        min_length (int): Minimum required text length
        
    Returns:
        Dict[str, Any]: Validation results and cleaned text
    """
    result = {
        'original_text': text,
        'cleaned_text': '',
        'is_valid': False,
        'issues': [],
        'original_length': 0,
        'cleaned_length': 0
    }
    
    if not isinstance(text, str):
        result['issues'].append('Text is not a string')
        return result
    
    result['original_length'] = len(text)
    
    # Clean the text
    cleaned = preprocess_claim_text(text)
    result['cleaned_text'] = cleaned
    result['cleaned_length'] = len(cleaned)
    
    # Validate cleaned text
    if len(cleaned) < min_length:
        result['issues'].append(f'Text too short (minimum {min_length} characters)')
    
    # Check for common issues
    if re.search(r'test|lorem ipsum|placeholder', cleaned, re.IGNORECASE):
        result['issues'].append('Text appears to be test/placeholder data')
    
    if len(set(cleaned.lower().split())) < 3:
        result['issues'].append('Text has very low vocabulary diversity')
    
    # Check for excessive repetition
    words = cleaned.lower().split()
    if len(words) > 0:
        most_common_word_count = max(words.count(word) for word in set(words))
        if most_common_word_count > len(words) * 0.5:
            result['issues'].append('Text has excessive word repetition')
    
    result['is_valid'] = len(result['issues']) == 0
    
    return result


def extract_claim_features(text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Extract comprehensive features from claim text.
    
    Args:
        text (str): Claim text
        metadata (Dict[str, Any], optional): Additional claim metadata
        
    Returns:
        Dict[str, Any]: Extracted features
    """
    if metadata is None:
        metadata = {}
    
    # Basic text metrics
    features = {
        'text_length': len(text) if text else 0,
        'word_count': len(text.split()) if text else 0,
        'sentence_count': len(re.findall(r'[.!?]+', text)) if text else 0,
        'avg_word_length': 0,
        'uppercase_ratio': 0,
        'digit_ratio': 0,
        'punctuation_ratio': 0
    }
    
    if text and features['word_count'] > 0:
        words = text.split()
        features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        
        total_chars = len(text)
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / total_chars
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / total_chars
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / total_chars
    
    # Keyword analysis
    keyword_analysis = extract_keywords(text)
    features.update({
        'building_keywords': keyword_analysis,
        'building_score': keyword_analysis['total_score'],
        'has_strong_building_indicators': keyword_analysis['strong_building_indicators'] > 0,
        'building_keyword_density': keyword_analysis['building_keyword_density']
    })
    
    # Text quality indicators
    quality_check = clean_and_validate_text(text)
    features.update({
        'text_quality': {
            'is_valid': quality_check['is_valid'],
            'issues': quality_check['issues'],
            'quality_score': 1.0 if quality_check['is_valid'] else 0.5
        }
    })
    
    # Domain-specific patterns
    damage_patterns = _extract_damage_patterns(text)
    features['damage_patterns'] = damage_patterns
    
    # Metadata features
    if metadata:
        features['metadata'] = {
            'lob_code': metadata.get('LOBCD'),
            'loss_date': metadata.get('LOSSDT'),
            'claim_age_days': _calculate_claim_age(metadata.get('LOSSDT')),
            'has_loss_description': bool(metadata.get('LOSSDESC'))
        }
    
    return features


def _extract_damage_patterns(text: str) -> Dict[str, Any]:
    """
    Extract damage-related patterns from text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        Dict[str, Any]: Damage pattern analysis
    """
    if not text:
        return {'patterns_found': [], 'damage_types': [], 'severity_indicators': []}
    
    text_lower = text.lower()
    
    # Damage type patterns
    damage_types = {
        'water_damage': r'\b(water|flood|leak|moisture|wet|damp)\b',
        'fire_damage': r'\b(fire|burn|smoke|char|flame|ignit)\b',
        'wind_damage': r'\b(wind|storm|hurricane|tornado|gust)\b',
        'structural_damage': r'\b(crack|collapse|settle|shift|buckle|warp)\b',
        'impact_damage': r'\b(hit|strike|impact|collision|crash)\b'
    }
    
    # Severity indicators
    severity_patterns = {
        'severe': r'\b(severe|extensive|major|significant|catastrophic|destroyed|demolished)\b',
        'moderate': r'\b(moderate|substantial|considerable|damaged)\b',
        'minor': r'\b(minor|slight|small|minimal|superficial)\b'
    }
    
    results = {
        'damage_types': [],
        'severity_indicators': [],
        'patterns_found': []
    }
    
    # Check damage types
    for damage_type, pattern in damage_types.items():
        if re.search(pattern, text_lower):
            results['damage_types'].append(damage_type)
            results['patterns_found'].append(f'damage_type:{damage_type}')
    
    # Check severity
    for severity, pattern in severity_patterns.items():
        if re.search(pattern, text_lower):
            results['severity_indicators'].append(severity)
            results['patterns_found'].append(f'severity:{severity}')
    
    return results


def _calculate_claim_age(loss_date) -> int:
    """
    Calculate claim age in days.
    
    Args:
        loss_date: Loss date (various formats)
        
    Returns:
        int: Age in days, or -1 if cannot calculate
    """
    try:
        if pd.isna(loss_date):
            return -1
        
        if isinstance(loss_date, str):
            loss_date = pd.to_datetime(loss_date)
        
        today = pd.Timestamp.now()
        age_days = (today - loss_date).days
        
        return max(0, age_days)
        
    except Exception:
        return -1


def batch_preprocess_texts(texts: List[str], show_progress: bool = False) -> List[str]:
    """
    Preprocess multiple texts in batch.
    
    Args:
        texts (List[str]): List of texts to preprocess
        show_progress (bool): Whether to show progress
        
    Returns:
        List[str]: List of preprocessed texts
    """
    results = []
    
    for i, text in enumerate(texts):
        if show_progress and i % 100 == 0:
            print(f"Processing text {i+1}/{len(texts)}")
        
        processed = preprocess_claim_text(text)
        results.append(processed)
    
    return results