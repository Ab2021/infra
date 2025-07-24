"""
Text chunking utilities for RAG processing.

This module provides functions for splitting text into chunks
while preserving sentence boundaries and maintaining context.
"""

import re
from typing import List, Dict, Any, Optional


def split_text_into_chunks(
    text: str, 
    chunking_params: Dict[str, Any]
) -> List[str]:
    """
    Split text into chunks based on specified parameters.
    
    Args:
        text (str): Text to split
        chunking_params (Dict[str, Any]): Chunking configuration
        
    Returns:
        List[str]: List of text chunks
    """
    if not text or not text.strip():
        return []
    
    chunk_size = chunking_params.get('chunk_size', 1500)
    chunk_overlap = chunking_params.get('chunk_overlap', 150)
    separators = chunking_params.get('separators', ["\n\n", "\n", ". ", "; ", ", ", " "])
    keep_separator = chunking_params.get('keep_separator', True)
    min_chunk_size = chunking_params.get('min_chunk_size', 100)
    max_chunks = chunking_params.get('max_chunks_per_claim', 10)
    prioritize_sentences = chunking_params.get('prioritize_sentences', True)
    
    # If text is small enough, return as single chunk
    if len(text) <= chunk_size:
        return [text]
    
    # Use sentence-aware chunking if enabled
    if prioritize_sentences:
        chunks = _split_with_sentence_awareness(
            text, chunk_size, chunk_overlap, min_chunk_size
        )
    else:
        chunks = _split_with_separators(
            text, chunk_size, chunk_overlap, separators, keep_separator
        )
    
    # Filter out chunks that are too small
    chunks = [chunk for chunk in chunks if len(chunk.strip()) >= min_chunk_size]
    
    # Limit number of chunks
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
    
    return chunks


def _split_with_sentence_awareness(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int
) -> List[str]:
    """
    Split text into chunks while trying to preserve sentence boundaries.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum chunk size
        chunk_overlap (int): Overlap between chunks
        min_chunk_size (int): Minimum chunk size
        
    Returns:
        List[str]: List of text chunks
    """
    # Split into sentences
    sentences = _split_into_sentences(text)
    
    if not sentences:
        return [text]  # Fallback to original text
    
    chunks = []
    current_chunk = ""
    current_length = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_length = len(sentence)
        
        # If single sentence exceeds chunk size, split it
        if sentence_length > chunk_size:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_length = 0
            
            # Split the long sentence
            sentence_chunks = _split_long_sentence(sentence, chunk_size)
            chunks.extend(sentence_chunks)
            
            i += 1
            continue
        
        # Check if adding this sentence would exceed chunk size
        if current_length + sentence_length > chunk_size and current_chunk.strip():
            # Save current chunk
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            if chunk_overlap > 0:
                overlap_text = _get_overlap_text(current_chunk, chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                current_chunk = sentence
                current_length = sentence_length
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_length = len(current_chunk)
        
        i += 1
    
    # Add remaining content
    if current_chunk.strip() and len(current_chunk.strip()) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex patterns.
    
    Args:
        text (str): Text to split
        
    Returns:
        List[str]: List of sentences
    """
    # Enhanced sentence splitting pattern
    # Matches periods, exclamation marks, question marks followed by whitespace or end of string
    # Avoids splitting on common abbreviations
    sentence_pattern = r'(?<![A-Z][a-z]\.)|(?<![A-Z]\.)(?<![0-9]\.)(?<![a-z]\.)(?<=[.!?])\s+(?=[A-Z])'
    
    # Split into potential sentences
    potential_sentences = re.split(sentence_pattern, text)
    
    # Clean and filter sentences
    sentences = []
    for sentence in potential_sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Filter very short fragments
            sentences.append(sentence)
    
    # If regex splitting didn't work well, fall back to simple split
    if len(sentences) <= 1 and len(text) > 100:
        sentences = _simple_sentence_split(text)
    
    return sentences


def _simple_sentence_split(text: str) -> List[str]:
    """
    Simple sentence splitting as fallback.
    
    Args:
        text (str): Text to split
        
    Returns:
        List[str]: List of sentences
    """
    # Split on common sentence endings
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Clean and filter
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences if cleaned_sentences else [text]


def _split_long_sentence(sentence: str, chunk_size: int) -> List[str]:
    """
    Split a sentence that's too long into smaller chunks.
    
    Args:
        sentence (str): Long sentence to split
        chunk_size (int): Maximum chunk size
        
    Returns:
        List[str]: List of sentence chunks
    """
    if len(sentence) <= chunk_size:
        return [sentence]
    
    chunks = []
    words = sentence.split()
    current_chunk = ""
    
    for word in words:
        if len(current_chunk + " " + word) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = word
        else:
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _get_overlap_text(text: str, overlap_size: int) -> str:
    """
    Get overlap text from the end of a chunk.
    
    Args:
        text (str): Source text
        overlap_size (int): Number of characters for overlap
        
    Returns:
        str: Overlap text
    """
    if len(text) <= overlap_size:
        return text
    
    # Try to get overlap at word boundary
    overlap_text = text[-overlap_size:]
    
    # Find the first space to avoid cutting words
    first_space = overlap_text.find(' ')
    if first_space > 0:
        overlap_text = overlap_text[first_space:].strip()
    
    return overlap_text


def _split_with_separators(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
    keep_separator: bool
) -> List[str]:
    """
    Split text using hierarchical separators.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum chunk size
        chunk_overlap (int): Overlap between chunks
        separators (List[str]): List of separators in order of preference
        keep_separator (bool): Whether to keep separators in chunks
        
    Returns:
        List[str]: List of text chunks
    """
    # Start with the full text
    splits = [text]
    
    # Apply separators hierarchically
    for separator in separators:
        new_splits = []
        
        for split in splits:
            if len(split) <= chunk_size:
                new_splits.append(split)
            else:
                # Split this chunk further
                if separator == " ":
                    # Special handling for word-level splitting
                    sub_chunks = _split_by_words(split, chunk_size, chunk_overlap)
                else:
                    sub_chunks = _split_by_separator(
                        split, separator, chunk_size, keep_separator
                    )
                new_splits.extend(sub_chunks)
        
        splits = new_splits
        
        # Check if all chunks are small enough
        if all(len(chunk) <= chunk_size for chunk in splits):
            break
    
    return [chunk.strip() for chunk in splits if chunk.strip()]


def _split_by_separator(
    text: str,
    separator: str,
    chunk_size: int,
    keep_separator: bool
) -> List[str]:
    """
    Split text by a specific separator.
    
    Args:
        text (str): Text to split
        separator (str): Separator to use
        chunk_size (int): Maximum chunk size
        keep_separator (bool): Whether to keep separator
        
    Returns:
        List[str]: List of chunks
    """
    if separator not in text:
        return [text]
    
    parts = text.split(separator)
    chunks = []
    current_chunk = ""
    
    for i, part in enumerate(parts):
        # Add separator back if keeping it and not the last part
        if keep_separator and i < len(parts) - 1:
            part_with_sep = part + separator
        else:
            part_with_sep = part
        
        if len(current_chunk + part_with_sep) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = part_with_sep
        else:
            current_chunk += part_with_sep
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_by_words(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Split text by words with overlap.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum chunk size
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List[str]: List of word-based chunks
    """
    words = text.split()
    chunks = []
    current_chunk_words = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        
        if current_length + word_length > chunk_size and current_chunk_words:
            # Save current chunk
            chunk_text = " ".join(current_chunk_words)
            chunks.append(chunk_text)
            
            # Calculate overlap
            if chunk_overlap > 0:
                overlap_words = _get_overlap_words(current_chunk_words, chunk_overlap)
                current_chunk_words = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk_words)
            else:
                current_chunk_words = [word]
                current_length = word_length
        else:
            current_chunk_words.append(word)
            current_length += word_length
    
    # Add remaining words
    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        chunks.append(chunk_text)
    
    return chunks


def _get_overlap_words(words: List[str], overlap_size: int) -> List[str]:
    """
    Get overlap words from the end of a word list.
    
    Args:
        words (List[str]): List of words
        overlap_size (int): Number of characters for overlap
        
    Returns:
        List[str]: Overlap words
    """
    if not words:
        return []
    
    # Calculate how many words to include for overlap
    overlap_words = []
    current_length = 0
    
    for word in reversed(words):
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length <= overlap_size:
            overlap_words.insert(0, word)
            current_length += word_length
        else:
            break
    
    return overlap_words


def validate_chunks(
    chunks: List[str],
    original_text: str,
    chunking_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate that chunks meet quality criteria.
    
    Args:
        chunks (List[str]): Generated chunks
        original_text (str): Original text
        chunking_params (Dict[str, Any]): Chunking parameters
        
    Returns:
        Dict[str, Any]: Validation results
    """
    chunk_size = chunking_params.get('chunk_size', 1500)
    min_chunk_size = chunking_params.get('min_chunk_size', 100)
    
    results = {
        'is_valid': True,
        'issues': [],
        'statistics': {
            'total_chunks': len(chunks),
            'total_length': sum(len(chunk) for chunk in chunks),
            'avg_chunk_length': 0,
            'min_chunk_length': 0,
            'max_chunk_length': 0,
            'original_length': len(original_text),
            'length_ratio': 0
        }
    }
    
    if not chunks:
        results['is_valid'] = False
        results['issues'].append('No chunks generated')
        return results
    
    # Calculate statistics
    chunk_lengths = [len(chunk) for chunk in chunks]
    results['statistics']['avg_chunk_length'] = sum(chunk_lengths) / len(chunk_lengths)
    results['statistics']['min_chunk_length'] = min(chunk_lengths)
    results['statistics']['max_chunk_length'] = max(chunk_lengths)
    results['statistics']['length_ratio'] = sum(chunk_lengths) / len(original_text)
    
    # Validate chunk sizes
    oversized_chunks = [i for i, length in enumerate(chunk_lengths) if length > chunk_size]
    if oversized_chunks:
        results['is_valid'] = False
        results['issues'].append(f'Chunks exceed size limit: {oversized_chunks}')
    
    undersized_chunks = [i for i, length in enumerate(chunk_lengths) if length < min_chunk_size]
    if undersized_chunks:
        results['issues'].append(f'Chunks below minimum size: {undersized_chunks}')
    
    # Check for empty chunks
    empty_chunks = [i for i, chunk in enumerate(chunks) if not chunk.strip()]
    if empty_chunks:
        results['is_valid'] = False
        results['issues'].append(f'Empty chunks found: {empty_chunks}')
    
    # Check for excessive length expansion
    if results['statistics']['length_ratio'] > 1.5:
        results['issues'].append('Chunking resulted in significant length expansion')
    
    return results


def optimize_chunking_params(
    text: str,
    target_chunk_count: int = 3,
    max_chunk_size: int = 1500
) -> Dict[str, Any]:
    """
    Optimize chunking parameters for a specific text.
    
    Args:
        text (str): Text to optimize for
        target_chunk_count (int): Desired number of chunks
        max_chunk_size (int): Maximum allowed chunk size
        
    Returns:
        Dict[str, Any]: Optimized chunking parameters
    """
    text_length = len(text)
    
    if text_length <= max_chunk_size:
        # Text fits in single chunk
        return {
            'chunk_size': max_chunk_size,
            'chunk_overlap': 0,
            'separators': ["\n\n", "\n", ". ", " "],
            'min_chunk_size': min(100, text_length // 2),
            'estimated_chunks': 1
        }
    
    # Calculate optimal chunk size
    optimal_chunk_size = min(max_chunk_size, text_length // target_chunk_count)
    
    # Calculate overlap (10-15% of chunk size)
    overlap = min(150, optimal_chunk_size // 8)
    
    # Adjust for text characteristics
    sentence_count = len(re.findall(r'[.!?]+', text))
    if sentence_count > target_chunk_count * 3:
        # Lots of sentences, prioritize sentence boundaries
        separators = [". ", "\n\n", "\n", "; ", ", ", " "]
    else:
        # Fewer sentences, use paragraph breaks
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    
    return {
        'chunk_size': optimal_chunk_size,
        'chunk_overlap': overlap,
        'separators': separators,
        'min_chunk_size': max(50, optimal_chunk_size // 10),
        'prioritize_sentences': True,
        'estimated_chunks': target_chunk_count
    }