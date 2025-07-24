"""
Parallel processing components.

This module contains components for parallel processing of claims data
including multi-threaded RAG processing and text analysis.
"""

from .parallel_rag import ParallelRAGProcessor

__all__ = ["ParallelRAGProcessor"]