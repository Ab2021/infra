"""
Modular components for building coverage system.

This package contains the new modular components that enhance the original
building coverage system with parallel processing and advanced features.
"""

__version__ = "1.0.0"
__author__ = "Insurance AI Team"

from modules.core.pipeline import CoveragePipeline
from modules.core.loader import ConfigLoader
from modules.core.monitor import PerformanceMonitor

__all__ = [
    "CoveragePipeline",
    "ConfigLoader", 
    "PerformanceMonitor"
]