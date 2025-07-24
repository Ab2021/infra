"""
Storage components.

This module contains components for storing processed data to multiple
destinations including SQL warehouses, Snowflake, and local storage.
"""

from .multi_writer import MultiWriter

__all__ = ["MultiWriter"]