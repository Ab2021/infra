"""
Utilities module for building coverage system.

This module provides utility functions for cryptography, detokenization,
and SQL data warehouse operations used by the original Codebase 1.
"""

from .cryptography import (
    CryptoManager,
    SecureCredentialManager,
    create_crypto_manager,
    create_secure_credential_manager,
    generate_encryption_key,
    hash_claim_text,
    secure_compare_strings
)

from .detokenization import (
    TokenManager,
    TextRedactor,
    create_token_manager,
    create_text_redactor,
    quick_tokenize_claim_numbers,
    quick_redact_text
)

from .sql_data_warehouse import (
    SQLDataWarehouse,
    create_sql_data_warehouse,
    optimize_table_performance,
    batch_process_data
)

__version__ = "1.0.0"

__all__ = [
    # Cryptography utilities
    'CryptoManager',
    'SecureCredentialManager',
    'create_crypto_manager',
    'create_secure_credential_manager',
    'generate_encryption_key',
    'hash_claim_text',
    'secure_compare_strings',
    
    # Detokenization utilities
    'TokenManager',
    'TextRedactor',
    'create_token_manager',
    'create_text_redactor',
    'quick_tokenize_claim_numbers',
    'quick_redact_text',
    
    # SQL data warehouse utilities
    'SQLDataWarehouse',
    'create_sql_data_warehouse',
    'optimize_table_performance',
    'batch_process_data'
]