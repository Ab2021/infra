"""
Cryptography utilities for building coverage system.

This module provides encryption, decryption, and security utilities
for protecting sensitive data in the building coverage pipeline.
"""

import base64
import hashlib
import secrets
from typing import Union, Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import logging
import os

logger = logging.getLogger(__name__)


class CryptoManager:
    """
    Cryptography manager for secure data handling.
    
    This class provides encryption/decryption capabilities for sensitive
    data such as database credentials, API keys, and claim data.
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize the crypto manager.
        
        Args:
            master_key (Optional[str]): Master encryption key. If None, generates new key.
        """
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = self._generate_master_key()
        
        self.fernet = self._create_fernet_cipher()
        
        logger.info("CryptoManager initialized")
    
    def encrypt_string(self, plaintext: str) -> str:
        """
        Encrypt a string value.
        
        Args:
            plaintext (str): String to encrypt
            
        Returns:
            str: Base64 encoded encrypted string
        """
        if not isinstance(plaintext, str):
            raise ValueError("Input must be a string")
        
        try:
            encrypted_bytes = self.fernet.encrypt(plaintext.encode('utf-8'))
            return base64.b64encode(encrypted_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """
        Decrypt a string value.
        
        Args:
            encrypted_text (str): Base64 encoded encrypted string
            
        Returns:
            str: Decrypted plaintext string
        """
        if not isinstance(encrypted_text, str):
            raise ValueError("Input must be a string")
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise
    
    def encrypt_dict(self, data_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        Encrypt sensitive values in a dictionary.
        
        Args:
            data_dict (Dict[str, Any]): Dictionary with values to encrypt
            
        Returns:
            Dict[str, str]: Dictionary with encrypted values
        """
        encrypted_dict = {}
        
        for key, value in data_dict.items():
            if value is not None:
                if isinstance(value, (str, int, float)):
                    encrypted_dict[key] = self.encrypt_string(str(value))
                else:
                    encrypted_dict[key] = self.encrypt_string(str(value))
            else:
                encrypted_dict[key] = None
        
        return encrypted_dict
    
    def decrypt_dict(self, encrypted_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Decrypt values in a dictionary.
        
        Args:
            encrypted_dict (Dict[str, str]): Dictionary with encrypted values
            
        Returns:
            Dict[str, str]: Dictionary with decrypted values
        """
        decrypted_dict = {}
        
        for key, encrypted_value in encrypted_dict.items():
            if encrypted_value is not None:
                try:
                    decrypted_dict[key] = self.decrypt_string(encrypted_value)
                except Exception as e:
                    logger.warning(f"Failed to decrypt key '{key}': {str(e)}")
                    decrypted_dict[key] = None
            else:
                decrypted_dict[key] = None
        
        return decrypted_dict
    
    def hash_string(self, text: str, algorithm: str = 'sha256') -> str:
        """
        Create a hash of a string.
        
        Args:
            text (str): Text to hash
            algorithm (str): Hashing algorithm ('sha256', 'md5', 'sha1')
            
        Returns:
            str: Hexadecimal hash string
        """
        if algorithm == 'sha256':
            hash_obj = hashlib.sha256(text.encode('utf-8'))
        elif algorithm == 'md5':
            hash_obj = hashlib.md5(text.encode('utf-8'))
        elif algorithm == 'sha1':
            hash_obj = hashlib.sha1(text.encode('utf-8'))
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return hash_obj.hexdigest()
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token.
        
        Args:
            length (int): Token length in bytes
            
        Returns:
            str: Base64 encoded secure token
        """
        token_bytes = secrets.token_bytes(length)
        return base64.b64encode(token_bytes).decode('utf-8')
    
    def mask_sensitive_data(self, text: str, mask_char: str = '*') -> str:
        """
        Mask sensitive data in text for logging.
        
        Args:
            text (str): Text containing sensitive data
            mask_char (str): Character to use for masking
            
        Returns:
            str: Text with sensitive data masked
        """
        if len(text) <= 4:
            return mask_char * len(text)
        
        # Show first 2 and last 2 characters, mask the middle
        masked_middle = mask_char * (len(text) - 4)
        return text[:2] + masked_middle + text[-2:]
    
    def _generate_master_key(self) -> bytes:
        """
        Generate a new master encryption key.
        
        Returns:
            bytes: Master key
        """
        return secrets.token_bytes(32)  # 256-bit key
    
    def _create_fernet_cipher(self) -> Fernet:
        """
        Create a Fernet cipher instance.
        
        Returns:
            Fernet: Fernet cipher instance
        """
        # Derive a key from the master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'building_coverage_salt',  # Fixed salt for consistency
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)
    
    def export_key(self) -> str:
        """
        Export the master key as a base64 string.
        
        Returns:
            str: Base64 encoded master key
        """
        return base64.b64encode(self.master_key).decode('utf-8')
    
    @classmethod
    def from_key(cls, key_string: str) -> 'CryptoManager':
        """
        Create a CryptoManager from an exported key string.
        
        Args:
            key_string (str): Base64 encoded key string
            
        Returns:
            CryptoManager: Crypto manager instance
        """
        master_key = base64.b64decode(key_string.encode('utf-8')).decode('utf-8')
        return cls(master_key=master_key)


class SecureCredentialManager:
    """
    Manager for securely storing and retrieving database credentials.
    
    This class provides secure storage of database credentials using
    encryption and environment variable integration.
    """
    
    def __init__(self, crypto_manager: CryptoManager):
        """
        Initialize the credential manager.
        
        Args:
            crypto_manager (CryptoManager): Cryptography manager instance
        """
        self.crypto = crypto_manager
        self.encrypted_credentials = {}
        
        logger.info("SecureCredentialManager initialized")
    
    def store_credentials(
        self,
        credential_name: str,
        credentials: Dict[str, str]
    ) -> None:
        """
        Store encrypted credentials.
        
        Args:
            credential_name (str): Name/identifier for the credentials
            credentials (Dict[str, str]): Credentials dictionary
        """
        try:
            encrypted_creds = self.crypto.encrypt_dict(credentials)
            self.encrypted_credentials[credential_name] = encrypted_creds
            
            logger.info(f"Credentials stored for '{credential_name}'")
            
        except Exception as e:
            logger.error(f"Failed to store credentials for '{credential_name}': {str(e)}")
            raise
    
    def retrieve_credentials(self, credential_name: str) -> Dict[str, str]:
        """
        Retrieve and decrypt credentials.
        
        Args:
            credential_name (str): Name/identifier for the credentials
            
        Returns:
            Dict[str, str]: Decrypted credentials dictionary
        """
        if credential_name not in self.encrypted_credentials:
            raise ValueError(f"Credentials not found for '{credential_name}'")
        
        try:
            encrypted_creds = self.encrypted_credentials[credential_name]
            decrypted_creds = self.crypto.decrypt_dict(encrypted_creds)
            
            logger.debug(f"Credentials retrieved for '{credential_name}'")
            return decrypted_creds
            
        except Exception as e:
            logger.error(f"Failed to retrieve credentials for '{credential_name}': {str(e)}")
            raise
    
    def store_credentials_from_env(
        self,
        credential_name: str,
        env_mapping: Dict[str, str]
    ) -> None:
        """
        Store credentials from environment variables.
        
        Args:
            credential_name (str): Name/identifier for the credentials
            env_mapping (Dict[str, str]): Mapping of credential keys to env var names
        """
        credentials = {}
        
        for cred_key, env_var in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value:
                credentials[cred_key] = env_value
            else:
                logger.warning(f"Environment variable '{env_var}' not found")
        
        if credentials:
            self.store_credentials(credential_name, credentials)
        else:
            raise ValueError(f"No credentials found in environment variables for '{credential_name}'")
    
    def update_credential(
        self,
        credential_name: str,
        key: str,
        value: str
    ) -> None:
        """
        Update a specific credential value.
        
        Args:
            credential_name (str): Name/identifier for the credentials
            key (str): Credential key to update
            value (str): New value
        """
        if credential_name not in self.encrypted_credentials:
            raise ValueError(f"Credentials not found for '{credential_name}'")
        
        # Retrieve, update, and re-store
        credentials = self.retrieve_credentials(credential_name)
        credentials[key] = value
        self.store_credentials(credential_name, credentials)
        
        logger.info(f"Updated credential '{key}' for '{credential_name}'")
    
    def delete_credentials(self, credential_name: str) -> None:
        """
        Delete stored credentials.
        
        Args:
            credential_name (str): Name/identifier for the credentials
        """
        if credential_name in self.encrypted_credentials:
            del self.encrypted_credentials[credential_name]
            logger.info(f"Deleted credentials for '{credential_name}'")
        else:
            logger.warning(f"No credentials found to delete for '{credential_name}'")
    
    def list_stored_credentials(self) -> List[str]:
        """
        List all stored credential names.
        
        Returns:
            List[str]: List of credential names
        """
        return list(self.encrypted_credentials.keys())
    
    def mask_credentials_for_logging(self, credentials: Dict[str, str]) -> Dict[str, str]:
        """
        Mask credentials for safe logging.
        
        Args:
            credentials (Dict[str, str]): Credentials to mask
            
        Returns:
            Dict[str, str]: Masked credentials
        """
        sensitive_keys = {'password', 'pwd', 'pass', 'secret', 'key', 'token'}
        
        masked_creds = {}
        for key, value in credentials.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                masked_creds[key] = self.crypto.mask_sensitive_data(value) if value else None
            else:
                masked_creds[key] = value
        
        return masked_creds


def create_crypto_manager(master_key: Optional[str] = None) -> CryptoManager:
    """
    Factory function to create a crypto manager.
    
    Args:
        master_key (Optional[str]): Master encryption key
        
    Returns:
        CryptoManager: Crypto manager instance
    """
    return CryptoManager(master_key=master_key)


def create_secure_credential_manager(crypto_manager: CryptoManager) -> SecureCredentialManager:
    """
    Factory function to create a secure credential manager.
    
    Args:
        crypto_manager (CryptoManager): Crypto manager instance
        
    Returns:
        SecureCredentialManager: Credential manager instance
    """
    return SecureCredentialManager(crypto_manager)


def generate_encryption_key() -> str:
    """
    Generate a new encryption key for use with the crypto manager.
    
    Returns:
        str: Base64 encoded encryption key
    """
    key_bytes = secrets.token_bytes(32)
    return base64.b64encode(key_bytes).decode('utf-8')


def hash_claim_text(claim_text: str, algorithm: str = 'sha256') -> str:
    """
    Create a hash of claim text for deduplication.
    
    Args:
        claim_text (str): Claim text to hash
        algorithm (str): Hashing algorithm
        
    Returns:
        str: Hexadecimal hash
    """
    crypto = CryptoManager()
    return crypto.hash_string(claim_text, algorithm)


def secure_compare_strings(str1: str, str2: str) -> bool:
    """
    Securely compare two strings to prevent timing attacks.
    
    Args:
        str1 (str): First string
        str2 (str): Second string
        
    Returns:
        bool: True if strings are equal
    """
    return secrets.compare_digest(str1.encode('utf-8'), str2.encode('utf-8'))