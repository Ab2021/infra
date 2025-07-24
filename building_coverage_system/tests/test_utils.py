"""
Tests for utils module.

This module contains unit tests for the utility functions including
cryptography, detokenization, and SQL data warehouse operations.
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from building_coverage_system.utils.cryptography import (
    CryptoManager,
    SecureCredentialManager,
    create_crypto_manager,
    generate_encryption_key,
    hash_claim_text,
    secure_compare_strings
)
from building_coverage_system.utils.detokenization import (
    TokenManager,
    TextRedactor,
    create_token_manager,
    quick_tokenize_claim_numbers,
    quick_redact_text
)
from building_coverage_system.utils.sql_data_warehouse import (
    SQLDataWarehouse,
    create_sql_data_warehouse,
    optimize_table_performance,
    batch_process_data
)
from building_coverage_system.tests.fixtures.sample_data import (
    create_sample_claims_data,
    create_sample_credentials
)


class TestCryptoManager:
    """Test cases for CryptoManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.crypto_manager = CryptoManager()
    
    def test_encrypt_decrypt_string(self):
        """Test string encryption and decryption."""
        plaintext = "This is a test string for encryption"
        
        # Encrypt the string
        encrypted = self.crypto_manager.encrypt_string(plaintext)
        
        # Verify encrypted string is different and not empty
        assert encrypted != plaintext
        assert len(encrypted) > 0
        
        # Decrypt the string
        decrypted = self.crypto_manager.decrypt_string(encrypted)
        
        # Verify decryption matches original
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_dict(self):
        """Test dictionary encryption and decryption."""
        test_dict = {
            'username': 'test_user',
            'password': 'secret_password',
            'api_key': 'abc123def456',
            'number_value': 12345
        }
        
        # Encrypt dictionary
        encrypted_dict = self.crypto_manager.encrypt_dict(test_dict)
        
        # Verify all values are encrypted
        for key, value in encrypted_dict.items():
            if value is not None:
                assert value != str(test_dict[key])
        
        # Decrypt dictionary
        decrypted_dict = self.crypto_manager.decrypt_dict(encrypted_dict)
        
        # Verify decryption matches original (as strings)
        for key in test_dict:
            assert decrypted_dict[key] == str(test_dict[key])
    
    def test_hash_string(self):
        """Test string hashing functionality."""
        test_string = "Test string for hashing"
        
        # Test SHA256 (default)
        hash_sha256 = self.crypto_manager.hash_string(test_string)
        assert len(hash_sha256) == 64  # SHA256 produces 64-char hex string
        
        # Test MD5
        hash_md5 = self.crypto_manager.hash_string(test_string, 'md5')
        assert len(hash_md5) == 32  # MD5 produces 32-char hex string
        
        # Test SHA1
        hash_sha1 = self.crypto_manager.hash_string(test_string, 'sha1')
        assert len(hash_sha1) == 40  # SHA1 produces 40-char hex string
        
        # Same input should produce same hash
        hash_sha256_2 = self.crypto_manager.hash_string(test_string)
        assert hash_sha256 == hash_sha256_2
    
    def test_generate_secure_token(self):
        """Test secure token generation."""
        # Test default length
        token = self.crypto_manager.generate_secure_token()
        assert len(token) > 0
        
        # Test custom length
        token_32 = self.crypto_manager.generate_secure_token(32)
        token_64 = self.crypto_manager.generate_secure_token(64)
        
        # Tokens should be different
        assert token_32 != token_64
        assert len(token_32) != len(token_64)
    
    def test_mask_sensitive_data(self):
        """Test sensitive data masking."""
        # Test short string
        short_string = "ab"
        masked = self.crypto_manager.mask_sensitive_data(short_string)
        assert masked == "**"
        
        # Test normal string
        normal_string = "password123"
        masked = self.crypto_manager.mask_sensitive_data(normal_string)
        assert masked.startswith("pa")
        assert masked.endswith("23")
        assert "*" in masked
    
    def test_export_import_key(self):
        """Test key export and import functionality."""
        # Export key
        exported_key = self.crypto_manager.export_key()
        assert len(exported_key) > 0
        
        # Create new manager with imported key
        new_manager = CryptoManager.from_key(exported_key)
        
        # Test that both managers can encrypt/decrypt consistently
        test_text = "Test encryption consistency"
        
        encrypted_original = self.crypto_manager.encrypt_string(test_text)
        decrypted_new = new_manager.decrypt_string(encrypted_original)
        
        assert decrypted_new == test_text
    
    def test_encryption_errors(self):
        """Test encryption error handling."""
        # Test encrypting non-string
        with pytest.raises(ValueError):
            self.crypto_manager.encrypt_string(123)
        
        # Test decrypting invalid data
        with pytest.raises(Exception):
            self.crypto_manager.decrypt_string("invalid_encrypted_data")


class TestSecureCredentialManager:
    """Test cases for SecureCredentialManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.crypto_manager = CryptoManager()
        self.credential_manager = SecureCredentialManager(self.crypto_manager)
    
    def test_store_retrieve_credentials(self):
        """Test storing and retrieving credentials."""
        test_credentials = {
            'username': 'test_user',
            'password': 'secret_password',
            'server': 'test-server.com',
            'database': 'test_db'
        }
        
        # Store credentials
        self.credential_manager.store_credentials('test_db', test_credentials)
        
        # Retrieve credentials
        retrieved = self.credential_manager.retrieve_credentials('test_db')
        
        # Verify credentials match
        for key, value in test_credentials.items():
            assert retrieved[key] == value
    
    def test_update_credential(self):
        """Test updating individual credential values."""
        test_credentials = {
            'username': 'old_user',
            'password': 'old_password'
        }
        
        # Store initial credentials
        self.credential_manager.store_credentials('update_test', test_credentials)
        
        # Update password
        self.credential_manager.update_credential('update_test', 'password', 'new_password')
        
        # Retrieve and verify update
        updated = self.credential_manager.retrieve_credentials('update_test')
        assert updated['username'] == 'old_user'
        assert updated['password'] == 'new_password'
    
    def test_delete_credentials(self):
        """Test deleting stored credentials."""
        test_credentials = {'username': 'test', 'password': 'test'}
        
        # Store credentials
        self.credential_manager.store_credentials('delete_test', test_credentials)
        
        # Verify they exist
        stored_names = self.credential_manager.list_stored_credentials()
        assert 'delete_test' in stored_names
        
        # Delete credentials
        self.credential_manager.delete_credentials('delete_test')
        
        # Verify they're gone
        stored_names = self.credential_manager.list_stored_credentials()
        assert 'delete_test' not in stored_names
    
    def test_mask_credentials_for_logging(self):
        """Test credential masking for logging."""
        test_credentials = {
            'username': 'test_user',
            'password': 'secret_password',
            'api_key': 'abc123def456',
            'server': 'test-server.com'
        }
        
        masked = self.credential_manager.mask_credentials_for_logging(test_credentials)
        
        # Username and server should not be masked
        assert masked['username'] == 'test_user'
        assert masked['server'] == 'test-server.com'
        
        # Password and API key should be masked
        assert masked['password'] != 'secret_password'
        assert masked['api_key'] != 'abc123def456'
        assert '*' in masked['password']
        assert '*' in masked['api_key']
    
    @patch.dict(os.environ, {
        'DB_USERNAME': 'env_user',
        'DB_PASSWORD': 'env_password',
        'DB_SERVER': 'env_server'
    })
    def test_store_credentials_from_env(self):
        """Test storing credentials from environment variables."""
        env_mapping = {
            'username': 'DB_USERNAME',
            'password': 'DB_PASSWORD',
            'server': 'DB_SERVER'
        }
        
        self.credential_manager.store_credentials_from_env('env_test', env_mapping)
        
        # Retrieve and verify
        retrieved = self.credential_manager.retrieve_credentials('env_test')
        
        assert retrieved['username'] == 'env_user'
        assert retrieved['password'] == 'env_password'
        assert retrieved['server'] == 'env_server'


class TestTokenManager:
    """Test cases for TokenManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.token_manager = TokenManager()
    
    def test_tokenize_detokenize_claim_number(self):
        """Test claim number tokenization and detokenization."""
        claim_number = "CLM123456789"
        
        # Tokenize
        token = self.token_manager.tokenize_claim_number(claim_number)
        
        assert token != claim_number
        assert token.startswith("TOK_")
        
        # Detokenize
        detokenized = self.token_manager.detokenize_claim_number(token)
        
        assert detokenized == claim_number
    
    def test_tokenize_text_content(self):
        """Test text content tokenization."""
        test_text = "Claim CLM123456789 reported by John Doe at 555-123-4567 for policy ABC12345678"
        
        tokenized = self.token_manager.tokenize_text_content(test_text)
        
        # Original claim number should be replaced with token
        assert "CLM123456789" not in tokenized
        assert "TOK_" in tokenized
        
        # Detokenize to verify
        detokenized = self.token_manager.detokenize_text_content(tokenized)
        assert "CLM123456789" in detokenized
    
    def test_tokenize_dataframe(self):
        """Test dataframe tokenization."""
        test_df = pd.DataFrame({
            'CLAIMNO': ['CLM001', 'CLM002', 'CLM003'],
            'description': [
                'Claim CLM001 fire damage',
                'Claim CLM002 water damage', 
                'Claim CLM003 wind damage'
            ],
            'amount': [1000, 2000, 3000]
        })
        
        tokenized_df = self.token_manager.tokenize_dataframe(
            test_df, 
            ['CLAIMNO', 'description']
        )
        
        # Claim numbers should be tokenized
        assert all(token.startswith('TOK_') for token in tokenized_df['CLAIMNO'])
        
        # Descriptions should have tokenized claim numbers
        assert all('TOK_' in desc for desc in tokenized_df['description'])
        
        # Amount column should be unchanged
        assert tokenized_df['amount'].equals(test_df['amount'])
    
    def test_export_import_token_mappings(self):
        """Test token mapping export and import."""
        # Create some tokens
        claim_numbers = ['CLM001', 'CLM002', 'CLM003']
        tokens = [self.token_manager.tokenize_claim_number(cn) for cn in claim_numbers]
        
        # Export mappings
        exported_mappings = self.token_manager.export_token_mappings()
        
        assert len(exported_mappings) == 3
        for claim_num in claim_numbers:
            assert claim_num in exported_mappings
        
        # Create new manager and import mappings
        new_manager = TokenManager()
        new_manager.import_token_mappings(exported_mappings)
        
        # Verify imported mappings work
        for i, token in enumerate(tokens):
            detokenized = new_manager.detokenize_claim_number(token)
            assert detokenized == claim_numbers[i]
    
    def test_get_statistics(self):
        """Test tokenization statistics."""
        # Create various types of tokens
        self.token_manager.tokenize_claim_number('CLM123456789')
        self.token_manager.tokenize_text_content('Call 555-123-4567 about policy ABC12345678')
        
        stats = self.token_manager.get_statistics()
        
        assert 'total_tokens' in stats
        assert 'token_counter' in stats
        assert 'mapping_types' in stats
        assert stats['total_tokens'] > 0


class TestTextRedactor:
    """Test cases for TextRedactor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.text_redactor = TextRedactor()
    
    def test_redact_sensitive_info(self):
        """Test sensitive information redaction."""
        test_text = "Contact John Doe at john.doe@email.com or 555-123-4567. SSN: 123-45-6789"
        
        redacted = self.text_redactor.redact_sensitive_info(test_text)
        
        # Email, phone, and SSN should be redacted
        assert "john.doe@email.com" not in redacted
        assert "555-123-4567" not in redacted
        assert "123-45-6789" not in redacted
        
        # Redaction markers should be present
        assert "[EMAIL_REDACTED]" in redacted
        assert "[PHONE_REDACTED]" in redacted
        assert "[SSN_REDACTED]" in redacted
    
    def test_identify_sensitive_info(self):
        """Test identifying sensitive information without redacting."""
        test_text = "Contact: email@test.com, phone: 555-123-4567, SSN: 123-45-6789"
        
        identified = self.text_redactor.identify_sensitive_info(test_text)
        
        assert len(identified) >= 3  # Should find email, phone, SSN
        
        # Check that each identified item has required fields
        for item in identified:
            assert 'type' in item
            assert 'value' in item
            assert 'start' in item
            assert 'end' in item
            assert 'confidence' in item
    
    def test_partial_redact(self):
        """Test partial redaction functionality."""
        test_text = "password123"
        
        # Default partial redaction (show first 2, last 2)
        partial = self.text_redactor.partial_redact(test_text)
        
        assert partial.startswith("pa")
        assert partial.endswith("23")
        assert "X" in partial
        
        # Custom partial redaction
        custom_partial = self.text_redactor.partial_redact(test_text, show_first=1, show_last=1)
        
        assert custom_partial.startswith("p")
        assert custom_partial.endswith("3")


class TestSQLDataWarehouse:
    """Test cases for SQLDataWarehouse class."""
    
    @patch('building_coverage_system.utils.sql_data_warehouse.sqlalchemy.create_engine')
    def setup_method(self, mock_create_engine):
        """Set up test fixtures with mocked database connection."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        self.sample_credentials = create_sample_credentials()
        self.warehouse = SQLDataWarehouse(self.sample_credentials)
    
    def test_warehouse_initialization(self):
        """Test warehouse initialization."""
        assert self.warehouse.credentials == self.sample_credentials
        assert self.warehouse.performance_stats['total_operations'] == 0
    
    @patch('building_coverage_system.utils.sql_data_warehouse.ThreadPoolExecutor')
    def test_batch_process_claims(self, mock_executor):
        """Test batch processing of claims data."""
        # Setup mock executor
        mock_future = Mock()
        mock_future.result.return_value = {
            'success': True,
            'rows_processed': 100,
            'processing_time': 1.0,
            'error': None
        }
        
        mock_executor_instance = Mock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor.return_value = mock_executor_instance
        
        # Create sample claims data
        claims_df = create_sample_claims_data(500)
        
        # Mock the batch processing methods
        with patch.object(self.warehouse, '_create_batches') as mock_create_batches, \
             patch.object(self.warehouse, '_process_batches_parallel') as mock_process_batches:
            
            mock_create_batches.return_value = [claims_df[:100], claims_df[100:200]]
            mock_process_batches.return_value = [
                {'success': True, 'rows_processed': 100},
                {'success': True, 'rows_processed': 100}
            ]
            
            result = self.warehouse.batch_process_claims(claims_df, batch_size=100)
            
            assert 'total_claims' in result
            assert 'total_processed' in result
            assert 'processing_time' in result
            assert result['total_claims'] == len(claims_df)
    
    def test_get_performance_statistics(self):
        """Test performance statistics retrieval."""
        # Update some stats manually for testing
        self.warehouse.performance_stats['total_operations'] = 10
        self.warehouse.performance_stats['successful_operations'] = 8
        self.warehouse.performance_stats['total_processing_time'] = 50.0
        self.warehouse.performance_stats['total_rows_processed'] = 1000
        
        stats = self.warehouse.get_performance_statistics()
        
        assert stats['total_operations'] == 10
        assert stats['success_rate'] == 0.8
        assert stats['avg_processing_time'] == 5.0
        assert stats['avg_rows_per_operation'] == 125.0
    
    @patch('building_coverage_system.utils.sql_data_warehouse.sqlalchemy.text')
    def test_execute_bulk_operations(self, mock_text):
        """Test bulk operations execution."""
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.rowcount = 5
        mock_connection.execute.return_value = mock_result
        mock_connection.begin.return_value = Mock()
        
        with patch.object(self.warehouse, '_get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_connection
            mock_get_conn.return_value.__exit__.return_value = None
            
            operations = [
                {'query': 'INSERT INTO test VALUES (1)', 'type': 'INSERT'},
                {'query': 'UPDATE test SET value = 2', 'type': 'UPDATE'}
            ]
            
            with patch.object(self.warehouse, '_execute_single_operation') as mock_exec:
                mock_exec.return_value = {'rows_affected': 5}
                
                result = self.warehouse.execute_bulk_operations(operations)
                
                assert result['total_operations'] == 2
                assert result['successful_operations'] == 2
                assert result['total_rows_affected'] == 10


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_generate_encryption_key(self):
        """Test encryption key generation."""
        key = generate_encryption_key()
        
        assert len(key) > 0
        assert isinstance(key, str)
        
        # Should be base64 encoded
        import base64
        try:
            decoded = base64.b64decode(key)
            assert len(decoded) == 32  # 256 bits
        except Exception:
            pytest.fail("Generated key is not valid base64")
    
    def test_hash_claim_text(self):
        """Test claim text hashing utility."""
        claim_text = "Test claim description for hashing"
        
        hash_result = hash_claim_text(claim_text)
        
        assert len(hash_result) == 64  # SHA256 hex string
        assert isinstance(hash_result, str)
        
        # Same input should produce same hash
        hash_result_2 = hash_claim_text(claim_text)
        assert hash_result == hash_result_2
    
    def test_secure_compare_strings(self):
        """Test secure string comparison."""
        string1 = "test_string"
        string2 = "test_string"
        string3 = "different_string"
        
        # Same strings should be equal
        assert secure_compare_strings(string1, string2) is True
        
        # Different strings should not be equal
        assert secure_compare_strings(string1, string3) is False
    
    def test_quick_tokenize_claim_numbers(self):
        """Test quick claim number tokenization utility."""
        claim_numbers = ['CLM001', 'CLM002', 'CLM003']
        
        mappings = quick_tokenize_claim_numbers(claim_numbers)
        
        assert len(mappings) == 3
        for claim_num in claim_numbers:
            assert claim_num in mappings
            assert mappings[claim_num].startswith('TOK_')
    
    def test_quick_redact_text(self):
        """Test quick text redaction utility."""
        test_text = "Contact john@email.com or call 555-123-4567"
        
        redacted = quick_redact_text(test_text)
        
        assert "john@email.com" not in redacted
        assert "555-123-4567" not in redacted
        assert "[EMAIL_REDACTED]" in redacted
        assert "[PHONE_REDACTED]" in redacted


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_crypto_manager(self):
        """Test crypto manager factory function."""
        # Without master key
        manager1 = create_crypto_manager()
        assert isinstance(manager1, CryptoManager)
        
        # With master key
        manager2 = create_crypto_manager("test_master_key")
        assert isinstance(manager2, CryptoManager)
        
        # Should be different instances
        assert manager1 is not manager2
    
    def test_create_token_manager(self):
        """Test token manager factory function."""
        manager = create_token_manager()
        
        assert isinstance(manager, TokenManager)
        assert manager.token_counter == 0
        assert len(manager.token_mappings) == 0
    
    @patch('building_coverage_system.utils.sql_data_warehouse.sqlalchemy.create_engine')
    def test_create_sql_data_warehouse(self, mock_create_engine):
        """Test SQL data warehouse factory function."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        credentials = create_sample_credentials()
        warehouse = create_sql_data_warehouse(credentials)
        
        assert isinstance(warehouse, SQLDataWarehouse)
        assert warehouse.credentials == credentials
    
    @patch('building_coverage_system.utils.sql_data_warehouse.sqlalchemy.create_engine')
    def test_optimize_table_performance_utility(self, mock_create_engine):
        """Test table performance optimization utility function."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        credentials = create_sample_credentials()
        warehouse = create_sql_data_warehouse(credentials)
        
        with patch.object(warehouse, 'optimize_warehouse_performance') as mock_optimize:
            mock_optimize.return_value = {
                'tables_optimized': 2,
                'indexes_created': 5,
                'optimization_time': 30.0
            }
            
            result = optimize_table_performance(warehouse, ['table1', 'table2'])
            
            assert 'tables_optimized' in result
            assert result['tables_optimized'] == 2
    
    @patch('building_coverage_system.utils.sql_data_warehouse.sqlalchemy.create_engine')
    def test_batch_process_data_utility(self, mock_create_engine):
        """Test batch process data utility function."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        credentials = create_sample_credentials()
        warehouse = create_sql_data_warehouse(credentials)
        data_df = create_sample_claims_data(100)
        
        with patch.object(warehouse, 'batch_process_claims') as mock_batch:
            mock_batch.return_value = {
                'total_claims': 100,
                'total_processed': 100,
                'processing_time': 5.0
            }
            
            result = batch_process_data(warehouse, data_df)
            
            assert 'total_claims' in result
            assert result['total_claims'] == 100