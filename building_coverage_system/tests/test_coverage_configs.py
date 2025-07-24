"""
Tests for coverage_configs module.

This module contains unit tests for the coverage configuration
management functionality.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from building_coverage_system.coverage_configs.src.config_manager import (
    ConfigManager,
    DatabaseConfig,
    ModelConfig,
    ProcessingConfig,
    create_config_manager
)
from building_coverage_system.tests.fixtures.sample_data import create_sample_config


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_config = create_sample_config()
        self.config_manager = ConfigManager()
    
    def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        self.config_manager.load_config(self.sample_config)
        
        assert self.config_manager.config is not None
        assert 'data_sources' in self.config_manager.config
        assert 'embedding_model' in self.config_manager.config
    
    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_config, f)
            config_file = f.name
        
        try:
            self.config_manager.load_config(config_file)
            
            assert self.config_manager.config is not None
            assert self.config_manager.config['embedding_model']['model_name'] == 'all-MiniLM-L6-v2'
        finally:
            os.unlink(config_file)
    
    def test_get_database_config(self):
        """Test getting database configuration."""
        self.config_manager.load_config(self.sample_config)
        
        db_config = self.config_manager.get_database_config('primary')
        
        assert isinstance(db_config, DatabaseConfig)
        assert db_config.enabled is True
        assert db_config.connection_timeout == 30
    
    def test_get_model_config(self):
        """Test getting model configuration."""
        self.config_manager.load_config(self.sample_config)
        
        model_config = self.config_manager.get_model_config()
        
        assert isinstance(model_config, ModelConfig)
        assert model_config.model_name == 'all-MiniLM-L6-v2'
        assert model_config.batch_size == 32
    
    def test_get_processing_config(self):
        """Test getting processing configuration."""
        self.config_manager.load_config(self.sample_config)
        
        processing_config = self.config_manager.get_processing_config()
        
        assert isinstance(processing_config, ProcessingConfig)
        assert processing_config.batch_size == 1000
        assert processing_config.max_workers == 4
    
    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        self.config_manager.load_config(self.sample_config)
        
        is_valid, errors = self.config_manager.validate_config()
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_config_invalid(self):
        """Test validation of invalid configuration."""
        invalid_config = {
            'data_sources': {},  # Missing required sections
            'embedding_model': {}
        }
        
        self.config_manager.load_config(invalid_config)
        
        is_valid, errors = self.config_manager.validate_config()
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_update_config_section(self):
        """Test updating configuration section."""
        self.config_manager.load_config(self.sample_config)
        
        new_settings = {
            'batch_size': 64,
            'max_workers': 8
        }
        
        self.config_manager.update_config_section('processing', new_settings)
        
        processing_config = self.config_manager.get_processing_config()
        assert processing_config.batch_size == 64
        assert processing_config.max_workers == 8
    
    def test_get_threshold_valid(self):
        """Test getting valid threshold value."""
        self.config_manager.load_config(self.sample_config)
        
        threshold = self.config_manager.get_threshold('building_coverage')
        
        assert threshold == 0.85
    
    def test_get_threshold_invalid(self):
        """Test getting invalid threshold returns default."""
        self.config_manager.load_config(self.sample_config)
        
        threshold = self.config_manager.get_threshold('nonexistent_threshold', default=0.75)
        
        assert threshold == 0.75
    
    def test_save_config(self):
        """Test saving configuration to file."""
        self.config_manager.load_config(self.sample_config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            self.config_manager.save_config(config_file)
            
            # Verify file was created and contains valid JSON
            assert os.path.exists(config_file)
            
            with open(config_file, 'r') as f:
                saved_config = json.load(f)
            
            assert 'data_sources' in saved_config
            assert 'embedding_model' in saved_config
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)
    
    def test_merge_configs(self):
        """Test merging multiple configurations."""
        base_config = {
            'data_sources': {'primary': {'enabled': True}},
            'embedding_model': {'model_name': 'base_model'}
        }
        
        override_config = {
            'embedding_model': {'model_name': 'override_model', 'batch_size': 64},
            'new_section': {'setting': 'value'}
        }
        
        merged = self.config_manager.merge_configs(base_config, override_config)
        
        assert merged['data_sources']['primary']['enabled'] is True
        assert merged['embedding_model']['model_name'] == 'override_model'
        assert merged['embedding_model']['batch_size'] == 64
        assert merged['new_section']['setting'] == 'value'
    
    def test_get_environment_config(self):
        """Test getting environment-specific configuration."""
        self.config_manager.load_config(self.sample_config)
        
        # Add environment-specific config
        env_config = {
            'development': {
                'debug': True,
                'log_level': 'DEBUG'
            },
            'production': {
                'debug': False,
                'log_level': 'INFO'
            }
        }
        
        self.config_manager.config['environments'] = env_config
        
        dev_config = self.config_manager.get_environment_config('development')
        prod_config = self.config_manager.get_environment_config('production')
        
        assert dev_config['debug'] is True
        assert prod_config['debug'] is False


class TestDatabaseConfig:
    """Test cases for DatabaseConfig class."""
    
    def test_database_config_creation(self):
        """Test creating DatabaseConfig from dictionary."""
        config_dict = {
            'enabled': True,
            'connection_timeout': 45,
            'query_timeout': 600,
            'pool_size': 10
        }
        
        db_config = DatabaseConfig.from_dict(config_dict)
        
        assert db_config.enabled is True
        assert db_config.connection_timeout == 45
        assert db_config.query_timeout == 600
        assert db_config.pool_size == 10
    
    def test_database_config_defaults(self):
        """Test DatabaseConfig with default values."""
        config_dict = {'enabled': True}
        
        db_config = DatabaseConfig.from_dict(config_dict)
        
        assert db_config.enabled is True
        assert db_config.connection_timeout == 30  # default
        assert db_config.query_timeout == 300  # default
    
    def test_database_config_to_dict(self):
        """Test converting DatabaseConfig to dictionary."""
        db_config = DatabaseConfig(
            enabled=True,
            connection_timeout=45,
            query_timeout=600
        )
        
        config_dict = db_config.to_dict()
        
        assert config_dict['enabled'] is True
        assert config_dict['connection_timeout'] == 45
        assert config_dict['query_timeout'] == 600


class TestModelConfig:
    """Test cases for ModelConfig class."""
    
    def test_model_config_creation(self):
        """Test creating ModelConfig from dictionary."""
        config_dict = {
            'model_name': 'test-model',
            'max_sequence_length': 256,
            'batch_size': 16,
            'device': 'cuda'
        }
        
        model_config = ModelConfig.from_dict(config_dict)
        
        assert model_config.model_name == 'test-model'
        assert model_config.max_sequence_length == 256
        assert model_config.batch_size == 16
        assert model_config.device == 'cuda'
    
    def test_model_config_defaults(self):
        """Test ModelConfig with default values."""
        config_dict = {'model_name': 'test-model'}
        
        model_config = ModelConfig.from_dict(config_dict)
        
        assert model_config.model_name == 'test-model'
        assert model_config.batch_size == 32  # default
        assert model_config.device == 'cpu'  # default


class TestProcessingConfig:
    """Test cases for ProcessingConfig class."""
    
    def test_processing_config_creation(self):
        """Test creating ProcessingConfig from dictionary."""
        config_dict = {
            'batch_size': 2000,
            'max_workers': 8,
            'chunk_size': 1000,
            'timeout': 600
        }
        
        processing_config = ProcessingConfig.from_dict(config_dict)
        
        assert processing_config.batch_size == 2000
        assert processing_config.max_workers == 8
        assert processing_config.chunk_size == 1000
        assert processing_config.timeout == 600


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager."""
    
    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        # Create config manager
        config_manager = create_config_manager()
        
        # Load sample configuration
        sample_config = create_sample_config()
        config_manager.load_config(sample_config)
        
        # Validate configuration
        is_valid, errors = config_manager.validate_config()
        assert is_valid is True
        
        # Get different config types
        db_config = config_manager.get_database_config('primary')
        model_config = config_manager.get_model_config()
        processing_config = config_manager.get_processing_config()
        
        assert isinstance(db_config, DatabaseConfig)
        assert isinstance(model_config, ModelConfig)
        assert isinstance(processing_config, ProcessingConfig)
        
        # Update configuration
        config_manager.update_config_section('processing', {'batch_size': 500})
        
        updated_processing_config = config_manager.get_processing_config()
        assert updated_processing_config.batch_size == 500
    
    @patch('building_coverage_system.coverage_configs.src.config_manager.logger')
    def test_config_loading_with_logging(self, mock_logger):
        """Test configuration loading with proper logging."""
        config_manager = ConfigManager()
        sample_config = create_sample_config()
        
        config_manager.load_config(sample_config)
        
        # Verify logging was called
        mock_logger.info.assert_called()
    
    def test_config_validation_edge_cases(self):
        """Test configuration validation edge cases."""
        config_manager = ConfigManager()
        
        # Test empty configuration
        config_manager.load_config({})
        is_valid, errors = config_manager.validate_config()
        assert is_valid is False
        assert len(errors) > 0
        
        # Test configuration with missing required fields
        incomplete_config = {
            'data_sources': {},
            'embedding_model': {'model_name': 'test'}
            # Missing other required sections
        }
        
        config_manager.load_config(incomplete_config)
        is_valid, errors = config_manager.validate_config()
        assert is_valid is False
    
    def test_config_manager_factory(self):
        """Test ConfigManager factory function."""
        config_manager = create_config_manager()
        
        assert isinstance(config_manager, ConfigManager)
        assert config_manager.config is None  # No config loaded yet