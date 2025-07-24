# Plugin Architecture Integration Guide: Codebase 1 Transformation

## Executive Summary

This guide outlines how to integrate Codebase 2's plugin architecture into Codebase 1's building coverage match system. The transformation will convert the current monolithic structure into a flexible, extensible plugin-based system while maintaining all existing functionality.

## Table of Contents
1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Plugin Integration Points](#plugin-integration-points)
3. [Plugin Interface Design](#plugin-interface-design)
4. [Migration Strategy](#migration-strategy)
5. [Configuration Management](#configuration-management)
6. [Implementation Examples](#implementation-examples)
7. [Benefits and Trade-offs](#benefits-and-trade-offs)

---

## Current Architecture Analysis

### Codebase 1 Integration Points Identified

Based on the existing modular structure, we've identified five key areas where plugin architecture can be integrated:

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT MONOLITHIC                      │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Configuration   │  │ RAG Processing  │  │ Rules Engine│ │
│  │ - Fixed config  │  │ - Single impl   │  │ - Fixed     │ │
│  │ - Hard-coded    │  │ - Tight coupling│  │   rules     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐ │
│  │ SQL Pipelines   │  │          Utilities                  │ │
│  │ - Fixed sources │  │ - Crypto, Detoken, SQL DW          │ │
│  │ - Hard-coded    │  │ - Fixed implementations             │ │
│  └─────────────────┘  └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Transformation Target:**

```
┌─────────────────────────────────────────────────────────────┐
│                    PLUGIN-BASED ARCHITECTURE               │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  PLUGIN MANAGER                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │ │
│  │  │   Plugin    │  │   Plugin    │  │     Plugin      │ │ │
│  │  │   Loader    │  │   Registry  │  │   Lifecycle     │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  INTERFACE LAYER                       │ │
│  │  IDataSource │ IProcessor │ IRulesEngine │ IStorage    │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PLUGIN ECOSYSTEM                        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Data Source │  │ Processor   │  │    Rules Engine     │  │
│  │  Plugins    │  │   Plugins   │  │      Plugins        │  │
│  │             │  │             │  │                     │  │
│  │ • AIP SQL   │  │ • RAG Proc  │  │ • Building Rules    │  │
│  │ • Atlas SQL │  │ • Text Proc │  │ • Coverage Rules    │  │
│  │ • Snowflake │  │ • GPT API   │  │ • Custom Rules      │  │
│  │ • Custom    │  │ • Custom    │  │ • ML-based Rules    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Storage    │  │ Validation  │  │    Utilities        │  │
│  │  Plugins    │  │   Plugins   │  │     Plugins         │  │
│  │             │  │             │  │                     │  │
│  │ • SQL DW    │  │ • Pydantic  │  │ • Crypto Utils      │  │
│  │ • ADLS      │  │ • Custom    │  │ • Token Utils       │  │
│  │ • Local     │  │ • Schema    │  │ • Custom Utils      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Plugin Integration Points

### 1. **Data Source Plugins** 
Replace fixed SQL connections with pluggable data sources:

**Current (Monolithic):**
```python
# coverage_sql_pipelines/src/sql_extract.py
class FeatureExtractor:
    def __init__(self, credentials_dict, sql_queries, crypto_spark, logger, SQL_QUERY_CONFIGS):
        # Hard-coded data sources
        self.aip_connection = self._get_aip_connection(credentials_dict)
        self.atlas_connection = self._get_atlas_connection(credentials_dict)
        self.snowflake_connection = self._get_snowflake_connection(credentials_dict)
```

**Target (Plugin-based):**
```python
# New plugin interface
class IDataSourcePlugin:
    def load_data(self, query: str, params: Dict) -> pd.DataFrame:
        pass
    
    def validate_connection(self) -> bool:
        pass
    
    def get_schema(self) -> Dict:
        pass
```

### 2. **Processing Plugins**
Convert RAG and text processing into pluggable components:

**Current (Fixed):**
```python
# coverage_rag_implementation/src/rag_predictor.py
class RAGPredictor:
    def __init__(self, get_prompt: Any, rag_params: Dict[str, Any]) -> None:
        # Fixed implementation
        self.rg_processor = RAGProcessor(get_prompt, self.rag_params)
        self.text_processor = TextProcessor()
```

**Target (Plugin-based):**
```python
class IProcessorPlugin:
    def process_text(self, text: str, config: Dict) -> str:
        pass
    
    def generate_summary(self, chunks: List[str], prompt: str) -> str:
        pass
```

### 3. **Rules Engine Plugins**
Make rule evaluation pluggable and configurable:

**Current (Fixed):**
```python
# coverage_rules/src/coverage_rules.py  
class CoverageRules:
    def __init__(self, rule_type: str, rule_conditions: List[str]):
        # Fixed rule implementation
        self.rule_conditions = rule_conditions
```

**Target (Plugin-based):**
```python
class IRulesEnginePlugin:
    def evaluate_rules(self, data: pd.DataFrame, conditions: List[Dict]) -> pd.DataFrame:
        pass
    
    def add_rule(self, rule: Dict) -> None:
        pass
    
    def validate_rule(self, rule: Dict) -> bool:
        pass
```

### 4. **Configuration Plugins**
Replace code-based configuration with YAML-driven approach:

**Current (Code-based):**
```python
# coverage_configs/src/environment.py
class DatabricksEnv:
    def __init__(self, databricks_dictionary):
        self.credentials_dict = get_credentials(databricks_dictionary)
        self.sql_queries = get_sql_query()
        self.rag_params = get_rag_params(databricks_dictionary)
```

**Target (YAML-based):**
```yaml
# coverage_match_config.yaml
data_sources:
  primary:
    type: aip_sql
    connection_params:
      server: "${AIP_SERVER}"
      database: "${AIP_DATABASE}"
  secondary:
    type: atlas_sql
    connection_params:
      server: "${ATLAS_SERVER}"

processors:
  text_processor:
    type: advanced_text
    config:
      cleaning_rules: ["remove_duplicates", "normalize_case"]
  
  rag_processor:
    type: gpt_rag
    config:
      model: "gpt-4"
      max_tokens: 4000
      temperature: 0.1

rules_engines:
  building_coverage:
    type: condition_based
    rules_file: "config/building_rules.yaml"
```

### 5. **Storage Plugins**
Pluggable output destinations:

**Current (Fixed):**
```python
# Fixed output to specific databases
final_df.write_to_sql_warehouse()
```

**Target (Plugin-based):**
```python
class IStoragePlugin:
    def save_data(self, data: pd.DataFrame, destination: str, config: Dict) -> bool:
        pass
    
    def validate_schema(self, data: pd.DataFrame) -> bool:
        pass
```

---

## Plugin Interface Design

### Core Plugin Interfaces

#### 1. **Base Plugin Interface**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class IPlugin(ABC):
    """Base interface for all plugins"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration"""
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources (optional)"""
        pass
```

#### 2. **Data Source Plugin Interface**
```python
class IDataSourcePlugin(IPlugin):
    """Interface for data source plugins"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    def load_data(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Load data using query"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if connection is valid"""
        pass
    
    @abstractmethod
    def get_schema(self, table_name: str) -> Dict[str, str]:
        """Get table schema"""
        pass
    
    def disconnect(self) -> None:
        """Close connection"""
        pass
```

#### 3. **Processor Plugin Interface**
```python
class IProcessorPlugin(IPlugin):
    """Interface for processing plugins"""
    
    @abstractmethod
    def process_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Process input data"""
        pass
    
    @abstractmethod
    def supports_parallel_processing(self) -> bool:
        """Check if plugin supports parallel processing"""
        pass
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {}
```

#### 4. **Rules Engine Plugin Interface**
```python
class IRulesEnginePlugin(IPlugin):
    """Interface for rules engine plugins"""
    
    @abstractmethod
    def load_rules(self, rules_config: Dict[str, Any]) -> None:
        """Load rules from configuration"""
        pass
    
    @abstractmethod
    def evaluate_rules(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply rules to data"""
        pass
    
    @abstractmethod
    def add_dynamic_rule(self, rule: Dict[str, Any]) -> bool:
        """Add rule at runtime"""
        pass
    
    def validate_rules(self) -> List[str]:
        """Validate loaded rules, return errors if any"""
        return []
```

#### 5. **Storage Plugin Interface**
```python
class IStoragePlugin(IPlugin):
    """Interface for storage plugins"""
    
    @abstractmethod
    def save_data(self, data: pd.DataFrame, destination: str, config: Dict[str, Any]) -> bool:
        """Save data to destination"""
        pass
    
    @abstractmethod
    def validate_schema(self, data: pd.DataFrame, expected_schema: Dict[str, str]) -> bool:
        """Validate data schema before saving"""
        pass
    
    def supports_batch_write(self) -> bool:
        """Check if batch writing is supported"""
        return False
```

### Plugin Manager Design

```python
class PluginManager:
    """Central plugin management system"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.plugins: Dict[str, Dict[str, IPlugin]] = {
            'data_sources': {},
            'processors': {},
            'rules_engines': {},
            'storage': {}
        }
        self.plugin_registry = {}
    
    def register_plugin(self, plugin_type: str, plugin_name: str, plugin_class: type) -> None:
        """Register a plugin class"""
        if plugin_type not in self.plugins:
            raise ValueError(f"Unknown plugin type: {plugin_type}")
        
        self.plugin_registry[f"{plugin_type}.{plugin_name}"] = plugin_class
    
    def load_plugin(self, plugin_type: str, plugin_name: str, config: Dict[str, Any]) -> IPlugin:
        """Load and initialize a plugin"""
        registry_key = f"{plugin_type}.{plugin_name}"
        
        if registry_key not in self.plugin_registry:
            raise ValueError(f"Plugin not found: {registry_key}")
        
        plugin_class = self.plugin_registry[registry_key]
        plugin = plugin_class()
        
        # Validate configuration
        if not plugin.validate_config(config):
            raise ValueError(f"Invalid configuration for plugin: {registry_key}")
        
        # Initialize plugin
        plugin.initialize(config)
        
        # Store in active plugins
        self.plugins[plugin_type][plugin_name] = plugin
        
        return plugin
    
    def get_plugin(self, plugin_type: str, plugin_name: str) -> Optional[IPlugin]:
        """Get an active plugin"""
        return self.plugins.get(plugin_type, {}).get(plugin_name)
    
    def load_all_plugins(self) -> None:
        """Load all plugins from configuration"""
        for plugin_type, plugins_config in self.config.items():
            if plugin_type in self.plugins:
                for plugin_name, plugin_config in plugins_config.items():
                    self.load_plugin(plugin_type, plugin_name, plugin_config)
    
    def cleanup_all_plugins(self) -> None:
        """Cleanup all loaded plugins"""
        for plugin_type_dict in self.plugins.values():
            for plugin in plugin_type_dict.values():
                plugin.cleanup()
```

---

## Migration Strategy

### Phase 1: Foundation Setup (Week 1-2)

#### Step 1: Create Plugin Framework
```python
# New file: coverage_plugins/core/interfaces.py
# Implement all plugin interfaces defined above

# New file: coverage_plugins/core/manager.py  
# Implement PluginManager class

# New file: coverage_plugins/core/registry.py
# Plugin registration and discovery utilities
```

#### Step 2: Convert Configuration System
```python
# New file: coverage_plugins/config/loader.py
class ConfigLoader:
    def load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        pass
    
    def validate_config_schema(self, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema"""
        pass
    
    def merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        """Merge configuration files"""
        pass
```

#### Step 3: Maintain Backward Compatibility
```python
# coverage_plugins/compat/legacy_wrapper.py
class LegacyCompatibilityWrapper:
    """Wrapper to maintain existing API while using plugins internally"""
    
    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
        # Initialize with existing behavior
        self._setup_default_plugins()
    
    def _setup_default_plugins(self):
        """Setup plugins that replicate current behavior exactly"""
        # Load existing implementations as plugins
        pass
```

### Phase 2: Plugin Implementation (Week 3-4)

#### Step 1: Data Source Plugins
```python
# coverage_plugins/data_sources/aip_sql_plugin.py
class AIPSQLPlugin(IDataSourcePlugin):
    name = "aip_sql"
    version = "1.0.0"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        self.connection_params = config['connection_params']
        # Use existing AIP connection logic
        
    def load_data(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        # Wrap existing FeatureExtractor._get_aip_data logic
        pass

# coverage_plugins/data_sources/atlas_sql_plugin.py  
class AtlasSQLPlugin(IDataSourcePlugin):
    # Similar implementation for Atlas
    pass

# coverage_plugins/data_sources/snowflake_plugin.py
class SnowflakePlugin(IDataSourcePlugin):
    # Similar implementation for Snowflake
    pass
```

#### Step 2: Processor Plugins
```python
# coverage_plugins/processors/rag_processor_plugin.py
class RAGProcessorPlugin(IProcessorPlugin):
    name = "rag_processor"
    version = "1.0.0"
    
    def process_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        # Wrap existing RAGPredictor logic
        rag_predictor = RAGPredictor(config['get_prompt'], config['rag_params'])
        return rag_predictor.get_summary(data)

# coverage_plugins/processors/text_processor_plugin.py
class TextProcessorPlugin(IProcessorPlugin):
    name = "text_processor"
    version = "1.0.0"
    
    def process_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        # Wrap existing text processing logic
        pass
```

#### Step 3: Rules Engine Plugins  
```python
# coverage_plugins/rules/coverage_rules_plugin.py
class CoverageRulesPlugin(IRulesEnginePlugin):
    name = "coverage_rules"
    version = "1.0.0"
    
    def load_rules(self, rules_config: Dict[str, Any]) -> None:
        # Load existing rule conditions
        self.rule_conditions = rules_config['conditions']
        
    def evaluate_rules(self, data: pd.DataFrame) -> pd.DataFrame:
        # Wrap existing CoverageRules.classify_rule_conditions logic
        rules_engine = CoverageRules(self.rule_type, self.rule_conditions)
        return rules_engine.classify_rule_conditions(data)
```

### Phase 3: Integration (Week 5-6)

#### Step 1: Update Main Execution Pipeline
```python
# Update BLDG_COV_MATCH_EXECUTION.py.ipynb to use plugin system

# Instead of:
env = DatabricksEnv(databricks_dict)
sql_query = FeatureExtractor(credentials_dict, sql_queries, crypto_spark, logger, SQL_QUERY_CONFIGS)
rag = RAGPredictor(get_prompt, rag_params)
bldg_rules = CoverageRules('BLDG INDICATOR', bldg_conditions)

# Use:
plugin_manager = PluginManager('config/coverage_match_config.yaml')
plugin_manager.load_all_plugins()

# Get data source plugin
data_source = plugin_manager.get_plugin('data_sources', 'primary')
feature_df = data_source.load_data(sql_queries['feature_query'])

# Get processor plugin
processor = plugin_manager.get_plugin('processors', 'rag_processor')
summary_df = processor.process_data(filtered_claims_df, rag_config)

# Get rules engine plugin
rules_engine = plugin_manager.get_plugin('rules_engines', 'building_coverage')
rule_predictions = rules_engine.evaluate_rules(merged_df)
```

#### Step 2: Configuration Migration
```yaml
# config/coverage_match_config.yaml - New YAML configuration
data_sources:
  primary:
    type: aip_sql
    connection_params:
      server: "${AIP_SQL_SERVER}"
      database: "${AIP_DATABASE}"
      spn:
        client_id: "${AIP_CLIENT_ID}"
        client_secret: "${AIP_CLIENT_SECRET}"
        tenant_id: "${AIP_TENANT_ID}"
    queries:
      feature_query: |
        SELECT cc.CLAIMNO, cc.CLAIMKEY, cc.LOBCD, 
               cc.LOSSDESC, cc.LOSSDT, cc.REPORTEDDT
        FROM claim_core cc
        WHERE cc.STATUSCD = 'O'
  
  secondary:
    type: atlas_sql
    connection_params:
      server: "${ATLAS_SQL_SERVER}"
      database: "${ATLAS_DATABASE}"
      spn:
        client_id: "${ATLAS_CLIENT_ID}"
        client_secret: "${ATLAS_CLIENT_SECRET}"
        tenant_id: "${ATLAS_TENANT_ID}"
        
  supplementary:
    type: snowflake
    connection_params:
      account: "${SNOWFLAKE_ACCOUNT}"
      user: "${SNOWFLAKE_USER}"
      password: "${SNOWFLAKE_PASSWORD}"
      warehouse: "${SNOWFLAKE_WAREHOUSE}"
      database: "${SNOWFLAKE_DATABASE}"

processors:
  rag_processor:
    type: rag_processor
    config:
      gpt_config:
        api_url: "${GPT_API_URL}"
        api_key: "${GPT_API_KEY}"
        model: "gpt-4"
        max_tokens: 4000
        temperature: 0.1
      chunking:
        strategy: "sentence"
        max_chunk_size: 8000
        overlap: 200
        
  text_processor:
    type: text_processor
    config:
      cleaning_rules:
        - remove_duplicates
        - normalize_whitespace
        - remove_special_chars
      deduplication:
        similarity_threshold: 0.95

rules_engines:
  building_coverage:
    type: coverage_rules
    config:
      rule_type: "BLDG INDICATOR" 
      rules_file: "config/building_rules.yaml"
      conditions:
        - "BLDG in LOSSDESC"
        - "BUILDING in LOSSDESC" 
        - "STRUCTURE in LOSSDESC"

storage:
  primary_output:
    type: sql_data_warehouse
    config:
      connection: "primary"
      table: "building_coverage_predictions"
      schema: "dbo"
      write_mode: "append"
```

### Phase 4: Testing and Validation (Week 7)

#### Step 1: Comprehensive Testing
```python
# coverage_plugins/test/integration/test_plugin_pipeline.py
class TestPluginPipeline:
    def test_end_to_end_with_plugins(self):
        """Test complete pipeline using plugin architecture"""
        # Load configuration
        plugin_manager = PluginManager('test/config/test_config.yaml')
        plugin_manager.load_all_plugins()
        
        # Test data source plugin
        data_source = plugin_manager.get_plugin('data_sources', 'test_source')
        test_data = data_source.load_data("SELECT * FROM test_claims LIMIT 10")
        assert not test_data.empty
        
        # Test processor plugin
        processor = plugin_manager.get_plugin('processors', 'rag_processor')
        processed_data = processor.process_data(test_data, {})
        assert 'summary' in processed_data.columns
        
        # Test rules engine plugin
        rules_engine = plugin_manager.get_plugin('rules_engines', 'building_coverage')
        results = rules_engine.evaluate_rules(processed_data)
        assert 'prediction' in results.columns

    def test_plugin_hot_swap(self):
        """Test ability to swap plugins at runtime"""
        # Test changing from one data source to another without restart
        pass

    def test_backward_compatibility(self):
        """Ensure existing code still works"""
        # Test that legacy interfaces still function
        pass
```

#### Step 2: Performance Validation
```python
# coverage_plugins/test/performance/test_plugin_performance.py
class TestPluginPerformance:
    def test_plugin_overhead(self):
        """Measure performance overhead of plugin system"""
        # Compare plugin-based vs direct implementation
        pass
        
    def test_parallel_plugin_execution(self):
        """Test parallel execution of plugins"""
        # Implement Codebase 2's threading model
        pass
```

---

## Configuration Management

### YAML-Based Configuration Structure

```yaml
# config/coverage_match_config.yaml
version: "1.0"
environment: "production"  # development, staging, production

# Plugin configuration
plugins:
  data_sources:
    primary:
      type: "aip_sql"
      enabled: true
      config:
        server: "${AIP_SQL_SERVER}"
        database: "${AIP_DATABASE}"
        authentication:
          type: "spn"
          client_id: "${AIP_CLIENT_ID}"
          client_secret: "${AIP_CLIENT_SECRET}"
          tenant_id: "${AIP_TENANT_ID}"
        connection_pool:
          min_connections: 1
          max_connections: 10
          timeout: 30
          
    secondary:
      type: "atlas_sql"
      enabled: true
      fallback_for: ["primary"]  # Use as fallback if primary fails
      config:
        server: "${ATLAS_SQL_SERVER}"
        database: "${ATLAS_DATABASE}"
        authentication:
          type: "spn"
          client_id: "${ATLAS_CLIENT_ID}"
          client_secret: "${ATLAS_CLIENT_SECRET}"
          tenant_id: "${ATLAS_TENANT_ID}"
          
  processors:
    text_processor:
      type: "advanced_text"
      enabled: true
      config:
        cleaning:
          remove_duplicates: true
          normalize_case: true
          remove_special_chars: true
          custom_replacements:
            - pattern: "\\b\\d{1,2}/\\d{1,2}/\\d{2,4}\\b"
              replacement: "[DATE]"
        chunking:
          strategy: "sentence"
          max_length: 8000
          overlap: 200
          respect_boundaries: true
          
    rag_processor:
      type: "gpt_rag"
      enabled: true
      config:
        api:
          url: "${GPT_API_URL}"
          key: "${GPT_API_KEY}"
          version: "2023-12-01-preview"
        model:
          name: "gpt-4"
          max_tokens: 4000
          temperature: 0.1
          frequency_penalty: 0.0
          presence_penalty: 0.0
        retry:
          max_attempts: 3
          backoff_factor: 2
          fallback_model: "gpt-3.5-turbo"
          
  rules_engines:
    building_coverage:
      type: "condition_based"
      enabled: true
      config:
        rules_source: "file"
        rules_file: "config/rules/building_coverage_rules.yaml"
        evaluation_mode: "all"  # all, any, weighted
        confidence_threshold: 0.7
        
  storage:
    primary_output:
      type: "sql_data_warehouse"
      enabled: true
      config:
        connection: "primary"
        schema: "dbo"
        table: "building_coverage_predictions_v2"
        write_mode: "append"
        batch_size: 1000
        validation:
          schema_check: true
          duplicate_check: true

# Pipeline configuration
pipeline:
  execution_mode: "sequential"  # sequential, parallel
  max_workers: 4  # For parallel mode
  batch_size: 100
  checkpoint_enabled: true
  checkpoint_interval: 1000
  
# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - type: "console"
      enabled: true
    - type: "file"
      enabled: true
      filename: "logs/coverage_match.log"
      max_size: "10MB"
      backup_count: 5

# Monitoring and metrics
monitoring:
  enabled: true
  metrics:
    - plugin_execution_time
    - data_source_latency
    - gpt_api_calls
    - rule_evaluation_time
  export:
    type: "prometheus"
    endpoint: "/metrics"
```

### Environment-Specific Configuration

```yaml
# config/environments/development.yaml
extends: "../coverage_match_config.yaml"

plugins:
  data_sources:
    primary:
      config:
        server: "${DEV_AIP_SQL_SERVER}"
        database: "${DEV_AIP_DATABASE}"
  processors:
    rag_processor:
      config:
        model:
          name: "gpt-3.5-turbo"  # Use cheaper model for dev
          max_tokens: 2000

pipeline:
  batch_size: 10  # Smaller batches for dev
  
logging:
  level: "DEBUG"
```

### Configuration Validation Schema

```python
# coverage_plugins/config/schema.py
from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Union

class DataSourceConfig(BaseModel):
    type: str
    enabled: bool = True
    config: Dict[str, Union[str, int, Dict]]
    fallback_for: Optional[List[str]] = None
    
    @validator('type')
    def validate_data_source_type(cls, v):
        allowed_types = ['aip_sql', 'atlas_sql', 'snowflake', 'local_file']
        if v not in allowed_types:
            raise ValueError(f'Data source type must be one of {allowed_types}')
        return v

class ProcessorConfig(BaseModel):
    type: str
    enabled: bool = True
    config: Dict[str, Union[str, int, float, Dict, List]]
    
class RulesEngineConfig(BaseModel):
    type: str
    enabled: bool = True
    config: Dict[str, Union[str, float, Dict, List]]

class StorageConfig(BaseModel):
    type: str
    enabled: bool = True
    config: Dict[str, Union[str, int, Dict]]

class PluginConfiguration(BaseModel):
    data_sources: Dict[str, DataSourceConfig]
    processors: Dict[str, ProcessorConfig]
    rules_engines: Dict[str, RulesEngineConfig]
    storage: Dict[str, StorageConfig]

class PipelineConfig(BaseModel):
    execution_mode: str = "sequential"
    max_workers: int = 4
    batch_size: int = 100
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 1000
    
    @validator('execution_mode')
    def validate_execution_mode(cls, v):
        if v not in ['sequential', 'parallel']:
            raise ValueError('execution_mode must be sequential or parallel')
        return v

class CoverageMatchConfig(BaseModel):
    version: str
    environment: str
    plugins: PluginConfiguration
    pipeline: PipelineConfig
    logging: Optional[Dict]
    monitoring: Optional[Dict]
```

---

## Implementation Examples

### Complete Example: Converting RAG Processing

#### Before (Monolithic):
```python
# coverage_rag_implementation/src/rag_predictor.py (Current)
class RAGPredictor:
    def __init__(self, get_prompt: Any, rag_params: Dict[str, Any]) -> None:
        self.get_prompt = get_prompt
        self.rag_params = rag_params
        self.rg_processor = RAGProcessor(get_prompt, self.rag_params)
        self.text_processor = TextProcessor()

    def get_summary(self, scoring_df: pd.DataFrame) -> pd.DataFrame:
        filtered_df = scoring_df[
            (scoring_df['clean_FN_TEXT'].str.len() >= 100) &
            (scoring_df['LOBCD'].isin(['15', '17']))
        ]
        
        out_list = []
        for claimno in filtered_df['CLAIMNO'].unique():
            claim_df = filtered_df[filtered_df['CLAIMNO'] == claimno]
            
            # Fixed processing logic
            res_dict = self.rg_processor.get_summary_and_loss_desc_b_code(
                claim_df, 
                self.rag_params['params_for_chunking']['chunk_size'],
                self.rag_params['rag_query']
            )
            out_list.append(res_dict)
            
        return pd.DataFrame(out_list)
```

#### After (Plugin-based):
```python
# coverage_plugins/processors/rag_processor_plugin.py (New)
class RAGProcessorPlugin(IProcessorPlugin):
    name = "rag_processor"
    version = "1.0.0"
    
    def __init__(self):
        self.rag_processor = None
        self.text_processor = None
        self.config = None
        
    def initialize(self, config: Dict[str, Any]) -> None:
        self.config = config
        # Initialize with existing logic but configurable
        self.rag_processor = RAGProcessor(
            config['get_prompt'], 
            config['rag_params']
        )
        self.text_processor = TextProcessor()
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        required_keys = ['get_prompt', 'rag_params', 'filtering']
        return all(key in config for key in required_keys)
    
    def process_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        # Configurable filtering instead of hard-coded
        filter_config = self.config.get('filtering', {})
        min_text_length = filter_config.get('min_text_length', 100)
        allowed_lob_codes = filter_config.get('allowed_lob_codes', ['15', '17'])
        
        filtered_df = data[
            (data['clean_FN_TEXT'].str.len() >= min_text_length) &
            (data['LOBCD'].isin(allowed_lob_codes))
        ]
        
        # Support parallel processing
        if self.supports_parallel_processing() and self.config.get('parallel_enabled', False):
            return self._process_parallel(filtered_df)
        else:
            return self._process_sequential(filtered_df)
    
    def supports_parallel_processing(self) -> bool:
        return True
    
    def _process_sequential(self, df: pd.DataFrame) -> pd.DataFrame:
        """Original sequential processing"""
        out_list = []
        for claimno in df['CLAIMNO'].unique():
            claim_df = df[df['CLAIMNO'] == claimno]
            res_dict = self.rag_processor.get_summary_and_loss_desc_b_code(
                claim_df,
                self.config['rag_params']['params_for_chunking']['chunk_size'],
                self.config['rag_params']['rag_query']
            )
            out_list.append(res_dict)
        return pd.DataFrame(out_list)
    
    def _process_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """New parallel processing capability"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        max_workers = self.config.get('max_workers', 4)
        claim_groups = [df[df['CLAIMNO'] == claimno] for claimno in df['CLAIMNO'].unique()]
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_single_claim, claim_df) 
                for claim_df in claim_groups
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Log error but continue processing
                    print(f"Error processing claim: {e}")
                    
        return pd.DataFrame(results)
    
    def _process_single_claim(self, claim_df: pd.DataFrame) -> Dict:
        """Process a single claim - used in parallel processing"""
        return self.rag_processor.get_summary_and_loss_desc_b_code(
            claim_df,
            self.config['rag_params']['params_for_chunking']['chunk_size'],
            self.config['rag_params']['rag_query']
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Return processing statistics"""
        return {
            'claims_processed': getattr(self, '_claims_processed', 0),
            'average_processing_time': getattr(self, '_avg_processing_time', 0),
            'parallel_enabled': self.config.get('parallel_enabled', False)
        }
```

#### Usage with Plugin System:
```python
# Updated execution notebook
# BLDG_COV_MATCH_EXECUTION_PLUGIN.py.ipynb

# Initialize plugin system
plugin_manager = PluginManager('config/coverage_match_config.yaml')

# Register custom plugins
plugin_manager.register_plugin('processors', 'rag_processor', RAGProcessorPlugin)
plugin_manager.register_plugin('data_sources', 'aip_sql', AIPSQLPlugin)
plugin_manager.register_plugin('rules_engines', 'coverage_rules', CoverageRulesPlugin)

# Load all plugins from configuration
plugin_manager.load_all_plugins()

# Execute pipeline using plugins
data_source = plugin_manager.get_plugin('data_sources', 'primary')
feature_df = data_source.load_data('feature_extraction_query')

processor = plugin_manager.get_plugin('processors', 'rag_processor')
summary_df = processor.process_data(filtered_claims_df, {})

rules_engine = plugin_manager.get_plugin('rules_engines', 'building_coverage')
predictions_df = rules_engine.evaluate_rules(merged_df)

# Output results
storage_plugin = plugin_manager.get_plugin('storage', 'primary_output')
storage_plugin.save_data(final_df, 'building_coverage_predictions', {})

# Cleanup
plugin_manager.cleanup_all_plugins()
```

### Configuration for the Example:
```yaml
# config/coverage_match_config.yaml
plugins:
  processors:
    rag_processor:
      type: "rag_processor"
      config:
        get_prompt: "${PROMPT_FUNCTION}"
        rag_params:
          gpt_config_params:
            api_url: "${GPT_API_URL}"
            max_tokens: 4000
          params_for_chunking:
            chunk_size: 8000
          rag_query: "Analyze the following insurance claim text..."
        filtering:
          min_text_length: 100
          allowed_lob_codes: ["15", "17"]
        parallel_enabled: true
        max_workers: 4
```

---

## Benefits and Trade-offs

### Benefits of Plugin Architecture Integration

#### 1. **Enhanced Flexibility**
- **Runtime Configuration**: Change data sources, processing logic, and rules without code changes
- **A/B Testing**: Easy to test different processors or rules engines side-by-side
- **Environment Adaptation**: Different plugin configurations for dev/staging/production

```yaml
# Easy A/B testing configuration
processors:
  rag_processor_a:
    type: "rag_processor"
    config:
      model: "gpt-4"
      temperature: 0.1
  rag_processor_b:
    type: "rag_processor"  
    config:
      model: "gpt-3.5-turbo"
      temperature: 0.3

pipeline:
  ab_testing:
    enabled: true
    split_ratio: 0.5
    processors: ["rag_processor_a", "rag_processor_b"]
```

#### 2. **Improved Scalability**
- **Parallel Processing**: Easy integration of Codebase 2's threading model
- **Horizontal Scaling**: Plugins can be distributed across multiple nodes
- **Resource Optimization**: Different plugins can have different resource requirements

#### 3. **Better Maintainability**
- **Isolated Changes**: Plugin updates don't affect other components
- **Version Management**: Each plugin can be versioned independently
- **Clear Interfaces**: Well-defined contracts between components

#### 4. **Enhanced Testing**
- **Mock Plugins**: Easy to create mock plugins for testing
- **Integration Testing**: Test different plugin combinations
- **Isolated Unit Tests**: Test each plugin independently

```python
# Easy mocking for tests
class MockRAGProcessorPlugin(IProcessorPlugin):
    def process_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        # Return predictable test data
        return data.assign(summary="Mock summary", confidence=0.9)

# Use in tests
plugin_manager.register_plugin('processors', 'mock_rag', MockRAGProcessorPlugin)
```

#### 5. **Enterprise Integration**
- **Third-party Plugins**: Easy integration of external systems
- **Vendor Flexibility**: Switch between different vendors (e.g., OpenAI vs Azure OpenAI)
- **Compliance**: Different plugins for different regulatory requirements

### Trade-offs and Considerations

#### 1. **Increased Complexity**
- **Learning Curve**: Team needs to understand plugin architecture
- **Configuration Management**: More complex configuration files
- **Debugging**: Plugin boundaries can make debugging harder

**Mitigation Strategies:**
- Comprehensive documentation and examples
- Good logging and monitoring at plugin boundaries
- Development tools for plugin debugging

#### 2. **Performance Overhead**
- **Interface Abstraction**: Small overhead from plugin interfaces
- **Dynamic Loading**: Runtime plugin loading takes time
- **Memory Usage**: Plugin registry and manager consume memory

**Mitigation Strategies:**
- Benchmark plugin overhead vs direct calls
- Implement plugin caching
- Lazy loading of plugins

#### 3. **Configuration Complexity**
- **YAML Management**: Large configuration files can become unwieldy
- **Environment Variables**: More environment variables to manage
- **Validation**: Need robust configuration validation

**Mitigation Strategies:**
- Configuration file templating and inheritance
- Configuration validation schemas
- Environment-specific configuration files

#### 4. **Version Compatibility**
- **Plugin API Changes**: Changes to plugin interfaces affect all plugins
- **Dependency Management**: Plugins may have conflicting dependencies
- **Backward Compatibility**: Need to maintain compatibility with existing plugins

**Mitigation Strategies:**
- Semantic versioning for plugin APIs
- Plugin compatibility matrix
- Deprecation strategies for API changes

### Migration Risk Assessment

#### Low Risk Areas ✅
- **Configuration System**: YAML configuration is straightforward
- **Data Source Plugins**: Well-defined database interfaces
- **Storage Plugins**: Simple output operations

#### Medium Risk Areas ⚠️
- **RAG Processing**: Complex logic with many dependencies
- **Text Processing**: Performance-critical operations
- **Rules Engine**: Complex business logic

#### High Risk Areas ⚠️⚠️
- **Plugin Manager**: Core system that affects everything
- **Interface Design**: Changes require updates to all plugins
- **Parallel Processing**: Threading and concurrency complexity

### Recommended Migration Timeline

#### Phase 1 (2 weeks): Foundation
- Implement plugin interfaces
- Create plugin manager
- Basic configuration system
- Backward compatibility wrappers

#### Phase 2 (2 weeks): Core Plugins  
- Data source plugins
- Basic processor plugins
- Simple rules engine plugins
- Unit tests for all plugins

#### Phase 3 (2 weeks): Integration
- Update main execution pipeline
- End-to-end testing
- Performance validation
- Documentation

#### Phase 4 (1 week): Advanced Features
- Parallel processing
- Advanced configuration features
- Monitoring and metrics
- Production deployment

### Success Metrics

#### Technical Metrics
- **Plugin Load Time**: < 100ms per plugin
- **Processing Overhead**: < 5% compared to monolithic
- **Configuration Validation**: 100% schema coverage
- **Test Coverage**: > 90% for all plugins

#### Business Metrics
- **Development Velocity**: Faster feature development
- **System Flexibility**: Ability to change configurations without deployment
- **Maintenance Cost**: Reduced debugging and maintenance time
- **Integration Time**: Faster integration of new data sources/processors

---

## Conclusion

Integrating Codebase 2's plugin architecture into Codebase 1 will transform it from a monolithic system into a flexible, scalable, and maintainable platform. The migration strategy outlined provides a gradual, low-risk approach that maintains backward compatibility while introducing modern architectural patterns.

**Key Success Factors:**
1. **Gradual Migration**: Phase approach minimizes risk
2. **Backward Compatibility**: Existing code continues to work
3. **Clear Interfaces**: Well-defined plugin contracts
4. **Comprehensive Testing**: Validate each phase thoroughly
5. **Team Training**: Ensure team understands new architecture

**Expected Outcomes:**
- **50% faster feature development** due to plugin-based architecture
- **90% reduction in configuration changes** requiring code deployment
- **Enhanced system reliability** through isolated plugin failures
- **Improved scalability** with parallel processing capabilities
- **Better maintainability** with clear separation of concerns

The plugin architecture will position Codebase 1 for future growth while leveraging the proven patterns from Codebase 2's successful implementation.