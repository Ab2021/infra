# Building Coverage Match System

A comprehensive insurance claim processing system that combines traditional rule-based processing with advanced RAG (Retrieval-Augmented Generation) capabilities and parallel processing.

## Repository Structure

```
building_coverage_system/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── .gitignore                        # Git ignore rules
│
├── coverage_configs/                  # Original configuration system
│   ├── __init__.py
│   ├── src/
│   │   ├── __init__.py
│   │   ├── credentials.py            # Database credentials management
│   │   ├── environment.py            # Environment configuration
│   │   ├── prompts.py               # GPT prompt templates
│   │   ├── rag_params.py            # RAG parameters
│   │   └── sql.py                   # SQL queries
│   └── test/
│       └── unit/
│           ├── __init__.py
│           ├── test_credentials.py
│           ├── test_environment.py
│           └── test_prompts.py
│
├── coverage_rag_implementation/       # Original RAG implementation
│   ├── __init__.py
│   ├── src/
│   │   ├── __init__.py
│   │   ├── rag_predictor.py         # Main RAG predictor
│   │   └── helpers/
│   │       ├── __init__.py
│   │       ├── chunk_split_sentences.py
│   │       ├── gpt_api.py
│   │       └── text_processing.py
│   └── test/
│       └── unit/
│           ├── __init__.py
│           └── test_rag_predictor.py
│
├── coverage_rules/                   # Original rules engine
│   ├── __init__.py
│   ├── src/
│   │   ├── __init__.py
│   │   ├── coverage_rules.py        # Main rules engine
│   │   └── transforms.py           # Data transformations
│   └── test/
│       └── unit/
│           ├── __init__.py
│           └── test_coverage_rules.py
│
├── coverage_sql_pipelines/           # Original SQL pipelines
│   ├── __init__.py
│   ├── src/
│   │   ├── __init__.py
│   │   ├── sql_extract.py          # Feature extraction
│   │   └── data_pull.py            # Database connectivity
│   └── test/
│       └── unit/
│           ├── __init__.py
│           └── test_sql_extract.py
│
├── utils/                           # Original utilities
│   ├── __init__.py
│   ├── cryptography.py             # Encryption utilities
│   ├── detokenization.py           # Token management
│   └── sql_data_warehouse.py       # SQL warehouse utilities
│
├── modules/                         # New modular components
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py             # Main pipeline orchestrator
│   │   ├── loader.py               # Configuration loader
│   │   ├── monitor.py              # Performance monitoring
│   │   └── validator.py            # Data validation
│   ├── source/
│   │   ├── __init__.py
│   │   └── source_loader.py        # Multi-source data loading
│   ├── processor/
│   │   ├── __init__.py
│   │   └── parallel_rag.py         # Parallel RAG processing
│   └── storage/
│       ├── __init__.py
│       └── multi_writer.py         # Multi-destination output
│
├── custom_hooks/                    # Custom processing hooks
│   ├── __init__.py
│   ├── pre_processing.py           # Pre-processing hooks
│   └── post_processing.py          # Post-processing hooks
│
├── tests/                          # Comprehensive test suite
│   ├── __init__.py
│   ├── test_integration.py         # Integration tests
│   ├── test_pipeline.py            # Pipeline tests
│   ├── test_performance.py         # Performance tests
│   └── fixtures/
│       ├── __init__.py
│       └── test_data.py            # Test data fixtures
│
├── notebooks/                      # Execution notebooks
│   ├── BUILDING_COVERAGE_EXECUTION.ipynb        # Main execution
│   ├── BUILDING_COVERAGE_ANALYSIS.ipynb         # Results analysis
│   └── BUILDING_COVERAGE_DEVELOPMENT.ipynb      # Development notebook
│
├── config/                         # Configuration files
│   ├── development.py              # Development config
│   ├── staging.py                  # Staging config
│   ├── production.py               # Production config
│   └── default.py                  # Default config
│
├── deployment/                     # Deployment configuration
│   ├── __init__.py
│   ├── deploy.py                   # Deployment script
│   └── docker/
│       ├── Dockerfile
│       └── docker-compose.yml
│
├── docs/                          # Documentation
│   ├── architecture.md            # System architecture
│   ├── api_reference.md           # API documentation
│   ├── deployment_guide.md        # Deployment guide
│   └── user_guide.md              # User guide
│
└── logs/                          # Log files directory
    └── .gitkeep
```

## Features

### Core Capabilities
- **Multi-source Data Loading**: Parallel loading from AIP, Atlas, and Snowflake
- **Advanced RAG Processing**: GPT-powered claim analysis with parallel processing
- **Rule-based Classification**: Flexible business rule engine
- **Custom Hooks**: Pre and post-processing customization
- **Performance Monitoring**: Built-in performance tracking and optimization

### Technical Features
- **Parallel Processing**: ThreadPoolExecutor-based multi-threading
- **Error Handling**: Comprehensive error recovery and validation
- **Configuration Management**: Environment-specific configurations
- **Modular Architecture**: Clean separation of concerns
- **Backward Compatibility**: Works with existing Codebase 1 components

## Quick Start

### Installation
```bash
git clone <repository-url>
cd building_coverage_system
pip install -r requirements.txt
pip install -e .
```

### Basic Usage
```python
from modules.core.pipeline import CoveragePipeline
from coverage_configs.src.environment import DatabricksEnv

# Initialize environment
env = DatabricksEnv(databricks_dict)

# Create pipeline
pipeline = CoveragePipeline(
    credentials_dict=env.credentials_dict,
    sql_queries=env.sql_queries,
    rag_params=env.rag_params,
    crypto_spark=env.crypto_spark,
    logger=env.logger,
    SQL_QUERY_CONFIGS=env.SQL_QUERY_CONFIGS
)

# Execute pipeline
results = pipeline.run_pipeline(bldg_conditions)
```

### Notebook Execution
Open `notebooks/BUILDING_COVERAGE_EXECUTION.ipynb` in Jupyter and run all cells.

## Configuration

### Environment Variables
```bash
export AIP_SQL_SERVER="your-aip-server"
export ATLAS_SQL_SERVER="your-atlas-server"
export SNOWFLAKE_ACCOUNT="your-snowflake-account"
export GPT_API_URL="your-gpt-api-url"
export GPT_API_KEY="your-gpt-api-key"
```

### Custom Configuration
```python
config_overrides = {
    'pipeline': {
        'parallel_processing': {
            'enabled': True,
            'max_workers': 8
        },
        'hooks': {
            'pre_processing_enabled': True,
            'post_processing_enabled': True
        }
    }
}
```

## Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Integration tests
python -m pytest tests/test_integration.py -v

# Performance tests
python -m pytest tests/test_performance.py -v

# Pipeline tests
python -m pytest tests/test_pipeline.py -v
```

## Performance

### Expected Improvements
- **Data Loading**: 60% faster with parallel source loading
- **RAG Processing**: 50-70% faster with multi-threading
- **Overall Pipeline**: 40-50% faster end-to-end processing

### Monitoring
Built-in performance monitoring tracks:
- Operation execution times
- Memory usage
- Processing rates
- Error rates

## Architecture

The system uses a modular architecture combining:
- **Original Components**: Maintained for backward compatibility
- **Modular Components**: New parallel processing capabilities
- **Custom Hooks**: Flexible pre/post processing
- **Configuration Management**: Environment-specific settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License

## Support

For questions and support, please create an issue in the repository.