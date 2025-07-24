# Building Coverage System Documentation

Welcome to the Building Coverage System documentation. This system combines the original Codebase 1 functionality with modern RAG-enhanced AI capabilities for intelligent insurance claim analysis and building coverage determination.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [API Reference](#api-reference)
- [Configuration Guide](#configuration-guide)
- [Development Guide](#development-guide)
- [Deployment Guide](#deployment-guide)
- [User Guide](#user-guide)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/company/building-coverage-system.git
cd building-coverage-system

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings

# Run the system
python -m building_coverage_system.main
```

### Docker Quick Start

```bash
# Using Docker Compose
docker-compose up -d

# The system will be available at http://localhost:8000
```

## Architecture Overview

The Building Coverage System consists of two main components:

### 1. Original Codebase 1 Components
- **Coverage Configs**: Configuration management for database connections and processing parameters
- **Coverage RAG Implementation**: RAG-enhanced text analysis and similarity matching
- **Coverage Rules**: Business rules engine for automated coverage determination
- **Coverage SQL Pipelines**: Data extraction and processing from multiple SQL sources
- **Utils**: Cryptography, tokenization, and data warehouse utilities

### 2. New Modular Architecture
- **Core Pipeline**: Main processing pipeline orchestration
- **Document Processing**: Advanced document analysis and chunking
- **Embedding Service**: Text embedding generation and management
- **Search Engine**: Vector-based similarity search and retrieval
- **Classification Service**: ML-powered coverage classification
- **API Layer**: RESTful API for system integration

## Key Features

- **Hybrid Architecture**: Combines proven original codebase with modern AI capabilities
- **RAG Enhancement**: Retrieval-augmented generation for improved accuracy
- **Multi-source Data**: Extracts from primary, AIP, and Atlas databases
- **Advanced Security**: Encryption, tokenization, and secure credential management
- **Scalable Processing**: Batch processing and parallel execution
- **Business Rules**: Configurable rules engine for coverage determination
- **Comprehensive Testing**: Unit tests, integration tests, and performance tests

## Directory Structure

```
building_coverage_system/
├── docs/                           # Documentation
├── building_coverage_system/       # Main application code
│   ├── new_architecture/           # Modern RAG-enhanced components
│   ├── coverage_configs/           # Configuration management
│   ├── coverage_rag_implementation/ # RAG text analysis
│   ├── coverage_rules/             # Business rules engine
│   ├── coverage_sql_pipelines/     # SQL data processing
│   └── utils/                      # Utility functions
├── tests/                          # Test suite
├── deploy/                         # Deployment configurations
├── notebooks/                      # Jupyter notebooks
└── config/                         # Configuration files
```

## Getting Started

1. **[Installation Guide](installation.md)** - Set up your development environment
2. **[Configuration Guide](configuration.md)** - Configure the system for your environment
3. **[API Reference](api-reference.md)** - Learn about the REST API endpoints
4. **[User Guide](user-guide.md)** - How to use the system effectively
5. **[Development Guide](development.md)** - Contributing to the codebase

## Support

- **Issues**: Report bugs and feature requests on GitHub
- **Documentation**: Check the docs/ directory for detailed guides
- **Examples**: See notebooks/ for usage examples

## License

This project is proprietary software. All rights reserved.