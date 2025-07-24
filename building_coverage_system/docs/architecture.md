# System Architecture

The Building Coverage System features a hybrid architecture that combines the proven functionality of the original Codebase 1 with modern RAG-enhanced AI capabilities.

## Overview

The system is designed with two complementary architectural layers:

1. **Original Codebase 1 Components** - Battle-tested modules for data processing and business logic
2. **New Modular Architecture** - Modern AI-powered components for enhanced analysis

## Architectural Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Building Coverage System                      │
├─────────────────────────────────────────────────────────────────┤
│                         API Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   REST API      │  │   GraphQL API   │  │   Batch API     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Processing Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  New Pipeline   │  │  RAG Enhanced   │  │ Classification  │ │
│  │  Orchestration  │  │  Processing     │  │    Service      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                 Original Codebase 1 Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Coverage Rules  │  │  SQL Pipelines  │  │   Coverage      │ │
│  │    Engine       │  │   (Primary,     │  │   Configs       │ │
│  │                 │  │   AIP, Atlas)   │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                     Data Layer                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Vector DB     │  │   SQL Server    │  │     Redis       │ │
│  │  (Embeddings)   │  │   (Claims)      │  │    (Cache)      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### New Modular Architecture

#### 1. Core Pipeline (`core/`)
- **PipelineOrchestrator**: Main processing coordinator
- **DataValidator**: Input validation and sanitization
- **ResultAggregator**: Combines results from multiple sources

#### 2. Document Processing (`document_processing/`)
- **DocumentProcessor**: Text extraction and preprocessing
- **ChunkingService**: Intelligent text chunking for RAG
- **ContentAnalyzer**: Document structure analysis

#### 3. Embedding Service (`embedding/`)
- **EmbeddingGenerator**: Text-to-vector conversion
- **ModelManager**: ML model lifecycle management
- **BatchProcessor**: Efficient batch embedding generation

#### 4. Search Engine (`search/`)
- **VectorSearch**: Semantic similarity search
- **IndexManager**: Vector index optimization
- **QueryProcessor**: Search query enhancement

#### 5. Classification Service (`classification/`)
- **CoverageClassifier**: ML-powered coverage determination
- **FeatureExtractor**: Claim feature extraction
- **ModelTrainer**: Classification model training

### Original Codebase 1 Components

#### 1. Coverage Configs (`coverage_configs/`)
- **ConfigManager**: Centralized configuration management
- **DatabaseConfig**: Database connection settings
- **ModelConfig**: ML model configurations
- **ProcessingConfig**: Processing parameters

#### 2. Coverage RAG Implementation (`coverage_rag_implementation/`)
- **RAGProcessor**: Retrieval-augmented generation
- **SimilarityMatcher**: Text similarity analysis
- **ContextRetriever**: Relevant context extraction

#### 3. Coverage Rules (`coverage_rules/`)
- **BusinessRulesEngine**: Rule evaluation engine
- **RuleManager**: Rule lifecycle management
- **TransformEngine**: Data transformation rules

#### 4. Coverage SQL Pipelines (`coverage_sql_pipelines/`)
- **SQLExtractor**: Multi-source data extraction
- **DataPuller**: Database connectivity and querying
- **PipelineManager**: ETL process coordination

#### 5. Utils (`utils/`)
- **CryptoManager**: Encryption and security
- **TokenManager**: Data tokenization for privacy
- **SQLDataWarehouse**: Large-scale data operations

## Data Flow

### 1. Input Processing
```
Raw Claim Text → Document Processing → Text Chunks → Embeddings
```

### 2. RAG Enhancement
```
Query → Vector Search → Context Retrieval → Augmented Processing
```

### 3. Classification Pipeline
```
Features → Business Rules → ML Classification → Coverage Determination
```

### 4. Data Integration
```
SQL Sources → ETL Pipeline → Feature Engineering → Model Input
```

## Integration Patterns

### 1. Hybrid Processing
The system seamlessly integrates original and new components:

```python
# Example integration
from building_coverage_system.coverage_rules import BusinessRulesEngine
from building_coverage_system.new_architecture.classification import CoverageClassifier

rules_engine = BusinessRulesEngine()
ml_classifier = CoverageClassifier()

# Combine rule-based and ML approaches
rule_result = rules_engine.evaluate(claim_data)
ml_result = ml_classifier.predict(claim_features)

final_result = combine_results(rule_result, ml_result)
```

### 2. Configuration Sharing
Both architectures share configuration through the unified config system:

```python
from building_coverage_system.coverage_configs import ConfigManager

config = ConfigManager()
config.load_config('production.yaml')

# Used by both original and new components
db_config = config.get_database_config('primary')
model_config = config.get_model_config()
```

### 3. Data Pipeline Integration
Original SQL pipelines feed the new RAG-enhanced processing:

```python
from building_coverage_system.coverage_sql_pipelines import SQLExtractor
from building_coverage_system.new_architecture.core import PipelineOrchestrator

extractor = SQLExtractor(credentials, queries)
orchestrator = PipelineOrchestrator()

# Extract data using original pipeline
raw_data = extractor.extract_building_coverage_features()

# Process using new architecture
results = orchestrator.process_batch(raw_data)
```

## Scalability Design

### Horizontal Scaling
- **Microservices**: Independent scaling of components
- **Load Balancing**: Distributed request handling
- **Container Orchestration**: Kubernetes deployment

### Vertical Scaling
- **Resource Optimization**: Memory and CPU efficiency
- **Batch Processing**: Large dataset handling
- **Caching**: Redis-based performance optimization

### Data Scaling
- **Partitioning**: Time-based and feature-based data splits
- **Indexing**: Optimized database and vector indexes
- **Archiving**: Historical data management

## Security Architecture

### Data Protection
- **Encryption**: At-rest and in-transit encryption
- **Tokenization**: PII protection through tokenization
- **Access Control**: Role-based permissions

### System Security
- **Authentication**: JWT-based API authentication
- **Authorization**: Fine-grained access control
- **Audit Logging**: Comprehensive activity tracking

## Performance Considerations

### Optimization Strategies
1. **Caching**: Multi-level caching (Redis, application, model)
2. **Batching**: Efficient batch processing for embeddings and predictions
3. **Parallel Processing**: Multi-threaded and multi-process execution
4. **Index Optimization**: Database and vector index tuning

### Monitoring and Metrics
- **Performance Metrics**: Response time, throughput, accuracy
- **Resource Monitoring**: CPU, memory, storage usage
- **Business Metrics**: Processing success rates, classification accuracy

## Deployment Architecture

### Environment Support
- **Development**: Docker Compose for local development
- **Staging**: Kubernetes with reduced resources
- **Production**: Full Kubernetes deployment with HA

### Infrastructure Components
- **Application Pods**: Main processing containers
- **Worker Pods**: Background task processing
- **Database**: PostgreSQL for metadata, SQL Server for claims
- **Cache**: Redis for session and result caching
- **Storage**: Persistent volumes for models and data

This architecture provides the flexibility to leverage existing proven components while enabling modern AI capabilities for enhanced claim analysis and coverage determination.