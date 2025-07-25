# Core Functionalities of Building Coverage Match System

## Overview
This system is designed for building coverage matching and prediction using insurance claim data. It consists of 4 main modules that work together to extract, process, and analyze insurance claims data to determine building coverage indicators.

---

## 1. Coverage Configs Module (`coverage_configs`)

### Purpose
Central configuration management for the entire system, providing credentials, SQL queries, prompts, and validation.

### Core Components

#### 1.1 Environment Management (`environment.py`)
- **DatabricksEnv Class**: Main environment handler for Databricks-based operations
- **Functionalities**:
  - Validates and manages databricks dictionary parameters
  - Provides centralized access to credentials, SQL queries, prompts, and RAG parameters
  - Configures logging for py4j operations
  - Manages building rules and validation parameters

#### 1.2 Credentials Management (`credentials.py`)
- **Functions**:
  - `get_adls_config()`: Configures Azure Data Lake Storage connections
  - `get_atlas_adls_config()`: Manages Atlas-specific ADLS configurations
  - `get_snowflake_config()`: Sets up Snowflake database connections
  - `get_credentials()`: Centralized credential retrieval combining all data sources

#### 1.3 SQL Query Management (`sql.py`)
- **Query Repository**: Contains pre-defined SQL queries for different data sources
  - `claim_line_prtcpt_feature`: Main claim and participant data extraction
  - `filenotes_snowflake_query`: File notes retrieval from Snowflake
  - `snowflake_query`: Policy and coverage data from Snowflake
  - `bar_atlas_query_1`: Atlas data warehouse queries
- **Validation**: Ensures all required queries are present using Pydantic validation

#### 1.4 Prompt Engineering (`prompts.py`)
- **GPT Prompt Generation**: Creates detailed prompts for building coverage analysis
- **Key Features**:
  - 22 different building indicator extractions
  - Complex building loss amount calculation logic
  - Validation prompts for accuracy checking
  - Structured JSON output formatting

#### 1.5 RAG Parameters (`rag_params.py`)
- **Configuration for RAG Processing**:
  - GPT API parameters (temperature, tokens, penalties)
  - Text chunking parameters
  - Model paths and configurations
  - Query templates for retrieval

#### 1.6 Building Rules (`bldg_rules.py`)
- **Rule Definitions**: Simple binary classification rules
  - `BLDG_Indicator_no`: Default classification
  - `BLDG_Indicator_yes`: Positive building loss indicator

#### 1.7 Validation Classes (`base_class.py`)
- **Pydantic Validators**:
  - `ValidateRAGParamsKeys`: Ensures required RAG parameters
  - `ValidateSQLQueries`: Validates SQL query completeness
  - `ValidateDatabricksParams`: Comprehensive parameter validation

---

## 2. Coverage RAG Implementation Module (`coverage_rag_implementation`)

### Purpose
Implements Retrieval-Augmented Generation for extracting building coverage insights from claim text data.

### Core Components

#### 2.1 RAG Predictor (`rag_predictor.py`)
- **RAGPredictor Class**: Main orchestrator for RAG-based analysis
- **Key Methods**:
  - `get_summary()`: Processes claim data and generates summaries with 22 building indicators
  - Filters valid claim text data
  - Batch processes claims with rate limiting (4-second delays)
  - Returns structured DataFrame with building damage indicators

#### 2.2 RAG Processor (`rag_processor.py`)
- **RAGProcessor Class**: Core RAG implementation
- **Functionalities**:
  - Text preprocessing and deduplication
  - GPT API integration for content generation
  - Structured extraction of 22 building indicators:
    - Financial: `BLDG_LOSS_AMOUNT`
    - Damage Types: Exterior, Interior, Roof, Plumbing, Electrical, etc.
    - Operational: Tenable, Unoccupiable, Complete Loss indicators
    - Contextual: Primary Building, Adjacent/Direct Origin indicators

#### 2.3 Text Processing (`text_processor.py`)
- **TextProcessor Class**: Comprehensive text cleaning and processing
- **Key Features**:
  - Duplicate sentence removal
  - Contraction expansion
  - Important keyword filtering for insurance claims
  - Regular expression-based data extraction
  - Numeric value extraction and normalization

#### 2.4 Chunk Splitting (`chunk_splitter.py`)
- **TextChunkSplitter Class**: Intelligent text segmentation
- **Capabilities**:
  - Multiple splitting strategies (word, sentence, passage)
  - Configurable overlap and length parameters
  - Sentence boundary respect
  - Windowed chunk creation for context preservation

#### 2.5 GPT API Integration (`gpt_api.py`)
- **GptApi Class**: Interface to enterprise GPT services
- **Features**:
  - Token-based authentication
  - Retry logic with configurable attempts
  - Parameter validation using Pydantic
  - Error handling and timeout management

#### 2.6 LLM Configuration (`llm_config.py`)
- **LLMConfig Class**: Manages GPT API configurations
- **Responsibilities**:
  - Environment variable validation
  - API endpoint management
  - Authentication parameter handling

---

## 3. Coverage Rules Module (`coverage_rules`)

### Purpose
Applies business rules and transformations to processed claim data for final classification and formatting.

### Core Components

#### 3.1 Coverage Rules Engine (`coverage_rules.py`)
- **CoverageRules Class**: Rule-based classification system
- **Functionalities**:
  - Dynamic rule application using pandas `eval()`
  - Configurable condition-classification mapping
  - Support for complex boolean expressions
  - Error handling for syntax and name errors

#### 3.2 Data Transformations (`coverage_transformations.py`)
- **DataFrameTransformations Class**: Final data preparation for output
- **Key Methods**:
  - `_filter_needed_columns()`: Selects only required columns for final output
  - `rename_columns()`: Standardizes column naming conventions
  - `select_and_rename_bldg_predictions_for_db()`: Comprehensive output formatting
- **Output Formatting**:
  - Creates period date combinations
  - Maps binary indicators to Y/N format
  - Generates GPT_BLDG_STATUS based on classification results
  - Produces 46 standardized output columns for database insertion

---

## 4. Coverage SQL Pipelines Module (`coverage_sql_pipelines`)

### Purpose
Handles data extraction and pipeline operations from multiple data sources (SQL Server, Snowflake, Atlas).

### Core Components

#### 4.1 Feature Extractor (`sql_extract.py`)
- **FeatureExtractor Class**: Main data extraction orchestrator
- **Key Methods**:
  - `_get_claim_line_df()`: Extracts claim and participant data
  - `_get_filenotes_df()`: Retrieves and processes claim file notes
  - `get_feature_df()`: Combines claim and file note data
  - `clean_claim()`: Advanced text cleaning for claim notes

#### 4.2 Data Pipeline (`datapull.py`)
- **Datapull Class**: Multi-source data extraction engine
- **Data Sources**:
  - **AIP**: Primary SQL Data Warehouse
  - **ATLAS**: Secondary data warehouse for enhanced claim data
  - **SNOWFLAKE**: Policy and coverage information
- **Key Features**:
  - Multi-stage data joining and aggregation
  - Policy period matching with loss dates
  - Complex aggregation logic for claim-level summaries
  - Spark-based processing for large datasets

#### 4.3 Data Processing Pipeline
The pipeline follows this sequence:
1. **Atlas Query**: Extract claim identifiers and coverage information
2. **AIP Integration**: Join with detailed claim and participant data
3. **Snowflake Integration**: Add policy and coverage premium data
4. **Policy Period Matching**: Ensure claims fall within active policy periods
5. **Data Aggregation**: Create claim-level summaries with concatenated details

---

## System Integration Flow

### 1. Configuration Phase
- `coverage_configs` module initializes all system parameters
- Validates credentials and establishes database connections
- Loads SQL queries, prompts, and processing parameters

### 2. Data Extraction Phase
- `coverage_sql_pipelines` extracts raw claim and policy data
- Multi-source joining creates comprehensive claim dataset
- File notes are retrieved and cleaned

### 3. AI Processing Phase
- `coverage_rag_implementation` processes claim text using GPT
- Extracts 22 building-related indicators
- Generates summaries and loss descriptions

### 4. Rule Application Phase
- `coverage_rules` applies business logic for final classification
- Transforms data into standardized output format
- Prepares data for database insertion or reporting

---

## Key Technical Features

### Scalability
- Spark-based processing for large datasets
- Batch processing with rate limiting
- Configurable sampling ratios for testing

### Reliability
- Comprehensive error handling and logging
- Pydantic validation throughout the system
- Retry logic for API calls

### Flexibility
- Configurable rules and parameters
- Multiple data source support
- Extensible prompt and rule systems

### Security
- Credential management through secure vaults
- Encryption/decryption capabilities
- Service principal authentication

---

## Output Schema

The system produces a comprehensive dataset with 46 columns including:
- **Claim Identifiers**: Claim Number, Status, Adjuster Name
- **Policy Information**: Original Policy Number, Premium Amounts, Coverage Details
- **Financial Data**: Loss amounts, payment information
- **Building Indicators**: 22 specific building damage and operational indicators
- **Metadata**: Processing timestamps, validation information

This system enables automated building coverage determination for insurance claims processing, reducing manual review time and improving consistency in coverage decisions.