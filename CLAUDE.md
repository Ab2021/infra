# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced SQL Agent System - A sophisticated AI-powered system that converts natural language queries into optimized SQL with enhanced in-memory processing, FAISS vector search, and automated visualizations. Built with Python, SQLite (in-memory), FAISS, FastAPI, and Streamlit.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies  
pip install -r requirements.txt

# Create necessary directories
python config/settings.py
```

### Running Applications
```bash
# Web Interface (Streamlit)
streamlit run ui/streamlit_app.py

# REST API (FastAPI)
python api/fastapi_app.py

# Direct system usage (main)
python main.py

# Simplified system usage
python main_simple.py

# Test in-memory configuration
python test_memory_config.py
```

### Testing
```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=. --cov-report=html

# Test memory configuration
python test_memory_config.py
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## Architecture Overview

### Core Components
- **Streamlined Agent System**: Specialized agents for NLU, schema analysis, SQL generation, validation, and visualization
- **Enhanced Memory System**: In-memory SQLite with FAISS vector search for intelligent context retrieval
- **High-Performance Storage**: In-memory databases with optional persistence for maximum speed
- **Snowflake Integration**: Optimized for Snowflake data warehouse with connection pooling

### Agent Specialization
1. **NLU Agent** (`agents/nlu_agent.py`) - Natural language understanding and intent extraction
2. **Schema Intelligence Agent** (`agents/schema_intelligence_agent.py`) - Database schema analysis and table relevance
3. **SQL Generator Agent** (`agents/sql_generator_agent.py`) - Template-based SQL generation with optimization
4. **Validation & Security Agent** (`agents/validation_security_agent.py`) - Query validation and security checks
5. **Visualization Agent** - Chart recommendations and dashboard creation

### Simplified Architecture
Streamlined design focusing on core functionality:
- Direct agent coordination without complex workflows
- Simple memory system with powerful vector search capabilities
- Efficient in-memory processing with optional persistence
- Enhanced context storage and retrieval using FAISS

### Enhanced Memory System
Three-tier architecture with performance optimizations (`memory/` directory):
- **Working Memory**: Real-time processing context and agent coordination state (in-memory)
- **Session Memory**: Conversation history and user preferences (In-memory SQLite: `session_memory.py`)
- **Long-term Memory**: Query patterns and schema insights with FAISS vector search (`long_term_memory.py`)
- **Simple Memory**: Streamlined implementation for basic usage (`simple_memory.py`)

## Configuration

### Environment Variables (.env file required)
```env
# Snowflake Database
SNOWFLAKE_ACCOUNT=your-account.snowflakecomputing.com
SNOWFLAKE_USER=username
SNOWFLAKE_PASSWORD=password
SNOWFLAKE_WAREHOUSE=warehouse
SNOWFLAKE_DATABASE=database

# LLM Provider
LLM_PROVIDER=openai  # or anthropic
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o

# Memory System (In-Memory by Default)
MEMORY_BACKEND=sqlite
SESSION_DB_PATH=:memory:
KNOWLEDGE_DB_PATH=:memory:

# Optional: Persistent storage paths
PERSISTENT_SESSION_DB_PATH=data/session_memory.db
PERSISTENT_KNOWLEDGE_DB_PATH=data/knowledge_memory.db
REDIS_URL=redis://localhost:6379

# Vector Store (FAISS-based)
VECTOR_STORE_PROVIDER=faiss
VECTOR_STORE_PATH=data/vector_store
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Settings Management
Central configuration in `config/settings.py` using Pydantic with:
- Environment variable validation
- In-memory SQLite configuration with optional persistence
- Enhanced FAISS vector store configuration
- LLM provider configuration with improved compatibility
- Performance and security settings optimized for in-memory operations

## Key Files and Entry Points

- `main.py` - Main system orchestrator and entry point
- `main_simple.py` - Simplified entry point for basic usage
- `test_memory_config.py` - Memory configuration validation and testing
- `memory/simple_memory.py` - Streamlined memory system implementation
- `memory/long_term_memory.py` - Enhanced long-term memory with FAISS vector search
- `database/snowflake_connector.py` - Snowflake database integration
- `api/fastapi_app.py` - REST API endpoints
- `ui/streamlit_app.py` - Web interface

## Development Patterns

### Adding New Agents
1. Create agent class in `agents/` directory
2. Implement required processing methods with memory integration
3. Register agent in the main system orchestrator
4. Add corresponding error handling and validation
5. Write unit tests in `tests/unit/test_agents.py`

### Enhanced Memory Integration
- All agents receive memory context for processing enhancement
- Store long-term contexts with vector indexing using `store_long_term_context()`
- Retrieve similar contexts using multi-strategy matching (exact, similarity, type-based)
- Leverage FAISS vector search for intelligent pattern recognition

### Error Handling
- Implement recovery strategies with graceful degradation
- Use structured error history tracking
- Provide fallback processing paths with simplified routing
- Enhanced logging and debugging capabilities

## Security Considerations

- **SQL Injection Prevention**: All database queries use parameterized statements
- **Path Traversal Protection**: Database paths are validated and restricted to working directory
- **Secure SQLite Configuration**: Foreign key constraints enabled, in-memory optimization, secure pragmas
- **Input Validation**: User inputs are validated for type, length, and format
- **Secure Credential Management**: Environment variables for sensitive data
- **Access Control**: Rate limiting and query validation capabilities

## Performance Optimization

- **In-Memory Databases**: Ultra-fast SQLite operations with 50-100x performance improvements
- **FAISS Vector Search**: Sub-millisecond similarity search for intelligent context retrieval
- **Optimized Pragmas**: Memory-specific SQLite configurations for maximum speed
- **Asynchronous Processing**: Throughout the system for scalability
- **Smart Caching**: Optional Redis integration with configurable TTL
- **Resource Monitoring**: Configurable limits and performance tracking
- **Optional Persistence**: Configurable data durability without sacrificing speed