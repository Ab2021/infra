# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced SQL Agent System - A sophisticated AI-powered system that converts natural language queries into optimized SQL with memory-driven learning, multi-agent coordination, and automated visualizations. Built with Python, LangGraph, FastAPI, and Streamlit.

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

# Direct system usage
python main.py
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
- **Multi-Agent System**: Five specialized agents coordinated through LangGraph workflows
- **Three-Tier Memory**: Working memory (real-time), session memory (conversation), long-term memory (patterns)
- **LangGraph Orchestration**: Sophisticated workflow management with dynamic routing and error recovery
- **Snowflake Integration**: Optimized for Snowflake data warehouse with connection pooling

### Agent Specialization
1. **NLU Agent** (`agents/nlu_agent.py`) - Natural language understanding and intent extraction
2. **Schema Intelligence Agent** (`agents/schema_intelligence_agent.py`) - Database schema analysis and table relevance
3. **SQL Generator Agent** (`agents/sql_generator_agent.py`) - Template-based SQL generation with optimization
4. **Validation & Security Agent** (`agents/validation_security_agent.py`) - Query validation and security checks
5. **Visualization Agent** - Chart recommendations and dashboard creation

### Workflow Orchestration
The main workflow (`workflows/sql_workflow.py`) implements sophisticated routing logic:
- Dynamic routing based on confidence scores and processing results
- Error recovery with iteration limits and fallback strategies
- Quality assessment loops for result validation
- Memory integration at every processing step

### Memory System
Three-tier architecture using SQLite and FAISS (`memory/` directory):
- **Working Memory**: Real-time processing context and agent coordination state (in-memory)
- **Session Memory**: Conversation history and user preferences (SQLite: `session_memory.py`)
- **Long-term Memory**: Query patterns and schema insights (SQLite + FAISS: `long_term_memory.py`)

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

# Memory System (SQLite-based)
MEMORY_BACKEND=sqlite
SESSION_DB_PATH=data/session_memory.db
KNOWLEDGE_DB_PATH=data/knowledge_memory.db
REDIS_URL=redis://localhost:6379

# Vector Store (FAISS-based)
VECTOR_STORE_PROVIDER=faiss
VECTOR_STORE_PATH=data/vector_store
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Settings Management
Central configuration in `config/settings.py` using Pydantic with:
- Environment variable validation
- SQLite database path configuration
- FAISS vector store configuration
- LLM provider configuration
- Performance and security settings

## Key Files and Entry Points

- `main.py` - Main system orchestrator and entry point
- `workflows/sql_workflow.py` - LangGraph workflow implementation with sophisticated routing
- `memory/memory_manager.py` - Memory system coordination
- `database/snowflake_connector.py` - Snowflake database integration
- `api/fastapi_app.py` - REST API endpoints
- `ui/streamlit_app.py` - Web interface

## Development Patterns

### Adding New Agents
1. Create agent class in `agents/` directory
2. Implement required processing methods with memory integration
3. Register agent in workflow (`workflows/sql_workflow.py`)
4. Add corresponding routing logic and error handling
5. Write unit tests in `tests/unit/test_agents.py`

### Memory Integration
- All agents receive memory context for processing enhancement
- Update memory with processing results using `memory_manager.update_memory_from_processing()`
- Query contextual memories for similar past queries and learned patterns

### Error Handling
- Implement recovery strategies with iteration limits
- Use structured error history tracking
- Provide fallback processing paths in workflow routing

## Security Considerations

- **SQL Injection Prevention**: All database queries use parameterized statements
- **Path Traversal Protection**: Database paths are validated and restricted to working directory
- **Secure SQLite Configuration**: Foreign key constraints enabled, WAL mode, secure pragmas
- **Input Validation**: User inputs are validated for type, length, and format
- **Secure Credential Management**: Environment variables for sensitive data
- **Access Control**: Rate limiting and query validation capabilities

## Performance Optimization

- **Asynchronous Processing**: Throughout the system for scalability
- **SQLite Optimizations**: WAL mode, memory temp store, optimized cache settings
- **FAISS Vector Search**: Efficient similarity search for knowledge retrieval
- **Intelligent Caching**: Redis integration with configurable TTL
- **Resource Monitoring**: Configurable limits and performance tracking