# Advanced SQL Agent System

ðŸ¤– **Transform natural language into SQL queries with intelligent memory and visualization**

An sophisticated AI-powered system that converts natural language queries into optimized SQL, featuring memory-driven learning, multi-agent coordination, and automated visualizations.

## ðŸŒŸ Key Features

- **Natural Language to SQL**: Convert plain English into optimized SQL queries
- **Memory-Driven Intelligence**: System learns from interactions and improves over time
- **Multi-Agent Architecture**: Specialized agents for different aspects of query processing
- **Automated Visualizations**: Generate charts and dashboards automatically
- **Snowflake Integration**: Optimized for Snowflake data warehouse
- **Real-time Processing**: LangGraph-powered workflow orchestration
- **REST API**: Programmatic access to all capabilities
- **Web Interface**: User-friendly Streamlit dashboard

## ðŸ—ï¸ Architecture Overview

The system uses a sophisticated three-tier memory architecture with specialized agents:

### Memory Tiers
- **Working Memory**: Real-time processing context
- **Session Memory**: Conversation context across queries  
- **Long-term Memory**: Accumulated knowledge and patterns

### Specialized Agents
- **NLU Agent**: Natural language understanding
- **Schema Intelligence Agent**: Database architecture expert
- **SQL Generator Agent**: Query crafting specialist
- **Validation & Security Agent**: Quality assurance
- **Visualization Agent**: Chart and dashboard creator

## ðŸš€ Quick Start

### 1. Installation

```bash
# Clone or create project
git clone <your-repo> # or use the PowerShell generator script
cd advanced_sql_agent_system

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your settings
notepad .env  # Windows
nano .env     # Linux/Mac
```

Required configuration:
- Snowflake connection details
- OpenAI or Anthropic API key
- Database for memory storage (optional)

### 3. Run the Application

#### Web Interface (Streamlit)
```bash
streamlit run ui/streamlit_app.py
```

#### REST API (FastAPI)
```bash
python api/fastapi_app.py
```

#### Direct Usage
```python
from main import SQLAgentSystem

# Initialize system
system = SQLAgentSystem()

# Process query
result = await system.process_query(
    "Show me total sales by product category for this quarter"
)

print(result)
```

## ðŸ’¡ Example Queries

The system handles a wide variety of natural language queries:

```python
# Sales Analytics
"Show me total sales by product category for this quarter"
"What are the top 10 customers by revenue?"
"Compare this year's performance to last year"

# Performance Analysis
"Which regions are underperforming?"
"Show me monthly trends for the past year"
"What products have declining sales?"

# Customer Insights
"Who are our most valuable customers?"
"Show customer retention rates by segment"
"What's the average order value by customer type?"
```

## ðŸ”§ Configuration Options

### Database Settings
```env
SNOWFLAKE_ACCOUNT="your-account.snowflakecomputing.com"
SNOWFLAKE_USER="your-username"
SNOWFLAKE_PASSWORD="your-password"
SNOWFLAKE_WAREHOUSE="your-warehouse"
SNOWFLAKE_DATABASE="your-database"
```

### LLM Provider
```env
LLM_PROVIDER="openai"  # or "anthropic"
OPENAI_API_KEY="sk-your-api-key"
OPENAI_MODEL="gpt-4o"
```

### Memory System
```env
MEMORY_BACKEND="sqlite"
SESSION_DB_PATH="data/session_memory.db"
KNOWLEDGE_DB_PATH="data/knowledge_memory.db"
REDIS_URL="redis://localhost:6379"
VECTOR_STORE_PROVIDER="faiss"
VECTOR_STORE_PATH="data/vector_store"
EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

## ðŸ“Š API Endpoints

### Query Processing
```http
POST /query
{
    "query": "Show me total sales by category",
    "user_id": "user123",
    "preferences": {"visualization": "auto"}
}
```

### Memory Insights
```http
GET /memory/insights/{user_id}
```

### Schema Information
```http
GET /schema/tables
```

### Example Queries
```http
GET /examples
```

## ðŸ§ª Testing

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=. --cov-report=html
```

## ðŸ“ˆ Performance Optimization

### Query Optimization
- Automatic query plan analysis
- Index usage recommendations
- Performance warning detection

### Memory Efficiency
- Smart caching strategies
- Connection pooling
- Asynchronous processing

### Scalability
- Configurable concurrency limits
- Rate limiting
- Resource monitoring

## ðŸ”’ Security Features

- SQL injection prevention
- Query validation and sanitization
- Access control and rate limiting
- Secure credential management

## ðŸ“ Development

### Project Structure
```
advanced_sql_agent_system/
â”œâ”€â”€ agents/                 # Specialized agent implementations
â”œâ”€â”€ memory/                # Memory system components  
â”œâ”€â”€ workflows/             # LangGraph workflow definitions
â”œâ”€â”€ database/              # Database connectors
â”œâ”€â”€ api/                   # REST API implementation
â”œâ”€â”€ ui/                    # Streamlit web interface
â”œâ”€â”€ config/                # Configuration management
â”œâ”€â”€ tests/                 # Test suites
â”œâ”€â”€ docs/                  # Documentation
â””â”€â”€ examples/              # Usage examples
```

### Adding New Agents
1. Create agent class in `agents/`
2. Implement required methods
3. Register in workflow
4. Add tests

### Extending Memory System
1. Define new memory structures
2. Implement storage/retrieval logic
3. Update memory manager
4. Add integration tests

## ðŸ¤ Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## ðŸ“œ License

This project is licensed under the MIT License - see the LICENSE file for details.

## ðŸ†˜ Support

- **Documentation**: `/docs` directory
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## ðŸŽ¯ Roadmap

- [ ] Support for additional databases (PostgreSQL, BigQuery)
- [ ] Enhanced vector search capabilities with advanced FAISS indices
- [ ] Advanced visualization types
- [ ] Real-time query streaming
- [ ] Multi-tenant support
- [ ] Advanced security features
- [ ] Performance analytics dashboard

---

Built with â¤ï¸ using LangGraph, Streamlit, FastAPI, and advanced AI technologies.
