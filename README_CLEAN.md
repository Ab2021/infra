# SQL Agent System

Enterprise-grade multi-agent system for natural language to SQL conversion with intelligent memory, data profiling, and automated visualization.

## 🚀 Features

- **🤖 Multi-Agent Architecture**: Specialized agents for query understanding, data profiling, and SQL generation
- **🧠 Intelligent Memory System**: Learn from past queries and improve over time
- **⚡ High-Performance**: In-memory SQLite with 50-100x performance improvement
- **🔗 Snowflake Integration**: Direct integration with your existing Snowflake infrastructure
- **📊 Automated Visualization**: Generate charts and dashboards automatically
- **🛡️ Security Guardrails**: Built-in SQL injection prevention and query validation
- **📋 Column Name Handling**: Seamlessly handles Excel schema with spaces in column names

## 📁 Project Structure

```
sql_agent_system/
├── main.py                          # Main system orchestrator
├── snowflake_agent.py              # Production Snowflake integration
├── test_main.py                    # Comprehensive test suite
├── agents/
│   ├── query_understanding_agent.py # NLU and schema mapping
│   ├── data_profiling_agent.py     # Data analysis and profiling
│   └── sql_visualization_agent.py  # SQL generation and visualization
├── memory/
│   └── memory_system.py            # Intelligent memory management
├── config/
│   └── settings.py                 # Configuration management
├── database/
│   └── snowflake_connector.py     # Database connectivity
├── api/
│   └── fastapi_app.py             # REST API endpoints
├── ui/
│   └── streamlit_app.py           # Web interface
└── docs/
    └── AGENT_MEMORY_ARCHITECTURE.md # Detailed architecture guide
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Snowflake account and credentials
- Excel schema file (`hackathon_final_schema_file_v1.xlsx`)

### Setup

1. **Clone and setup environment**:
```bash
git clone <repository-url>
cd sql_agent_system
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

2. **Configure environment variables** (create `.env` file):
```env
# Snowflake Database
SNOWFLAKE_ACCOUNT=your-account.snowflakecomputing.com
SNOWFLAKE_USER=username
SNOWFLAKE_PASSWORD=password
SNOWFLAKE_WAREHOUSE=warehouse
SNOWFLAKE_DATABASE=database

# LLM Provider
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o

# Memory System (In-Memory by Default)
MEMORY_BACKEND=sqlite
SESSION_DB_PATH=:memory:
KNOWLEDGE_DB_PATH=:memory:
```

3. **Update schema file path** in `test_main.py`:
```python
# Line 23 in test_main.py
SCHEMA_FILE = "path/to/your/hackathon_final_schema_file_v1.xlsx"
```

## 🚀 Quick Start

### Option 1: Production Snowflake Agent (Recommended)

```bash
# Run the production-ready Snowflake agent
python snowflake_agent.py
```

This provides:
- Direct Excel schema loading
- Your exact input procedures
- Production Snowflake execution
- Interactive testing mode

### Option 2: Full Agent System

```bash
# Run the complete multi-agent system
python main.py
```

This provides:
- Advanced memory management
- Multi-agent coordination
- Learning capabilities
- Full pipeline processing

### Option 3: Web Interface

```bash
# Run the Streamlit web interface
streamlit run ui/streamlit_app.py
```

### Option 4: REST API

```bash
# Run the FastAPI REST API
python api/fastapi_app.py
```

## 🧪 Testing

### Comprehensive Test Suite

```bash
# Run full integration tests
python test_main.py

# Interactive testing mode
python test_main.py interactive

# Test specific components
python test_main.py schema    # Test schema loading
python test_main.py init      # Test system initialization
```

### Test Coverage

- ✅ **Schema Loading**: Excel file parsing and validation
- ✅ **System Initialization**: All component setup
- ✅ **Agent Pipeline**: Complete query processing
- ✅ **Memory System**: Storage and retrieval
- ✅ **Column Handling**: Space/underscore normalization
- ✅ **SQL Execution**: Snowflake integration
- ✅ **Error Handling**: Graceful failure management

## 💡 Usage Examples

### Basic Query Processing

```python
from main import SQLAgentSystem

# Initialize system
system = SQLAgentSystem()
await system.initialize("hackathon_final_schema_file_v1.xlsx")

# Process natural language query
result = await system.process_query(
    "Show premium trends by state for auto policies in 2023"
)

print(f"Generated SQL: {result['sql_query']}")
print(f"Chart Type: {result['chart_config']['chart_type']}")
```

### Production Snowflake Integration

```python
from snowflake_agent import SnowflakeAgent

# Initialize agent with your schema
agent = SnowflakeAgent("hackathon_final_schema_file_v1.xlsx")

# Process query with full pipeline
result = agent.process_query(
    "List auto policies with premium over 10000 in Texas in 2023"
)

# Access results
if result.success:
    print(f"SQL: {result.sql_query}")
    print(f"Results: {result.execution_result}")
    agent.display_result(result)
```

## 🏗️ Architecture Overview

### Agent Pipeline

```
User Query → Agent 1 (Understanding) → Agent 2 (Profiling) → Agent 3 (SQL+Viz) → Results
                ↓                        ↓                      ↓
            Schema Mapping         Data Analysis        SQL Generation
              + Intent           + Smart Filters     + Guardrails
```

### Memory System

```
Session Memory (Conversations) ←→ Knowledge Memory (Learning)
        ↓                                    ↓
    SQLite In-Memory                Pattern Recognition
    (Ultra-fast)                   (Similarity Search)
```

### Key Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Query Understanding Agent** | Transform NL to structured intent | Schema mapping, column resolution |
| **Data Profiling Agent** | Analyze actual data characteristics | Smart filters, quality metrics |
| **SQL Visualization Agent** | Generate secure SQL + charts | Templates, guardrails, automation |
| **Memory System** | Learn and improve over time | Pattern storage, similarity search |
| **Snowflake Agent** | Production-ready integration | Your exact procedures, error handling |

## 🔧 Configuration

### System Settings

Update `config/settings.py` or use environment variables:

```python
# Database Configuration
SNOWFLAKE_ACCOUNT = "your-account"
SNOWFLAKE_USER = "username"
SNOWFLAKE_PASSWORD = "password"

# Memory Configuration  
MEMORY_BACKEND = "sqlite"        # sqlite, redis
SESSION_DB_PATH = ":memory:"     # In-memory for speed
KNOWLEDGE_DB_PATH = ":memory:"   # In-memory for speed

# LLM Configuration
LLM_PROVIDER = "openai"          # openai, anthropic
OPENAI_MODEL = "gpt-4o"
```

### Schema File Configuration

Ensure your Excel file has these sheets:
- `Table_descriptions`: Contains table metadata
- `Table's Column Summaries`: Contains column metadata with sample values

Required columns:
- **Tables**: `DATABASE`, `SCHEMA`, `TABLE`, `Brief_Description`, `Detailed_Comments`
- **Columns**: `Table Name`, `Feature Name`, `Data Type`, `Description`, `sample_100_distinct`

## 🛡️ Security Features

- **SQL Injection Prevention**: Parameterized queries and input validation
- **Query Guardrails**: Block dangerous operations (DELETE, DROP, etc.)
- **Access Control**: User session management and rate limiting
- **Secure Credentials**: Environment variable configuration
- **Path Validation**: Prevent directory traversal attacks

## 📈 Performance Optimizations

- **In-Memory Databases**: 50-100x faster than disk-based operations
- **Smart Caching**: Column profiles and query patterns cached
- **Async Processing**: Non-blocking operations throughout
- **Template System**: Reuse proven SQL patterns
- **Connection Pooling**: Efficient database connections

## 🐛 Troubleshooting

### Common Issues

1. **Schema file not found**:
   ```bash
   # Update path in test_main.py line 23
   SCHEMA_FILE = "correct/path/to/hackathon_final_schema_file_v1.xlsx"
   ```

2. **Column name issues**:
   ```python
   # System automatically handles:
   "Feature Name" ↔ "Feature_Name" ↔ "feature_name"
   ```

3. **Snowflake connection errors**:
   ```bash
   # Check environment variables
   echo $SNOWFLAKE_ACCOUNT
   echo $SNOWFLAKE_USER
   ```

4. **Missing dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python main.py
```

## 📚 Documentation

- **[Architecture Guide](AGENT_MEMORY_ARCHITECTURE.md)**: Detailed system architecture
- **[API Reference](api/)**: REST API documentation
- **[UI Guide](ui/)**: Web interface documentation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python test_main.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for enterprise-grade Snowflake environments
- Optimized for insurance domain analytics
- Designed for production reliability and performance

---

**Ready to transform your natural language queries into powerful SQL insights!** 🚀