# 🤖 Advanced SQL Agent System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/langgraph-latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Transform natural language into intelligent SQL queries with AI-powered visualization and real-time dashboard streaming.

## 🌟 Overview

The Advanced SQL Agent System is a sophisticated AI-powered platform that converts natural language queries into optimized SQL with intelligent visualizations. Built on a multi-agent architecture using LangGraph, it provides enterprise-grade features including memory-driven learning, real-time streaming dashboards, and automated chart recommendations.

### ✨ Key Features

- **🧠 Natural Language Understanding**: Advanced NLP with entity extraction and intent recognition
- **🗄️ Intelligent Schema Analysis**: Deep database understanding with relationship mapping
- **⚡ Optimized SQL Generation**: Template-based generation with performance optimization
- **🛡️ Enterprise Security**: Comprehensive validation and SQL injection prevention
- **📊 AI-Powered Visualizations**: Automatic chart recommendations with code generation
- **🎯 Real-Time Streaming**: Live agent status and progressive result display
- **🧮 Memory-Driven Learning**: Three-tier memory system for continuous improvement
- **📱 Professional Dashboard**: Modern UI with responsive design and animations

---

## 🏗️ Architecture

### Multi-Agent System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  📱 Advanced Dashboard    │  🖥️ Streamlit UI    │  🔌 FastAPI  │
└─────────────────────────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────┐
│                   LANGGRAPH WORKFLOW                        │
├─────────────────────────────────────────────────────────────┤
│  🔄 Orchestration Engine with Dynamic Routing & Recovery    │
└─────────────────────────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────┐
│                    AGENT ECOSYSTEM                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│  🧠 NLU Agent   │  🗄️ Schema Agent │  ⚡ SQL Generator       │
│  Entity Extract │  Table Analysis │  Query Optimization     │
│  Intent Recog   │  Pattern Detect │  Template Matching      │
├─────────────────┼─────────────────┼─────────────────────────┤
│  🛡️ Validator   │  📊 Visualizer  │  🧮 Memory Manager      │
│  Security Check │  Chart Recommend│  Learning & Context     │
│  Performance    │  Code Generation│  Session Tracking       │
└─────────────────┴─────────────────┴─────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────┐
│                   INFRASTRUCTURE LAYER                      │
├─────────────────┬─────────────────┬─────────────────────────┤
│  🗃️ Snowflake   │  💾 SQLite      │  🔍 FAISS Vector Store  │
│  Data Warehouse │  Memory Storage │  Similarity Search      │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Core Components

#### 🎯 **Agent Specialization**
- **NLU Agent**: Converts natural language to structured intent
- **Schema Intelligence Agent**: Analyzes database schema and relationships
- **SQL Generator Agent**: Creates optimized SQL with visualization metadata
- **Validation & Security Agent**: Ensures query safety and performance
- **Visualization Agent**: Recommends charts and generates plotting code

#### 🧮 **Three-Tier Memory System**
- **Working Memory**: Real-time processing context and agent coordination
- **Session Memory**: Conversation history and user preferences (SQLite)
- **Long-term Memory**: Query patterns and schema insights (SQLite + FAISS)

#### 🔄 **LangGraph Workflow**
- Dynamic routing based on confidence scores and processing results
- Error recovery with iteration limits and fallback strategies
- Quality assessment loops for result validation
- Memory integration at every processing step

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Access to a Snowflake database (or modify for your database)
- OpenAI API key or Anthropic API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/advanced_sql_agent_system.git
   cd advanced_sql_agent_system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env with your credentials
   nano .env
   ```

5. **Initialize the system**
   ```bash
   python config/settings.py
   ```

### Environment Configuration

Create a `.env` file in the root directory:

```env
# Database Configuration
SNOWFLAKE_ACCOUNT=your-account.snowflakecomputing.com
SNOWFLAKE_USER=username
SNOWFLAKE_PASSWORD=password
SNOWFLAKE_WAREHOUSE=warehouse
SNOWFLAKE_DATABASE=database

# LLM Provider (choose one)
LLM_PROVIDER=openai  # or anthropic
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o

# Optional: Anthropic Configuration
ANTHROPIC_API_KEY=your-anthropic-key
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Memory System
MEMORY_BACKEND=sqlite
SESSION_DB_PATH=data/session_memory.db
KNOWLEDGE_DB_PATH=data/knowledge_memory.db

# Vector Store
VECTOR_STORE_PROVIDER=faiss
VECTOR_STORE_PATH=data/vector_store
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional: Redis for Caching
REDIS_URL=redis://localhost:6379
```

---

## 📱 Usage

### Option 1: Advanced Professional Dashboard (Recommended)

```bash
streamlit run ui/advanced_dashboard.py
```

**Features:**
- Real-time agent processing pipeline with animated status indicators
- Professional metrics dashboard with gradient styling
- AI-powered visualization recommendations with interactive controls
- Streaming query processing with progress tracking
- Advanced result analysis with multi-tab display
- Export capabilities and code generation

### Option 2: Enhanced Original Interface

```bash
streamlit run ui/streamlit_app.py
```

**Features:**
- Enhanced visualization integration
- AI chart recommendations display
- Query history and session management
- Simplified interface for basic usage

### Option 3: REST API

```bash
python api/fastapi_app.py
```

**Endpoints:**
- `POST /query`: Process natural language queries
- `GET /health`: System health check
- `GET /metrics`: Performance metrics
- `POST /feedback`: User feedback collection

### Option 4: Direct System Access

```bash
python main.py
```

**Example:**
```python
from main import SQLAgentSystem

# Initialize system
system = SQLAgentSystem()

# Process query
result = await system.process_query(
    "Show me sales trends by region for the last quarter",
    user_id="user123"
)

print(result)
```

---

## 💬 Example Queries

### Sales & Revenue Analysis
```
"Show me total sales by product category for this quarter"
"What are the top 10 customers by revenue?"
"Compare this year's performance to last year"
"Which regions are underperforming in Q3?"
```

### Trend & Time Series
```
"Show me monthly revenue trends for the past year"
"What's the sales pattern by day of week?"
"Display quarterly growth rates over time"
"Show seasonal variations in customer activity"
```

### Comparative Analysis
```
"Compare product performance across different regions"
"Show market share by competitor for each segment"
"Analyze conversion rates by marketing channel"
"Compare customer acquisition costs by source"
```

### Advanced Analytics
```
"Show correlation between marketing spend and sales"
"Identify top-performing sales representatives"
"Display customer churn analysis by segment"
"Show inventory turnover rates by product line"
```

---

## 📊 Visualization Capabilities

### AI-Powered Chart Recommendations

The system automatically analyzes your query results and recommends optimal visualizations:

#### **Automatic Chart Type Selection**
- **Bar Charts**: For categorical comparisons and aggregated data
- **Line Charts**: For time series and trend analysis
- **Pie Charts**: For composition and proportion analysis
- **Scatter Plots**: For correlation and relationship analysis
- **Heatmaps**: For pattern identification in multi-dimensional data

#### **Intelligent Axis Selection**
- **X-Axis**: Automatically identifies categorical or temporal dimensions
- **Y-Axis**: Selects appropriate metrics and measures
- **Color Coding**: Suggests secondary dimensions for enhanced insights

#### **Code Generation**
- **Plotly Code**: Production-ready interactive visualizations
- **Streamlit Code**: Dashboard-ready components
- **Customization**: Easily modifiable templates for specific needs

### Example Visualization Output

```python
# Auto-generated Plotly code
import plotly.express as px

fig = px.bar(df, x='category', y='sales', 
             color='region',
             title='Sales by Category and Region')
fig.show()

# Auto-generated Streamlit code
import streamlit as st
import plotly.express as px

fig = px.bar(df, x='category', y='sales', 
             color='region',
             title='Sales by Category and Region')
st.plotly_chart(fig, use_container_width=True)
```

---

## 🧠 Memory & Learning System

### Three-Tier Memory Architecture

#### **Working Memory** (Real-time)
- Current session context and agent state
- Processing results and intermediate data
- Dynamic routing decisions and error handling

#### **Session Memory** (SQLite)
- User conversation history and preferences
- Query patterns within current session
- Personalization data and settings

#### **Long-term Memory** (SQLite + FAISS)
- Successful query patterns and templates
- Schema insights and table relationships
- User behavior patterns for recommendation improvement

### Learning Capabilities

- **Query Pattern Recognition**: Learns from successful queries to improve future recommendations
- **Schema Understanding**: Builds knowledge of table relationships and data patterns
- **User Personalization**: Adapts to individual user preferences and query styles
- **Performance Optimization**: Learns which queries perform well and suggests optimizations

---

## 🛡️ Security & Validation

### Multi-Layer Security

#### **SQL Injection Prevention**
- Pattern-based detection of malicious SQL constructs
- Parameterized query enforcement
- Dangerous operation blocking (DROP, DELETE, TRUNCATE, etc.)

#### **Query Validation**
- Syntax validation using `sqlparse`
- Business logic alignment checking
- Performance impact assessment

#### **Access Control**
- Database permission validation
- Query result size limitations
- Rate limiting and abuse prevention

### Security Features

```python
# Example security validation
security_patterns = [
    r";\s*(DROP|DELETE|TRUNCATE|ALTER)\s+",
    r"UNION\s+SELECT.*--",
    r"1\s*=\s*1",
    r"OR\s+1\s*=\s*1",
    r"--\s*$",
    r"/\*.*\*/"
]

# Performance analysis
if "FULL TABLE SCAN" in execution_plan:
    warnings.append("Query requires full table scan")
```

---

## ⚙️ Configuration

### System Settings

The system uses Pydantic for configuration management with environment variable validation:

```python
# config/settings.py
class Settings(BaseSettings):
    # Database Configuration
    snowflake_account: str
    snowflake_user: str
    snowflake_password: str
    snowflake_warehouse: str
    snowflake_database: str
    
    # LLM Configuration
    llm_provider: str = "openai"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    
    # Memory Configuration
    memory_backend: str = "sqlite"
    session_db_path: str = "data/session_memory.db"
    knowledge_db_path: str = "data/knowledge_memory.db"
    
    # Vector Store Configuration
    vector_store_provider: str = "faiss"
    vector_store_path: str = "data/vector_store"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
```

### Performance Tuning

#### **Database Optimization**
- Connection pooling for improved performance
- Query caching with configurable TTL
- Optimized SQLite configuration with WAL mode

#### **Memory Management**
- Configurable memory limits for large result sets
- Efficient vector storage with FAISS indexing
- Session cleanup and garbage collection

#### **LLM Optimization**
- Model selection based on query complexity
- Context window optimization for large schemas
- Caching of common entity extractions

---

## 🧪 Testing

### Test Suite

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/          # End-to-end tests

# Run with coverage
pytest --cov=. --cov-report=html
```

### Test Structure

```
tests/
├── unit/
│   ├── test_agents.py              # Agent functionality
│   ├── test_memory.py              # Memory system
│   ├── test_validation.py          # Security validation
│   └── test_visualization.py       # Chart recommendations
├── integration/
│   ├── test_workflow.py            # LangGraph workflow
│   ├── test_database.py            # Database integration
│   └── test_system.py              # Full system integration
└── fixtures/
    ├── sample_data.json            # Test data
    ├── mock_responses.json         # Mock LLM responses
    └── test_schemas.sql            # Test database schemas
```

### Example Tests

```python
# tests/unit/test_agents.py
import pytest
from agents.nlu_agent import NLUAgent

@pytest.mark.asyncio
async def test_nlu_agent_intent_extraction():
    """Test NLU agent extracts correct intent from query."""
    agent = NLUAgent(mock_memory_system, mock_llm)
    
    result = await agent.process_query(
        "Show me sales by region for last quarter",
        context={}
    )
    
    assert result["query_intent"]["primary_action"] == "aggregate"
    assert "sales" in result["entities_extracted"][0]["value"]
    assert result["confidence_scores"]["overall"] > 0.7
```

---

## 📈 Performance Monitoring

### Built-in Metrics

- **Query Processing Time**: End-to-end latency tracking
- **Agent Performance**: Individual agent execution times
- **Memory Usage**: System resource monitoring
- **Success Rates**: Query completion and accuracy metrics
- **User Satisfaction**: Feedback and rating collection

### Monitoring Dashboard

Access real-time metrics through the advanced dashboard:

```
http://localhost:8501/metrics
```

**Available Metrics:**
- Processing pipeline performance
- Database connection health
- Memory system statistics
- Visualization generation success rates
- User interaction patterns

---

## 🛠️ Development

### Adding New Agents

1. **Create Agent Class**
   ```python
   # agents/custom_agent.py
   class CustomAgent:
       def __init__(self, memory_system, additional_deps):
           self.memory_system = memory_system
           self.agent_name = "custom_agent"
       
       async def process(self, input_data: Dict) -> Dict:
           # Implementation
           pass
   ```

2. **Register in Workflow**
   ```python
   # workflows/sql_workflow.py
   def _initialize_agents(self):
       return {
           # ... existing agents
           "custom": CustomAgent(self.memory_system, deps)
       }
   ```

3. **Add Workflow Node**
   ```python
   async def _custom_processing_node(self, state: SQLAgentState):
       result = await self.agents["custom"].process(state)
       return {**state, **result}
   ```

### Extending Visualization Types

1. **Add Chart Type to VisualizationAgent**
   ```python
   # agents/visualization_agent.py
   def _recommend_custom_chart(self, df, insights):
       return ChartRecommendation(
           chart_type="custom_chart",
           priority=0.8,
           x_axis="column_x",
           y_axis="column_y",
           rationale="Custom chart for specific use case"
       )
   ```

2. **Generate Code Templates**
   ```python
   def _generate_custom_chart_code(self, recommendation):
       return """
       import plotly.graph_objects as go
       
       fig = go.Figure(data=go.CustomChart(...))
       fig.show()
       """
   ```

### Database Integration

The system currently supports Snowflake but can be extended to other databases:

1. **Create Database Connector**
   ```python
   # database/custom_connector.py
   class CustomDatabaseConnector:
       async def execute_query(self, sql: str):
           # Implementation
           pass
       
       async def get_schema_metadata(self):
           # Implementation
           pass
   ```

2. **Update Configuration**
   ```python
   # config/settings.py
   database_type: str = "custom"
   custom_connection_string: str = "..."
   ```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. **Fork and clone the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```
4. **Run tests**
   ```bash
   pytest
   ```
5. **Submit a pull request**

### Code Standards

- **Type Hints**: Use comprehensive type annotations
- **Docstrings**: Follow Google/NumPy docstring conventions
- **Testing**: Maintain >90% test coverage
- **Linting**: Use `black`, `flake8`, and `mypy`
- **Security**: Follow OWASP guidelines for data handling

---

## 📚 API Reference

### Core Classes

#### **SQLAgentSystem**
Main orchestrator class for the entire system.

```python
class SQLAgentSystem:
    async def process_query(self, query: str, user_id: str) -> Dict:
        """Process natural language query and return results."""
        pass
    
    async def get_query_history(self, user_id: str) -> List[Dict]:
        """Retrieve user's query history."""
        pass
    
    async def provide_feedback(self, query_id: str, feedback: Dict):
        """Collect user feedback for system improvement."""
        pass
```

#### **VisualizationAgent**
Handles chart recommendations and code generation.

```python
class VisualizationAgent:
    async def analyze_and_recommend(
        self, 
        query_results: List[Dict], 
        query_intent: Dict,
        schema_context: Dict, 
        entities: List[Dict]
    ) -> VisualizationResult:
        """Generate visualization recommendations."""
        pass
```

#### **SchemaIntelligenceAgent**
Manages database schema analysis and relationship detection.

```python
class SchemaIntelligenceAgent:
    async def analyze_schema_requirements(
        self, 
        entities: List[Dict], 
        intent: Dict, 
        context: Dict
    ) -> SchemaAnalysisResult:
        """Analyze database schema for query optimization."""
        pass
```

---

## ❓ Troubleshooting

### Common Issues

#### **Database Connection Errors**
```
Error: Could not connect to Snowflake database
```
**Solution:** Verify your `.env` file contains correct database credentials and the database is accessible.

#### **LLM API Errors**
```
Error: OpenAI API rate limit exceeded
```
**Solution:** Check your API key validity and rate limits. Consider switching to Anthropic or implementing request throttling.

#### **Memory System Issues**
```
Error: SQLite database locked
```
**Solution:** Ensure no other instances are running. Delete lock files in the `data/` directory if necessary.

#### **Visualization Generation Failures**
```
Error: No suitable columns found for visualization
```
**Solution:** Check that your query returns data with appropriate column types (numeric/categorical).

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL=DEBUG
```

### Performance Issues

1. **Slow Query Processing**
   - Check database connection latency
   - Optimize SQL queries with appropriate indexes
   - Reduce LLM context window size

2. **Memory Usage**
   - Clear session memory periodically
   - Implement result pagination for large datasets
   - Optimize vector store indexing

3. **UI Responsiveness**
   - Enable caching for common queries
   - Implement progressive loading
   - Use Streamlit's session state efficiently

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain & LangGraph**: For the powerful agent orchestration framework
- **Streamlit**: For the excellent dashboard framework
- **Plotly**: For interactive visualization capabilities
- **Snowflake**: For the robust data warehouse platform
- **OpenAI & Anthropic**: For advanced language model capabilities

---

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/advanced_sql_agent_system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/advanced_sql_agent_system/discussions)
- **Email**: support@your-org.com

---

<div align="center">

**Built with ❤️ for the Data Science Community**

[⭐ Star us on GitHub](https://github.com/your-org/advanced_sql_agent_system) | [🐛 Report Bug](https://github.com/your-org/advanced_sql_agent_system/issues) | [💡 Request Feature](https://github.com/your-org/advanced_sql_agent_system/issues)

</div>