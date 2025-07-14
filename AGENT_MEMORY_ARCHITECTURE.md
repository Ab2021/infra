# SQL Agent System: Agent & Memory Architecture Guide

## Table of Contents
1. [Introduction to Agentic Systems](#introduction-to-agentic-systems)
2. [System Overview](#system-overview)
3. [Agent Architecture](#agent-architecture)
4. [Memory System Architecture](#memory-system-architecture)
5. [Data Flow & Workflows](#data-flow--workflows)
6. [Implementation Details](#implementation-details)
7. [Beginner's Guide to Concepts](#beginners-guide-to-concepts)
8. [Advanced Patterns](#advanced-patterns)

---

## Introduction to Agentic Systems

### What is an Agentic System?

An **agentic system** is a software architecture where autonomous agents work together to solve complex problems. Each agent has:

- **Specialized capabilities** (like a team member with specific skills)
- **Autonomy** to make decisions within their domain
- **Communication** abilities to coordinate with other agents
- **Memory** to learn from past experiences
- **Goals** they work towards achieving

Think of it like a software development team where:
- One person handles UI design (UI Agent)
- Another handles backend logic (Logic Agent) 
- A third manages databases (Data Agent)
- They all communicate and coordinate to build software

### Why Use Agents for SQL Generation?

Traditional SQL generation approaches often struggle with:
- **Context understanding**: What does the user really want?
- **Data awareness**: What tables/columns are available and relevant?
- **Quality assurance**: Is the generated SQL safe and performant?
- **Visualization**: How should results be presented?

Our agent system breaks this complex problem into **specialized, manageable tasks**:

```
User Query: "Show me sales trends by region for 2024"
    ↓
Agent 1: "This wants trend analysis using sales and region data from 2024"
    ↓
Agent 2: "Sales data is in fact_sales table, regions in dim_geography, filter by year"
    ↓
Agent 3: "Generate SQL with time series, create line chart visualization"
```

---

## System Overview

### Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                    SQL Agent System                             │
├─────────────────────────────────────────────────────────────────┤
│  User Query: "Show sales trends by region this year"            │
└─────────────────┬───────────────────────────────────────────────┘
                  │
    ┌─────────────▼─────────────┐
    │   SQLAgentSystem          │ ◄── Main Orchestrator
    │   (main.py:37)            │
    └─────────────┬─────────────┘
                  │
         ┌────────▼────────┐
         │ Memory System   │ ◄── Shared Intelligence
         │ (memory_system) │
         └────────┬────────┘
                  │
    ┌─────────────▼─────────────────────────────────────────┐
    │                Agent Pipeline                        │
    └─────────────┬───────────┬───────────┬─────────────────┘
                  │           │           │
         ┌────────▼──────┐   │  ┌────────▼──────┐   ┌────────▼──────┐
         │   Agent 1     │   │  │   Agent 2     │   │   Agent 3     │
         │Query Understanding│ │Data Profiling │   │SQL Visualization│
         │               │   │  │               │   │               │
         └───────────────┘   │  └───────────────┘   └───────────────┘
                             │
                    ┌────────▼──────┐
                    │   Database    │
                    │  (Snowflake)  │
                    └───────────────┘
```

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Main Orchestrator** | `main.py:37` | Coordinates all agents and manages system lifecycle |
| **Memory System** | `memory/memory_system.py:16` | Provides intelligence and learning capabilities |
| **Agent 1** | `agents/query_understanding_agent.py:24` | Understands user intent and identifies required data |
| **Agent 2** | `agents/data_profiling_agent.py:30` | Analyzes actual data to determine optimal filters |
| **Agent 3** | `agents/sql_visualization_agent.py:34` | Generates SQL and determines visualization |
| **Snowflake Agent** | `snowflake_agent.py:65` | Production Snowflake integration with your procedures |

---

## Agent Architecture

### Agent Design Principles

Each agent follows the **Single Responsibility Principle**:

```python
# Agent Interface Pattern
class BaseAgent:
    def __init__(self, llm_provider, memory_system, **dependencies):
        self.llm_provider = llm_provider      # AI capabilities
        self.memory_system = memory_system    # Shared memory
        # Agent-specific dependencies
    
    async def process(self, input_data, context) -> Dict:
        # 1. Analyze input
        # 2. Retrieve relevant memories
        # 3. Apply agent logic
        # 4. Store learnings
        # 5. Return structured output
        pass
```

### Agent 1: Query Understanding Agent

**File**: `agents/query_understanding_agent.py`  
**Purpose**: Transform natural language into structured intent

```
Input: "Show me sales trends by region for 2024"
    ↓
┌─────────────────────────────────────────────────────────────────┐
│                Query Understanding Agent                        │
├─────────────────────────────────────────────────────────────────┤
│ 1. Extract Basic Intent (Line 197)                             │
│    • Keywords: "trends", "by region", "2024"                   │
│    • Action: trend_analysis                                     │
│    • Metrics: [sales]                                          │
│    • Dimensions: [region]                                      │
│    • Time: 2024                                                │
│                                                                 │
│ 2. Enhance with LLM (Line 236)                                 │
│    • Use schema context                                         │
│    • Previous conversation history                              │
│    • Generate structured intent                                 │
│                                                                 │
│ 3. Map to Schema (Line 377)                                    │
│    • Find relevant tables: fact_sales, dim_geography           │
│    • Find columns: sales_amount, region_name, date             │
│    • Calculate confidence scores                                │
│                                                                 │
│ 4. Validate & Store (Line 483)                                 │
│    • Ensure tables/columns exist                               │
│    • Store successful patterns                                  │
└─────────────────────────────────────────────────────────────────┘
    ↓
Output: {
  "intent": QueryIntent(action="trend_analysis", metrics=["sales"], 
                       dimensions=["region"], time_scope="2024"),
  "identified_tables": [{"name": "fact_sales", "confidence": 0.9}],
  "required_columns": [{"name": "sales_amount", "table": "fact_sales"}]
}
```

**Key Intelligence**:
- **Keyword Mapping** (Line 56): Maps common words to SQL concepts
- **LLM Enhancement** (Line 236): Uses AI for complex understanding
- **Schema Awareness** (Line 377): Maps intent to actual database structure
- **Column Name Resolution** (Line 100): Handles spaces/underscores correctly
- **Pattern Learning** (Line 631): Stores successful mappings for future use

### Agent 2: Data Profiling Agent

**File**: `agents/data_profiling_agent.py`  
**Purpose**: Understand actual data characteristics and suggest optimal filters

```
Input: Tables[fact_sales, dim_geography], Columns[sales_amount, region_name]
    ↓
┌─────────────────────────────────────────────────────────────────┐
│                Data Profiling Agent                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. Profile Columns (Line 97)                                   │
│    • Execute: SELECT COUNT(DISTINCT region_name),              │
│               COUNT(*), MIN(sales_amount)... FROM fact_sales   │
│    • Analyze: Data types, null counts, value distributions     │
│    • Score: Data quality metrics                               │
│                                                                 │
│ 2. Analyze Relationships (Line 352)                            │
│    • Find potential joins between tables                       │
│    • Identify common columns                                    │
│    • Assess data consistency                                    │
│                                                                 │
│ 3. Generate Smart Filters (Line 449)                           │
│    • Time filters: "WHERE YEAR(date) = 2024"                  │
│    • Quality filters: "WHERE sales_amount IS NOT NULL"         │
│    • Categorical filters: "WHERE region IN (top_regions)"      │
│                                                                 │
│ 4. Store Insights (Line 643)                                   │
│    • Cache column profiles for performance                      │
│    • Store successful filter patterns                          │
└─────────────────────────────────────────────────────────────────┘
    ↓
Output: {
  "column_profiles": {
    "fact_sales.sales_amount": {type: "float", quality: 0.95},
    "dim_geography.region_name": {type: "categorical", values: [...]}
  },
  "suggested_filters": [
    {type: "time_filter", condition: "YEAR(date) = 2024", confidence: 0.9}
  ]
}
```

### Agent 3: SQL Visualization Agent

**File**: `agents/sql_visualization_agent.py`  
**Purpose**: Generate secure SQL and determine optimal visualization

```
Input: Intent, Column Profiles, Filter Suggestions
    ↓
┌─────────────────────────────────────────────────────────────────┐
│                SQL Visualization Agent                         │
├─────────────────────────────────────────────────────────────────┤
│ 1. Generate SQL (Line 133)                                     │
│    • Check memory for similar patterns                         │
│    • Use templates: trend_analysis, comparison, etc.           │
│    • Fill template with actual table/column names              │
│                                                                 │
│ 2. Apply Guardrails (Line 437)                                 │
│    • Security: Block DELETE, DROP, injection patterns          │
│    • Performance: Check row limits, complexity scores          │
│    • Quality: Validate syntax and logic                        │
│                                                                 │
│ 3. Determine Visualization (Line 525)                          │
│    • Analyze SQL structure (aggregation, grouping, time)       │
│    • Map to chart types: line_chart, bar_chart, table          │
│    • Generate chart configuration                               │
│                                                                 │
│ 4. Generate Code (Line 701)                                    │
│    • Create Streamlit visualization code                       │
│    • Include error handling and download options               │
│    • Store successful patterns                                 │
└─────────────────────────────────────────────────────────────────┘
    ↓
Output: {
  "sql_query": "SELECT region_name, SUM(sales_amount) FROM fact_sales fs 
                JOIN dim_geography dg ON fs.geography_id = dg.id 
                WHERE YEAR(date) = 2024 GROUP BY region_name",
  "chart_config": {type: "line_chart", x_axis: "date", y_axis: "sales"},
  "plotting_code": "st.line_chart(df.set_index('date')['sales'])"
}
```

### Snowflake Production Agent

**File**: `snowflake_agent.py`  
**Purpose**: Production-ready Snowflake integration matching your input procedures

```
Input: "List auto policies with premium over 10000 in Texas in 2023"
    ↓
┌─────────────────────────────────────────────────────────────────┐
│              Snowflake Production Agent                        │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load Schema from Excel (Line 63)                            │
│    • Read hackathon_final_schema_file_v1.xlsx                  │
│    • Parse Table_descriptions and Column Summaries             │
│    • Create column mappings for space/underscore handling      │
│                                                                 │
│ 2. Schema Analysis (Line 169)                                  │
│    • Use GPT with exact column names from Excel                │
│    • Map user query to relevant tables and columns             │
│    • Validate column name variations                           │
│                                                                 │
│ 3. SQL Generation (Line 232)                                   │
│    • Build detailed context with sample values                 │
│    • Use few-shot examples if available                        │
│    • Generate SQL with proper column quoting                   │
│                                                                 │
│ 4. Snowflake Execution (Line 366)                              │
│    • Execute on actual Snowflake using your connection         │
│    • Return pandas DataFrame with results                      │
│    • Handle errors gracefully                                  │
└─────────────────────────────────────────────────────────────────┘
    ↓
Output: QueryResult(
  success=True, sql_query="SELECT ...", 
  execution_result=DataFrame([...]), processing_time=2.1
)
```

---

## Memory System Architecture

### Memory Design Philosophy

The memory system acts as the **collective intelligence** of all agents:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory System Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Session Memory │    │ Knowledge Memory │                    │
│  │  (Conversations)│    │ (Learning Store) │                    │
│  └─────────────────┘    └─────────────────┘                    │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               SQLite In-Memory Databases                   │ │
│  │  • Ultra-fast: 50-100x faster than disk                   │ │
│  │  • ACID compliant: Reliable transactions                   │ │
│  │  • Optional persistence: Best of both worlds               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Memory Operations                        │ │
│  │  • Store successful patterns                               │ │
│  │  • Find similar queries (text similarity)                  │ │
│  │  • Cache column profiles                                    │ │
│  │  • Track conversation context                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Components

#### 1. Session Memory (Conversation Context)

**Tables**: `sessions`, `conversation_history`  
**Purpose**: Track user interactions and maintain context

```sql
-- Session tracking (memory_system.py:67)
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_activity TEXT NOT NULL,
    context_data TEXT
);

-- Conversation history (memory_system.py:75)
CREATE TABLE conversation_history (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,           -- Original user query
    intent TEXT,                   -- Structured intent (JSON)
    tables_used TEXT,              -- Tables accessed (JSON array)
    chart_type TEXT,               -- Visualization type
    success INTEGER DEFAULT 1,     -- Whether query succeeded
    timestamp TEXT NOT NULL
);
```

#### 2. Knowledge Memory (Learning Store)

**Tables**: `successful_queries`, `query_patterns`, `schema_insights`  
**Purpose**: Store learning and enable pattern reuse

```sql
-- Successful query patterns (memory_system.py:89)
CREATE TABLE successful_queries (
    id TEXT PRIMARY KEY,
    query_pattern TEXT NOT NULL,    -- Extracted pattern for similarity
    sql_query TEXT NOT NULL,        -- Generated SQL
    tables_used TEXT,               -- Tables involved
    chart_type TEXT,                -- Visualization used
    execution_time REAL DEFAULT 0.0, -- Performance metric
    result_count INTEGER DEFAULT 0,  -- Result size
    success_score REAL DEFAULT 1.0,  -- Quality score
    usage_count INTEGER DEFAULT 1,   -- Reuse frequency
    created_at TEXT NOT NULL,
    last_used TEXT NOT NULL,
    metadata TEXT                    -- Additional context (JSON)
);
```

---

## Data Flow & Workflows

### Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Request Workflow                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ 1. User Input                      │
    │ "Show sales trends by region 2024" │
    └─────────────────┬──────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ 2. Session Setup (main.py:224)     │
    │ • Create/retrieve session          │
    │ • Get conversation history         │
    │ • Initialize context               │
    └─────────────────┬──────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ 3. Agent 1: Query Understanding    │
    │ • Extract intent: trend_analysis   │
    │ • Identify metrics: [sales]        │
    │ • Identify dimensions: [region]    │
    │ • Map to schema: fact_sales table  │
    │ • Confidence: 0.9                  │
    └─────────────────┬──────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ 4. Agent 2: Data Profiling        │
    │ • Query: SELECT COUNT DISTINCT...  │
    │ • Profile sales_amount column      │
    │ • Profile region_name column       │
    │ • Suggest time filter: YEAR = 2024 │
    │ • Quality score: 0.95              │
    └─────────────────┬──────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ 5. Agent 3: SQL & Visualization   │
    │ • Generate SQL from template       │
    │ • Apply guardrails (security)      │
    │ • Choose chart: line_chart         │
    │ • Generate Streamlit code          │
    │ • Validation: PASSED               │
    └─────────────────┬──────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ 6. Memory Storage                  │
    │ • Store conversation history       │
    │ • Cache successful patterns        │
    │ • Update usage statistics          │
    │ • Store schema insights            │
    └─────────────────┬──────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ 7. Response Assembly               │
    │ • Combine all agent outputs        │
    │ • Include metadata and timing      │
    │ • Format for user interface        │
    └─────────────────┬──────────────────┘
                      │
    ┌─────────────────▼──────────────────┐
    │ 8. Result Delivery                 │
    │ • SQL: SELECT region, SUM(sales).. │
    │ • Chart: Line chart config         │
    │ • Code: st.line_chart(df)          │
    │ • Time: 2.3 seconds                │
    └────────────────────────────────────┘
```

---

## Implementation Details

### Key Files and Their Roles

| File | Lines of Code | Key Responsibilities |
|------|---------------|---------------------|
| `main.py` | 540 | System orchestration, agent coordination, lifecycle management |
| `memory/memory_system.py` | 470 | Memory operations, pattern storage, similarity search |
| `agents/query_understanding_agent.py` | 647 | NLU, intent extraction, schema mapping, column resolution |
| `agents/data_profiling_agent.py` | 674 | Data analysis, column profiling, filter suggestions |
| `agents/sql_visualization_agent.py` | 837 | SQL generation, guardrails, visualization |
| `snowflake_agent.py` | 650 | Production Snowflake integration with your procedures |
| `test_main.py` | 800 | Comprehensive testing suite for all components |

### Agent Coordination (main.py)

```python
class SQLAgentSystem:
    async def process_query(self, user_query: str, user_id: str = "anonymous", 
                          session_id: str = None) -> Dict[str, Any]:
        """3-agent pipeline coordination"""
        
        # Step 1: Query Understanding Agent (Line 230)
        understanding_result = await self.query_agent.process(
            user_query=user_query,
            session_context=session_context
        )
        
        # Step 2: Data Profiling Agent (Line 244)  
        profiling_result = await self.profiling_agent.process(
            tables=understanding_result.get("identified_tables", []),
            columns=understanding_result.get("required_columns", []),
            intent=understanding_result.get("query_intent", {})
        )
        
        # Step 3: SQL Visualization Agent (Line 258)
        sql_viz_result = await self.sql_viz_agent.process(
            query_intent=understanding_result.get("query_intent", {}),
            column_profiles=profiling_result.get("column_profiles", {}),
            suggested_filters=profiling_result.get("suggested_filters", [])
        )
        
        # Combine and return results (Line 276)
        return self._combine_results(understanding_result, profiling_result, sql_viz_result)
```

### Column Name Handling System

The system now properly handles your Excel schema with spaces in column names:

```python
# Column name resolution (query_understanding_agent.py:100)
def resolve_column_name(self, column_name: str, table_name: str = None) -> str:
    """Resolve column name variations to correct database names"""
    # Handles: "Feature Name" ↔ "Feature_Name" ↔ "feature_name"
    
def quote_column_name(self, column_name: str) -> str:
    """Quote column names that contain spaces for SQL"""
    if ' ' in column_name:
        return f'"{column_name}"'  # "Feature Name"
    return column_name            # regular_column
```

---

## Beginner's Guide to Concepts

### Core Concepts Explained

#### 1. What is an "Agent"?

Think of an agent as a **specialized worker** in a team:

```python
# Agent Concept
class Agent:
    def __init__(self, specialty, tools, memory):
        self.specialty = specialty  # What I'm good at
        self.tools = tools         # What I can use (LLM, database, etc.)
        self.memory = memory       # What I remember
    
    async def process(self, task):
        # 1. Understand the task
        # 2. Use my specialty and tools
        # 3. Learn from the result
        # 4. Return my contribution
        pass
```

#### 2. Why Memory Matters

```
❌ Without Memory:
User: "Show sales by region"
System: Starts from scratch every time, slow and inefficient

✅ With Memory:
User: "Show sales by region"  
System: "I remember doing this before, let me reuse what worked"
```

#### 3. Column Name Challenges

Your Excel schema has columns like "Feature Name" and "Policy Number" (with spaces), but sometimes systems expect "Feature_Name" (with underscores). Our system handles this automatically:

```python
# Automatic resolution:
"Feature_Name" → "Feature Name" → "Feature Name" (SQL: "Feature Name")
"feature name" → "Feature Name" → "Feature Name" (SQL: "Feature Name")  
"Policy Number" → "Policy Number" → "Policy Number" (SQL: "Policy Number")
```

---

## Testing Your System

### Running Tests

```bash
# Full integration test
python test_main.py

# Interactive testing mode
python test_main.py interactive

# Test specific components
python test_main.py schema    # Test schema loading
python test_main.py init      # Test initialization

# Production Snowflake agent
python snowflake_agent.py
```

### Test Coverage

The test suite covers:
- ✅ **Schema Loading**: Excel file parsing and validation
- ✅ **System Initialization**: All component setup
- ✅ **Agent Pipeline**: Complete query processing
- ✅ **Memory System**: Storage and retrieval
- ✅ **Column Handling**: Space/underscore normalization
- ✅ **SQL Execution**: Snowflake integration
- ✅ **Error Handling**: Graceful failure management

### Example Test Output

```
🧪 SQL AGENT SYSTEM - COMPREHENSIVE TEST SUITE
============================================================
📊 System Information:
   schema_file: hackathon_final_schema_file_v1.xlsx
   total_tables: 15
   total_columns: 250
   column_mappings: 500
   gpt_available: True

✅ Schema Loading PASSED
✅ System Initialization PASSED  
✅ Query Pipeline Tests: 4/5 PASSED (80%)
✅ Memory System PASSED
✅ Column Handling PASSED

🎉 INTEGRATION TESTS PASSED! System is ready for production.
```

---

This comprehensive documentation now reflects your clean, production-ready SQL Agent System without any prefixes or suffixes, properly integrated with your Snowflake infrastructure and column naming conventions.