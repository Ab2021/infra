# SQL Agent System: Complete Workflow Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Entry Points and Initialization](#entry-points-and-initialization)
3. [Complete Workflow: Mock Query Execution](#complete-workflow-mock-query-execution)
4. [Detailed Step-by-Step Execution](#detailed-step-by-step-execution)
5. [Code References and Function Calls](#code-references-and-function-calls)
6. [Testing Individual Components](#testing-individual-components)
7. [Error Handling and Recovery](#error-handling-and-recovery)
8. [Performance Monitoring](#performance-monitoring)

---

## System Overview

The SQL Agent System processes natural language queries through a sophisticated multi-agent pipeline that converts user requests into optimized SQL queries and visualizations.

### Architecture Flow
```
User Input → Schema Loading → Agent Pipeline → SQL Generation → Execution → Visualization
     ↓            ↓              ↓               ↓              ↓           ↓
Entry Points → Initialization → 3-Agent Flow → SQL Creation → Database → Dashboard
```

### Key Components
- **Entry Points**: `main.py`, `snowflake_agent.py`, `ui/streamlit_app.py`
- **Core Agents**: Query Understanding, Data Profiling, SQL Visualization
- **Memory System**: Session and knowledge management
- **Database Integration**: Snowflake connector with column name handling

---

## Entry Points and Initialization

### 1. Production Snowflake Agent (`snowflake_agent.py`)
**Primary entry point for production use**

```python
# File: snowflake_agent.py:797-798
if __name__ == "__main__":
    main()
```

**Initialization Flow:**
```python
# snowflake_agent.py:51-80
def __init__(self, schema_file_path: str = "hackathon_final_schema_file_v1.xlsx"):
    self.schema_file_path = schema_file_path
    self.logger = logging.getLogger(__name__)
    
    # Core initialization steps
    self._load_schema()           # Line 78
    self._create_column_mappings() # Line 79  
    self._initialize_gpt()        # Line 80
```

### 2. Full Agent System (`main.py`)
**Entry point for complete multi-agent system**

```python
# File: main.py:540
if __name__ == "__main__":
    asyncio.run(main())
```

**System Creation:**
```python
# main.py:37-60
class SQLAgentSystem:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Component initialization
        self.memory_system = None      # Line 45
        self.query_agent = None        # Line 46
        self.profiling_agent = None    # Line 47
        self.sql_viz_agent = None      # Line 48
```

### 3. Web Interface (`ui/streamlit_app.py`)
**Entry point for web-based interaction**

```python
# ui/streamlit_app.py (entry point)
import streamlit as st
# Web interface initialization and user interaction
```

---

## Complete Workflow: Mock Query Execution

Let's trace a complete execution using the mock query: **"Show premium trends by state for auto policies in 2023"**

### Workflow Overview
```
Input: "Show premium trends by state for auto policies in 2023"
  ↓
[1] Entry Point Selection (snowflake_agent.py)
  ↓  
[2] Schema Loading & Column Mapping
  ↓
[3] Agent 1: Query Understanding 
  ↓
[4] Agent 2: Data Profiling (if using main.py)
  ↓
[5] Agent 3: SQL Generation & Visualization
  ↓
[6] Snowflake Execution
  ↓
[7] Result Processing & Display
  ↓
Output: SQL Query + Execution Results + Chart Config
```

---

## Detailed Step-by-Step Execution

### Step 1: Entry Point and System Initialization

#### Option A: Production Snowflake Agent
```python
# File: snowflake_agent.py:740-750
def main():
    print(f"\n🚀 SNOWFLAKE SQL AGENT - PRODUCTION SYSTEM")
    
    # Initialize agent with schema file
    agent = SnowflakeAgent()  # Line 748
    
    # Display system information
    info = agent.get_system_info()  # Line 751
```

**Code Reference**: `snowflake_agent.py:747-748`
```python
agent = SnowflakeAgent()
```

**Function Called**: `SnowflakeAgent.__init__()`
- **Location**: `snowflake_agent.py:51-80`
- **Purpose**: Initialize agent with Excel schema loading

#### Option B: Full Agent System  
```python
# File: main.py:495-520
async def main():
    # Load settings
    settings = get_settings()  # Line 498
    
    # Create system instance  
    system = SQLAgentSystem(settings)  # Line 501
    
    # Initialize with schema
    await system.initialize(SCHEMA_FILE)  # Line 504
```

**Code Reference**: `main.py:501`
```python
system = SQLAgentSystem(settings)
```

**Function Called**: `SQLAgentSystem.__init__()`
- **Location**: `main.py:37-60`
- **Purpose**: Create multi-agent system with memory

### Step 2: Schema Loading and Column Mapping

```python
# File: snowflake_agent.py:82-126
def _load_schema(self):
    # Load table descriptions
    self.table_catalog = pd.read_excel(
        self.schema_file_path,
        sheet_name='Table_descriptions'  # Line 90
    )[['DATABASE', 'SCHEMA', 'TABLE', 'Brief_Description', 'Detailed_Comments']]
    
    # Load column descriptions  
    self.column_catalog = pd.read_excel(
        self.schema_file_path,
        sheet_name="Table's Column Summaries"  # Line 96
    )[['Table Name', 'Feature Name', 'Data Type', 'Description', 'sample_100_distinct']]
```

**Column Mapping Creation:**
```python
# File: snowflake_agent.py:128-145
def _create_column_mappings(self):
    for col in self.all_columns:
        original = str(col["column_name"]).strip()
        normalized = original.replace(' ', '_').replace('-', '_')
        
        # Create bidirectional mappings
        self.column_mapping[normalized.lower()] = original      # Line 138
        self.reverse_mapping[original.lower()] = normalized     # Line 139
```

**Testing**: 
```bash
python -c "
from snowflake_agent import SnowflakeAgent
# Test with mock Excel file
agent = SnowflakeAgent('test_schema.xlsx')
print(f'Loaded {len(agent.all_tables)} tables, {len(agent.all_columns)} columns')
"
```

### Step 3: Query Processing - Agent 1 (Query Understanding)

#### Production Path (Snowflake Agent)
```python
# File: snowflake_agent.py:590-607
def process_query(self, user_question: str) -> QueryResult:
    # Step 1: Schema Analysis
    relevant_tables, relevant_columns, relevant_joins = self.match_query_to_schema(user_question)
```

**Schema Analysis Function:**
```python
# File: snowflake_agent.py:261-395
def match_query_to_schema(self, user_question: str) -> tuple:
    uq = user_question.lower()
    
    # Use LLM if available
    if self.gpt_object is not None:
        # Build context for LLM
        gpt_prompt = f"""
        You are an expert in the actuarial insurance domain...
        
        User Question: {user_question}
        
        IMPORTANT COLUMN NAMING RULES:
        - Column names in the database may contain SPACES
        - You MUST use the EXACT column names as shown
        """  # Lines 288-329
```

#### Full System Path (Multi-Agent)
```python
# File: main.py:224-230
async def process_query(self, user_query: str, user_id: str = "anonymous", 
                      session_id: str = None) -> Dict[str, Any]:
    # Step 1: Query Understanding Agent
    understanding_result = await self.query_agent.process(
        user_query=user_query,
        session_context=session_context
    )
```

**Query Understanding Agent Process:**
```python
# File: agents/query_understanding_agent.py:147-187
async def process(self, user_query: str, session_context: Dict = None) -> Dict:
    # Step 1: Extract basic intent using keywords
    basic_intent = self._extract_basic_intent(user_query)  # Line 162
    
    # Step 2: Get enhanced understanding using LLM
    if self.llm_provider:
        enhanced_intent = await self._get_llm_understanding(user_query, basic_intent, session_context)  # Line 166
```

**Mock Query Processing:**
```
Input: "Show premium trends by state for auto policies in 2023"

Basic Intent Extraction (Line 197-234):
- primary_action: "trend_analysis" (detected from "trends")
- potential_metrics: ["premium"] (detected from "premium")  
- potential_dimensions: ["location"] (detected from "state")
- time_scope: "2023" (detected from "2023")
```

**Testing Agent 1:**
```python
# Test file: test_main.py:248-319
async def test_agent_pipeline(system, test_query: Dict):
    query_text = test_query["query"]
    result = await system.process_query(query_text)
    
    understanding = result.get('query_understanding', {})
    print(f"Intent: {understanding.get('query_intent')}")
```

### Step 4: Data Profiling (Full System Only)

```python
# File: main.py:244-250
# Step 2: Data Profiling Agent  
profiling_result = await self.profiling_agent.process(
    tables=understanding_result.get("identified_tables", []),
    columns=understanding_result.get("required_columns", []),
    intent=understanding_result.get("query_intent", {})
)
```

**Data Profiling Agent Process:**
```python
# File: agents/data_profiling_agent.py:97-200 (example structure)
async def process(self, tables: List[Dict], columns: List[Dict], intent: Dict) -> Dict:
    # Profile each column for data quality and characteristics
    column_profiles = await self._profile_columns(columns)
    
    # Generate smart filters based on data analysis
    suggested_filters = await self._generate_smart_filters(tables, columns, intent)
```

**Mock Data Profiling Result:**
```python
{
    "column_profiles": {
        "policies.premium_amount": {"type": "float", "quality": 0.95, "min": 500, "max": 50000},
        "policies.state": {"type": "categorical", "distinct_values": 50, "quality": 0.98},
        "policies.policy_type": {"type": "categorical", "values": ["auto", "home", "life"]}
    },
    "suggested_filters": [
        {"type": "time_filter", "condition": "YEAR(effective_date) = 2023", "confidence": 0.9},
        {"type": "category_filter", "condition": "policy_type = 'auto'", "confidence": 0.85}
    ]
}
```

### Step 5: SQL Generation and Visualization

#### Production Path (Snowflake Agent)
```python
# File: snowflake_agent.py:622-624
# Step 2: SQL Generation
reasoning, sql_query = self.generate_sql(user_question, relevant_columns, relevant_joins)
```

**SQL Generation Function:**
```python
# File: snowflake_agent.py:435-541
def generate_sql(self, user_question: str, relevant_columns: List[Dict], 
                relevant_joins: List[Dict]) -> tuple:
    
    # Build context block
    context_block = self.build_column_context_block(relevant_columns)  # Line 448
    
    # Enhanced prompt for GPT
    gpt_prompt = f"""
    **Relevant Columns and Tables with Sample Values:**
    ```{context_block}```
    
    **User Question:**
    ```{user_question}```
    
    **MANDATORY Instructions:**
    - Use EXACT column names as shown in the context above
    - If a column name contains spaces, wrap it in double quotes
    """  # Lines 459-511
```

#### Full System Path
```python
# File: main.py:258-264
# Step 3: SQL Visualization Agent
sql_viz_result = await self.sql_viz_agent.process(
    query_intent=understanding_result.get("query_intent", {}),
    column_profiles=profiling_result.get("column_profiles", {}),
    suggested_filters=profiling_result.get("suggested_filters", [])
)
```

**SQL Visualization Agent Process:**
```python
# File: agents/sql_visualization_agent.py:133-200 (example structure)
async def process(self, query_intent: Dict, column_profiles: Dict, suggested_filters: List) -> Dict:
    # Generate SQL from templates
    sql_query = await self._generate_sql_from_template(query_intent, column_profiles)
    
    # Apply security guardrails
    validated_sql = await self._apply_guardrails(sql_query)
    
    # Determine optimal visualization
    chart_config = await self._determine_visualization(query_intent, sql_query)
```

**Mock SQL Generation Result:**
```sql
-- Generated SQL for: "Show premium trends by state for auto policies in 2023"
SELECT 
    state_name,
    YEAR(effective_date) as year,
    SUM("Premium Amount") as total_premium,
    COUNT(*) as policy_count
FROM policies p
JOIN geography g ON p.state_id = g.id  
WHERE YEAR(effective_date) = 2023 
    AND policy_type = 'auto'
GROUP BY state_name, YEAR(effective_date)
ORDER BY total_premium DESC;
```

### Step 6: Column Name Resolution and SQL Validation

```python
# File: snowflake_agent.py:161-196
def resolve_column_name(self, column_name: str, table_name: str = None) -> str:
    # Try exact match first
    if table_name:
        for col in self.all_columns:
            if (col["table_name"].lower() == table_short.lower() and 
                col["column_name"].lower() == cleaned.lower()):
                return col["column_name"]  # Line 177
```

**Column Quoting for SQL:**
```python  
# File: snowflake_agent.py:198-209
def quote_column_name(self, column_name: str) -> str:
    if (' ' in column_name or 
        '-' in column_name or 
        any(char in column_name for char in ['(', ')', '.', ',', ';'])):
        return f'"{column_name}"'  # Line 207
    return column_name
```

**SQL Validation:**
```python
# File: snowflake_agent.py:211-244
def validate_and_fix_sql_column_names(self, sql_query: str, relevant_columns: List[Dict]) -> str:
    # Create mapping of variations to properly quoted names
    for col in relevant_columns:
        original_name = col["column_name"]
        quoted_name = self.quote_column_name(original_name)
        
        # Apply fixes using word boundaries
        pattern = r'\b' + re.escape(wrong_name) + r'\b'
        fixed_sql = re.sub(pattern, correct_name, fixed_sql, flags=re.IGNORECASE)
```

**Testing Column Resolution:**
```python
# Test in test_main.py:426-466
async def test_column_name_handling(system):
    test_cases = [
        ("Feature Name", "Feature Name"),
        ("Feature_Name", "Feature Name"), 
        ("Policy Number", "Policy Number"),
    ]
    
    for input_name, expected in test_cases:
        resolved = agent.resolve_column_name(input_name)
        quoted = agent.quote_column_name(resolved or input_name)
        print(f"'{input_name}' → '{resolved}' → '{quoted}'")
```

### Step 7: Snowflake Execution

```python
# File: snowflake_agent.py:639-641
# Step 3: SQL Execution
execution_result = self.execute_sql_query(sql_query)
```

**SQL Execution Function:**
```python
# File: snowflake_agent.py:547-584
def execute_sql_query(self, sql_query: str) -> pd.DataFrame:
    try:
        # Import Snowflake connection
        from src.snowflake_connection import create_snowflake_connection  # Line 563
        
        conn = create_snowflake_connection()  # Line 567
        with conn.cursor() as cursor:
            cursor.execute(sql_query)           # Line 569
            rows = cursor.fetchall()            # Line 570
            colnames = [d[0] for d in cursor.description]  # Line 571
            df = pd.DataFrame(rows, columns=colnames)      # Line 572
```

**Testing SQL Execution:**
```python
# Test in test_main.py:325-360
async def test_sql_execution(system):
    test_queries = [
        "SELECT 1 as test_column",
        "SELECT 'Hello' as message, 123 as number",
    ]
    
    for sql_query in test_queries:
        execution_result = await system.execute_sql_and_get_results(sql_query)
        print(f"Success: {execution_result.get('success', False)}")
        print(f"Rows: {execution_result.get('result_count', 0)}")
```

### Step 8: Result Processing and Visualization Configuration

```python
# File: snowflake_agent.py:660-672
return QueryResult(
    success=True,
    user_query=user_question,
    reasoning=reasoning,
    sql_query=sql_query,
    relevant_tables=relevant_tables,
    relevant_columns=relevant_columns,
    relevant_joins=relevant_joins,
    execution_result=execution_result,
    processing_time=processing_time
)
```

**Chart Configuration (Full System):**
```python
# agents/sql_visualization_agent.py (example structure)
def _determine_visualization(self, query_intent: Dict, sql_query: str) -> Dict:
    if query_intent.action == "trend_analysis":
        return {
            "chart_type": "line_chart",
            "x_axis": "year", 
            "y_axis": "total_premium",
            "color_by": "state_name",
            "title": "Premium Trends by State (2023)"
        }
```

### Step 9: Result Display

```python
# File: snowflake_agent.py:705-734
def display_result(self, result: QueryResult):
    print(f"🔍 QUERY PROCESSING RESULT")
    print(f"✅ Success: {result.success}")
    print(f"⏱️  Processing Time: {result.processing_time:.2f} seconds")
    
    if result.execution_result is not None:
        print(f"📊 EXECUTION RESULTS:")
        print(f"   Rows: {len(result.execution_result)}")
        print(result.execution_result.head().to_string(index=False))
```

**Mock Result Display:**
```
🔍 QUERY PROCESSING RESULT
====================================================
📝 Query: Show premium trends by state for auto policies in 2023
✅ Success: True
⏱️  Processing Time: 2.34 seconds

📊 ANALYSIS RESULTS:
   Tables: 2
   Columns: 4
   Joins: 1

🧠 REASONING:
   Identified trend analysis request for premium data by state with auto policy filter and 2023 time scope

📝 GENERATED SQL:
   SELECT state_name, SUM("Premium Amount") as total_premium...

📊 EXECUTION RESULTS:
   Rows: 50
   Columns: ['state_name', 'year', 'total_premium', 'policy_count']
   
   Sample Data:
   state_name    year  total_premium  policy_count
   California    2023     125000000         15234
   Texas         2023      98000000         12456
   Florida       2023      87000000         11234
```

---

## Code References and Function Calls

### Complete Function Call Chain

#### Production Snowflake Agent Path
```
snowflake_agent.py:main() [Line 740]
├── SnowflakeAgent.__init__() [Line 51]
│   ├── _load_schema() [Line 78]
│   ├── _create_column_mappings() [Line 79]
│   └── _initialize_gpt() [Line 80]
│
├── process_query() [Line 590]
│   ├── match_query_to_schema() [Line 261]
│   │   ├── LLM analysis [Line 275-369] OR
│   │   └── keyword_based_mapping() [Line 448-481]
│   │
│   ├── generate_sql() [Line 435]
│   │   ├── build_column_context_block() [Line 401]
│   │   └── validate_and_fix_sql_column_names() [Line 532-534]
│   │
│   └── execute_sql_query() [Line 547]
│       └── create_snowflake_connection() [External: src/snowflake_connection.py]
│
└── display_result() [Line 705]
```

#### Full Agent System Path
```
main.py:main() [Line 495]
├── SQLAgentSystem.__init__() [Line 37]
├── system.initialize() [Line 85]
│   ├── MemorySystem.initialize() [memory/memory_system.py:64]
│   ├── QueryUnderstandingAgent.__init__() [agents/query_understanding_agent.py:36]
│   ├── DataProfilingAgent.__init__() [agents/data_profiling_agent.py:30]
│   └── SQLVisualizationAgent.__init__() [agents/sql_visualization_agent.py:34]
│
└── system.process_query() [Line 224]
    ├── query_agent.process() [agents/query_understanding_agent.py:147]
    │   ├── _extract_basic_intent() [Line 197]
    │   ├── _get_llm_understanding() [Line 236]
    │   ├── _map_to_schema() [Line 377]
    │   └── _validate_mapping() [Line 483]
    │
    ├── profiling_agent.process() [agents/data_profiling_agent.py]
    │   ├── _profile_columns() [Example method]
    │   └── _generate_smart_filters() [Example method]
    │
    └── sql_viz_agent.process() [agents/sql_visualization_agent.py]
        ├── _generate_sql_from_template() [Example method]
        ├── _apply_guardrails() [Example method]
        └── _determine_visualization() [Example method]
```

### Key Configuration Points

#### Settings Configuration
```python
# File: config/settings.py:20-82
class Settings(BaseSettings):
    # Snowflake Database Settings
    snowflake_account: str = Field(..., description="Snowflake account")    # Line 33
    snowflake_user: str = Field(..., description="Snowflake username")     # Line 34
    
    # Memory System Settings  
    memory_backend: str = Field(default="sqlite")                          # Line 48
    session_db_path: str = Field(default=":memory:")                       # Line 49
    
    # LLM Provider Settings
    llm_provider: str = Field(default="openai")                           # Line 42
    openai_model: str = Field(default="gpt-4o")                           # Line 44
```

#### Database Connection
```python
# File: database/snowflake_connector.py (referenced in snowflake_agent.py:563)
def create_snowflake_connection():
    # Creates authenticated Snowflake connection
    # Returns connection object for query execution
```

---

## Testing Individual Components

### 1. Schema Loading Test
```python
# File: test_main.py:128-186
async def test_schema_loading():
    # Test Excel file parsing
    table_catalog = pd.read_excel(
        TestConfig.SCHEMA_FILE,
        sheet_name='Table_descriptions'
    )
    
    column_catalog = pd.read_excel(
        TestConfig.SCHEMA_FILE, 
        sheet_name="Table's Column Summaries"
    )
    
    # Validate loaded data
    print(f"Tables: {len(table_catalog)}, Columns: {len(column_catalog)}")
```

**Run Test:**
```bash
python test_main.py schema
```

### 2. System Initialization Test
```python
# File: test_main.py:192-242
async def test_system_initialization():
    from main import SQLAgentSystem, get_settings
    
    settings = get_settings()
    system = SQLAgentSystem(settings)
    await system.initialize(TestConfig.SCHEMA_FILE)
    
    # Verify all components
    status = await system.get_system_status()
    print(f"System Status: {status}")
```

**Run Test:**
```bash
python test_main.py init
```

### 3. Agent Pipeline Test
```python
# File: test_main.py:248-319
async def test_agent_pipeline(system, test_query: Dict):
    query_text = test_query["query"]
    result = await system.process_query(query_text)
    
    # Validate pipeline results
    print(f"Processing time: {result.get('processing_time', 0):.2f}s")
    print(f"Confidence: {result.get('query_understanding', {}).get('confidence', 0):.2f}")
```

**Test Queries:**
```python
# File: test_main.py:44-84
TEST_QUERIES = [
    {
        "query": "Show total premium by state for 2023",
        "expected_action": "aggregation",
        "description": "Basic aggregation with time filter"
    },
    {
        "query": "Show premium trends over time by quarter", 
        "expected_action": "trend_analysis",
        "description": "Time-based trend analysis"
    }
]
```

### 4. Memory System Test
```python
# File: test_main.py:366-420
async def test_memory_system(system):
    memory_system = system.memory_system
    
    # Test session creation
    session_id = await memory_system.create_session("test_user")
    
    # Test conversation storage
    await memory_system.add_to_conversation(
        session_id=session_id,
        query="Test query for memory",
        intent={"action": "test"},
        tables_used=["test_table"],
        chart_type="bar_chart"
    )
    
    # Test pattern search
    similar_queries = await memory_system.find_similar_queries("Test query", top_k=3)
    print(f"Similar queries found: {len(similar_queries)}")
```

### 5. Column Name Handling Test
```python
# File: test_main.py:426-466
async def test_column_name_handling(system):
    agent = system.query_agent
    
    test_cases = [
        ("Feature Name", "Feature Name"),
        ("Feature_Name", "Feature Name"),
        ("feature name", "Feature Name"),
        ("Policy Number", "Policy Number"),
    ]
    
    for input_name, expected in test_cases:
        resolved = agent.resolve_column_name(input_name)
        quoted = agent.quote_column_name(resolved or input_name)
        print(f"'{input_name}' → '{resolved}' → '{quoted}'")
```

### 6. Snowflake Agent Standalone Test
```python
# Create test file: test_snowflake_agent.py
import asyncio
from snowflake_agent import SnowflakeAgent

async def test_snowflake_agent():
    # Test with mock schema file
    agent = SnowflakeAgent("test_schema.xlsx")
    
    # Test system info
    info = agent.get_system_info()
    print(f"System Info: {info}")
    
    # Test query processing (without actual Snowflake)
    tables, columns, joins = agent.match_query_to_schema(
        "Show premium trends by state for auto policies in 2023"
    )
    print(f"Schema Analysis: {len(tables)} tables, {len(columns)} columns")

if __name__ == "__main__":
    asyncio.run(test_snowflake_agent())
```

**Run Test:**
```bash
python test_snowflake_agent.py
```

### 7. Interactive Testing Mode
```python
# File: test_main.py:582-685
async def interactive_test_mode():
    # Initialize system
    system = SQLAgentSystem(settings)
    await system.initialize(TestConfig.SCHEMA_FILE)
    
    # Interactive loop
    while True:
        user_input = input(f"\n[TEST] Test Command: ").strip()
        
        if user_input.startswith('test '):
            query = user_input[5:].strip()
            result = await system.process_query(query)
            
            if result.get('success'):
                print(f"[PASS] Success!")
                print(f"SQL: {result.get('sql_query', '')[:200]}...")
```

**Run Interactive Test:**
```bash
python test_main.py interactive
```

### 8. Component-Specific Unit Tests

#### Test Agent Creation
```python
def test_agent_creation():
    """Test individual agent instantiation"""
    from agents.query_understanding_agent import QueryUnderstandingAgent
    
    mock_tables = [{"table_name": "policies", "brief_description": "Policy data"}]
    mock_columns = [{"table_name": "policies", "column_name": "Premium Amount", "data_type": "DECIMAL"}]
    
    agent = QueryUnderstandingAgent(None, None, {}, mock_tables, mock_columns)
    assert agent is not None
    assert len(agent.all_columns) == 1
```

#### Test Column Resolution
```python
def test_column_resolution():
    """Test column name resolution logic"""
    from snowflake_agent import SnowflakeAgent
    
    # Create agent with mock data
    agent = SnowflakeAgent()
    agent.all_columns = [
        {"table_name": "policies", "column_name": "Premium Amount", "data_type": "DECIMAL"}
    ]
    agent._create_column_mappings()
    
    # Test resolution
    assert agent.resolve_column_name("Premium_Amount") == "Premium Amount"
    assert agent.quote_column_name("Premium Amount") == '"Premium Amount"'
```

### 9. Performance Testing
```python
def test_performance():
    """Test system performance with multiple queries"""
    import time
    
    queries = [
        "Show total premium by state",
        "List top 10 customers by claims",
        "Show premium trends over time",
        "Compare claims by region"
    ]
    
    start_time = time.time()
    for query in queries:
        result = agent.process_query(query)
        print(f"Query: {query[:30]}... Time: {result.processing_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Total time for {len(queries)} queries: {total_time:.2f}s")
```

---

## Error Handling and Recovery

### 1. Schema Loading Errors
```python
# File: snowflake_agent.py:124-126
except Exception as e:
    self.logger.error(f"Failed to load schema: {e}")
    raise
```

**Recovery Strategy:**
- Validate Excel file exists and has correct sheets
- Check column names match expected format
- Provide fallback to default schema if needed

### 2. LLM Processing Errors
```python
# File: snowflake_agent.py:366-369
except Exception as ex:
    self.logger.warning(f"LLM schema mapping failed: {ex}")
    matched_tables = []
    matched_columns = []
```

**Recovery Strategy:**
- Fallback to keyword-based mapping
- Use cached patterns from memory system
- Provide default table suggestions

### 3. Database Connection Errors
```python
# File: snowflake_agent.py:579-584
except Exception as ex:
    self.logger.error(f"Snowflake query execution failed: {ex}")
    return pd.DataFrame({
        "error": [f"Execution failed: {str(ex)}"],
        "query": [sql_query]
    })
```

**Recovery Strategy:**
- Return error information in structured format
- Log detailed error for debugging
- Suggest query modifications if possible

### 4. Memory System Errors
```python
# File: memory/memory_system.py:79-81
except Exception as e:
    self.logger.error(f"Failed to setup databases: {e}")
    raise
```

**Recovery Strategy:**
- Graceful degradation without memory features
- Fallback to stateless operation
- Automatic retry with different configuration

---

## Performance Monitoring

### 1. Query Processing Metrics
```python
# File: snowflake_agent.py:600, 643
start_time = time.time()
# ... processing ...
processing_time = time.time() - start_time
```

**Metrics Tracked:**
- Total processing time
- Schema analysis time  
- SQL generation time
- Database execution time
- Result formatting time

### 2. Memory Usage Monitoring
```python
# File: memory/memory_system.py (example structure)
async def get_memory_stats(self) -> Dict[str, Any]:
    return {
        "session_count": await self._count_sessions(),
        "conversation_count": await self._count_conversations(),
        "successful_queries": await self._count_successful_queries(),
        "memory_usage_mb": self._get_memory_usage()
    }
```

### 3. Component Performance Tracking
```python
# Add to each agent process method
def track_performance(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        
        logger.info(f"{func.__name__} completed in {duration:.2f}s")
        return result
    return wrapper
```

### 4. System Health Checks
```python
# File: main.py (example structure)
async def get_system_status(self) -> Dict[str, Any]:
    return {
        "system_status": "healthy",
        "components_active": len([c for c in [self.query_agent, self.profiling_agent] if c]),
        "memory_initialized": self.memory_system._initialized if self.memory_system else False,
        "schema_info": {
            "total_tables": len(self.all_tables) if hasattr(self, 'all_tables') else 0,
            "total_columns": len(self.all_columns) if hasattr(self, 'all_columns') else 0
        }
    }
```

---

## Summary

This workflow documentation provides a complete trace through the SQL Agent System from initial user input to final dashboard visualization. Each step includes:

1. **Exact code references** with file names and line numbers
2. **Function call chains** showing the complete execution path  
3. **Testing procedures** for each component individually
4. **Error handling** and recovery strategies
5. **Performance monitoring** and optimization points

The system supports multiple execution paths:
- **Production**: `snowflake_agent.py` for direct Snowflake integration
- **Development**: `main.py` for full multi-agent system
- **Web Interface**: `ui/streamlit_app.py` for browser-based interaction

All components are thoroughly tested through the comprehensive test suite in `test_main.py`, with individual component testing available for debugging and development.