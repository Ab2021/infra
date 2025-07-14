"""
Snowflake SQL Agent - Production Implementation
Integrates with existing Snowflake infrastructure and GPT API for natural language to SQL conversion.
Aligns with your latest input procedures and handles column name variations correctly.
"""

import pandas as pd
import json
import time
import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# CORE DATA STRUCTURES
# ========================================

@dataclass
class QueryResult:
    """Structured result from query processing"""
    success: bool
    user_query: str
    reasoning: str
    sql_query: str
    relevant_tables: List[str]
    relevant_columns: List[Dict[str, str]]
    relevant_joins: List[Dict[str, str]]
    execution_result: Optional[pd.DataFrame] = None
    processing_time: float = 0.0
    error_message: str = ""

class SnowflakeAgent:
    """
    Production Snowflake SQL Agent System
    
    Features:
    - Intelligent schema analysis from Excel files
    - Natural language to SQL conversion using GPT
    - Column name normalization (handles spaces/underscores)
    - Template-based SQL generation with guardrails
    - Direct Snowflake execution with error handling
    - Comprehensive logging and debugging
    """
    
    def __init__(self, schema_file_path: str = "hackathon_final_schema_file_v1.xlsx"):
        """
        Initialize Snowflake Agent
        
        Args:
            schema_file_path: Path to Excel schema file
        """
        self.schema_file_path = schema_file_path
        self.logger = logging.getLogger(__name__)
        
        # Schema data
        self.table_catalog = None
        self.column_catalog = None
        self.all_tables = []
        self.all_columns = []
        
        # Column name mappings for handling variations
        self.column_mapping = {}
        self.reverse_mapping = {}
        
        # GPT API instance
        self.gpt_object = None
        
        # Sample value field name
        self.SAMPLE_100_VALUE_FIELD = "sample_100_distinct"
        
        # Initialize components
        self._load_schema()
        self._create_column_mappings()
        self._initialize_gpt()
    
    def _load_schema(self):
        """Load schema information from Excel file"""
        try:
            self.logger.info(f"Loading schema from {self.schema_file_path}")
            
            # Load table descriptions
            self.table_catalog = pd.read_excel(
                self.schema_file_path,
                sheet_name='Table_descriptions'
            )[['DATABASE', 'SCHEMA', 'TABLE', 'Brief_Description', 'Detailed_Comments']]
            
            # Load column descriptions
            self.column_catalog = pd.read_excel(
                self.schema_file_path,
                sheet_name="Table's Column Summaries"
            )[['Table Name', 'Feature Name', 'Data Type', 'Description', 'sample_100_distinct']]
            
            # Convert to structured format
            self.all_tables = [
                {
                    "database": row["DATABASE"],
                    "schema": row["SCHEMA"],
                    "table_name": row["TABLE"],
                    "brief_description": row["Brief_Description"],
                    "detailed_comments": row["Detailed_Comments"]
                }
                for idx, row in self.table_catalog.iterrows()
            ]
            
            self.all_columns = [
                {
                    "table_name": row["Table Name"],
                    "column_name": row["Feature Name"],
                    "data_type": row["Data Type"],
                    "description": row["Description"],
                    "sample_100_distinct": row[self.SAMPLE_100_VALUE_FIELD]
                }
                for idx, row in self.column_catalog.iterrows()
            ]
            
            self.logger.info(f"Loaded {len(self.all_tables)} tables and {len(self.all_columns)} columns")
            
        except Exception as e:
            self.logger.error(f"Failed to load schema: {e}")
            raise
    
    def _create_column_mappings(self):
        """Create mappings for column name variations (spaces, underscores, etc.)"""
        self.column_mapping = {}
        self.reverse_mapping = {}
        
        for col in self.all_columns:
            original = str(col["column_name"]).strip()
            normalized = original.replace(' ', '_').replace('-', '_')
            
            # Create bidirectional mappings
            self.column_mapping[normalized.lower()] = original
            self.reverse_mapping[original.lower()] = normalized
            
            # Also map exact matches
            self.column_mapping[original.lower()] = original
            self.reverse_mapping[normalized.lower()] = normalized
        
        self.logger.info(f"Created {len(self.column_mapping)} column name mappings")
    
    def _initialize_gpt(self):
        """Initialize GPT API connection"""
        try:
            from src.gpt_class import GptApi
            self.gpt_object = GptApi()
            self.logger.info("GPT API initialized successfully")
        except ImportError as e:
            self.logger.warning(f"GPT API not available: {e}")
            self.gpt_object = None
    
    # ========================================
    # COLUMN NAME UTILITIES
    # ========================================
    
    def resolve_column_name(self, column_name: str, table_name: str = None) -> str:
        """
        Resolve column name variations to correct database names
        Handles spaces, underscores, and case variations
        """
        if pd.isna(column_name) or not column_name:
            return None
        
        cleaned = str(column_name).strip()
        
        # Try exact match first
        if table_name:
            table_short = table_name.split('.')[-1] if '.' in table_name else table_name
            for col in self.all_columns:
                if (col["table_name"].lower() == table_short.lower() and 
                    col["column_name"].lower() == cleaned.lower()):
                    return col["column_name"]
        
        # Try mapping lookup
        lower_cleaned = cleaned.lower()
        if lower_cleaned in self.column_mapping:
            return self.column_mapping[lower_cleaned]
        
        # Try variations
        variations = [
            cleaned.replace('_', ' '),
            cleaned.replace(' ', '_'),
            cleaned.replace('-', '_'),
            cleaned.replace('_', '-')
        ]
        
        for variation in variations:
            if variation.lower() in self.column_mapping:
                return self.column_mapping[variation.lower()]
        
        return cleaned
    
    def quote_column_name(self, column_name: str) -> str:
        """Quote column names that contain spaces or special characters for SQL"""
        if not column_name:
            return column_name
        
        if (' ' in column_name or 
            '-' in column_name or 
            any(char in column_name for char in ['(', ')', '.', ',', ';']) or
            column_name.upper() in ['ORDER', 'GROUP', 'SELECT', 'FROM', 'WHERE', 'JOIN']):
            return f'"{column_name}"'
        
        return column_name
    
    def validate_and_fix_sql_column_names(self, sql_query: str, relevant_columns: List[Dict]) -> str:
        """Validate and fix column names in SQL query to ensure they match schema"""
        if not sql_query or not relevant_columns:
            return sql_query
        
        fixed_sql = sql_query
        
        # Create mapping of variations to properly quoted names
        column_fixes = {}
        for col in relevant_columns:
            original_name = col["column_name"]
            quoted_name = self.quote_column_name(original_name)
            
            # Map various forms to the properly quoted version
            variations = [
                original_name,
                original_name.replace(' ', '_'),
                original_name.replace('_', ' '),
                original_name.lower(),
                original_name.upper(),
                original_name.replace(' ', '_').lower(),
                original_name.replace(' ', '_').upper()
            ]
            
            for variation in variations:
                if variation != quoted_name:
                    column_fixes[variation] = quoted_name
        
        # Apply fixes using word boundaries
        for wrong_name, correct_name in column_fixes.items():
            pattern = r'\b' + re.escape(wrong_name) + r'\b'
            fixed_sql = re.sub(pattern, correct_name, fixed_sql, flags=re.IGNORECASE)
        
        return fixed_sql
    
    # ========================================
    # SCHEMA ANALYSIS (AGENT 1)
    # ========================================
    
    def extract_unique_joins(self, columns: List[Dict]) -> List[Dict]:
        """Extract unique join key information from columns"""
        join_pairs = set()
        for col in columns:
            join_key = col.get('JOIN_KEY', None)
            table_name = col.get('table_name', None)
            if join_key and table_name:
                join_pairs.add((table_name, join_key))
        
        return [{"table_name": t, "join_key": k} for (t, k) in sorted(join_pairs)]
    
    def match_query_to_schema(self, user_question: str) -> tuple:
        """
        Map user query to relevant tables and columns using LLM or keyword matching
        
        Returns:
            (relevant_tables, relevant_columns, relevant_joins)
        """
        self.logger.info(f"Analyzing query: {user_question}")
        
        uq = user_question.lower()
        matched_tables = []
        matched_columns = []
        
        # Use LLM if available
        if self.gpt_object is not None:
            try:
                # Build context for LLM
                tables_context = "\n".join([
                    f"{tbl['database']}.{tbl['schema']}.{tbl['table_name']}: {tbl['brief_description']}"
                    for tbl in self.all_tables
                ])
                
                columns_context = "\n".join([
                    f"{col['table_name']}.{col['column_name']}: {col['description']}"
                    for col in self.all_columns
                ])
                
                gpt_prompt = f"""
                You are an expert in the actuarial insurance domain with advanced experience in Snowflake SQL and complex insurance data modeling.
                
                Task:
                I will provide you a specific USER QUESTION related to insurance data with SCHEMA, TABLE, COLUMNS given below.
                
                IMPORTANT COLUMN NAMING RULES:
                - Column names in the database may contain SPACES (e.g., "Feature Name", "Policy Number")
                - You MUST use the EXACT column names as shown in the columns context below
                - Do NOT convert spaces to underscores or modify column names
                - If a column name has spaces, use it exactly as shown
                
                REASONING:
                First, provide step-by-step logical reasoning for your design choices:
                - Table Selection: Which tables are relevant, and why?
                - Column Selection: Which fields are required, and why?
                - Join_Key Selection: Which fields are common across tables for joining
                
                CHECK SCHEMAs as well:
                1. schema 'RAW_CI_CAT_ANALYSIS' is majorly linked with CAT Policies
                2. schema 'RAW_CI_INFORCE' contains policy, auto vehicle details
                
                User Question: {user_question}
                
                Database Tables:
                {tables_context}
                
                Table Columns (use EXACT names as shown):
                {columns_context}
                
                Q: From the above, which tables, columns and join_key would you use to answer the question?
                
                CRITICAL: Use the EXACT column names as shown in the Table Columns section above.
                
                Provide as JSON like this:
                {{
                    "relevant_tables": ["..."],
                    "relevant_columns": [{{"table_name": "...", "column_name": "...", "JOIN_KEY": "..."}}]
                }}
                
                Strictly output JSON, do NOT explain. VALIDATE that column names match exactly.
                """
                
                payload = {
                    "username": "SNOWFLAKE_SCHEMA_AGENT",
                    "session_id": "1",
                    "messages": [{"role": "user", "content": gpt_prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1024
                }
                
                resp = self.gpt_object.get_gpt_response_non_streaming(payload)
                content = resp.json()['choices'][0]['message']['content']
                first = content.find('{')
                last = content.rfind('}') + 1
                parsed = json.loads(content[first:last])
                
                matched_tables = parsed.get("relevant_tables", [])
                matched_columns = parsed.get("relevant_columns", [])
                
                # Validate and resolve column names
                validated_columns = []
                for col in matched_columns:
                    table_name = col.get("table_name", "")
                    column_name = col.get("column_name", "")
                    resolved_name = self.resolve_column_name(column_name, table_name)
                    
                    if resolved_name:
                        validated_columns.append({
                            "table_name": table_name,
                            "column_name": resolved_name,
                            "JOIN_KEY": col.get("JOIN_KEY", "")
                        })
                    else:
                        self.logger.warning(f"Could not resolve column: {table_name}.{column_name}")
                
                matched_columns = validated_columns
                
            except Exception as ex:
                self.logger.warning(f"LLM schema mapping failed: {ex}")
                matched_tables = []
                matched_columns = []
        
        # Fallback to keyword matching
        if not matched_tables and not matched_columns:
            self.logger.info("Using keyword-based schema matching")
            keywords = set(uq.replace(",", " ").replace("_", " ").split())
            
            matched_tables = [
                f"{tbl['database']}.{tbl['schema']}.{tbl['table_name']}"
                for tbl in self.all_tables
                if any(k in (tbl['table_name'].lower() + " " +
                           (str(tbl['brief_description']) or "")).lower()
                       for k in keywords)
            ]
            
            matched_columns = [
                {"table_name": col['table_name'], "column_name": col['column_name']}
                for col in self.all_columns
                if any(k in (col['column_name'] + " " +
                           (str(col['description']) or "")).lower()
                       for k in keywords)
            ]
        
        matched_joins = self.extract_unique_joins(matched_columns)
        
        self.logger.info(f"Schema analysis complete: {len(matched_tables)} tables, {len(matched_columns)} columns")
        return matched_tables, matched_columns, matched_joins
    
    # ========================================
    # SQL GENERATION (AGENT 2)
    # ========================================
    
    def build_column_context_block(self, relevant_columns: List[Dict]) -> str:
        """Build detailed context block for GPT prompt with column metadata"""
        blocks = []
        
        for rel in relevant_columns:
            tname = rel["table_name"]
            cname = rel["column_name"]
            
            # Find column info from catalog
            table_short_name = tname.split('.')[-1] if '.' in tname else tname
            col_row = None
            
            # Find matching column
            for c in self.all_columns:
                if (c["table_name"].lower() == table_short_name.lower() and 
                    c["column_name"].lower() == cname.lower()):
                    col_row = c
                    break
            
            if not col_row:
                self.logger.warning(f"Column metadata not found: {tname}.{cname}")
                continue
            
            block = (
                f"\n ||| Table_name: {tname} "
                f"|| Feature_name: {col_row['column_name']} "
                f"|| Data_Type: {col_row['data_type']} "
                f"|| Description: {col_row['description']} "
                f"|| 100 Sample Values (separated by ,): {col_row['sample_100_distinct']} ||| \n"
            )
            blocks.append(block)
        
        return "".join(blocks)
    
    def generate_sql(self, user_question: str, relevant_columns: List[Dict], 
                    relevant_joins: List[Dict]) -> tuple:
        """
        Generate SQL query using GPT with enhanced prompting
        
        Returns:
            (reasoning, sql_query)
        """
        if not self.gpt_object:
            return "GPT not available", "SELECT 1 as placeholder"
        
        try:
            # Build context block
            context_block = self.build_column_context_block(relevant_columns)
            
            # Load few-shot examples if available
            few_shot_examples = ""
            try:
                with open('few_shot_examples.txt', 'r', encoding='utf-8') as f:
                    few_shot_examples = f.read()
            except FileNotFoundError:
                self.logger.warning("Few-shot examples file not found")
            
            # Compose enhanced prompt
            gpt_prompt = f"""
            ****FEW-SHOT EXAMPLES:****
            {few_shot_examples}
            ****END EXAMPLES****

            You must answer ONLY using the following columns and tables. Do NOT invent or guess any table or column names or join_keys.

            **Relevant Columns and Tables with Sample Values:**
            ```{context_block}```

            **Relevant Tables and Join_Keys :**
            ```{relevant_joins}```

            **User Question:**
            ```{user_question}```

            **MANDATORY Instructions:**

            - Only reference columns and tables listed above; ignore any others.
            - CRITICAL: Use EXACT column names as shown in the context above.
            - If a column name contains spaces, wrap it in double quotes in the SQL (e.g., "Feature Name").
            - USE the joins based on Relevant Tables and Join_Keys.
            - THINK BEFORE YOU CREATE THE LOGIC; WE MIGHT NOT NEED JOINS IN EVERY PROVIDED TABLE ALWAYS IF FEATURES ARE PRESENT.
            - For any user-specified filter value, find the closest corresponding value from the '100 sample values' shown.
            - If a user-filtered column is not listed, explicitly note its unavailability and do NOT invent data.
            - For DATE or _DT columns:
                - **Do NOT perform mapping for user values. State "N/A" for these columns in the Mapping section.**
                - **Do NOT consider or match actual sample date values. Restrict logic to the format (e.g., Year, Month, or Timestamp).**
            - If you create any mappings, list them clearly.
            - Follow the output template exactly.
            - THINK ABOUT AGGREGATIONS, CASE WHEN CONDITIONAL STATEMENTS WHEN CREATING THE LOGIC
            - Ensure all column names in SQL match exactly with the provided context and are properly quoted if they contain spaces.

            **Your output must have this structure:**

            **Mapping:**  
            (User value → Actual DB/sample value for WHERE clause. 
            DO NOT apply the filtering condition if value not found.)

            **Reasoning:**  
            (Brief explanation in 100 words of your mapping choices, treatment of any unavailable columns/filters, and your general approach to the SQL construction.)

            **SQL Query:**  
            (The corresponding query, using only the mapped values from Mapping above and properly quoted column names.)

            From the above:
            Fetch Reasoning and 'SQL Query' (that can be passed to snowflake) and Provide them as JSON like this:
            {{
                "Reasoning": "",
                "SQL Query": ""
            }}
            Strictly output JSON, do NOT explain. VALIDATE that all column names in SQL match the provided context exactly.
            """
            
            payload = {
                "username": "SNOWFLAKE_SQL_AGENT",
                "session_id": "1",
                "messages": [{"role": "user", "content": gpt_prompt}],
                "temperature": 0.2,
                "max_tokens": 2048
            }
            
            resp = self.gpt_object.get_gpt_response_non_streaming(payload)
            gpt_output = resp.json()['choices'][0]['message']['content']
            
            # Parse response
            first = gpt_output.find('{')
            last = gpt_output.rfind('}') + 1
            parsed = json.loads(gpt_output[first:last])
            
            reasoning = parsed.get("Reasoning", "")
            sql_query = parsed.get("SQL Query", "")
            
            # Validate and fix column names in SQL
            if sql_query:
                sql_query = self.validate_and_fix_sql_column_names(sql_query, relevant_columns)
            
            self.logger.info(f"SQL generation complete: {len(sql_query)} characters")
            return reasoning, sql_query
            
        except Exception as ex:
            self.logger.error(f"SQL generation failed: {ex}")
            return f"Error: {ex}", "SELECT 1 as error_query"
    
    # ========================================
    # SNOWFLAKE EXECUTION
    # ========================================
    
    def execute_sql_query(self, sql_query: str) -> pd.DataFrame:
        """
        Execute SQL query on Snowflake and return results as DataFrame
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            DataFrame with query results
        """
        if not sql_query:
            self.logger.error("No SQL query provided for execution")
            return pd.DataFrame({"error": ["No SQL query provided"]})
        
        try:
            # Import Snowflake connection
            from src.snowflake_connection import create_snowflake_connection
            
            self.logger.info(f"Executing SQL query: {sql_query[:100]}...")
            
            conn = create_snowflake_connection()
            with conn.cursor() as cursor:
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                colnames = [d[0] for d in cursor.description]
                df = pd.DataFrame(rows, columns=colnames)
            
            conn.close()
            
            self.logger.info(f"Query executed successfully: {len(df)} rows returned")
            return df
            
        except Exception as ex:
            self.logger.error(f"Snowflake query execution failed: {ex}")
            return pd.DataFrame({
                "error": [f"Execution failed: {str(ex)}"],
                "query": [sql_query]
            })
    
    # ========================================
    # MAIN PIPELINE
    # ========================================
    
    def process_query(self, user_question: str) -> QueryResult:
        """
        Process user query through the complete pipeline
        
        Args:
            user_question: Natural language query
            
        Returns:
            QueryResult with all processing information
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting query processing: {user_question}")
            
            # Step 1: Schema Analysis
            self.logger.info("Step 1: Analyzing schema and identifying relevant tables/columns")
            relevant_tables, relevant_columns, relevant_joins = self.match_query_to_schema(user_question)
            
            if not relevant_tables or not relevant_columns:
                return QueryResult(
                    success=False,
                    user_query=user_question,
                    reasoning="No relevant tables or columns found for the query",
                    sql_query="",
                    relevant_tables=[],
                    relevant_columns=[],
                    relevant_joins=[],
                    processing_time=time.time() - start_time,
                    error_message="Schema analysis failed to find relevant entities"
                )
            
            # Step 2: SQL Generation
            self.logger.info("Step 2: Generating SQL query")
            reasoning, sql_query = self.generate_sql(user_question, relevant_columns, relevant_joins)
            
            if not sql_query or sql_query.strip() == "":
                return QueryResult(
                    success=False,
                    user_query=user_question,
                    reasoning=reasoning,
                    sql_query="",
                    relevant_tables=relevant_tables,
                    relevant_columns=relevant_columns,
                    relevant_joins=relevant_joins,
                    processing_time=time.time() - start_time,
                    error_message="SQL generation failed"
                )
            
            # Step 3: SQL Execution
            self.logger.info("Step 3: Executing SQL query on Snowflake")
            execution_result = self.execute_sql_query(sql_query)
            
            processing_time = time.time() - start_time
            
            # Check if execution was successful
            if "error" in execution_result.columns:
                return QueryResult(
                    success=False,
                    user_query=user_question,
                    reasoning=reasoning,
                    sql_query=sql_query,
                    relevant_tables=relevant_tables,
                    relevant_columns=relevant_columns,
                    relevant_joins=relevant_joins,
                    execution_result=execution_result,
                    processing_time=processing_time,
                    error_message=execution_result["error"].iloc[0] if len(execution_result) > 0 else "Unknown execution error"
                )
            
            # Success!
            self.logger.info(f"Query processing completed successfully in {processing_time:.2f} seconds")
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
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Query processing failed: {e}")
            return QueryResult(
                success=False,
                user_query=user_question,
                reasoning="",
                sql_query="",
                relevant_tables=[],
                relevant_columns=[],
                relevant_joins=[],
                processing_time=processing_time,
                error_message=str(e)
            )
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics"""
        return {
            "schema_file": self.schema_file_path,
            "total_tables": len(self.all_tables),
            "total_columns": len(self.all_columns),
            "column_mappings": len(self.column_mapping),
            "gpt_available": self.gpt_object is not None,
            "sample_tables": [t["table_name"] for t in self.all_tables[:5]],
            "sample_columns": [c["column_name"] for c in self.all_columns[:10]]
        }
    
    def display_result(self, result: QueryResult):
        """Display query result in a formatted way"""
        print(f"\n{'='*60}")
        print(f"🔍 QUERY PROCESSING RESULT")
        print(f"{'='*60}")
        print(f"📝 Query: {result.user_query}")
        print(f"✅ Success: {result.success}")
        print(f"⏱️  Processing Time: {result.processing_time:.2f} seconds")
        
        if not result.success:
            print(f"❌ Error: {result.error_message}")
            return
        
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   Tables: {len(result.relevant_tables)}")
        print(f"   Columns: {len(result.relevant_columns)}")
        print(f"   Joins: {len(result.relevant_joins)}")
        
        print(f"\n🧠 REASONING:")
        print(f"   {result.reasoning}")
        
        print(f"\n📝 GENERATED SQL:")
        print(f"   {result.sql_query}")
        
        if result.execution_result is not None and not result.execution_result.empty:
            print(f"\n📊 EXECUTION RESULTS:")
            print(f"   Rows: {len(result.execution_result)}")
            print(f"   Columns: {list(result.execution_result.columns)}")
            print(f"\n   Sample Data:")
            print(result.execution_result.head().to_string(index=False))

# ========================================
# MAIN EXECUTION
# ========================================

def main():
    """Main execution function for testing"""
    
    print(f"\n🚀 SNOWFLAKE SQL AGENT - PRODUCTION SYSTEM")
    print(f"{'='*60}")
    
    try:
        # Initialize agent
        agent = SnowflakeAgent()
        
        # Display system info
        info = agent.get_system_info()
        print(f"📊 System Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Example test query
        test_query = "List auto policies with premium over 10000 in Texas in 2023"
        
        print(f"\n🧪 Testing with example query:")
        print(f"   '{test_query}'")
        
        # Process query
        result = agent.process_query(test_query)
        
        # Display results
        agent.display_result(result)
        
        # Interactive mode
        print(f"\n{'='*60}")
        print(f"🎮 INTERACTIVE MODE")
        print(f"Enter queries or 'exit' to quit")
        print(f"{'='*60}")
        
        while True:
            try:
                user_input = input(f"\n📝 Query: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                print(f"\n🔄 Processing...")
                result = agent.process_query(user_input)
                agent.display_result(result)
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    except Exception as e:
        print(f"\n💥 System initialization failed: {e}")

if __name__ == "__main__":
    main()