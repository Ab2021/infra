import pandas as pd
import json
import time
import re
from src.snowflake_connection import create_snowflake_connection

# ------------------ INITIAL SETUP: LOAD DATA --------------------

from src.gpt_class import GptApi

gpt_object = GptApi()

# Load table descriptions (schema, table, and per-table details)
table_catalog = pd.read_excel(
    "hackathon_final_schema_file_v1.xlsx",
    sheet_name='Table_descriptions'
)[['DATABASE', 'SCHEMA', 'TABLE', 'Brief_Description', 'Detailed_Comments']]

# Load column descriptions (per-table, per-column, and EDA)
column_catalog_df = pd.read_excel(
    "hackathon_final_schema_file_v1.xlsx",
    sheet_name="Table's Column Summaries"
)[['Table Name', 'Feature Name', 'Data Type', 'Description', 'sample_100_distinct']]

SAMPLE_100_VALUE_FIELD = "sample_100_distinct"  

# -- Convert to list-of-dicts for uniform access --
all_tables = [
    {
        "database": row["DATABASE"],
        "schema": row["SCHEMA"],
        "table_name": row["TABLE"],
        "brief_description": row["Brief_Description"],
        "detailed_comments": row["Detailed_Comments"]
    }
    for idx, row in table_catalog.iterrows()
]

all_columns = [
    {
        "table_name": row["Table Name"],
        "column_name": row["Feature Name"],
        "data_type": row["Data Type"],
        "description": row["Description"],
        "sample_100_distinct": row[SAMPLE_100_VALUE_FIELD]
    }
    for idx, row in column_catalog_df.iterrows()
]

# ------------------ COLUMN NAME UTILITIES -------------------------

def normalize_column_name(column_name):
    """
    Normalize column names to handle spaces vs underscores consistently.
    Returns both the original name and a normalized version.
    """
    if pd.isna(column_name):
        return "", ""
    
    original = str(column_name).strip()
    # Replace spaces with underscores for LLM consistency
    normalized = original.replace(' ', '_').replace('-', '_')
    return original, normalized

def create_column_mapping():
    """
    Create a mapping between normalized column names and original column names.
    This helps resolve mismatches between LLM output and actual database columns.
    """
    column_mapping = {}
    reverse_mapping = {}
    
    for col in all_columns:
        original, normalized = normalize_column_name(col["column_name"])
        if original and normalized:
            column_mapping[normalized.lower()] = original
            reverse_mapping[original.lower()] = normalized
            # Also map exact matches
            column_mapping[original.lower()] = original
            reverse_mapping[normalized.lower()] = normalized
    
    return column_mapping, reverse_mapping

# Global mappings for column name resolution
COLUMN_MAPPING, REVERSE_MAPPING = create_column_mapping()

def resolve_column_name(column_name, table_name=None):
    """
    Resolve column name variations to the correct database column name.
    Handles spaces, underscores, and case variations.
    """
    if pd.isna(column_name):
        return None
    
    # Clean the input
    cleaned = str(column_name).strip()
    
    # Try exact match first
    if table_name:
        # Look for column in specific table
        for col in all_columns:
            if (col["table_name"].lower() == table_name.lower() and 
                col["column_name"].lower() == cleaned.lower()):
                return col["column_name"]
    
    # Try mapping lookup
    lower_cleaned = cleaned.lower()
    if lower_cleaned in COLUMN_MAPPING:
        return COLUMN_MAPPING[lower_cleaned]
    
    # Try with space/underscore variations
    variations = [
        cleaned.replace('_', ' '),
        cleaned.replace(' ', '_'),
        cleaned.replace('-', '_'),
        cleaned.replace('_', '-')
    ]
    
    for variation in variations:
        if variation.lower() in COLUMN_MAPPING:
            return COLUMN_MAPPING[variation.lower()]
    
    # If no mapping found, return original but with proper SQL quoting
    return cleaned

def quote_column_name(column_name):
    """
    Properly quote column names for SQL if they contain spaces or special characters.
    """
    if not column_name:
        return column_name
    
    # If column name contains spaces, special chars, or is a reserved word, quote it
    if (' ' in column_name or 
        '-' in column_name or 
        any(char in column_name for char in ['(', ')', '.', ',', ';']) or
        column_name.upper() in ['ORDER', 'GROUP', 'SELECT', 'FROM', 'WHERE', 'JOIN']):
        return f'"{column_name}"'
    
    return column_name

# ------------------ PROMPT UTILITIES -------------------------

def build_column_context_block(relevant_columns, all_columns):
    """
    Return a string giving, for all columns-of-interest, precise metadata block for GPT prompt.
    Enhanced to handle column name resolution and provide clear naming guidance.
    """
    blocks = []
    for rel in relevant_columns:
        tname = rel["table_name"]
        cname = rel["column_name"]
        
        # Resolve column name variations
        resolved_cname = resolve_column_name(cname, tname.split('.')[-1])
        
        # Find column info
        table_short_name = tname.split('.')[-1] if '.' in tname else tname
        col_row = None
        
        # Try exact match first
        for c in all_columns:
            if (c["table_name"].lower() == table_short_name.lower() and 
                c["column_name"].lower() == resolved_cname.lower()):
                col_row = c
                break
        
        # Fallback to fuzzy match
        if not col_row:
            for c in all_columns:
                if (c["table_name"].lower() == table_short_name.lower() and 
                    (c["column_name"].lower() == cname.lower() or
                     c["column_name"].replace(' ', '_').lower() == cname.lower() or
                     c["column_name"].replace('_', ' ').lower() == cname.lower())):
                    col_row = c
                    break
        
        if not col_row:
            print(f"[WARNING] Column not found: {tname}.{cname}")
            continue
        
        # Use the actual column name from the catalog
        actual_column_name = col_row['column_name']
        
        block = (
            f"\n ||| Table_name: {tname} "
            f"|| Feature_name: {actual_column_name} "
            f"|| Data_Type: {col_row['data_type']} "
            f"|| Description: {col_row['description']} "
            f"|| 100 Sample Values (separated by ,): {col_row['sample_100_distinct']} ||| \n"
        )
        blocks.append(block)
    
    return "".join(blocks)

# ------------------ AGENT: NLQ → RELEVANT SCHEMA (LLM OR KEYWORD) --------------------------

def extract_unique_joins(columns):
    """
    Given a list of column dicts, extract unique {table_name, join_key} pairs.
    """
    join_pairs = set()
    for col in columns:
        # Safely handle 'JOIN_KEY' possibly missing in fallback scenario
        join_key = col.get('JOIN_KEY', None)
        table_name = col.get('table_name', None)
        if join_key and table_name:
            join_pairs.add( (table_name, join_key) )
    # Convert set of tuples to list of dicts if preferred
    return [ {"table_name": t, "join_key": k} for (t,k) in sorted(join_pairs) ]

def match_query_to_schema(user_question, all_tables, all_columns, gpt_object=None):
    """
    Given user NLQ, return (relevant_tables, relevant_columns)
    - If gpt_object provided: call GPT for mapping.
    - Else: fallback on keyword-matching.
    Enhanced with better column name handling.
    """
    uq = user_question.lower()
    matched_tables = []
    matched_columns = []
    print(f"User Question: {user_question}")

    # --- LLM Option ---
    if gpt_object is not None:
        # Build context with clear column naming instructions
        tables_context = "\n".join(
            [f"{tbl['database']}.{tbl['schema']}.{tbl['table_name']}: {tbl['brief_description']}" for tbl in all_tables])
        
        # Enhanced columns context with exact naming
        columns_context = "\n".join(
            [f"{col['table_name']}.{col['column_name']}: {col['description']}" for col in all_columns])
        
        gpt_prompt = f"""
                        You are an expert in the actuarial insurance domain with advanced experience in Snowflake SQL and complex insurance data modeling.
                        
                        Task:
                            I will provide you a specific USER QUESTION or ANALYTICAL REQUIREMENT related to insurance data with SCHEMA, TABLE, COLUMNS given below, all related to Insurance Domain.
                           
                        IMPORTANT COLUMN NAMING RULES:
                        - Column names in the database may contain SPACES (e.g., "Feature Name", "Table Name")
                        - You MUST use the EXACT column names as shown in the columns context below
                        - Do NOT convert spaces to underscores or modify column names
                        - If a column name has spaces, use it exactly as shown
                        
                        REASONING:
                        First, provide step-by-step logical reasoning for your design choices, considering:
                        Table Selection: Which tables are relevant, and why?
                        Column Selection: Which fields are required, and why are they necessary for the analysis?
                        Join_Key Selection: Which fields from the table are common across tables and are mentioned as identifiers/keys in brief_description
                        
                        CHECK SCHEMAs as well in the Database Table. 
                        Eg. 
                        1.schema 'RAW_CI_CAT_ANALYSIS' is majorly linked with CAT Policies
                        2.schema 'RAW_CI_INFORCE' contains policy, auto vehicle details

                        Make sure to identify the main tables first and all required columns in those tables only, containing the relevant information. 
                        Check question if it is related to catastrophic insurance or not and if not then do not return the CAT tables for non-CAT insurance.

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
                        
                        Strictly output JSON, do NOT explain. VALIDATE that column names match exactly with the provided columns.
                        """
        
        messages = [{"role": "user", "content": gpt_prompt}]
        payload = {
            "username": "GPT_SCHEMA_AGENT",
            "session_id": "1",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1024
        }
        try:
            resp = gpt_object.get_gpt_response_non_streaming(payload)
            content = resp.json()['choices'][0]['message']['content']
            first = content.find('{')
            last = content.rfind('}')+1
            parsed = json.loads(content[first:last])
            matched_tables = parsed.get("relevant_tables", [])
            matched_columns = parsed.get("relevant_columns", [])
            
            # Validate and resolve column names
            validated_columns = []
            for col in matched_columns:
                table_name = col.get("table_name", "")
                column_name = col.get("column_name", "")
                resolved_name = resolve_column_name(column_name, table_name)
                
                if resolved_name:
                    validated_columns.append({
                        "table_name": table_name,
                        "column_name": resolved_name,
                        "JOIN_KEY": col.get("JOIN_KEY", "")
                    })
                else:
                    print(f"[WARNING] Could not resolve column: {table_name}.{column_name}")
            
            matched_columns = validated_columns
            
        except Exception as ex:
            print("[WARN] GPT table/column mapping error:", ex)
    
    matched_joins = []

    # --- If GPT failed/fallback: do simple keyword-based matching ---
    if not matched_tables and not matched_columns:
        keywords = set(uq.replace(",", " ").replace("_", " ").split())
        matched_tables = [
            f"{tbl['database']}.{tbl['schema']}.{tbl['table_name']}"
            for tbl in all_tables
            if any(k in (tbl['table_name'].lower() + " " +
                         (str(tbl['brief_description']) or "")).lower()
                   for k in keywords)
        ]
        matched_columns = [
            {"table_name": col['table_name'], "column_name": col['column_name']}
            for col in all_columns
            if any(k in (col['column_name'] + " " +
                         (str(col['description']) or "")).lower()
                   for k in keywords)
        ]

    matched_joins = extract_unique_joins(matched_columns)
    return matched_tables, matched_columns, matched_joins

def first_tool_call(state):
    """
    Node 1: Given NLQ and catalogs, pick tables and columns.
    Enhanced with column name validation.
    """
    user_question = state.get("user_question")
    all_tables    = state.get("table_schema")
    all_columns   = state.get("columns_info")
    gpt_object    = state.get("gpt_object", None)

    relevant_tables, relevant_columns, relevant_joins = match_query_to_schema(
        user_question, all_tables, all_columns, gpt_object
    )
    
    # Log column resolution results
    print(f"[INFO] Resolved {len(relevant_columns)} columns:")
    for col in relevant_columns:
        print(f"  - {col['table_name']}.{col['column_name']}")
    
    state["relevant_tables"] = relevant_tables
    state["relevant_columns"] = relevant_columns
    state["relevant_joins"] = relevant_joins
    return state

# -------------------- AGENT: GPT SQL GENERATOR ----------------------

def query_gen_node(state):
    """
    Node 2: Given selected columns and question, ask GPT to reason and write SQL.
    Enhanced with proper column name quoting and validation.
    """
    user_question = state.get("user_question")
    all_columns = state.get("columns_info")
    relevant_columns = state.get("relevant_columns")
    relevant_joins = state.get("relevant_joins")
    gpt_object = state.get("gpt_object")

    # Step 1: Build context for the question
    context_block = build_column_context_block(relevant_columns, all_columns)

    with open('few_shot_examples.txt', 'r', encoding='utf-8') as f:
        few_shot_examples = f.read()

    # Step 2: Compose final prompt with enhanced column handling
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
        - For any user-specified filter value, find the closest corresponding value from the '100 sample values' shown. For example, if user says "Texas" and the sample value is "TX", map as "Texas" → "TX" in your mapping, and use only "TX" in the SQL.
        - If a user-filtered column is not listed, explicitly note its unavailability and do NOT invent data or answer as if it exists.
        - For DATE or _DT columns:
            - **Do NOT perform mapping for user values. State "N/A" for these columns in the Mapping section.**
            - **Do NOT consider or match actual sample date values. Restrict logic to the format (e.g., Year, Month, or Timestamp) as illustrated in provided columns.**
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

    messages = [{"role": "user", "content": gpt_prompt}]
    payload = {
        "username": "GPT_SQL_AGENT",
        "session_id": "1",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048
    }

    matched_sql_query = []
    matched_reasoning = []

    try:
        resp = gpt_object.get_gpt_response_non_streaming(payload)
        gpt_output = resp.json()['choices'][0]['message']['content']

        first = gpt_output.find('{')
        last = gpt_output.rfind('}')+1
        parsed = json.loads(gpt_output[first:last])
        matched_reasoning = parsed.get("Reasoning", [])
        matched_sql_query = parsed.get("SQL Query", [])
        
        # Validate and fix column names in SQL
        if matched_sql_query:
            matched_sql_query = validate_and_fix_sql_column_names(matched_sql_query, relevant_columns)
        
    except Exception as ex:
        print("[WARN] GPT reasoning/sqlquery mapping error:", ex)
        
    state["query_llm_prompt"] = gpt_prompt
    state["query_llm_result"] = gpt_output
    state["query_reasoning"] = matched_reasoning
    state["query_sql"] = matched_sql_query
    return state

def validate_and_fix_sql_column_names(sql_query, relevant_columns):
    """
    Validate and fix column names in the SQL query to ensure they match the schema.
    """
    if not sql_query or not relevant_columns:
        return sql_query
    
    fixed_sql = sql_query
    
    # Create a mapping of all possible column name variations to properly quoted names
    column_fixes = {}
    for col in relevant_columns:
        original_name = col["column_name"]
        quoted_name = quote_column_name(original_name)
        
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
    
    # Apply fixes using word boundaries to avoid partial matches
    for wrong_name, correct_name in column_fixes.items():
        # Use word boundaries and handle both quoted and unquoted cases
        pattern = r'\b' + re.escape(wrong_name) + r'\b'
        fixed_sql = re.sub(pattern, correct_name, fixed_sql, flags=re.IGNORECASE)
    
    return fixed_sql

# ------------------ SNOWFLAKE SQL EXECUTION: STATE["query_sql"] → DataFrame --------------------

def run_query_and_return_df(state):
    """
    Executes the SQL query found in state['query_sql'] on Snowflake and returns DataFrame result.
    Updates state['snowflake_result'] with a pandas DataFrame (or None on error).
    Enhanced with better error handling and column name debugging.
    """
    query_sql = state.get("query_sql", None)
    if not query_sql:
        print("[ERROR] No SQL found in state['query_sql']")
        state['snowflake_result'] = None
        return state
    
    print(f"[INFO] Executing SQL: {query_sql}")
    
    # Connect to Snowflake
    conn = None
    try:
        conn = create_snowflake_connection()
        with conn.cursor() as cursor:
            cursor.execute(query_sql)
            # Fetch all rows and columns
            rows = cursor.fetchall()
            colnames = [d[0] for d in cursor.description]
            df = pd.DataFrame(rows, columns=colnames)
            print(f"[SUCCESS] Query returned {len(df)} rows and {len(df.columns)} columns")
        state['snowflake_result'] = df
    except Exception as ex:
        print(f"[ERROR] Snowflake query execution failed: {ex}")
        print(f"[ERROR] Problematic SQL: {query_sql}")
        
        # Try to provide helpful debugging info
        if "invalid identifier" in str(ex).lower() or "column" in str(ex).lower():
            print("[DEBUG] This might be a column name issue. Check:")
            print("1. Column names with spaces should be quoted")
            print("2. Column names should match exactly with schema")
            print("3. Check for typos in column names")
        
        state['snowflake_result'] = None
        state['sql_error'] = str(ex)
    finally:
        if conn is not None:
            conn.close()
    return state
    
# ----------- PIPELINE ORCHESTRATION -----------

if __name__ == "__main__":
    # EXAMPLE pipeline run:
    state = {
        "user_question": "List auto policies with premium over 10000 in Texas in 2023",
        "table_schema": all_tables,
        "columns_info": all_columns,
        "gpt_object": gpt_object
    }
    
    print("=== COLUMN MAPPING INFO ===")
    print(f"Total columns in schema: {len(all_columns)}")
    print("Sample column names:")
    for i, col in enumerate(all_columns[:5]):
        print(f"  {i+1}. {col['table_name']}.{col['column_name']}")
    
    print("\n=== RUNNING PIPELINE ===")
    print("Running first_tool_call...")
    state = first_tool_call(state)
    print("Relevant tables:", state["relevant_tables"])
    print("Relevant columns:", state["relevant_columns"])
    print("Relevant Joins:", state["relevant_joins"])
    
    print("\nRunning query_gen_node (this may take up to a minute)...")
    state = query_gen_node(state)
    print("\n========== GPT Reasoning ===============")
    print(state["query_reasoning"])
    print("\n========== SQL Output ===============")
    print(state["query_sql"])
    
    # ---- Run SQL on Snowflake! ----
    print("\nRunning query on Snowflake...")
    state = run_query_and_return_df(state)
    print("\n========== Query Result (DataFrame head) ===============")
    if isinstance(state['snowflake_result'], pd.DataFrame):
        print(state['snowflake_result'].head())
        print(f"\nDataFrame shape: {state['snowflake_result'].shape}")
        print(f"Columns: {list(state['snowflake_result'].columns)}")
    else:
        print("No results or query failed.")
        if 'sql_error' in state:
            print(f"Error details: {state['sql_error']}")