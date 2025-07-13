"""
Streamlit Dashboard for Advanced SQL Agent System
Provides intuitive interface for natural language to SQL conversion with visualizations
"""

import streamlit as st
import asyncio
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Advanced SQL Agent System",
    page_icon="ðŸ¤–",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-status {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .sql-code {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.25rem;
        padding: 1rem;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'visualization_type': 'auto',
            'sql_complexity': 'balanced',
            'show_query_explanation': True
        }

async def process_query_async(query, user_id="streamlit_user"):
    """Async wrapper for query processing."""
    
    # Import here to avoid circular imports
    from main import SQLAgentSystem
    
    try:
        system = SQLAgentSystem()
        result = await system.process_query(query, user_id)
        return result
    except Exception as e:
        return {"error": str(e), "success": False}

def process_natural_language_query(query):
    """Process natural language query through the SQL agent system."""
    
    # Run async function in streamlit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(process_query_async(query))
        return result
    finally:
        loop.close()

def display_agent_processing_status(status_data):
    """Display real-time agent processing status."""
    
    if not status_data:
        return
    
    st.subheader("ðŸ¤– Agent Processing Status")
    
    # Create columns for each agent
    agents = ['NLU', 'Schema', 'SQL Generator', 'Validator', 'Visualizer']
    cols = st.columns(len(agents))
    
    for i, agent in enumerate(agents):
        with cols[i]:
            agent_status = status_data.get(agent.lower().replace(' ', '_'), 'pending')
            if agent_status == 'completed':
                st.success(f"âœ… {agent}")
            elif agent_status == 'processing':
                st.info(f"â³ {agent}")
            elif agent_status == 'error':
                st.error(f"âŒ {agent}")
            else:
                st.info(f"â­• {agent}")

def display_query_results(results):
    """Display query results with visualizations."""
    
    if not results or 'query_results' not in results:
        return
    
    st.subheader("ðŸ“Š Query Results")
    
    # Display generated SQL
    if 'generated_sql' in results:
        st.subheader("Generated SQL")
        st.code(results['generated_sql'], language='sql')
        
        # Add copy button
        if st.button("ðŸ“‹ Copy SQL"):
            st.code(results['generated_sql'])
            st.success("SQL copied to display!")
    
    # Display results table
    query_results = results['query_results']
    if query_results:
        df = pd.DataFrame(query_results)
        
        st.subheader("Results Table")
        st.dataframe(df, use_container_width=True)
        
        # Auto-generate visualizations if numeric data is present
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_columns) > 0 and len(categorical_columns) > 0:
            st.subheader("ðŸ“ˆ Auto-Generated Visualizations")
            
            # Create tabs for different chart types
            chart_tabs = st.tabs(["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot"])
            
            with chart_tabs[0]:
                if len(categorical_columns) > 0 and len(numeric_columns) > 0:
                    fig = px.bar(df, x=categorical_columns[0], y=numeric_columns[0],
                               title=f"{numeric_columns[0]} by {categorical_columns[0]}")
                    st.plotly_chart(fig, use_container_width=True)
            
            with chart_tabs[1]:
                if len(numeric_columns) >= 2:
                    fig = px.line(df, x=df.columns[0], y=numeric_columns[0],
                                title=f"{numeric_columns[0]} Over Time")
                    st.plotly_chart(fig, use_container_width=True)
            
            with chart_tabs[2]:
                if len(categorical_columns) > 0 and len(numeric_columns) > 0:
                    fig = px.pie(df, names=categorical_columns[0], values=numeric_columns[0],
                               title=f"Distribution of {numeric_columns[0]}")
                    st.plotly_chart(fig, use_container_width=True)
            
            with chart_tabs[3]:
                if len(numeric_columns) >= 2:
                    fig = px.scatter(df, x=numeric_columns[0], y=numeric_columns[1],
                                   title=f"{numeric_columns[1]} vs {numeric_columns[0]}")
                    st.plotly_chart(fig, use_container_width=True)

def display_memory_insights(results):
    """Display memory and learning insights."""
    
    if not results or 'memory_insights' not in results:
        return
    
    st.subheader("ðŸ§  Memory & Learning Insights")
    
    memory_data = results['memory_insights']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Similar Past Queries", len(memory_data.get('similar_queries', [])))
        st.metric("Confidence Score", f"{memory_data.get('confidence', 0):.2f}")
    
    with col2:
        st.metric("Processing Time", f"{memory_data.get('processing_time', 0):.2f}s")
        st.metric("Memory Utilization", memory_data.get('memory_usage', 'N/A'))

def main():
    """Main Streamlit application."""
    
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">ðŸ¤– Advanced SQL Agent System</h1>', unsafe_allow_html=True)
    st.markdown("Transform natural language into SQL queries with intelligent memory and visualization")
    
    # Sidebar configuration
    st.sidebar.header("âš™ï¸ Configuration")
    
    # User preferences
    st.sidebar.subheader("User Preferences")
    visualization_type = st.sidebar.selectbox(
        "Visualization Preference",
        ["auto", "table", "chart", "dashboard"],
        index=0
    )
    
    sql_complexity = st.sidebar.selectbox(
        "SQL Complexity",
        ["simple", "balanced", "advanced"],
        index=1
    )
    
    show_explanation = st.sidebar.checkbox("Show Query Explanation", True)
    
    # Update session state
    st.session_state.user_preferences.update({
        'visualization_type': visualization_type,
        'sql_complexity': sql_complexity,
        'show_query_explanation': show_explanation
    })
    
    # Main interface
    st.subheader("ðŸ’¬ Natural Language Query")
    
    # Query input
    query_input = st.text_area(
        "Enter your question in natural language:",
        placeholder="e.g., Show me total sales by product category for this quarter",
        height=100
    )
    
    # Example queries
    st.subheader("ðŸ“ Example Queries")
    example_queries = [
        "Show me total sales by product category for this quarter",
        "What are the top 10 customers by revenue?",
        "Compare this year's performance to last year",
        "Which regions are underperforming?",
        "Show me monthly trends for the past year"
    ]
    
    selected_example = st.selectbox("Or select an example:", [""] + example_queries)
    if selected_example:
        query_input = selected_example
    
    # Process query button
    if st.button("ðŸš€ Process Query", type="primary", disabled=not query_input):
        if query_input:
            with st.spinner("Processing your query through the agent system..."):
                
                # Show processing status
                status_placeholder = st.empty()
                
                # Process the query
                results = process_natural_language_query(query_input)
                
                if results.get('success', False):
                    # Store in session history
                    st.session_state.query_history.append({
                        'query': query_input,
                        'results': results,
                        'timestamp': datetime.now()
                    })
                    
                    st.session_state.current_results = results
                    
                    # Display success message
                    st.markdown('<div class="success-message">âœ… Query processed successfully!</div>', 
                              unsafe_allow_html=True)
                    
                    # Display results
                    display_query_results(results)
                    
                    # Display memory insights
                    display_memory_insights(results)
                    
                else:
                    # Display error
                    error_msg = results.get('error', 'Unknown error occurred')
                    st.markdown(f'<div class="error-message">âŒ Error: {error_msg}</div>', 
                              unsafe_allow_html=True)
        else:
            st.warning("Please enter a query or select an example.")
    
    # Query history
    if st.session_state.query_history:
        st.subheader("ðŸ“š Query History")
        
        for i, history_item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Query {len(st.session_state.query_history) - i}: {history_item['query'][:50]}..."):
                st.write(f"**Query:** {history_item['query']}")
                st.write(f"**Timestamp:** {history_item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                if 'generated_sql' in history_item['results']:
                    st.code(history_item['results']['generated_sql'], language='sql')
                
                if st.button(f"Rerun Query {len(st.session_state.query_history) - i}", key=f"rerun_{i}"):
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("Built with â¤ï¸ using Streamlit, LangGraph, and Advanced Memory Architecture")

if __name__ == "__main__":
    main()
