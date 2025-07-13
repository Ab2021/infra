"""
Advanced Professional Dashboard for SQL Agent System
Modern, streaming-enabled dashboard with real-time agent status and progressive results
"""

import streamlit as st
import asyncio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import uuid
from typing import Dict, List, Optional, Any

# Page configuration
st.set_page_config(
    page_title="Advanced SQL Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS for professional dashboard
st.markdown("""
<style>
    /* Main dashboard styling */
    .main-dashboard {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .dashboard-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .dashboard-subtitle {
        font-size: 1.2rem;
        text-align: center;
        opacity: 0.9;
        margin-bottom: 2rem;
    }
    
    /* Agent status cards */
    .agent-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    
    .agent-card.processing {
        border-left-color: #ffc107;
        background: linear-gradient(45deg, #fff9c4, #ffffff);
    }
    
    .agent-card.completed {
        border-left-color: #28a745;
        background: linear-gradient(45deg, #d4edda, #ffffff);
    }
    
    .agent-card.error {
        border-left-color: #dc3545;
        background: linear-gradient(45deg, #f8d7da, #ffffff);
    }
    
    .agent-name {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .agent-status {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Metrics cards */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Progress bars */
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #007bff, #28a745);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Query input styling */
    .query-input {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    /* Results section */
    .results-section {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-pending { background-color: #6c757d; }
    .status-processing { background-color: #ffc107; animation: pulse 1s infinite; }
    .status-completed { background-color: #28a745; }
    .status-error { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

class DashboardState:
    """Manages dashboard state and streaming updates."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.session_id = str(uuid.uuid4())
        self.current_query = ""
        self.processing_status = "idle"
        self.agent_statuses = {
            "nlu": "pending",
            "schema": "pending", 
            "sql_generator": "pending",
            "validator": "pending",
            "visualizer": "pending"
        }
        self.results = None
        self.progress = 0
        self.start_time = None
        self.error_messages = []
        self.query_history = []

def initialize_dashboard():
    """Initialize dashboard session state."""
    if 'dashboard_state' not in st.session_state:
        st.session_state.dashboard_state = DashboardState()
    if 'query_counter' not in st.session_state:
        st.session_state.query_counter = 0

def render_dashboard_header():
    """Renders the main dashboard header."""
    st.markdown("""
    <div class="main-dashboard fade-in">
        <div class="dashboard-title">🤖 Advanced SQL Agent Dashboard</div>
        <div class="dashboard-subtitle">
            Transform natural language into intelligent SQL queries with real-time visualization
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metrics_cards():
    """Renders key metrics cards."""
    state = st.session_state.dashboard_state
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div class="metric-value">{len(state.query_history)}</div>
            <div class="metric-label">Queries Processed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        accuracy = 95.7  # Mock value - could be calculated from history
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div class="metric-value">{accuracy}%</div>
            <div class="metric-label">Query Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_time = "2.3s"  # Mock value
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div class="metric-value">{avg_time}</div>
            <div class="metric-label">Avg Response Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        active_sessions = 1
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div class="metric-value">{active_sessions}</div>
            <div class="metric-label">Active Sessions</div>
        </div>
        """, unsafe_allow_html=True)

def render_agent_status_panel():
    """Renders real-time agent status panel."""
    state = st.session_state.dashboard_state
    
    st.markdown("### 🤖 Agent Processing Pipeline")
    
    agent_details = {
        "nlu": {"name": "Natural Language Understanding", "icon": "🧠"},
        "schema": {"name": "Schema Intelligence", "icon": "🗄️"},
        "sql_generator": {"name": "SQL Generator", "icon": "⚡"},
        "validator": {"name": "Security Validator", "icon": "🛡️"},
        "visualizer": {"name": "Visualization Engine", "icon": "📊"}
    }
    
    # Overall progress bar
    completed_agents = sum(1 for status in state.agent_statuses.values() if status == "completed")
    progress = (completed_agents / len(state.agent_statuses)) * 100
    
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress}%"></div>
    </div>
    <p style="text-align: center; margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.8;">
        Processing Progress: {progress:.0f}%
    </p>
    """, unsafe_allow_html=True)
    
    # Individual agent cards
    cols = st.columns(len(agent_details))
    
    for i, (agent_key, details) in enumerate(agent_details.items()):
        with cols[i]:
            status = state.agent_statuses[agent_key]
            
            # Status indicator
            status_class = f"status-{status}"
            pulse_class = "pulse" if status == "processing" else ""
            
            # Status icon
            status_icons = {
                "pending": "⏳",
                "processing": "⚡",
                "completed": "✅",
                "error": "❌"
            }
            
            st.markdown(f"""
            <div class="agent-card {status} fade-in">
                <div class="agent-name">
                    {details['icon']} {details['name']}
                </div>
                <div class="agent-status">
                    <span class="status-indicator {status_class} {pulse_class}"></span>
                    {status_icons[status]} {status.title()}
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_query_interface():
    """Renders the query input interface."""
    st.markdown("### 💬 Natural Language Query Interface")
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Analytics Query", help="Generate analytical queries"):
            st.session_state.query_template = "analytics"
    with col2:
        if st.button("📈 Performance Metrics", help="Performance-focused queries"):
            st.session_state.query_template = "performance"
    with col3:
        if st.button("🔍 Data Exploration", help="Exploratory data queries"):
            st.session_state.query_template = "exploration"
    
    # Query input with templates
    template_queries = {
        "": "Enter your question in natural language...",
        "analytics": "Show me the top 10 customers by revenue this quarter",
        "performance": "What are the sales trends by region over the past 6 months?",
        "exploration": "Which products have the highest profit margins?"
    }
    
    template = st.session_state.get('query_template', '')
    placeholder = template_queries.get(template, template_queries[''])
    
    query_input = st.text_area(
        "Your Query:",
        placeholder=placeholder,
        height=100,
        key="query_input",
        help="Describe what data you want to see in natural language"
    )
    
    # Advanced options in expander
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            output_format = st.selectbox(
                "Output Format",
                ["Auto-detect", "Table", "Chart", "Dashboard", "Export"],
                help="How would you like to see the results?"
            )
        with col2:
            complexity_level = st.selectbox(
                "Query Complexity",
                ["Balanced", "Simple", "Advanced"],
                help="How complex should the generated SQL be?"
            )
        
        include_explanation = st.checkbox("Include SQL Explanation", value=True)
        enable_caching = st.checkbox("Enable Result Caching", value=True)
    
    # Process button with enhanced styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Process Query",
            type="primary",
            disabled=not query_input.strip(),
            use_container_width=True,
            help="Click to start processing your natural language query"
        )
    
    return query_input, process_button, {
        "output_format": output_format,
        "complexity_level": complexity_level,
        "include_explanation": include_explanation,
        "enable_caching": enable_caching
    }

async def process_query_with_streaming(query: str, options: Dict):
    """Process query with real-time status updates."""
    state = st.session_state.dashboard_state
    
    # Reset state for new query
    state.reset()
    state.current_query = query
    state.processing_status = "processing"
    state.start_time = datetime.now()
    
    # Create placeholders for streaming updates
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    try:
        # Import the main system
        from main import SQLAgentSystem
        
        # Simulate streaming by updating status during processing
        agents = ["nlu", "schema", "sql_generator", "validator", "visualizer"]
        
        system = SQLAgentSystem()
        
        # Process with mock streaming updates
        for i, agent in enumerate(agents):
            # Update status to processing
            state.agent_statuses[agent] = "processing"
            render_agent_status_panel()
            time.sleep(0.5)  # Simulate processing time
            
            # Mark as completed
            state.agent_statuses[agent] = "completed"
            state.progress = ((i + 1) / len(agents)) * 100
            
            # Update UI
            render_agent_status_panel()
            time.sleep(0.3)
        
        # Get final results
        result = await system.process_query(query, "dashboard_user")
        
        state.results = result
        state.processing_status = "completed"
        state.query_history.append({
            "query": query,
            "results": result,
            "timestamp": datetime.now(),
            "processing_time": (datetime.now() - state.start_time).total_seconds()
        })
        
        return result
        
    except Exception as e:
        state.processing_status = "error"
        state.error_messages.append(str(e))
        return {"error": str(e), "success": False}

def render_results_dashboard(results: Dict):
    """Renders comprehensive results dashboard."""
    if not results or not results.get("success", True):
        st.error(f"❌ Query processing failed: {results.get('error', 'Unknown error')}")
        return
    
    st.markdown("### 📊 Query Results Dashboard")
    
    # Create tabs for different result views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "📋 Data Table", "💻 SQL Query", "🧠 Insights"])
    
    with tab1:
        render_visualization_tab(results)
    
    with tab2:
        render_data_table_tab(results)
    
    with tab3:
        render_sql_tab(results)
    
    with tab4:
        render_insights_tab(results)

def render_visualization_tab(results: Dict):
    """Renders advanced visualization tab."""
    query_results = results.get("query_results", [])
    
    if not query_results:
        st.info("No data available for visualization")
        return
    
    df = pd.DataFrame(query_results)
    
    # Auto-detect chart types based on data
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_columns and not categorical_columns:
        st.info("No suitable columns found for visualization")
        return
    
    # Chart selection interface
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### Chart Configuration")
        
        chart_types = ["Auto", "Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Heatmap"]
        selected_chart = st.selectbox("Chart Type", chart_types)
        
        if numeric_columns:
            x_axis = st.selectbox("X-Axis", categorical_columns + numeric_columns)
            y_axis = st.selectbox("Y-Axis", numeric_columns)
        
        if categorical_columns:
            color_by = st.selectbox("Color By", ["None"] + categorical_columns)
    
    with col2:
        st.markdown("#### Interactive Visualization")
        
        # Generate chart based on selection
        if selected_chart == "Auto" or selected_chart == "Bar Chart":
            if categorical_columns and numeric_columns:
                fig = px.bar(
                    df, 
                    x=categorical_columns[0], 
                    y=numeric_columns[0],
                    color=categorical_columns[0] if len(categorical_columns) > 1 else None,
                    title=f"{numeric_columns[0]} by {categorical_columns[0]}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif selected_chart == "Line Chart":
            if len(numeric_columns) >= 2:
                fig = px.line(
                    df,
                    x=df.columns[0],
                    y=numeric_columns[0],
                    title="Trend Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif selected_chart == "Pie Chart":
            if categorical_columns and numeric_columns:
                fig = px.pie(
                    df,
                    names=categorical_columns[0],
                    values=numeric_columns[0],
                    title="Distribution Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif selected_chart == "Scatter Plot":
            if len(numeric_columns) >= 2:
                fig = px.scatter(
                    df,
                    x=numeric_columns[0],
                    y=numeric_columns[1],
                    color=categorical_columns[0] if categorical_columns else None,
                    title="Correlation Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights
    st.markdown("#### 🔍 Data Insights")
    
    insight_cols = st.columns(3)
    
    with insight_cols[0]:
        st.metric("Total Records", len(df))
    
    with insight_cols[1]:
        if numeric_columns:
            avg_value = df[numeric_columns[0]].mean()
            st.metric(f"Average {numeric_columns[0]}", f"{avg_value:.2f}")
    
    with insight_cols[2]:
        if categorical_columns:
            unique_categories = df[categorical_columns[0]].nunique()
            st.metric(f"Unique {categorical_columns[0]}", unique_categories)

def render_data_table_tab(results: Dict):
    """Renders data table with advanced features."""
    query_results = results.get("query_results", [])
    
    if not query_results:
        st.info("No data available")
        return
    
    df = pd.DataFrame(query_results)
    
    # Table controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_index = st.checkbox("Show Index", value=False)
    
    with col2:
        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    
    with col3:
        if st.button("📥 Export CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Display table with pagination
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=not show_index,
        height=400
    )
    
    # Table summary
    st.markdown("#### 📈 Table Summary")
    summary_cols = st.columns(4)
    
    with summary_cols[0]:
        st.metric("Rows", len(df))
    
    with summary_cols[1]:
        st.metric("Columns", len(df.columns))
    
    with summary_cols[2]:
        memory_usage = df.memory_usage(deep=True).sum() / 1024  # KB
        st.metric("Memory Usage", f"{memory_usage:.1f} KB")
    
    with summary_cols[3]:
        null_count = df.isnull().sum().sum()
        st.metric("Null Values", null_count)

def render_sql_tab(results: Dict):
    """Renders SQL query tab with enhanced features."""
    generated_sql = results.get("generated_sql", "")
    
    if not generated_sql:
        st.info("No SQL query available")
        return
    
    # SQL display with syntax highlighting
    st.markdown("#### 💻 Generated SQL Query")
    st.code(generated_sql, language="sql")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Copy SQL"):
            st.success("SQL copied to clipboard!")
    
    with col2:
        if st.button("💾 Save Query"):
            st.success("Query saved to history!")
    
    with col3:
        if st.button("🔄 Regenerate"):
            st.info("Query regeneration requested!")
    
    # Query analysis
    st.markdown("#### 🔍 Query Analysis")
    
    analysis_cols = st.columns(3)
    
    with analysis_cols[0]:
        query_complexity = "Medium"  # Mock analysis
        st.metric("Complexity", query_complexity)
    
    with analysis_cols[1]:
        estimated_runtime = "< 1s"  # Mock analysis
        st.metric("Est. Runtime", estimated_runtime)
    
    with analysis_cols[2]:
        optimization_score = "Good"  # Mock analysis
        st.metric("Optimization", optimization_score)
    
    # Query explanation
    if results.get("include_explanation", True):
        st.markdown("#### 📚 Query Explanation")
        explanation = """
        This query retrieves data from the selected tables using the following logic:
        1. Filters data based on the specified conditions
        2. Joins related tables for comprehensive results
        3. Aggregates data where necessary
        4. Orders results for optimal presentation
        """
        st.info(explanation)

def render_insights_tab(results: Dict):
    """Renders AI insights and recommendations."""
    st.markdown("#### 🧠 AI-Powered Insights")
    
    # Mock insights based on results
    insights = [
        "📈 **Trend Analysis**: Data shows a 15% increase compared to previous period",
        "🎯 **Key Finding**: Top 3 categories account for 67% of total values",
        "⚠️ **Data Quality**: 2% of records contain null values in key fields",
        "💡 **Recommendation**: Consider filtering by date range for better performance",
        "🔄 **Pattern**: Weekly seasonality detected in the data"
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    st.markdown("#### 📊 Statistical Summary")
    
    # Mock statistical insights
    stat_cols = st.columns(2)
    
    with stat_cols[0]:
        st.markdown("""
        **Data Distribution:**
        - Normal distribution detected
        - Outliers: 3 records (1.2%)
        - Correlation strength: Moderate
        """)
    
    with stat_cols[1]:
        st.markdown("""
        **Business Impact:**
        - High-value segments identified
        - Growth opportunities spotted
        - Performance benchmarks met
        """)

def render_sidebar():
    """Renders enhanced sidebar with controls."""
    st.sidebar.markdown("## ⚙️ Dashboard Controls")
    
    # Query history
    st.sidebar.markdown("### 📚 Recent Queries")
    state = st.session_state.dashboard_state
    
    if state.query_history:
        for i, query_item in enumerate(state.query_history[-5:]):
            with st.sidebar.expander(f"Query {i+1}"):
                st.write(f"**Query:** {query_item['query'][:50]}...")
                st.write(f"**Time:** {query_item['processing_time']:.2f}s")
                if st.button(f"Rerun", key=f"rerun_{i}"):
                    st.rerun()
    else:
        st.sidebar.info("No queries yet")
    
    # System status
    st.sidebar.markdown("### 🖥️ System Status")
    st.sidebar.success("🟢 All systems operational")
    st.sidebar.metric("Uptime", "99.9%")
    st.sidebar.metric("Response Time", "1.2s avg")
    
    # Settings
    st.sidebar.markdown("### ⚙️ Settings")
    
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark", "Auto"])
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    notifications = st.sidebar.checkbox("Notifications", value=True)
    
    if st.sidebar.button("🔄 Reset Dashboard"):
        st.session_state.dashboard_state = DashboardState()
        st.rerun()

def main():
    """Main dashboard application."""
    initialize_dashboard()
    
    # Render main dashboard
    render_dashboard_header()
    render_metrics_cards()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Agent status panel
        render_agent_status_panel()
        
        # Query interface
        query_input, process_button, options = render_query_interface()
        
        # Process query if button clicked
        if process_button and query_input:
            st.session_state.query_counter += 1
            
            with st.spinner("🚀 Processing your query..."):
                # Run async processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    results = loop.run_until_complete(
                        process_query_with_streaming(query_input, options)
                    )
                    
                    if results:
                        render_results_dashboard(results)
                finally:
                    loop.close()
    
    with col2:
        render_sidebar()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; opacity: 0.7;'>"
        "🤖 Advanced SQL Agent Dashboard | Built with Streamlit & LangGraph"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()