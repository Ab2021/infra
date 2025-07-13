"""
Visualization Agent - Chart recommendation and plotting code specialist
Analyzes query results to recommend optimal visualizations and generate plotting code
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from langchain_core.messages import HumanMessage

@dataclass
class ChartRecommendation:
    """Represents a chart recommendation with rationale."""
    chart_type: str
    priority: int
    x_axis: str
    y_axis: str
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    rationale: str = ""
    plotly_code: str = ""
    streamlit_code: str = ""

@dataclass 
class VisualizationResult:
    """Complete visualization analysis results."""
    recommended_charts: List[ChartRecommendation]
    data_insights: Dict[str, Any]
    dashboard_layout: Dict[str, Any]
    interactive_features: List[str]
    confidence_score: float

class VisualizationAgent:
    """
    Sophisticated visualization agent that analyzes query results and intent
    to recommend optimal chart types and generate plotting code.
    """
    
    def __init__(self, memory_system, llm_provider):
        self.memory_system = memory_system
        self.llm_provider = llm_provider
        self.agent_name = "visualizer"
        
        # Chart type mapping based on data characteristics
        self.chart_type_rules = {
            "categorical_vs_numerical": ["bar", "column", "pie"],
            "time_series": ["line", "area", "scatter"],
            "comparison": ["bar", "radar", "heatmap"],
            "distribution": ["histogram", "box", "violin"],
            "correlation": ["scatter", "heatmap", "bubble"],
            "geographical": ["map", "choropleth"],
            "hierarchical": ["treemap", "sunburst", "sankey"]
        }
    
    async def analyze_and_recommend(self, query_results: List[Dict], query_intent: Dict, 
                                  schema_context: Dict, entities: List[Dict]) -> VisualizationResult:
        """
        Analyzes query results and generates intelligent visualization recommendations.
        
        This method serves as the main entry point for the visualization analysis pipeline.
        It combines data pattern analysis, user intent understanding, and schema context
        to generate contextually appropriate chart recommendations with executable code.
        
        Args:
            query_results (List[Dict]): Raw query results from database execution.
                Example: [{"category": "Electronics", "sales": 15000}, 
                         {"category": "Clothing", "sales": 12000}]
            
            query_intent (Dict): Structured query intent from NLU processing.
                Contains fields like:
                - primary_action: "aggregate", "filter", "sort", "compare"
                - data_focus: Description of what data user wants
                - output_preference: "table", "chart", "dashboard"
                - temporal_scope: Time period if specified
                - analysis_requirements: List of analysis types needed
            
            schema_context (Dict): Schema analysis context from SchemaIntelligenceAgent.
                Includes:
                - relevant_tables: List of identified relevant tables
                - column_metadata: Detailed column information
                - data_patterns: Detected cross-table patterns
                - table_relationships: Foreign key relationships
            
            entities (List[Dict]): Extracted entities from NLU processing.
                Each entity contains:
                - type: "table", "column", "value", "date", "metric"
                - value: Extracted entity text
                - confidence: Confidence score (0.0-1.0)
            
        Returns:
            VisualizationResult: Comprehensive visualization analysis containing:
                - recommended_charts: List of ChartRecommendation objects with
                  chart types, axes, rationale, and generated code
                - data_insights: Statistical analysis of the dataset
                - dashboard_layout: Optimal layout configuration for multiple charts
                - interactive_features: Suggested interactive capabilities
                - confidence_score: Overall confidence in recommendations (0.0-1.0)
        
        Raises:
            Exception: If data analysis fails or no valid visualizations can be generated
            
        Example:
            >>> query_results = [{"month": "Jan", "revenue": 10000}, 
            ...                  {"month": "Feb", "revenue": 12000}]
            >>> intent = {"primary_action": "aggregate", "temporal_scope": "monthly"}
            >>> result = await agent.analyze_and_recommend(query_results, intent, {}, [])
            >>> print(result.recommended_charts[0].chart_type)  # "line"
            >>> print(result.recommended_charts[0].rationale)   # "Time series data..."
        
        Note:
            - Requires non-empty query_results for meaningful recommendations
            - Leverages memory system to improve recommendations over time
            - Generates both Plotly and Streamlit code for each recommendation
            - Supports interactive features like filtering and drill-down capabilities
        """
        
        try:
            if not query_results:
                return self._create_empty_result("No data to visualize")
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(query_results)
            
            # Analyze data characteristics
            data_insights = self._analyze_data_characteristics(df)
            
            # Get visualization preferences from memory
            viz_context = await self._get_visualization_context(query_intent, entities)
            
            # Generate chart recommendations
            recommendations = await self._generate_chart_recommendations(
                df, query_intent, data_insights, viz_context
            )
            
            # Design dashboard layout
            dashboard_layout = self._design_dashboard_layout(recommendations, data_insights)
            
            # Identify interactive features
            interactive_features = self._identify_interactive_features(df, query_intent)
            
            # Calculate confidence score
            confidence = self._calculate_visualization_confidence(recommendations, data_insights)
            
            # Update memory with visualization insights
            await self._update_visualization_memory(query_intent, recommendations, data_insights)
            
            return VisualizationResult(
                recommended_charts=recommendations,
                data_insights=data_insights,
                dashboard_layout=dashboard_layout,
                interactive_features=interactive_features,
                confidence_score=confidence
            )
            
        except Exception as e:
            return self._create_empty_result(f"Visualization analysis failed: {str(e)}")
    
    def _analyze_data_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs comprehensive analysis of dataset characteristics for visualization guidance.
        
        This method examines the structure, types, and statistical properties of the dataset
        to inform intelligent visualization choices. It categorizes columns, detects patterns,
        and calculates quality metrics essential for chart recommendation algorithms.
        
        Args:
            df (pd.DataFrame): Input dataset converted from query results.
                Must contain at least one row and column for meaningful analysis.
        
        Returns:
            Dict[str, Any]: Comprehensive data characteristics analysis containing:
                - row_count (int): Total number of data rows
                - column_count (int): Total number of columns
                - numeric_columns (List[str]): Columns with numeric data types
                - categorical_columns (List[str]): Columns with text/categorical data
                - datetime_columns (List[str]): Columns with date/time data
                - data_types (Dict[str, str]): Mapping of column names to data types
                - null_percentages (Dict[str, float]): Null value percentages per column
                - unique_value_counts (Dict[str, int]): Unique value counts per column
                - statistical_summary (Dict[str, Dict]): Statistical measures for numeric columns
                - patterns_detected (List[str]): Identified data patterns for visualization
        
        Example:
            >>> df = pd.DataFrame({
            ...     "category": ["A", "B", "A"], 
            ...     "sales": [100, 200, 150],
            ...     "date": ["2023-01-01", "2023-01-02", "2023-01-03"]
            ... })
            >>> characteristics = agent._analyze_data_characteristics(df)
            >>> print(characteristics["numeric_columns"])  # ["sales"]
            >>> print(characteristics["patterns_detected"])  # ["categorical_vs_numerical"]
        
        Note:
            - Automatically detects time series patterns from datetime columns
            - Identifies categorical vs numerical relationships for chart recommendations
            - Calculates statistical summaries only for numeric columns
            - Handles missing values gracefully in all calculations
        """
        
        insights = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "data_types": {},
            "null_percentages": {},
            "unique_value_counts": {},
            "statistical_summary": {},
            "patterns_detected": []
        }
        
        for column in df.columns:
            col_data = df[column]
            
            # Determine data type
            if pd.api.types.is_numeric_dtype(col_data):
                insights["numeric_columns"].append(column)
                insights["statistical_summary"][column] = {
                    "mean": float(col_data.mean()) if not col_data.empty else 0,
                    "std": float(col_data.std()) if not col_data.empty else 0,
                    "min": float(col_data.min()) if not col_data.empty else 0,
                    "max": float(col_data.max()) if not col_data.empty else 0
                }
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                insights["datetime_columns"].append(column)
            else:
                insights["categorical_columns"].append(column)
            
            # Basic statistics
            insights["data_types"][column] = str(col_data.dtype)
            insights["null_percentages"][column] = float(col_data.isnull().mean() * 100)
            insights["unique_value_counts"][column] = int(col_data.nunique())
        
        # Detect patterns
        insights["patterns_detected"] = self._detect_data_patterns(df, insights)
        
        return insights
    
    def _detect_data_patterns(self, df: pd.DataFrame, insights: Dict) -> List[str]:
        """Detects common data patterns for visualization guidance."""
        
        patterns = []
        
        # Time series pattern
        if insights["datetime_columns"]:
            patterns.append("time_series")
        
        # Categorical vs numerical pattern
        if insights["categorical_columns"] and insights["numeric_columns"]:
            patterns.append("categorical_vs_numerical")
        
        # Correlation pattern
        if len(insights["numeric_columns"]) >= 2:
            patterns.append("correlation")
        
        # Distribution pattern
        if len(insights["numeric_columns"]) >= 1:
            patterns.append("distribution")
        
        # Hierarchical pattern (detected by column names)
        hierarchical_keywords = ["category", "subcategory", "region", "country", "state", "city"]
        if any(any(keyword in col.lower() for keyword in hierarchical_keywords) 
               for col in insights["categorical_columns"]):
            patterns.append("hierarchical")
        
        # High cardinality categorical
        high_cardinality_cats = [col for col in insights["categorical_columns"] 
                               if insights["unique_value_counts"][col] > 10]
        if high_cardinality_cats:
            patterns.append("high_cardinality_categorical")
        
        return patterns
    
    async def _generate_chart_recommendations(self, df: pd.DataFrame, query_intent: Dict, 
                                            data_insights: Dict, viz_context: Dict) -> List[ChartRecommendation]:
        """Generates chart recommendations based on data analysis."""
        
        recommendations = []
        patterns = data_insights["patterns_detected"]
        
        # Primary recommendation based on strongest pattern
        if "time_series" in patterns:
            recommendations.extend(self._recommend_time_series_charts(df, data_insights))
        
        if "categorical_vs_numerical" in patterns:
            recommendations.extend(self._recommend_categorical_numerical_charts(df, data_insights))
        
        if "correlation" in patterns:
            recommendations.extend(self._recommend_correlation_charts(df, data_insights))
        
        if "distribution" in patterns:
            recommendations.extend(self._recommend_distribution_charts(df, data_insights))
        
        # Sort by priority and limit to top 4 recommendations
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        # Generate plotting code for each recommendation
        for rec in recommendations[:4]:
            rec.plotly_code = self._generate_plotly_code(rec, df)
            rec.streamlit_code = self._generate_streamlit_code(rec, df)
        
        return recommendations[:4]
    
    def _recommend_time_series_charts(self, df: pd.DataFrame, insights: Dict) -> List[ChartRecommendation]:
        """Recommends charts for time series data."""
        
        recommendations = []
        datetime_col = insights["datetime_columns"][0] if insights["datetime_columns"] else None
        numeric_cols = insights["numeric_columns"]
        
        if datetime_col and numeric_cols:
            # Line chart for time series
            recommendations.append(ChartRecommendation(
                chart_type="line",
                priority=90,
                x_axis=datetime_col,
                y_axis=numeric_cols[0],
                rationale="Line chart ideal for showing trends over time"
            ))
            
            # Area chart for cumulative data
            if len(numeric_cols) > 1:
                recommendations.append(ChartRecommendation(
                    chart_type="area",
                    priority=70,
                    x_axis=datetime_col,
                    y_axis=numeric_cols[1],
                    rationale="Area chart effective for showing cumulative values over time"
                ))
        
        return recommendations
    
    def _recommend_categorical_numerical_charts(self, df: pd.DataFrame, insights: Dict) -> List[ChartRecommendation]:
        """Recommends charts for categorical vs numerical data."""
        
        recommendations = []
        cat_cols = insights["categorical_columns"]
        num_cols = insights["numeric_columns"]
        
        if cat_cols and num_cols:
            cat_col = cat_cols[0]
            num_col = num_cols[0]
            unique_categories = insights["unique_value_counts"][cat_col]
            
            # Bar chart - most versatile
            recommendations.append(ChartRecommendation(
                chart_type="bar",
                priority=85,
                x_axis=cat_col,
                y_axis=num_col,
                rationale=f"Bar chart ideal for comparing {num_col} across {unique_categories} categories"
            ))
            
            # Pie chart for composition (if reasonable number of categories)
            if unique_categories <= 8:
                recommendations.append(ChartRecommendation(
                    chart_type="pie",
                    priority=75,
                    x_axis=cat_col,
                    y_axis=num_col,
                    rationale=f"Pie chart shows composition of {num_col} across categories"
                ))
        
        return recommendations
    
    def _recommend_correlation_charts(self, df: pd.DataFrame, insights: Dict) -> List[ChartRecommendation]:
        """Recommends charts for correlation analysis."""
        
        recommendations = []
        num_cols = insights["numeric_columns"]
        
        if len(num_cols) >= 2:
            # Scatter plot for correlation
            recommendations.append(ChartRecommendation(
                chart_type="scatter",
                priority=80,
                x_axis=num_cols[0],
                y_axis=num_cols[1],
                color_column=insights["categorical_columns"][0] if insights["categorical_columns"] else None,
                rationale=f"Scatter plot reveals relationship between {num_cols[0]} and {num_cols[1]}"
            ))
        
        return recommendations
    
    def _recommend_distribution_charts(self, df: pd.DataFrame, insights: Dict) -> List[ChartRecommendation]:
        """Recommends charts for distribution analysis."""
        
        recommendations = []
        num_cols = insights["numeric_columns"]
        
        if num_cols:
            # Histogram for distribution
            recommendations.append(ChartRecommendation(
                chart_type="histogram",
                priority=65,
                x_axis=num_cols[0],
                y_axis="count",
                rationale=f"Histogram shows distribution of {num_cols[0]} values"
            ))
        
        return recommendations
    
    def _generate_plotly_code(self, recommendation: ChartRecommendation, df: pd.DataFrame) -> str:
        """Generates Plotly code for the recommendation."""
        
        chart_type = recommendation.chart_type
        x_col = recommendation.x_axis
        y_col = recommendation.y_axis
        color_col = recommendation.color_column
        
        if chart_type == "line":
            code = f"""
import plotly.express as px

fig = px.line(df, x='{x_col}', y='{y_col}', 
              title='{y_col} over {x_col}')
fig.show()
"""
        elif chart_type == "bar":
            color_param = f", color='{color_col}'" if color_col else ""
            code = f"""
import plotly.express as px

fig = px.bar(df, x='{x_col}', y='{y_col}'{color_param}, 
             title='{y_col} by {x_col}')
fig.show()
"""
        elif chart_type == "scatter":
            color_param = f", color='{color_col}'" if color_col else ""
            code = f"""
import plotly.express as px

fig = px.scatter(df, x='{x_col}', y='{y_col}'{color_param}, 
                 title='{y_col} vs {x_col}')
fig.show()
"""
        elif chart_type == "pie":
            code = f"""
import plotly.express as px

fig = px.pie(df, names='{x_col}', values='{y_col}', 
             title='Distribution of {y_col}')
fig.show()
"""
        elif chart_type == "histogram":
            code = f"""
import plotly.express as px

fig = px.histogram(df, x='{x_col}', 
                   title='Distribution of {x_col}')
fig.show()
"""
        elif chart_type == "area":
            code = f"""
import plotly.express as px

fig = px.area(df, x='{x_col}', y='{y_col}', 
              title='{y_col} over {x_col}')
fig.show()
"""
        else:
            code = f"""
# Chart type '{chart_type}' not yet implemented
import plotly.express as px

fig = px.bar(df, x='{x_col}', y='{y_col}')
fig.show()
"""
        
        return code.strip()
    
    def _generate_streamlit_code(self, recommendation: ChartRecommendation, df: pd.DataFrame) -> str:
        """Generates Streamlit code for the recommendation."""
        
        chart_type = recommendation.chart_type
        x_col = recommendation.x_axis
        y_col = recommendation.y_axis
        color_col = recommendation.color_column
        
        if chart_type == "line":
            code = f"""
import streamlit as st
import plotly.express as px

fig = px.line(df, x='{x_col}', y='{y_col}', 
              title='{y_col} over {x_col}')
st.plotly_chart(fig, use_container_width=True)
"""
        elif chart_type == "bar":
            color_param = f", color='{color_col}'" if color_col else ""
            code = f"""
import streamlit as st
import plotly.express as px

fig = px.bar(df, x='{x_col}', y='{y_col}'{color_param}, 
             title='{y_col} by {x_col}')
st.plotly_chart(fig, use_container_width=True)
"""
        elif chart_type == "scatter":
            color_param = f", color='{color_col}'" if color_col else ""
            code = f"""
import streamlit as st
import plotly.express as px

fig = px.scatter(df, x='{x_col}', y='{y_col}'{color_param}, 
                 title='{y_col} vs {x_col}')
st.plotly_chart(fig, use_container_width=True)
"""
        elif chart_type == "pie":
            code = f"""
import streamlit as st
import plotly.express as px

fig = px.pie(df, names='{x_col}', values='{y_col}', 
             title='Distribution of {y_col}')
st.plotly_chart(fig, use_container_width=True)
"""
        elif chart_type == "histogram":
            code = f"""
import streamlit as st
import plotly.express as px

fig = px.histogram(df, x='{x_col}', 
                   title='Distribution of {x_col}')
st.plotly_chart(fig, use_container_width=True)
"""
        else:
            code = f"""
# Chart type '{chart_type}' not yet implemented
import streamlit as st
import plotly.express as px

fig = px.bar(df, x='{x_col}', y='{y_col}')
st.plotly_chart(fig, use_container_width=True)
"""
        
        return code.strip()
    
    def _design_dashboard_layout(self, recommendations: List[ChartRecommendation], 
                               insights: Dict) -> Dict[str, Any]:
        """Designs optimal dashboard layout for visualizations."""
        
        layout = {
            "layout_type": "grid",
            "columns": 2,
            "sections": [],
            "responsive": True
        }
        
        # Main chart (highest priority)
        if recommendations:
            layout["sections"].append({
                "type": "main_chart",
                "chart": recommendations[0],
                "size": "large",
                "position": {"row": 1, "col": 1, "span": 2}
            })
        
        # Secondary charts
        for i, rec in enumerate(recommendations[1:3], 1):
            layout["sections"].append({
                "type": "secondary_chart", 
                "chart": rec,
                "size": "medium",
                "position": {"row": 2, "col": i, "span": 1}
            })
        
        # Data summary section
        layout["sections"].append({
            "type": "data_summary",
            "content": {
                "total_rows": insights["row_count"],
                "total_columns": insights["column_count"],
                "patterns": insights["patterns_detected"]
            },
            "size": "small",
            "position": {"row": 3, "col": 1, "span": 2}
        })
        
        return layout
    
    def _identify_interactive_features(self, df: pd.DataFrame, query_intent: Dict) -> List[str]:
        """Identifies interactive features to add to visualizations."""
        
        features = ["zoom", "pan", "hover_data"]
        
        # Filter capabilities
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            features.append("category_filter")
        
        # Date range selection for time series
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            features.append("date_range_selector")
        
        # Cross-filtering for multiple charts
        if len(df.columns) > 3:
            features.append("cross_filter")
        
        return features
    
    def _calculate_visualization_confidence(self, recommendations: List[ChartRecommendation], 
                                          insights: Dict) -> float:
        """Calculates confidence score for visualization recommendations."""
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on clear patterns
        if insights["patterns_detected"]:
            confidence += 0.2 * len(insights["patterns_detected"])
        
        # Boost confidence based on data quality
        if insights["row_count"] > 10:
            confidence += 0.1
        
        # Reduce confidence for sparse data
        if insights["row_count"] < 5:
            confidence -= 0.2
        
        # Boost confidence if we have good recommendations
        if recommendations and len(recommendations) >= 2:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _get_visualization_context(self, query_intent: Dict, entities: List[Dict]) -> Dict:
        """Gets visualization context from memory and preferences."""
        
        return await self.memory_system.get_contextual_memories(
            query=f"visualization for {query_intent.get('data_focus', '')}",
            context_type="visualization"
        )
    
    async def _update_visualization_memory(self, query_intent: Dict, 
                                         recommendations: List[ChartRecommendation], 
                                         insights: Dict):
        """Updates memory with visualization insights."""
        
        await self.memory_system.working_memory.update_context(
            agent_name=self.agent_name,
            update_data={
                "visualization_recommendations": [rec.__dict__ for rec in recommendations],
                "data_insights": insights,
                "query_intent": query_intent,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _create_empty_result(self, error_message: str) -> VisualizationResult:
        """Creates empty result for error cases."""
        
        return VisualizationResult(
            recommended_charts=[],
            data_insights={"error": error_message},
            dashboard_layout={"layout_type": "empty"},
            interactive_features=[],
            confidence_score=0.0
        )