"""
Minimal Memory System - Optimized for no external dependencies
Provides memory functionality with minimal embedding usage and in-memory SQLite support
"""

import asyncio
import aiosqlite
import json
import logging
import pickle
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import os

class MinimalMemorySystem:
    """
    Lightweight memory system that prioritizes text-based similarity over vector embeddings.
    Supports both in-memory and file-based SQLite storage.
    """
    
    def __init__(self, 
                 session_db_path: str = ":memory:",
                 knowledge_db_path: str = "data/knowledge_memory.db",
                 enable_vector_search: bool = False,
                 similarity_threshold: float = 0.3):
        """
        Initialize minimal memory system.
        
        Args:
            session_db_path: SQLite path for session memory (":memory:" for in-memory)
            knowledge_db_path: SQLite path for knowledge memory
            enable_vector_search: Whether to enable vector similarity (requires embeddings)
            similarity_threshold: Threshold for text-based similarity matching
        """
        self.session_db_path = session_db_path
        self.knowledge_db_path = knowledge_db_path
        self.enable_vector_search = enable_vector_search
        self.similarity_threshold = similarity_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize databases
        self._setup_databases()
        
    def _setup_databases(self):
        """Setup database connections and tables."""
        try:
            # Create knowledge directory if using file path
            if self.knowledge_db_path != ":memory:":
                Path(os.path.dirname(self.knowledge_db_path)).mkdir(parents=True, exist_ok=True)
            
            # Initialize databases synchronously
            asyncio.run(self._create_tables())
            
        except Exception as e:
            self.logger.error(f"Failed to setup databases: {e}")
            raise
    
    async def _create_tables(self):
        """Create necessary tables in both databases."""
        
        # Session memory tables
        async with aiosqlite.connect(self.session_db_path) as session_db:
            await session_db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            await session_db.execute("""
                CREATE TABLE IF NOT EXISTS session_queries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    intent TEXT,
                    entities TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            await session_db.commit()
        
        # Knowledge memory tables
        async with aiosqlite.connect(self.knowledge_db_path) as knowledge_db:
            await knowledge_db.execute("""
                CREATE TABLE IF NOT EXISTS successful_queries (
                    id TEXT PRIMARY KEY,
                    original_query TEXT NOT NULL,
                    generated_sql TEXT NOT NULL,
                    execution_time REAL,
                    result_count INTEGER,
                    tables_used TEXT,
                    success_score REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            await knowledge_db.execute("""
                CREATE TABLE IF NOT EXISTS query_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    pattern_description TEXT,
                    example_queries TEXT,
                    sql_template TEXT,
                    usage_count INTEGER DEFAULT 1,
                    last_used TEXT NOT NULL,
                    effectiveness_score REAL DEFAULT 1.0
                )
            """)
            
            await knowledge_db.execute("""
                CREATE TABLE IF NOT EXISTS schema_insights (
                    id TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    column_name TEXT,
                    insight_type TEXT NOT NULL,
                    insight_data TEXT,
                    confidence_score REAL DEFAULT 1.0,
                    last_updated TEXT NOT NULL
                )
            """)
            
            await knowledge_db.commit()
    
    async def initialize_session(self, user_id: str, session_id: str = None) -> str:
        """Initialize a new user session."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.session_db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO sessions (id, user_id, created_at, last_activity, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, user_id, timestamp, timestamp, "{}"))
            await db.commit()
        
        return session_id
    
    async def store_session_query(self, session_id: str, query: str, 
                                intent: Dict = None, entities: List = None):
        """Store query in session memory."""
        query_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.session_db_path) as db:
            await db.execute("""
                INSERT INTO session_queries (id, session_id, query, intent, entities, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (query_id, session_id, query, 
                  json.dumps(intent) if intent else None,
                  json.dumps(entities) if entities else None,
                  timestamp))
            await db.commit()
        
        return query_id
    
    async def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent queries from session."""
        async with aiosqlite.connect(self.session_db_path) as db:
            cursor = await db.execute("""
                SELECT query, intent, entities, timestamp 
                FROM session_queries 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (session_id, limit))
            
            rows = await cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    "query": row[0],
                    "intent": json.loads(row[1]) if row[1] else None,
                    "entities": json.loads(row[2]) if row[2] else None,
                    "timestamp": row[3]
                })
            
            return history
    
    async def store_successful_query(self, query: str, sql: str, execution_time: float,
                                   result_count: int, tables_used: List[str],
                                   metadata: Dict = None):
        """Store successful query for future learning."""
        query_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            await db.execute("""
                INSERT INTO successful_queries 
                (id, original_query, generated_sql, execution_time, result_count, 
                 tables_used, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (query_id, query, sql, execution_time, result_count,
                  json.dumps(tables_used), timestamp, 
                  json.dumps(metadata) if metadata else None))
            await db.commit()
        
        return query_id
    
    async def find_similar_queries(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find similar queries using simple text matching."""
        query_words = set(query.lower().split())
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            cursor = await db.execute("""
                SELECT original_query, generated_sql, execution_time, 
                       result_count, tables_used, success_score, created_at
                FROM successful_queries 
                ORDER BY created_at DESC 
                LIMIT 100
            """)
            
            rows = await cursor.fetchall()
            
            similar_queries = []
            for row in rows:
                stored_query = row[0].lower()
                stored_words = set(stored_query.split())
                
                # Calculate Jaccard similarity
                intersection = len(query_words & stored_words)
                union = len(query_words | stored_words)
                similarity = intersection / union if union > 0 else 0
                
                if similarity >= self.similarity_threshold:
                    similar_queries.append({
                        "original_query": row[0],
                        "generated_sql": row[1],
                        "execution_time": row[2],
                        "result_count": row[3],
                        "tables_used": json.loads(row[4]) if row[4] else [],
                        "success_score": row[5],
                        "similarity_score": similarity,
                        "created_at": row[6]
                    })
            
            # Sort by similarity and return top k
            similar_queries.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similar_queries[:top_k]
    
    async def store_query_pattern(self, pattern_type: str, pattern_description: str,
                                example_queries: List[str], sql_template: str):
        """Store a reusable query pattern."""
        pattern_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            await db.execute("""
                INSERT INTO query_patterns 
                (id, pattern_type, pattern_description, example_queries, 
                 sql_template, last_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (pattern_id, pattern_type, pattern_description,
                  json.dumps(example_queries), sql_template, timestamp))
            await db.commit()
        
        return pattern_id
    
    async def get_query_patterns(self, pattern_type: str = None) -> List[Dict]:
        """Get stored query patterns."""
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            if pattern_type:
                cursor = await db.execute("""
                    SELECT pattern_type, pattern_description, example_queries, 
                           sql_template, usage_count, effectiveness_score
                    FROM query_patterns 
                    WHERE pattern_type = ?
                    ORDER BY effectiveness_score DESC
                """, (pattern_type,))
            else:
                cursor = await db.execute("""
                    SELECT pattern_type, pattern_description, example_queries, 
                           sql_template, usage_count, effectiveness_score
                    FROM query_patterns 
                    ORDER BY effectiveness_score DESC
                """)
            
            rows = await cursor.fetchall()
            
            patterns = []
            for row in rows:
                patterns.append({
                    "pattern_type": row[0],
                    "pattern_description": row[1],
                    "example_queries": json.loads(row[2]) if row[2] else [],
                    "sql_template": row[3],
                    "usage_count": row[4],
                    "effectiveness_score": row[5]
                })
            
            return patterns
    
    async def store_schema_insight(self, table_name: str, insight_type: str,
                                 insight_data: Dict, column_name: str = None,
                                 confidence_score: float = 1.0):
        """Store schema insights for future reference."""
        insight_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO schema_insights 
                (id, table_name, column_name, insight_type, insight_data, 
                 confidence_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (insight_id, table_name, column_name, insight_type,
                  json.dumps(insight_data), confidence_score, timestamp))
            await db.commit()
        
        return insight_id
    
    async def get_schema_insights(self, table_name: str = None, 
                                insight_type: str = None) -> List[Dict]:
        """Get stored schema insights."""
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            conditions = []
            params = []
            
            if table_name:
                conditions.append("table_name = ?")
                params.append(table_name)
            
            if insight_type:
                conditions.append("insight_type = ?")
                params.append(insight_type)
            
            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
            
            cursor = await db.execute(f"""
                SELECT table_name, column_name, insight_type, insight_data, 
                       confidence_score, last_updated
                FROM schema_insights{where_clause}
                ORDER BY confidence_score DESC
            """, params)
            
            rows = await cursor.fetchall()
            
            insights = []
            for row in rows:
                insights.append({
                    "table_name": row[0],
                    "column_name": row[1],
                    "insight_type": row[2],
                    "insight_data": json.loads(row[3]) if row[3] else {},
                    "confidence_score": row[4],
                    "last_updated": row[5]
                })
            
            return insights
    
    async def update_pattern_effectiveness(self, pattern_type: str, 
                                         effectiveness_score: float):
        """Update effectiveness score of a query pattern."""
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            await db.execute("""
                UPDATE query_patterns 
                SET effectiveness_score = ?, usage_count = usage_count + 1,
                    last_used = ?
                WHERE pattern_type = ?
            """, (effectiveness_score, timestamp, pattern_type))
            await db.commit()
    
    async def cleanup_old_sessions(self, days_old: int = 7):
        """Clean up old session data."""
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        async with aiosqlite.connect(self.session_db_path) as db:
            # Delete old queries first (foreign key constraint)
            await db.execute("""
                DELETE FROM session_queries 
                WHERE session_id IN (
                    SELECT id FROM sessions WHERE last_activity < ?
                )
            """, (cutoff_date,))
            
            # Delete old sessions
            await db.execute("""
                DELETE FROM sessions WHERE last_activity < ?
            """, (cutoff_date,))
            
            await db.commit()
    
    async def get_memory_stats(self) -> Dict:
        """Get memory system statistics."""
        stats = {}
        
        # Session stats
        async with aiosqlite.connect(self.session_db_path) as session_db:
            cursor = await session_db.execute("SELECT COUNT(*) FROM sessions")
            stats["active_sessions"] = (await cursor.fetchone())[0]
            
            cursor = await session_db.execute("SELECT COUNT(*) FROM session_queries")
            stats["total_session_queries"] = (await cursor.fetchone())[0]
        
        # Knowledge stats
        async with aiosqlite.connect(self.knowledge_db_path) as knowledge_db:
            cursor = await knowledge_db.execute("SELECT COUNT(*) FROM successful_queries")
            stats["successful_queries"] = (await cursor.fetchone())[0]
            
            cursor = await knowledge_db.execute("SELECT COUNT(*) FROM query_patterns")
            stats["stored_patterns"] = (await cursor.fetchone())[0]
            
            cursor = await knowledge_db.execute("SELECT COUNT(*) FROM schema_insights")
            stats["schema_insights"] = (await cursor.fetchone())[0]
        
        stats["vector_search_enabled"] = self.enable_vector_search
        stats["similarity_threshold"] = self.similarity_threshold
        
        return stats


class MemoryManager:
    """
    Memory manager that coordinates between session and knowledge memory.
    Provides a unified interface for the agent system.
    """
    
    def __init__(self, memory_system: MinimalMemorySystem):
        self.memory_system = memory_system
        self.working_memory = WorkingMemory()
        self.logger = logging.getLogger(__name__)
    
    async def initialize_processing_session(self, user_id: str, session_id: str, query: str) -> Dict:
        """Initialize a processing session with memory context."""
        # Initialize session
        session_id = await self.memory_system.initialize_session(user_id, session_id)
        
        # Store current query
        await self.memory_system.store_session_query(session_id, query)
        
        # Get context
        context = {
            "session_id": session_id,
            "user_id": user_id,
            "current_query": query,
            "session_history": await self.memory_system.get_session_history(session_id),
            "similar_queries": await self.memory_system.find_similar_queries(query)
        }
        
        return context
    
    async def get_contextual_memories(self, query: str, user_id: str, context_type: str = "general") -> Dict:
        """Get relevant contextual memories for processing."""
        return {
            "session_memories": {
                "similar_queries": await self.memory_system.find_similar_queries(query),
                "query_patterns": await self.memory_system.get_query_patterns()
            },
            "knowledge_memories": {
                "patterns": await self.memory_system.get_query_patterns(),
                "schema_insights": await self.memory_system.get_schema_insights()
            }
        }
    
    async def update_memory_from_processing(self, session_id: str, agent_name: str,
                                          processing_data: Dict, success: bool):
        """Update memory with processing results."""
        # Update working memory
        await self.working_memory.update_context(agent_name, processing_data)
        
        # Store successful patterns if applicable
        if success and "generated_sql" in processing_data:
            await self.memory_system.store_successful_query(
                query=processing_data.get("original_query", ""),
                sql=processing_data["generated_sql"],
                execution_time=processing_data.get("execution_time", 0),
                result_count=processing_data.get("result_count", 0),
                tables_used=processing_data.get("tables_used", [])
            )
    
    async def finalize_session(self, session_id: str, final_results: Dict, user_feedback: Dict = None):
        """Finalize session and update long-term memories."""
        # Store final results if successful
        if final_results.get("success") and final_results.get("generated_sql"):
            await self.memory_system.store_successful_query(
                query=final_results.get("user_query", ""),
                sql=final_results["generated_sql"],
                execution_time=final_results.get("execution_time", 0),
                result_count=len(final_results.get("query_results", [])),
                tables_used=final_results.get("tables_used", []),
                metadata=final_results
            )
        
        # Clear working memory
        self.working_memory.clear()


class WorkingMemory:
    """Simple in-memory storage for current processing context."""
    
    def __init__(self):
        self.context = {}
        self.agent_states = {}
    
    async def update_context(self, agent_name: str, update_data: Dict):
        """Update context with agent processing data."""
        self.agent_states[agent_name] = update_data
        self.context.update(update_data)
    
    def get_context(self) -> Dict:
        """Get current context."""
        return self.context.copy()
    
    def get_agent_state(self, agent_name: str) -> Dict:
        """Get state for specific agent."""
        return self.agent_states.get(agent_name, {})
    
    def clear(self):
        """Clear working memory."""
        self.context.clear()
        self.agent_states.clear()