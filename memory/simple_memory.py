"""
Simple Memory System - Streamlined for Direct Agent Usage
Combines session and knowledge memory with minimal overhead
"""

import asyncio
import aiosqlite
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
import sqlite3

class SimpleMemorySystem:
    """
    Streamlined memory system focused on essential functionality
    Supports both in-memory and persistent storage
    """
    
    def __init__(self, 
                 session_db_path: str = ":memory:",
                 knowledge_db_path: str = ":memory:",
                 similarity_threshold: float = 0.3,
                 enable_persistence: bool = False,
                 persistent_session_path: str = "data/session_memory.db",
                 persistent_knowledge_path: str = "data/knowledge_memory.db"):
        """
        Initialize simple memory system
        
        Args:
            session_db_path: SQLite path for session memory (":memory:" for in-memory)
            knowledge_db_path: SQLite path for persistent knowledge
            similarity_threshold: Threshold for text-based similarity matching
        """
        self.session_db_path = session_db_path
        self.knowledge_db_path = knowledge_db_path
        self.similarity_threshold = similarity_threshold
        self.enable_persistence = enable_persistence
        self.persistent_session_path = persistent_session_path
        self.persistent_knowledge_path = persistent_knowledge_path
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize databases
        self._setup_databases()
        
        # Session context cache
        self.session_contexts = {}
        
        # In-memory backup for persistence
        self._session_backup = {}
        self._knowledge_backup = {}
    
    def _setup_databases(self):
        """Setup database connections and tables"""
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
        """Create necessary tables in both databases"""
        
        # Session memory tables (conversation context)
        async with aiosqlite.connect(self.session_db_path) as session_db:
            await session_db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    context_data TEXT
                )
            """)
            
            await session_db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    intent TEXT,
                    tables_used TEXT,
                    chart_type TEXT,
                    success INTEGER DEFAULT 1,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            await session_db.commit()
        
        # Knowledge memory tables (learning and patterns)
        async with aiosqlite.connect(self.knowledge_db_path) as knowledge_db:
            await knowledge_db.execute("""
                CREATE TABLE IF NOT EXISTS successful_queries (
                    id TEXT PRIMARY KEY,
                    query_pattern TEXT NOT NULL,
                    sql_query TEXT NOT NULL,
                    tables_used TEXT,
                    chart_type TEXT,
                    execution_time REAL DEFAULT 0.0,
                    result_count INTEGER DEFAULT 0,
                    success_score REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_used TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            await knowledge_db.execute("""
                CREATE TABLE IF NOT EXISTS query_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    pattern_description TEXT,
                    template_sql TEXT,
                    example_queries TEXT,
                    success_rate REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_used TEXT NOT NULL,
                    metadata TEXT
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
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)
            
            # Create indexes for better performance
            await knowledge_db.execute("CREATE INDEX IF NOT EXISTS idx_query_pattern ON successful_queries(query_pattern)")
            await knowledge_db.execute("CREATE INDEX IF NOT EXISTS idx_last_used ON successful_queries(last_used)")
            await knowledge_db.execute("CREATE INDEX IF NOT EXISTS idx_pattern_type ON query_patterns(pattern_type)")
            await knowledge_db.execute("CREATE INDEX IF NOT EXISTS idx_table_name ON schema_insights(table_name)")
            
            await knowledge_db.commit()
    
    async def _load_persistent_data(self):
        """Load data from persistent storage into in-memory databases"""
        if not self.enable_persistence:
            return
        
        try:
            # Load session data
            if os.path.exists(self.persistent_session_path):
                await self._copy_data_between_dbs(self.persistent_session_path, self.session_db_path)
            
            # Load knowledge data  
            if os.path.exists(self.persistent_knowledge_path):
                await self._copy_data_between_dbs(self.persistent_knowledge_path, self.knowledge_db_path)
                
        except Exception as e:
            self.logger.warning(f"Failed to load persistent data: {e}")
    
    async def save_to_persistent_storage(self):
        """Save in-memory data to persistent storage"""
        if not self.enable_persistence:
            return
        
        try:
            # Save session data
            await self._copy_data_between_dbs(self.session_db_path, self.persistent_session_path)
            
            # Save knowledge data
            await self._copy_data_between_dbs(self.knowledge_db_path, self.persistent_knowledge_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save persistent data: {e}")
    
    async def _copy_data_between_dbs(self, source_path: str, target_path: str):
        """Copy data between SQLite databases"""
        async with aiosqlite.connect(source_path) as source_db:
            async with aiosqlite.connect(target_path) as target_db:
                # First ensure target has same schema
                await self._create_tables_for_db(target_db, target_path)
                
                # Get all table names
                cursor = await source_db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = await cursor.fetchall()
                
                for (table_name,) in tables:
                    # Copy table data
                    cursor = await source_db.execute(f"SELECT * FROM {table_name}")
                    rows = await cursor.fetchall()
                    
                    if rows:
                        # Get column info
                        cursor = await source_db.execute(f"PRAGMA table_info({table_name})")
                        columns = await cursor.fetchall()
                        column_names = [col[1] for col in columns]
                        placeholders = ','.join(['?' for _ in column_names])
                        
                        # Insert data
                        await target_db.executemany(
                            f"INSERT OR REPLACE INTO {table_name} VALUES ({placeholders})", 
                            rows
                        )
                
                await target_db.commit()
    
    async def _create_tables_for_db(self, db, db_path: str):
        """Create tables for a specific database connection"""
        session_pragmas = [
            "PRAGMA journal_mode = WAL;",
            "PRAGMA synchronous = NORMAL;", 
            "PRAGMA temp_store = MEMORY;",
            "PRAGMA cache_size = 10000;",
            "PRAGMA foreign_keys = ON;"
        ]
        
        # Apply pragmas (different for persistent vs in-memory)
        if db_path == ":memory:":
            await db.execute("PRAGMA journal_mode = MEMORY;")
            await db.execute("PRAGMA synchronous = OFF;")
        else:
            for pragma in session_pragmas:
                await db.execute(pragma)
        
        # Create all necessary tables (same structure as _create_tables)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                context_data TEXT
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                intent TEXT,
                tables_used TEXT,
                chart_type TEXT,
                success INTEGER DEFAULT 1,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS successful_queries (
                id TEXT PRIMARY KEY,
                query_pattern TEXT NOT NULL,
                sql_query TEXT NOT NULL,
                tables_used TEXT,
                chart_type TEXT,
                execution_time REAL DEFAULT 0.0,
                result_count INTEGER DEFAULT 0,
                success_score REAL DEFAULT 1.0,
                usage_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                last_used TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS query_patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                pattern_description TEXT,
                template_sql TEXT,
                example_queries TEXT,
                success_rate REAL DEFAULT 1.0,
                usage_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                last_used TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS schema_insights (
                id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                column_name TEXT,
                insight_type TEXT NOT NULL,
                insight_data TEXT,
                confidence_score REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
    
    # =========================
    # Session Management
    # =========================
    
    async def create_session(self, user_id: str, session_id: str = None) -> str:
        """Create or update a user session"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.session_db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO sessions (id, user_id, created_at, last_activity, context_data)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, user_id, timestamp, timestamp, "{}"))
            await db.commit()
        
        # Initialize session context cache
        self.session_contexts[session_id] = {
            "user_id": user_id,
            "created_at": timestamp,
            "conversation_history": []
        }
        
        return session_id
    
    async def get_session_context(self, session_id: str) -> Dict:
        """Get session context including conversation history"""
        
        # Check cache first
        if session_id in self.session_contexts:
            context = self.session_contexts[session_id].copy()
        else:
            context = {"conversation_history": []}
        
        # Get recent conversation history from database
        async with aiosqlite.connect(self.session_db_path) as db:
            cursor = await db.execute("""
                SELECT query, intent, tables_used, chart_type, success, timestamp
                FROM conversation_history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 10
            """, (session_id,))
            
            rows = await cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    "query": row[0],
                    "intent": json.loads(row[1]) if row[1] else {},
                    "tables_used": json.loads(row[2]) if row[2] else [],
                    "chart_type": row[3],
                    "success": bool(row[4]),
                    "timestamp": row[5]
                })
            
            context["conversation_history"] = list(reversed(history))  # Chronological order
        
        return context
    
    async def add_to_conversation(self, session_id: str, query: str, intent: Dict = None, 
                                tables_used: List[str] = None, chart_type: str = None, 
                                success: bool = True):
        """Add interaction to conversation history"""
        
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.session_db_path) as db:
            await db.execute("""
                INSERT INTO conversation_history 
                (id, session_id, query, intent, tables_used, chart_type, success, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (interaction_id, session_id, query, 
                  json.dumps(intent) if intent else None,
                  json.dumps(tables_used) if tables_used else None,
                  chart_type, int(success), timestamp))
            
            # Update session last activity
            await db.execute("""
                UPDATE sessions SET last_activity = ? WHERE id = ?
            """, (timestamp, session_id))
            
            await db.commit()
        
        # Update cache
        if session_id in self.session_contexts:
            self.session_contexts[session_id]["conversation_history"].append({
                "query": query,
                "intent": intent or {},
                "tables_used": tables_used or [],
                "chart_type": chart_type,
                "success": success,
                "timestamp": timestamp
            })
            
            # Keep only recent history in cache
            if len(self.session_contexts[session_id]["conversation_history"]) > 10:
                self.session_contexts[session_id]["conversation_history"] = \
                    self.session_contexts[session_id]["conversation_history"][-10:]
    
    # =========================
    # Knowledge Management
    # =========================
    
    async def store_successful_query(self, query: str, sql: str, execution_time: float = 0.0,
                                   result_count: int = 0, tables_used: List[str] = None,
                                   chart_type: str = None, metadata: Dict = None):
        """Store successful query pattern for learning"""
        
        query_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create query pattern for similarity matching
        query_pattern = self._extract_query_pattern(query)
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            await db.execute("""
                INSERT INTO successful_queries 
                (id, query_pattern, sql_query, tables_used, chart_type, execution_time, 
                 result_count, created_at, last_used, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (query_id, query_pattern, sql, 
                  json.dumps(tables_used) if tables_used else None,
                  chart_type, execution_time, result_count, timestamp, timestamp,
                  json.dumps(metadata) if metadata else None))
            await db.commit()
        
        return query_id
    
    async def find_similar_queries(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find similar queries using text-based similarity"""
        
        query_pattern = self._extract_query_pattern(query)
        query_words = set(query_pattern.lower().split())
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            cursor = await db.execute("""
                SELECT query_pattern, sql_query, tables_used, chart_type, 
                       execution_time, result_count, success_score, usage_count, metadata
                FROM successful_queries 
                ORDER BY last_used DESC 
                LIMIT 50
            """)
            
            rows = await cursor.fetchall()
            
            similar_queries = []
            for row in rows:
                stored_pattern = row[0].lower()
                stored_words = set(stored_pattern.split())
                
                # Calculate Jaccard similarity
                intersection = len(query_words & stored_words)
                union = len(query_words | stored_words)
                similarity = intersection / union if union > 0 else 0
                
                if similarity >= self.similarity_threshold:
                    similar_queries.append({
                        "query_pattern": row[0],
                        "generated_sql": row[1],
                        "tables_used": json.loads(row[2]) if row[2] else [],
                        "chart_type": row[3],
                        "execution_time": row[4],
                        "result_count": row[5],
                        "success_score": row[6],
                        "usage_count": row[7],
                        "similarity_score": similarity,
                        "metadata": json.loads(row[8]) if row[8] else {}
                    })
            
            # Sort by similarity score and return top k
            similar_queries.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similar_queries[:top_k]
    
    async def store_query_pattern(self, pattern_type: str, pattern_description: str,
                                example_queries: List[str], template_sql: str = None,
                                metadata: Dict = None):
        """Store reusable query pattern"""
        
        pattern_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            await db.execute("""
                INSERT INTO query_patterns 
                (id, pattern_type, pattern_description, template_sql, example_queries, 
                 created_at, last_used, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (pattern_id, pattern_type, pattern_description, template_sql,
                  json.dumps(example_queries), timestamp, timestamp,
                  json.dumps(metadata) if metadata else None))
            await db.commit()
        
        return pattern_id
    
    async def get_query_patterns(self, pattern_type: str = None) -> List[Dict]:
        """Get stored query patterns"""
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            if pattern_type:
                cursor = await db.execute("""
                    SELECT pattern_type, pattern_description, template_sql, example_queries, 
                           success_rate, usage_count, metadata
                    FROM query_patterns 
                    WHERE pattern_type = ?
                    ORDER BY success_rate DESC, usage_count DESC
                """, (pattern_type,))
            else:
                cursor = await db.execute("""
                    SELECT pattern_type, pattern_description, template_sql, example_queries, 
                           success_rate, usage_count, metadata
                    FROM query_patterns 
                    ORDER BY success_rate DESC, usage_count DESC
                """)
            
            rows = await cursor.fetchall()
            
            patterns = []
            for row in rows:
                patterns.append({
                    "pattern_type": row[0],
                    "pattern_description": row[1],
                    "template_sql": row[2],
                    "example_queries": json.loads(row[3]) if row[3] else [],
                    "success_rate": row[4],
                    "usage_count": row[5],
                    "metadata": json.loads(row[6]) if row[6] else {}
                })
            
            return patterns
    
    async def store_schema_insight(self, table_name: str, insight_type: str,
                                 insight_data: Dict, column_name: str = None,
                                 confidence_score: float = 1.0):
        """Store schema insights for future reference"""
        
        insight_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO schema_insights 
                (id, table_name, column_name, insight_type, insight_data, 
                 confidence_score, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (insight_id, table_name, column_name, insight_type,
                  json.dumps(insight_data), confidence_score, timestamp, timestamp))
            await db.commit()
        
        return insight_id
    
    async def get_schema_insights(self, table_name: str = None, 
                                insight_type: str = None) -> List[Dict]:
        """Get stored schema insights"""
        
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
                ORDER BY confidence_score DESC, last_updated DESC
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
    
    # =========================
    # Memory Utilities
    # =========================
    
    def _extract_query_pattern(self, query: str) -> str:
        """Extract searchable pattern from natural language query"""
        
        # Convert to lowercase and remove common words
        common_words = {
            "show", "me", "get", "find", "tell", "what", "is", "are", "the", "a", "an",
            "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"
        }
        
        words = query.lower().split()
        meaningful_words = [word for word in words if word not in common_words and len(word) > 2]
        
        return " ".join(meaningful_words)
    
    async def update_query_usage(self, query_pattern: str, execution_time: float = 0.0, 
                               success: bool = True):
        """Update usage statistics for a query pattern"""
        
        async with aiosqlite.connect(self.knowledge_db_path) as db:
            # Find matching pattern
            cursor = await db.execute("""
                SELECT id, usage_count, success_score 
                FROM successful_queries 
                WHERE query_pattern = ?
            """, (query_pattern,))
            
            row = await cursor.fetchone()
            
            if row:
                query_id, usage_count, success_score = row
                
                # Update statistics
                new_usage_count = usage_count + 1
                new_success_score = ((success_score * usage_count) + (1.0 if success else 0.0)) / new_usage_count
                
                await db.execute("""
                    UPDATE successful_queries 
                    SET usage_count = ?, success_score = ?, last_used = ?, execution_time = ?
                    WHERE id = ?
                """, (new_usage_count, new_success_score, datetime.now().isoformat(), 
                          execution_time, query_id))
                
                await db.commit()
    
    async def cleanup_old_data(self, days_old: int = 30):
        """Clean up old session data and low-usage patterns"""
        
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        # Clean up old sessions
        async with aiosqlite.connect(self.session_db_path) as session_db:
            await session_db.execute("""
                DELETE FROM conversation_history 
                WHERE session_id IN (
                    SELECT id FROM sessions WHERE last_activity < ?
                )
            """, (cutoff_date,))
            
            await session_db.execute("""
                DELETE FROM sessions WHERE last_activity < ?
            """, (cutoff_date,))
            
            await session_db.commit()
        
        # Clean up low-performing patterns
        async with aiosqlite.connect(self.knowledge_db_path) as knowledge_db:
            await knowledge_db.execute("""
                DELETE FROM successful_queries 
                WHERE last_used < ? AND usage_count < 3 AND success_score < 0.5
            """, (cutoff_date,))
            
            await knowledge_db.commit()
        
        # Clear session cache
        self.session_contexts.clear()
    
    async def get_memory_stats(self) -> Dict:
        """Get memory system statistics"""
        
        stats = {}
        
        # Session stats
        try:
            async with aiosqlite.connect(self.session_db_path) as session_db:
                cursor = await session_db.execute("SELECT COUNT(*) FROM sessions")
                stats["active_sessions"] = (await cursor.fetchone())[0]
                
                cursor = await session_db.execute("SELECT COUNT(*) FROM conversation_history")
                stats["total_conversations"] = (await cursor.fetchone())[0]
        except:
            stats["active_sessions"] = 0
            stats["total_conversations"] = 0
        
        # Knowledge stats
        try:
            async with aiosqlite.connect(self.knowledge_db_path) as knowledge_db:
                cursor = await knowledge_db.execute("SELECT COUNT(*) FROM successful_queries")
                stats["successful_queries"] = (await cursor.fetchone())[0]
                
                cursor = await knowledge_db.execute("SELECT COUNT(*) FROM query_patterns")
                stats["stored_patterns"] = (await cursor.fetchone())[0]
                
                cursor = await knowledge_db.execute("SELECT COUNT(*) FROM schema_insights")
                stats["schema_insights"] = (await cursor.fetchone())[0]
                
                cursor = await knowledge_db.execute("""
                    SELECT AVG(success_score) FROM successful_queries
                """)
                avg_success = await cursor.fetchone()
                stats["average_success_rate"] = round(avg_success[0] or 0.0, 2)
        except:
            stats["successful_queries"] = 0
            stats["stored_patterns"] = 0
            stats["schema_insights"] = 0
            stats["average_success_rate"] = 0.0
        
        stats["similarity_threshold"] = self.similarity_threshold
        stats["session_cache_size"] = len(self.session_contexts)
        
        return stats