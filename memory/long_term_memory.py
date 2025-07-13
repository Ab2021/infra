"""
Long-term Knowledge Memory - SQLite + FAISS for persistent learning
Manages accumulated knowledge, patterns, and schema intelligence using SQLite and FAISS
"""

import sqlite3
import asyncio
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import logging
from pathlib import Path

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS or sentence-transformers not available. Vector search disabled.")

class LongTermKnowledgeMemory:
    """
    Manages long-term knowledge and learning patterns using SQLite for metadata
    and FAISS for vector similarity search.
    """
    
    def __init__(self, db_path: str = ":memory:", 
                 vector_path: str = "data/vector_store",
                 enable_persistence: bool = True,
                 persistent_db_path: str = "data/knowledge_memory.db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 max_context_length: int = 10000):
        self.db_path = db_path
        self.vector_path = str(Path(vector_path).resolve())
        self.enable_persistence = enable_persistence
        self.persistent_db_path = persistent_db_path
        self.embedding_model_name = embedding_model
        self.max_context_length = max_context_length
        
        # Security: Resolve and validate paths to prevent directory traversal
        cwd = str(Path.cwd())
        if not self.vector_path.startswith(cwd):
            raise ValueError("Vector path must be within current working directory")
        
        if self.enable_persistence and self.persistent_db_path != ":memory:":
            self.persistent_db_path = str(Path(persistent_db_path).resolve())
            if not self.persistent_db_path.startswith(cwd):
                raise ValueError("Persistent database path must be within current working directory")
            Path(os.path.dirname(self.persistent_db_path)).mkdir(parents=True, exist_ok=True, mode=0o750)
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure vector directory exists with secure permissions
        Path(self.vector_path).mkdir(parents=True, exist_ok=True, mode=0o750)
        
        # Initialize database
        self._initialize_database()
        
        # Initialize vector store if available
        self.vector_store = None
        self.embedding_model = None
        self.index_to_id = {}
        self.id_to_index = {}
        self.context_cache = {}  # Cache for frequently accessed contexts
        
        if FAISS_AVAILABLE:
            self._initialize_vector_store()
        
        # Load from persistent storage if enabled
        if self.enable_persistence and os.path.exists(self.persistent_db_path):
            asyncio.run(self._load_from_persistent())
        
        # Connection pool for thread safety
        self._connection_lock = asyncio.Lock()
        self._vector_lock = asyncio.Lock()
    
    def _get_secure_connection(self):
        """Returns a secure SQLite connection with proper settings."""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        # Security: Enable foreign key constraints and set secure pragmas
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Configure for in-memory vs persistent storage
        if self.db_path == ":memory:":
            conn.execute("PRAGMA journal_mode = MEMORY")
            conn.execute("PRAGMA synchronous = OFF")
        else:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
        return conn
    
    def _initialize_database(self):
        """Creates SQLite tables for long-term knowledge storage."""
        
        with sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False) as conn:
            # Security: Enable foreign key constraints and set secure pragmas
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Configure for in-memory vs persistent storage
            if self.db_path == ":memory:":
                conn.execute("PRAGMA journal_mode = MEMORY")
                conn.execute("PRAGMA synchronous = OFF")
            else:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
            
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
            conn.execute("PRAGMA cache_size = 10000")
            cursor = conn.cursor()
            
            # Query patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    success_rate REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Schema insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    insight_data TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Learning patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_category TEXT NOT NULL,
                    pattern_content TEXT NOT NULL,
                    effectiveness REAL DEFAULT 0.5,
                    usage_frequency INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Vector embeddings metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vector_metadata (
                    vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_type TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    faiss_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON query_patterns(pattern_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_schema_table ON schema_insights(table_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_category ON learning_patterns(pattern_category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vector_hash ON vector_metadata(content_hash)")
            
            conn.commit()
    
    async def store_long_term_context(self, context_type: str, context_key: str, 
                                    context_data: Dict, metadata: Dict = None,
                                    ttl_hours: int = None) -> int:
        """
        Store long-term context with FAISS vector indexing
        
        Args:
            context_type: Type of context (e.g., 'query_pattern', 'schema_insight', 'user_preference')
            context_key: Unique key for the context
            context_data: The actual context data
            metadata: Additional metadata
            ttl_hours: Time to live in hours (None for permanent)
        
        Returns:
            Context ID
        """
        
        # Create text representation for vector embedding
        text_content = self._create_context_text(context_type, context_key, context_data)
        
        # Store in vector database if available
        embedding_id = None
        if self.vector_store is not None and len(text_content.strip()) > 0:
            embedding_id = await self._add_context_to_vector_store(text_content, {
                'context_type': context_type,
                'context_key': context_key,
                'context_data': context_data,
                'metadata': metadata
            })
        
        # Calculate expiration time
        expires_at = None
        if ttl_hours:
            from datetime import datetime, timedelta
            expires_at = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                # Check if we need to add new columns for enhanced context storage
                cursor.execute("PRAGMA table_info(query_patterns)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'embedding_id' not in columns:
                    # Add enhanced context storage table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS long_term_contexts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            context_type TEXT NOT NULL,
                            context_key TEXT NOT NULL,
                            context_data TEXT NOT NULL,
                            embedding_id INTEGER,
                            relevance_score REAL DEFAULT 1.0,
                            access_count INTEGER DEFAULT 0,
                            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP,
                            metadata TEXT
                        )
                    """)
                    
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_context_type_key 
                        ON long_term_contexts(context_type, context_key)
                    """)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO long_term_contexts 
                    (context_type, context_key, context_data, embedding_id, 
                     expires_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    context_type, context_key, json.dumps(context_data),
                    embedding_id, expires_at, json.dumps(metadata) if metadata else None
                ))
                
                context_id = cursor.lastrowid
                conn.commit()
                
                self.logger.debug(f"Stored long-term context: {context_type}/{context_key} (ID: {context_id})")
                return context_id
    
    async def retrieve_long_term_context(self, context_type: str = None, 
                                       context_key: str = None,
                                       query_text: str = None,
                                       top_k: int = 10,
                                       similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Retrieve long-term contexts using multiple strategies
        
        Args:
            context_type: Filter by context type
            context_key: Exact context key match
            query_text: Text for similarity search
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity score for vector search
        
        Returns:
            List of matching contexts with relevance scores
        """
        
        results = []
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                # Check if enhanced table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='long_term_contexts'")
                if not cursor.fetchone():
                    return []
                
                # Strategy 1: Exact key match (highest priority)
                if context_key:
                    if context_type:
                        cursor.execute("""
                            SELECT id, context_type, context_key, context_data, 
                                   relevance_score, access_count, metadata
                            FROM long_term_contexts 
                            WHERE context_type = ? AND context_key = ?
                            AND (expires_at IS NULL OR expires_at > datetime('now'))
                        """, (context_type, context_key))
                    else:
                        cursor.execute("""
                            SELECT id, context_type, context_key, context_data, 
                                   relevance_score, access_count, metadata
                            FROM long_term_contexts 
                            WHERE context_key = ?
                            AND (expires_at IS NULL OR expires_at > datetime('now'))
                        """, (context_key,))
                    
                    for row in cursor.fetchall():
                        results.append({
                            'id': row[0],
                            'context_type': row[1],
                            'context_key': row[2],
                            'context_data': json.loads(row[3]),
                            'relevance_score': row[4],
                            'access_count': row[5],
                            'metadata': json.loads(row[6]) if row[6] else {},
                            'match_type': 'exact_key'
                        })
                
                # Strategy 2: Vector similarity search
                if query_text and self.vector_store is not None:
                    vector_matches = await self._search_contexts_by_similarity(
                        query_text, top_k, similarity_threshold
                    )
                    results.extend(vector_matches)
                
                # Strategy 3: Type-based retrieval
                if context_type and not context_key:
                    cursor.execute("""
                        SELECT id, context_type, context_key, context_data, 
                               relevance_score, access_count, metadata
                        FROM long_term_contexts 
                        WHERE context_type = ?
                        AND (expires_at IS NULL OR expires_at > datetime('now'))
                        ORDER BY relevance_score DESC, access_count DESC
                        LIMIT ?
                    """, (context_type, top_k))
                    
                    for row in cursor.fetchall():
                        results.append({
                            'id': row[0],
                            'context_type': row[1],
                            'context_key': row[2],
                            'context_data': json.loads(row[3]),
                            'relevance_score': row[4],
                            'access_count': row[5],
                            'metadata': json.loads(row[6]) if row[6] else {},
                            'match_type': 'type_based'
                        })
        
        # Remove duplicates and rank by relevance
        seen_ids = set()
        unique_results = []
        for result in results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        # Sort by relevance score and access frequency
        unique_results.sort(key=lambda x: (x.get('relevance_score', 0), x.get('access_count', 0)), reverse=True)
        
        # Update access statistics
        if unique_results:
            await self._update_context_access_stats([r['id'] for r in unique_results[:top_k]])
        
        return unique_results[:top_k]
    
    async def _search_contexts_by_similarity(self, query_text: str, top_k: int, threshold: float) -> List[Dict]:
        """Search contexts using vector similarity"""
        
        if not FAISS_AVAILABLE or self.vector_store is None or self.vector_store.ntotal == 0:
            return []
        
        try:
            async with self._vector_lock:
                # Generate query embedding
                query_embedding = self.embedding_model.encode([query_text])
                faiss.normalize_L2(query_embedding)
                
                # Search
                scores, indices = self.vector_store.search(query_embedding, min(top_k * 2, self.vector_store.ntotal))
                
                # Retrieve context metadata
                results = []
                with self._get_secure_connection() as conn:
                    cursor = conn.cursor()
                    
                    for score, index in zip(scores[0], indices[0]):
                        if index == -1 or score < threshold:
                            continue
                            
                        vector_id = self.index_to_id.get(index)
                        if vector_id:
                            cursor.execute("""
                                SELECT vm.metadata, ltc.id, ltc.context_type, ltc.context_key,
                                       ltc.context_data, ltc.relevance_score, ltc.access_count,
                                       ltc.metadata as context_metadata
                                FROM vector_metadata vm
                                LEFT JOIN long_term_contexts ltc ON vm.vector_id = ltc.embedding_id
                                WHERE vm.vector_id = ?
                            """, (vector_id,))
                            
                            row = cursor.fetchone()
                            if row and row[1]:  # Has context data
                                results.append({
                                    'id': row[1],
                                    'context_type': row[2],
                                    'context_key': row[3],
                                    'context_data': json.loads(row[4]),
                                    'relevance_score': max(float(score), row[5] or 0),
                                    'access_count': row[6] or 0,
                                    'metadata': json.loads(row[7]) if row[7] else {},
                                    'similarity_score': float(score),
                                    'match_type': 'vector_similarity'
                                })
                
                return results
                
        except Exception as e:
            self.logger.error(f"Vector similarity search failed: {e}")
            return []
    
    async def _update_context_access_stats(self, context_ids: List[int]):
        """Update access statistics for contexts"""
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                for context_id in context_ids:
                    cursor.execute("""
                        UPDATE long_term_contexts 
                        SET access_count = access_count + 1,
                            last_accessed = datetime('now')
                        WHERE id = ?
                    """, (context_id,))
                
                conn.commit()
    
    def _create_context_text(self, context_type: str, context_key: str, context_data: Dict) -> str:
        """Create searchable text representation of context"""
        
        text_parts = [f"Type: {context_type}", f"Key: {context_key}"]
        
        # Extract meaningful text from context data
        if isinstance(context_data, dict):
            for key, value in context_data.items():
                if isinstance(value, (str, int, float)):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    text_parts.append(f"{key}: {', '.join(map(str, value[:5]))}")
        
        return " | ".join(text_parts)
    
    async def _add_context_to_vector_store(self, text_content: str, context_metadata: Dict) -> int:
        """Add context to FAISS vector store"""
        
        if not FAISS_AVAILABLE or self.vector_store is None:
            return None
        
        try:
            async with self._vector_lock:
                # Generate embedding
                embedding = self.embedding_model.encode([text_content])
                faiss.normalize_L2(embedding)
                
                # Add to FAISS index
                next_index = self.vector_store.ntotal
                self.vector_store.add(embedding)
                
                # Store metadata in database
                content_hash = str(hash(text_content))
                
                with self._get_secure_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO vector_metadata 
                        (content_type, content_text, content_hash, faiss_index, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        context_metadata.get('context_type', 'unknown'),
                        text_content[:self.max_context_length],  # Truncate if too long
                        content_hash,
                        next_index,
                        json.dumps(context_metadata)
                    ))
                    
                    vector_id = cursor.lastrowid
                    
                    # Update mappings
                    self.index_to_id[next_index] = vector_id
                    self.id_to_index[vector_id] = next_index
                    
                    conn.commit()
                
                # Save updated index
                self._save_vector_store()
                
                return vector_id
                
        except Exception as e:
            self.logger.error(f"Failed to add context to vector store: {e}")
            return None
    
    def _initialize_vector_store(self):
        """Initializes FAISS vector store for similarity search."""
        
        if not FAISS_AVAILABLE:
            return
        
        try:
            # Initialize sentence transformer model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            self.logger.info(f"Initialized embedding model: {self.embedding_model_name} (dim: {embedding_dim})")
            
            # Initialize or load FAISS index
            index_path = os.path.join(self.vector_path, "knowledge.index")
            mapping_path = os.path.join(self.vector_path, "index_mapping.pkl")
            
            if os.path.exists(index_path) and os.path.exists(mapping_path):
                # Load existing index
                self.vector_store = faiss.read_index(index_path)
                with open(mapping_path, 'rb') as f:
                    mapping_data = pickle.load(f)
                    self.index_to_id = mapping_data.get('index_to_id', {})
                    self.id_to_index = mapping_data.get('id_to_index', {})
                    
                self.logger.info(f"Loaded existing FAISS index with {self.vector_store.ntotal} vectors")
            else:
                # Create new index with better performance characteristics
                # Use IndexIVFFlat for better performance with large datasets
                if embedding_dim > 0:
                    quantizer = faiss.IndexFlatIP(embedding_dim)
                    nlist = min(100, max(1, embedding_dim // 4))  # Number of clusters
                    self.vector_store = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
                    # Train with dummy data if no existing vectors
                    dummy_vectors = np.random.random((max(nlist, 100), embedding_dim)).astype('float32')
                    faiss.normalize_L2(dummy_vectors)
                    self.vector_store.train(dummy_vectors)
                else:
                    self.vector_store = faiss.IndexFlatIP(embedding_dim)
                
                self.index_to_id = {}
                self.id_to_index = {}
                
                self.logger.info(f"Created new FAISS index (type: {type(self.vector_store).__name__})")
                
            # Set search parameters for IVF index
            if hasattr(self.vector_store, 'nprobe'):
                self.vector_store.nprobe = min(10, self.vector_store.nlist)  # Number of clusters to search
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {str(e)}")
            self.vector_store = None
    
    async def get_relevant_context(self, query: str, user_id: str, session_context: Dict) -> Dict:
        """
        Retrieves relevant long-term context for query processing.
        
        Args:
            query: Current query
            user_id: User identifier
            session_context: Current session context
            
        Returns:
            Relevant long-term knowledge and patterns
        """
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                # Get relevant query patterns
                cursor.execute("""
                    SELECT pattern_type, pattern_data, frequency, success_rate
                    FROM query_patterns 
                    ORDER BY frequency DESC, success_rate DESC 
                    LIMIT 10
                """, )
                
                query_patterns = []
                for row in cursor.fetchall():
                    query_patterns.append({
                        "type": row[0],
                        "data": json.loads(row[1]),
                        "frequency": row[2],
                        "success_rate": row[3]
                    })
                
                # Get schema insights if query involves database tables
                schema_insights = []
                table_keywords = self._extract_table_keywords(query)
                if table_keywords:
                    for table in table_keywords:
                        cursor.execute("""
                            SELECT insight_type, insight_data, confidence
                            FROM schema_insights 
                            WHERE table_name LIKE ? 
                            ORDER BY confidence DESC 
                            LIMIT 5
                        """, (f"%{table}%",))
                        
                        for row in cursor.fetchall():
                            schema_insights.append({
                                "table": table,
                                "type": row[0],
                                "data": json.loads(row[1]),
                                "confidence": row[2]
                            })
                
                # Vector similarity search for related content
                similar_content = []
                if self.vector_store is not None:
                    similar_content = await self._vector_similarity_search(query, top_k=5)
                
                return {
                    "query_patterns": query_patterns,
                    "schema_insights": schema_insights,
                    "similar_content": similar_content,
                    "total_patterns": len(query_patterns),
                    "knowledge_retrieval_time": datetime.now().isoformat()
                }
    
    async def get_relevant_patterns(self, query: str, context_type: str) -> Dict:
        """
        Retrieves patterns relevant to specific processing context.
        
        Args:
            query: Current query
            context_type: Type of processing context
            
        Returns:
            Relevant patterns for the context
        """
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                # Context-specific pattern retrieval
                if context_type == "sql_generation":
                    cursor.execute("""
                        SELECT pattern_content, effectiveness, usage_frequency
                        FROM learning_patterns 
                        WHERE pattern_category = 'sql_generation'
                        ORDER BY effectiveness DESC, usage_frequency DESC 
                        LIMIT 5
                    """)
                    
                elif context_type == "schema_analysis":
                    cursor.execute("""
                        SELECT pattern_content, effectiveness, usage_frequency
                        FROM learning_patterns 
                        WHERE pattern_category = 'schema_analysis'
                        ORDER BY effectiveness DESC, usage_frequency DESC 
                        LIMIT 5
                    """)
                    
                else:
                    cursor.execute("""
                        SELECT pattern_content, effectiveness, usage_frequency
                        FROM learning_patterns 
                        ORDER BY effectiveness DESC, usage_frequency DESC 
                        LIMIT 10
                    """)
                
                patterns = []
                for row in cursor.fetchall():
                    patterns.append({
                        "content": json.loads(row[0]),
                        "effectiveness": row[1],
                        "frequency": row[2]
                    })
                
                return {
                    "patterns": patterns,
                    "context_type": context_type,
                    "total_found": len(patterns)
                }
    
    async def learn_from_interaction(self, agent_name: str, interaction_data: Dict):
        """
        Learns from successful agent interactions.
        
        Args:
            agent_name: Name of the agent
            interaction_data: Data from the interaction
        """
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                # Extract learning patterns based on agent type
                if agent_name == "sql_generator" and "generated_sql" in interaction_data:
                    await self._learn_sql_patterns(cursor, interaction_data)
                
                elif agent_name == "schema_analyzer" and "schema_info" in interaction_data:
                    await self._learn_schema_patterns(cursor, interaction_data)
                
                elif agent_name == "nlu_processor" and "entities_extracted" in interaction_data:
                    await self._learn_nlu_patterns(cursor, interaction_data)
                
                # Store in vector store if available
                if self.vector_store is not None:
                    await self._add_to_vector_store(interaction_data)
                
                conn.commit()
    
    async def extract_learning_patterns(self, session_summary: Dict):
        """
        Extracts learning patterns from completed session.
        
        Args:
            session_summary: Complete session summary
        """
        
        if not session_summary.get("success", False):
            return
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                # Extract high-level query patterns
                query = session_summary.get("original_query", "")
                success_rate = 1.0 if session_summary.get("success") else 0.0
                
                # Categorize query type
                query_type = self._categorize_query(query)
                
                # Check if pattern exists
                cursor.execute("""
                    SELECT id, frequency, success_rate 
                    FROM query_patterns 
                    WHERE pattern_type = ?
                """, (query_type,))
                
                result = cursor.fetchone()
                
                if result:
                    # Update existing pattern
                    pattern_id, frequency, current_success_rate = result
                    new_frequency = frequency + 1
                    new_success_rate = (current_success_rate * frequency + success_rate) / new_frequency
                    
                    cursor.execute("""
                        UPDATE query_patterns 
                        SET frequency = ?, success_rate = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (new_frequency, new_success_rate, pattern_id))
                    
                else:
                    # Create new pattern
                    cursor.execute("""
                        INSERT INTO query_patterns (pattern_type, pattern_data, success_rate)
                        VALUES (?, ?, ?)
                    """, (query_type, json.dumps({"query": query}), success_rate))
                
                conn.commit()
    
    async def get_learning_insights(self) -> Dict:
        """
        Provides insights about learning progress and knowledge accumulation.
        
        Returns:
            Learning insights and statistics
        """
        
        async with self._connection_lock:
            with self._get_secure_connection() as conn:
                cursor = conn.cursor()
                
                # Pattern statistics
                cursor.execute("SELECT COUNT(*), AVG(success_rate) FROM query_patterns")
                pattern_count, avg_success_rate = cursor.fetchone()
                
                # Schema insights statistics
                cursor.execute("SELECT COUNT(*), AVG(confidence) FROM schema_insights")
                schema_count, avg_confidence = cursor.fetchone()
                
                # Most effective patterns
                cursor.execute("""
                    SELECT pattern_category, AVG(effectiveness), COUNT(*)
                    FROM learning_patterns 
                    GROUP BY pattern_category
                    ORDER BY AVG(effectiveness) DESC
                """)
                
                category_effectiveness = [
                    {"category": row[0], "effectiveness": row[1], "count": row[2]}
                    for row in cursor.fetchall()
                ]
                
                return {
                    "total_patterns": pattern_count or 0,
                    "average_success_rate": avg_success_rate or 0,
                    "total_schema_insights": schema_count or 0,
                    "average_confidence": avg_confidence or 0,
                    "category_effectiveness": category_effectiveness,
                    "vector_store_size": self.vector_store.ntotal if self.vector_store else 0
                }
    
    async def _learn_sql_patterns(self, cursor, interaction_data: Dict):
        """Learns SQL generation patterns."""
        
        sql_data = {
            "sql": interaction_data.get("generated_sql"),
            "confidence": interaction_data.get("confidence", 0.5),
            "performance": interaction_data.get("performance_metrics", {})
        }
        
        cursor.execute("""
            INSERT INTO learning_patterns (pattern_category, pattern_content, effectiveness)
            VALUES (?, ?, ?)
        """, ("sql_generation", json.dumps(sql_data), interaction_data.get("confidence", 0.5)))
    
    async def _learn_schema_patterns(self, cursor, interaction_data: Dict):
        """Learns schema analysis patterns."""
        
        schema_info = interaction_data.get("schema_info", {})
        if "table_name" in schema_info:
            cursor.execute("""
                INSERT OR REPLACE INTO schema_insights 
                (table_name, insight_type, insight_data, confidence)
                VALUES (?, ?, ?, ?)
            """, (
                schema_info["table_name"],
                "relevance_pattern",
                json.dumps(schema_info),
                interaction_data.get("confidence", 0.5)
            ))
    
    async def _learn_nlu_patterns(self, cursor, interaction_data: Dict):
        """Learns NLU processing patterns."""
        
        nlu_data = {
            "entities": interaction_data.get("entities_extracted", []),
            "intent": interaction_data.get("query_intent", {}),
            "confidence": interaction_data.get("confidence", 0.5)
        }
        
        cursor.execute("""
            INSERT INTO learning_patterns (pattern_category, pattern_content, effectiveness)
            VALUES (?, ?, ?)
        """, ("nlu_processing", json.dumps(nlu_data), interaction_data.get("confidence", 0.5)))
    
    async def _add_to_vector_store(self, interaction_data: Dict):
        """Adds interaction data to vector store for similarity search."""
        
        if not FAISS_AVAILABLE or self.vector_store is None:
            return
        
        # Create text representation of interaction
        text_content = self._create_text_representation(interaction_data)
        
        # Generate embedding
        try:
            async with self._vector_lock:
                embedding = self.embedding_model.encode([text_content])
                
                # Normalize for cosine similarity
                faiss.normalize_L2(embedding)
                
                # Add to FAISS index
                next_index = self.vector_store.ntotal
                self.vector_store.add(embedding)
                
                # Store metadata in database
                with self._get_secure_connection() as conn:
                    cursor = conn.cursor()
                    
                    content_hash = str(hash(text_content))
                    cursor.execute("""
                        INSERT OR REPLACE INTO vector_metadata 
                        (content_type, content_text, content_hash, faiss_index, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        "interaction",
                        text_content,
                        content_hash,
                        next_index,
                        json.dumps(interaction_data)
                    ))
                    
                    # Update mappings
                    vector_id = cursor.lastrowid
                    self.index_to_id[next_index] = vector_id
                    self.id_to_index[vector_id] = next_index
                    
                    conn.commit()
                
                # Save updated index
                self._save_vector_store()
                
        except Exception as e:
            self.logger.error(f"Failed to add to vector store: {str(e)}")
    
    async def _vector_similarity_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Performs vector similarity search."""
        
        if not FAISS_AVAILABLE or self.vector_store is None or self.vector_store.ntotal == 0:
            return []
        
        try:
            async with self._vector_lock:
                # Generate query embedding
                query_embedding = self.embedding_model.encode([query])
                faiss.normalize_L2(query_embedding)
                
                # Search
                scores, indices = self.vector_store.search(query_embedding, min(top_k, self.vector_store.ntotal))
                
                # Retrieve metadata
                results = []
                with self._get_secure_connection() as conn:
                    cursor = conn.cursor()
                    
                    for score, index in zip(scores[0], indices[0]):
                        if index == -1:  # Invalid index
                            continue
                            
                        vector_id = self.index_to_id.get(index)
                        if vector_id:
                            cursor.execute("""
                                SELECT content_text, metadata 
                                FROM vector_metadata 
                                WHERE vector_id = ?
                            """, (vector_id,))
                            
                            row = cursor.fetchone()
                            if row:
                                results.append({
                                    "content": row[0],
                                    "metadata": json.loads(row[1]) if row[1] else {},
                                    "similarity_score": float(score)
                                })
                
                return results
                
        except Exception as e:
            self.logger.error(f"Vector similarity search failed: {str(e)}")
            return []
    
    def _save_vector_store(self):
        """Saves FAISS index and mappings to disk."""
        
        if not FAISS_AVAILABLE or self.vector_store is None:
            return
        
        try:
            index_path = os.path.join(self.vector_path, "knowledge.index")
            mapping_path = os.path.join(self.vector_path, "index_mapping.pkl")
            
            faiss.write_index(self.vector_store, index_path)
            
            with open(mapping_path, 'wb') as f:
                pickle.dump({
                    'index_to_id': self.index_to_id,
                    'id_to_index': self.id_to_index
                }, f)
                
        except Exception as e:
            self.logger.error(f"Failed to save vector store: {str(e)}")
    
    def _extract_table_keywords(self, query: str) -> List[str]:
        """Extracts potential table names from query."""
        # Simple keyword extraction - can be enhanced
        common_table_words = ['table', 'from', 'join', 'sales', 'users', 'orders', 'products']
        words = query.lower().split()
        
        potential_tables = []
        for i, word in enumerate(words):
            if word in ['from', 'join'] and i + 1 < len(words):
                potential_tables.append(words[i + 1])
            elif word in common_table_words:
                potential_tables.append(word)
        
        return list(set(potential_tables))
    
    def _categorize_query(self, query: str) -> str:
        """Categorizes query type for pattern learning."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['sum', 'count', 'avg', 'total']):
            return 'aggregation'
        elif 'join' in query_lower:
            return 'join'
        elif any(word in query_lower for word in ['where', 'filter', 'condition']):
            return 'filtering'
        elif any(word in query_lower for word in ['order by', 'sort', 'rank']):
            return 'sorting'
        elif any(word in query_lower for word in ['group by', 'group']):
            return 'grouping'
        else:
            return 'general'
    
    def _create_text_representation(self, interaction_data: Dict) -> str:
        """Creates text representation of interaction for embedding."""
        
        text_parts = []
        
        if "generated_sql" in interaction_data:
            text_parts.append(f"SQL: {interaction_data['generated_sql']}")
        
        if "query_intent" in interaction_data:
            intent = interaction_data["query_intent"]
            if isinstance(intent, dict):
                text_parts.append(f"Intent: {intent.get('primary_action', '')}")
        
        if "entities_extracted" in interaction_data:
            entities = interaction_data["entities_extracted"]
            if isinstance(entities, list):
                entity_texts = [str(e) for e in entities]
                text_parts.append(f"Entities: {', '.join(entity_texts)}")
        
        return " | ".join(text_parts) if text_parts else "Empty interaction"