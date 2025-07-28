# Agentic Building Coverage Analysis

## Overview
AI-powered building coverage analysis system using GPT-4o-mini and FAISS vector memory for intelligent claim processing.

## Features
- **1+3 Agent Architecture**: Single extraction agent + 3 specialized financial reasoning agents
- **GPT-4o-mini Integration**: Cost-effective AI processing with JSON mode
- **FAISS Vector Memory**: Semantic similarity matching without database dependencies
- **Complete Keyword Preservation**: All 21 building indicators + hierarchical loss amounts
- **Memory Learning**: Pattern recognition and confidence calibration

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Test the system
python test_faiss_integration.py

# Run main implementation
python complete_agentic_implementation.py
```

## Architecture
- **Stage 1**: Unified extraction agent (21 building indicators + monetary candidates)
- **Stage 2**: Context analysis → Calculation → Validation & reflection
- **Memory**: FAISS vector similarity for pattern learning
- **API**: GPT-4o-mini with JSON mode for structured extraction

## Files
- `complete_agentic_implementation.py` - Main agentic system
- `faiss_memory_store.py` - Vector-based memory management
- `gpt_api_wrapper.py` - GPT-4o-mini API wrapper
- `config.py` - System configuration
- `test_faiss_integration.py` - Test suite

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Update requirements.txt to add FAISS and remove database dependencies", "status": "completed", "priority": "high"}, {"id": "2", "content": "Create FAISS-based memory store to replace SQLite", "status": "completed", "priority": "high"}, {"id": "3", "content": "Update main implementation to use FAISS memory store", "status": "completed", "priority": "high"}, {"id": "4", "content": "Create test script for FAISS integration", "status": "completed", "priority": "medium"}, {"id": "5", "content": "Update documentation for FAISS migration", "status": "completed", "priority": "medium"}]