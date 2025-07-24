# Development Guide

This guide provides comprehensive information for developers working on the Building Coverage System, including setup, architecture, coding standards, and contribution guidelines.

## Development Environment Setup

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- Git
- PostgreSQL (for local development)
- Redis (for caching)

### Local Development Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/company/building-coverage-system.git
   cd building-coverage-system
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Set Up Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

5. **Start Local Services**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

6. **Run Database Migrations**
   ```bash
   python -m alembic upgrade head
   ```

7. **Start the Application**
   ```bash
   python -m building_coverage_system.main
   ```

### Docker Development Environment

For a fully containerized development environment:

```bash
# Build and start development containers
docker-compose -f docker-compose.dev.yml up --build

# Run tests in container
docker-compose -f docker-compose.dev.yml exec app pytest

# Access application shell
docker-compose -f docker-compose.dev.yml exec app bash
```

## Project Structure

### High-Level Architecture

```
building_coverage_system/
├── building_coverage_system/           # Main application package
│   ├── new_architecture/               # Modern RAG-enhanced components
│   │   ├── core/                      # Core pipeline orchestration
│   │   ├── document_processing/       # Document analysis
│   │   ├── embedding/                 # Text embedding service
│   │   ├── search/                    # Vector search engine
│   │   └── classification/            # ML classification
│   ├── coverage_configs/              # Configuration management
│   ├── coverage_rag_implementation/   # RAG text analysis
│   ├── coverage_rules/                # Business rules engine
│   ├── coverage_sql_pipelines/        # SQL data processing
│   ├── utils/                         # Utility functions
│   └── api/                          # REST API endpoints
├── tests/                             # Test suite
├── docs/                             # Documentation
├── deploy/                           # Deployment configurations
├── notebooks/                        # Jupyter notebooks
└── config/                          # Configuration files
```

### Code Organization Principles

1. **Modular Design**: Each component is self-contained with clear interfaces
2. **Separation of Concerns**: Business logic, data access, and API layers are separate
3. **Dependency Injection**: Use dependency injection for testability
4. **Factory Pattern**: Factory functions for creating configured instances
5. **Configuration-Driven**: Behavior controlled through configuration

## Development Workflow

### 1. Feature Development Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/coverage-enhancement
   ```

2. **Implement Feature**
   - Write code following coding standards
   - Add comprehensive tests
   - Update documentation

3. **Run Quality Checks**
   ```bash
   # Run tests
   pytest tests/
   
   # Run linting
   flake8 building_coverage_system/
   
   # Run type checking
   mypy building_coverage_system/
   
   # Run security checks
   bandit -r building_coverage_system/
   ```

4. **Submit Pull Request**
   - Create detailed PR description
   - Include test results and performance impact
   - Request code review

### 2. Testing Strategy

#### Unit Tests

Write unit tests for all business logic:

```python
# tests/test_coverage_classifier.py
import pytest
from unittest.mock import Mock, patch
from building_coverage_system.new_architecture.classification import CoverageClassifier

class TestCoverageClassifier:
    def setup_method(self):
        self.classifier = CoverageClassifier()
    
    def test_classify_dwelling_coverage(self):
        claim_text = "Fire damage to kitchen walls and ceiling"
        
        result = self.classifier.classify(claim_text)
        
        assert result.primary_coverage == "DWELLING_COVERAGE_A"
        assert result.confidence_score > 0.85
    
    @patch('building_coverage_system.new_architecture.classification.ModelManager')
    def test_model_loading_failure(self, mock_model_manager):
        mock_model_manager.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception):
            CoverageClassifier()
```

#### Integration Tests

Test component interactions:

```python
# tests/test_integration.py
import pytest
from building_coverage_system.new_architecture.core import PipelineOrchestrator
from building_coverage_system.coverage_configs import ConfigManager

class TestPipelineIntegration:
    def setup_method(self):
        config = ConfigManager()
        config.load_config('tests/fixtures/test_config.yaml')
        self.orchestrator = PipelineOrchestrator(config)
    
    def test_end_to_end_processing(self):
        claim_data = {
            'claim_id': 'TEST001',
            'claim_text': 'Water damage from burst pipe',
            'loss_date': '2023-11-15'
        }
        
        result = self.orchestrator.process_claim(claim_data)
        
        assert result.claim_id == 'TEST001'
        assert result.coverage_determination is not None
        assert result.processing_time_ms > 0
```

#### Performance Tests

Monitor system performance:

```python
# tests/test_performance.py
import time
import pytest
from building_coverage_system.new_architecture.embedding import EmbeddingGenerator

class TestPerformance:
    def setup_method(self):
        self.embedding_generator = EmbeddingGenerator()
    
    def test_embedding_generation_speed(self):
        texts = ["Sample claim text"] * 100
        
        start_time = time.time()
        embeddings = self.embedding_generator.generate_batch(texts)
        end_time = time.time()
        
        processing_time = end_time - start_time
        texts_per_second = len(texts) / processing_time
        
        assert texts_per_second > 50  # Minimum performance requirement
        assert len(embeddings) == len(texts)
```

### 3. Code Review Guidelines

#### What to Look For

1. **Code Quality**
   - Follows coding standards and conventions
   - Proper error handling and logging
   - Appropriate comments and documentation

2. **Architecture**
   - Follows established patterns
   - Proper separation of concerns
   - Appropriate use of abstractions

3. **Testing**
   - Adequate test coverage (>80%)
   - Tests cover edge cases
   - Integration tests for critical paths

4. **Performance**
   - No obvious performance bottlenecks
   - Efficient algorithms and data structures
   - Proper resource management

5. **Security**
   - No hardcoded secrets or credentials
   - Proper input validation
   - Secure data handling

## Coding Standards

### Python Style Guide

Follow PEP 8 with these specific guidelines:

#### 1. Code Formatting

```python
# Good: Clear function with proper typing
def analyze_claim_coverage(
    claim_text: str,
    claim_metadata: Dict[str, Any],
    confidence_threshold: float = 0.85
) -> CoverageAnalysisResult:
    """
    Analyze claim text to determine coverage type.
    
    Args:
        claim_text: The claim description text
        claim_metadata: Additional claim information
        confidence_threshold: Minimum confidence for classification
        
    Returns:
        CoverageAnalysisResult with classification and confidence
        
    Raises:
        ValueError: If claim_text is empty or invalid
    """
    if not claim_text or not claim_text.strip():
        raise ValueError("claim_text cannot be empty")
    
    # Implementation here
    pass

# Bad: Poor formatting and no documentation
def analyze(txt,meta,thresh=0.85):
    if not txt:return None
    # implementation
```

#### 2. Error Handling

```python
# Good: Specific exception handling with logging
import logging

logger = logging.getLogger(__name__)

def process_claim_data(claim_data: Dict[str, Any]) -> ProcessingResult:
    try:
        validated_data = validate_claim_data(claim_data)
        result = perform_analysis(validated_data)
        return result
        
    except ValidationError as e:
        logger.error(f"Claim data validation failed: {e}")
        raise
    except ModelError as e:
        logger.error(f"Model processing failed: {e}")
        raise ProcessingError(f"Unable to process claim: {e}") from e
    except Exception as e:
        logger.exception("Unexpected error during claim processing")
        raise ProcessingError("Internal processing error") from e

# Bad: Generic exception handling
def process_claim_data(claim_data):
    try:
        # processing logic
        pass
    except:
        return None
```

#### 3. Configuration and Dependencies

```python
# Good: Dependency injection with clear interfaces
from abc import ABC, abstractmethod
from typing import Protocol

class EmbeddingService(Protocol):
    def generate_embedding(self, text: str) -> List[float]:
        ...

class CoverageClassifier:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        config: ClassificationConfig
    ):
        self.embedding_service = embedding_service
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def classify(self, text: str) -> ClassificationResult:
        embedding = self.embedding_service.generate_embedding(text)
        # Classification logic
        pass

# Bad: Hard dependencies and global state
class CoverageClassifier:
    def __init__(self):
        self.model = load_model("/hardcoded/path/model.pkl")
        global_config = get_global_config()
        self.threshold = global_config["threshold"]
```

### Documentation Standards

#### 1. Module Documentation

```python
"""
Coverage classification module.

This module provides functionality for classifying insurance claims
into appropriate coverage categories using machine learning models
and business rules.

Classes:
    CoverageClassifier: Main classification engine
    ClassificationResult: Result container
    ClassificationError: Custom exception for classification errors

Functions:
    create_classifier: Factory function for creating classifier instances
    
Example:
    classifier = create_classifier(config)
    result = classifier.classify("Fire damage to kitchen")
    print(f"Coverage: {result.primary_coverage}")
"""
```

#### 2. Class Documentation

```python
class CoverageClassifier:
    """
    Machine learning-based coverage classifier.
    
    This class combines ML models with business rules to determine
    the most appropriate coverage type for insurance claims.
    
    Attributes:
        model: The trained classification model
        rules_engine: Business rules engine for validation
        config: Classification configuration
        
    Example:
        classifier = CoverageClassifier(model, rules_engine, config)
        result = classifier.classify("Water damage in basement")
    """
```

#### 3. Function Documentation

```python
def classify_coverage(
    self,
    claim_text: str,
    claim_metadata: Optional[Dict[str, Any]] = None
) -> ClassificationResult:
    """
    Classify a claim into appropriate coverage categories.
    
    Uses a combination of ML models and business rules to determine
    the most likely coverage type for the given claim.
    
    Args:
        claim_text: The description of the claim damage
        claim_metadata: Optional additional claim information including
            loss_date, lob_code, policy_limits, etc.
            
    Returns:
        ClassificationResult containing:
            - primary_coverage: Most likely coverage type
            - confidence_score: Confidence level (0.0-1.0)
            - alternative_coverages: Other possible coverage types
            - rule_matches: Business rules that matched
            
    Raises:
        ValueError: If claim_text is empty or invalid
        ModelError: If the ML model fails to process the claim
        
    Example:
        result = classifier.classify_coverage(
            "Fire damage to kitchen cabinets",
            {"loss_date": "2023-11-15", "lob_code": "HO"}
        )
        print(f"Coverage: {result.primary_coverage}")
        print(f"Confidence: {result.confidence_score:.2%}")
    """
```

## Architecture Patterns

### 1. Factory Pattern for Configuration

```python
# config_factory.py
from typing import Dict, Any
from building_coverage_system.coverage_configs import ConfigManager

def create_database_config(environment: str) -> DatabaseConfig:
    """Create database configuration for specified environment."""
    config_manager = ConfigManager()
    config_manager.load_config(f'config/{environment}.yaml')
    return config_manager.get_database_config('primary')

def create_classifier(config: Dict[str, Any]) -> CoverageClassifier:
    """Create configured coverage classifier."""
    embedding_service = create_embedding_service(config['embedding'])
    rules_engine = create_rules_engine(config['rules'])
    
    return CoverageClassifier(
        embedding_service=embedding_service,
        rules_engine=rules_engine,
        config=ClassificationConfig.from_dict(config['classification'])
    )
```

### 2. Repository Pattern for Data Access

```python
# repositories.py
from abc import ABC, abstractmethod
from typing import List, Optional

class ClaimRepository(ABC):
    """Abstract repository for claim data access."""
    
    @abstractmethod
    def get_claim_by_id(self, claim_id: str) -> Optional[Claim]:
        """Retrieve claim by ID."""
        pass
    
    @abstractmethod
    def search_similar_claims(
        self,
        embedding: List[float],
        limit: int = 10
    ) -> List[SimilarClaim]:
        """Find similar claims using vector search."""
        pass

class SqlClaimRepository(ClaimRepository):
    """SQL-based claim repository implementation."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
    
    def get_claim_by_id(self, claim_id: str) -> Optional[Claim]:
        query = "SELECT * FROM claims WHERE claim_id = %s"
        result = self.db.execute(query, (claim_id,))
        
        if result:
            return Claim.from_db_row(result[0])
        return None
```

### 3. Service Layer Pattern

```python
# services.py
class ClaimAnalysisService:
    """High-level service for claim analysis operations."""
    
    def __init__(
        self,
        classifier: CoverageClassifier,
        claim_repository: ClaimRepository,
        similarity_service: SimilarityService,
        rules_engine: BusinessRulesEngine
    ):
        self.classifier = classifier
        self.claim_repository = claim_repository
        self.similarity_service = similarity_service
        self.rules_engine = rules_engine
        self.logger = logging.getLogger(__name__)
    
    async def analyze_claim(
        self,
        claim_data: ClaimData
    ) -> ClaimAnalysisResult:
        """Perform comprehensive claim analysis."""
        self.logger.info(f"Starting analysis for claim {claim_data.claim_id}")
        
        try:
            # Step 1: Classify coverage
            classification = await self.classifier.classify_async(claim_data.text)
            
            # Step 2: Apply business rules
            rule_results = self.rules_engine.evaluate_rules(claim_data)
            
            # Step 3: Find similar claims
            similar_claims = await self.similarity_service.find_similar(
                claim_data.text,
                limit=5
            )
            
            # Step 4: Combine results
            result = ClaimAnalysisResult(
                claim_id=claim_data.claim_id,
                classification=classification,
                rule_results=rule_results,
                similar_claims=similar_claims,
                processing_metadata=self._create_metadata()
            )
            
            self.logger.info(
                f"Analysis completed for claim {claim_data.claim_id}: "
                f"{classification.primary_coverage} ({classification.confidence_score:.2%})"
            )
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Analysis failed for claim {claim_data.claim_id}")
            raise AnalysisError(f"Failed to analyze claim: {e}") from e
```

## Performance Optimization

### 1. Async Processing

```python
import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor

class AsyncCoverageClassifier:
    """Async version of coverage classifier for better throughput."""
    
    def __init__(self, classifier: CoverageClassifier, max_workers: int = 4):
        self.classifier = classifier
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def classify_batch(
        self,
        claims: List[ClaimData]
    ) -> List[ClassificationResult]:
        """Classify multiple claims concurrently."""
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(
                self.executor,
                self.classifier.classify,
                claim.text
            )
            for claim in claims
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Classification failed for claim {claims[i].claim_id}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
```

### 2. Caching Strategies

```python
from functools import lru_cache
import redis
import pickle
from typing import Optional

class CachedEmbeddingService:
    """Embedding service with Redis caching."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator, redis_client: redis.Redis):
        self.embedding_generator = embedding_generator
        self.redis = redis_client
        self.cache_ttl = 86400  # 24 hours
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with caching."""
        cache_key = f"embedding:{hash(text)}"
        
        # Try cache first
        cached_embedding = self._get_from_cache(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate new embedding
        embedding = self.embedding_generator.generate(text)
        
        # Cache the result
        self._set_cache(cache_key, embedding)
        
        return embedding
    
    def _get_from_cache(self, key: str) -> Optional[List[float]]:
        try:
            cached_data = self.redis.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            self.logger.warning(f"Cache read failed: {e}")
        return None
    
    def _set_cache(self, key: str, embedding: List[float]) -> None:
        try:
            self.redis.setex(key, self.cache_ttl, pickle.dumps(embedding))
        except Exception as e:
            self.logger.warning(f"Cache write failed: {e}")
```

## Debugging and Monitoring

### 1. Logging Best Practices

```python
import logging
import structlog
from building_coverage_system.utils import get_request_id

# Configure structured logging
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

class CoverageClassifier:
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    def classify(self, claim_text: str) -> ClassificationResult:
        request_id = get_request_id()
        
        self.logger.info(
            "Starting claim classification",
            request_id=request_id,
            text_length=len(claim_text),
            claim_preview=claim_text[:100]
        )
        
        start_time = time.time()
        
        try:
            result = self._perform_classification(claim_text)
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                "Classification completed",
                request_id=request_id,
                primary_coverage=result.primary_coverage,
                confidence=result.confidence_score,
                processing_time_ms=processing_time * 1000
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.logger.error(
                "Classification failed",
                request_id=request_id,
                error=str(e),
                processing_time_ms=processing_time * 1000,
                exc_info=True
            )
            raise
```

### 2. Performance Monitoring

```python
import time
from contextlib import contextmanager
from prometheus_client import Counter, Histogram, Gauge

# Metrics
CLASSIFICATION_REQUESTS = Counter('coverage_classification_requests_total', 'Total classification requests')
CLASSIFICATION_DURATION = Histogram('coverage_classification_duration_seconds', 'Classification duration')
CLASSIFICATION_ERRORS = Counter('coverage_classification_errors_total', 'Classification errors')
ACTIVE_REQUESTS = Gauge('coverage_classification_active_requests', 'Active classification requests')

@contextmanager
def monitor_classification():
    """Context manager for monitoring classification performance."""
    CLASSIFICATION_REQUESTS.inc()
    ACTIVE_REQUESTS.inc()
    
    start_time = time.time()
    
    try:
        yield
    except Exception:
        CLASSIFICATION_ERRORS.inc()
        raise
    finally:
        duration = time.time() - start_time
        CLASSIFICATION_DURATION.observe(duration)
        ACTIVE_REQUESTS.dec()

class MonitoredCoverageClassifier:
    def classify(self, claim_text: str) -> ClassificationResult:
        with monitor_classification():
            return self._perform_classification(claim_text)
```

## Contributing Guidelines

### 1. Pull Request Process

1. **Branch Naming**
   - Feature: `feature/description-of-feature`
   - Bug fix: `fix/description-of-bug`
   - Documentation: `docs/description-of-changes`

2. **Commit Messages**
   ```
   feat(classification): add support for multi-coverage claims
   
   - Implement multi-label classification capability
   - Add confidence thresholds for secondary coverages
   - Update API response format to include all applicable coverages
   
   Closes #123
   ```

3. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Performance tests run
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No new warnings or errors
   ```

### 2. Code Quality Gates

All code must pass these quality gates:

```bash
# Run all quality checks
make quality-check

# Individual checks
pytest tests/ --cov=building_coverage_system --cov-report=html
flake8 building_coverage_system/
mypy building_coverage_system/
bandit -r building_coverage_system/
safety check
```

### 3. Release Process

1. **Version Bumping**
   ```bash
   # Update version in setup.py and __init__.py
   git tag v1.2.3
   git push origin v1.2.3
   ```

2. **Release Notes**
   Document all changes in CHANGELOG.md following Keep a Changelog format

3. **Deployment**
   Automated deployment through CI/CD pipeline

This development guide provides the foundation for contributing to the Building Coverage System. For questions or clarifications, please reach out to the development team or create an issue in the repository.