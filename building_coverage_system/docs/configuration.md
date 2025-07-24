# Configuration Guide

The Building Coverage System uses a hierarchical configuration system that supports multiple environments and flexible parameter management.

## Configuration Overview

The system supports configuration through:
- YAML configuration files
- Environment variables
- Runtime configuration updates
- Database-stored settings

## Configuration Structure

### Main Configuration File

The primary configuration is stored in `config/config.yaml`:

```yaml
# Application Configuration
app:
  name: "Building Coverage System"
  version: "1.0.0"
  environment: "${APP_ENV:development}"
  debug: false
  log_level: "${LOG_LEVEL:INFO}"

# Database Configuration
database:
  # Primary database (claims data)
  primary:
    enabled: true
    server: "${DATABASE_HOST:localhost}"
    database: "${DATABASE_NAME:building_coverage}"
    username: "${DATABASE_USER:postgres}"
    password: "${DATABASE_PASSWORD}"
    driver: "ODBC Driver 17 for SQL Server"
    connection_timeout: 30
    query_timeout: 300
    pool_size: 10
    max_overflow: 20
    
  # AIP database
  aip:
    enabled: "${AIP_ENABLED:false}"
    server: "${AIP_SERVER}"
    database: "${AIP_DATABASE}"
    username: "${AIP_USERNAME}"
    password: "${AIP_PASSWORD}"
    connection_timeout: 45
    query_timeout: 600
    
  # Atlas database
  atlas:
    enabled: "${ATLAS_ENABLED:false}"
    server: "${ATLAS_SERVER}"
    database: "${ATLAS_DATABASE}"
    username: "${ATLAS_USERNAME}"
    password: "${ATLAS_PASSWORD}"
    connection_timeout: 30
    query_timeout: 300

# Embedding Model Configuration
embedding_model:
  model_name: "${EMBEDDING_MODEL:all-MiniLM-L6-v2}"
  model_path: "${MODEL_PATH:/app/models}"
  max_sequence_length: 512
  batch_size: 32
  device: "${DEVICE:cpu}"
  cache_embeddings: true
  embedding_dimension: 384

# Processing Configuration
processing:
  batch_size: 1000
  max_workers: "${MAX_WORKERS:4}"
  chunk_size: 500
  timeout: 300
  parallel_processing: true
  memory_limit_mb: 2048

# Classification Thresholds
classification_thresholds:
  building_coverage: 0.85
  personal_property: 0.80
  liability: 0.90
  other_structures: 0.75
  medical_payments: 0.85
  loss_of_use: 0.80

# Business Rules Configuration
business_rules:
  enabled: true
  rule_cache_ttl: 3600
  max_rules_per_evaluation: 50
  parallel_rule_evaluation: true
  rule_priority_enabled: true

# Cache Configuration
cache:
  redis:
    url: "${REDIS_URL:redis://localhost:6379}"
    password: "${REDIS_PASSWORD:}"
    db: 0
    max_connections: 20
    socket_timeout: 5
    socket_connect_timeout: 5
    retry_on_timeout: true
  
  # Cache TTL settings (in seconds)
  ttl:
    embeddings: 86400  # 24 hours
    similarity_results: 3600  # 1 hour
    classification_results: 7200  # 2 hours
    rule_results: 1800  # 30 minutes
    configuration: 300  # 5 minutes

# Security Configuration
security:
  encryption:
    algorithm: "fernet"
    key_derivation: "pbkdf2"
    iterations: 100000
    master_key: "${ENCRYPTION_KEY}"
  
  tokenization:
    enabled: true
    preserve_format: true
    token_prefix: "TOK_"
    
  authentication:
    jwt_secret: "${JWT_SECRET}"
    jwt_expiration: 3600  # 1 hour
    refresh_token_expiration: 604800  # 7 days

# Monitoring Configuration
monitoring:
  metrics_enabled: true
  health_check_interval: 30
  performance_tracking: true
  log_retention_days: 30
  
  prometheus:
    enabled: "${PROMETHEUS_ENABLED:true}"
    port: 9090
    metrics_path: "/metrics"
  
  logging:
    level: "${LOG_LEVEL:INFO}"
    format: "json"
    file_path: "/app/logs/application.log"
    max_file_size_mb: 100
    backup_count: 5

# API Configuration
api:
  host: "${API_HOST:0.0.0.0}"
  port: "${API_PORT:8000}"
  workers: "${API_WORKERS:4}"
  max_request_size: 52428800  # 50MB
  timeout: 60
  
  rate_limiting:
    enabled: true
    default_limit: "100/minute"
    premium_limit: "500/minute"
    burst_limit: 10
  
  cors:
    enabled: true
    allow_origins: ["*"]
    allow_methods: ["GET", "POST", "PUT", "DELETE"]
    allow_headers: ["*"]

# Celery Configuration (for background tasks)
celery:
  broker_url: "${CELERY_BROKER_URL:redis://localhost:6379/1}"
  result_backend: "${CELERY_RESULT_BACKEND:redis://localhost:6379/2}"
  task_serializer: "json"
  accept_content: ["json"]
  result_serializer: "json"
  timezone: "UTC"
  
  worker:
    concurrency: 4
    max_tasks_per_child: 1000
    task_soft_time_limit: 300
    task_time_limit: 600
```

## Environment Variables

Key environment variables used by the system:

### Database Configuration
```bash
# Primary Database
DATABASE_HOST=your-sql-server.com
DATABASE_NAME=building_coverage_db
DATABASE_USER=app_user
DATABASE_PASSWORD=secure_password

# AIP Database (optional)
AIP_ENABLED=true
AIP_SERVER=aip-server.com
AIP_DATABASE=aip_claims_db
AIP_USERNAME=aip_user
AIP_PASSWORD=aip_password

# Atlas Database (optional)
ATLAS_ENABLED=false
ATLAS_SERVER=atlas-server.com
ATLAS_DATABASE=atlas_db
ATLAS_USERNAME=atlas_user
ATLAS_PASSWORD=atlas_password
```

### Security Configuration
```bash
# Encryption
ENCRYPTION_KEY=your-base64-encoded-encryption-key

# JWT Authentication
JWT_SECRET=your-jwt-secret-key

# Redis (if using external Redis)
REDIS_URL=redis://redis-server:6379
REDIS_PASSWORD=redis_password
```

### Application Configuration
```bash
# Environment
APP_ENV=production
LOG_LEVEL=INFO

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Processing
MAX_WORKERS=8
EMBEDDING_MODEL=all-MiniLM-L6-v2
DEVICE=cuda  # or cpu
MODEL_PATH=/app/models

# Monitoring
PROMETHEUS_ENABLED=true
```

## Environment-Specific Configuration

### Development Environment

Create `config/development.yaml`:

```yaml
app:
  debug: true
  log_level: "DEBUG"

database:
  primary:
    server: "localhost"
    database: "building_coverage_dev"
    pool_size: 5
  
processing:
  batch_size: 100
  max_workers: 2

cache:
  ttl:
    embeddings: 300  # Shorter TTL for development

monitoring:
  log_retention_days: 7
```

### Production Environment

Create `config/production.yaml`:

```yaml
app:
  debug: false
  log_level: "INFO"

database:
  primary:
    pool_size: 20
    max_overflow: 50
    connection_timeout: 60
    query_timeout: 600

processing:
  batch_size: 5000
  max_workers: 16
  memory_limit_mb: 8192

security:
  tokenization:
    enabled: true
  authentication:
    jwt_expiration: 1800  # Shorter expiration in production

monitoring:
  metrics_enabled: true
  performance_tracking: true
  log_retention_days: 90

api:
  workers: 8
  rate_limiting:
    enabled: true
    default_limit: "200/minute"
```

## Configuration Loading

### Programmatic Configuration

```python
from building_coverage_system.coverage_configs import ConfigManager

# Load configuration
config_manager = ConfigManager()
config_manager.load_config('config/config.yaml')

# Override with environment-specific config
if config_manager.get_app_config().environment == 'production':
    config_manager.load_config('config/production.yaml', merge=True)

# Get specific configurations
db_config = config_manager.get_database_config('primary')
model_config = config_manager.get_model_config()
processing_config = config_manager.get_processing_config()
```

### Runtime Configuration Updates

```python
# Update configuration at runtime
config_manager.update_config_section('processing', {
    'batch_size': 2000,
    'max_workers': 8
})

# Update thresholds
config_manager.update_thresholds({
    'building_coverage': 0.90,
    'personal_property': 0.85
})
```

## Configuration Validation

The system automatically validates configuration on startup:

```python
# Validate configuration
is_valid, errors = config_manager.validate_config()

if not is_valid:
    for error in errors:
        print(f"Configuration error: {error}")
    exit(1)
```

### Required Configuration Fields

The following fields are required for system operation:

#### Database Configuration
- `database.primary.server`
- `database.primary.database`
- Authentication credentials (username/password or trusted connection)

#### Security Configuration
- `security.encryption.master_key` (for data encryption)
- `security.authentication.jwt_secret` (for API authentication)

#### Model Configuration
- `embedding_model.model_name`
- `embedding_model.model_path`

## Advanced Configuration

### Custom Configuration Sources

You can implement custom configuration sources:

```python
from building_coverage_system.coverage_configs import ConfigSource

class DatabaseConfigSource(ConfigSource):
    """Load configuration from database."""
    
    def load_config(self) -> dict:
        # Implementation to load from database
        pass

# Register custom source
config_manager.add_config_source(DatabaseConfigSource())
```

### Configuration Encryption

Sensitive configuration values can be encrypted:

```python
from building_coverage_system.utils import CryptoManager

crypto = CryptoManager()

# Encrypt sensitive values
encrypted_password = crypto.encrypt_string("sensitive_password")

# Use in configuration
config = {
    "database": {
        "password": encrypted_password,
        "_encrypted_fields": ["password"]
    }
}
```

### Dynamic Configuration Updates

The system supports hot-reloading of certain configuration sections:

```python
# Monitor configuration file for changes
config_manager.enable_auto_reload(
    watch_files=['config/config.yaml'],
    reload_sections=['processing', 'classification_thresholds']
)
```

## Configuration Best Practices

### 1. Environment Variables for Secrets
Always use environment variables for sensitive information:

```yaml
database:
  password: "${DATABASE_PASSWORD}"  # Good
  # password: "hardcoded_password"  # Bad
```

### 2. Separate Configuration by Purpose
Organize configuration into logical sections:

```yaml
# Good: Well-organized sections
database:
  primary: {...}
  secondary: {...}

processing:
  batch_settings: {...}
  worker_settings: {...}

# Bad: Flat structure
primary_db_host: "..."
batch_size: 1000
primary_db_port: 5432
max_workers: 4
```

### 3. Use Defaults Appropriately
Provide sensible defaults for optional settings:

```yaml
processing:
  batch_size: "${BATCH_SIZE:1000}"  # Default to 1000 if not set
  timeout: "${TIMEOUT:300}"         # Default to 5 minutes
```

### 4. Document Configuration Options
Include comments explaining configuration options:

```yaml
processing:
  # Number of claims to process in each batch
  # Higher values improve throughput but use more memory
  batch_size: 1000
  
  # Maximum number of parallel worker processes
  # Should not exceed CPU core count
  max_workers: 4
```

### 5. Validate Configuration
Always validate configuration values:

```python
def validate_batch_size(value):
    if not isinstance(value, int) or value <= 0:
        raise ValueError("batch_size must be a positive integer")
    if value > 10000:
        raise ValueError("batch_size cannot exceed 10000")
    return value
```

## Troubleshooting Configuration

### Common Issues

1. **Missing Environment Variables**
   ```
   Error: Environment variable 'DATABASE_PASSWORD' is not set
   Solution: Set the required environment variable
   ```

2. **Invalid Configuration Format**
   ```
   Error: Invalid YAML syntax in config file
   Solution: Validate YAML syntax using a YAML validator
   ```

3. **Database Connection Failures**
   ```
   Error: Cannot connect to database
   Solution: Verify database configuration and network connectivity
   ```

4. **Model Loading Failures**
   ```
   Error: Cannot load embedding model
   Solution: Verify model path and ensure model files are present
   ```

### Configuration Debugging

Enable debug logging to troubleshoot configuration issues:

```bash
export LOG_LEVEL=DEBUG
python -m building_coverage_system.main
```

Check configuration loading:

```python
from building_coverage_system.coverage_configs import ConfigManager

config_manager = ConfigManager()
config_manager.load_config('config/config.yaml')

# Print loaded configuration (excluding secrets)
config_manager.print_config(hide_secrets=True)
```