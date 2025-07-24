# API Reference

The Building Coverage System provides a comprehensive REST API for claim processing, coverage analysis, and system management.

## Base URL

```
Production: https://api.building-coverage.company.com
Staging: https://api-staging.building-coverage.company.com
Development: http://localhost:8000
```

## Authentication

All API endpoints require authentication using JWT tokens.

### Get Authentication Token

```http
POST /auth/token
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using the Token

Include the token in the Authorization header:

```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

## Core API Endpoints

### Health Check

#### Get System Health
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T10:00:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "embedding_service": "healthy"
  }
}
```

#### Get Readiness Status
```http
GET /ready
```

**Response:**
```json
{
  "ready": true,
  "services": {
    "models_loaded": true,
    "database_connected": true,
    "cache_available": true
  }
}
```

## Claim Processing API

### Process Single Claim

#### Analyze Claim for Coverage
```http
POST /api/v1/claims/analyze
Content-Type: application/json
Authorization: Bearer {token}

{
  "claim_id": "CLM123456789",
  "claim_text": "Fire damage to kitchen caused by electrical short circuit. Smoke damage throughout first floor.",
  "loss_date": "2023-11-15",
  "policy_number": "POL987654321",
  "lob_code": "HO",
  "loss_amount": 15000.00
}
```

**Response:**
```json
{
  "claim_id": "CLM123456789",
  "processing_id": "proc_abc123def456",
  "results": {
    "coverage_determination": {
      "primary_coverage": "DWELLING_COVERAGE_A",
      "confidence_score": 0.92,
      "applicable_coverages": [
        {
          "coverage_type": "DWELLING_COVERAGE_A",
          "confidence": 0.92,
          "reasoning": "Fire damage to structure (kitchen) is covered under dwelling coverage"
        },
        {
          "coverage_type": "PERSONAL_PROPERTY_COVERAGE_C",
          "confidence": 0.78,
          "reasoning": "Smoke damage may affect personal property"
        }
      ]
    },
    "business_rules_results": {
      "rules_matched": [
        {
          "rule_id": "RULE_001",
          "rule_name": "Fire Damage Coverage Rule",
          "matched": true,
          "action": "APPROVE_DWELLING_COVERAGE"
        }
      ],
      "exceptions": []
    },
    "similar_claims": [
      {
        "claim_id": "CLM987654321",
        "similarity_score": 0.87,
        "coverage_decision": "DWELLING_COVERAGE_A",
        "summary": "Kitchen fire with smoke damage"
      }
    ]
  },
  "processing_time_ms": 1250,
  "timestamp": "2023-12-01T10:15:30Z"
}
```

### Batch Processing

#### Submit Batch Job
```http
POST /api/v1/claims/batch
Content-Type: application/json
Authorization: Bearer {token}

{
  "batch_name": "December_Claims_Batch_001",
  "claims": [
    {
      "claim_id": "CLM001",
      "claim_text": "Water damage from burst pipe...",
      "loss_date": "2023-11-20",
      "policy_number": "POL001",
      "lob_code": "HO"
    },
    {
      "claim_id": "CLM002",
      "claim_text": "Hail damage to roof...",
      "loss_date": "2023-11-22",
      "policy_number": "POL002",
      "lob_code": "HO"
    }
  ],
  "processing_options": {
    "include_similar_claims": true,
    "confidence_threshold": 0.85,
    "max_parallel_processing": 10
  }
}
```

**Response:**
```json
{
  "batch_id": "batch_xyz789abc123",
  "status": "submitted",
  "total_claims": 2,
  "estimated_completion_time": "2023-12-01T10:20:00Z",
  "tracking_url": "/api/v1/batches/batch_xyz789abc123/status"
}
```

#### Get Batch Status
```http
GET /api/v1/batches/{batch_id}/status
Authorization: Bearer {token}
```

**Response:**
```json
{
  "batch_id": "batch_xyz789abc123",
  "status": "processing",
  "progress": {
    "total_claims": 2,
    "processed_claims": 1,
    "failed_claims": 0,
    "percentage_complete": 50
  },
  "started_at": "2023-12-01T10:10:00Z",
  "estimated_completion": "2023-12-01T10:20:00Z",
  "results_available": false
}
```

#### Get Batch Results
```http
GET /api/v1/batches/{batch_id}/results
Authorization: Bearer {token}
```

**Response:**
```json
{
  "batch_id": "batch_xyz789abc123",
  "status": "completed",
  "summary": {
    "total_claims": 2,
    "successful_processing": 2,
    "failed_processing": 0,
    "average_confidence": 0.89,
    "processing_time_total_ms": 2340
  },
  "results": [
    {
      "claim_id": "CLM001",
      "status": "success",
      "coverage_determination": {
        "primary_coverage": "DWELLING_COVERAGE_A",
        "confidence_score": 0.91
      }
    },
    {
      "claim_id": "CLM002",
      "status": "success",
      "coverage_determination": {
        "primary_coverage": "OTHER_STRUCTURES_COVERAGE_B",
        "confidence_score": 0.87
      }
    }
  ]
}
```

## Search and Similarity API

### Similar Claims Search

#### Find Similar Claims
```http
POST /api/v1/search/similar-claims
Content-Type: application/json
Authorization: Bearer {token}

{
  "query_text": "Fire damage to kitchen and smoke throughout house",
  "filters": {
    "lob_codes": ["HO", "CO"],
    "date_range": {
      "start": "2020-01-01",
      "end": "2023-12-01"
    },
    "min_similarity": 0.7
  },
  "limit": 10
}
```

**Response:**
```json
{
  "query": "Fire damage to kitchen and smoke throughout house",
  "total_results": 25,
  "returned_results": 10,
  "results": [
    {
      "claim_id": "CLM987654321",
      "similarity_score": 0.94,
      "claim_summary": "Kitchen fire with extensive smoke damage",
      "coverage_decision": "DWELLING_COVERAGE_A",
      "loss_amount": 18500.00,
      "loss_date": "2023-10-15"
    },
    {
      "claim_id": "CLM876543210",
      "similarity_score": 0.89,
      "claim_summary": "Electrical fire in kitchen, smoke damage to living areas",
      "coverage_decision": "DWELLING_COVERAGE_A",
      "loss_amount": 22000.00,
      "loss_date": "2023-09-22"
    }
  ],
  "processing_time_ms": 850
}
```

### Vector Search

#### Semantic Search
```http
POST /api/v1/search/semantic
Content-Type: application/json
Authorization: Bearer {token}

{
  "query": "water damage basement flooding",
  "search_type": "claims",
  "filters": {
    "coverage_types": ["DWELLING_COVERAGE_A", "PERSONAL_PROPERTY_COVERAGE_C"]
  },
  "limit": 20
}
```

## Configuration API

### System Configuration

#### Get Current Configuration
```http
GET /api/v1/config
Authorization: Bearer {token}
```

**Response:**
```json
{
  "version": "1.0.0",
  "environment": "production",
  "model_settings": {
    "embedding_model": "all-MiniLM-L6-v2",
    "classification_threshold": 0.85,
    "batch_size": 32
  },
  "processing_settings": {
    "max_workers": 4,
    "timeout_seconds": 300,
    "chunk_size": 500
  }
}
```

#### Update Configuration
```http
PUT /api/v1/config
Content-Type: application/json
Authorization: Bearer {token}

{
  "processing_settings": {
    "max_workers": 8,
    "batch_size": 64
  },
  "classification_thresholds": {
    "building_coverage": 0.90,
    "personal_property": 0.85
  }
}
```

### Business Rules Management

#### List Rules
```http
GET /api/v1/rules
Authorization: Bearer {token}
```

**Response:**
```json
{
  "total_rules": 15,
  "active_rules": 12,
  "rules": [
    {
      "rule_id": "RULE_001",
      "name": "Fire Damage Coverage Rule",
      "description": "Determines coverage for fire-related claims",
      "active": true,
      "priority": 100,
      "last_modified": "2023-11-15T14:30:00Z"
    }
  ]
}
```

#### Create Rule
```http
POST /api/v1/rules
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "Water Damage Coverage Rule",
  "description": "Determines coverage for water damage claims",
  "conditions": [
    {
      "field": "loss_description",
      "operator": "contains",
      "value": "water damage"
    },
    {
      "field": "loss_amount",
      "operator": ">",
      "value": 5000
    }
  ],
  "action": {
    "type": "classify",
    "value": "DWELLING_COVERAGE_A"
  },
  "priority": 90
}
```

## Analytics and Reporting API

### System Metrics

#### Get Processing Statistics
```http
GET /api/v1/analytics/processing-stats
Authorization: Bearer {token}

Query Parameters:
- period: daily|weekly|monthly (default: daily)
- start_date: YYYY-MM-DD
- end_date: YYYY-MM-DD
```

**Response:**
```json
{
  "period": "daily",
  "date_range": {
    "start": "2023-11-01",
    "end": "2023-11-30"
  },
  "metrics": {
    "total_claims_processed": 1250,
    "average_processing_time_ms": 1850,
    "accuracy_rate": 0.94,
    "coverage_distribution": {
      "DWELLING_COVERAGE_A": 450,
      "PERSONAL_PROPERTY_COVERAGE_C": 320,
      "OTHER_STRUCTURES_COVERAGE_B": 180,
      "LIABILITY_COVERAGE_E": 150,
      "OTHER": 150
    }
  }
}
```

### Model Performance

#### Get Model Metrics
```http
GET /api/v1/analytics/model-performance
Authorization: Bearer {token}
```

**Response:**
```json
{
  "classification_model": {
    "accuracy": 0.94,
    "precision": 0.92,
    "recall": 0.91,
    "f1_score": 0.915,
    "confusion_matrix": {
      "dwelling_coverage": {"tp": 450, "fp": 25, "fn": 30, "tn": 495},
      "personal_property": {"tp": 320, "fp": 40, "fn": 35, "tn": 605}
    }
  },
  "embedding_model": {
    "average_similarity_score": 0.78,
    "processing_speed_tokens_per_second": 1200
  }
}
```

## Error Responses

### Standard Error Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data provided",
    "details": {
      "field": "claim_text",
      "issue": "Field is required but was not provided"
    },
    "timestamp": "2023-12-01T10:15:30Z",
    "request_id": "req_abc123def456"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid input data |
| `AUTHENTICATION_REQUIRED` | 401 | Missing or invalid token |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks required permissions |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource doesn't exist |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `PROCESSING_ERROR` | 500 | Internal processing error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## Rate Limiting

API requests are subject to rate limiting:

- **Standard Users**: 100 requests per minute
- **Premium Users**: 500 requests per minute
- **Batch Processing**: 10 concurrent batch jobs

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1701432000
```

## Pagination

List endpoints support pagination:

```http
GET /api/v1/claims?page=2&per_page=50&sort=created_at&order=desc
```

**Response includes pagination metadata:**
```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "per_page": 50,
    "total_pages": 10,
    "total_items": 500,
    "has_next": true,
    "has_prev": true
  }
}
```

## Webhooks

The system supports webhooks for event notifications:

### Batch Completion Webhook

```http
POST https://your-endpoint.com/webhook
Content-Type: application/json

{
  "event": "batch.completed",
  "batch_id": "batch_xyz789abc123",
  "status": "completed",
  "summary": {
    "total_claims": 100,
    "successful": 98,
    "failed": 2
  },
  "timestamp": "2023-12-01T10:30:00Z"
}
```

## SDK Examples

### Python SDK

```python
from building_coverage_client import CoverageClient

client = CoverageClient(
    base_url="https://api.building-coverage.company.com",
    api_key="your_api_key"
)

# Analyze single claim
result = client.analyze_claim({
    "claim_id": "CLM123456789",
    "claim_text": "Fire damage to kitchen...",
    "loss_date": "2023-11-15",
    "policy_number": "POL987654321"
})

print(f"Coverage: {result.primary_coverage}")
print(f"Confidence: {result.confidence_score}")
```

### JavaScript SDK

```javascript
import { CoverageClient } from '@company/building-coverage-client';

const client = new CoverageClient({
  baseUrl: 'https://api.building-coverage.company.com',
  apiKey: 'your_api_key'
});

// Analyze claim
const result = await client.analyzeClaim({
  claimId: 'CLM123456789',
  claimText: 'Fire damage to kitchen...',
  lossDate: '2023-11-15',
  policyNumber: 'POL987654321'
});

console.log('Coverage:', result.primaryCoverage);
console.log('Confidence:', result.confidenceScore);
```