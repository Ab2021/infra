# User Guide

This guide provides comprehensive instructions for using the Building Coverage System to analyze insurance claims and determine coverage types.

## Getting Started

### Prerequisites

Before using the system, ensure you have:
- Valid user credentials and API access
- Basic understanding of insurance claim terminology
- Familiarity with coverage types (Dwelling, Personal Property, Liability, etc.)

### Initial Setup

1. **Access the System**
   ```
   Web Interface: https://building-coverage.company.com
   API Endpoint: https://api.building-coverage.company.com
   ```

2. **Authentication**
   - Log in with your corporate credentials
   - Generate API tokens for programmatic access
   - Configure your client applications

## Core Functionality

### 1. Single Claim Analysis

#### Using the Web Interface

1. **Navigate to Claim Analysis**
   - Go to "Claims" → "Analyze Single Claim"
   - Enter claim details in the form

2. **Input Claim Information**
   ```
   Claim ID: CLM123456789
   Claim Text: "Fire damage to kitchen caused by electrical short circuit. 
                Smoke damage throughout first floor of house."
   Loss Date: 2023-11-15
   Policy Number: POL987654321
   Line of Business: HO (Homeowners)
   Loss Amount: $15,000
   ```

3. **Review Results**
   The system will provide:
   - **Primary Coverage Determination**: DWELLING_COVERAGE_A
   - **Confidence Score**: 92%
   - **Supporting Evidence**: Rule matches and similar claims
   - **Alternative Coverages**: Other applicable coverage types

#### Using the API

```python
import requests

# API request for claim analysis
response = requests.post(
    'https://api.building-coverage.company.com/api/v1/claims/analyze',
    headers={
        'Authorization': 'Bearer your_token_here',
        'Content-Type': 'application/json'
    },
    json={
        'claim_id': 'CLM123456789',
        'claim_text': 'Fire damage to kitchen caused by electrical short circuit...',
        'loss_date': '2023-11-15',
        'policy_number': 'POL987654321',
        'lob_code': 'HO',
        'loss_amount': 15000.00
    }
)

result = response.json()
print(f"Primary Coverage: {result['results']['coverage_determination']['primary_coverage']}")
print(f"Confidence: {result['results']['coverage_determination']['confidence_score']}")
```

### 2. Batch Processing

#### Preparing Batch Files

Create a CSV file with your claims data:

```csv
claim_id,claim_text,loss_date,policy_number,lob_code,loss_amount
CLM001,"Water damage from burst pipe in basement",2023-11-01,POL001,HO,8500.00
CLM002,"Hail damage to roof and siding",2023-11-02,POL002,HO,12000.00
CLM003,"Theft of jewelry and electronics",2023-11-03,POL003,HO,5500.00
```

#### Submitting Batch Jobs

1. **Web Interface Method**
   - Go to "Claims" → "Batch Processing"
   - Upload your CSV file
   - Configure processing options
   - Submit the batch job

2. **API Method**
   ```python
   import pandas as pd
   
   # Read claims from CSV
   claims_df = pd.read_csv('claims_batch.csv')
   
   # Convert to API format
   claims_data = []
   for _, row in claims_df.iterrows():
       claims_data.append({
           'claim_id': row['claim_id'],
           'claim_text': row['claim_text'],
           'loss_date': row['loss_date'],
           'policy_number': row['policy_number'],
           'lob_code': row['lob_code'],
           'loss_amount': row['loss_amount']
       })
   
   # Submit batch
   response = requests.post(
       'https://api.building-coverage.company.com/api/v1/claims/batch',
       headers={'Authorization': 'Bearer your_token_here'},
       json={
           'batch_name': 'November_Claims_Analysis',
           'claims': claims_data,
           'processing_options': {
               'include_similar_claims': True,
               'confidence_threshold': 0.85
           }
       }
   )
   
   batch_id = response.json()['batch_id']
   print(f"Batch submitted: {batch_id}")
   ```

#### Monitoring Batch Progress

1. **Check Status**
   ```python
   # Check batch status
   status_response = requests.get(
       f'https://api.building-coverage.company.com/api/v1/batches/{batch_id}/status',
       headers={'Authorization': 'Bearer your_token_here'}
   )
   
   status = status_response.json()
   print(f"Progress: {status['progress']['percentage_complete']}%")
   print(f"Status: {status['status']}")
   ```

2. **Retrieve Results**
   ```python
   # Get results when completed
   results_response = requests.get(
       f'https://api.building-coverage.company.com/api/v1/batches/{batch_id}/results',
       headers={'Authorization': 'Bearer your_token_here'}
   )
   
   results = results_response.json()
   
   # Process results
   for result in results['results']:
       claim_id = result['claim_id']
       coverage = result['coverage_determination']['primary_coverage']
       confidence = result['coverage_determination']['confidence_score']
       print(f"{claim_id}: {coverage} ({confidence:.2%})")
   ```

### 3. Similar Claims Search

#### Finding Similar Claims

Use the similar claims feature to find precedent cases:

1. **Web Interface**
   - Go to "Search" → "Similar Claims"
   - Enter your search text
   - Apply filters (date range, LOB, coverage types)
   - Review similar claims with similarity scores

2. **API Usage**
   ```python
   # Search for similar claims
   search_response = requests.post(
       'https://api.building-coverage.company.com/api/v1/search/similar-claims',
       headers={'Authorization': 'Bearer your_token_here'},
       json={
           'query_text': 'Fire damage to kitchen and smoke throughout house',
           'filters': {
               'lob_codes': ['HO', 'CO'],
               'date_range': {
                   'start': '2020-01-01',
                   'end': '2023-12-01'
               },
               'min_similarity': 0.7
           },
           'limit': 10
       }
   )
   
   similar_claims = search_response.json()
   
   for claim in similar_claims['results']:
       print(f"Claim {claim['claim_id']}: {claim['similarity_score']:.2%} similar")
       print(f"Coverage: {claim['coverage_decision']}")
       print(f"Summary: {claim['claim_summary']}")
       print("---")
   ```

## Understanding Results

### Coverage Types

The system classifies claims into standard coverage categories:

#### **DWELLING_COVERAGE_A**
- **Description**: Covers damage to the physical structure of the home
- **Examples**: Fire damage, wind damage, structural damage
- **Typical Claims**: "Fire damaged kitchen walls and ceiling"

#### **OTHER_STRUCTURES_COVERAGE_B**
- **Description**: Covers detached structures on the property
- **Examples**: Garage, shed, fence damage
- **Typical Claims**: "Tornado damaged detached garage"

#### **PERSONAL_PROPERTY_COVERAGE_C**
- **Description**: Covers personal belongings and contents
- **Examples**: Furniture, electronics, clothing
- **Typical Claims**: "Theft of jewelry and electronics"

#### **LOSS_OF_USE_COVERAGE_D**
- **Description**: Covers additional living expenses
- **Examples**: Hotel costs, temporary housing
- **Typical Claims**: "House uninhabitable due to fire damage"

#### **PERSONAL_LIABILITY_COVERAGE_E**
- **Description**: Covers liability for injuries to others
- **Examples**: Slip and fall, dog bite incidents
- **Typical Claims**: "Guest injured on icy walkway"

#### **MEDICAL_PAYMENTS_COVERAGE_F**
- **Description**: Covers medical expenses for others
- **Examples**: Minor injuries to guests
- **Typical Claims**: "Neighbor cut by broken glass"

### Confidence Scores

The system provides confidence scores to help you assess the reliability of classifications:

- **90-100%**: Very High Confidence - Strong textual evidence and rule matches
- **80-89%**: High Confidence - Clear coverage indicators present
- **70-79%**: Medium Confidence - Some ambiguity, review recommended
- **60-69%**: Low Confidence - Multiple possible coverages, manual review needed
- **Below 60%**: Very Low Confidence - Insufficient information for reliable classification

### Supporting Evidence

Each analysis includes supporting evidence:

1. **Rule Matches**
   ```
   Rule: Fire Damage Coverage Rule (RULE_001)
   Status: Matched
   Action: Approve Dwelling Coverage
   ```

2. **Similar Claims**
   ```
   Claim CLM987654321 (87% similar)
   Coverage: DWELLING_COVERAGE_A
   Summary: Kitchen fire with smoke damage
   ```

3. **Key Terms Identified**
   ```
   Building-related terms: kitchen, walls, ceiling, structure
   Damage terms: fire, smoke, electrical, damaged
   ```

## Advanced Features

### 1. Business Rules Management

#### Viewing Active Rules

```python
# Get current business rules
rules_response = requests.get(
    'https://api.building-coverage.company.com/api/v1/rules',
    headers={'Authorization': 'Bearer your_token_here'}
)

rules = rules_response.json()
for rule in rules['rules']:
    print(f"Rule {rule['rule_id']}: {rule['name']}")
    print(f"Status: {'Active' if rule['active'] else 'Inactive'}")
    print(f"Priority: {rule['priority']}")
```

#### Creating Custom Rules

```python
# Create a new business rule
new_rule = {
    'name': 'High-Value Water Damage Rule',
    'description': 'Classifies high-value water damage claims',
    'conditions': [
        {
            'field': 'loss_description',
            'operator': 'contains',
            'value': 'water damage'
        },
        {
            'field': 'loss_amount',
            'operator': '>',
            'value': 10000
        }
    ],
    'action': {
        'type': 'classify',
        'value': 'DWELLING_COVERAGE_A'
    },
    'priority': 85
}

create_response = requests.post(
    'https://api.building-coverage.company.com/api/v1/rules',
    headers={'Authorization': 'Bearer your_token_here'},
    json=new_rule
)
```

### 2. Configuration Management

#### Updating Processing Settings

```python
# Update system configuration
config_update = {
    'processing_settings': {
        'batch_size': 2000,
        'max_workers': 8
    },
    'classification_thresholds': {
        'building_coverage': 0.90,
        'personal_property': 0.85
    }
}

config_response = requests.put(
    'https://api.building-coverage.company.com/api/v1/config',
    headers={'Authorization': 'Bearer your_token_here'},
    json=config_update
)
```

### 3. Analytics and Reporting

#### Processing Statistics

```python
# Get processing statistics
stats_response = requests.get(
    'https://api.building-coverage.company.com/api/v1/analytics/processing-stats',
    headers={'Authorization': 'Bearer your_token_here'},
    params={
        'period': 'monthly',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31'
    }
)

stats = stats_response.json()
print(f"Total claims processed: {stats['metrics']['total_claims_processed']}")
print(f"Average processing time: {stats['metrics']['average_processing_time_ms']}ms")
print(f"Accuracy rate: {stats['metrics']['accuracy_rate']:.2%}")
```

## Best Practices

### 1. Input Data Quality

**Good Claim Text:**
```
Fire damage to kitchen caused by electrical short circuit in outlet near stove. 
Flames spread to cabinets and ceiling. Smoke damage throughout first floor 
including living room and dining room. Kitchen requires complete renovation.
```

**Poor Claim Text:**
```
Kitchen fire. Damage.
```

**Tips for Better Results:**
- Include specific damage descriptions
- Mention affected areas/structures
- Describe the cause of loss
- Include relevant details about extent of damage

### 2. Batch Processing Optimization

- **Optimal Batch Size**: 500-2000 claims per batch
- **File Format**: Use CSV with proper encoding (UTF-8)
- **Data Validation**: Ensure all required fields are populated
- **Processing Time**: Allow adequate time for large batches

### 3. Interpreting Results

- **High Confidence (>85%)**: Generally reliable, can be used for automated processing
- **Medium Confidence (70-85%)**: Review for accuracy, especially for high-value claims
- **Low Confidence (<70%)**: Requires manual review and additional information

### 4. Error Handling

```python
try:
    response = requests.post(api_url, json=data, headers=headers)
    response.raise_for_status()  # Raises exception for HTTP errors
    
    result = response.json()
    if 'error' in result:
        print(f"API Error: {result['error']['message']}")
    else:
        # Process successful result
        pass
        
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except ValueError as e:
    print(f"Invalid JSON response: {e}")
```

## Troubleshooting

### Common Issues

1. **Low Confidence Scores**
   - **Cause**: Insufficient or unclear claim text
   - **Solution**: Provide more detailed descriptions

2. **Incorrect Classifications**
   - **Cause**: Unusual claim circumstances not covered by training data
   - **Solution**: Review and provide feedback for model improvement

3. **Batch Processing Failures**
   - **Cause**: Invalid data format or missing required fields
   - **Solution**: Validate input data before submission

4. **API Authentication Errors**
   - **Cause**: Expired or invalid tokens
   - **Solution**: Refresh authentication tokens

### Getting Help

- **Documentation**: Refer to this user guide and API documentation
- **Support Portal**: Submit tickets through the corporate support system
- **Training**: Contact the team for additional training sessions
- **Feedback**: Provide feedback on classification accuracy to improve the system

## Workflow Examples

### Example 1: Daily Claim Processing Workflow

```python
def daily_claim_processing():
    # 1. Extract new claims from database
    new_claims = extract_new_claims_from_db()
    
    # 2. Submit batch for processing
    batch_response = submit_batch_processing(new_claims)
    batch_id = batch_response['batch_id']
    
    # 3. Monitor progress
    while not is_batch_complete(batch_id):
        time.sleep(60)  # Check every minute
        
    # 4. Retrieve and process results
    results = get_batch_results(batch_id)
    
    # 5. Update database with classifications
    update_claims_database(results)
    
    # 6. Generate report
    generate_daily_report(results)
```

### Example 2: Quality Assurance Workflow

```python
def quality_assurance_review():
    # 1. Get claims with low confidence scores
    low_confidence_claims = get_claims_by_confidence(max_confidence=0.75)
    
    # 2. Review each claim
    for claim in low_confidence_claims:
        # Get similar claims for context
        similar_claims = search_similar_claims(claim['claim_text'])
        
        # Present for manual review
        manual_classification = review_claim_manually(claim, similar_claims)
        
        # Update classification if needed
        if manual_classification != claim['ai_classification']:
            update_claim_classification(claim['id'], manual_classification)
            
            # Provide feedback for model improvement
            submit_feedback(claim, manual_classification)
```

This user guide provides comprehensive instructions for effectively using the Building Coverage System. For additional support or advanced use cases, please refer to the API documentation or contact the system administrators.