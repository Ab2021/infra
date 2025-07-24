"""
Sample data fixtures for testing building coverage system.

This module provides sample data for testing various components
of the building coverage pipeline and analytics system.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json


class SampleDataGenerator:
    """
    Generator for sample building coverage data.
    
    Provides realistic sample data for testing various components
    of the building coverage system.
    """
    
    def __init__(self):
        """Initialize the sample data generator."""
        self.claim_counter = 1000
        self.policy_counter = 50000
    
    def generate_sample_claims(self, num_claims: int = 100) -> pd.DataFrame:
        """
        Generate sample claims data.
        
        Args:
            num_claims (int): Number of claims to generate
            
        Returns:
            pd.DataFrame: Sample claims data
        """
        claims_data = []
        
        lob_codes = ['HO', 'CO', 'DP', 'CL', 'UM']
        status_codes = ['O', 'C', 'R', 'S']
        loss_descriptions = [
            'Water damage from burst pipe in basement',
            'Fire damage to kitchen and living room',
            'Wind damage to roof and siding',
            'Theft of personal property',
            'Hail damage to roof and vehicles',
            'Lightning strike caused electrical damage',
            'Ice dam caused interior water damage',
            'Tree fell on house during storm',
            'Vandalism to property exterior',
            'Smoke damage throughout house'
        ]
        
        for i in range(num_claims):
            claim_date = datetime.now() - timedelta(days=i * 2)
            
            claim_data = {
                'CLAIMNO': f'CLM{self.claim_counter + i:06d}',
                'CLAIMKEY': self.claim_counter + i,
                'LOBCD': lob_codes[i % len(lob_codes)],
                'STATUSCD': status_codes[i % len(status_codes)],
                'LOSSDT': claim_date,
                'REPORTEDDT': claim_date + timedelta(days=1),
                'LOSSDESC': loss_descriptions[i % len(loss_descriptions)],
                'clean_FN_TEXT': f"Claim for {loss_descriptions[i % len(loss_descriptions)].lower()}. Customer reported damage on {claim_date.strftime('%m/%d/%Y')}. Initial assessment shows significant damage requiring repairs.",
                'RESERVE_TOTAL': round(5000 + (i * 100), 2),
                'PAID_TOTAL': round(2000 + (i * 50), 2),
                'INCURRED_TOTAL': round(7000 + (i * 150), 2),
                'POLICY_NUMBER': f'POL{self.policy_counter + i:08d}',
                'INSURED_NAME': f'John Doe {i}',
                'PROPERTY_ADDRESS': f'{100 + i} Main Street, City, State 12345'
            }
            
            claims_data.append(claim_data)
        
        self.claim_counter += num_claims
        return pd.DataFrame(claims_data)
    
    def generate_sample_embeddings(self, num_embeddings: int = 100) -> pd.DataFrame:
        """
        Generate sample text embeddings.
        
        Args:
            num_embeddings (int): Number of embeddings to generate
            
        Returns:
            pd.DataFrame: Sample embeddings data
        """
        import numpy as np
        
        embeddings_data = []
        
        for i in range(num_embeddings):
            # Generate random embedding vector (384 dimensions for sentence-transformers)
            embedding_vector = np.random.normal(0, 1, 384).tolist()
            
            embedding_data = {
                'text_id': f'TXT{i:06d}',
                'original_text': f'Sample claim text {i} with building damage description',
                'embedding_vector': json.dumps(embedding_vector),
                'embedding_model': 'all-MiniLM-L6-v2',
                'created_at': datetime.now() - timedelta(hours=i),
                'chunk_index': i % 5,
                'source_document': f'claim_doc_{i // 5}'
            }
            
            embeddings_data.append(embedding_data)
        
        return pd.DataFrame(embeddings_data)
    
    def generate_sample_classifications(self, num_classifications: int = 100) -> pd.DataFrame:
        """
        Generate sample building coverage classifications.
        
        Args:
            num_classifications (int): Number of classifications to generate
            
        Returns:
            pd.DataFrame: Sample classifications data
        """
        classifications_data = []
        
        coverage_types = [
            'DWELLING_COVERAGE',
            'PERSONAL_PROPERTY_COVERAGE',
            'LIABILITY_COVERAGE',
            'ADDITIONAL_LIVING_EXPENSE',
            'MEDICAL_PAYMENTS',
            'OTHER_STRUCTURES'
        ]
        
        confidence_scores = [0.95, 0.87, 0.92, 0.89, 0.94, 0.83, 0.90, 0.96]
        
        for i in range(num_classifications):
            classification_data = {
                'claim_id': f'CLM{1000 + i:06d}',
                'coverage_type': coverage_types[i % len(coverage_types)],
                'confidence_score': confidence_scores[i % len(confidence_scores)],
                'model_version': 'v2.1.0',
                'classification_date': datetime.now() - timedelta(hours=i * 2),
                'features_used': json.dumps(['text_embedding', 'loss_description', 'damage_amount']),
                'is_building_related': (i % 3 == 0),  # 1/3 are building-related
                'manual_review_required': (confidence_scores[i % len(confidence_scores)] < 0.90)
            }
            
            classifications_data.append(classification_data)
        
        return pd.DataFrame(classifications_data)
    
    def generate_sample_rules_data(self, num_rules: int = 20) -> List[Dict[str, Any]]:
        """
        Generate sample business rules data.
        
        Args:
            num_rules (int): Number of rules to generate
            
        Returns:
            List[Dict[str, Any]]: Sample rules data
        """
        rules_data = []
        
        rule_types = ['coverage_determination', 'exclusion_check', 'limit_validation', 'deductible_calculation']
        conditions = [
            {'field': 'loss_amount', 'operator': '>', 'value': 10000},
            {'field': 'lob_code', 'operator': 'in', 'value': ['HO', 'CO']},
            {'field': 'loss_description', 'operator': 'contains', 'value': 'fire'},
            {'field': 'policy_limit', 'operator': '>=', 'value': 100000}
        ]
        
        for i in range(num_rules):
            rule_data = {
                'rule_id': f'RULE_{i:03d}',
                'rule_name': f'Building Coverage Rule {i}',
                'rule_type': rule_types[i % len(rule_types)],
                'conditions': conditions[i % len(conditions)],
                'action': {
                    'type': 'classify',
                    'value': 'BUILDING_COVERAGE' if i % 2 == 0 else 'OTHER_COVERAGE'
                },
                'priority': 100 + i,
                'active': True,
                'created_date': datetime.now() - timedelta(days=i * 10),
                'last_modified': datetime.now() - timedelta(days=i),
                'description': f'Rule for determining building coverage based on specific conditions - Rule {i}'
            }
            
            rules_data.append(rule_data)
        
        return rules_data
    
    def generate_sample_sql_queries(self) -> Dict[str, str]:
        """
        Generate sample SQL queries for testing.
        
        Returns:
            Dict[str, str]: Sample SQL queries
        """
        return {
            'feature_queries': {
                'main_claims_query': """
                    SELECT 
                        CLAIMNO,
                        CLAIMKEY,
                        LOBCD,
                        STATUSCD,
                        LOSSDT,
                        REPORTEDDT,
                        LOSSDESC,
                        clean_FN_TEXT,
                        RESERVE_TOTAL,
                        PAID_TOTAL,
                        INCURRED_TOTAL
                    FROM claims_data
                    WHERE STATUSCD IN ('O', 'C')
                    AND LOSSDT >= '2020-01-01'
                    ORDER BY LOSSDT DESC
                """,
                'aip_claims_query': """
                    SELECT 
                        claim_number as CLAIMNO,
                        claim_id as CLAIMKEY,
                        line_of_business as LOBCD,
                        status as STATUSCD,
                        loss_date as LOSSDT,
                        reported_date as REPORTEDDT,
                        loss_description as LOSSDESC,
                        claim_text as clean_FN_TEXT,
                        reserve_amount as RESERVE_TOTAL,
                        paid_amount as PAID_TOTAL,
                        incurred_amount as INCURRED_TOTAL
                    FROM aip_claims
                    WHERE status IN ('OPEN', 'CLOSED')
                    ORDER BY loss_date DESC
                """,
                'atlas_claims_query': """
                    SELECT 
                        CLAIM_NUM as CLAIMNO,
                        CLAIM_KEY as CLAIMKEY,
                        LOB as LOBCD,
                        CLAIM_STATUS as STATUSCD,
                        LOSS_DT as LOSSDT,
                        RPT_DT as REPORTEDDT,
                        LOSS_DESC as LOSSDESC,
                        DESCRIPTION as clean_FN_TEXT,
                        RES_AMT as RESERVE_TOTAL,
                        PAID_AMT as PAID_TOTAL,
                        INC_AMT as INCURRED_TOTAL
                    FROM atlas_claims_view
                    WHERE CLAIM_STATUS NOT IN ('DECLINED', 'VOID')
                    ORDER BY LOSS_DT DESC
                """
            }
        }
    
    def generate_sample_credentials(self) -> Dict[str, Any]:
        """
        Generate sample database credentials for testing.
        
        Returns:
            Dict[str, Any]: Sample credentials
        """
        return {
            'server': 'test-sql-server.company.com',
            'database': 'test_claims_db',
            'username': 'test_user',
            'password': 'test_password',
            'driver': 'ODBC Driver 17 for SQL Server',
            'aip_server': 'test-aip-server.company.com',
            'aip_database': 'test_aip_db',
            'aip_username': 'aip_user',
            'aip_password': 'aip_password',
            'atlas_server': 'test-atlas-server.company.com',
            'atlas_database': 'test_atlas_db',
            'atlas_username': 'atlas_user',
            'atlas_password': 'atlas_password'
        }
    
    def generate_sample_config(self) -> Dict[str, Any]:
        """
        Generate sample configuration for testing.
        
        Returns:
            Dict[str, Any]: Sample configuration
        """
        return {
            'data_sources': {
                'primary': {
                    'enabled': True,
                    'connection_timeout': 30,
                    'query_timeout': 300
                },
                'aip': {
                    'enabled': True,
                    'connection_timeout': 45,
                    'query_timeout': 600
                },
                'atlas': {
                    'enabled': False,
                    'connection_timeout': 30,
                    'query_timeout': 300
                }
            },
            'embedding_model': {
                'model_name': 'all-MiniLM-L6-v2',
                'max_sequence_length': 512,
                'batch_size': 32
            },
            'classification_thresholds': {
                'building_coverage': 0.85,
                'personal_property': 0.80,
                'liability': 0.90
            },
            'processing': {
                'batch_size': 1000,
                'max_workers': 4,
                'chunk_size': 500
            }
        }


# Factory functions for easy fixture creation
def create_sample_claims_data(num_claims: int = 100) -> pd.DataFrame:
    """
    Create sample claims data for testing.
    
    Args:
        num_claims (int): Number of claims to generate
        
    Returns:
        pd.DataFrame: Sample claims data
    """
    generator = SampleDataGenerator()
    return generator.generate_sample_claims(num_claims)


def create_sample_embeddings_data(num_embeddings: int = 100) -> pd.DataFrame:
    """
    Create sample embeddings data for testing.
    
    Args:
        num_embeddings (int): Number of embeddings to generate
        
    Returns:
        pd.DataFrame: Sample embeddings data
    """
    generator = SampleDataGenerator()
    return generator.generate_sample_embeddings(num_embeddings)


def create_sample_classifications_data(num_classifications: int = 100) -> pd.DataFrame:
    """
    Create sample classifications data for testing.
    
    Args:
        num_classifications (int): Number of classifications to generate
        
    Returns:
        pd.DataFrame: Sample classifications data
    """
    generator = SampleDataGenerator()
    return generator.generate_sample_classifications(num_classifications)


def create_sample_rules_data(num_rules: int = 20) -> List[Dict[str, Any]]:
    """
    Create sample rules data for testing.
    
    Args:
        num_rules (int): Number of rules to generate
        
    Returns:
        List[Dict[str, Any]]: Sample rules data
    """
    generator = SampleDataGenerator()
    return generator.generate_sample_rules_data(num_rules)


def create_sample_sql_queries() -> Dict[str, str]:
    """
    Create sample SQL queries for testing.
    
    Returns:
        Dict[str, str]: Sample SQL queries
    """
    generator = SampleDataGenerator()
    return generator.generate_sample_sql_queries()


def create_sample_credentials() -> Dict[str, Any]:
    """
    Create sample database credentials for testing.
    
    Returns:
        Dict[str, Any]: Sample credentials
    """
    generator = SampleDataGenerator()
    return generator.generate_sample_credentials()


def create_sample_config() -> Dict[str, Any]:
    """
    Create sample configuration for testing.
    
    Returns:
        Dict[str, Any]: Sample configuration
    """
    generator = SampleDataGenerator()
    return generator.generate_sample_config()


# Predefined test data sets
SAMPLE_CLAIM_TEXTS = [
    "Water damage from burst pipe in basement flooded finished basement area",
    "Fire in kitchen caused extensive smoke damage throughout first floor",
    "Hail storm damaged roof shingles and gutters, water entered attic space",
    "Tree fell on house during windstorm, damaged roof and bedroom wall",
    "Lightning strike caused electrical surge, damaged appliances and wiring",
    "Ice dam on roof caused water backup into living room and bedroom",
    "Vandalism to exterior doors and windows, glass broken and frames damaged",
    "Smoke damage from neighboring house fire affected entire second floor",
    "Frozen pipes burst in crawl space, water damage to hardwood floors",
    "Wind damage during tornado removed section of roof and exterior wall"
]

SAMPLE_BUILDING_TERMS = [
    "roof damage", "foundation crack", "siding replacement", "window replacement",
    "door frame", "structural damage", "load bearing wall", "exterior wall",
    "interior wall", "ceiling damage", "floor damage", "basement flooding",
    "attic insulation", "electrical wiring", "plumbing system", "HVAC system",
    "chimney damage", "deck replacement", "porch repair", "garage door"
]

SAMPLE_COVERAGE_TYPES = [
    "DWELLING_COVERAGE_A",
    "OTHER_STRUCTURES_COVERAGE_B", 
    "PERSONAL_PROPERTY_COVERAGE_C",
    "LOSS_OF_USE_COVERAGE_D",
    "PERSONAL_LIABILITY_COVERAGE_E",
    "MEDICAL_PAYMENTS_COVERAGE_F"
]