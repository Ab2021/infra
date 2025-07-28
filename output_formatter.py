"""
Output Formatter for Agentic Building Coverage Analysis
Essential DataFrameTransformations methods for database-ready output
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class OutputFormatter:
    """Format extraction results into database-ready structure"""
    
    def __init__(self):
        # Define the 40+ column output schema
        self.output_schema = {
            # Claim identification
            'CLAIM_ID': 'string',
            'EXTRACTION_TIMESTAMP': 'datetime',
            'SOURCE_TEXT_LENGTH': 'integer',
            
            # Damage indicators (15) - Yes/No + Confidence
            'BLDG_FIRE_DMG': 'string',
            'BLDG_FIRE_DMG_CONF': 'float',
            'BLDG_WATER_DMG': 'string', 
            'BLDG_WATER_DMG_CONF': 'float',
            'BLDG_WIND_DMG': 'string',
            'BLDG_WIND_DMG_CONF': 'float',
            'BLDG_HAIL_DMG': 'string',
            'BLDG_HAIL_DMG_CONF': 'float',
            'BLDG_LIGHTNING_DMG': 'string',
            'BLDG_LIGHTNING_DMG_CONF': 'float',
            'BLDG_VANDALISM_DMG': 'string',
            'BLDG_VANDALISM_DMG_CONF': 'float',
            'BLDG_THEFT_DMG': 'string',
            'BLDG_THEFT_DMG_CONF': 'float',
            'BLDG_ROOF_DMG': 'string',
            'BLDG_ROOF_DMG_CONF': 'float',
            'BLDG_WALLS_DMG': 'string',
            'BLDG_WALLS_DMG_CONF': 'float',
            'BLDG_FLOORING_DMG': 'string',
            'BLDG_FLOORING_DMG_CONF': 'float',
            'BLDG_CEILING_DMG': 'string',
            'BLDG_CEILING_DMG_CONF': 'float',
            'BLDG_WINDOWS_DMG': 'string',
            'BLDG_WINDOWS_DMG_CONF': 'float',
            'BLDG_DOORS_DMG': 'string',
            'BLDG_DOORS_DMG_CONF': 'float',
            'BLDG_ELECTRICAL_DMG': 'string',
            'BLDG_ELECTRICAL_DMG_CONF': 'float',
            'BLDG_PLUMBING_DMG': 'string',
            'BLDG_PLUMBING_DMG_CONF': 'float',
            
            # Operational indicators (3)
            'BLDG_INTERIOR_DMG': 'string',
            'BLDG_INTERIOR_DMG_CONF': 'float',
            'BLDG_TENABLE': 'string',
            'BLDG_TENABLE_CONF': 'float',
            'BLDG_PRIMARY_STRUCTURE': 'string',
            'BLDG_PRIMARY_STRUCTURE_CONF': 'float',
            
            # Contextual indicators (4)
            'BLDG_OCCUPANCY_TYPE': 'string',
            'BLDG_OCCUPANCY_TYPE_CONF': 'float',
            'BLDG_SQUARE_FOOTAGE': 'string',
            'BLDG_SQUARE_FOOTAGE_CONF': 'float',
            'BLDG_YEAR_BUILT': 'string',
            'BLDG_YEAR_BUILT_CONF': 'float',
            'BLDG_CONSTRUCTION_TYPE': 'string',
            'BLDG_CONSTRUCTION_TYPE_CONF': 'float',
            
            # Financial analysis
            'MONETARY_CANDIDATES_COUNT': 'integer',
            'MONETARY_CANDIDATES_JSON': 'string',
            'BLDG_LOSS_AMOUNT': 'float',
            'BLDG_LOSS_AMOUNT_CONF': 'float',
            'LOSS_CALCULATION_METHOD': 'string',
            
            # Summary metrics
            'TOTAL_DAMAGE_INDICATORS': 'integer',
            'HIGH_CONFIDENCE_INDICATORS': 'integer',
            'EXTRACTION_COMPLETENESS': 'float',
            'VALIDATION_PASSED': 'boolean',
            'PROCESSING_STATUS': 'string'
        }
    
    def format_extraction_results(self, extraction_results: Dict[str, Any],
                                monetary_analysis: Optional[Dict[str, Any]] = None,
                                validation_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format complete extraction results into database schema"""
        
        # Initialize output record with default values
        formatted_record = self._initialize_record()
        
        # Add basic claim information
        formatted_record['CLAIM_ID'] = extraction_results.get('claim_id', '')
        formatted_record['EXTRACTION_TIMESTAMP'] = datetime.now().isoformat()
        formatted_record['SOURCE_TEXT_LENGTH'] = len(extraction_results.get('source_text', ''))
        
        # Process damage indicators
        self._format_damage_indicators(extraction_results, formatted_record)
        
        # Process operational indicators
        self._format_operational_indicators(extraction_results, formatted_record)
        
        # Process contextual indicators
        self._format_contextual_indicators(extraction_results, formatted_record)
        
        # Process monetary analysis
        if monetary_analysis:
            self._format_monetary_analysis(monetary_analysis, formatted_record)
        
        # Process validation results
        if validation_results:
            self._format_validation_results(validation_results, formatted_record)
        
        # Calculate summary metrics
        self._calculate_summary_metrics(extraction_results, formatted_record)
        
        return formatted_record
    
    def _initialize_record(self) -> Dict[str, Any]:
        """Initialize record with default values based on schema"""
        record = {}
        
        for column, data_type in self.output_schema.items():
            if data_type == 'string':
                record[column] = ''
            elif data_type == 'float':
                record[column] = 0.0
            elif data_type == 'integer':
                record[column] = 0
            elif data_type == 'boolean':
                record[column] = False
            elif data_type == 'datetime':
                record[column] = datetime.now().isoformat()
        
        return record
    
    def _format_damage_indicators(self, extraction_results: Dict[str, Any], 
                                formatted_record: Dict[str, Any]):
        """Format damage indicators into output schema"""
        
        damage_indicators = [
            "BLDG_FIRE_DMG", "BLDG_WATER_DMG", "BLDG_WIND_DMG", "BLDG_HAIL_DMG",
            "BLDG_LIGHTNING_DMG", "BLDG_VANDALISM_DMG", "BLDG_THEFT_DMG",
            "BLDG_ROOF_DMG", "BLDG_WALLS_DMG", "BLDG_FLOORING_DMG",
            "BLDG_CEILING_DMG", "BLDG_WINDOWS_DMG", "BLDG_DOORS_DMG",
            "BLDG_ELECTRICAL_DMG", "BLDG_PLUMBING_DMG"
        ]
        
        for indicator in damage_indicators:
            result = extraction_results.get(indicator, {})
            if isinstance(result, dict):
                formatted_record[indicator] = result.get('value', 'N')
                formatted_record[f"{indicator}_CONF"] = float(result.get('confidence', 0.0))
            else:
                formatted_record[indicator] = 'N'
                formatted_record[f"{indicator}_CONF"] = 0.0
    
    def _format_operational_indicators(self, extraction_results: Dict[str, Any],
                                     formatted_record: Dict[str, Any]):
        """Format operational indicators into output schema"""
        
        operational_indicators = ["BLDG_INTERIOR_DMG", "BLDG_TENABLE", "BLDG_PRIMARY_STRUCTURE"]
        
        for indicator in operational_indicators:
            result = extraction_results.get(indicator, {})
            if isinstance(result, dict):
                formatted_record[indicator] = result.get('value', 'N')
                formatted_record[f"{indicator}_CONF"] = float(result.get('confidence', 0.0))
            else:
                formatted_record[indicator] = 'N'
                formatted_record[f"{indicator}_CONF"] = 0.0
    
    def _format_contextual_indicators(self, extraction_results: Dict[str, Any],
                                    formatted_record: Dict[str, Any]):
        """Format contextual indicators into output schema"""
        
        contextual_indicators = [
            "BLDG_OCCUPANCY_TYPE", "BLDG_SQUARE_FOOTAGE", 
            "BLDG_YEAR_BUILT", "BLDG_CONSTRUCTION_TYPE"
        ]
        
        for indicator in contextual_indicators:
            result = extraction_results.get(indicator, {})
            if isinstance(result, dict):
                formatted_record[indicator] = str(result.get('value', 'unknown'))
                formatted_record[f"{indicator}_CONF"] = float(result.get('confidence', 0.0))
            else:
                formatted_record[indicator] = 'unknown'
                formatted_record[f"{indicator}_CONF"] = 0.0
    
    def _format_monetary_analysis(self, monetary_analysis: Dict[str, Any],
                                formatted_record: Dict[str, Any]):
        """Format monetary analysis results"""
        
        # Monetary candidates
        candidates = monetary_analysis.get('monetary_candidates', [])
        formatted_record['MONETARY_CANDIDATES_COUNT'] = len(candidates)
        formatted_record['MONETARY_CANDIDATES_JSON'] = json.dumps(candidates)
        
        # Final loss amount
        final_result = monetary_analysis.get('final_calculation', {})
        formatted_record['BLDG_LOSS_AMOUNT'] = float(final_result.get('final_amount', 0))
        formatted_record['BLDG_LOSS_AMOUNT_CONF'] = float(final_result.get('confidence', 0.0))
        formatted_record['LOSS_CALCULATION_METHOD'] = final_result.get('method', 'unknown')
    
    def _format_validation_results(self, validation_results: Dict[str, Any],
                                 formatted_record: Dict[str, Any]):
        """Format validation results"""
        
        formatted_record['VALIDATION_PASSED'] = validation_results.get('validation_passed', False)
        formatted_record['PROCESSING_STATUS'] = validation_results.get('status', 'completed')
    
    def _calculate_summary_metrics(self, extraction_results: Dict[str, Any],
                                 formatted_record: Dict[str, Any]):
        """Calculate summary metrics for the record"""
        
        # Count damage indicators found
        damage_count = 0
        high_confidence_count = 0
        total_indicators = 0
        
        for key, value in extraction_results.items():
            if key.startswith('BLDG_') and isinstance(value, dict):
                total_indicators += 1
                confidence = value.get('confidence', 0)
                
                if value.get('value') == 'Y':
                    damage_count += 1
                
                if confidence >= 0.8:
                    high_confidence_count += 1
        
        formatted_record['TOTAL_DAMAGE_INDICATORS'] = damage_count
        formatted_record['HIGH_CONFIDENCE_INDICATORS'] = high_confidence_count
        
        # Calculate extraction completeness (non-empty/non-N values out of 22 total)
        non_empty_count = len([
            v for k, v in extraction_results.items() 
            if k.startswith('BLDG_') and isinstance(v, dict) 
            and v.get('value') not in ['N', 'unknown', '']
        ])
        
        formatted_record['EXTRACTION_COMPLETENESS'] = min(1.0, non_empty_count / 22.0)
    
    def create_dataframe(self, formatted_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create pandas DataFrame from formatted records"""
        
        if not formatted_records:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=list(self.output_schema.keys()))
        
        df = pd.DataFrame(formatted_records)
        
        # Ensure all schema columns are present
        for column in self.output_schema.keys():
            if column not in df.columns:
                data_type = self.output_schema[column]
                if data_type == 'string':
                    df[column] = ''
                elif data_type == 'float':
                    df[column] = 0.0
                elif data_type == 'integer':
                    df[column] = 0
                elif data_type == 'boolean':
                    df[column] = False
                elif data_type == 'datetime':
                    df[column] = datetime.now().isoformat()
        
        # Reorder columns to match schema
        df = df[list(self.output_schema.keys())]
        
        return df
    
    def export_to_csv(self, formatted_records: List[Dict[str, Any]], 
                     output_file: str):
        """Export formatted records to CSV file"""
        
        df = self.create_dataframe(formatted_records)
        df.to_csv(output_file, index=False)
        print(f"Exported {len(formatted_records)} records to {output_file}")
    
    def export_to_json(self, formatted_records: List[Dict[str, Any]], 
                      output_file: str):
        """Export formatted records to JSON file"""
        
        with open(output_file, 'w') as f:
            json.dump(formatted_records, f, indent=2, default=str)
        print(f"Exported {len(formatted_records)} records to {output_file}")
    
    def validate_record_schema(self, record: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate a record against the expected schema"""
        
        issues = {
            'missing_columns': [],
            'type_mismatches': [],
            'invalid_values': []
        }
        
        # Check for missing columns
        for column in self.output_schema.keys():
            if column not in record:
                issues['missing_columns'].append(column)
        
        # Check data types and values
        for column, value in record.items():
            if column in self.output_schema:
                expected_type = self.output_schema[column]
                
                # Type validation
                if expected_type == 'string' and not isinstance(value, str):
                    issues['type_mismatches'].append(f"{column}: expected string, got {type(value)}")
                elif expected_type == 'float' and not isinstance(value, (int, float)):
                    issues['type_mismatches'].append(f"{column}: expected float, got {type(value)}")
                elif expected_type == 'integer' and not isinstance(value, int):
                    issues['type_mismatches'].append(f"{column}: expected integer, got {type(value)}")
                elif expected_type == 'boolean' and not isinstance(value, bool):
                    issues['type_mismatches'].append(f"{column}: expected boolean, got {type(value)}")
                
                # Value validation
                if column.endswith('_CONF') and isinstance(value, (int, float)):
                    if not (0.0 <= value <= 1.0):
                        issues['invalid_values'].append(f"{column}: confidence must be between 0.0 and 1.0")
                
                if column.endswith('_DMG') and value not in ['Y', 'N', '']:
                    issues['invalid_values'].append(f"{column}: damage indicator must be 'Y' or 'N'")
        
        return issues
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the output schema"""
        
        return {
            'total_columns': len(self.output_schema),
            'column_types': dict(Counter(self.output_schema.values())),
            'damage_indicator_columns': len([c for c in self.output_schema.keys() 
                                           if c.startswith('BLDG_') and c.endswith('_DMG')]),
            'confidence_columns': len([c for c in self.output_schema.keys() 
                                     if c.endswith('_CONF')]),
            'schema': self.output_schema
        }