"""
Data Quality module for movie recommendation system.
Implements schema validation and drift detection for production monitoring.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

from config import config

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Monitors data quality including schema validation and drift detection"""
    
    def __init__(self, monitoring_config=None):
        """Initialize data quality monitor"""
        self.config = monitoring_config or config.monitoring
        self.baseline_stats = {}
        self.schema_definitions = self._define_schemas()
        
    def _define_schemas(self) -> Dict[str, Dict]:
        """Define expected schemas for each dataset"""
        return {
            'users': {
                'required_columns': ['user_id', 'age', 'occupation', 'gender'],
                'column_types': {
                    'user_id': 'int64',
                    'age': 'int64',
                    'occupation': 'object',
                    'gender': 'object'
                },
                'constraints': {
                    'user_id': {'min': 1, 'max': 999999, 'unique': True},
                    'age': {'min': 8, 'max': 120},
                    'gender': {'allowed_values': ['M', 'F']},
                    'occupation': {'min_length': 1}
                }
            },
            'movies': {
                'required_columns': ['movie_id', 'title'],
                'column_types': {
                    'movie_id': 'object',
                    'title': 'object',
                    'vote_average': 'float64',
                    'vote_count': 'float64'
                },
                'constraints': {
                    'movie_id': {'min_length': 1, 'unique': True},
                    'title': {'min_length': 1},
                    'vote_average': {'min': 0.0, 'max': 10.0},
                    'vote_count': {'min': 0}
                }
            },
            'interactions': {
                'required_columns': ['user_id', 'movie_id', 'total_minutes'],
                'column_types': {
                    'user_id': 'int64',
                    'movie_id': 'object',
                    'total_minutes': 'float64',
                    'rating': 'float64',
                    'final_rating': 'float64'
                },
                'constraints': {
                    'user_id': {'min': 1},
                    'movie_id': {'min_length': 1},
                    'total_minutes': {'min': 0.1, 'max': 720},  # Max 12 hours
                    'rating': {'min': 1.0, 'max': 5.0},
                    'final_rating': {'min': 1.0, 'max': 5.0}
                }
            }
        }
    
    def validate_schema(self, df: pd.DataFrame, dataset_type: str) -> Dict[str, Any]:
        """
        Validate dataset against expected schema
        
        Args:
            df: DataFrame to validate
            dataset_type: Type of dataset ('users', 'movies', 'interactions')
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Validating schema for {dataset_type} dataset...")
        
        if dataset_type not in self.schema_definitions:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        schema = self.schema_definitions[dataset_type]
        validation_results = {
            'dataset_type': dataset_type,
            'validation_timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'issues': [],
            'passed': True
        }
        
        # Check required columns
        missing_columns = self._check_required_columns(df, schema, validation_results)
        
        # Check column types
        self._check_column_types(df, schema, validation_results)
        
        # Check constraints
        self._check_constraints(df, schema, validation_results)
        
        # Check for duplicate rows
        self._check_duplicates(df, validation_results)
        
        # Overall validation status
        validation_results['passed'] = len(validation_results['issues']) == 0
        
        if validation_results['passed']:
            logger.info(f"Schema validation passed for {dataset_type}")
        else:
            logger.warning(f"Schema validation failed for {dataset_type}: {len(validation_results['issues'])} issues found")
        
        return validation_results
    
    def _check_required_columns(self, df: pd.DataFrame, schema: Dict, validation_results: Dict) -> List[str]:
        """Check if all required columns are present"""
        required_columns = schema['required_columns']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['issues'].append({
                'type': 'missing_columns',
                'severity': 'critical',
                'details': f"Missing required columns: {missing_columns}"
            })
        
        return missing_columns
    
    def _check_column_types(self, df: pd.DataFrame, schema: Dict, validation_results: Dict):
        """Check if columns have expected data types"""
        expected_types = schema.get('column_types', {})
        
        for column, expected_type in expected_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                
                # Allow for some flexibility in numeric types
                if expected_type in ['int64', 'float64'] and actual_type in ['int64', 'float64', 'int32', 'float32']:
                    continue
                
                if actual_type != expected_type:
                    validation_results['issues'].append({
                        'type': 'incorrect_type',
                        'severity': 'warning',
                        'details': f"Column '{column}' has type '{actual_type}', expected '{expected_type}'"
                    })
    
    def _check_constraints(self, df: pd.DataFrame, schema: Dict, validation_results: Dict):
        """Check value constraints for each column"""
        constraints = schema.get('constraints', {})
        
        for column, column_constraints in constraints.items():
            if column not in df.columns:
                continue
            
            # Check numeric constraints
            if 'min' in column_constraints:
                min_violations = df[df[column] < column_constraints['min']]
                if len(min_violations) > 0:
                    validation_results['issues'].append({
                        'type': 'constraint_violation',
                        'severity': 'warning',
                        'details': f"Column '{column}': {len(min_violations)} values below minimum {column_constraints['min']}"
                    })
            
            if 'max' in column_constraints:
                max_violations = df[df[column] > column_constraints['max']]
                if len(max_violations) > 0:
                    validation_results['issues'].append({
                        'type': 'constraint_violation',
                        'severity': 'warning',
                        'details': f"Column '{column}': {len(max_violations)} values above maximum {column_constraints['max']}"
                    })
            
            # Check uniqueness
            if column_constraints.get('unique', False):
                duplicates = df[column].duplicated().sum()
                if duplicates > 0:
                    validation_results['issues'].append({
                        'type': 'uniqueness_violation',
                        'severity': 'critical',
                        'details': f"Column '{column}': {duplicates} duplicate values found"
                    })
            
            # Check allowed values
            if 'allowed_values' in column_constraints:
                invalid_values = df[~df[column].isin(column_constraints['allowed_values'])]
                if len(invalid_values) > 0:
                    validation_results['issues'].append({
                        'type': 'invalid_values',
                        'severity': 'warning',
                        'details': f"Column '{column}': {len(invalid_values)} values not in allowed set {column_constraints['allowed_values']}"
                    })
            
            # Check string length
            if 'min_length' in column_constraints and df[column].dtype == 'object':
                short_values = df[df[column].str.len() < column_constraints['min_length']]
                if len(short_values) > 0:
                    validation_results['issues'].append({
                        'type': 'length_violation',
                        'severity': 'warning',
                        'details': f"Column '{column}': {len(short_values)} values shorter than minimum length {column_constraints['min_length']}"
                    })
    
    def _check_duplicates(self, df: pd.DataFrame, validation_results: Dict):
        """Check for duplicate rows"""
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation_results['issues'].append({
                'type': 'duplicate_rows',
                'severity': 'warning',
                'details': f"{duplicates} duplicate rows found"
            })
    
    def detect_drift(self, current_df: pd.DataFrame, baseline_df: pd.DataFrame, 
                    dataset_type: str) -> Dict[str, Any]:
        """
        Detect statistical drift between current and baseline datasets
        
        Args:
            current_df: Current dataset
            baseline_df: Baseline dataset for comparison
            dataset_type: Type of dataset
            
        Returns:
            Dictionary containing drift detection results
        """
        logger.info(f"Detecting drift in {dataset_type} dataset...")
        
        drift_results = {
            'dataset_type': dataset_type,
            'detection_timestamp': datetime.now().isoformat(),
            'baseline_size': len(baseline_df),
            'current_size': len(current_df),
            'drift_detected': False,
            'drift_details': []
        }
        
        # Get common numeric columns
        numeric_columns = self._get_common_numeric_columns(current_df, baseline_df)
        
        for column in numeric_columns:
            drift_score = self._calculate_drift_score(
                baseline_df[column].dropna(),
                current_df[column].dropna()
            )
            
            if drift_score > self.config.drift_detection_threshold:
                drift_results['drift_detected'] = True
                drift_results['drift_details'].append({
                    'column': column,
                    'drift_score': drift_score,
                    'threshold': self.config.drift_detection_threshold,
                    'severity': 'high' if drift_score > 0.2 else 'medium'
                })
        
        # Check categorical columns
        categorical_columns = self._get_common_categorical_columns(current_df, baseline_df)
        
        for column in categorical_columns:
            categorical_drift = self._detect_categorical_drift(
                baseline_df[column].dropna(),
                current_df[column].dropna()
            )
            
            if categorical_drift['drift_detected']:
                drift_results['drift_detected'] = True
                drift_results['drift_details'].append({
                    'column': column,
                    'drift_type': 'categorical',
                    'new_categories': categorical_drift['new_categories'],
                    'missing_categories': categorical_drift['missing_categories'],
                    'distribution_change': categorical_drift['distribution_change']
                })
        
        # Overall drift assessment
        if drift_results['drift_detected']:
            logger.warning(f"Drift detected in {dataset_type} dataset: {len(drift_results['drift_details'])} issues")
        else:
            logger.info(f"No significant drift detected in {dataset_type} dataset")
        
        return drift_results
    
    def _get_common_numeric_columns(self, df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
        """Get numeric columns common to both dataframes"""
        numeric_cols_1 = df1.select_dtypes(include=[np.number]).columns
        numeric_cols_2 = df2.select_dtypes(include=[np.number]).columns
        return list(set(numeric_cols_1) & set(numeric_cols_2))
    
    def _get_common_categorical_columns(self, df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
        """Get categorical columns common to both dataframes"""
        cat_cols_1 = df1.select_dtypes(include=['object']).columns
        cat_cols_2 = df2.select_dtypes(include=['object']).columns
        return list(set(cat_cols_1) & set(cat_cols_2))
    
    def _calculate_drift_score(self, baseline_series: pd.Series, current_series: pd.Series) -> float:
        """
        Calculate drift score using robust statistical measures
        Accounts for dataset growth and scale differences
        """
        if len(baseline_series) == 0 or len(current_series) == 0:
            return 0.0
        
        # Calculate robust statistics using percentiles
        baseline_median = baseline_series.median()
        current_median = current_series.median()
        baseline_iqr = baseline_series.quantile(0.75) - baseline_series.quantile(0.25)
        current_iqr = current_series.quantile(0.75) - current_series.quantile(0.25)
        
        # Avoid division by zero
        if baseline_iqr == 0:
            baseline_iqr = 1e-6
        
        # Relative change in central tendency
        median_shift = abs(current_median - baseline_median) / (abs(baseline_median) + 1e-6)
        
        # Relative change in spread
        spread_change = abs(current_iqr - baseline_iqr) / (baseline_iqr + 1e-6)
        
        # Account for dataset size differences
        size_factor = min(1.0, len(baseline_series) / len(current_series))
        
        # Combined drift score with size adjustment
        drift_score = (median_shift + spread_change) * size_factor
        
        return drift_score
    
    def _detect_categorical_drift(self, baseline_series: pd.Series, current_series: pd.Series) -> Dict[str, Any]:
        """Detect drift in categorical variables"""
        baseline_categories = set(baseline_series.unique())
        current_categories = set(current_series.unique())
        
        new_categories = list(current_categories - baseline_categories)
        missing_categories = list(baseline_categories - current_categories)
        
        # Calculate distribution changes for common categories
        common_categories = baseline_categories & current_categories
        distribution_changes = {}
        
        if len(common_categories) > 0:
            baseline_dist = baseline_series.value_counts(normalize=True)
            current_dist = current_series.value_counts(normalize=True)
            
            for category in common_categories:
                baseline_prop = baseline_dist.get(category, 0)
                current_prop = current_dist.get(category, 0)
                change = abs(current_prop - baseline_prop)
                if change > 0.05:  # 5% threshold
                    distribution_changes[category] = {
                        'baseline_proportion': baseline_prop,
                        'current_proportion': current_prop,
                        'absolute_change': change
                    }
        
        drift_detected = len(new_categories) > 0 or len(missing_categories) > 0 or len(distribution_changes) > 0
        
        return {
            'drift_detected': drift_detected,
            'new_categories': new_categories,
            'missing_categories': missing_categories,
            'distribution_change': distribution_changes
        }
    
    def create_baseline(self, df: pd.DataFrame, dataset_type: str):
        """Create baseline statistics for drift detection"""
        logger.info(f"Creating baseline statistics for {dataset_type}...")
        
        baseline_stats = {
            'dataset_type': dataset_type,
            'creation_timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        # Numeric column statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            series = df[column].dropna()
            baseline_stats['numeric_stats'][column] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'median': series.median(),
                'percentiles': {
                    '25': series.quantile(0.25),
                    '75': series.quantile(0.75),
                    '95': series.quantile(0.95)
                }
            }
        
        # Categorical column statistics
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            series = df[column].dropna()
            value_counts = series.value_counts()
            baseline_stats['categorical_stats'][column] = {
                'unique_values': len(value_counts),
                'top_categories': value_counts.head(10).to_dict(),
                'value_distribution': value_counts.to_dict()
            }
        
        self.baseline_stats[dataset_type] = baseline_stats
        logger.info(f"Baseline created for {dataset_type} with {len(numeric_columns)} numeric and {len(categorical_columns)} categorical columns")
    
    def save_baseline(self, filepath: str):
        """Save baseline statistics to file"""
        with open(filepath, 'w') as f:
            json.dump(self.baseline_stats, f, indent=2, default=str)
        logger.info(f"Baseline statistics saved to {filepath}")
    
    def load_baseline(self, filepath: str):
        """Load baseline statistics from file"""
        with open(filepath, 'r') as f:
            self.baseline_stats = json.load(f)
        logger.info(f"Baseline statistics loaded from {filepath}")
    
    def generate_data_quality_report(self, validation_results: List[Dict], 
                                   drift_results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'validation_summary': {
                'total_datasets': len(validation_results),
                'passed_validation': sum(1 for r in validation_results if r['passed']),
                'failed_validation': sum(1 for r in validation_results if not r['passed'])
            },
            'drift_summary': {
                'total_datasets': len(drift_results),
                'drift_detected': sum(1 for r in drift_results if r['drift_detected']),
                'no_drift': sum(1 for r in drift_results if not r['drift_detected'])
            },
            'detailed_results': {
                'validation': validation_results,
                'drift_detection': drift_results
            }
        }
        
        # Determine overall health status
        critical_issues = sum(1 for result in validation_results 
                            for issue in result.get('issues', []) 
                            if issue.get('severity') == 'critical')
        
        high_drift_count = sum(1 for result in drift_results 
                             for detail in result.get('drift_details', [])
                             if detail.get('severity') == 'high')
        
        if critical_issues > 0 or high_drift_count > 0:
            report['overall_status'] = 'critical'
        elif report['validation_summary']['failed_validation'] > 0 or report['drift_summary']['drift_detected'] > 0:
            report['overall_status'] = 'warning'
        
        return report


def validate_and_monitor_data(users_df: pd.DataFrame, movies_df: pd.DataFrame, 
                            interactions_df: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function to validate and monitor all datasets"""
    monitor = DataQualityMonitor()
    
    # Validate schemas
    validation_results = [
        monitor.validate_schema(users_df, 'users'),
        monitor.validate_schema(movies_df, 'movies'),
        monitor.validate_schema(interactions_df, 'interactions')
    ]
    
    # Create baselines (in production, these would be loaded from storage)
    monitor.create_baseline(users_df, 'users')
    monitor.create_baseline(movies_df, 'movies')
    monitor.create_baseline(interactions_df, 'interactions')
    
    # Note: Drift detection would require historical data
    # For now, returning validation results only
    drift_results = []
    
    return monitor.generate_data_quality_report(validation_results, drift_results)


if __name__ == "__main__":
    # Test data quality monitoring
    from data_loader import load_and_validate_data
    from data_preprocessor import preprocess_data
    
    try:
        # Load data
        users_df, movies_df, interactions_df = load_and_validate_data()
        users_clean, movies_clean, interactions_clean = preprocess_data(users_df, movies_df, interactions_df)
        
        # Load baseline interactions data
        baseline_interactions = pd.read_csv('baseline_data/baseline_interactions.csv')
        
        # Initialize monitor
        monitor = DataQualityMonitor()
        
        # Validate schemas
        validation_results = [
            monitor.validate_schema(users_clean, 'users'),
            monitor.validate_schema(movies_clean, 'movies'),
            monitor.validate_schema(interactions_clean, 'interactions')
        ]
        
        # Perform drift detection for interactions
        drift_results = [
            monitor.detect_drift(interactions_clean, baseline_interactions, 'interactions')
        ]
        
        # Generate report
        report = monitor.generate_data_quality_report(validation_results, drift_results)
        
        # Save report to file
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"data_quality_report_{timestamp}.json")
        
        # Save the report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nData quality report saved to: {report_path}")
        
        # Print drift details
        print("\nDrift Detection Details:")
        for drift_result in drift_results:
            print(f"\nDataset: {drift_result['dataset_type']}")
            print(f"Drift detected: {drift_result['drift_detected']}")
            if drift_result['drift_details']:
                print("\nDrift details:")
                for detail in drift_result['drift_details']:
                    if 'column' in detail:
                        print(f"\nColumn: {detail['column']}")
                        if 'drift_score' in detail:
                            print(f"Drift score: {detail['drift_score']:.4f}")
                            print(f"Severity: {detail['severity']}")
                        if 'drift_type' in detail:
                            print(f"Type: {detail['drift_type']}")
                            if detail.get('new_categories'):
                                print(f"New categories: {detail['new_categories']}")
                            if detail.get('missing_categories'):
                                print(f"Missing categories: {detail['missing_categories']}")
                            if detail.get('distribution_change'):
                                print("Distribution changes:")
                                for cat, changes in detail['distribution_change'].items():
                                    print(f"  {cat}: {changes['absolute_change']:.4f} change")
        print(f"\nData quality report saved to: {report_path}")
        
        print("\nData quality monitoring completed!")
        print(f"Overall status: {report['overall_status']}")
        print(f"Validation passed: {report['validation_summary']['passed_validation']}/{report['validation_summary']['total_datasets']}")
        
        # Print any issues found
        for result in report['detailed_results']['validation']:
            if not result['passed']:
                print(f"\nIssues in {result['dataset_type']}:")
                for issue in result['issues']:
                    print(f"  - {issue['severity']}: {issue['details']}")
        
    except Exception as e:
        print(f"Data quality monitoring failed: {e}")
        exit(1)
