"""
Additional focused tests to boost coverage for Milestone 2
Targets specific functions and edge cases in core modules
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Import core modules
from config import Config
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor  
from feature_engineer import FeatureEngineer
from data_quality import DataQualityMonitor
from data_splitter import DataSplitter
from svd_model_trainer import SVDModelTrainer
from model_evaluator import ModelEvaluator
from main_pipeline import Pipeline


class TestDataLoaderCoverage:
    """Focused tests to improve DataLoader coverage"""
    
    def test_load_datasets_file_not_found(self):
        """Test error handling when files don't exist"""
        config = Config()
        config.data.users_file = "nonexistent_users.csv"
        config.data.movies_file = "nonexistent_movies.csv"
        config.data.interactions_file = "nonexistent_interactions.csv"
        
        loader = DataLoader(config)
        
        with pytest.raises((FileNotFoundError, Exception)):
            loader.load_datasets()

    def test_validate_users_data_invalid(self):
        """Test user data validation with invalid data"""
        config = Config()
        loader = DataLoader(config)
        
        # Invalid users data - missing required columns
        invalid_users = pd.DataFrame({
            'wrong_column': [1, 2, 3]
        })
        
        is_valid = loader._validate_users_data(invalid_users)
        assert not is_valid

    def test_validate_movies_data_invalid(self):
        """Test movie data validation with invalid data"""
        config = Config()
        loader = DataLoader(config)
        
        # Invalid movies data
        invalid_movies = pd.DataFrame({
            'wrong_movie_column': ['movie1', 'movie2']
        })
        
        is_valid = loader._validate_movies_data(invalid_movies)
        assert not is_valid

    def test_validate_interactions_data_invalid(self):
        """Test interactions data validation with invalid data"""
        config = Config()
        loader = DataLoader(config)
        
        # Invalid interactions data
        invalid_interactions = pd.DataFrame({
            'wrong_interaction_column': [1, 2, 3]
        })
        
        is_valid = loader._validate_interactions_data(invalid_interactions)
        assert not is_valid


class TestDataPreprocessorCoverage:
    """Focused tests to improve DataPreprocessor coverage"""
    
    def test_preprocess_movies(self, sample_movies_data):
        """Test movie preprocessing"""
        preprocessor = DataPreprocessor()
        processed = preprocessor.preprocess_movies(sample_movies_data)
        
        assert isinstance(processed, pd.DataFrame)
        assert len(processed) > 0
        assert 'movie_id' in processed.columns

    def test_preprocess_datasets_comprehensive(self, sample_users_data, sample_movies_data, sample_interactions_data):
        """Test comprehensive dataset preprocessing"""
        preprocessor = DataPreprocessor()
        
        users_clean, movies_clean, interactions_clean = preprocessor.preprocess_datasets(
            sample_users_data, sample_movies_data, sample_interactions_data
        )
        
        # Test all outputs
        assert isinstance(users_clean, pd.DataFrame)
        assert isinstance(movies_clean, pd.DataFrame)
        assert isinstance(interactions_clean, pd.DataFrame)
        
        # Test data quality
        assert len(users_clean) > 0
        assert len(movies_clean) > 0
        assert len(interactions_clean) > 0

    def test_filter_low_activity_users(self):
        """Test filtering of low activity users"""
        preprocessor = DataPreprocessor()
        
        # Create test data with low activity users
        test_data = pd.DataFrame({
            'user_id': [1, 1, 2, 3, 3, 3, 3],  # User 2 has only 1 interaction
            'movie_id': ['m1', 'm2', 'm1', 'm1', 'm2', 'm3', 'm4'],
            'final_rating': [4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0]
        })
        
        filtered = preprocessor._filter_low_activity_users(test_data, min_interactions=2)
        
        # User 2 should be filtered out
        assert 2 not in filtered['user_id'].values
        assert len(filtered) < len(test_data)

    def test_filter_low_activity_movies(self):
        """Test filtering of low activity movies"""
        preprocessor = DataPreprocessor()
        
        # Create test data with low activity movies
        test_data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5, 6, 7],
            'movie_id': ['m1', 'm1', 'm1', 'm2', 'm3', 'm3', 'm3'],  # m2 has only 1 interaction
            'final_rating': [4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0]
        })
        
        filtered = preprocessor._filter_low_activity_movies(test_data, min_interactions=2)
        
        # Movie m2 should be filtered out
        assert 'm2' not in filtered['movie_id'].values
        assert len(filtered) < len(test_data)


class TestFeatureEngineerCoverage:
    """Focused tests to improve FeatureEngineer coverage"""
    
    def test_create_features_comprehensive(self):
        """Test comprehensive feature creation"""
        engineer = FeatureEngineer()
        
        # Create realistic test data
        test_data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5] * 20,
            'movie_id': [f'movie{i}' for i in range(1, 6)] * 20,
            'total_minutes': np.random.randint(30, 300, 100),
            'rating': [4.0, np.nan, 5.0, np.nan, 3.0] * 20,
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
        })
        
        features = engineer.create_features(test_data)
        
        assert isinstance(features, pd.DataFrame)
        assert 'implicit_rating' in features.columns
        assert 'final_rating' in features.columns
        assert len(features) > 0

    def test_create_implicit_rating_edge_cases(self):
        """Test implicit rating creation with edge cases"""
        engineer = FeatureEngineer()
        
        # Test with all explicit ratings
        all_explicit = pd.DataFrame({
            'user_id': [1, 2, 3],
            'movie_id': ['m1', 'm2', 'm3'],
            'total_minutes': [120, 150, 90],
            'rating': [4.0, 5.0, 3.0]  # All explicit
        })
        
        result = engineer._create_implicit_rating(all_explicit)
        assert 'implicit_rating' in result.columns

    def test_create_final_rating_edge_cases(self):
        """Test final rating creation with edge cases"""
        engineer = FeatureEngineer()
        
        # Test with mixed data
        mixed_data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'movie_id': ['m1', 'm2', 'm3', 'm4', 'm5'],
            'rating': [4.0, np.nan, 5.0, np.nan, 3.0],
            'implicit_rating': [3.5, 4.2, 4.8, 2.1, 3.8]
        })
        
        result = engineer._create_final_rating(mixed_data)
        assert 'final_rating' in result.columns
        
        # Check that explicit ratings are preserved
        assert result.loc[result['rating'].notna(), 'final_rating'].equals(
            result.loc[result['rating'].notna(), 'rating']
        )


class TestDataQualityCoverage:
    """Focused tests to improve DataQualityMonitor coverage"""
    
    def test_detect_drift_basic(self):
        """Test basic drift detection"""
        monitor = DataQualityMonitor()
        
        # Create baseline and current data
        baseline = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'total_minutes': [120, 130, 140, 150, 160],
            'rating': [4.0, 4.5, 5.0, 3.5, 4.0]
        })
        
        # Current data with slight drift
        current = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'total_minutes': [125, 135, 145, 155, 165],  # Slight increase
            'rating': [4.2, 4.7, 5.0, 3.7, 4.2]
        })
        
        drift_results = monitor.detect_drift(baseline, current)
        assert isinstance(drift_results, dict)
        assert 'drift_detected' in drift_results

    def test_statistical_tests(self):
        """Test statistical test methods"""
        monitor = DataQualityMonitor()
        
        # Test data
        baseline_data = np.random.normal(100, 15, 100)
        current_data = np.random.normal(105, 15, 100)  # Slight shift
        
        # Test KS test
        ks_stat, ks_p = monitor._ks_test(baseline_data, current_data)
        assert isinstance(ks_stat, float)
        assert isinstance(ks_p, float)
        
        # Test chi-square test for categorical data
        baseline_cat = pd.Series(['A', 'B', 'C'] * 30)
        current_cat = pd.Series(['A', 'A', 'B', 'C'] * 22 + ['C', 'C'])
        
        chi2_stat, chi2_p = monitor._chi2_test(baseline_cat, current_cat)
        assert isinstance(chi2_stat, float)
        assert isinstance(chi2_p, float)

    def test_generate_comprehensive_quality_report(self):
        """Test comprehensive quality report generation"""
        monitor = DataQualityMonitor()
        
        test_data = pd.DataFrame({
            'user_id': range(1, 101),
            'movie_id': [f'movie{i%20}' for i in range(1, 101)],
            'total_minutes': np.random.randint(30, 300, 100),
            'rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], 100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
        })
        
        report = monitor.generate_quality_report(test_data)
        
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'total_rows' in report
        assert 'missing_values' in report
        assert 'data_types' in report


class TestDataSplitterCoverage:
    """Focused tests to improve DataSplitter coverage"""
    
    def test_temporal_split_edge_cases(self):
        """Test temporal split with edge cases"""
        splitter = DataSplitter()
        
        # Test with very small dataset
        small_data = pd.DataFrame({
            'user_id': [1, 2],
            'movie_id': ['m1', 'm2'],
            'final_rating': [4.0, 5.0],
            'timestamp': pd.date_range('2024-01-01', periods=2)
        })
        
        train, test = splitter.temporal_split(small_data, test_size=0.5)
        assert len(train) >= 1
        assert len(test) >= 1

    def test_random_split(self):
        """Test random split functionality"""
        splitter = DataSplitter()
        
        test_data = pd.DataFrame({
            'user_id': range(1, 21),
            'movie_id': [f'movie{i%5}' for i in range(1, 21)],
            'final_rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], 20)
        })
        
        train, test = splitter.random_split(test_data, test_size=0.3)
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert len(train) + len(test) == len(test_data)
        assert len(test) / len(test_data) == pytest.approx(0.3, abs=0.1)


class TestSVDModelTrainerCoverage:
    """Focused tests to improve SVDModelTrainer coverage"""
    
    def test_create_user_item_matrix(self):
        """Test user-item matrix creation"""
        config = Config()
        trainer = SVDModelTrainer(config)
        
        test_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'movie_id': ['m1', 'm2', 'm1', 'm3', 'm2'],
            'final_rating': [4.0, 5.0, 3.0, 4.0, 5.0]
        })
        
        user_item_matrix, user_map, item_map = trainer._create_user_item_matrix(test_data)
        
        assert user_item_matrix.shape[0] == len(test_data['user_id'].unique())
        assert user_item_matrix.shape[1] == len(test_data['movie_id'].unique())
        assert isinstance(user_map, dict)
        assert isinstance(item_map, dict)

    def test_save_and_load_model(self, minimal_training_data, tmp_path):
        """Test model saving and loading"""
        config = Config()
        config.model.n_components = 5
        config.model.max_iter = 10
        config.data.models_dir = str(tmp_path)
        
        trainer = SVDModelTrainer(config)
        model = trainer.train(minimal_training_data)
        
        # Test save
        model_path = trainer.save_model(model, "test_model")
        assert os.path.exists(model_path)
        
        # Test load
        loaded_model = trainer.load_model("test_model")
        assert loaded_model is not None
        
        # Test that loaded model works
        prediction = loaded_model.predict_rating(1, 'movie1')
        assert isinstance(prediction, (int, float))


class TestModelEvaluatorCoverage:
    """Focused tests to improve ModelEvaluator coverage"""
    
    def test_calculate_recall_at_k_variations(self):
        """Test recall@k calculation with different k values"""
        evaluator = ModelEvaluator()
        
        # Create test data
        test_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'movie_id': ['m1', 'm2', 'm3', 'm4', 'm5', 'm6'],
            'final_rating': [5.0, 4.0, 5.0, 4.0, 5.0, 4.0]
        })
        
        # Create mock model
        mock_model = Mock()
        mock_model.recommend_movies.return_value = ['m1', 'm2', 'm3', 'm4', 'm5']
        
        # Test different k values
        for k in [5, 10, 20]:
            recall = evaluator._calculate_recall_at_k(test_data, mock_model, k=k)
            assert isinstance(recall, float)
            assert 0 <= recall <= 1

    def test_calculate_precision_at_k(self):
        """Test precision@k calculation"""
        evaluator = ModelEvaluator()
        
        test_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'movie_id': ['m1', 'm2', 'm3', 'm4'],
            'final_rating': [5.0, 4.0, 5.0, 4.0]
        })
        
        mock_model = Mock()
        mock_model.recommend_movies.return_value = ['m1', 'm2', 'm5', 'm6', 'm7']
        
        precision = evaluator._calculate_precision_at_k(test_data, mock_model, k=5)
        assert isinstance(precision, float)
        assert 0 <= precision <= 1

    def test_evaluate_model_comprehensive(self, minimal_training_data):
        """Test comprehensive model evaluation"""
        evaluator = ModelEvaluator()
        
        # Create simple mock model for testing
        mock_model = Mock()
        mock_model.recommend_movies.return_value = ['movie1', 'movie2', 'movie3']
        mock_model.predict_rating.return_value = 4.0
        
        # Split data for evaluation
        train_data = minimal_training_data.head(100)
        test_data = minimal_training_data.tail(50)
        
        results = evaluator.evaluate_model(mock_model, train_data, test_data)
        
        assert isinstance(results, dict)
        expected_keys = ['recall_at_k', 'precision_at_k', 'coverage']
        for key in expected_keys:
            assert key in results


class TestMainPipelineCoverage:
    """Focused tests to improve MainPipeline coverage"""
    
    def test_pipeline_initialization_with_config(self):
        """Test pipeline initialization with custom config"""
        config = Config()
        config.model.n_components = 10
        
        pipeline = Pipeline(config)
        assert pipeline.config.model.n_components == 10

    def test_run_data_quality_checks(self, sample_interactions_data):
        """Test data quality checks in pipeline"""
        config = Config()
        pipeline = Pipeline(config)
        
        # Mock the quality monitor
        with patch.object(pipeline, 'quality_monitor') as mock_monitor:
            mock_monitor.generate_quality_report.return_value = {
                'status': 'success',
                'total_rows': len(sample_interactions_data),
                'issues_found': 0
            }
            
            result = pipeline._run_quality_checks(sample_interactions_data)
            assert isinstance(result, dict)
            assert 'status' in result

    @patch('main_pipeline.DataLoader')
    @patch('main_pipeline.DataPreprocessor') 
    @patch('main_pipeline.FeatureEngineer')
    def test_load_and_preprocess_data(self, mock_engineer, mock_preprocessor, mock_loader):
        """Test data loading and preprocessing in pipeline"""
        config = Config()
        pipeline = Pipeline(config)
        
        # Configure mocks
        mock_loader.return_value.load_datasets.return_value = (
            pd.DataFrame({'user_id': [1]}),
            pd.DataFrame({'movie_id': ['m1']}), 
            pd.DataFrame({'user_id': [1], 'movie_id': ['m1']})
        )
        
        mock_preprocessor.return_value.preprocess_datasets.return_value = (
            pd.DataFrame({'user_id': [1]}),
            pd.DataFrame({'movie_id': ['m1']}),
            pd.DataFrame({'user_id': [1], 'movie_id': ['m1']})
        )
        
        mock_engineer.return_value.create_features.return_value = pd.DataFrame({
            'user_id': [1], 'movie_id': ['m1'], 'final_rating': [4.0]
        })
        
        result = pipeline._load_and_preprocess_data()
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=term-missing"])
