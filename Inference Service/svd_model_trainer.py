"""
SVD Model Training module for movie recommendation system.
FINAL FIXED VERSION - Removes deprecated max_iter parameter.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
import time
import pickle
import os
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from config import config
from data_splitter import temporal_train_test_split, TemporalSplitResult, temporal_train_val_test_split, TemporalSplitResult3Way

logger = logging.getLogger(__name__)


class SVDModelTrainer:
    """Handles SVD model training for collaborative filtering - FINAL FIXED VERSION"""
    
    def __init__(self, model_config=None):
        """Initialize SVD trainer with configuration"""
        self.config = model_config or config.model
        self.model = None
        self.user_mapper = None
        self.movie_mapper = None
        self.reverse_user_mapper = None
        self.reverse_movie_mapper = None
        self.global_mean = None
        self.user_factors = None  # Store user factors
        self.item_factors = None  # Store item factors
        self.training_stats = {}
        self.temporal_split_result: Optional[TemporalSplitResult] = None
        self.temporal_split_result_3way: Optional[TemporalSplitResult3Way] = None
        self.train_interactions_by_user: Dict[int, set] = {}
        self.item_popularity: Optional[pd.Series] = None
        self.best_n_components: Optional[int] = None  # Store best hyperparameter
        
    def train_model(self, training_df: pd.DataFrame, perform_split: bool = True) -> Dict[str, Any]:
        """
        Train SVD model on the dataset with optional hyperparameter tuning
        
        Args:
            training_df: DataFrame with user_id, movie_id, final_rating columns
            perform_split: Whether to create an internal validation split (default: True)
            
        Returns:
            Dictionary containing training results and model metrics
        """
        logger.info("Starting SVD model training...")
        start_time = time.time()
        
        # Reset state before training
        self.training_stats = {}
        self.temporal_split_result = None
        self.temporal_split_result_3way = None
        self.train_interactions_by_user = {}
        
        if perform_split and self.config.enable_hyperparameter_tuning:
            # Use 3-way split for hyperparameter tuning
            results = self._train_with_hyperparameter_tuning(training_df)
            results['training_time_seconds'] = time.time() - start_time
            return results
        elif perform_split:
            # Legacy 2-way split (no tuning)
            train_data, test_data = self._prepare_training_data_legacy(training_df)
            self._train_on_data(train_data)
            
            training_time = time.time() - start_time
            self.training_stats['training_time_seconds'] = training_time
            self.training_stats['train_rows'] = len(train_data)
            
            results = self._evaluate_model(test_data)
            results['training_time_seconds'] = training_time
            results['split_strategy'] = self.training_stats.get('split_strategy', 'random')
            if self.temporal_split_result is not None:
                results['temporal_cutoff'] = self.temporal_split_result.cutoff_timestamp.isoformat()
                results['temporal_split_metadata'] = self.temporal_split_result.metadata
                results['temporal_cold_user_rows'] = len(self.temporal_split_result.cold_user_test)
                results['temporal_cold_item_rows'] = len(self.temporal_split_result.cold_item_test)
        else:
            # No split - train on full dataset
            logger.info("Training on full dataset without holdout split.")
            train_data = training_df.copy()
            self._train_on_data(train_data)
            
            training_time = time.time() - start_time
            self.training_stats['training_time_seconds'] = training_time
            self.training_stats['split_strategy'] = 'none'
            
            results = {
                'training_time_seconds': training_time,
                'split_strategy': 'none',
                'model_size_mb': self._calculate_model_size(),
                'inference_time_ms': self._measure_inference_time(),
            }
        
        logger.info(f"SVD model training completed in {time.time() - start_time:.2f} seconds")
        return results
    
    def _train_with_hyperparameter_tuning(self, training_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train model with hyperparameter tuning using validation set (60/20/20 split)
        
        Args:
            training_df: Full dataset with timestamps
            
        Returns:
            Dictionary with training results including best hyperparameters
        """
        logger.info("="*80)
        logger.info("HYPERPARAMETER TUNING ENABLED - Using 60/20/20 temporal split")
        logger.info("="*80)
        
        # Create 3-way temporal split
        if 'timestamp' not in training_df.columns:
            logger.warning("No timestamp column found. Cannot perform temporal 3-way split.")
            logger.warning("Falling back to legacy 2-way split without hyperparameter tuning.")
            return self._train_with_hyperparameter_tuning_fallback(training_df)
        
        try:
            split = temporal_train_val_test_split(
                training_df,
                timestamp_col='timestamp',
                train_fraction=self.config.train_fraction,
                val_fraction=self.config.val_fraction,
                ensure_warm_start=True,
            )
            self.temporal_split_result_3way = split
            self.training_stats['split_strategy'] = 'temporal_3way'
            self.training_stats['train_cutoff'] = split.train_cutoff_timestamp.isoformat()
            self.training_stats['val_cutoff'] = split.val_cutoff_timestamp.isoformat()
            
            logger.info(f"Train set: {len(split.train):,} interactions (60%)")
            logger.info(f"Validation set: {len(split.validation):,} interactions (20%)")
            logger.info(f"Test set: {len(split.test):,} interactions (20%)")
            logger.info(f"Train cutoff: {split.train_cutoff_timestamp}")
            logger.info(f"Validation cutoff: {split.val_cutoff_timestamp}")
            
        except ValueError as exc:
            logger.warning(f"3-way temporal split failed: {exc}")
            logger.warning("Falling back to legacy 2-way split without hyperparameter tuning.")
            return self._train_with_hyperparameter_tuning_fallback(training_df)
        
        # Hyperparameter search
        best_rmse = float('inf')
        best_n_components = self.config.n_components
        search_results = []
        
        logger.info("\n" + "="*80)
        logger.info("HYPERPARAMETER SEARCH: Testing different n_components values")
        logger.info("="*80)
        
        for n_comp in self.config.n_components_search:
            logger.info(f"\n--- Testing n_components={n_comp} ---")
            
            # Temporarily set n_components
            original_n_comp = self.config.n_components
            self.config.n_components = n_comp
            
            # Train model on train set only
            self._train_on_data(split.train)
            
            # Evaluate on validation set
            val_rmse = self._evaluate_on_validation(split.validation)
            
            search_results.append({
                'n_components': n_comp,
                'validation_rmse': val_rmse,
            })
            
            logger.info(f"n_components={n_comp} -> Validation RMSE: {val_rmse:.4f}")
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_n_components = n_comp
                logger.info(f"  ✓ New best model!")
            
            # Restore original value
            self.config.n_components = original_n_comp
        
        logger.info("\n" + "="*80)
        logger.info(f"BEST HYPERPARAMETERS: n_components={best_n_components} (RMSE={best_rmse:.4f})")
        logger.info("="*80)
        
        # Train final model with best hyperparameters on train set
        self.config.n_components = best_n_components
        self.best_n_components = best_n_components
        self._train_on_data(split.train)
        
        # Evaluate on test set (ONLY ONCE - final evaluation)
        logger.info("\n" + "="*80)
        logger.info("FINAL EVALUATION ON TEST SET (Never seen during training/tuning)")
        logger.info("="*80)
        
        test_results = self._evaluate_model(split.test)
        
        # Combine results
        results = {
            **test_results,
            'split_strategy': 'temporal_3way_with_tuning',
            'best_n_components': best_n_components,
            'validation_rmse': best_rmse,
            'hyperparameter_search_results': search_results,
            'train_cutoff': split.train_cutoff_timestamp.isoformat(),
            'val_cutoff': split.val_cutoff_timestamp.isoformat(),
            'train_size': len(split.train),
            'val_size': len(split.validation),
            'test_size': len(split.test),
        }
        
        logger.info(f"Final Test RMSE: {test_results.get('accuracy_rmse', 'N/A'):.4f}")
        logger.info(f"Model trained with n_components={best_n_components}")
        
        return results
    
    def _train_with_hyperparameter_tuning_fallback(self, training_df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback to legacy 2-way split if 3-way split fails"""
        train_data, test_data = self._prepare_training_data_legacy(training_df)
        self._train_on_data(train_data)
        results = self._evaluate_model(test_data)
        results['split_strategy'] = 'fallback_2way'
        return results
    
    def _evaluate_on_validation(self, validation_data: pd.DataFrame) -> float:
        """
        Evaluate model on validation set (returns RMSE only for speed)
        
        Args:
            validation_data: Validation interactions
            
        Returns:
            RMSE on validation set
        """
        if len(validation_data) == 0:
            return float('inf')
        
        # Sample for faster evaluation
        eval_sample_size = min(5000, len(validation_data))
        val_sample = validation_data.sample(n=eval_sample_size, random_state=self.config.random_state)
        
        predictions = []
        actuals = []
        
        for _, row in val_sample.iterrows():
            pred = self.predict_rating(row['user_id'], row['movie_id'])
            predictions.append(pred)
            actuals.append(row['final_rating'])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        return rmse
    
    def _prepare_training_data_legacy(self, training_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Legacy 2-way split (80/20) - kept for backward compatibility"""
        logger.info("Using legacy 2-way split (80/20)...")
        
        # Prefer temporal split when timestamp information is available
        if 'timestamp' in training_df.columns:
            try:
                split = temporal_train_test_split(
                    training_df,
                    timestamp_col='timestamp',
                    train_fraction=self.config.train_test_split_ratio,
                    ensure_warm_start=True,
                )
                self.temporal_split_result = split
                self.training_stats['split_strategy'] = 'temporal'
                self.training_stats['temporal_cutoff'] = split.cutoff_timestamp.isoformat()
                self.training_stats['temporal_split_metadata'] = split.metadata
                self.training_stats['temporal_cold_user_rows'] = len(split.cold_user_test)
                self.training_stats['temporal_cold_item_rows'] = len(split.cold_item_test)
                
                logger.info(
                    "Using temporal split with cutoff %s (train=%s, warm_test=%s)",
                    split.cutoff_timestamp,
                    len(split.train),
                    len(split.warm_test),
                )
                return split.train, split.warm_test
            except ValueError as exc:
                logger.warning(
                    "Temporal split failed (%s). Falling back to random split.", exc
                )
        
        # Fallback: random split (original behaviour)
        train_data, test_data = train_test_split(
            training_df,
            test_size=1 - self.config.train_test_split_ratio,
            random_state=self.config.random_state,
            stratify=None  # Cannot stratify on continuous ratings
        )
        
        self.training_stats['split_strategy'] = 'random'
        logger.info(f"Train set: {len(train_data):,} interactions")
        logger.info(f"Test set: {len(test_data):,} interactions")
        
        return train_data, test_data
    
    def _train_on_data(self, train_data: pd.DataFrame):
        """Train SVD model on provided training interactions without creating a holdout split."""
        if train_data.empty:
            raise ValueError("Training data is empty. Cannot train SVD model.")
        
        # Track per-user history for later filtering
        self.train_interactions_by_user = train_data.groupby('user_id')['movie_id'].agg(lambda x: set(x)).to_dict()
        self.item_popularity = train_data['movie_id'].value_counts()
        
        # Create user-item matrix
        user_item_matrix, self.user_mapper, self.movie_mapper = self._create_user_item_matrix(train_data)
        
        # Create reverse mappers
        self.reverse_user_mapper = {idx: user for user, idx in self.user_mapper.items()}
        self.reverse_movie_mapper = {idx: movie for movie, idx in self.movie_mapper.items()}
        
        # Train model on the full matrix
        self.model = self._train_svd(user_item_matrix)
    
    def _create_user_item_matrix(self, train_data: pd.DataFrame) -> Tuple[csr_matrix, Dict, Dict]:
        """Create sparse user-item matrix for SVD training"""
        logger.info("Creating user-item matrix...")
        
        # Create mappings for users and movies to matrix indices
        unique_users = train_data['user_id'].unique()
        unique_movies = train_data['movie_id'].unique()
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
        
        # Map user and movie IDs to matrix indices
        user_indices = train_data['user_id'].map(user_to_idx)
        movie_indices = train_data['movie_id'].map(movie_to_idx)
        ratings = train_data['final_rating'].values
        
        # Calculate global mean rating
        self.global_mean = ratings.mean()
        
        # Create sparse matrix
        n_users = len(unique_users)
        n_movies = len(unique_movies)
        
        user_item_matrix = csr_matrix(
            (ratings, (user_indices, movie_indices)),
            shape=(n_users, n_movies)
        )
        
        # Calculate sparsity
        sparsity = 1 - (user_item_matrix.nnz / (n_users * n_movies))
        
        logger.info(f"Matrix dimensions: {n_users:,} users × {n_movies:,} movies")
        logger.info(f"Matrix sparsity: {sparsity:.2%}")
        logger.info(f"Global mean rating: {self.global_mean:.2f}")
        
        return user_item_matrix, user_to_idx, movie_to_idx
    
    def _train_svd(self, user_item_matrix: csr_matrix) -> TruncatedSVD:
        """Train SVD model on user-item matrix"""
        logger.info(f"Training SVD with {self.config.n_components} components...")
        
        # Initialize SVD model - FIXED: Removed max_iter parameter
        svd = TruncatedSVD(
            n_components=self.config.n_components,
            random_state=self.config.random_state
        )
        
        # Fit model and get user factors
        self.user_factors = svd.fit_transform(user_item_matrix)
        self.item_factors = svd.components_.T  # Transpose to get (n_movies, n_components)
        
        logger.info("SVD training completed")
        logger.info(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
        
        return svd
    
    def predict_rating(self, user_id: int, movie_id: str) -> float:
        """
        Predict rating for a user-movie pair - FIXED VERSION
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating (float)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Check if user and movie are in training data
        if user_id not in self.user_mapper or movie_id not in self.movie_mapper:
            return self.global_mean  # Return global mean for unknown users/movies
        
        # Get matrix indices
        user_idx = self.user_mapper[user_id]
        movie_idx = self.movie_mapper[movie_id]
        
        # Get user and movie factors - FIXED LOGIC
        user_vector = self.user_factors[user_idx, :]  # Shape: (n_components,)
        movie_vector = self.item_factors[movie_idx, :]  # Shape: (n_components,)
        
        # Predict rating
        predicted_rating = np.dot(user_vector, movie_vector) + self.global_mean
        
        # Clip to valid rating range
        predicted_rating = np.clip(predicted_rating, 1.0, 5.0)
        
        return predicted_rating
    
    def recommend_movies(self, user_id: int, n_recommendations: int = 20, exclude_seen: Optional[set] = None) -> list:
        """
        Get movie recommendations for a user - FIXED VERSION
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to return
            exclude_seen: Optional iterable of movie IDs to exclude from results
            
        Returns:
            List of recommended movie IDs
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        if user_id not in self.user_mapper:
            # Return most popular movies for unknown users
            return self._get_popular_movies(n_recommendations)
        
        user_idx = self.user_mapper[user_id]
        
        # Get user vector - FIXED LOGIC
        user_vector = self.user_factors[user_idx, :]  # Shape: (n_components,)
        
        # Calculate scores for all movies
        # item_factors has shape (n_movies, n_components)
        scores = np.dot(self.item_factors, user_vector)  # Shape: (n_movies,)
        
        # Build exclusion set (seen items, optional explicit exclusions)
        seen_items = set()
        if self.train_interactions_by_user:
            seen_items.update(self.train_interactions_by_user.get(user_id, set()))
        if exclude_seen:
            seen_items.update(exclude_seen)
        
        # Iterate over sorted scores, skipping excluded items
        recommendations = []
        for idx in np.argsort(scores)[::-1]:
            movie_id = self.reverse_movie_mapper[idx]
            if movie_id in seen_items:
                continue
            if movie_id in recommendations:
                continue
            recommendations.append(movie_id)
            if len(recommendations) >= n_recommendations:
                break
        
        # Fallback with popular movies if we could not fill the quota
        if len(recommendations) < n_recommendations:
            fallback = self._get_popular_movies(n_recommendations * 2)
            for movie_id in fallback:
                if movie_id in seen_items or movie_id in recommendations:
                    continue
                recommendations.append(movie_id)
                if len(recommendations) >= n_recommendations:
                    break
        
        return recommendations
    
    def _get_popular_movies(self, n_recommendations: int) -> list:
        """Get most popular movies as fallback recommendations"""
        if self.item_popularity is not None and not self.item_popularity.empty:
            popular_ids = [movie for movie in self.item_popularity.index if movie in self.movie_mapper]
        else:
            popular_ids = list(self.movie_mapper.keys())
        
        recommendations = []
        for movie_id in popular_ids:
            if movie_id in recommendations:
                continue
            recommendations.append(movie_id)
            if len(recommendations) >= n_recommendations:
                break
        
        return recommendations[:n_recommendations]
    
    def _evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        logger.info("Evaluating model performance...")
        
        # Prediction accuracy metrics
        predictions = []
        actuals = []
        
        # Sample test data for faster evaluation
        eval_sample_size = min(10000, len(test_data))
        test_sample = test_data.sample(n=eval_sample_size, random_state=self.config.random_state)
        
        for _, row in test_sample.iterrows():
            pred = self.predict_rating(row['user_id'], row['movie_id'])
            predictions.append(pred)
            actuals.append(row['final_rating'])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        
        # Ranking metrics
        ranking_metrics = self._calculate_ranking_metrics(test_data)
        
        # Model size calculation
        model_size_mb = self._calculate_model_size()
        
        # Inference time measurement
        inference_time_ms = self._measure_inference_time()
        
        results = {
            'accuracy_rmse': rmse,
            'accuracy_mae': mae,
            'model_size_mb': model_size_mb,
            'inference_time_ms': inference_time_ms,
            **ranking_metrics
        }
        
        logger.info(f"Model evaluation completed:")
        logger.info(f"  RMSE: {rmse:.3f}")
        logger.info(f"  MAE: {mae:.3f}")
        logger.info(f"  Model size: {model_size_mb:.1f} MB")
        logger.info(f"  Inference time: {inference_time_ms:.3f} ms")
        
        return results
    
    def _calculate_ranking_metrics(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate precision and recall at K"""
        # Sample users for ranking evaluation
        sample_users = test_data['user_id'].unique()[:100]  # Evaluate on 100 users for speed
        
        precisions_at_k = {k: [] for k in self.config.precision_recall_k}
        recalls_at_k = {k: [] for k in self.config.precision_recall_k}
        
        for user_id in sample_users:
            user_test_movies = set(test_data[test_data['user_id'] == user_id]['movie_id'])
            
            if len(user_test_movies) == 0:
                continue
            
            try:
                # Get recommendations
                recommendations = self.recommend_movies(user_id, max(self.config.precision_recall_k))
                
                for k in self.config.precision_recall_k:
                    top_k_recs = set(recommendations[:k])
                    
                    # Calculate precision and recall
                    relevant_recs = top_k_recs.intersection(user_test_movies)
                    
                    precision = len(relevant_recs) / k if k > 0 else 0
                    recall = len(relevant_recs) / len(user_test_movies) if len(user_test_movies) > 0 else 0
                    
                    precisions_at_k[k].append(precision)
                    recalls_at_k[k].append(recall)
            except Exception as e:
                logger.warning(f"Error calculating metrics for user {user_id}: {e}")
                continue
        
        # Average metrics
        metrics = {}
        for k in self.config.precision_recall_k:
            metrics[f'precision_at_{k}'] = np.mean(precisions_at_k[k]) if precisions_at_k[k] else 0
            metrics[f'recall_at_{k}'] = np.mean(recalls_at_k[k]) if recalls_at_k[k] else 0
        
        return metrics
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB"""
        # Estimate size of SVD factors and mappings
        user_factors_size = self.user_factors.nbytes if self.user_factors is not None else 0
        item_factors_size = self.item_factors.nbytes if self.item_factors is not None else 0
        mapper_size = len(str(self.user_mapper)) + len(str(self.movie_mapper))
        total_size = user_factors_size + item_factors_size + mapper_size
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _measure_inference_time(self) -> float:
        """Measure average inference time for a single prediction"""
        # Sample user and movie for timing
        if not self.user_mapper or not self.movie_mapper:
            return 0.0
        
        sample_user = list(self.user_mapper.keys())[0]
        sample_movie = list(self.movie_mapper.keys())[0]
        
        # Measure time for 100 predictions
        start_time = time.time()
        for _ in range(100):
            self.predict_rating(sample_user, sample_movie)
        end_time = time.time()
        
        # Return average time in milliseconds
        return ((end_time - start_time) / 100) * 1000
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """Save entire trained model object (for direct use in inference)"""
        if filepath is None:
            filepath = config.get_model_path()
        
        # Save the entire trainer object instead of a dictionary
        # This allows the model to be loaded and used directly without reconstruction
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: Optional[str] = None) -> bool:
        """Load entire trained model object"""
        if filepath is None:
            filepath = config.get_model_path()
        
        try:
            with open(filepath, 'rb') as f:
                loaded_trainer = pickle.load(f)
            
            # Copy all attributes from loaded trainer to self
            self.model = loaded_trainer.model
            self.user_mapper = loaded_trainer.user_mapper
            self.movie_mapper = loaded_trainer.movie_mapper
            self.reverse_user_mapper = loaded_trainer.reverse_user_mapper
            self.reverse_movie_mapper = loaded_trainer.reverse_movie_mapper
            self.global_mean = loaded_trainer.global_mean
            self.user_factors = loaded_trainer.user_factors
            self.item_factors = loaded_trainer.item_factors
            self.config = loaded_trainer.config
            self.training_stats = loaded_trainer.training_stats
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            return False


def train_svd_model(training_df: pd.DataFrame) -> Tuple[SVDModelTrainer, Dict[str, Any]]:
    """Convenience function to train SVD model"""
    trainer = SVDModelTrainer()
    results = trainer.train_model(training_df)
    return trainer, results


if __name__ == "__main__":
    # Test SVD training
    from data_loader import load_and_validate_data
    from data_preprocessor import preprocess_data
    from feature_engineer import engineer_features, FeatureEngineer
    
    try:
        # Load and prepare data
        users_df, movies_df, interactions_df = load_and_validate_data()
        users_clean, movies_clean, interactions_clean = preprocess_data(users_df, movies_df, interactions_df)
        features_df = engineer_features(interactions_clean)
        
        # Prepare training data
        engineer = FeatureEngineer()
        training_df = engineer.prepare_training_data(features_df)
        
        # Train model
        trainer, results = train_svd_model(training_df)
        
        # Save model
        trainer.save_model()
        
        print("SVD model training successful!")
        print(f"Training time: {results['training_time_seconds']:.2f} seconds")
        print(f"RMSE: {results['accuracy_rmse']:.3f}")
        print(f"Model size: {results['model_size_mb']:.1f} MB")
        
        # Test recommendations
        sample_user = training_df['user_id'].iloc[0]
        recommendations = trainer.recommend_movies(sample_user, 5)
        print(f"Sample recommendations for user {sample_user}: {recommendations}")
        
    except Exception as e:
        print(f"SVD training failed: {e}")
        exit(1)
