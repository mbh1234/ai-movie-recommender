"""
Feature engineering module for movie recommendation system.
Creates features for model training, especially the hybrid final_rating.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any

from config import config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for recommendation system"""
    
    def __init__(self, data_config=None):
        """Initialize feature engineer with configuration"""
        self.config = data_config or config.data
        
    def create_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            interactions_df: Preprocessed interactions dataset
            
        Returns:
            DataFrame with engineered features including final_rating
        """
        logger.info("Starting feature engineering...")
        
        # Create a copy to avoid modifying original
        features_df = interactions_df.copy()
        
        # Create implicit rating from watch time
        features_df = self._create_implicit_rating(features_df)
        
        # Create final hybrid rating
        features_df = self._create_final_rating(features_df)
        
        # Create additional features
        features_df = self._create_user_features(features_df)
        features_df = self._create_movie_features(features_df)
        
        logger.info(f"Feature engineering completed. Final shape: {features_df.shape}")
        return features_df
    
    def _create_implicit_rating(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create implicit rating from watch time using percentile-based approach
        This replicates the logic from the original notebook
        """
        logger.info("Creating implicit ratings from watch time...")
        
        # Calculate watch time percentiles
        watch_times = interactions_df['total_minutes'].dropna()
        percentiles = np.percentile(watch_times, [25, 50, 75, 90])
        
        logger.info(f"Watch time percentiles - 25th: {percentiles[0]:.1f}, 50th: {percentiles[1]:.1f}, "
                   f"75th: {percentiles[2]:.1f}, 90th: {percentiles[3]:.1f}")
        
        # Create implicit rating based on watch time percentiles
        def calculate_implicit_rating(minutes):
            if pd.isna(minutes):
                return 2.5  # Default neutral rating for missing watch time
            elif minutes <= percentiles[0]:  # Bottom 25%
                return 2.0
            elif minutes <= percentiles[1]:  # 25-50%
                return 2.5
            elif minutes <= percentiles[2]:  # 50-75%
                return 3.0
            elif minutes <= percentiles[3]:  # 75-90%
                return 4.0
            else:  # Top 10%
                return 5.0
        
        interactions_df['implicit_rating'] = interactions_df['total_minutes'].apply(calculate_implicit_rating)
        
        # Log distribution of implicit ratings
        implicit_dist = interactions_df['implicit_rating'].value_counts().sort_index()
        logger.info(f"Implicit rating distribution:\n{implicit_dist}")
        
        return interactions_df
    
    def _create_final_rating(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create final hybrid rating combining explicit and implicit ratings
        This is the key feature used by the SVD model
        """
        logger.info("Creating final hybrid rating...")
        
        def combine_ratings(row):
            """Combine explicit and implicit ratings with preference for explicit"""
            if pd.notna(row.get('rating')):
                # Use explicit rating if available
                return row['rating']
            else:
                # Use implicit rating derived from watch time
                return row['implicit_rating']
        
        interactions_df['final_rating'] = interactions_df.apply(combine_ratings, axis=1)
        
        # Log rating type distribution
        explicit_count = interactions_df['rating'].notna().sum()
        implicit_count = interactions_df['rating'].isna().sum()
        total_count = len(interactions_df)
        
        logger.info(f"Final rating composition:")
        logger.info(f"  Explicit ratings: {explicit_count:,} ({explicit_count/total_count:.1%})")
        logger.info(f"  Implicit ratings: {implicit_count:,} ({implicit_count/total_count:.1%})")
        
        # Log final rating distribution
        final_dist = interactions_df['final_rating'].value_counts().sort_index()
        logger.info(f"Final rating distribution:\n{final_dist}")
        
        return interactions_df
    
    def _create_user_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create user-level aggregated features"""
        logger.info("Creating user-level features...")
        
        # User activity features
        user_stats = interactions_df.groupby('user_id').agg({
            'total_minutes': ['count', 'sum', 'mean'],
            'final_rating': ['mean', 'std'],
            'movie_id': 'nunique'
        }).round(3)
        
        # Flatten column names
        user_stats.columns = [f'user_{col[1]}_{col[0]}' if col[1] else f'user_{col[0]}' for col in user_stats.columns]
        user_stats = user_stats.rename(columns={
            'user_count_total_minutes': 'user_interaction_count',
            'user_sum_total_minutes': 'user_total_watch_time',
            'user_mean_total_minutes': 'user_avg_watch_time',
            'user_mean_final_rating': 'user_avg_rating',
            'user_std_final_rating': 'user_rating_std',
            'user_nunique_movie_id': 'user_unique_movies'
        })
        
        # Fill NaN std with 0 (users with only one rating)
        user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)
        
        # Merge back to interactions
        interactions_df = interactions_df.merge(user_stats, on='user_id', how='left')
        
        logger.info(f"Created {len(user_stats.columns)} user-level features")
        return interactions_df
    
    def _create_movie_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create movie-level aggregated features"""
        logger.info("Creating movie-level features...")
        
        # Movie popularity features
        movie_stats = interactions_df.groupby('movie_id').agg({
            'total_minutes': ['count', 'sum', 'mean'],
            'final_rating': ['mean', 'std'],
            'user_id': 'nunique'
        }).round(3)
        
        # Flatten column names
        movie_stats.columns = [f'movie_{col[1]}_{col[0]}' if col[1] else f'movie_{col[0]}' for col in movie_stats.columns]
        movie_stats = movie_stats.rename(columns={
            'movie_count_total_minutes': 'movie_interaction_count',
            'movie_sum_total_minutes': 'movie_total_watch_time',
            'movie_mean_total_minutes': 'movie_avg_watch_time',
            'movie_mean_final_rating': 'movie_avg_rating',
            'movie_std_final_rating': 'movie_rating_std',
            'movie_nunique_user_id': 'movie_unique_users'
        })
        
        # Fill NaN std with 0 (movies with only one rating)
        movie_stats['movie_rating_std'] = movie_stats['movie_rating_std'].fillna(0)
        
        # Merge back to interactions
        interactions_df = interactions_df.merge(movie_stats, on='movie_id', how='left')
        
        logger.info(f"Created {len(movie_stats.columns)} movie-level features")
        return interactions_df
    
    def prepare_training_data(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare final dataset for model training
        Returns dataset ready for SVD model with minimal required columns
        """
        logger.info("Preparing training data...")
        
        # Select essential columns for training
        training_columns = ['user_id', 'movie_id', 'final_rating']
        
        # Add additional columns if they exist and might be useful
        optional_columns = ['total_minutes', 'rating', 'implicit_rating', 'timestamp']
        for col in optional_columns:
            if col in features_df.columns:
                training_columns.append(col)
        
        training_df = features_df[training_columns].copy()
        
        # Ensure no missing values in key columns
        training_df = training_df.dropna(subset=['user_id', 'movie_id', 'final_rating'])
        
        if 'timestamp' in training_df.columns:
            training_df['timestamp'] = pd.to_datetime(training_df['timestamp'], errors='coerce')
            missing_timestamps = training_df['timestamp'].isna().sum()
            if missing_timestamps > 0:
                logger.info(f"Dropping {missing_timestamps} interactions with invalid timestamps from training data")
                training_df = training_df[training_df['timestamp'].notna()]
        
        # Convert user_id and movie_id to consistent types
        training_df['user_id'] = training_df['user_id'].astype(int)
        training_df['movie_id'] = training_df['movie_id'].astype(str)
        
        logger.info(f"Training data prepared: {training_df.shape}")
        logger.info(f"Final rating range: {training_df['final_rating'].min():.2f} - {training_df['final_rating'].max():.2f}")
        
        return training_df
    
    def get_feature_statistics(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive feature statistics"""
        
        stats = {
            "total_interactions": len(features_df),
            "unique_users": features_df['user_id'].nunique(),
            "unique_movies": features_df['movie_id'].nunique(),
            "rating_composition": {
                "explicit_ratings": features_df['rating'].notna().sum(),
                "implicit_ratings": features_df['rating'].isna().sum(),
                "explicit_percentage": features_df['rating'].notna().mean() * 100
            },
            "final_rating_stats": {
                "mean": features_df['final_rating'].mean(),
                "std": features_df['final_rating'].std(),
                "min": features_df['final_rating'].min(),
                "max": features_df['final_rating'].max(),
                "distribution": features_df['final_rating'].value_counts().sort_index().to_dict()
            },
            "watch_time_stats": {
                "mean": features_df['total_minutes'].mean(),
                "median": features_df['total_minutes'].median(),
                "std": features_df['total_minutes'].std()
            }
        }
        
        return stats


def engineer_features(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to run complete feature engineering pipeline"""
    engineer = FeatureEngineer()
    return engineer.create_features(interactions_df)


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import load_and_validate_data
    from data_preprocessor import preprocess_data
    
    try:
        # Load and preprocess data
        users_df, movies_df, interactions_df = load_and_validate_data()
        users_clean, movies_clean, interactions_clean = preprocess_data(users_df, movies_df, interactions_df)
        
        # Engineer features
        features_df = engineer_features(interactions_clean)
        
        # Prepare training data
        engineer = FeatureEngineer()
        training_df = engineer.prepare_training_data(features_df)
        
        # Generate statistics
        stats = engineer.get_feature_statistics(features_df)
        
        print("Feature engineering successful!")
        print(f"Training data shape: {training_df.shape}")
        print(f"Explicit rating percentage: {stats['rating_composition']['explicit_percentage']:.1f}%")
        print(f"Final rating range: {stats['final_rating_stats']['min']:.1f} - {stats['final_rating_stats']['max']:.1f}")
        
    except Exception as e:
        print(f"Feature engineering failed: {e}")
        exit(1)
