"""
Data preprocessing module for movie recommendation system.
Handles data cleaning, filtering, and preparation for model training.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

from config import config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data cleaning and preprocessing for recommendation system"""
    
    def __init__(self, data_config=None):
        """Initialize preprocessor with configuration"""
        self.config = data_config or config.data
        
    def preprocess_datasets(self, users_df: pd.DataFrame, movies_df: pd.DataFrame, 
                          interactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Main preprocessing pipeline for all datasets
        
        Args:
            users_df: Raw users dataset
            movies_df: Raw movies dataset  
            interactions_df: Raw interactions dataset
            
        Returns:
            Tuple of preprocessed (users_df, movies_df, interactions_df)
        """
        logger.info("Starting data preprocessing pipeline...")
        
        # Preprocess each dataset
        users_clean = self._preprocess_users(users_df.copy())
        movies_clean = self._preprocess_movies(movies_df.copy())
        interactions_clean = self._preprocess_interactions(interactions_df.copy())
        
        # Filter for consistency across datasets
        users_clean, movies_clean, interactions_clean = self._filter_consistent_data(
            users_clean, movies_clean, interactions_clean
        )
        
        logger.info("Data preprocessing completed")
        logger.info(f"Final shapes - Users: {users_clean.shape}, Movies: {movies_clean.shape}, Interactions: {interactions_clean.shape}")
        
        return users_clean, movies_clean, interactions_clean
    
    def _preprocess_users(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess users dataset"""
        logger.info("Preprocessing users dataset...")
        
        # Remove duplicates
        initial_count = len(users_df)
        users_df = users_df.drop_duplicates(subset=['user_id'])
        logger.info(f"Removed {initial_count - len(users_df)} duplicate users")
        
        # Clean age data
        users_df = users_df[(users_df['age'] > 0) & (users_df['age'] < 120)]
        
        # Clean gender data
        users_df = users_df[users_df['gender'].isin(['M', 'F'])]
        
        # Clean occupation data
        users_df['occupation'] = users_df['occupation'].fillna('other or not specified')
        
        # Remove users with missing user_id
        users_df = users_df.dropna(subset=['user_id'])
        
        logger.info(f"Users dataset after cleaning: {users_df.shape}")
        return users_df
    
    def _preprocess_movies(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess movies dataset"""
        logger.info("Preprocessing movies dataset...")
        
        # Remove duplicates
        initial_count = len(movies_df)
        movies_df = movies_df.drop_duplicates(subset=['movie_id'])
        logger.info(f"Removed {initial_count - len(movies_df)} duplicate movies")
        
        # Clean movie_id - remove malformed entries
        movies_df = movies_df[~movies_df['movie_id'].astype(str).str.contains('#NAME?|NaN', na=False)]
        movies_df = movies_df.dropna(subset=['movie_id'])
        movies_df = movies_df[movies_df['movie_id'] != '']
        
        # Clean title
        movies_df = movies_df.dropna(subset=['title'])
        movies_df = movies_df[movies_df['title'] != '']
        
        # Handle missing numeric fields
        numeric_columns = ['vote_average', 'vote_count', 'popularity', 'runtime']
        for col in numeric_columns:
            if col in movies_df.columns:
                movies_df[col] = pd.to_numeric(movies_df[col], errors='coerce')
                movies_df[col] = movies_df[col].fillna(0)
        
        # Clean vote_average (should be 0-10)
        if 'vote_average' in movies_df.columns:
            movies_df.loc[movies_df['vote_average'] < 0, 'vote_average'] = 0
            movies_df.loc[movies_df['vote_average'] > 10, 'vote_average'] = 10
        
        # Clean genres_list
        if 'genres_list' in movies_df.columns:
            movies_df['genres_list'] = movies_df['genres_list'].fillna('unknown')
        
        logger.info(f"Movies dataset after cleaning: {movies_df.shape}")
        return movies_df
    
    def _preprocess_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess interactions dataset"""
        logger.info("Preprocessing interactions dataset...")
        
        # Remove missing user_id and movie_id
        initial_count = len(interactions_df)
        interactions_df = interactions_df.dropna(subset=['user_id', 'movie_id'])
        logger.info(f"Removed {initial_count - len(interactions_df)} interactions with missing IDs")
        
        # Clean total_minutes (watch time)
        interactions_df = interactions_df[interactions_df['total_minutes'] > 0]
        
        # Cap extremely long watch times (>12 hours = 720 minutes)
        max_minutes = 720
        long_watches = interactions_df['total_minutes'] > max_minutes
        if long_watches.sum() > 0:
            logger.info(f"Capping {long_watches.sum()} extremely long watch times")
            interactions_df.loc[long_watches, 'total_minutes'] = max_minutes
        
        # Clean explicit ratings if present
        if 'rating' in interactions_df.columns:
            # Keep rating as NaN for implicit interactions, clean explicit ones
            valid_rating_mask = interactions_df['rating'].notna()
            invalid_ratings = (interactions_df['rating'] < self.config.min_rating) | (interactions_df['rating'] > self.config.max_rating)
            
            if invalid_ratings.sum() > 0:
                logger.info(f"Removing {invalid_ratings.sum()} interactions with invalid ratings")
                # Set invalid ratings to NaN rather than removing the interaction
                interactions_df.loc[invalid_ratings, 'rating'] = np.nan
        
        # Parse watch timestamps for temporal modeling
        if 'last_watch_time' in interactions_df.columns:
            interactions_df['timestamp'] = pd.to_datetime(
                interactions_df['last_watch_time'],
                errors='coerce'
            )
            invalid_timestamps = interactions_df['timestamp'].isna().sum()
            if invalid_timestamps > 0:
                logger.info(f"Removing {invalid_timestamps} interactions with invalid last_watch_time")
                interactions_df = interactions_df[interactions_df['timestamp'].notna()]
        
        # Remove duplicate interactions (same user-movie pair)
        interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
        
        logger.info(f"Interactions dataset after cleaning: {interactions_df.shape}")
        return interactions_df
    
    def _filter_consistent_data(self, users_df: pd.DataFrame, movies_df: pd.DataFrame, 
                              interactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter datasets to ensure consistency across user and movie IDs"""
        logger.info("Filtering for consistent data across datasets...")
        
        # Get valid user and movie IDs
        valid_user_ids = set(users_df['user_id'].unique())
        valid_movie_ids = set(movies_df['movie_id'].unique())
        
        # Filter interactions to only include valid users and movies
        initial_interactions = len(interactions_df)
        interactions_df = interactions_df[
            interactions_df['user_id'].isin(valid_user_ids) & 
            interactions_df['movie_id'].isin(valid_movie_ids)
        ]
        logger.info(f"Removed {initial_interactions - len(interactions_df)} interactions with invalid user/movie IDs")
        
        # Apply minimum interaction filters
        user_interaction_counts = interactions_df['user_id'].value_counts()
        movie_interaction_counts = interactions_df['movie_id'].value_counts()
        
        # Filter users with minimum interactions
        valid_users = user_interaction_counts[user_interaction_counts >= self.config.min_interactions_per_user].index
        interactions_df = interactions_df[interactions_df['user_id'].isin(valid_users)]
        users_df = users_df[users_df['user_id'].isin(valid_users)]
        
        # Filter movies with minimum interactions
        valid_movies = movie_interaction_counts[movie_interaction_counts >= self.config.min_interactions_per_movie].index
        interactions_df = interactions_df[interactions_df['movie_id'].isin(valid_movies)]
        movies_df = movies_df[movies_df['movie_id'].isin(valid_movies)]
        
        logger.info(f"After minimum interaction filtering:")
        logger.info(f"  Users: {len(users_df)} (min {self.config.min_interactions_per_user} interactions)")
        logger.info(f"  Movies: {len(movies_df)} (min {self.config.min_interactions_per_movie} interactions)")
        logger.info(f"  Interactions: {len(interactions_df)}")
        
        return users_df, movies_df, interactions_df
    
    def get_preprocessing_statistics(self, users_before: pd.DataFrame, movies_before: pd.DataFrame, 
                                   interactions_before: pd.DataFrame, users_after: pd.DataFrame, 
                                   movies_after: pd.DataFrame, interactions_after: pd.DataFrame) -> Dict[str, Any]:
        """Generate preprocessing statistics"""
        
        return {
            "users": {
                "before": len(users_before),
                "after": len(users_after),
                "removed": len(users_before) - len(users_after),
                "removal_rate": (len(users_before) - len(users_after)) / len(users_before)
            },
            "movies": {
                "before": len(movies_before),
                "after": len(movies_after),
                "removed": len(movies_before) - len(movies_after),
                "removal_rate": (len(movies_before) - len(movies_after)) / len(movies_before)
            },
            "interactions": {
                "before": len(interactions_before),
                "after": len(interactions_after),
                "removed": len(interactions_before) - len(interactions_after),
                "removal_rate": (len(interactions_before) - len(interactions_after)) / len(interactions_before)
            }
        }


def preprocess_data(users_df: pd.DataFrame, movies_df: pd.DataFrame, 
                   interactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to preprocess all datasets"""
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess_datasets(users_df, movies_df, interactions_df)


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_and_validate_data
    
    try:
        # Load data
        users_df, movies_df, interactions_df = load_and_validate_data()
        
        # Preprocess data
        users_clean, movies_clean, interactions_clean = preprocess_data(
            users_df, movies_df, interactions_df
        )
        
        print("Data preprocessing successful!")
        print(f"Final dataset sizes:")
        print(f"  Users: {len(users_clean)}")
        print(f"  Movies: {len(movies_clean)}")
        print(f"  Interactions: {len(interactions_clean)}")
        
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        exit(1)
