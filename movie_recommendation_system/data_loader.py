"""
Data loading and validation module for movie recommendation system.
Handles loading datasets and basic validation checks.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any
from pathlib import Path

from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.monitoring.log_level),
    format=config.monitoring.log_format
)
logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Custom exception for data loading errors"""
    pass


class DataLoader:
    """Handles loading and basic validation of movie recommendation datasets"""
    
    def __init__(self, data_config=None):
        """Initialize data loader with configuration"""
        self.config = data_config or config.data
        
    def load_datasets(self, base_path: str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets (users, movies, interactions)
        
        Args:
            base_path: Base directory path for data files
            
        Returns:
            Tuple of (users_df, movies_df, interactions_df)
            
        Raises:
            DataLoadError: If any dataset fails to load or validate
        """
        logger.info("Loading datasets...")
        
        try:
            # Construct absolute file paths
            project_root = Path(__file__).parent  # Get project root directory
            users_path = project_root / self.config.users_file
            movies_path = project_root / self.config.movies_file
            interactions_path = project_root / self.config.interactions_file
            
            # Load datasets
            users_df = self._load_csv(users_path, "users")
            movies_df = self._load_csv(movies_path, "movies") 
            interactions_df = self._load_csv(interactions_path, "interactions")
            
            # Basic validation
            self._validate_datasets(users_df, movies_df, interactions_df)
            
            # Log dataset info
            logger.info(f"Loaded datasets - Users: {users_df.shape}, Movies: {movies_df.shape}, Interactions: {interactions_df.shape}")
            
            return users_df, movies_df, interactions_df
            
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise DataLoadError(f"Dataset loading failed: {e}")
    
    def _load_csv(self, filepath: Path, dataset_name: str) -> pd.DataFrame:
        """Load a single CSV file with error handling"""
        try:
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            # Try different encodings and parsing strategies
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings_to_try:
                try:
                    # First try standard parsing
                    df = pd.read_csv(filepath, encoding=encoding)
                    logger.info(f"Loaded {dataset_name} dataset: {df.shape} (encoding: {encoding})")
                    return df
                except (UnicodeDecodeError, pd.errors.ParserError):
                    # If standard parsing fails, try with error handling
                    try:
                        df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', engine='python')
                        logger.warning(f"Loaded {dataset_name} dataset with skipped bad lines: {df.shape} (encoding: {encoding})")
                        return df
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
            
            # Final fallback with most permissive settings
            try:
                df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip', 
                               engine='python', quoting=3, sep=None)  # quoting=3 means QUOTE_NONE, sep=None auto-detects
                logger.warning(f"Loaded {dataset_name} dataset with auto-detected separator and relaxed parsing: {df.shape}")
                return df
            except Exception:
                # Last resort - try with manual error handling
                df = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip', engine='python')
                logger.warning(f"Loaded {dataset_name} dataset with latin-1 encoding and skipped errors: {df.shape}")
                return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load {dataset_name} from {filepath}: {e}")
    
    def _validate_datasets(self, users_df: pd.DataFrame, movies_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Perform basic validation on loaded datasets"""
        
        # Check if datasets are not empty
        if len(users_df) == 0:
            raise DataLoadError("Users dataset is empty")
        if len(movies_df) == 0:
            raise DataLoadError("Movies dataset is empty")
        if len(interactions_df) == 0:
            raise DataLoadError("Interactions dataset is empty")
        
        # Check required columns
        required_user_cols = ['user_id', 'age', 'occupation', 'gender']
        required_movie_cols = ['movie_id', 'title']
        required_interaction_cols = ['user_id', 'movie_id', 'total_minutes']
        
        self._check_required_columns(users_df, required_user_cols, "users")
        self._check_required_columns(movies_df, required_movie_cols, "movies")
        self._check_required_columns(interactions_df, required_interaction_cols, "interactions")
        
        # Check data types
        if not pd.api.types.is_integer_dtype(users_df['user_id']):
            raise DataLoadError("User ID must be integer type")
        if not pd.api.types.is_integer_dtype(interactions_df['user_id']):
            raise DataLoadError("Interaction user_id must be integer type")
        
        # Check for completely missing key columns
        if users_df['user_id'].isnull().all():
            raise DataLoadError("All user IDs are missing")
        if interactions_df['user_id'].isnull().all():
            raise DataLoadError("All interaction user IDs are missing")
        if interactions_df['movie_id'].isnull().all():
            raise DataLoadError("All interaction movie IDs are missing")
        
        logger.info("Basic dataset validation passed")
    
    def _check_required_columns(self, df: pd.DataFrame, required_cols: list, dataset_name: str):
        """Check if required columns exist in dataframe"""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataLoadError(f"Missing required columns in {dataset_name} dataset: {missing_cols}")
    
    def get_dataset_statistics(self, users_df: pd.DataFrame, movies_df: pd.DataFrame, interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics"""
        
        stats = {
            "users": {
                "total_users": len(users_df),
                "unique_users": users_df['user_id'].nunique(),
                "age_range": (users_df['age'].min(), users_df['age'].max()),
                "age_mean": users_df['age'].mean(),
                "gender_distribution": users_df['gender'].value_counts().to_dict(),
                "missing_values": users_df.isnull().sum().to_dict()
            },
            "movies": {
                "total_movies": len(movies_df),
                "unique_movies": movies_df['movie_id'].nunique(),
                "missing_values": movies_df.isnull().sum().to_dict()
            },
            "interactions": {
                "total_interactions": len(interactions_df),
                "unique_users": interactions_df['user_id'].nunique(),
                "unique_movies": interactions_df['movie_id'].nunique(),
                "watch_time_stats": {
                    "min": interactions_df['total_minutes'].min(),
                    "max": interactions_df['total_minutes'].max(),
                    "mean": interactions_df['total_minutes'].mean(),
                    "median": interactions_df['total_minutes'].median()
                },
                "missing_values": interactions_df.isnull().sum().to_dict()
            }
        }
        
        # Add rating statistics if rating column exists
        if 'rating' in interactions_df.columns:
            valid_ratings = interactions_df['rating'].dropna()
            stats["interactions"]["rating_stats"] = {
                "total_ratings": len(valid_ratings),
                "rating_coverage": len(valid_ratings) / len(interactions_df),
                "rating_distribution": interactions_df['rating'].value_counts().to_dict(),
                "mean_rating": valid_ratings.mean()
            }
        
        return stats
    
    def explore_dataset(self, df: pd.DataFrame, name: str) -> None:
        """Print exploratory data analysis for a dataset (from original notebook)"""
        print(f"\n{name.upper()} DATASET EXPLORATION")
        print("-" * 50)
        print(f"Shape: {df.shape}")
        print(f"\nColumn types:")
        print(df.dtypes)
        print(f"\nMissing values:")
        print(df.isnull().sum())
        print(f"\nFirst 5 rows:")
        print(df.head())
        if len(df) > 0:
            print(f"\nBasic statistics:")
            print(df.describe(include='all'))
        print("\n" + "="*60)


def load_and_validate_data(base_path: str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to load and validate all datasets"""
    loader = DataLoader()
    return loader.load_datasets(base_path)


if __name__ == "__main__":
    # Test data loading
    try:
        users_df, movies_df, interactions_df = load_and_validate_data()
        
        # Generate statistics
        loader = DataLoader()
        stats = loader.get_dataset_statistics(users_df, movies_df, interactions_df)
        
        print("Dataset loading successful!")
        print(f"Users: {stats['users']['total_users']}")
        print(f"Movies: {stats['movies']['total_movies']}")
        print(f"Interactions: {stats['interactions']['total_interactions']}")
        
    except DataLoadError as e:
        print(f"Data loading failed: {e}")
        exit(1)
