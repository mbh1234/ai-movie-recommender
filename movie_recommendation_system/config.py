"""
Configuration management for movie recommendation system.
Centralizes all configuration parameters for easy management and testing.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DataConfig:
    """Data loading and processing configuration"""
    # File paths
    users_file: str = "data/users.csv"
    movies_file: str = "data/movies_table_clean.csv" 
    interactions_file: str = "data/interactions_table.csv"
    
    # Data validation
    min_interactions_per_user: int = 5
    min_interactions_per_movie: int = 10
    max_rating: float = 5.0
    min_rating: float = 1.0
    
    # Feature engineering
    watch_time_percentiles: List[float] = None
    
    def __post_init__(self):
        if self.watch_time_percentiles is None:
            self.watch_time_percentiles = [0.25, 0.50, 0.75, 0.90]


@dataclass 
class ModelConfig:
    """SVD model configuration"""
    # SVD parameters
    n_components: int = 50  # Will be tuned via hyperparameter search
    max_iter: int = 100
    random_state: int = 42
    
    # Training parameters
    train_fraction: float = 0.6  # 60% train
    val_fraction: float = 0.2    # 20% validation
    test_fraction: float = 0.2   # 20% test
    train_test_split_ratio: float = 0.8  # Legacy parameter for backward compatibility
    sample_size_training: int = None  # For memory efficiency
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = True
    n_components_search: List[int] = None  # Search space for SVD components
    
    # Evaluation parameters
    precision_recall_k: List[int] = None
    recommendation_count: int = 20
    
    def __post_init__(self):
        if self.precision_recall_k is None:
            self.precision_recall_k = [10, 20]
        if self.n_components_search is None:
            self.n_components_search = [20, 50, 100, 150]  # Components to try


@dataclass
class APIConfig:
    """API service configuration"""
    host: str = "0.0.0.0"
    port: int = 8082
    debug: bool = False
    max_response_time_ms: int = 600
    default_recommendation_count: int = 20


@dataclass
class MonitoringConfig:
    """Monitoring and telemetry configuration"""
    # Online evaluation
    online_eval_window_hours: int = 24
    min_interactions_for_online_eval: int = 100
    
    # Data quality monitoring
    # drift_detection_threshold: float = 0.1
    drift_detection_threshold: float = 0.3  # Detect changes greater than 30%
    drift_detection_relative_threshold: float = 0.3
    schema_validation_enabled: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Config:
    """Main configuration class that combines all config sections"""
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.monitoring = MonitoringConfig()
        
        # Model persistence
        self.model_save_path = "models/"
        self.model_filename = "svd_model_final.pkl"
        self.results_filename = "svd_model_results.pkl"
        
        # Ensure model directory exists
        os.makedirs(self.model_save_path, exist_ok=True)
    
    def get_model_path(self) -> str:
        """Get full path to saved model"""
        return os.path.join(self.model_save_path, self.model_filename)
    
    def get_results_path(self) -> str:
        """Get full path to saved results"""
        return os.path.join(self.model_save_path, self.results_filename)
    
    # def update_from_env(self):
    #     """Update configuration from environment variables"""
    #     # API configuration
    #     self.api.host = os.getenv("API_HOST", self.api.host)
    #     self.api.port = int(os.getenv("API_PORT", self.api.port))
    #     self.api.debug = os.getenv("DEBUG", "false").lower() == "true"
        
    #     # Model configuration
    #     self.model.n_components = int(os.getenv("SVD_COMPONENTS", self.model.n_components))
    #     self.model.sample_size_training = int(os.getenv("TRAINING_SAMPLE_SIZE", self.model.sample_size_training))
        
    #     # Monitoring
    #     self.monitoring.log_level = os.getenv("LOG_LEVEL", self.monitoring.log_level)
    # def update_from_env(self):
    #     """Update configuration from environment variables - FIXED"""
    #     # API configuration
    #     self.api.host = os.getenv("API_HOST", self.api.host)
    #     self.api.port = int(os.getenv("API_PORT", str(self.api.port)))
    #     self.api.debug = os.getenv("DEBUG", "false").lower() == "true"

    #     # Model configuration - FIXED: Handle None values properly
    #     self.model.n_components = int(os.getenv("SVD_COMPONENTS", str(self.model.n_components)))
    #     self.model.sample_size_training = int(os.getenv("TRAINING_SAMPLE_SIZE", str(self.model.sample_size_training)))
    #     self.model.max_iter = int(os.getenv("SVD_MAX_ITER", str(self.model.max_iter)))
        
    #     # Data configuration
    #     self.data.min_interactions_per_user = int(os.getenv("MIN_USER_INTERACTIONS", str(self.data.min_interactions_per_user)))
    #     self.data.min_interactions_per_movie = int(os.getenv("MIN_MOVIE_INTERACTIONS", str(self.data.min_interactions_per_movie)))
    def update_from_env(self):
        """Update configuration from environment variables - PROPERLY FIXED"""
        # API configuration
        self.api.host = os.getenv("API_HOST", self.api.host)
        self.api.port = int(os.getenv("API_PORT", str(self.api.port)))
        self.api.debug = os.getenv("DEBUG", "false").lower() == "true"

        # Model configuration - PROPERLY FIXED: Handle None values
        self.model.n_components = int(os.getenv("SVD_COMPONENTS", str(self.model.n_components)))
        
        # Handle sample_size_training which might be None
        default_sample_size = getattr(self.model, 'sample_size_training', None)
        if default_sample_size is None:
            default_sample_size = 1000  # Set a reasonable default
        self.model.sample_size_training = int(os.getenv("TRAINING_SAMPLE_SIZE", str(default_sample_size)))
        
        # Handle max_iter which might be None  
        default_max_iter = getattr(self.model, 'max_iter', None)
        if default_max_iter is None:
            default_max_iter = 100  # Set a reasonable default
        self.model.max_iter = int(os.getenv("SVD_MAX_ITER", str(default_max_iter)))
        
        # Data configuration - with defaults for None values
        default_min_user = getattr(self.data, 'min_interactions_per_user', None)
        if default_min_user is None:
            default_min_user = 5
        self.data.min_interactions_per_user = int(os.getenv("MIN_USER_INTERACTIONS", str(default_min_user)))
        
        default_min_movie = getattr(self.data, 'min_interactions_per_movie', None) 
        if default_min_movie is None:
            default_min_movie = 10
        self.data.min_interactions_per_movie = int(os.getenv("MIN_MOVIE_INTERACTIONS", str(default_min_movie)))


# Global configuration instance
config = Config()
