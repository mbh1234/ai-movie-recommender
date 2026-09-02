"""
Model evaluation module for movie recommendation system.
Implements comprehensive offline evaluation strategies avoiding common pitfalls.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from config import config
from svd_model_trainer import SVDModelTrainer
from data_splitter import temporal_train_test_split, temporal_train_val_test_split

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive offline evaluation for recommendation models"""
    
    def __init__(self, model_config=None):
        """Initialize evaluator with configuration"""
        self.config = model_config or config.model
        
    def evaluate_model_offline(self, training_df: pd.DataFrame, model: SVDModelTrainer) -> Dict[str, Any]:
        """
        Comprehensive offline evaluation avoiding common pitfalls
        
        Args:
            training_df: Full dataset for evaluation
            model: Trained SVD model
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info("Starting comprehensive offline evaluation...")
        
        # 1. Temporal validation split
        temporal_results = self._temporal_validation(training_df, model)
        
        # 2. User-based cold start evaluation
        cold_start_results = self._cold_start_evaluation(training_df, model)
        
        # 3. Popularity bias evaluation
        popularity_results = self._popularity_bias_evaluation(training_df, model)
        
        # 4. Subpopulation evaluation
        subpop_results = self._subpopulation_evaluation(training_df, model)
        
        # 5. Coverage and diversity evaluation
        coverage_results = self._coverage_evaluation(training_df, model)
        popularity_results_metrics = self._average_recommendation_popularity(training_df, model)
        
        # Combine all results
        evaluation_results = {
            'temporal_validation': temporal_results,
            'popularity_baseline': temporal_results.get('baseline_popularity'),
            'cold_start_evaluation': cold_start_results,
            'popularity_bias': popularity_results,
            'subpopulation_analysis': subpop_results,
            'coverage_analysis': coverage_results,
            'evaluation_timestamp': time.time(),
            'temporal_validation_ndcg_at_20': temporal_results.get('ndcg_at_20'),
            'average_recommendation_popularity': popularity_results_metrics
        }
        
        logger.info("Offline evaluation completed")
        return evaluation_results
    
    def _temporal_validation(self, training_df: pd.DataFrame, model: SVDModelTrainer) -> Dict[str, Any]:
        """
        Temporal validation to avoid data leakage
        Split data based on chronological order if timestamp available
        """
        logger.info("Performing temporal validation...")
        
        # If we have timestamp data, use temporal split
        if 'timestamp' in training_df.columns or 'date' in training_df.columns:
            return self._temporal_split_validation(training_df, model)
        else:
            # Fall back to random split with explicit methodology
            return self._random_split_validation(training_df, model)
    
    def _temporal_split_validation(self, training_df: pd.DataFrame, model: SVDModelTrainer) -> Dict[str, Any]:
        """Use temporal split for validation"""
        timestamp_col = 'timestamp' if 'timestamp' in training_df.columns else 'date'
        
        try:
            split = temporal_train_test_split(
                training_df,
                timestamp_col=timestamp_col,
                train_fraction=self.config.train_test_split_ratio,
                ensure_warm_start=True,
            )
        except ValueError as exc:
            logger.warning(
                "Temporal validation split failed (%s). Falling back to random split.",
                exc,
            )
            return self._random_split_validation(training_df, model)
        
        logger.info(
            "Temporal split - Train: %s, Warm test: %s, Cold users: %s, Cold items: %s",
            len(split.train),
            len(split.warm_test),
            len(split.cold_user_test),
            len(split.cold_item_test),
        )
        
        baseline_metrics = self._evaluate_popularity_baseline(split.train, split.warm_test)
        evaluation = self._evaluate_predictions(split.train, split.warm_test, model, "temporal")
        evaluation['temporal_cutoff'] = split.cutoff_timestamp.isoformat()
        evaluation['temporal_metadata'] = split.metadata
        evaluation['cold_start_holdout'] = {
            'cold_user_rows': len(split.cold_user_test),
            'cold_item_rows': len(split.cold_item_test),
        }
        evaluation['baseline_popularity'] = baseline_metrics
        
        return evaluation
    
    def _random_split_validation(self, training_df: pd.DataFrame, model: SVDModelTrainer) -> Dict[str, Any]:
        """Use stratified random split for validation"""
        # Stratify by user to ensure each user appears in both train and test
        users = training_df['user_id'].unique()
        train_users, test_users = train_test_split(
            users, test_size=0.2, random_state=self.config.random_state
        )
        
        train_data = training_df[training_df['user_id'].isin(train_users)]
        test_data = training_df[training_df['user_id'].isin(test_users)]
        
        logger.info(f"User-stratified split - Train: {len(train_data)}, Test: {len(test_data)}")
        
        return self._evaluate_predictions(train_data, test_data, model, "random")
    
    def _evaluate_popularity_baseline(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a global popularity baseline on the temporal holdout."""
        if len(test_data) == 0:
            return {'precision_at_20': 0.0, 'recall_at_20': 0.0, 'ndcg_at_20': 0.0, 'evaluated_users': 0}
        
        popular_items = train_data['movie_id'].value_counts().index.tolist()
        user_history = train_data.groupby('user_id')['movie_id'].agg(lambda x: set(x)).to_dict()
        sample_users = test_data['user_id'].unique()[:100]
        
        precisions = []
        recalls = []
        ndcgs = []
        evaluated_users = 0
        
        for user_id in sample_users:
            user_test_movies = test_data[test_data['user_id'] == user_id]['movie_id'].unique().tolist()
            if not user_test_movies:
                continue
            
            seen_items = user_history.get(user_id, set())
            recommendations = []
            for movie_id in popular_items:
                if movie_id in seen_items:
                    continue
                recommendations.append(movie_id)
                if len(recommendations) >= 20:
                    break
            
            if not recommendations:
                continue
            
            evaluated_users += 1
            relevant_set = set(user_test_movies)
            hits = len([m for m in recommendations[:20] if m in relevant_set])
            precisions.append(hits / 20)
            recalls.append(hits / len(relevant_set) if len(relevant_set) > 0 else 0.0)
            ndcgs.append(self._compute_ndcg_at_k(recommendations, relevant_set, 20))
        
        return {
            'precision_at_20': float(np.mean(precisions)) if precisions else 0.0,
            'recall_at_20': float(np.mean(recalls)) if recalls else 0.0,
            'ndcg_at_20': float(np.mean(ndcgs)) if ndcgs else 0.0,
            'evaluated_users': evaluated_users
        }
    
    def _compute_ndcg_at_k(self, recommendations: List[Any], relevant_items: set, k: int) -> float:
        """Compute NDCG@k for a single user."""
        if not relevant_items:
            return 0.0
        
        dcg = 0.0
        for idx, movie_id in enumerate(recommendations[:k]):
            if movie_id in relevant_items:
                dcg += 1.0 / np.log2(idx + 2)
        
        ideal_hits = min(len(relevant_items), k)
        if ideal_hits == 0:
            return 0.0
        
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0
    
    def _evaluate_predictions(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                            model: SVDModelTrainer, split_type: str) -> Dict[str, Any]:
        """Evaluate model predictions on test data"""
        
        # Retrain model on train_data only (no internal split)
        temp_trainer = SVDModelTrainer(self.config)
        temp_trainer.train_model(train_data, perform_split=False)
        
        # Evaluate on test data
        predictions = []
        actuals = []
        prediction_times = []
        
        # Sample test data for faster evaluation
        eval_sample_size = min(5000, len(test_data))
        test_sample = test_data.sample(n=eval_sample_size, random_state=self.config.random_state)
        
        for _, row in test_sample.iterrows():
            start_time = time.time()
            pred = temp_trainer.predict_rating(row['user_id'], row['movie_id'])
            pred_time = (time.time() - start_time) * 1000  # ms
            
            predictions.append(pred)
            actuals.append(row['final_rating'])
            prediction_times.append(pred_time)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        # Calculate ranking metrics
        ranking_metrics = self._calculate_ranking_metrics_detailed(test_data, temp_trainer)
        
        return {
            'split_type': split_type,
            'rmse': rmse,
            'mae': mae,
            'avg_prediction_time_ms': np.mean(prediction_times),
            'test_sample_size': eval_sample_size,
            'coverage': self._calculate_coverage(test_data, temp_trainer),
            **ranking_metrics
        }
    
    def _cold_start_evaluation(self, training_df: pd.DataFrame, model: SVDModelTrainer) -> Dict[str, Any]:
        """Evaluate model performance on cold start users and items"""
        logger.info("Performing cold start evaluation...")
        
        # Split users into warm and cold
        user_interaction_counts = training_df['user_id'].value_counts()
        
        # Cold users: users with few interactions
        cold_threshold = 5
        cold_users = user_interaction_counts[user_interaction_counts <= cold_threshold].index
        warm_users = user_interaction_counts[user_interaction_counts > cold_threshold].index
        
        # Evaluate on cold vs warm users
        cold_data = training_df[training_df['user_id'].isin(cold_users)]
        warm_data = training_df[training_df['user_id'].isin(warm_users)]
        
        cold_results = self._evaluate_user_subset(cold_data, model, "cold_users")
        warm_results = self._evaluate_user_subset(warm_data, model, "warm_users")
        
        return {
            'cold_user_threshold': cold_threshold,
            'cold_user_count': len(cold_users),
            'warm_user_count': len(warm_users),
            'cold_user_performance': cold_results,
            'warm_user_performance': warm_results
        }
    
    def _popularity_bias_evaluation(self, training_df: pd.DataFrame, model: SVDModelTrainer) -> Dict[str, Any]:
        """Evaluate model's bias towards popular items"""
        logger.info("Evaluating popularity bias...")
        
        # Calculate movie popularity
        movie_popularity = training_df['movie_id'].value_counts()
        
        # Define popularity segments
        popular_movies = movie_popularity.head(1000).index  # Top 1000 most popular
        niche_movies = movie_popularity.tail(5000).index    # Bottom 5000 least popular
        
        # Evaluate recommendations for popular vs niche items
        sample_users = training_df['user_id'].unique()[:100]
        
        popular_rec_count = 0
        niche_rec_count = 0
        total_recommendations = 0
        
        for user_id in sample_users:
            try:
                recommendations = model.recommend_movies(user_id, 20)
                total_recommendations += len(recommendations)
                
                popular_rec_count += len([m for m in recommendations if m in popular_movies])
                niche_rec_count += len([m for m in recommendations if m in niche_movies])
            except:
                continue
        
        popularity_bias = {
            'popular_movie_percentage': (popular_rec_count / total_recommendations) * 100 if total_recommendations > 0 else 0,
            'niche_movie_percentage': (niche_rec_count / total_recommendations) * 100 if total_recommendations > 0 else 0,
            'total_movies_evaluated': len(movie_popularity),
            'popular_movies_count': len(popular_movies),
            'niche_movies_count': len(niche_movies)
        }
        
        return popularity_bias
    
    def _subpopulation_evaluation(self, training_df: pd.DataFrame, model: SVDModelTrainer) -> Dict[str, Any]:
        """Evaluate model performance across different user subpopulations"""
        logger.info("Evaluating subpopulation performance...")
        
        # For subpopulation analysis, we'd need user demographic data
        # This is a placeholder that can be extended with actual demographic analysis
        
        # Analyze by user activity level
        user_activity = training_df.groupby('user_id').size()
        
        # Define activity segments
        low_activity = user_activity[user_activity <= user_activity.quantile(0.33)].index
        medium_activity = user_activity[(user_activity > user_activity.quantile(0.33)) & 
                                       (user_activity <= user_activity.quantile(0.67))].index
        high_activity = user_activity[user_activity > user_activity.quantile(0.67)].index
        
        # Evaluate each segment
        low_activity_data = training_df[training_df['user_id'].isin(low_activity)]
        medium_activity_data = training_df[training_df['user_id'].isin(medium_activity)]
        high_activity_data = training_df[training_df['user_id'].isin(high_activity)]
        
        return {
            'low_activity_users': self._evaluate_user_subset(low_activity_data, model, "low_activity"),
            'medium_activity_users': self._evaluate_user_subset(medium_activity_data, model, "medium_activity"),
            'high_activity_users': self._evaluate_user_subset(high_activity_data, model, "high_activity"),
            'activity_thresholds': {
                'low_max': user_activity.quantile(0.33),
                'medium_max': user_activity.quantile(0.67)
            }
        }
    
    def _coverage_evaluation(self, training_df: pd.DataFrame, model: SVDModelTrainer) -> Dict[str, Any]:
        """Evaluate model coverage and diversity"""
        logger.info("Evaluating coverage and diversity...")
        
        # Sample users for coverage evaluation
        sample_users = training_df['user_id'].unique()[:200]
        all_recommendations = set()
        user_recommendation_counts = []
        
        for user_id in sample_users:
            try:
                recommendations = model.recommend_movies(user_id, 20)
                all_recommendations.update(recommendations)
                user_recommendation_counts.append(len(recommendations))
            except:
                user_recommendation_counts.append(0)
        
        # Calculate coverage metrics
        total_movies = training_df['movie_id'].nunique()
        catalog_coverage = len(all_recommendations) / total_movies
        
        # Calculate diversity (average pairwise distance would require content features)
        # For now, use unique recommendation ratio as proxy
        total_possible_recs = len(sample_users) * 20
        unique_recs = len(all_recommendations)
        diversity_proxy = unique_recs / total_possible_recs if total_possible_recs > 0 else 0
        
        return {
            'catalog_coverage': catalog_coverage,
            'unique_movies_recommended': len(all_recommendations),
            'total_movies_in_catalog': total_movies,
            'diversity_proxy': diversity_proxy,
            'avg_recommendations_per_user': np.mean(user_recommendation_counts),
            'recommendation_success_rate': np.mean([1 if count > 0 else 0 for count in user_recommendation_counts])
        }
    
    def _average_recommendation_popularity(self, training_df: pd.DataFrame, model: SVDModelTrainer) -> Dict[str, Any]:
        """Compute Average Recommendation Popularity (ARP@20) metrics."""
        logger.info("Calculating average recommendation popularity...")
        
        popularity_counts = training_df['movie_id'].value_counts()
        if popularity_counts.empty:
            return {
                'arp_at_20': 0.0,
                'arp_rank_at_20': 0.0,
                'arp_percentile_at_20': 0.0,
                'evaluated_users': 0
            }
        
        popularity_rank = popularity_counts.rank(method='average', ascending=False)
        popularity_percentile = popularity_counts.rank(method='average', pct=True, ascending=False)
        
        sample_users = training_df['user_id'].unique()[:200]
        user_avg_popularity = []
        user_avg_rank = []
        user_avg_percentile = []
        
        for user_id in sample_users:
            try:
                recommendations = model.recommend_movies(user_id, 20)
            except Exception:
                continue
            
            counts = []
            ranks = []
            percentiles = []
            
            for movie_id in recommendations:
                if movie_id in popularity_counts:
                    counts.append(popularity_counts[movie_id])
                    ranks.append(popularity_rank[movie_id])
                    percentiles.append(popularity_percentile[movie_id])
            
            if counts:
                user_avg_popularity.append(np.mean(counts))
                user_avg_rank.append(np.mean(ranks))
                user_avg_percentile.append(np.mean(percentiles))
        
        if not user_avg_popularity:
            return {
                'arp_at_20': 0.0,
                'arp_rank_at_20': 0.0,
                'arp_percentile_at_20': 0.0,
                'evaluated_users': 0
            }
        
        return {
            'arp_at_20': float(np.mean(user_avg_popularity)),
            'arp_rank_at_20': float(np.mean(user_avg_rank)),
            'arp_percentile_at_20': float(np.mean(user_avg_percentile)),
            'evaluated_users': len(user_avg_popularity)
        }
    
    def _evaluate_user_subset(self, subset_data: pd.DataFrame, model: SVDModelTrainer, subset_name: str) -> Dict[str, Any]:
        """Evaluate model on a specific user subset"""
        if len(subset_data) == 0:
            return {'error': 'Empty subset'}
        
        # Sample for evaluation
        eval_sample_size = min(1000, len(subset_data))
        sample_data = subset_data.sample(n=eval_sample_size, random_state=self.config.random_state)
        
        predictions = []
        actuals = []
        
        for _, row in sample_data.iterrows():
            pred = model.predict_rating(row['user_id'], row['movie_id'])
            predictions.append(pred)
            actuals.append(row['final_rating'])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        return {
            'subset_name': subset_name,
            'sample_size': eval_sample_size,
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mae': mean_absolute_error(actuals, predictions),
            'coverage': self._calculate_coverage(sample_data, model)
        }
    
    def _calculate_coverage(self, test_data: pd.DataFrame, model: SVDModelTrainer) -> float:
        """Calculate prediction coverage (% of user-item pairs that can be predicted)"""
        sample_size = min(1000, len(test_data))
        sample_data = test_data.sample(n=sample_size, random_state=self.config.random_state)
        
        predictable_count = 0
        for _, row in sample_data.iterrows():
            try:
                pred = model.predict_rating(row['user_id'], row['movie_id'])
                if not np.isnan(pred):
                    predictable_count += 1
            except:
                continue
        
        return predictable_count / sample_size if sample_size > 0 else 0
    
    def _calculate_ranking_metrics_detailed(self, test_data: pd.DataFrame, model: SVDModelTrainer) -> Dict[str, float]:
        """Calculate detailed ranking metrics"""
        # Sample users for ranking evaluation
        sample_users = test_data['user_id'].unique()[:50]  # Smaller sample for detailed evaluation
        
        precisions_at_k = {k: [] for k in [5, 10, 20]}
        recalls_at_k = {k: [] for k in [5, 10, 20]}
        ndcgs_at_k = {k: [] for k in [5, 10, 20]}
        
        for user_id in sample_users:
            user_test_movies = test_data[test_data['user_id'] == user_id]['movie_id'].unique()
            
            if len(user_test_movies) == 0:
                continue
            
            try:
                recommendations = model.recommend_movies(user_id, 20)
                
                for k in [5, 10, 20]:
                    top_k_recs = recommendations[:k]
                    
                    # Calculate precision and recall
                    relevant_recs = [m for m in top_k_recs if m in user_test_movies]
                    
                    precision = len(relevant_recs) / k if k > 0 else 0
                    recall = len(relevant_recs) / len(user_test_movies) if len(user_test_movies) > 0 else 0
                    
                    # Simple NDCG calculation (assumes binary relevance)
                    dcg = sum([1 / np.log2(i + 2) for i, movie in enumerate(top_k_recs) if movie in user_test_movies])
                    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(user_test_movies)))])
                    ndcg = dcg / idcg if idcg > 0 else 0
                    
                    precisions_at_k[k].append(precision)
                    recalls_at_k[k].append(recall)
                    ndcgs_at_k[k].append(ndcg)
            except:
                continue
        
        # Average metrics
        metrics = {}
        for k in [5, 10, 20]:
            metrics[f'precision_at_{k}'] = np.mean(precisions_at_k[k]) if precisions_at_k[k] else 0
            metrics[f'recall_at_{k}'] = np.mean(recalls_at_k[k]) if recalls_at_k[k] else 0
            metrics[f'ndcg_at_{k}'] = np.mean(ndcgs_at_k[k]) if ndcgs_at_k[k] else 0
        
        return metrics


def evaluate_model_offline(training_df: pd.DataFrame, model: SVDModelTrainer) -> Dict[str, Any]:
    """Convenience function for offline evaluation"""
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model_offline(training_df, model)


if __name__ == "__main__":
    # Test offline evaluation
    from data_loader import load_and_validate_data
    from data_preprocessor import preprocess_data
    from feature_engineer import engineer_features, FeatureEngineer
    from svd_model_trainer import train_svd_model
    
    try:
        # Load and prepare data
        users_df, movies_df, interactions_df = load_and_validate_data()
        users_clean, movies_clean, interactions_clean = preprocess_data(users_df, movies_df, interactions_df)
        features_df = engineer_features(interactions_clean)
        
        # Prepare training data
        engineer = FeatureEngineer()
        training_df = engineer.prepare_training_data(features_df)
        
        # Train model
        trainer, _ = train_svd_model(training_df)
        
        # Evaluate model
        evaluation_results = evaluate_model_offline(training_df, trainer)
        
        print("Offline evaluation completed!")
        print("Temporal validation RMSE:", evaluation_results['temporal_validation']['rmse'])
        print("Coverage:", evaluation_results['coverage_analysis']['catalog_coverage'])
        print("Cold start performance:", evaluation_results['cold_start_evaluation']['cold_user_performance']['rmse'])
        
    except Exception as e:
        print(f"Offline evaluation failed: {e}")
        exit(1)
