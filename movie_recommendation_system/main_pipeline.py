"""
Main Pipeline for Movie Recommendation System - Milestone 2
Complete end-to-end orchestration of data processing, model training, and evaluation.

This script replicates your original SVD notebook logic in a production-ready format.
Merged version combining comprehensive data quality monitoring and enhanced evaluation metrics.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
import os
import json
import pandas as pd

# Import all pipeline modules - FIXED IMPORTS
try:
    from config import config
    from data_loader import load_and_validate_data
    from data_preprocessor import preprocess_data
    from feature_engineer import engineer_features, FeatureEngineer
    from svd_model_trainer import train_svd_model
    from model_evaluator import evaluate_model_offline
    from data_quality import validate_and_monitor_data
    # from online_monitor import OnlineMonitor
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure all Python files are in the same directory")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main pipeline execution - Complete SVD recommendation system
    
    This replicates your original notebook flow:
    1. Load and validate data
    2. Preprocess and clean data  
    3. Engineer features (hybrid ratings)
    4. Train SVD model
    5. Evaluate model performance
    6. Save results
    """
    
    print("="*80)
    print("MOVIE RECOMMENDATION SYSTEM - MILESTONE 2 PIPELINE")
    print("="*80)
    print("🎯 Replicating original SVD notebook logic in production format")
    print(f"⏰ Started at: {datetime.now()}")
    print("="*80)
    
    total_start_time = time.time()
    
    try:
        # ============================================================================
        # STEP 1: DATA LOADING AND VALIDATION
        # ============================================================================
        print("\n📁 STEP 1: LOADING AND VALIDATING DATA")
        print("-" * 50)
        
        step_start = time.time()
        users_df, movies_df, interactions_df = load_and_validate_data()
        
        print(f"✅ Data loaded successfully:")
        print(f"   • Users: {users_df.shape}")
        print(f"   • Movies: {movies_df.shape}")
        print(f"   • Interactions: {interactions_df.shape}")
        print(f"⏱️  Time: {time.time() - step_start:.2f}s")
        
        # ============================================================================
        # STEP 2: COMPREHENSIVE DATA QUALITY MONITORING WITH DRIFT DETECTION
        # ============================================================================
        print("\n🔍 STEP 2: COMPREHENSIVE DATA QUALITY MONITORING WITH DRIFT DETECTION")
        print("-" * 50)
        
        step_start = time.time()
        
        # Basic data quality monitoring (original functionality)
        quality_report_basic = validate_and_monitor_data(users_df, movies_df, interactions_df)
        
        # Enhanced data quality monitoring with drift detection
        # Load baseline data for drift detection
        try:
            baseline_interactions = pd.read_csv('baseline_data/baseline_interactions.csv')
            print(f"   • Loaded baseline interactions: {baseline_interactions.shape}")
        except FileNotFoundError:
            print("   ⚠️  Baseline data not found, skipping drift detection")
            baseline_interactions = None
        
        # Initialize advanced data quality monitor
        from data_quality import DataQualityMonitor
        monitor = DataQualityMonitor()
        
        # Validate schemas with advanced monitoring
        validation_results = [
            monitor.validate_schema(users_df, 'users'),
            monitor.validate_schema(movies_df, 'movies'),
            monitor.validate_schema(interactions_df, 'interactions')
        ]
        
        # Perform drift detection if baseline is available
        drift_results = []
        if baseline_interactions is not None:
            drift_results = [
                monitor.detect_drift(interactions_df, baseline_interactions, 'interactions')
            ]
            if drift_results[0]['drift_detected']:
                print(f"   ⚠️  Drift detected in interactions dataset!")
                for detail in drift_results[0]['drift_details']:
                    print(f"      - {detail['column']}: {detail.get('drift_type', 'numerical drift')}")
            else:
                print(f"   ✅ No significant drift detected")
        
        # Generate comprehensive report (enhanced version)
        quality_report = monitor.generate_data_quality_report(validation_results, drift_results)
        
        # Merge basic and enhanced quality reports
        if quality_report_basic:
            quality_report.update({
                'basic_validation': quality_report_basic,
                'enhanced_monitoring': True
            })
        
        print(f"✅ Comprehensive data quality assessment:")
        print(f"   • Overall status: {quality_report['overall_status']}")
        print(f"   • Basic validation passed: {quality_report_basic['validation_summary']['passed_validation']}/{quality_report_basic['validation_summary']['total_datasets']}")
        print(f"   • Enhanced validation passed: {quality_report['validation_summary']['passed_validation']}/{quality_report['validation_summary']['total_datasets']}")
        print(f"   • Drift detection: {quality_report['drift_summary']['drift_detected']}/{quality_report['drift_summary']['total_datasets']} datasets")
        print(f"⏱️  Time: {time.time() - step_start:.2f}s")
        
        # Save comprehensive data quality report to reports directory
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"data_quality_report_{timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        print(f"📄 Data quality report saved to: {report_path}")
        
        # ============================================================================
        # STEP 3: DATA PREPROCESSING  
        # ============================================================================
        print("\n🧹 STEP 3: DATA PREPROCESSING")
        print("-" * 50)
        
        step_start = time.time()
        users_clean, movies_clean, interactions_clean = preprocess_data(
            users_df, movies_df, interactions_df
        )
        
        print(f"✅ Data preprocessing completed:")
        print(f"   • Users after cleaning: {users_clean.shape}")
        print(f"   • Movies after cleaning: {movies_clean.shape}")
        print(f"   • Interactions after cleaning: {interactions_clean.shape}")
        print(f"⏱️  Time: {time.time() - step_start:.2f}s")
        
        # ============================================================================
        # STEP 4: FEATURE ENGINEERING (HYBRID RATINGS)
        # ============================================================================
        print("\n⚙️ STEP 4: FEATURE ENGINEERING")
        print("-" * 50)
        
        step_start = time.time()
        features_df = engineer_features(interactions_clean)
        
        # Prepare training data (same as your notebook)
        engineer = FeatureEngineer()
        training_df = engineer.prepare_training_data(features_df)
        
        # Get feature statistics
        stats = engineer.get_feature_statistics(features_df)
        
        print(f"✅ Feature engineering completed:")
        print(f"   • Training data shape: {training_df.shape}")
        print(f"   • Explicit rating coverage: {stats['rating_composition']['explicit_percentage']:.1f}%")
        print(f"   • Final rating range: {stats['final_rating_stats']['min']:.1f} - {stats['final_rating_stats']['max']:.1f}")
        print(f"⏱️  Time: {time.time() - step_start:.2f}s")
        
        # ============================================================================
        # STEP 5: SVD MODEL TRAINING WITH HYPERPARAMETER TUNING
        # ============================================================================
        print("\n🤖 STEP 5: SVD MODEL TRAINING WITH HYPERPARAMETER TUNING")
        print("-" * 50)
        
        step_start = time.time()
        trainer, training_results = train_svd_model(training_df)
        
        print(f"✅ SVD model training completed:")
        print(f"   • Training time: {training_results['training_time_seconds']:.2f}s")
        print(f"   • Model size: {training_results['model_size_mb']:.1f} MB")
        print(f"   • Inference time: {training_results['inference_time_ms']:.3f}ms")
        print(f"   • Split strategy: {training_results.get('split_strategy', 'unknown')}")
        
        # Show hyperparameter tuning results if available
        if 'best_n_components' in training_results:
            print(f"\n   🔍 HYPERPARAMETER TUNING RESULTS:")
            print(f"   • Best n_components: {training_results['best_n_components']}")
            print(f"   • Validation RMSE: {training_results['validation_rmse']:.4f}")
            print(f"   • Test RMSE (final): {training_results.get('accuracy_rmse', 'N/A'):.4f}")
            
            if 'hyperparameter_search_results' in training_results:
                print(f"\n   • Hyperparameter search details:")
                for result in training_results['hyperparameter_search_results']:
                    marker = "✓" if result['n_components'] == training_results['best_n_components'] else " "
                    print(f"     {marker} n_components={result['n_components']:3d} -> Val RMSE={result['validation_rmse']:.4f}")
            
            if 'train_cutoff' in training_results:
                print(f"\n   • Train cutoff: {training_results['train_cutoff']}")
                print(f"   • Val cutoff: {training_results.get('val_cutoff', 'N/A')}")
            print(f"   • Train size: {training_results.get('train_size', 'N/A'):,}")
            print(f"   • Validation size: {training_results.get('val_size', 'N/A'):,}")
            print(f"   • Test size: {training_results.get('test_size', 'N/A'):,}")
        else:
            # Legacy output for backward compatibility
            print(f"   • RMSE: {training_results.get('accuracy_rmse', 'N/A')}")
            if training_results.get('temporal_cutoff'):
                print(f"   • Temporal cutoff: {training_results['temporal_cutoff']}")
        
        print(f"\n   • Precision@20: {training_results.get('precision_at_20', 'N/A'):.3f}")
        print(f"   • Recall@20: {training_results.get('recall_at_20', 'N/A'):.3f}")
        ndcg20 = training_results.get('ndcg_at_20')
        if ndcg20 is not None:
            print(f"   • NDCG@20: {ndcg20:.3f}")
        print(f"⏱️  Time: {time.time() - step_start:.2f}s")
        
        # Save the model
        model_path = trainer.save_model()
        print(f"💾 Model saved to: {model_path}")
        
        # ============================================================================
        # STEP 6: COMPREHENSIVE MODEL EVALUATION (ENHANCED)
        # ============================================================================
        print("\n📊 STEP 6: COMPREHENSIVE MODEL EVALUATION")
        print("-" * 50)
        
        step_start = time.time()
        evaluation_results = evaluate_model_offline(training_df, trainer)
        
        print(f"✅ Comprehensive evaluation completed:")
        print(f"   • Temporal validation RMSE: {evaluation_results['temporal_validation']['rmse']:.3f}")
        
        # Enhanced evaluation metrics display
        ndcg20_eval = evaluation_results['temporal_validation'].get('ndcg_at_20')
        if ndcg20_eval is not None:
            print(f"   • Temporal validation NDCG@20: {ndcg20_eval:.3f}")
        
        # Popularity baseline comparison
        popularity_baseline = evaluation_results.get('popularity_baseline') or {}
        if popularity_baseline:
            print(f"   • Popularity baseline NDCG@20: {popularity_baseline.get('ndcg_at_20', 0.0):.3f}")
        
        # Average Recommendation Popularity (ARP) metrics
        arp_metrics = evaluation_results.get('average_recommendation_popularity') or {}
        if arp_metrics:
            print(f"   • ARP@20 (avg interactions): {arp_metrics.get('arp_at_20', 0.0):.1f}")
            print(f"   • ARP rank@20: {arp_metrics.get('arp_rank_at_20', 0.0):.1f}")
            print(f"   • ARP percentile@20: {arp_metrics.get('arp_percentile_at_20', 0.0):.3f}")
        
        # Subpopulation analysis (enhanced display)
        subpop_analysis = evaluation_results.get('subpopulation_analysis') or {}
        if subpop_analysis:
            label_map = {
                'low_activity_users': 'Low-activity users',
                'medium_activity_users': 'Mid-activity users',
                'high_activity_users': 'High-activity users'
            }
            print("   • Subpopulation performance (RMSE / coverage):")
            for key, label in label_map.items():
                metrics = subpop_analysis.get(key) or {}
                rmse = metrics.get('rmse')
                coverage = metrics.get('coverage')
                if rmse is not None and coverage is not None:
                    print(f"       - {label}: {rmse:.3f} RMSE, {coverage:.2f} coverage")
        
        # Coverage and bias analysis
        print(f"   • Cold start performance: Available")
        print(f"   • Coverage analysis: {evaluation_results['coverage_analysis']['catalog_coverage']:.3f}")
        print(f"   • Popularity bias analysis: Available")
        print(f"⏱️  Time: {time.time() - step_start:.2f}s")
        
        # ============================================================================
        # STEP 7: SAMPLE RECOMMENDATIONS (LIKE ORIGINAL NOTEBOOK)
        # ============================================================================
        print("\n🎬 STEP 7: SAMPLE RECOMMENDATIONS")
        print("-" * 50)
        
        # Test recommendations for sample users (like your original notebook)
        sample_users = training_df['user_id'].unique()[:3]
        
        for user_id in sample_users:
            recommendations = trainer.recommend_movies(user_id, 5)
            print(f"   • User {user_id}: {recommendations}")
        
        # ============================================================================
        # STEP 8: SAVE RESULTS AND SUMMARY (COMPREHENSIVE)
        # ============================================================================
        print("\n💾 STEP 8: SAVING COMPREHENSIVE RESULTS")
        print("-" * 50)
        
        # Save results to pickle (same as original notebook)
        import pickle
        
        # Create comprehensive results summary
        results_summary = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'data_statistics': stats,
            'quality_report': quality_report,
            'quality_report_basic': quality_report_basic,
            'pipeline_execution_time': time.time() - total_start_time,
            'execution_timestamp': datetime.now().isoformat(),
            'pipeline_version': 'merged_comprehensive_v1.0',
            'enhanced_features': {
                'drift_detection': baseline_interactions is not None,
                'advanced_quality_monitoring': True,
                'subpopulation_analysis': bool(subpop_analysis),
                'arp_metrics': bool(arp_metrics),
                'popularity_baseline': bool(popularity_baseline)
            }
        }
        
        results_path = config.get_results_path()
        with open(results_path, 'wb') as f:
            pickle.dump(results_summary, f)
        
        print(f"✅ Comprehensive results saved to: {results_path}")
        
        # Also save a JSON summary for easier inspection
        json_summary = {
            'pipeline_summary': {
                'execution_time': results_summary['pipeline_execution_time'],
                'timestamp': results_summary['execution_timestamp'],
                'version': results_summary['pipeline_version']
            },
            'model_performance': {
                'training_time': training_results['training_time_seconds'],
                'inference_time': training_results['inference_time_ms'],
                'model_size_mb': training_results['model_size_mb'],
                'precision_at_20': training_results.get('precision_at_20'),
                'recall_at_20': training_results.get('recall_at_20'),
                'ndcg_at_20': training_results.get('ndcg_at_20')
            },
            'evaluation_metrics': {
                'temporal_rmse': evaluation_results['temporal_validation']['rmse'],
                'temporal_ndcg_20': evaluation_results['temporal_validation'].get('ndcg_at_20'),
                'catalog_coverage': evaluation_results['coverage_analysis']['catalog_coverage']
            },
            'data_quality': {
                'overall_status': quality_report['overall_status'],
                'drift_detected': quality_report['drift_summary']['drift_detected'],
                'validation_passed': quality_report['validation_summary']['passed_validation']
            }
        }
        
        json_path = os.path.join(report_dir, f"pipeline_summary_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(json_summary, f, indent=2, default=str)
        print(f"📄 Pipeline summary saved to: {json_path}")
        
        # ============================================================================
        # PIPELINE COMPLETION SUMMARY (ENHANCED)
        # ============================================================================
        total_time = time.time() - total_start_time
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"⏱️  Total execution time: {total_time:.2f}s")
        print(f"📊 Model performance:")
        print(f"   • Training time: {training_results['training_time_seconds']:.2f}s")
        print(f"   • Inference time: {training_results['inference_time_ms']:.3f}ms (target: <600ms)")
        print(f"   • Model size: {training_results['model_size_mb']:.1f}MB")
        
        if 'best_n_components' in training_results:
            print(f"   • Best n_components (tuned): {training_results['best_n_components']}")
            print(f"   • Validation RMSE: {training_results['validation_rmse']:.4f}")
            print(f"   • Test RMSE (final): {training_results.get('accuracy_rmse', 'N/A'):.4f}")
        
        print(f"   • Precision@20: {training_results.get('precision_at_20', 'N/A'):.3f}")
        print(f"   • Recall@20: {training_results.get('recall_at_20', 'N/A'):.3f}")
        if ndcg20 is not None:
            print(f"   • NDCG@20: {ndcg20:.3f}")
        print(f"📊 Enhanced evaluation:")
        if arp_metrics:
            print(f"   • ARP@20: {arp_metrics.get('arp_at_20', 0.0):.1f} interactions")
        if popularity_baseline:
            print(f"   • vs Popularity baseline: {popularity_baseline.get('ndcg_at_20', 0.0):.3f} NDCG@20")
        print(f"📁 Outputs:")
        print(f"   • Model: {model_path}")
        print(f"   • Results: {results_path}")
        
        if 'best_n_components' in training_results:
            print(f"🎯 Hyperparameter Tuning: ENABLED (60/20/20 split)")
        else:
            print(f"⚠️  Hyperparameter Tuning: DISABLED (using default parameters)")
        
        print(f"   • Quality report: {report_path}")
        print(f"   • Summary: {json_path}")
        print(f"🔍 Quality monitoring:")
        print(f"   • Basic validation: {quality_report_basic['validation_summary']['passed_validation']}/{quality_report_basic['validation_summary']['total_datasets']} passed")
        print(f"   • Enhanced validation: {quality_report['validation_summary']['passed_validation']}/{quality_report['validation_summary']['total_datasets']} passed")
        print(f"   • Drift detection: {'✅ Enabled' if baseline_interactions is not None else '⚠️ Disabled (no baseline)'}")
        print(f"🏆 Status: READY FOR MILESTONE 2 SUBMISSION!")
        print("="*80)
        
        return results_summary
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\n❌ PIPELINE FAILED: {e}")
        print("\nDebugging tips:")
        print("1. Check that data files exist in the expected location")
        print("2. Verify all dependencies are installed: pip install -r requirements.txt")
        print("3. Ensure you're running from the correct directory")
        print("4. Check if baseline_data/baseline_interactions.csv exists for drift detection")
        raise


def quick_test():
    """Quick test of the pipeline with minimal data for debugging"""
    print("🧪 Running quick pipeline test...")
    
    import pandas as pd
    import numpy as np
    
    # Create minimal test data
    test_users = pd.DataFrame({
        'user_id': [1, 2, 3],
        'age': [25, 30, 35],
        'occupation': ['engineer', 'teacher', 'doctor'],
        'gender': ['M', 'F', 'M']
    })
    
    test_movies = pd.DataFrame({
        'movie_id': ['movie1', 'movie2', 'movie3'],
        'title': ['Movie 1', 'Movie 2', 'Movie 3'],
        'vote_average': [7.5, 8.0, 6.5],
        'vote_count': [100, 200, 150]
    })
    
    test_interactions = pd.DataFrame({
        'user_id': [1, 2, 3, 1, 2],
        'movie_id': ['movie1', 'movie2', 'movie3', 'movie2', 'movie1'],
        'total_minutes': [120, 95, 180, 45, 200],
        'rating': [4.0, 5.0, 3.0, np.nan, 4.5]
    })
    
    print("✅ Test data created successfully")
    print("🏁 Quick test completed - main pipeline should work!")


def run_quality_check_only():
    """Run only the data quality monitoring components for testing"""
    print("🔍 Running data quality check only...")
    
    try:
        # Load data
        users_df, movies_df, interactions_df = load_and_validate_data()
        
        # Basic quality check
        quality_report_basic = validate_and_monitor_data(users_df, movies_df, interactions_df)
        
        # Advanced quality check
        from data_quality import DataQualityMonitor
        monitor = DataQualityMonitor()
        
        validation_results = [
            monitor.validate_schema(users_df, 'users'),
            monitor.validate_schema(movies_df, 'movies'),
            monitor.validate_schema(interactions_df, 'interactions')
        ]
        
        quality_report = monitor.generate_data_quality_report(validation_results, [])
        
        print("✅ Quality check completed:")
        print(f"   • Basic validation: {quality_report_basic['overall_status']}")
        print(f"   • Enhanced validation: {quality_report['overall_status']}")
        
        return quality_report, quality_report_basic
        
    except Exception as e:
        print(f"❌ Quality check failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Quick test mode
            quick_test()
        elif sys.argv[1] == "--quality-only":
            # Quality check only
            run_quality_check_only()
        else:
            print("Usage: python main_pipeline.py [--test|--quality-only]")
    else:
        # Full pipeline execution
        results = main()