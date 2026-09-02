#!/usr/bin/env python3
"""
Online Evaluation Telemetry Analysis - Kafka Production Version
===============================================================
This script performs TRUE online evaluation using real telemetry data from Kafka.
No fallbacks to CSV simulation - production-ready for Milestone 2.

Author: Sarah Lang
Date: October 25, 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import warnings
import re
from collections import Counter
warnings.filterwarnings('ignore')

# Core libraries
import os
import sys

# Create reports directory if it doesn't exist
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Kafka libraries
try:
    from kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    print("Warning: kafka-python not installed. Install with: pip install kafka-python")
    KAFKA_AVAILABLE = False

print("Libraries imported successfully!")
print("Online evaluation telemetry analysis ready")

# ============================================================================
# Section 1: Kafka Configuration and Data Loading
# ============================================================================

# Kafka Configuration for Course Telemetry Data Collection
KAFKA_CONFIG = {
    'bootstrap_servers': 'localhost:9092',
    'topic': 'movielog6',
    'group_id': 'online-eval-course-data'
}

def setup_kafka_connection():
    """Setup Kafka consumer for telemetry data collection"""
    if not KAFKA_AVAILABLE:
        print("❌ Kafka not available - install kafka-python to enable")
        return None
        
    try:
        consumer = KafkaConsumer(
            KAFKA_CONFIG['topic'],
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            consumer_timeout_ms=150000,
            value_deserializer=lambda x: x.decode('utf-8')
        )
        print("✅ Kafka connection established successfully")
        return consumer
        
    except Exception as e:
        print(f"❌ Failed to connect to Kafka: {e}")
        return None

def load_real_kafka_data():
    """Load real telemetry data from COURSE Kafka stream - PRIMARY DATA SOURCE"""
    print("🔄 Loading REAL production data from COURSE Kafka broker...")
    print("📡 Connecting via SSH tunnel to 128.2.220.241...")
    
    consumer = setup_kafka_connection()
    if not consumer:
        raise Exception("❌ Cannot connect to COURSE Kafka broker - ensure SSH tunnel is active")
    
    events = []
    print("📥 Reading real recommendation and interaction logs from course broker...")
    
    try:
        for message in consumer:
            try:
                msg_str = message.value
                parts = msg_str.split(',', 2)  # Split only on first 2 commas!
                if len(parts) < 3:
                    continue

                # Parse the message format: timestamp,user_id,action
                timestamp = parts[0]
                user_id = parts[1]
                action = parts[2]  # Everything after second comma
                
                # Validate timestamp before processing
                try:
                    pd.to_datetime(timestamp)
                except:
                    continue
                
                # Determine event type based on action
                if action.startswith('GET /data/m/'):
                    # Extract movie title from URL
                    movie_title = None
                    try:
                        movie_title = action.split('/data/m/')[1].split('/')[0]
                    except:
                        pass
                    
                    event = {
                        'timestamp': timestamp,
                        'user_id': user_id,
                        'event_type': 'user_interaction',
                        'interaction_type': 'click',
                        'movie_id': movie_title,
                        'rating': None,
                        'event_details': action
                    }
                elif action.startswith('GET /rate/'):
                    # Rating event
                    movie_title = None
                    rating = None
                    try:
                        rating_part = action.split('GET /rate/')[1]
                        movie_title = rating_part.split('=')[0]
                        rating = float(rating_part.split('=')[1])
                    except:
                        pass
                    
                    event = {
                        'timestamp': timestamp,
                        'user_id': user_id,
                        'event_type': 'user_interaction',
                        'interaction_type': 'rating',
                        'movie_id': movie_title,
                        'rating': rating,
                        'event_details': action
                    }
                elif 'recommendation request' in action.lower():
                    # Parse recommendation response
                    recommendations = []
                    response_time = 0
                    server = 'unknown'
                    
                    try:
                        # Extract server
                        if 'recommendation request' in action:
                            server_part = action.split('recommendation request')[1].split(',')[0].strip()
                            server = server_part
                        
                        # Extract recommendations
                        if 'result:' in action:
                            result_part = action.split('result:')[1].strip()
                            
                            # Remove response time using regex
                            time_pattern = r',\s*(\d+)\s*ms\s*$'
                            time_match = re.search(time_pattern, result_part)
                            if time_match:
                                response_time = float(time_match.group(1))
                                result_part = result_part[:time_match.start()]
                            
                            # Split by comma to get movies
                            recommendations = [m.strip() for m in result_part.split(',') if m.strip()]
                        
                    except Exception as e:
                        pass
                    
                    event = {
                        'timestamp': timestamp,
                        'user_id': user_id,
                        'event_type': 'recommendation_request',
                        'recommendations': recommendations,
                        'model_used': server,
                        'response_time_ms': response_time,
                        'event_details': action
                    }
                else:
                    # Other event type
                    event = {
                        'timestamp': timestamp,
                        'user_id': user_id,
                        'event_type': 'other',
                        'event_details': action
                    }
                
                events.append(event)

                if len(events) % 50 == 0:
                    print(f"   📊 Collected {len(events)} real course events...")
                    
                if len(events) >= 10000:
                    print("   ⏸️ Reached 10000 events limit for analysis...")
                    break
                    
            except Exception as e:
                continue
        
        consumer.close()
        
        if len(events) == 0:
            raise Exception("❌ No telemetry data found in course Kafka broker")
        
        print(f"✅ Successfully loaded {len(events)} real course events from Kafka")
        
        # Convert to DataFrames
        df_all = pd.DataFrame(events)
        
        # Filter for recommendations
        df_recommendations = df_all[df_all['event_type'] == 'recommendation_request'].copy()
        
        if len(df_recommendations) > 0:
            if 'recommendations' not in df_recommendations.columns:
                df_recommendations['recommendations'] = [[] for _ in range(len(df_recommendations))]
            if 'model_used' not in df_recommendations.columns:
                df_recommendations['model_used'] = 'unknown'
            if 'response_time_ms' not in df_recommendations.columns:
                df_recommendations['response_time_ms'] = 0
        
        # Filter for interactions
        df_interactions = df_all[df_all['event_type'] == 'user_interaction'].copy()
        
        if len(df_interactions) > 0:
            required_cols = ['interaction_type', 'movie_id', 'rating']
            for col in required_cols:
                if col not in df_interactions.columns:
                    df_interactions[col] = None
        
        print(f"📊 REAL Course Data Summary:")
        print(f"   • Recommendation requests: {len(df_recommendations)}")
        print(f"   • User interactions: {len(df_interactions)}")
        if not df_interactions.empty and 'interaction_type' in df_interactions.columns:
            print(f"   • Interaction types: {df_interactions['interaction_type'].value_counts().to_dict()}")
        print(f"   • Event types: {df_all['event_type'].value_counts().to_dict()}")
        print(f"   • Data source: Course Kafka broker (128.2.220.241)")
        
        return df_recommendations, df_interactions, df_all
        
    except Exception as e:
        consumer.close()
        raise Exception(f"❌ Failed to load course Kafka data: {e}")
        
    finally:
        try:
            consumer.close()
        except:
            pass

# ============================================================================
# Section 2: Online Evaluation Metrics with Movie Name Normalization
# ============================================================================

def normalize_movie_name(movie_name):
    """Normalize movie names for consistent matching"""
    if pd.isna(movie_name) or movie_name is None:
        return ""
    
    movie_str = str(movie_name).lower().strip()
    # Remove URL encoding artifacts
    movie_str = movie_str.replace('+', ' ')
    movie_str = movie_str.replace('%20', ' ')
    # Remove extra whitespace
    movie_str = re.sub(r'\s+', ' ', movie_str)
    return movie_str.strip()


def calculate_click_through_rate(df_recommendations, df_interactions):
    """Calculate Click-Through Rate from real telemetry data"""
    if df_recommendations.empty:
        return {"ctr_percentage": 0.0, "total_recommendations": 0, "total_clicks": 0}
    
    total_recommendations = df_recommendations['recommendations'].apply(len).sum()
    
    # Find clicks that match ANY recommended movie (with time window)
    clicks_on_recommendations = 0
    
    for _, rec_row in df_recommendations.iterrows():
        user_id = rec_row['user_id']
        recommendations = rec_row.get('recommendations', [])
        rec_timestamp = pd.to_datetime(rec_row['timestamp'], errors='coerce')
        
        if pd.isna(rec_timestamp) or not recommendations:
            continue
        
        # Normalize recommendation names
        normalized_recs = [normalize_movie_name(m) for m in recommendations]
        
        # Look for clicks within 30 minutes AFTER recommendation
        time_window_end = rec_timestamp + pd.Timedelta(minutes=30)
        
        df_interactions_copy = df_interactions.copy()
        df_interactions_copy['timestamp'] = pd.to_datetime(df_interactions_copy['timestamp'], errors='coerce')
        df_interactions_copy['normalized_movie'] = df_interactions_copy['movie_id'].apply(normalize_movie_name)
        
        # Find user's clicks on recommended movies within time window
        user_clicks = df_interactions_copy[
            (df_interactions_copy['user_id'] == user_id) & 
            (df_interactions_copy['timestamp'] >= rec_timestamp) &
            (df_interactions_copy['timestamp'] <= time_window_end) &
            (df_interactions_copy['interaction_type'] == 'click') &
            (df_interactions_copy['normalized_movie'].isin(normalized_recs)) &
            (~df_interactions_copy['timestamp'].isna())
        ]
        
        clicks_on_recommendations += len(user_clicks)
    
    ctr_percentage = (clicks_on_recommendations / total_recommendations * 100) if total_recommendations > 0 else 0.0
    
    return {
        "ctr_percentage": ctr_percentage,
        "total_recommendations": total_recommendations,
        "total_clicks": clicks_on_recommendations
    }

def calculate_precision_at_k(df_recommendations, df_interactions, k=10):
    """Calculate Precision@K for top-K recommendations"""
    if df_recommendations.empty or df_interactions.empty:
        return {"precision_at_k": 0.0, "k": k, "num_users_evaluated": 0}
    
    precisions = []
    debug_info = {
        'total_recs': 0,
        'recs_with_clicks': 0,
        'total_matches': 0
    }
    
    try:
        for _, rec_row in df_recommendations.iterrows():
            user_id = rec_row['user_id']
            recommendations = rec_row.get('recommendations', [])
            rec_timestamp = pd.to_datetime(rec_row['timestamp'], errors='coerce')
            
            if pd.isna(rec_timestamp) or not isinstance(recommendations, list) or len(recommendations) == 0:
                continue
                
            debug_info['total_recs'] += 1
            top_k_recs = [normalize_movie_name(m) for m in recommendations[:k]]
            
            df_interactions_copy = df_interactions.copy()
            df_interactions_copy['timestamp'] = pd.to_datetime(df_interactions_copy['timestamp'], errors='coerce')
            df_interactions_copy['normalized_movie'] = df_interactions_copy['movie_id'].apply(normalize_movie_name)
            
            # Look for clicks within 30 minutes AFTER recommendation
            time_window_end = rec_timestamp + pd.Timedelta(minutes=30)
            
            user_interactions = df_interactions_copy[
                (df_interactions_copy['user_id'] == user_id) & 
                (df_interactions_copy['timestamp'] >= rec_timestamp) &
                (df_interactions_copy['timestamp'] <= time_window_end) &
                (~df_interactions_copy['timestamp'].isna())
            ]
            
            if len(user_interactions) > 0:
                debug_info['recs_with_clicks'] += 1
                clicked_movies = user_interactions[
                    user_interactions['interaction_type'] == 'click'
                ]['normalized_movie'].tolist()
                
                # Count matches
                relevant_in_topk = len(set(top_k_recs) & set(clicked_movies))
                if relevant_in_topk > 0:
                    debug_info['total_matches'] += relevant_in_topk
                
                precision = relevant_in_topk / k
                precisions.append(precision)
    except Exception as e:
        print(f"⚠️ Warning: Error calculating precision: {e}")
    
    avg_precision = np.mean(precisions) if precisions else 0.0
    
    # Print debug info
    print(f"   Debug: {debug_info['total_recs']} recommendations evaluated")
    print(f"   Debug: {debug_info['recs_with_clicks']} had subsequent user clicks")
    print(f"   Debug: {debug_info['total_matches']} total matches found")
    print(f"   Debug: {len(precisions)} users contributed to precision score")
    
    return {
        "precision_at_k": avg_precision,
        "k": k,
        "num_users_evaluated": len(precisions),
        "total_recommendations_evaluated": len(df_recommendations)
    }

def calculate_response_time_metrics(df_recommendations):
    """Calculate response time performance metrics"""
    if df_recommendations.empty or 'response_time_ms' not in df_recommendations.columns:
        return {"avg_response_time": 0, "median_response_time": 0, "p95_response_time": 0, "p99_response_time": 0}
    
    response_times = df_recommendations['response_time_ms'].dropna()
    
    if len(response_times) == 0:
        return {"avg_response_time": 0, "median_response_time": 0, "p95_response_time": 0, "p99_response_time": 0}
    
    return {
        "avg_response_time": response_times.mean(),
        "median_response_time": response_times.median(),
        "p95_response_time": response_times.quantile(0.95),
        "p99_response_time": response_times.quantile(0.99)
    }

def calculate_model_usage_metrics(df_recommendations):
    """Calculate model usage and fallback metrics"""
    if df_recommendations.empty or 'model_used' not in df_recommendations.columns:
        return {"fallback_rate": 0.0, "model_distribution": {}, "total_requests": 0}
    
    model_counts = df_recommendations['model_used'].value_counts()
    total_requests = len(df_recommendations)
    
    fallback_requests = len(df_recommendations[df_recommendations['model_used'] == 'ALS_fallback'])
    fallback_rate = fallback_requests / total_requests if total_requests > 0 else 0.0
    
    return {
        "fallback_rate": fallback_rate,
        "model_distribution": model_counts.to_dict(),
        "total_requests": total_requests
    }

# ============================================================================
# Section 2.5: Model Quality Metrics (Attribution-Free)
# ============================================================================

def calculate_model_quality_metrics(df_recommendations, df_interactions):
    """
    Calculate model quality metrics that work WITHOUT click attribution.
    These metrics assess recommendation quality independently.
    """
    if df_recommendations.empty:
        return {"error": "No recommendation data available"}
    
    metrics = {}
    
    # 1. DIVERSITY: How varied are the recommendations?
    all_recommendations = []
    for recs in df_recommendations['recommendations']:
        if isinstance(recs, list):
            all_recommendations.extend([normalize_movie_name(m) for m in recs])
    
    unique_movies = len(set(all_recommendations))
    total_slots = len(all_recommendations)
    
    metrics['recommendation_diversity'] = {
        'unique_movies_recommended': unique_movies,
        'total_recommendation_slots': total_slots,
        'diversity_ratio': unique_movies / total_slots if total_slots > 0 else 0,
        'interpretation': 'Low diversity suggests cold-start or popularity bias'
    }
    
    # 2. COVERAGE: What % of clicked movies appear in recommendations?
    if not df_interactions.empty:
        df_interactions_copy = df_interactions.copy()
        df_interactions_copy['normalized_movie'] = df_interactions_copy['movie_id'].apply(normalize_movie_name)
        
        clicked_movies = set(df_interactions_copy[
            df_interactions_copy['interaction_type'] == 'click'
        ]['normalized_movie'].dropna())
        
        recommended_movies = set(all_recommendations)
        
        coverage = len(clicked_movies & recommended_movies) / len(clicked_movies) if clicked_movies else 0
        
        metrics['catalog_coverage'] = {
            'movies_clicked_by_users': len(clicked_movies),
            'movies_in_recommendations': len(recommended_movies),
            'overlap': len(clicked_movies & recommended_movies),
            'coverage_percentage': coverage * 100,
            'interpretation': 'High coverage means recommendations include popular movies'
        }
    
    # 3. POPULARITY BIAS: Are recommendations concentrated on few movies?
    rec_counts = Counter(all_recommendations)
    
    if len(rec_counts) > 0:
        # Top-K concentration
        sorted_counts = sorted(rec_counts.values(), reverse=True)
        top_10_movies = sorted_counts[:10] if len(sorted_counts) >= 10 else sorted_counts
        top_10_percentage = sum(top_10_movies) / sum(sorted_counts) * 100 if sum(sorted_counts) > 0 else 0
        
        metrics['popularity_bias'] = {
            'unique_movies': len(rec_counts),
            'top_10_concentration': top_10_percentage,
            'most_recommended_movie': rec_counts.most_common(1)[0][0] if rec_counts else None,
            'most_recommended_count': rec_counts.most_common(1)[0][1] if rec_counts else 0,
            'interpretation': f'{top_10_percentage:.1f}% of recommendations are top-10 movies'
        }
    
    # 4. PERSONALIZATION: Are recommendations different per user?
    user_recs = {}
    for _, row in df_recommendations.iterrows():
        user_id = row['user_id']
        recs = [normalize_movie_name(m) for m in row.get('recommendations', [])]
        if user_id not in user_recs:
            user_recs[user_id] = []
        user_recs[user_id].extend(recs)
    
    if len(user_recs) > 1:
        # Calculate Jaccard similarity between users
        user_ids = list(user_recs.keys())
        similarities = []
        
        for i in range(min(10, len(user_ids))):  # Sample 10 users
            for j in range(i+1, min(10, len(user_ids))):
                set_i = set(user_recs[user_ids[i]])
                set_j = set(user_recs[user_ids[j]])
                if len(set_i | set_j) > 0:
                    jaccard = len(set_i & set_j) / len(set_i | set_j)
                    similarities.append(jaccard)
        
        avg_similarity = np.mean(similarities) if similarities else 1.0
        
        metrics['personalization'] = {
            'users_with_recommendations': len(user_recs),
            'avg_user_similarity': avg_similarity,
            'personalization_score': 1 - avg_similarity,  # Higher is more personalized
            'interpretation': f'Recommendations are {(1-avg_similarity)*100:.1f}% personalized'
        }
    
    # 5. CONSISTENCY: How stable are recommendations over time?
    df_recommendations_sorted = df_recommendations.copy()
    df_recommendations_sorted['timestamp'] = pd.to_datetime(df_recommendations_sorted['timestamp'], errors='coerce')
    df_recommendations_sorted = df_recommendations_sorted.sort_values('timestamp')
    
    if len(df_recommendations_sorted) > 1:
        first_half = df_recommendations_sorted.iloc[:len(df_recommendations_sorted)//2]
        second_half = df_recommendations_sorted.iloc[len(df_recommendations_sorted)//2:]
        
        first_half_movies = set()
        for recs in first_half['recommendations']:
            if isinstance(recs, list):
                first_half_movies.update([normalize_movie_name(m) for m in recs])
        
        second_half_movies = set()
        for recs in second_half['recommendations']:
            if isinstance(recs, list):
                second_half_movies.update([normalize_movie_name(m) for m in recs])
        
        consistency = len(first_half_movies & second_half_movies) / len(first_half_movies | second_half_movies) if (first_half_movies | second_half_movies) else 1.0
        
        metrics['temporal_consistency'] = {
            'first_half_unique_movies': len(first_half_movies),
            'second_half_unique_movies': len(second_half_movies),
            'consistency_score': consistency,
            'interpretation': 'High consistency may indicate static recommendations'
        }
    
    return metrics


def print_model_quality_report(metrics):
    """Pretty print model quality metrics"""
    print("\n" + "="*70)
    print("📊 MODEL QUALITY ASSESSMENT (Attribution-Free Metrics)")
    print("="*70)
    
    if 'error' in metrics:
        print(f"❌ {metrics['error']}")
        return
    
    # Diversity
    if 'recommendation_diversity' in metrics:
        div = metrics['recommendation_diversity']
        print(f"\n🎨 DIVERSITY:")
        print(f"   Unique movies recommended: {div['unique_movies_recommended']}")
        print(f"   Total recommendation slots: {div['total_recommendation_slots']}")
        print(f"   Diversity ratio: {div['diversity_ratio']:.3f}")
        print(f"   ⚠️  {div['interpretation']}")
    
    # Coverage
    if 'catalog_coverage' in metrics:
        cov = metrics['catalog_coverage']
        print(f"\n📚 CATALOG COVERAGE:")
        print(f"   Movies clicked by users: {cov['movies_clicked_by_users']}")
        print(f"   Movies in recommendations: {cov['movies_in_recommendations']}")
        print(f"   Overlap: {cov['overlap']} movies")
        print(f"   Coverage: {cov['coverage_percentage']:.1f}%")
        print(f"   ℹ️  {cov['interpretation']}")
    
    # Popularity bias
    if 'popularity_bias' in metrics:
        pop = metrics['popularity_bias']
        print(f"\n⭐ POPULARITY BIAS:")
        print(f"   Unique movies: {pop['unique_movies']}")
        print(f"   Top-10 concentration: {pop['top_10_concentration']:.1f}%")
        print(f"   Most recommended: {pop['most_recommended_movie']}")
        print(f"   Times recommended: {pop['most_recommended_count']}")
        print(f"   ℹ️  {pop['interpretation']}")
    
    # Personalization
    if 'personalization' in metrics:
        pers = metrics['personalization']
        print(f"\n👤 PERSONALIZATION:")
        print(f"   Users evaluated: {pers['users_with_recommendations']}")
        print(f"   Avg user similarity: {pers['avg_user_similarity']:.3f}")
        print(f"   Personalization score: {pers['personalization_score']:.3f}")
        print(f"   ℹ️  {pers['interpretation']}")
    
    # Consistency
    if 'temporal_consistency' in metrics:
        cons = metrics['temporal_consistency']
        print(f"\n⏱️  TEMPORAL CONSISTENCY:")
        print(f"   Early period movies: {cons['first_half_unique_movies']}")
        print(f"   Late period movies: {cons['second_half_unique_movies']}")
        print(f"   Consistency score: {cons['consistency_score']:.3f}")
        print(f"   ℹ️  {cons['interpretation']}")
    
    print("="*70)

# ============================================================================
# Section 3: Analysis Functions
# ============================================================================

def analyze_user_engagement_patterns(df_interactions):
    """Analyze user engagement patterns from real interactions"""
    if df_interactions.empty:
        return {"error": "No interaction data available"}
    
    user_activity = df_interactions.groupby('user_id').size()
    interaction_dist = df_interactions['interaction_type'].value_counts()
    
    hourly_activity = pd.Series()
    daily_activity = pd.Series()
    try:
        df_interactions_copy = df_interactions.copy()
        df_interactions_copy['timestamp'] = pd.to_datetime(df_interactions_copy['timestamp'], errors='coerce')
        df_interactions_copy = df_interactions_copy.dropna(subset=['timestamp'])
        
        if not df_interactions_copy.empty:
            df_interactions_copy['hour'] = df_interactions_copy['timestamp'].dt.hour
            df_interactions_copy['day_of_week'] = df_interactions_copy['timestamp'].dt.dayofweek
            
            hourly_activity = df_interactions_copy.groupby('hour').size()
            daily_activity = df_interactions_copy.groupby('day_of_week').size()
    except Exception as e:
        print(f"⚠️ Warning: Could not parse timestamps: {e}")
    
    rating_stats = None
    try:
        valid_ratings = df_interactions[df_interactions['rating'].notna()]['rating']
        if len(valid_ratings) > 0:
            rating_stats = valid_ratings.describe()
    except:
        pass
    
    return {
        'user_activity_summary': user_activity.describe(),
        'interaction_distribution': interaction_dist,
        'hourly_activity': hourly_activity,
        'daily_activity': daily_activity,
        'rating_statistics': rating_stats,
        'total_unique_users': df_interactions['user_id'].nunique(),
        'total_unique_movies': df_interactions['movie_id'].nunique()
    }

def analyze_recommendation_effectiveness(df_recommendations, df_interactions):
    """Analyze recommendation model effectiveness"""
    if df_recommendations.empty:
        return {"error": "No recommendation data available", "model_performance": {}, "hourly_performance": pd.DataFrame()}
    
    model_performance = {}
    
    try:
        for model in df_recommendations['model_used'].unique():
            model_recs = df_recommendations[df_recommendations['model_used'] == model]
            model_rec_count = model_recs['recommendations'].apply(len).sum()
            
            model_clicks = 0
            try:
                for _, rec_row in model_recs.iterrows():
                    user_id = rec_row['user_id']
                    rec_timestamp = pd.to_datetime(rec_row['timestamp'], errors='coerce')
                    recommendations = rec_row.get('recommendations', [])
                    
                    if pd.isna(rec_timestamp) or not recommendations:
                        continue
                    
                    normalized_recs = [normalize_movie_name(m) for m in recommendations]
                    
                    df_interactions_copy = df_interactions.copy()
                    df_interactions_copy['timestamp'] = pd.to_datetime(df_interactions_copy['timestamp'], errors='coerce')
                    df_interactions_copy['normalized_movie'] = df_interactions_copy['movie_id'].apply(normalize_movie_name)
                    
                    user_clicks = df_interactions_copy[
                        (df_interactions_copy['user_id'] == user_id) & 
                        (df_interactions_copy['timestamp'] > rec_timestamp) &
                        (df_interactions_copy['interaction_type'] == 'click') &
                        (df_interactions_copy['normalized_movie'].isin(normalized_recs)) &
                        (~df_interactions_copy['timestamp'].isna())
                    ]
                    model_clicks += len(user_clicks)
            except Exception as e:
                print(f"⚠️ Warning: Error calculating model clicks: {e}")
            
            model_ctr = (model_clicks / model_rec_count * 100) if model_rec_count > 0 else 0
            
            model_performance[model] = {
                'ctr': model_ctr,
                'avg_response_time': model_recs['response_time_ms'].mean(),
                'request_count': len(model_recs)
            }
    except Exception as e:
        print(f"⚠️ Warning: Error in model performance analysis: {e}")
    
    hourly_performance = pd.DataFrame()
    try:
        df_recommendations_copy = df_recommendations.copy()
        df_recommendations_copy['timestamp'] = pd.to_datetime(df_recommendations_copy['timestamp'], errors='coerce')
        df_recommendations_copy = df_recommendations_copy.dropna(subset=['timestamp'])
        
        if not df_recommendations_copy.empty:
            df_recommendations_copy['hour'] = df_recommendations_copy['timestamp'].dt.hour
            hourly_performance = df_recommendations_copy.groupby('hour').agg({
                'response_time_ms': 'mean',
                'user_id': 'count'
            }).reset_index()
    except Exception as e:
        print(f"⚠️ Warning: Error in time-based analysis: {e}")
    
    return {
        'model_performance': model_performance,
        'hourly_performance': hourly_performance,
        'total_recommendation_requests': len(df_recommendations)
    }

# ============================================================================
# Section 4: Visualization and Reporting
# ============================================================================

def create_performance_dashboard(df_recommendations, df_interactions):
    """Create performance dashboard from real telemetry data"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Real-Time Online Evaluation Dashboard - Course Kafka Data', fontsize=16, fontweight='bold')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.99, 0.01, f'Generated: {timestamp} | Data: Course Kafka (128.2.220.241)', 
             ha='right', va='bottom', fontsize=8, style='italic')
    
    # 1. Click-Through Rate
    ctr_data = calculate_click_through_rate(df_recommendations, df_interactions)
    axes[0, 0].bar(['CTR'], [ctr_data['ctr_percentage']], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Click-Through Rate (%)')
    axes[0, 0].set_ylabel('CTR %')
    axes[0, 0].set_ylim([0, max(ctr_data['ctr_percentage'] * 1.2, 1)])
    axes[0, 0].text(0, max(ctr_data['ctr_percentage']/2, 0.1), f"{ctr_data['ctr_percentage']:.2f}%", 
                    ha='center', va='center', fontweight='bold', fontsize=12)
    
    # 2. Response Time Distribution
    if not df_recommendations.empty and 'response_time_ms' in df_recommendations.columns:
        response_times = df_recommendations['response_time_ms'].dropna()
        if len(response_times) > 0:
            axes[0, 1].hist(response_times, bins=20, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('Response Time Distribution')
            axes[0, 1].set_xlabel('Response Time (ms)')
            axes[0, 1].set_ylabel('Frequency')
        else:
            axes[0, 1].text(0.5, 0.5, 'No response time data', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Response Time Distribution')
    else:
        axes[0, 1].text(0.5, 0.5, 'No response time data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Response Time Distribution')
    
    # 3. Model Usage Distribution
    if not df_recommendations.empty and 'model_used' in df_recommendations.columns:
        model_counts = df_recommendations['model_used'].value_counts()
        if len(model_counts) > 0:
            axes[0, 2].pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%', startangle=90)
            axes[0, 2].set_title('Model Usage Distribution')
        else:
            axes[0, 2].text(0.5, 0.5, 'No model data', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Model Usage Distribution')
    else:
        axes[0, 2].text(0.5, 0.5, 'No recommendation data', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Model Usage Distribution')
    
    # 4. Hourly Activity Pattern
    if not df_interactions.empty:
        try:
            df_interactions_copy = df_interactions.copy()
            df_interactions_copy['timestamp'] = pd.to_datetime(df_interactions_copy['timestamp'], errors='coerce')
            df_interactions_copy = df_interactions_copy.dropna(subset=['timestamp'])
            
            if not df_interactions_copy.empty:
                df_interactions_copy['hour'] = df_interactions_copy['timestamp'].dt.hour
                hourly_counts = df_interactions_copy.groupby('hour').size()
                axes[1, 0].bar(hourly_counts.index, hourly_counts.values, alpha=0.7, color='lightgreen')
                axes[1, 0].set_title('User Activity by Hour')
                axes[1, 0].set_xlabel('Hour of Day')
                axes[1, 0].set_ylabel('Number of Interactions')
            else:
                axes[1, 0].text(0.5, 0.5, 'No valid timestamps', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('User Activity by Hour')
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, 'Error parsing data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('User Activity by Hour')
    else:
        axes[1, 0].text(0.5, 0.5, 'No interaction data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('User Activity by Hour')
    
    # 5. Interaction Type Distribution
    if not df_interactions.empty and 'interaction_type' in df_interactions.columns:
        interaction_counts = df_interactions['interaction_type'].value_counts()
        if len(interaction_counts) > 0:
            axes[1, 1].bar(interaction_counts.index, interaction_counts.values, alpha=0.7, color='coral')
            axes[1, 1].set_title('Interaction Type Distribution')
            axes[1, 1].set_xlabel('Interaction Type')
            axes[1, 1].set_ylabel('Count')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            axes[1, 1].text(0.5, 0.5, 'No interaction types', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Interaction Type Distribution')
    else:
        axes[1, 1].text(0.5, 0.5, 'No interaction data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Interaction Type Distribution')
    
    # 6. Precision@K
    precision_data = calculate_precision_at_k(df_recommendations, df_interactions)
    axes[1, 2].bar(['Precision@10'], [precision_data['precision_at_k']], color='gold', alpha=0.7)
    axes[1, 2].set_title('Precision@10')
    axes[1, 2].set_ylabel('Precision')
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].text(0, max(precision_data['precision_at_k']/2, 0.05), f"{precision_data['precision_at_k']:.3f}", 
                    ha='center', va='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(REPORTS_DIR, "online_evaluation_dashboard.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Dashboard saved to: {output_path}")

def generate_evaluation_report(df_recommendations, df_interactions, model_quality_metrics, output_file=None):
    """Generate comprehensive evaluation report from real data"""
    if output_file is None:
        output_file = os.path.join(REPORTS_DIR, "online_evaluation_report.txt")
    
    ctr_metrics = calculate_click_through_rate(df_recommendations, df_interactions)
    precision_metrics = calculate_precision_at_k(df_recommendations, df_interactions)
    response_metrics = calculate_response_time_metrics(df_recommendations)
    model_metrics = calculate_model_usage_metrics(df_recommendations)
    engagement_analysis = analyze_user_engagement_patterns(df_interactions)
    
    report_lines = [
        "=" * 80,
        "ONLINE EVALUATION REPORT - MOVIE RECOMMENDATION SYSTEM",
        "=" * 80,
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATA SUMMARY:",
        f"  * Recommendation Requests: {len(df_recommendations)}",
        f"  * User Interactions: {len(df_interactions)}",
        f"  * Unique Users (Interactions): {engagement_analysis.get('total_unique_users', 0)}",
    ]
    
    if not df_recommendations.empty:
        try:
            df_recs_copy = df_recommendations.copy()
            df_recs_copy['timestamp'] = pd.to_datetime(df_recs_copy['timestamp'], errors='coerce')
            df_recs_copy = df_recs_copy.dropna(subset=['timestamp'])
            if not df_recs_copy.empty:
                min_time = df_recs_copy['timestamp'].min()
                max_time = df_recs_copy['timestamp'].max()
                report_lines.append(f"  * Date Range: {min_time} to {max_time}")
        except:
            pass
    
    report_lines.extend([
        "",
        "ACCURACY METRICS:",
        f"  * Click-Through Rate: {ctr_metrics['ctr_percentage']:.2f}%",
        f"  * Precision@10: {precision_metrics['precision_at_k']:.3f}",
        f"  * Total Recommendations Served: {ctr_metrics['total_recommendations']}",
        f"  * Total User Clicks (on recommended items): {ctr_metrics['total_clicks']}",
        "",
        "PERFORMANCE METRICS:",
        f"  * Average Response Time: {response_metrics['avg_response_time']:.2f}ms",
        f"  * Median Response Time: {response_metrics['median_response_time']:.2f}ms",
        f"  * P95 Response Time: {response_metrics['p95_response_time']:.2f}ms",
        f"  * P99 Response Time: {response_metrics['p99_response_time']:.2f}ms",
        "",
        "MODEL USAGE:",
    ])
    
    if model_metrics['model_distribution']:
        for model, count in model_metrics['model_distribution'].items():
            percentage = (count / model_metrics['total_requests'] * 100) if model_metrics['total_requests'] > 0 else 0
            report_lines.append(f"  * {model}: {count} requests ({percentage:.1f}%)")
        report_lines.append(f"  * Fallback Rate: {model_metrics['fallback_rate']:.2%}")
    else:
        report_lines.append("  * No recommendation data available")
    
    report_lines.extend([
        "",
        "USER ENGAGEMENT:",
        f"  * Total Unique Users: {engagement_analysis.get('total_unique_users', 0)}",
        f"  * Total Unique Movies: {engagement_analysis.get('total_unique_movies', 0)}",
        "",
        "=" * 80,
        "MODEL QUALITY ASSESSMENT (Attribution-Free Metrics)",
        "=" * 80,
    ])
    
    # Add model quality metrics
    if 'error' not in model_quality_metrics:
        if 'recommendation_diversity' in model_quality_metrics:
            div = model_quality_metrics['recommendation_diversity']
            report_lines.extend([
                "",
                "DIVERSITY:",
                f"  * Unique movies recommended: {div['unique_movies_recommended']}",
                f"  * Total recommendation slots: {div['total_recommendation_slots']}",
                f"  * Diversity ratio: {div['diversity_ratio']:.3f}",
                f"  * Note: {div['interpretation']}",
            ])
        
        if 'catalog_coverage' in model_quality_metrics:
            cov = model_quality_metrics['catalog_coverage']
            report_lines.extend([
                "",
                "CATALOG COVERAGE:",
                f"  * Movies clicked by users: {cov['movies_clicked_by_users']}",
                f"  * Movies in recommendations: {cov['movies_in_recommendations']}",
                f"  * Overlap: {cov['overlap']} movies",
                f"  * Coverage: {cov['coverage_percentage']:.1f}%",
                f"  * Note: {cov['interpretation']}",
            ])
        
        if 'popularity_bias' in model_quality_metrics:
            pop = model_quality_metrics['popularity_bias']
            report_lines.extend([
                "",
                "POPULARITY BIAS:",
                f"  * Unique movies: {pop['unique_movies']}",
                f"  * Top-10 concentration: {pop['top_10_concentration']:.1f}%",
                f"  * Most recommended: {pop['most_recommended_movie']}",
                f"  * Times recommended: {pop['most_recommended_count']}",
                f"  * Note: {pop['interpretation']}",
            ])
        
        if 'personalization' in model_quality_metrics:
            pers = model_quality_metrics['personalization']
            report_lines.extend([
                "",
                "PERSONALIZATION:",
                f"  * Users evaluated: {pers['users_with_recommendations']}",
                f"  * Avg user similarity: {pers['avg_user_similarity']:.3f}",
                f"  * Personalization score: {pers['personalization_score']:.3f}",
                f"  * Note: {pers['interpretation']}",
            ])
        
        if 'temporal_consistency' in model_quality_metrics:
            cons = model_quality_metrics['temporal_consistency']
            report_lines.extend([
                "",
                "TEMPORAL CONSISTENCY:",
                f"  * Early period movies: {cons['first_half_unique_movies']}",
                f"  * Late period movies: {cons['second_half_unique_movies']}",
                f"  * Consistency score: {cons['consistency_score']:.3f}",
                f"  * Note: {cons['interpretation']}",
            ])
    
    report_lines.extend([
        "",
        "=" * 80,
        "RECOMMENDATIONS:",
    ])
    
    if response_metrics['avg_response_time'] > 0:
        if response_metrics['avg_response_time'] < 200:
            report_lines.append("  - Response times are within acceptable range")
        else:
            report_lines.append("  - Consider optimizing response times")
    
    if model_metrics['fallback_rate'] > 0.2:
        report_lines.append("  - High fallback rate - investigate model issues")
    elif model_metrics['total_requests'] > 0:
        report_lines.append("  - Model fallback rate is acceptable")
    
    if ctr_metrics['ctr_percentage'] == 0 and precision_metrics['precision_at_k'] == 0:
        report_lines.extend([
            "",
            "⚠️  DATA QUALITY NOTE:",
            "  - CTR and Precision metrics are 0, indicating:",
            "    • Recommendations and clicks may be from different user sessions",
            "    • Users may not be clicking recommended items within the 30-min window",
            "    • Possible cold-start scenario with default recommendations",
            "  - Model quality metrics provide alternative assessment methods",
        ])
    
    report_lines.extend([
        "",
        "=" * 80,
    ])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📁 Report saved to: {output_file}")

# ============================================================================
# Section 5: Main Analysis Pipeline
# ============================================================================

def run_telemetry_analysis(generate_visualizations=True):
    """Run complete ONLINE evaluation using ONLY real COURSE Kafka telemetry data"""
    
    print("🚀 Starting TRUE Online Evaluation with Real COURSE Kafka Data")
    print("📡 Using SSH tunnel to course broker: 128.2.220.241")
    print("=" * 60)
    
    try:
        df_recommendations, df_interactions, df_all = load_real_kafka_data()
        data_source = "course_kafka"
        
        print("✅ Using REAL course telemetry data from official Kafka broker")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print("❌ Online evaluation requires real course telemetry data from Kafka")
        print("💡 Ensure SSH tunnel is active: ssh -L 9092:localhost:9092 tunnel@128.2.220.241 -NT")
        return False
    
    # Calculate metrics from real data
    print("\n📈 Calculating Online Evaluation Metrics from Real Data...")
    ctr_metrics = calculate_click_through_rate(df_recommendations, df_interactions)
    precision_metrics = calculate_precision_at_k(df_recommendations, df_interactions, k=10)
    response_metrics = calculate_response_time_metrics(df_recommendations)
    model_metrics = calculate_model_usage_metrics(df_recommendations)
    
    # Print results
    print("\n📊 REAL ONLINE EVALUATION RESULTS:")
    print("=" * 50)
    print(f"Click-Through Rate: {ctr_metrics['ctr_percentage']:.2f}%")
    print(f"Total Recommendations: {ctr_metrics['total_recommendations']}")
    print(f"Total Clicks (on recommended): {ctr_metrics['total_clicks']}")
    print(f"Precision@10: {precision_metrics['precision_at_k']:.3f}")
    print(f"Average Response Time: {response_metrics['avg_response_time']:.2f}ms")
    print(f"P95 Response Time: {response_metrics['p95_response_time']:.2f}ms")
    print(f"Model Fallback Rate: {model_metrics['fallback_rate']:.2%}")
    print("=" * 50)
    
    # Calculate and display model quality metrics
    print("\n🔍 Calculating Model Quality Metrics...")
    model_quality = calculate_model_quality_metrics(df_recommendations, df_interactions)
    print_model_quality_report(model_quality)
    
    # Analyze real user patterns
    print("\n🔍 Analyzing Real User Interaction Patterns...")
    engagement_analysis = analyze_user_engagement_patterns(df_interactions)
    effectiveness_analysis = analyze_recommendation_effectiveness(df_recommendations, df_interactions)
    
    print(f"Total Unique Users: {engagement_analysis.get('total_unique_users', 0)}")
    print(f"Total Unique Movies: {engagement_analysis.get('total_unique_movies', 0)}")
    
    # Print real model effectiveness
    print("\n⚡ REAL MODEL EFFECTIVENESS:")
    model_perf = effectiveness_analysis.get('model_performance', {})
    if model_perf:
        for model, metrics in model_perf.items():
            print(f"{model}: CTR={metrics['ctr']:.2f}%, Avg Response={metrics['avg_response_time']:.1f}ms, Requests={metrics['request_count']}")
    else:
        print("No recommendation data available for model effectiveness analysis")
    
    # Generate visualizations from real data
    if generate_visualizations:
        print("\n📈 Creating Performance Dashboard from Real Data...")
        try:
            create_performance_dashboard(df_recommendations, df_interactions)
        except Exception as e:
            print(f"⚠️ Warning: Could not create dashboard: {e}")
        
        print("\n📋 Generating Real Online Evaluation Report...")
        try:
            generate_evaluation_report(df_recommendations, df_interactions, model_quality)
        except Exception as e:
            print(f"⚠️ Warning: Could not generate report: {e}")
    
    print("\n✅ TRUE Online Evaluation completed successfully!")
    print("📊 Results based on REAL course telemetry data from official Kafka broker")
    print("📡 Data source: SSH tunnel to 128.2.220.241")
    return True

def main():
    """Main function - TRUE Online Evaluation with Real COURSE Kafka Data Only"""
    print("🚀 Online Evaluation System - Real COURSE Kafka Data Mode")
    print("📡 Connecting to course broker via SSH tunnel")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='Online Evaluation with Real COURSE Kafka Telemetry')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Check Kafka availability
    if not KAFKA_AVAILABLE:
        print("❌ ERROR: kafka-python not installed!")
        print("💡 Install with: pip install kafka-python")
        return False
    
    # Run TRUE online evaluation with COURSE Kafka data only
    print("🔄 Starting online evaluation with REAL COURSE Kafka telemetry data...")
    success = run_telemetry_analysis(generate_visualizations=not args.no_viz)
    
    if success:
        print("\n✅ TRUE Online Evaluation completed successfully!")
        print("📊 Dashboard and report files generated from REAL course telemetry data")
        print("📁 Files created: online_evaluation_dashboard.png, online_evaluation_report.txt")
    else:
        print("\n❌ Online evaluation failed - no real course telemetry data available")
        print("💡 To connect to course Kafka broker:")
        print("   1. Start SSH tunnel: ssh -L 9092:localhost:9092 tunnel@128.2.220.241 -NT")
        print("   2. Run this script again")
    
    return success

if __name__ == "__main__":
    main()