import pandas as pd
import json
from datetime import datetime

def calculate_baseline_statistics(df):
    """Calculate baseline statistics from the interactions data"""
    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_interactions': len(df),
        'unique_users': df['user_id'].nunique(),
        'unique_movies': df['movie_id'].nunique(),
        
        # Watch time statistics
        'watch_time_stats': {
            'mean': df['total_minutes'].mean(),
            'median': df['total_minutes'].median(),
            'std': df['total_minutes'].std(),
            'percentiles': {
                '25': df['total_minutes'].quantile(0.25),
                '75': df['total_minutes'].quantile(0.75),
                '90': df['total_minutes'].quantile(0.90)
            }
        },
        
        # Rating statistics (excluding null values)
        'rating_stats': {
            'rating_coverage': df['rating'].notna().mean(),
            'mean_rating': df['rating'].mean(),
            'rating_distribution': df['rating'].value_counts(normalize=True).to_dict()
        },
        
        # Engagement metrics
        'engagement_metrics': {
            'avg_watch_count_per_user': float(df.groupby('user_id')['watch_count'].mean().mean()),
            'avg_watch_time_per_user': float(df.groupby('user_id')['total_minutes'].mean().mean()),
            'movie_popularity': df.groupby('movie_id')['watch_count'].sum().describe().to_dict()
        }
    }
    return stats

def main():
    # Load baseline data
    print("Loading baseline data...")
    df = pd.read_csv('baseline_data/baseline_interactions.csv')
    
    # Calculate statistics
    print("Calculating baseline statistics...")
    baseline_stats = calculate_baseline_statistics(df)
    
    # Save baseline statistics
    output_file = 'baseline_data/baseline_statistics.json'
    print(f"Saving baseline statistics to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(baseline_stats, f, indent=4)
    
    print("Baseline statistics created successfully!")

if __name__ == "__main__":
    main()
