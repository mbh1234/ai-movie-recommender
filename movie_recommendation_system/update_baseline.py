"""
Script to update baseline data for drift detection
"""

import pandas as pd
from datetime import datetime
import os

def update_baseline():
    # Load current interaction data
    data_path = 'group-project-f25-pulp-predictions/data/interactions_table.csv'
    interactions_df = pd.read_csv(data_path)
    
    # Create baseline directory if it doesn't exist
    os.makedirs('baseline_data', exist_ok=True)
    
    # Save current data as new baseline
    timestamp = datetime.now().strftime("%Y%m%d")
    baseline_path = f'baseline_data/baseline_interactions_{timestamp}.csv'
    interactions_df.to_csv(baseline_path, index=False)
    
    # Also update the default baseline path
    interactions_df.to_csv('baseline_data/baseline_interactions.csv', index=False)
    
    print(f"Baseline updated successfully: {baseline_path}")

if __name__ == "__main__":
    update_baseline()
