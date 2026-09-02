
import pandas as pd
import os

print("Current working directory:", os.getcwd())

# Load CSV with header row
interactions = pd.read_csv('../CSV Data/interactions_table.csv', header=0)
print('First 5 rows:')
print(interactions.head())
print('Columns:', interactions.columns)

# --- Feedback Loop Analysis ---
# Assumption: feedback loop is popularity reinforcement (popular movies get recommended more, watched more)

# Check if 'movie_id' or similar column exists
movie_id_col = None
for col in interactions.columns:
	if 'movie' in col.lower():
		movie_id_col = col
		break
if not movie_id_col:
	print("No movie_id column found. Columns:", interactions.columns)
else:
	# Compute watch counts per movie
	watch_counts = interactions[movie_id_col].value_counts()
	print(f"Top 10 most watched movies (by {movie_id_col}):")
	print(watch_counts.head(10))

	# Analyze distribution
	import matplotlib.pyplot as plt
	plt.figure(figsize=(8,4))
	plt.hist(watch_counts, bins=30)
	plt.xlabel('Watch Count')
	plt.ylabel('Number of Movies')
	plt.title('Distribution of Movie Watch Counts')
	plt.tight_layout()
	plt.savefig('watch_count_distribution.png')
	print("Histogram saved as watch_count_distribution.png")

	# Key findings summary
	print("\n--- Feedback Loop Analysis Summary ---")
	print("We analyzed the distribution of movie watch counts to check for popularity reinforcement.")
	print("If a small number of movies have very high watch counts, this suggests a feedback loop.")
	print("See 'watch_count_distribution.png' for the histogram.")
	print("Code: Feature Engineering and Model Training/debug_interactions.py")
