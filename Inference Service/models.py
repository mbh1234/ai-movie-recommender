import pandas as pd
import numpy as np
import sys
import time

class MemoryEfficientSVD:
    """Memory-efficient SVD for collaborative filtering"""

    def __init__(self, n_components=25, learning_rate=0.02, regularization=0.1,
                 n_epochs=5, batch_size=5000, random_state=42):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

        # Model parameters
        self.global_mean = 0
        self.user_bias = {}
        self.item_bias = {}
        self.user_features = {}
        self.item_features = {}
        self.training_time = 0
        self.model_size_mb = 0

    def initialize_user_item(self, user_id, item_id):
        """Initialize user and item if not seen before"""
        if user_id not in self.user_features:
            self.user_features[user_id] = np.random.normal(0, 0.1, self.n_components)
            self.user_bias[user_id] = 0.0

        if item_id not in self.item_features:
            self.item_features[item_id] = np.random.normal(0, 0.1, self.n_components)
            self.item_bias[item_id] = 0.0

    def predict_rating(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if user_id not in self.user_features or item_id not in self.item_features:
            return self.global_mean

        prediction = (self.global_mean +
                     self.user_bias[user_id] +
                     self.item_bias[item_id] +
                     np.dot(self.user_features[user_id], self.item_features[item_id]))

        return np.clip(prediction, 1.0, 5.0)

    def train_batch(self, batch_data):
        """Train on a batch of interactions"""
        for _, row in batch_data.iterrows():
            user_id = row['user_id']
            item_id = row['movie_id']
            rating = row['final_rating']

            self.initialize_user_item(user_id, item_id)

            # Predict and calculate error
            prediction = self.predict_rating(user_id, item_id)
            error = rating - prediction

            # Update biases
            user_bias_old = self.user_bias[user_id]
            item_bias_old = self.item_bias[item_id]

            self.user_bias[user_id] += self.learning_rate * (error - self.regularization * user_bias_old)
            self.item_bias[item_id] += self.learning_rate * (error - self.regularization * item_bias_old)

            # Update features
            user_features_old = self.user_features[user_id].copy()
            item_features_old = self.item_features[item_id].copy()

            self.user_features[user_id] += self.learning_rate * (error * item_features_old - self.regularization * user_features_old)
            self.item_features[item_id] += self.learning_rate * (error * user_features_old - self.regularization * item_features_old)

    def train(self, interactions_df, sample_size=200000):
        """Train the model"""
        print("Training Memory-Efficient SVD...")
        start_time = time.time()

        # Sample data if too large
        if sample_size and len(interactions_df) > sample_size:
            interactions_df = interactions_df.sample(sample_size, random_state=self.random_state)
            print(f"Using sample of {len(interactions_df):,} interactions")

        self.global_mean = interactions_df['final_rating'].mean()

        # Training loop
        for epoch in range(self.n_epochs):
            shuffled_data = interactions_df.sample(frac=1, random_state=epoch).reset_index(drop=True)
            total_batches = len(shuffled_data) // self.batch_size + 1

            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(shuffled_data))

                if start_idx >= len(shuffled_data):
                    break

                batch = shuffled_data.iloc[start_idx:end_idx]
                self.train_batch(batch)

        self.training_time = time.time() - start_time

        # Calculate model size
        self.model_size_mb = (
            sys.getsizeof(self.user_features) +
            sys.getsizeof(self.item_features) +
            sys.getsizeof(self.user_bias) +
            sys.getsizeof(self.item_bias)
        ) / (1024 * 1024)

        print(f"Training completed in {self.training_time:.1f} seconds")
        print(f"Model size: {self.model_size_mb:.1f} MB")

    def get_recommendations(self, user_id, n_recommendations=20, exclude_seen=None):
        """Get recommendations for a user"""
        if user_id not in self.user_features:
            # Return popular items for unknown users
            popular_items = sorted(self.item_bias.items(), key=lambda x: x[1], reverse=True)
            return [item_id for item_id, _ in popular_items[:n_recommendations]]

        # Score all items
        item_scores = []
        for item_id in self.item_features.keys():
            if exclude_seen and item_id in exclude_seen:
                continue

            score = self.predict_rating(user_id, item_id)
            item_scores.append((item_id, score))

        # Sort and return top N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in item_scores[:n_recommendations]]


class PopularityBaseline:
    """Simple popularity-based recommendation model"""

    def __init__(self):
        self.global_mean = 0
        self.item_stats = None
        self.training_time = 0
        self.model_size_mb = 0

    def train(self, interactions_df):
        """Train the popularity model"""
        print("Training Popularity Baseline...")
        start_time = time.time()

        self.global_mean = interactions_df['final_rating'].mean()

        # Calculate item statistics
        self.item_stats = interactions_df.groupby('movie_id').agg({
            'final_rating': ['mean', 'count', 'std']
        }).round(3)

        self.item_stats.columns = ['avg_rating', 'count', 'std_rating']
        self.item_stats = self.item_stats.reset_index()

        # Calculate popularity score
        self.item_stats['popularity_score'] = (
            self.item_stats['avg_rating'] * 0.7 +
            np.log1p(self.item_stats['count']) * 0.3
        )

        self.training_time = time.time() - start_time
        self.model_size_mb = sys.getsizeof(self.item_stats) / (1024 * 1024)

        print(f"Training completed in {self.training_time:.3f} seconds")
        print(f"Model size: {self.model_size_mb:.3f} MB")

    def predict_rating(self, user_id, item_id):
        """Predict rating for user-item pair"""
        item_data = self.item_stats[self.item_stats['movie_id'] == item_id]

        if len(item_data) > 0 and item_data.iloc[0]['count'] >= 5:
            predicted = item_data.iloc[0]['avg_rating']
        else:
            predicted = self.global_mean

        return np.clip(predicted, 1.0, 5.0)

    def get_recommendations(self, user_id, n_recommendations=20, exclude_seen=None):
        """Get recommendations for a user"""
        # Sort by popularity score
        popular_items = self.item_stats.nlargest(n_recommendations * 2, 'popularity_score')

        recommendations = []
        for _, row in popular_items.iterrows():
            item_id = row['movie_id']
            if exclude_seen and item_id in exclude_seen:
                continue

            recommendations.append(item_id)
            if len(recommendations) >= n_recommendations:
                break

        return recommendations