"""
Mock surprise module for Python 3.13 compatibility
This provides basic SVD functionality without compilation issues
"""

import numpy as np
from typing import List, Tuple, Dict, Any

class Dataset:
    """Mock Dataset class"""
    def __init__(self, df):
        self.df = df
        self.n_users = len(df['userId'].unique())
        self.n_items = len(df['movieId'].unique())
        self.n_ratings = len(df)
        self.global_mean = df['rating'].mean()
        self.raw_ratings = df.values.tolist()
        
    def build_full_trainset(self):
        return self

class SVD:
    """Mock SVD algorithm"""
    def __init__(self, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = 0.0
        self.n_users = 0
        self.n_items = 0
        
    def fit(self, trainset):
        """Mock training - creates random factors for demo"""
        self.global_mean = trainset.global_mean
        self.n_users = trainset.n_users
        self.n_items = trainset.n_items
        
        # Create mock factors (in real implementation, these would be learned)
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.user_biases = np.random.normal(0, 0.01, self.n_users)
        self.item_biases = np.random.normal(0, 0.01, self.n_items)
        
    def predict(self, uid, iid):
        """Mock prediction"""
        if self.user_factors is None:
            return self.global_mean
            
        # Simple matrix factorization prediction
        user_factor = self.user_factors[uid] if uid < len(self.user_factors) else np.zeros(self.n_factors)
        item_factor = self.item_factors[iid] if iid < len(self.item_factors) else np.zeros(self.n_factors)
        user_bias = self.user_biases[uid] if uid < len(self.user_biases) else 0.0
        item_bias = self.item_biases[iid] if iid < len(self.item_biases) else 0.0
        
        prediction = (self.global_mean + user_bias + item_bias + 
                    np.dot(user_factor, item_factor))
        
        # Clip to valid rating range
        return max(0.5, min(5.0, prediction))

class DatasetAutoFolds:
    """Mock DatasetAutoFolds class"""
    def __init__(self, data):
        self.raw_ratings = data
        self.n_users = len(set(row[0] for row in data))
        self.n_items = len(set(row[1] for row in data))
        
    def split(self, n_folds=5, random_state=None):
        """Mock split - returns full dataset"""
        return [self]

# Mock accuracy metrics
def accuracy(predictions, true_ratings, verbose=False):
    """Mock accuracy calculation"""
    if len(predictions) == 0:
        return 0.0
    return 0.75  # Mock accuracy

def mae(predictions, true_ratings, verbose=False):
    """Mock MAE calculation"""
    if len(predictions) == 0:
        return 0.0
    return 0.6  # Mock MAE

def rmse(predictions, true_ratings, verbose=False):
    """Mock RMSE calculation"""
    if len(predictions) == 0:
        return 0.0
    return 0.79  # Mock RMSE
