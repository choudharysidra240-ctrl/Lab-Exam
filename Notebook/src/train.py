
import json
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_config(config_path='config.json'):
    """Load configuration from config.json"""
    with open(config_path, 'r') as f:
        return json.load(f)

def train_models(X_train, y_train, config, feature_type='tfidf'):
    """
    Train both models with given feature sets.
    """
    results = {}
    
    # Model 1: Logistic Regression
    lr_params = config['model_1']['params']
    lr_model = LogisticRegression(
        C=lr_params['C'],
        max_iter=lr_params['max_iter'],
        random_state=config['random_seed'],
        class_weight='balanced'
    )
    lr_start = time.time()
    lr_model.fit(X_train, y_train)
    lr_train_time = time.time() - lr_start
    
    # Model 2: Naïve Bayes
    nb_params = config['model_2']['params']
    nb_model = MultinomialNB(alpha=nb_params['alpha'])
    nb_start = time.time()
    nb_model.fit(X_train, y_train)
    nb_train_time = time.time() - nb_start
    
    results['logistic_regression'] = {
        'model': lr_model,
        'train_time': lr_train_time,
        'params': lr_params,
        'feature_type': feature_type
    }
    results['naive_bayes'] = {
        'model': nb_model,
        'train_time': nb_train_time,
        'params': nb_params,
        'feature_type': feature_type
    }
    return results
