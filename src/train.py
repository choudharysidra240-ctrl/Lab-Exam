"""
Model training module for legal document classification.
"""
import json
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from src import preprocess, features

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def train_models(X_train, y_train, X_test, y_test, config):
    """
    Train both models with given feature sets.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Configuration dictionary
        
    Returns:
        dict: Trained models and metadata
    """
    results = {}
    
    # Model 1: Logistic Regression (from config)
    print("\nTraining Logistic Regression...")
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
    
    # Model 2: Naïve Bayes (from config)
    print("Training Naïve Bayes...")
    nb_params = config['model_2']['params']
    nb_model = MultinomialNB(
        alpha=nb_params['alpha']
    )
    nb_start = time.time()
    nb_model.fit(X_train, y_train)
    nb_train_time = time.time() - nb_start
    
    # Store results
    results['logistic_regression'] = {
        'model': lr_model,
        'train_time': lr_train_time,
        'params': lr_params
    }
    results['naive_bayes'] = {
        'model': nb_model,
        'train_time': nb_train_time,
        'params': nb_params
    }
    
    return results

def main():
    """Main training pipeline."""
    config = load_config()
    
    # Load and preprocess data
    print("Loading data...")
    df = pd.read_csv('data/raw/legal_notices.csv')
    
    print("Preprocessing text...")
    df = preprocess.preprocess_dataframe(df, stem=True)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    class_names = le.classes_
    
    # Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['cleaned_text'],
        y,
        test_size=config['test_size'],
        random_state=config['random_seed'],
        stratify=y
    )
    
    print(f"Train size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")
    
    # Extract features - BOW
    print("\nExtracting BOW features...")
    X_train_bow, X_test_bow, bow_vectorizer = features.extract_features(
        X_train_raw, X_test_raw, 
        vectorizer_type='bow',
        max_features=config['max_features']
    )
    
    # Extract features - TF-IDF
    print("Extracting TF-IDF features...")
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = features.extract_features(
        X_train_raw, X_test_raw,
        vectorizer_type='tfidf',
        max_features=config['max_features'],
        sublinear_tf=True
    )
    
    # Train models with BOW features
    print("\n=== Training with BOW Features ===")
    bow_results = train_models(X_train_bow, y_train, X_test_bow, y_test, config)
    
    # Train models with TF-IDF features
    print("\n=== Training with TF-IDF Features ===")
    tfidf_results = train_models(X_train_tfidf, y_train, X_test_tfidf, y_test, config)
    
    return {
        'bow_results': bow_results,
        'tfidf_results': tfidf_results,
        'y_test': y_test,
        'class_names': class_names,
        'X_test_bow': X_test_bow,
        'X_test_tfidf': X_test_tfidf,
        'bow_vectorizer': bow_vectorizer,
        'tfidf_vectorizer': tfidf_vectorizer,
        'le': le
    }

if __name__ == "__main__":
    main()