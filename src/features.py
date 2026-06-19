"""
Feature extraction module for text classification.
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import json

def extract_features(train_texts, test_texts, vectorizer_type='tfidf', max_features=5000, sublinear_tf=True):
    """
    Extract features using Bag-of-Words or TF-IDF.
    
    Args:
        train_texts: Training text data
        test_texts: Test text data
        vectorizer_type: 'bow' or 'tfidf'
        max_features: Maximum number of features
        sublinear_tf: Apply sublinear tf scaling (TF-IDF only)
        
    Returns:
        tuple: (train_features, test_features, vectorizer)
    """
    if vectorizer_type == 'bow':
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=2,  # Ignore terms appearing in less than 2 documents
            max_df=0.95  # Ignore terms appearing in more than 95% of documents
        )
    else:  # tfidf
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            min_df=2,
            max_df=0.95
        )
    
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)
    
    return train_features, test_features, vectorizer

def get_top_terms(vectorizer, class_texts, class_names, n_terms=20):
    """
    Get top n terms for each class.
    
    Args:
        vectorizer: Fitted vectorizer
        class_texts: List of texts for each class
        class_names: List of class names
        n_terms: Number of top terms to return
        
    Returns:
        dict: Top terms for each class
    """
    top_terms = {}
    
    for class_name, texts in zip(class_names, class_texts):
        if not texts:
            continue
            
        # Transform class texts
        class_features = vectorizer.transform(texts)
        feature_weights = class_features.sum(axis=0).A1
        
        # Get top terms
        feature_names = vectorizer.get_feature_names_out()
        top_indices = feature_weights.argsort()[-n_terms:][::-1]
        
        top_terms[class_name] = [(feature_names[i], feature_weights[i]) for i in top_indices]
    
    return top_terms

def print_top_terms(top_terms):
    """
    Print top terms for each class in a readable format.
    """
    print("\nTop Terms by Class:")
    print("=" * 50)
    for class_name, terms in top_terms.items():
        print(f"\nClass: {class_name}")
        print("-" * 30)
        for term, weight in terms:
            print(f"  {term}: {weight:.4f}")