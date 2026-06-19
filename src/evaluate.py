"""
Evaluation module for classification models.
"""
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

def evaluate_model(model, X_test, y_test, model_name, feature_type, class_names):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        feature_type: 'bow' or 'tfidf'
        class_names: List of class names
        
    Returns:
        dict: Evaluation metrics
    """
    # Predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results = {
        'model_name': model_name,
        'feature_type': feature_type,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'inference_time': inference_time,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'class_names': class_names
    }
    
    return results

def plot_confusion_matrix(cm, class_names, model_name, feature_type, save_path=None):
    """
    Plot confusion matrix as heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name} ({feature_type})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_evaluation_results(results):
    """
    Print evaluation results in a readable format.
    """
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS: {results['model_name']} with {results['feature_type']}")
    print("="*60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision (macro): {results['precision_macro']:.4f}")
    print(f"Recall (macro): {results['recall_macro']:.4f}")
    print(f"F1-Score (macro): {results['f1_macro']:.4f}")
    print(f"F1-Score (weighted): {results['f1_weighted']:.4f}")
    print(f"Inference Time: {results['inference_time']:.4f} seconds")
    print("-"*60)

def compare_results(all_results):
    """
    Compare all models and create a summary table.
    """
    summary = []
    for result in all_results:
        summary.append({
            'Model': result['model_name'],
            'Features': result['feature_type'],
            'Accuracy': f"{result['accuracy']:.4f}",
            'F1 (Macro)': f"{result['f1_macro']:.4f}",
            'F1 (Weighted)': f"{result['f1_weighted']:.4f}",
            'Inference (s)': f"{result['inference_time']:.4f}"
        })
    
    df_summary = pd.DataFrame(summary)
    print("\n" + "="*70)
    print("COMPARATIVE RESULTS SUMMARY")
    print("="*70)
    print(df_summary.to_string(index=False))
    
    return df_summary