# In the Jupyter Notebook - Hyperparameter Tuning

import mlflow
import mlflow.sklearn
from src import train, evaluate
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Hyperparameter experiments for Logistic Regression with TF-IDF
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'max_features': [3000, 5000, 7000]
}

experiments = []
best_f1 = 0
best_config = None

for C in param_grid['C']:
    for max_features in param_grid['max_features']:
        # Update config
        config['model_1']['params']['C'] = C
        config['max_features'] = max_features
        
        # Run experiment
        with mlflow.start_run(run_name=f"LR_C{C}_MF{max_features}"):
            # Log parameters
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("C", C)
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("random_seed", config['random_seed'])
            
            # Train and evaluate
            # (This would be a function call to your training pipeline)
            # For demonstration, we'll just log metrics
            f1_score = 0.75 + np.random.normal(0, 0.05)  # Placeholder
            accuracy = 0.78 + np.random.normal(0, 0.04)
            
            mlflow.log_metric("f1_weighted", f1_score)
            mlflow.log_metric("accuracy", accuracy)
            
            experiments.append({
                'C': C,
                'max_features': max_features,
                'f1_weighted': f1_score,
                'accuracy': accuracy
            })
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_config = {'C': C, 'max_features': max_features}

# Results table
print("\nHyperparameter Experiment Results:")
print("="*60)
print(f"{'C':<10} {'Max Features':<15} {'F1 Weighted':<15} {'Accuracy':<10}")
print("-"*60)
for exp in experiments:
    print(f"{exp['C']:<10} {exp['max_features']:<15} {exp['f1_weighted']:.4f}     {exp['accuracy']:.4f}")

print("\n" + "="*60)
print(f"Best Configuration: C={best_config['C']}, max_features={best_config['max_features']}")
print(f"Best F1 Score: {best_f1:.4f}")