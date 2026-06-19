import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def run_experiment(X, y, model, model_name, feat_name, config, save_dir="../results"):
    """Train a model, evaluate it on a held-out test set, and save a confusion matrix.

    Args:
        X: Feature matrix (BoW or TF-IDF).
        y (pd.Series): Class labels.
        model: An unfitted sklearn estimator.
        model_name (str): Identifier used in filenames/logging (e.g. 'LogReg').
        feat_name (str): Feature representation name (e.g. 'TFIDF').
        config (dict): Loaded config.json, used for seed and test_size.
        save_dir (str): Where to save the confusion matrix image.

    Returns:
        dict: Metrics including accuracy, F1 scores, and timing.
    """
    # Stratified split keeps class proportions equal in train/test —
    # important here since we have exactly balanced classes and want to preserve that
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=config['random_seed'],  # seed comes from config, not hardcoded
        stratify=y
    )

    # Time the training step
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Time the inference step separately, since exam asks for both
    t0 = time.time()
    preds = model.predict(X_test)
    infer_time = time.time() - t0

    # Compute accuracy
    acc = accuracy_score(y_test, preds)

    # Compute macro and weighted precision/recall/F1
    # macro = unweighted mean across classes (good for balanced importance)
    # weighted = accounts for class support (useful if classes were imbalanced)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro')
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, preds, average='weighted')

    # Build and save confusion matrix heatmap
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} + {feat_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cm_{model_name}_{feat_name}.png")
    plt.close()  # close figure so it doesn't pile up in memory across runs

    return {
        "model": model_name,
        "features": feat_name,
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "train_time": train_time,
        "infer_time": infer_time
    }