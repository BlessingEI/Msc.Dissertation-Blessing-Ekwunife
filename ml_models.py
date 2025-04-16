import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
from pathlib import Path
import warnings
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            roc_curve, auc, f1_score, 
                            precision_score, recall_score)
from scipy import stats
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['savefig.dpi'] = 300

# Setting the paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "Thesis", "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Define color palettes
BINARY_PALETTE = ['#3498db', '#e74c3c']
MULTI_PALETTE = ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(REPORTS_DIR, 'ml_models.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_processed_data():
    """Load the preprocessed data"""
    logging.info("=== Loading Preprocessed Data ===")
    
    # Load the processed data from .npz file
    data_path = os.path.join(PROCESSED_DIR, 'processed_data.npz')
    data = np.load(data_path, allow_pickle=True)
    
    # Extract the arrays
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    features = data['feature_names']
    
    # Load encoders for later use
    encoders_path = os.path.join(PROCESSED_DIR, 'encoders.pkl')
    encoders = joblib.load(encoders_path)
    
    logging.info(f"Loaded preprocessed data successfully")
    logging.info(f"Training set: {X_train.shape[0]} samples with {X_train.shape[1]} features")
    logging.info(f"Testing set: {X_test.shape[0]} samples with {X_test.shape[1]} features")
    
    # Print class distribution
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    test_classes, test_counts = np.unique(y_test, return_counts=True)
    
    logging.info("\nClass distribution in training set:")
    for cls, count in zip(train_classes, train_counts):
        logging.info(f"  Class {cls}: {count} samples ({count/len(y_train)*100:.2f}%)")
    
    logging.info("\nClass distribution in testing set:")
    for cls, count in zip(test_classes, test_counts):
        logging.info(f"  Class {cls}: {count} samples ({count/len(y_test)*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test, features.tolist(), encoders

def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier"""
    logging.info("\n=== Training Random Forest Classifier ===")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1, n_jobs=-1)
    logging.info(f"Random Forest parameters: {rf.get_params()}")
    logging.info("Fitting Random Forest model...")
    
    start_time = datetime.now()
    rf.fit(X_train, y_train)
    training_time = datetime.now() - start_time
    
    logging.info(f"Random Forest training completed in {training_time}")
    
    # Save the model
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    logging.info(f"Random Forest model saved to {os.path.join(MODELS_DIR, 'random_forest.pkl')}")
    
    return rf

def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier"""
    logging.info("\n=== Training XGBoost Classifier ===")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=2,
        n_jobs=-1
    )
    logging.info(f"XGBoost parameters: {xgb_model.get_params()}")
    logging.info("Fitting XGBoost model...")
    
    # Create an evaluation set
    eval_set = [(X_train, y_train)]
    
    start_time = datetime.now()
    xgb_model.fit(
        X_train, 
        y_train,
        eval_set=eval_set,
        verbose=True
    )
    training_time = datetime.now() - start_time
    
    logging.info(f"XGBoost training completed in {training_time}")
    
    # Save the model
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost.pkl"))
    logging.info(f"XGBoost model saved to {os.path.join(MODELS_DIR, 'xgboost.pkl')}")
    
    return xgb_model

def train_svm(X_train, y_train):
    """Train a Support Vector Machine classifier"""
    logging.info("\n=== Training SVM Classifier ===")
    
    svm = SVC(kernel='rbf', probability=True, random_state=42, verbose=True)
    logging.info(f"SVM parameters: {svm.get_params()}")
    logging.info("Fitting SVM model (this may take some time)...")
    
    start_time = datetime.now()
    svm.fit(X_train, y_train)
    training_time = datetime.now() - start_time
    
    logging.info(f"SVM training completed in {training_time}")
    
    # Save the model
    joblib.dump(svm, os.path.join(MODELS_DIR, "svm.pkl"))
    logging.info(f"SVM model saved to {os.path.join(MODELS_DIR, 'svm.pkl')}")
    
    return svm

def evaluate_model(model, X_test, y_test, model_name, features=None):
    """Evaluate a model and create visualizations"""
    logging.info(f"\n=== Evaluating {model_name} ===")
    
    # Make predictions
    logging.info(f"Making predictions with {model_name}...")
    y_pred = model.predict(X_test)
    
    # Get probabilities (if available)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logging.info(f"{model_name} Performance:")
    logging.info(f"  Accuracy:  {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall:    {recall:.4f}")
    logging.info(f"  F1 Score:  {f1:.4f}")
    
    # Save metrics to a CSV file
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1]
    })
    
    metrics_file = os.path.join(REPORTS_DIR, f"{model_name.replace(' ', '_').lower()}_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    logging.info(f"Metrics saved to {metrics_file}")
    
    logging.info("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    logging.info(report)
    
    # Create confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'])
    plt.title(f'Confusion Matrix - {model_name}', fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'))
    plt.close()
    
    # Feature importance (if available)
    if features is not None and hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        # Sort features by importance
        sorted_idx = feature_importance.argsort()[-15:]  # top 15 features
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
        plt.title(f'Feature Importance - {model_name}', fontweight='bold')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'feature_importance_{model_name.replace(" ", "_").lower()}.png'))
        plt.close()
    
    # Create ROC curve (if probabilities are available)
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}', fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'roc_curve_{model_name.replace(" ", "_").lower()}.png'))
        plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def compare_models(results, y_test):
    """Compare the performance of different models"""
    logging.info("\n=== Model Performance Comparison ===")
    
    # Prepare data for comparison plots
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    precisions = [results[model]['precision'] for model in models]
    recalls = [results[model]['recall'] for model in models]
    f1_scores = [results[model]['f1'] for model in models]
    
    # Create a bar chart comparing all metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_data = np.array([accuracies, precisions, recalls, f1_scores])
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, metrics_data[i], width, label=metric, color=colors[i])
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison', fontweight='bold')
    plt.xticks(x + width*1.5, models)
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of the bars
    for i, metric_values in enumerate(metrics_data):
        for j, value in enumerate(metric_values):
            plt.text(j + i*width, value + 0.01, f'{value:.3f}', 
                     ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'model_metrics_comparison.png'))
    plt.close()
    
    # ROC curve comparison
    plt.figure(figsize=(10, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for i, (name, result) in enumerate(results.items()):
        if result['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                    label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves_comparison.png'))
    plt.close()
    
    # Create a table with performance metrics
    metrics_table = pd.DataFrame({
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    }, index=models)
    
    logging.info("\nPerformance Metrics:")
    logging.info(metrics_table)
    
    # Find and print the best model for each metric
    best_model = {
        'Accuracy': models[np.argmax(accuracies)],
        'Precision': models[np.argmax(precisions)],
        'Recall': models[np.argmax(recalls)],
        'F1 Score': models[np.argmax(f1_scores)]
    }
    
    logging.info("\nBest model for each metric:")
    for metric, model in best_model.items():
        logging.info(f"  {metric}: {model}")
    
    return metrics_table

def optimize_random_forest(X_train, y_train, X_test, y_test):
    """Optimize hyperparameters for Random Forest"""
    logging.info("\n=== Random Forest Hyperparameter Optimization ===")
    
    # Define parameter distributions for random search
    param_distributions = {
        'n_estimators': stats.randint(50, 300),
        'max_depth': [None] + list(range(5, 50, 5)),
        'min_samples_split': stats.randint(2, 20),
        'min_samples_leaf': stats.randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Create a cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create random search object
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=20,  # Number of parameter settings to try
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Perform random search
    logging.info("Starting random search for Random Forest...")
    start_time = datetime.now()
    random_search.fit(X_train, y_train)
    training_time = datetime.now() - start_time
    logging.info(f"Random search completed in {training_time}")
    
    # Get best model and parameters
    best_rf = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Evaluate best model
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Optimized accuracy: {accuracy:.4f}")
    logging.info(f"Optimized F1 score: {f1:.4f}")
    
    # Save the optimized model
    joblib.dump(best_rf, os.path.join(MODELS_DIR, "optimized_random_forest.pkl"))
    logging.info(f"Optimized Random Forest model saved to {os.path.join(MODELS_DIR, 'optimized_random_forest.pkl')}")
    
    return best_rf, best_params

def optimize_xgboost(X_train, y_train, X_test, y_test):
    """Optimize hyperparameters for XGBoost"""
    logging.info("\n=== XGBoost Hyperparameter Optimization ===")
    
    # Define parameter distributions for random search
    param_distributions = {
        'n_estimators': stats.randint(50, 300),
        'max_depth': stats.randint(3, 10),
        'learning_rate': stats.uniform(0.01, 0.2),
        'subsample': stats.uniform(0.7, 0.3),
        'colsample_bytree': stats.uniform(0.7, 0.3),
        'gamma': stats.uniform(0, 0.5),
        'min_child_weight': stats.randint(1, 6),
        'reg_alpha': [0, 0.001, 0.01, 0.1, 1],
        'reg_lambda': [0, 0.001, 0.01, 0.1, 1]
    }
    
    # Create a cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create random search object
    random_search = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        ),
        param_distributions=param_distributions,
        n_iter=20,  # Number of parameter settings to try
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Perform random search
    logging.info("Starting random search for XGBoost...")
    start_time = datetime.now()
    random_search.fit(X_train, y_train)
    training_time = datetime.now() - start_time
    logging.info(f"Random search completed in {training_time}")
    
    # Get best model and parameters
    best_xgb = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Evaluate best model
    y_pred = best_xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Optimized accuracy: {accuracy:.4f}")
    logging.info(f"Optimized F1 score: {f1:.4f}")
    
    # Save the optimized model
    joblib.dump(best_xgb, os.path.join(MODELS_DIR, "optimized_xgboost.pkl"))
    logging.info(f"Optimized XGBoost model saved to {os.path.join(MODELS_DIR, 'optimized_xgboost.pkl')}")
    
    return best_xgb, best_params

# Main execution
print("=====================================================")
print("NETWORK MALWARE DETECTION - MACHINE LEARNING MODELS")
print("=====================================================")
logging.info("Starting machine learning model training and evaluation")

# Load preprocessed data
logging.info("Loading preprocessed data from: " + os.path.join(PROCESSED_DIR, 'processed_data.npz'))
X_train, X_test, y_train, y_test, features, encoders = load_processed_data()

# Train the models
logging.info("\n=== Training Models ===")

# Train models (faster models first)
rf_model = train_random_forest(X_train, y_train)
xgb_model = train_xgboost(X_train, y_train)
svm_model = train_svm(X_train, y_train)  # SVM is typically slowest

# Evaluate the models
logging.info("\n=== Evaluating Models ===")
results = {}

if rf_model is not None:
    results['Random Forest'] = evaluate_model(rf_model, X_test, y_test, "Random Forest", features)

if xgb_model is not None:
    results['XGBoost'] = evaluate_model(xgb_model, X_test, y_test, "XGBoost", features)

if svm_model is not None:
    results['SVM'] = evaluate_model(svm_model, X_test, y_test, "SVM")

# Compare models if at least one was trained successfully
if results:
    metrics_table = compare_models(results, y_test)
    
    # Save the metrics table to CSV
    metrics_file = os.path.join(REPORTS_DIR, "model_comparison_metrics.csv")
    metrics_table.to_csv(metrics_file)
    logging.info(f"Model comparison metrics saved to {metrics_file}")
    
    # Optimize best models
    logging.info("\n=== Optimizing Models ===")
    
    if rf_model is not None:
        opt_rf, rf_params = optimize_random_forest(X_train, y_train, X_test, y_test)
        if opt_rf is not None:
            results['Optimized RF'] = evaluate_model(opt_rf, X_test, y_test, "Optimized Random Forest", features)
    
    if xgb_model is not None:
        opt_xgb, xgb_params = optimize_xgboost(X_train, y_train, X_test, y_test)
        if opt_xgb is not None:
            results['Optimized XGB'] = evaluate_model(opt_xgb, X_test, y_test, "Optimized XGBoost", features)
    
    # Final comparison with optimized models
    final_metrics = compare_models(results, y_test)
    
    # Save the final metrics table
    final_metrics_file = os.path.join(REPORTS_DIR, "final_model_comparison_metrics.csv")
    final_metrics.to_csv(final_metrics_file)
    logging.info(f"Final model comparison metrics saved to {final_metrics_file}")

logging.info("\n=== ML Model Development Complete ===")
logging.info(f"All models saved in: {MODELS_DIR}")
logging.info(f"All visualizations saved in: {FIGURES_DIR}")
logging.info(f"All metrics reports saved in: {REPORTS_DIR}")