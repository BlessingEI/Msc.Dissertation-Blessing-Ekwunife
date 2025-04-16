import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
from pathlib import Path
import warnings
import joblib
from datetime import datetime
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            roc_curve, auc, f1_score, 
                            precision_score, recall_score)
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
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "Thesis", "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(REPORTS_DIR, 'neural_network.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

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
    
    return X_train, X_test, y_train, y_test, features.tolist()

class MalwareDetectionNN(nn.Module):
    """Neural Network for IoT Malware Detection"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(MalwareDetectionNN, self).__init__()
        
        # Input layer
        layers = [nn.Linear(input_size, hidden_sizes[0]),
                  nn.BatchNorm1d(hidden_sizes[0]),
                  nn.ReLU(),
                  nn.Dropout(dropout_rate)]
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.extend([
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.BatchNorm1d(hidden_sizes[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_neural_network(X_train, y_train, X_test, y_test, input_size, batch_size=64, 
                         epochs=100, learning_rate=0.001, patience=10):
    """Train a neural network for malware detection"""
    logging.info("\n=== Training Neural Network ===")
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
    
    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    model = MalwareDetectionNN(input_size)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                    patience=5, verbose=True)
    
    # Training setup
    logging.info(f"Neural Network architecture:\n{model}")
    logging.info(f"Training with batch size: {batch_size}, learning rate: {learning_rate}")
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    best_model = None
    no_improve_epochs = 0
    
    start_time = datetime.now()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / len(test_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # Print epoch results
        logging.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if no_improve_epochs >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = datetime.now() - start_time
    logging.info(f"Training completed in {training_time}")
    
    # Load the best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "neural_network.pth"))
    logging.info(f"Model saved to {os.path.join(MODELS_DIR, 'neural_network.pth')}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'neural_network_training_curves.png'))
    plt.close()
    
    return model, (train_losses, val_losses, train_accs, val_accs)

def evaluate_neural_network(model, X_test, y_test):
    """Evaluate the neural network model"""
    logging.info("\n=== Evaluating Neural Network ===")
    
    # Convert test data to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        y_prob = model(X_test_tensor).cpu().numpy().flatten()
    
    # Convert probabilities to class labels
    y_pred = (y_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logging.info("Neural Network Performance:")
    logging.info(f"  Accuracy:  {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall:    {recall:.4f}")
    logging.info(f"  F1 Score:  {f1:.4f}")
    
    # Save metrics to a CSV file
    metrics_df = pd.DataFrame({
        'Model': ['Neural Network'],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1]
    })
    
    metrics_file = os.path.join(REPORTS_DIR, "neural_network_metrics.csv")
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
    plt.title('Confusion Matrix - Neural Network', fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix_neural_network.png'))
    plt.close()
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Neural Network', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curve_neural_network.png'))
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def compare_with_ml_models(nn_results):
    """Compare neural network with traditional ML models"""
    logging.info("\n=== Comparing Neural Network with ML Models ===")
    
    # Load ML model metrics if available
    ml_metrics_file = os.path.join(REPORTS_DIR, "model_comparison_metrics.csv")
    
    if os.path.exists(ml_metrics_file):
        ml_metrics = pd.read_csv(ml_metrics_file)
        
        # Add neural network results
        all_metrics = pd.concat([
            ml_metrics, 
            pd.DataFrame({
                'Model': ['Neural Network'],
                'Accuracy': [nn_results['accuracy']],
                'Precision': [nn_results['precision']],
                'Recall': [nn_results['recall']],
                'F1 Score': [nn_results['f1']]
            })
        ], ignore_index=True)
        
        # Save combined metrics
        combined_metrics_file = os.path.join(REPORTS_DIR, "all_models_comparison.csv")
        all_metrics.to_csv(combined_metrics_file, index=False)
        logging.info(f"Combined metrics saved to {combined_metrics_file}")
        
        # Create comparison chart
        plt.figure(figsize=(14, 10))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Set up the bar positions
        models = all_metrics['Model'].tolist()
        x = np.arange(len(models))
        width = 0.2
        
        # Plot bars for each metric
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, all_metrics[metric], width, label=metric, color=colors[i])
        
        # Add labels and formatting
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('All Models Performance Comparison', fontweight='bold')
        plt.xticks(x + width*1.5, models)
        plt.legend(loc='lower right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for i, metric in enumerate(metrics):
            for j, value in enumerate(all_metrics[metric]):
                plt.text(j + i*width, value + 0.01, f'{value:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'all_models_comparison.png'))
        plt.close()
        
        # Find the best model for each metric
        best_models = {}
        for metric in metrics:
            best_idx = all_metrics[metric].idxmax()
            best_models[metric] = all_metrics.loc[best_idx, 'Model']
        
        logging.info("\nBest model for each metric:")
        for metric, model in best_models.items():
            logging.info(f"  {metric}: {model}")
        
        return all_metrics
    else:
        logging.warning("ML model metrics file not found. Skipping comparison.")
        return None

# Main execution
print("=====================================================")
print("IOT MALWARE DETECTION - NEURAL NETWORK IMPLEMENTATION")
print("=====================================================")
logging.info("Starting neural network training and evaluation")

# Load preprocessed data
X_train, X_test, y_train, y_test, features = load_processed_data()

# Train neural network
# The input size is the number of features
input_size = X_train.shape[1]
model, training_history = train_neural_network(
    X_train, y_train, X_test, y_test, 
    input_size=input_size,
    batch_size=128,
    epochs=50,
    learning_rate=0.001,
    patience=7
)

# Evaluate neural network
nn_results = evaluate_neural_network(model, X_test, y_test)

# Compare with ML models
comparison_metrics = compare_with_ml_models(nn_results)

logging.info("\n=== Neural Network Implementation Complete ===")
logging.info(f"Neural Network model saved in: {os.path.join(MODELS_DIR, 'neural_network.pth')}")
logging.info(f"All visualizations saved in: {FIGURES_DIR}")
logging.info(f"All metrics reports saved in: {REPORTS_DIR}")