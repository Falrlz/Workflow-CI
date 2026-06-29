import os
import sys
import argparse

# Reconfigure stdout/stderr to handle Unicode emojis without crashing on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
import mlflow
import mlflow.sklearn

# Set up local paths relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "Telco_customer_churn_preprocessing", "train.csv")
test_path = os.path.join(script_dir, "Telco_customer_churn_preprocessing", "test.csv")
log_file = os.path.join(script_dir, "modelling.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("modelling_retrain")

def train_and_track(learning_rate, max_depth, max_iter):
    logger.info("==========================================")
    logger.info("Starting MLflow Project Retraining Pipeline")
    logger.info(f"Parameters: learning_rate={learning_rate}, max_depth={max_depth}, max_iter={max_iter}")
    logger.info("==========================================")
    
    logger.info("Loading preprocessed dataset splits...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error(f"Preprocessed train/test files not found at: {train_path} or {test_path}")
        raise FileNotFoundError("Preprocessed train/test files not found inside MLProject directory.")
        
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info("Dataset splits loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise e
    
    # Separate features and target
    X_train = train_df.drop(columns=['Churn Value'])
    y_train = train_df['Churn Value']
    X_test = test_df.drop(columns=['Churn Value'])
    y_test = test_df['Churn Value']
    
    logger.info(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")
    
    params = {
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "max_iter": max_iter,
        "class_weight": "balanced",
        "random_state": 42
    }
    
    # Set up outputs directory for local artifacts
    outputs_dir = os.path.join(script_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # MLflow experiment fallback
    if not mlflow.get_experiment_by_name("Telco_Customer_Churn_Retraining"):
        mlflow.set_experiment("Telco_Customer_Churn_Retraining")
        
    logger.info("Starting MLflow run...")
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_params(params)
        logger.info(f"Logged parameters to MLflow: {params}")
        
        # Train classifier
        logger.info("Training HistGradientBoostingClassifier...")
        model = HistGradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        logger.info("Model training completed.")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model performance
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        logger.info(f"Logged evaluation metrics to MLflow: {metrics}")
        
        # 1. Generate & save Confusion Matrix Plot
        logger.info("Generating Confusion Matrix plot...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Retained', 'Churn'], 
                    yticklabels=['Retained', 'Churn'])
        plt.title('Confusion Matrix - Retrained HGB')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_plot_path = os.path.join(outputs_dir, "confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=100)
        plt.close()
        
        mlflow.log_artifact(cm_plot_path)
        logger.info("Logged Confusion Matrix plot.")
        
        # 2. Generate & save Feature Importance Plot
        logger.info("Generating Feature Importance plot using Permutation Importance...")
        feature_names = X_train.columns
        result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
        importances = result.importances_mean
        
        indices = np.argsort(importances)[::-1]
        top_n = min(15, len(feature_names))
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Top {top_n} Feature Importances (Permutation Importance)")
        plt.bar(range(top_n), importances[indices[:top_n]], align="center", color='teal')
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.xlim([-1, top_n])
        plt.tight_layout()
        
        feat_plot_path = os.path.join(outputs_dir, "feature_importance.png")
        plt.savefig(feat_plot_path, dpi=100)
        plt.close()
        
        mlflow.log_artifact(feat_plot_path)
        logger.info("Logged Feature Importance plot.")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        logger.info("Logged model object to MLflow.")
        
    logger.info("==========================================")
    logger.info("Retraining Completed Successfully")
    logger.info("==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()
    
    try:
        train_and_track(args.learning_rate, args.max_depth, args.max_iter)
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        sys.exit(1)
