"""
Utility functions for SSAES model training pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def load_data(file_path):
    """
    Load dataset from CSV file with error handling.
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame or None: Loaded dataframe or None if file not found
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Please upload your dataset to the data/demo/ folder")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def train_test_split_save(df, target_col='pass_fail', test_size=0.2, random_state=42, save_path='data/demo/'):
    """
    Split dataset and optionally save splits to CSV files.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name for stratification
        test_size (float): Test set proportion
        random_state (int): Random seed
        save_path (str): Directory to save split files
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        stratify = y if y.dtype == 'object' or y.nunique() < 10 else None
    else:
        X = df.iloc[:, :-1]  # All columns except last
        y = df.iloc[:, -1]   # Last column as target
        stratify = None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    # Save splits if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv(os.path.join(save_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(save_path, 'test.csv'), index=False)
        print(f"✅ Train/test splits saved to {save_path}")
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def save_model(model, file_path):
    """
    Save trained model using joblib.
    
    Args:
        model: Trained scikit-learn model
        file_path (str): Path to save model
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"✅ Model saved to {file_path}")


def load_model(file_path):
    """
    Load saved model using joblib.
    
    Args:
        file_path (str): Path to saved model
        
    Returns:
        Loaded model or None if file not found
    """
    try:
        model = joblib.load(file_path)
        print(f"✅ Model loaded from {file_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found: {file_path}")
        return None


def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix"):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path (str): Path to save plot (optional)
        title (str): Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")
    
    plt.show()


def create_model_comparison_table(models_results, save_path=None):
    """
    Create and save model comparison table.
    
    Args:
        models_results (dict): Dictionary with model names as keys and metrics as values
        save_path (str): Path to save CSV (optional)
        
    Returns:
        pd.DataFrame: Comparison table
    """
    df_comparison = pd.DataFrame(models_results).T
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_comparison.to_csv(save_path)
        print(f"✅ Model comparison saved to {save_path}")
    
    return df_comparison


def plot_feature_importance(model, feature_names, save_path=None, top_n=10):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        save_path (str): Path to save plot (optional)
        top_n (int): Number of top features to display
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Feature importance plot saved to {save_path}")
        
        plt.show()
    else:
        print("Model does not have feature_importances_ attribute")