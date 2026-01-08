"""
Model Evaluation Module
Evaluates classification and regression models
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Evaluate trained models"""
    
    @staticmethod
    def evaluate_classifier(y_true, y_pred, class_names=None):
        """
        Evaluate classification model
        Returns dictionary with accuracy, confusion matrix, and report
        """
        results = {}
        
        # Accuracy
        results['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Confusion Matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification Report
        results['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names
        )
        
        return results
    
    @staticmethod
    def evaluate_regressor(y_true, y_pred):
        """
        Evaluate regression model
        Returns dictionary with MAE, RMSE, and R2 score
        """
        results = {}
        
        # Mean Absolute Error
        results['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Root Mean Squared Error
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # R-squared Score
        results['r2'] = r2_score(y_true, y_pred)
        
        return results
    
    @staticmethod
    def plot_confusion_matrix(confusion_mat, class_names, save_path=None):
        """Plot and optionally save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_mat, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_regression_results(y_true, y_pred, save_path=None):
        """Plot actual vs predicted values for regression"""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Score')
        plt.ylabel('Predicted Score')
        plt.title('Actual vs Predicted')
        
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Score')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Regression plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_class_distribution(y_class, save_path=None):
        """Plot class distribution"""
        plt.figure(figsize=(8, 6))
        unique, counts = np.unique(y_class, return_counts=True)
        colors = ['#2ecc71', '#e74c3c', '#f39c12']  # green, red, orange
        plt.bar(unique, counts, color=colors)
        plt.xlabel('Problem Class')
        plt.ylabel('Count')
        plt.title('Distribution of Problem Difficulty Classes')
        
        # Add count labels on bars
        for i, (u, c) in enumerate(zip(unique, counts)):
            plt.text(i, c + 20, str(c), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_score_distribution(y_score, save_path=None):
        """Plot score distribution"""
        plt.figure(figsize=(10, 6))
        plt.hist(y_score, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
        plt.xlabel('Problem Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Problem Difficulty Scores')
        plt.axvline(x=np.mean(y_score), color='red', linestyle='--', label=f'Mean: {np.mean(y_score):.2f}')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Score distribution plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def print_evaluation_results(classifier_results=None, regressor_results=None):
        """Print evaluation results in a formatted way"""
        if classifier_results:
            print("\n" + "="*50)
            print("CLASSIFICATION RESULTS")
            print("="*50)
            print(f"Accuracy: {classifier_results['accuracy']:.4f}")
            print("\nConfusion Matrix:")
            print(classifier_results['confusion_matrix'])
            print("\nClassification Report:")
            print(classifier_results['classification_report'])
        
        if regressor_results:
            print("\n" + "="*50)
            print("REGRESSION RESULTS")
            print("="*50)
            print(f"Mean Absolute Error (MAE): {regressor_results['mae']:.4f}")
            print(f"Root Mean Squared Error (RMSE): {regressor_results['rmse']:.4f}")
            print(f"R-squared Score: {regressor_results['r2']:.4f}")