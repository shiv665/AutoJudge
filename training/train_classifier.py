"""
Classification Model Training
Trains models to predict problem difficulty class (Easy/Medium/Hard)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

class DifficultyClassifier:
    """Train and manage classification models"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train):
        """Train the classification model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        print(f"Training {self.model_type} classifier...")
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.model
    
    def predict(self, X):
        """Predict difficulty class"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # For SVM, return decision function as proxy
            return self.model.decision_function(X_scaled)
    
    def save_model(self, classifier_path, scaler_path):
        """Save trained model and scaler"""
        joblib.dump(self.model, classifier_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Classifier saved to {classifier_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, classifier_path, scaler_path):
        """Load trained model and scaler"""
        self.model = joblib.load(classifier_path)
        self.scaler = joblib.load(scaler_path)
        print("Classifier and scaler loaded successfully")