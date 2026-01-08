"""
Regression Model Training
Trains models to predict problem difficulty score
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

# Try to import XGBoost, but make it optional
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

class DifficultyRegressor:
    """Train and manage regression models"""
    
    def __init__(self, model_type='gradient_boosting'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'extra_trees':
            self.model = ExtraTreesRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'linear_regression':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'elastic_net':
            self.model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            )
        elif model_type == 'xgboost' and HAS_XGBOOST:
            self.model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train):
        """Train the regression model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        print(f"Training {self.model_type} regressor...")
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation score (negative MSE)
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        rmse_scores = np.sqrt(-cv_scores)
        print(f"Cross-validation RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
        
        return self.model
    
    def predict(self, X):
        """Predict difficulty score (0-10 scale)"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Clip predictions to 0-10 range
        return np.clip(predictions, 0, 10)
    
    def save_model(self, regressor_path, scaler_path):
        """Save trained model and scaler"""
        joblib.dump(self.model, regressor_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Regressor saved to {regressor_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, regressor_path, scaler_path):
        """Load trained model and scaler"""
        self.model = joblib.load(regressor_path)
        self.scaler = joblib.load(scaler_path)
        print("Regressor and scaler loaded successfully")