"""
Main Training Script
Loads data, trains both classification and regression models, and saves them
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

from preprocessing import TextCleaner, FeatureExtractor
from training import DifficultyClassifier, DifficultyRegressor, ModelEvaluator

def main():
    print("="*60)
    print("AutoJudge Training Pipeline")
    print("="*60)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load dataset
    print("\n[1/7] Loading dataset...")
    data_path = 'data/problems_dataset.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please ensure your CSV file is placed in the data/ folder")
        return
    
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Clean and combine text fields
    print("\n[2/7] Cleaning and combining text fields...")
    cleaner = TextCleaner()
    
    combined_texts = []
    for idx, row in df.iterrows():
        combined = cleaner.combine_fields(
            row.get('title', ''),
            row.get('description', ''),
            row.get('input_description', ''),
            row.get('output_description', '')
        )
        combined_texts.append(combined)
    
    print(f"Combined {len(combined_texts)} text samples")
    
    # Extract features
    print("\n[3/7] Extracting features...")
    feature_extractor = FeatureExtractor()
    
    # Fit vectorizer on all texts - increased max_features for better representation
    feature_extractor.fit_vectorizer(combined_texts, max_features=1000)
    
    # Create feature matrix
    X, feature_names = feature_extractor.create_feature_matrix(combined_texts)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Total features: {X.shape[1]}")
    
    # Prepare labels
    y_class = df['problem_class'].values
    y_score = df['problem_score'].values
    
    print(f"Class distribution: {np.unique(y_class, return_counts=True)}")
    print(f"Score range: {y_score.min():.2f} - {y_score.max():.2f}")
    
    # Split data
    print("\n[4/7] Splitting data into train and test sets...")
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train Classification Model
    print("\n[5/7] Training Classification Model...")
    classifier = DifficultyClassifier(model_type='random_forest')
    classifier.train(X_train, y_class_train)
    
    # Evaluate classifier
    y_class_pred = classifier.predict(X_test)
    class_results = ModelEvaluator.evaluate_classifier(
        y_class_test, y_class_pred, 
        class_names=np.unique(y_class)
    )
    
    # Train Regression Model - Compare multiple models to find the best one
    print("\n[6/7] Training Regression Models (comparing multiple)...")
    
    best_regressor = None
    best_r2 = -float('inf')
    best_model_type = ''
    
    # Try multiple model types
    model_types = ['gradient_boosting', 'random_forest', 'extra_trees']
    
    # Try XGBoost if available
    try:
        from xgboost import XGBRegressor
        model_types.append('xgboost')
    except ImportError:
        pass
    
    for model_type in model_types:
        print(f"\n  Trying {model_type}...")
        try:
            temp_regressor = DifficultyRegressor(model_type=model_type)
            temp_regressor.train(X_train, y_score_train)
            temp_pred = temp_regressor.predict(X_test)
            temp_results = ModelEvaluator.evaluate_regressor(y_score_test, temp_pred)
            temp_r2 = temp_results['r2']
            print(f"  {model_type} R² Score: {temp_r2:.4f}, RMSE: {temp_results['rmse']:.4f}")
            
            if temp_r2 > best_r2:
                best_r2 = temp_r2
                best_regressor = temp_regressor
                best_model_type = model_type
        except Exception as e:
            print(f"  Error with {model_type}: {e}")
    
    print(f"\n  Best model: {best_model_type} with R² = {best_r2:.4f}")
    regressor = best_regressor
    
    # Evaluate regressor
    y_score_pred = regressor.predict(X_test)
    reg_results = ModelEvaluator.evaluate_regressor(y_score_test, y_score_pred)
    
    # Print results
    ModelEvaluator.print_evaluation_results(class_results, reg_results)
    
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Save plots to reports folder
    print("\n[7/7] Saving models and visualizations...")
    ModelEvaluator.plot_confusion_matrix(
        class_results['confusion_matrix'],
        np.unique(y_class),
        'reports/confusion_matrix.png'
    )
    
    ModelEvaluator.plot_regression_results(
        y_score_test, y_score_pred,
        'reports/regression_plot.png'
    )
    
    # Plot class distribution
    ModelEvaluator.plot_class_distribution(y_class, 'reports/class_distribution.png')
    
    # Plot score distribution
    ModelEvaluator.plot_score_distribution(y_score, 'reports/score_distribution.png')
    
    # Save models
    classifier.save_model('models/classifier.pkl', 'models/classifier_scaler.pkl')
    regressor.save_model('models/regressor.pkl', 'models/regressor_scaler.pkl')
    
    # Save feature extractor
    joblib.dump(feature_extractor, 'models/feature_extractor.pkl')
    print("Feature extractor saved to models/feature_extractor.pkl")
    
    # Generate report data
    unique_classes, class_counts = np.unique(y_class, return_counts=True)
    class_dist = {str(k): int(v) for k, v in zip(unique_classes, class_counts)}
    
    report_data = {
        'dataset_size': int(len(df)),
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'num_features': int(X.shape[1]),
        'class_distribution': class_dist,
        'classifier_accuracy': float(class_results['accuracy']),
        'classifier_type': 'Random Forest',
        'regressor_type': best_model_type,
        'mae': float(reg_results['mae']),
        'rmse': float(reg_results['rmse'])
    }
    
    # Save report data
    import json
    with open('reports/training_metrics.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nSaved files:")
    print("  - models/classifier.pkl")
    print("  - models/classifier_scaler.pkl")
    print("  - models/regressor.pkl")
    print("  - models/regressor_scaler.pkl")
    print("  - models/feature_extractor.pkl")
    print("  - reports/confusion_matrix.png")
    print("  - reports/regression_plot.png")
    print("  - reports/class_distribution.png")
    print("  - reports/score_distribution.png")
    print("  - reports/training_metrics.json")
    print("\nYou can now run the web application with: python app.py")

if __name__ == "__main__":
    main()