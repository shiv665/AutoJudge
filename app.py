"""
Flask Web Application for AutoJudge
Provides a web interface for predicting problem difficulty
"""

import os
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

from preprocessing import TextCleaner

app = Flask(__name__)

# Global variables for models
classifier = None
classifier_scaler = None
regressor = None
regressor_scaler = None
feature_extractor = None
text_cleaner = TextCleaner()

def load_models():
    """Load all trained models"""
    global classifier, classifier_scaler, regressor, regressor_scaler, feature_extractor
    
    try:
        print("Loading models...")
        classifier = joblib.load('models/classifier.pkl')
        classifier_scaler = joblib.load('models/classifier_scaler.pkl')
        regressor = joblib.load('models/regressor.pkl')
        regressor_scaler = joblib.load('models/regressor_scaler.pkl')
        feature_extractor = joblib.load('models/feature_extractor.pkl')
        print("All models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run 'python train_models.py' first to train the models.")
        return False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input data
        data = request.json
        title = data.get('title', '')
        description = data.get('description', '')
        input_desc = data.get('input_description', '')
        output_desc = data.get('output_description', '')
        
        # Validate input
        if not any([title, description, input_desc, output_desc]):
            return jsonify({
                'error': 'Please provide at least one text field'
            }), 400
        
        # Combine and clean text
        combined_text = text_cleaner.combine_fields(
            title, description, input_desc, output_desc
        )
        
        # Extract features
        X, _ = feature_extractor.create_feature_matrix([combined_text])
        
        # Make predictions
        # Classification
        X_scaled_class = classifier_scaler.transform(X)
        predicted_class = classifier.predict(X_scaled_class)[0]
        
        # Get prediction probabilities if available
        if hasattr(classifier, 'predict_proba'):
            class_proba = classifier.predict_proba(X_scaled_class)[0]
            confidence = float(max(class_proba)) * 100
        else:
            confidence = None
        
        # Regression
        X_scaled_reg = regressor_scaler.transform(X)
        predicted_score = float(regressor.predict(X_scaled_reg)[0])
        # Ensure score is within 0-10 range
        predicted_score = max(0, min(10, predicted_score))
        
        # Prepare response
        response = {
            'predicted_class': predicted_class,
            'predicted_score': round(predicted_score, 2),
            'confidence': round(confidence, 2) if confidence else None,
            'success': True
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    models_loaded = all([
        classifier is not None,
        regressor is not None,
        feature_extractor is not None
    ])
    
    return jsonify({
        'status': 'healthy' if models_loaded else 'models not loaded',
        'models_loaded': models_loaded
    })

if __name__ == '__main__':
    # Load models before starting the server
    models_loaded = load_models()
    
    if not models_loaded:
        print("\n" + "="*60)
        print("ERROR: Models not found!")
        print("="*60)
        print("Please run the training script first:")
        print("  python train_models.py")
        print("="*60)
        exit(1)
    
    print("\n" + "="*60)
    print("AutoJudge Web Server")
    print("="*60)
    print("Server starting at http://localhost:5000")
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)