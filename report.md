# AutoJudge - Problem Difficulty Prediction System

## Project Report

**Date:** January 8, 2026  
**Project:** AutoJudge - Automated Programming Problem Difficulty Assessment

---

## 1. Executive Summary

AutoJudge is a machine learning-based system designed to automatically predict the difficulty level of competitive programming problems. The system analyzes problem descriptions, input/output specifications, and other textual features to classify problems into difficulty categories (Easy, Medium, Hard) and predict difficulty scores.

---

## 2. Dataset Overview

### 2.1 Data Source
The dataset consists of competitive programming problems collected from online judges.

### 2.2 Dataset Statistics
| Metric | Value |
|--------|-------|
| Total Samples | 4,112 |
| Training Set | 3,289 (80%) |
| Test Set | 823 (20%) |
| Total Features Extracted | 1,094 |

### 2.3 Class Distribution
| Difficulty Class | Count | Percentage |
|-----------------|-------|------------|
| Easy | 766 | 18.6% |
| Medium | 1,405 | 34.2% |
| Hard | 1,941 | 47.2% |

![Class Distribution](class_distribution.png)

### 2.4 Score Distribution
The problem difficulty scores range from 1.10 to 9.70 on a 10-point scale.

![Score Distribution](score_distribution.png)

---

## 3. Feature Engineering

### 3.1 Text Preprocessing
- Combined multiple text fields: title, description, input/output specifications
- Applied text cleaning and normalization
- Removed stop words and special characters

### 3.2 Feature Categories

#### Basic Statistical Features
- Text length, word count, character count
- Average word length
- Mathematical symbol count
- Digit count and number patterns
- Power notation detection (e.g., 10^9)

#### Segmentation Features
- Sentence and paragraph counts
- Constraint section analysis
- Example section detection
- Problem statement complexity

#### Algorithm Keyword Features
Detection of 50+ algorithm-related keywords including:
- Graph algorithms (DFS, BFS, Dijkstra)
- Dynamic Programming patterns
- Data structures (Tree, Heap, Stack, Queue)
- Advanced concepts (Segment Tree, Trie, FFT)

#### TF-IDF Features
- N-gram range: (1, 3) - unigrams, bigrams, and trigrams
- Maximum features: 1,000
- Sublinear TF scaling applied

---

## 4. Model Architecture

### 4.1 Classification Model
**Model Type:** Random Forest Classifier

**Hyperparameters:**
- Number of estimators: 200
- Maximum depth: 20
- Minimum samples split: 5
- Minimum samples leaf: 2

### 4.2 Regression Model
**Model Type:** Gradient Boosting Regressor (Best performing)

**Hyperparameters:**
- Number of estimators: 300
- Learning rate: 0.05
- Maximum depth: 6
- Subsample ratio: 0.8
- Minimum samples split: 10

---

## 5. Classification Results

### 5.1 Overall Performance
| Metric | Value |
|--------|-------|
| **Accuracy** | **51.64%** |
| Macro Average F1-Score | 0.41 |
| Weighted Average F1-Score | 0.46 |

### 5.2 Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Easy | 0.55 | 0.24 | 0.33 | 153 |
| Medium | 0.43 | 0.17 | 0.25 | 281 |
| Hard | 0.53 | 0.87 | 0.66 | 389 |

### 5.3 Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

**Analysis:**
- The model shows strong performance in identifying "Hard" problems (87% recall)
- "Easy" and "Medium" classes show lower recall, indicating room for improvement
- The class imbalance (Hard class dominates) affects overall performance

---

## 6. Regression Results

### 6.1 Error Metrics
| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | 1.66 |
| Root Mean Squared Error (RMSE) | 2.00 |

### 6.2 Prediction Analysis
![Regression Results](regression_plot.png)

**Interpretation:**
- On average, predictions are within ±1.66 points of actual scores
- The model captures the general trend of difficulty scoring

---

## 7. Models Compared

| Model | RMSE | Performance |
|-------|------|-------------|
| Gradient Boosting | 1.99 | **Best** |
| XGBoost | 2.01 | Good |
| Extra Trees | 2.02 | Moderate |
| Random Forest | 2.04 | Baseline |

---

## 8. System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Input Text    │────▶│ Feature Extractor│────▶│  ML Models      │
│ (Problem Desc.) │     │                  │     │                 │
└─────────────────┘     │ - TF-IDF         │     │ - Classifier    │
                        │ - Keywords       │     │ - Regressor     │
                        │ - Statistics     │     │                 │
                        └──────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │    Output       │
                                                │ - Class (E/M/H) │
                                                │ - Score (1-10)  │
                                                └─────────────────┘
```

---

## 9. Files and Artifacts

### 9.1 Trained Models
| File | Description |
|------|-------------|
| `models/classifier.pkl` | Trained classification model |
| `models/classifier_scaler.pkl` | Feature scaler for classifier |
| `models/regressor.pkl` | Trained regression model |
| `models/regressor_scaler.pkl` | Feature scaler for regressor |
| `models/feature_extractor.pkl` | Fitted feature extractor |

### 9.2 Reports and Visualizations
| File | Description |
|------|-------------|
| `reports/confusion_matrix.png` | Classification confusion matrix |
| `reports/regression_plot.png` | Actual vs predicted scores |
| `reports/class_distribution.png` | Dataset class distribution |
| `reports/score_distribution.png` | Score histogram |
| `reports/training_metrics.json` | Training metrics in JSON format |

---

## 10. Usage Instructions

### 10.1 Training the Models
```bash
python -m train_models
```

### 10.2 Running the Web Application
```bash
python app.py
```

### 10.3 Making Predictions (Programmatic)
```python
import joblib

# Load models
classifier = joblib.load('models/classifier.pkl')
regressor = joblib.load('models/regressor.pkl')
feature_extractor = joblib.load('models/feature_extractor.pkl')

# Extract features from problem text
features = feature_extractor.create_feature_matrix([problem_text])

# Predict
difficulty_class = classifier.predict(features)
difficulty_score = regressor.predict(features)
```

---

## 11. Future Improvements

1. **Data Augmentation:** Collect more samples for underrepresented classes
2. **Advanced NLP:** Incorporate transformer-based embeddings (BERT, RoBERTa)
3. **Ensemble Methods:** Combine multiple models for better predictions
4. **Feature Selection:** Apply feature importance analysis to reduce dimensionality
5. **Active Learning:** Continuously improve with user feedback

---

## 12. Conclusion

AutoJudge successfully demonstrates the feasibility of automated problem difficulty assessment using machine learning. The system achieves:

- **51.64% classification accuracy** across three difficulty levels
- **MAE of 1.66** for score prediction on a 10-point scale
- Strong identification of hard problems (87% recall)

The model provides a solid foundation for assisting competitive programming platforms in automatically categorizing problem difficulty, with clear paths for future enhancement.

---

*Report generated by AutoJudge Training Pipeline*
