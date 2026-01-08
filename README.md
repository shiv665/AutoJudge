# AutoJudge: Programming Problem Difficulty Predictor

An intelligent system that automatically predicts the difficulty class (Easy/Medium/Hard) and numerical difficulty score of programming problems based solely on their textual descriptions.

## 🎯 Project Overview

AutoJudge uses machine learning and advanced text segmentation techniques to analyze programming problem descriptions and predict their difficulty levels. The system combines multiple feature extraction methods including:

- **Text Segmentation**: Breaks down problems into semantic components (problem statement, constraints, examples)
- **Statistical Features**: Text length, word count, mathematical symbols
- **Keyword Analysis**: Identifies algorithm-related keywords (graph, DP, recursion, etc.)
- **TF-IDF Vectorization**: Captures important terms and phrases

## 📁 Project Structure

```
autojudge/
│
├── data/
│   └── problems_dataset.csv          # Your dataset file
│
├── models/                            # Saved trained models (generated)
│   ├── classifier.pkl
│   ├── regressor.pkl
│   ├── feature_extractor.pkl
│   ├── classifier_scaler.pkl
│   ├── regressor_scaler.pkl
│   ├── confusion_matrix.png
│   └── regression_plot.png
│
├── preprocessing/
│   ├── __init__.py
│   ├── text_cleaner.py               # Text cleaning utilities
│   ├── feature_extractor.py          # Feature engineering
│   └── segmentation.py               # Text segmentation module
│
├── training/
│   ├── __init__.py
│   ├── train_classifier.py           # Classification model
│   ├── train_regressor.py            # Regression model
│   └── evaluate.py                   # Model evaluation
│
├── static/
│   └── style.css                     # Web interface styling
│
├── templates/
│   └── index.html                    # Web interface HTML
│
├── app.py                            # Flask application
├── train_models.py                   # Main training script
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🚀 Installation & Setup

### 1. Clone or Download the Project

Place all files in a directory called `autojudge/`

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Your Dataset

Place your dataset CSV file in the `data/` folder with the filename `problems_dataset.csv`

**Required CSV Columns:**
- `title` - Problem title
- `description` - Problem description
- `input_description` - Input format description
- `output_description` - Output format description
- `problem_class` - Difficulty class (Easy/Medium/Hard)
- `problem_score` - Numerical difficulty score

### 5. Train the Models

```bash
python train_models.py
```

This will:
- Load and preprocess the dataset
- Extract features using segmentation and NLP
- Train both classification and regression models
- Evaluate model performance
- Save trained models to the `models/` folder
- Generate visualization plots

**Expected Output:**
```
==============================================================
AutoJudge Training Pipeline
==============================================================

[1/7] Loading dataset...
Dataset loaded: 1000 samples
...
CLASSIFICATION RESULTS
Accuracy: 0.8500

REGRESSION RESULTS
MAE: 150.25
RMSE: 200.50
R-squared: 0.75
==============================================================
```

### 6. Run the Web Application

```bash
python app.py
```

Open your browser and navigate to: `http://localhost:5000`

## 🎨 Using the Web Interface

1. **Enter Problem Details:**
   - Problem Title (optional)
   - Problem Description (required)
   - Input Description (optional)
   - Output Description (optional)

2. **Click "Predict Difficulty"**

3. **View Results:**
   - Predicted Difficulty Class (Easy/Medium/Hard)
   - Predicted Difficulty Score (numerical value)
   - Confidence percentage (if available)

## 🧠 How It Works

### Text Segmentation

The system segments problem text into meaningful components:

```python
# Example segmentation
{
    'problem_statement': 'Given an array of integers...',
    'constraints': '1 ≤ n ≤ 10^5',
    'examples': 'Input: [1,2,3] Output: 6',
    'explanation': 'Additional clarifications...'
}
```

### Feature Extraction

Multiple feature types are extracted:

1. **Basic Features**: text length, word count, math symbols
2. **Segmentation Features**: number of sentences, paragraphs, presence of constraints
3. **Algorithmic Indicators**: counts of keywords like "graph", "dp", "recursion"
4. **TF-IDF Features**: important terms and phrases
5. **Keyword Features**: specific algorithm-related terms

### Models

**Classification Model:**
- Algorithm: Random Forest Classifier
- Task: Predict Easy/Medium/Hard
- Metrics: Accuracy, Confusion Matrix

**Regression Model:**
- Algorithm: Random Forest Regressor
- Task: Predict numerical difficulty score
- Metrics: MAE, RMSE, R²

## 📊 Model Performance

After training, check the `models/` folder for:
- `confusion_matrix.png` - Classification performance visualization
- `regression_plot.png` - Actual vs Predicted scores plot

Example metrics:
- **Classification Accuracy**: 80-90% (depends on dataset)
- **Regression MAE**: Varies by score range
- **Regression RMSE**: Varies by score range

## 🔧 Customization

### Using Different Models

Edit `train_models.py`:

```python
# For classification
classifier = DifficultyClassifier(model_type='logistic_regression')
# Options: 'random_forest', 'logistic_regression', 'svm'

# For regression
regressor = DifficultyRegressor(model_type='gradient_boosting')
# Options: 'random_forest', 'linear_regression', 'gradient_boosting'
```

### Adjusting Feature Extraction

Edit `preprocessing/feature_extractor.py` to:
- Add new algorithm keywords
- Modify TF-IDF parameters
- Add custom feature extraction logic

### Modifying Segmentation

Edit `preprocessing/segmentation.py` to:
- Add new segmentation patterns
- Customize constraint/example extraction
- Implement domain-specific segmentation

## 🐛 Troubleshooting

### Models Not Found Error

```
ERROR: Models not found!
Please run the training script first:
  python train_models.py
```

**Solution**: Train the models first before running the web app

### Import Errors

```
ModuleNotFoundError: No module named 'flask'
```

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### NLTK Data Error

```
LookupError: Resource punkt not found
```

**Solution**: The code automatically downloads required NLTK data. If it fails:
```python
import nltk
nltk.download('punkt')
```

### Dataset Not Found

```
Error: Dataset not found at data/problems_dataset.csv
```

**Solution**: Place your CSV file in the `data/` folder with exact filename

## 📈 Future Enhancements

Possible improvements:
- Deep learning models (BERT, transformers)
- Multi-language support
- Real-time learning from user feedback
- API endpoint for integration
- Batch prediction interface
- More detailed difficulty breakdowns
- Topic classification (graphs, DP, strings, etc.)

## 📝 Technical Details

### Dependencies
- **Flask**: Web framework
- **Scikit-learn**: Machine learning models
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **NLTK**: Natural language processing
- **Joblib**: Model serialization

### Key Algorithms
- Random Forest for classification and regression
- TF-IDF for text vectorization
- Standard scaling for feature normalization
- Cross-validation for model evaluation

### Segmentation Techniques
- Sentence tokenization using NLTK
- Paragraph segmentation using regex
- Semantic component extraction
- Constraint and example detection

## 🤝 Contributing

To extend this project:

1. Add new feature extractors in `preprocessing/feature_extractor.py`
2. Implement new models in `training/`
3. Enhance segmentation in `preprocessing/segmentation.py`
4. Improve the UI in `templates/index.html` and `static/style.css`


## 🙋 Support

If you encounter any issues:
1. Check that all files are in the correct folders
2. Ensure your dataset has the required columns
3. Verify all dependencies are installed
4. Check console output for error messages

---

