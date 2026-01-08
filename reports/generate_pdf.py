"""
Script to convert report.md to PDF using FPDF2
"""

import os
from fpdf import FPDF

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'AutoJudge - Project Report', 0, 0, 'C')
        self.ln(15)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font('Helvetica', 'B', 16)
            self.set_text_color(44, 62, 80)
        elif level == 2:
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(52, 73, 94)
        else:
            self.set_font('Helvetica', 'B', 12)
            self.set_text_color(127, 140, 141)
        self.cell(0, 10, title, 0, 1)
        self.ln(2)
        
    def body_text(self, text):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(51, 51, 51)
        self.multi_cell(0, 6, text)
        self.ln(2)
    
    def bold_text(self, text):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(41, 128, 185)
        self.multi_cell(0, 6, text)
        self.ln(2)
        
    def add_table(self, headers, data):
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(52, 152, 219)
        self.set_text_color(255, 255, 255)
        
        col_width = (self.w - 20) / len(headers)
        
        for header in headers:
            self.cell(col_width, 8, header, 1, 0, 'C', fill=True)
        self.ln()
        
        self.set_font('Helvetica', '', 10)
        self.set_text_color(51, 51, 51)
        
        fill = False
        for row in data:
            if fill:
                self.set_fill_color(249, 249, 249)
            else:
                self.set_fill_color(255, 255, 255)
            for cell in row:
                self.cell(col_width, 7, str(cell), 1, 0, 'C', fill=True)
            self.ln()
            fill = not fill
        self.ln(5)
    
    def add_image_if_exists(self, path, w=150):
        if os.path.exists(path):
            self.image(path, x=(self.w - w) / 2, w=w)
            self.ln(10)
        else:
            self.body_text(f"[Image not found: {path}]")

def generate_pdf():
    reports_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(reports_dir, 'report.pdf')
    
    pdf = PDFReport()
    
    # Title
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 20, 'AutoJudge', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Problem Difficulty Prediction System', 0, 1, 'C')
    pdf.cell(0, 10, 'Project Report - January 8, 2026', 0, 1, 'C')
    pdf.ln(10)
    
    # Section 1
    pdf.chapter_title('1. Executive Summary', 1)
    pdf.body_text('AutoJudge is a machine learning-based system designed to automatically predict the difficulty level of competitive programming problems. The system analyzes problem descriptions, input/output specifications, and other textual features to classify problems into difficulty categories (Easy, Medium, Hard) and predict difficulty scores.')
    pdf.ln(5)
    
    # Section 2
    pdf.chapter_title('2. Dataset Overview', 1)
    pdf.chapter_title('2.1 Dataset Statistics', 2)
    pdf.add_table(
        ['Metric', 'Value'],
        [
            ['Total Samples', '4,112'],
            ['Training Set', '3,289 (80%)'],
            ['Test Set', '823 (20%)'],
            ['Total Features', '1,094']
        ]
    )
    
    pdf.chapter_title('2.2 Class Distribution', 2)
    pdf.add_table(
        ['Difficulty Class', 'Count', 'Percentage'],
        [
            ['Easy', '766', '18.6%'],
            ['Medium', '1,405', '34.2%'],
            ['Hard', '1,941', '47.2%']
        ]
    )
    
    pdf.add_image_if_exists(os.path.join(reports_dir, 'class_distribution.png'), w=140)
    
    pdf.chapter_title('2.3 Score Distribution', 2)
    pdf.body_text('The problem difficulty scores range from 1.10 to 9.70 on a 10-point scale.')
    pdf.add_image_if_exists(os.path.join(reports_dir, 'score_distribution.png'), w=140)
    
    # Section 3
    pdf.add_page()
    pdf.chapter_title('3. Feature Engineering', 1)
    pdf.chapter_title('3.1 Text Preprocessing', 2)
    pdf.body_text('- Combined multiple text fields: title, description, input/output specifications')
    pdf.body_text('- Applied text cleaning and normalization')
    pdf.body_text('- Removed stop words and special characters')
    
    pdf.chapter_title('3.2 Feature Categories', 2)
    pdf.bold_text('Basic Statistical Features')
    pdf.body_text('Text length, word count, average word length, mathematical symbol count, digit patterns, power notation detection')
    
    pdf.bold_text('Algorithm Keyword Features')
    pdf.body_text('Detection of 50+ algorithm-related keywords including: Graph algorithms (DFS, BFS, Dijkstra), Dynamic Programming, Data structures (Tree, Heap, Stack), Advanced concepts (Segment Tree, Trie, FFT)')
    
    pdf.bold_text('TF-IDF Features')
    pdf.body_text('N-gram range: (1, 3), Maximum features: 1,000, Sublinear TF scaling applied')
    
    # Section 4
    pdf.add_page()
    pdf.chapter_title('4. Model Architecture', 1)
    pdf.chapter_title('4.1 Classification Model', 2)
    pdf.bold_text('Model Type: Random Forest Classifier')
    pdf.add_table(
        ['Parameter', 'Value'],
        [
            ['Number of Estimators', '200'],
            ['Maximum Depth', '20'],
            ['Min Samples Split', '5'],
            ['Min Samples Leaf', '2']
        ]
    )
    
    pdf.chapter_title('4.2 Regression Model', 2)
    pdf.bold_text('Model Type: Gradient Boosting Regressor (Best performing)')
    pdf.add_table(
        ['Parameter', 'Value'],
        [
            ['Number of Estimators', '300'],
            ['Learning Rate', '0.05'],
            ['Maximum Depth', '6'],
            ['Subsample Ratio', '0.8']
        ]
    )
    
    # Section 5
    pdf.add_page()
    pdf.chapter_title('5. Classification Results', 1)
    pdf.chapter_title('5.1 Overall Performance', 2)
    pdf.add_table(
        ['Metric', 'Value'],
        [
            ['Accuracy', '51.64%'],
            ['Macro Avg F1-Score', '0.41'],
            ['Weighted Avg F1-Score', '0.46']
        ]
    )
    
    pdf.chapter_title('5.2 Per-Class Performance', 2)
    pdf.add_table(
        ['Class', 'Precision', 'Recall', 'F1-Score'],
        [
            ['Easy', '0.55', '0.24', '0.33'],
            ['Medium', '0.43', '0.17', '0.25'],
            ['Hard', '0.53', '0.87', '0.66']
        ]
    )
    
    pdf.chapter_title('5.3 Confusion Matrix', 2)
    pdf.add_image_if_exists(os.path.join(reports_dir, 'confusion_matrix.png'), w=130)
    
    pdf.body_text('Analysis: The model shows strong performance in identifying "Hard" problems (87% recall). "Easy" and "Medium" classes show lower recall due to class imbalance.')
    
    # Section 6
    pdf.add_page()
    pdf.chapter_title('6. Regression Results', 1)
    pdf.chapter_title('6.1 Error Metrics', 2)
    pdf.add_table(
        ['Metric', 'Value'],
        [
            ['Mean Absolute Error (MAE)', '1.66'],
            ['Root Mean Squared Error (RMSE)', '2.00']
        ]
    )
    
    pdf.chapter_title('6.2 Prediction Analysis', 2)
    pdf.add_image_if_exists(os.path.join(reports_dir, 'regression_plot.png'), w=150)
    pdf.body_text('On average, predictions are within +/- 1.66 points of actual scores on a 10-point scale.')
    
    # Section 7
    pdf.chapter_title('7. Models Compared', 1)
    pdf.add_table(
        ['Model', 'RMSE', 'Performance'],
        [
            ['Gradient Boosting', '1.99', 'Best'],
            ['XGBoost', '2.01', 'Good'],
            ['Extra Trees', '2.02', 'Moderate'],
            ['Random Forest', '2.04', 'Baseline']
        ]
    )
    
    # Section 8
    pdf.add_page()
    pdf.chapter_title('8. Files and Artifacts', 1)
    pdf.chapter_title('8.1 Trained Models', 2)
    pdf.add_table(
        ['File', 'Description'],
        [
            ['classifier.pkl', 'Trained classification model'],
            ['classifier_scaler.pkl', 'Feature scaler for classifier'],
            ['regressor.pkl', 'Trained regression model'],
            ['regressor_scaler.pkl', 'Feature scaler for regressor'],
            ['feature_extractor.pkl', 'Fitted feature extractor']
        ]
    )
    
    # Section 9
    pdf.chapter_title('9. Future Improvements', 1)
    pdf.body_text('1. Data Augmentation: Collect more samples for underrepresented classes')
    pdf.body_text('2. Advanced NLP: Incorporate transformer-based embeddings (BERT, RoBERTa)')
    pdf.body_text('3. Ensemble Methods: Combine multiple models for better predictions')
    pdf.body_text('4. Feature Selection: Apply feature importance analysis')
    pdf.body_text('5. Active Learning: Continuously improve with user feedback')
    
    # Conclusion
    pdf.add_page()
    pdf.chapter_title('10. Conclusion', 1)
    pdf.body_text('AutoJudge successfully demonstrates the feasibility of automated problem difficulty assessment using machine learning. The system achieves:')
    pdf.ln(3)
    pdf.bold_text('- 51.64% classification accuracy across three difficulty levels')
    pdf.bold_text('- MAE of 1.66 for score prediction on a 10-point scale')
    pdf.bold_text('- Strong identification of hard problems (87% recall)')
    pdf.ln(5)
    pdf.body_text('The model provides a solid foundation for assisting competitive programming platforms in automatically categorizing problem difficulty, with clear paths for future enhancement.')
    
    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(128)
    pdf.cell(0, 10, 'Report generated by AutoJudge Training Pipeline', 0, 1, 'C')
    
    # Save PDF
    pdf.output(pdf_path)
    print(f"PDF report generated: {pdf_path}")

if __name__ == "__main__":
    generate_pdf()
