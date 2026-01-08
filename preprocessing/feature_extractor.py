"""
Feature Extraction Module
Extracts numerical features from text using segmentation and NLP techniques
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing.segmentation import TextSegmenter

class FeatureExtractor:
    """Extracts features from problem descriptions"""
    
    def __init__(self):
        self.segmenter = TextSegmenter()
        self.vectorizer = None
        
        # Keywords that indicate problem complexity - expanded list
        self.algorithm_keywords = [
            'graph', 'tree', 'dp', 'dynamic', 'greedy', 'recursion',
            'backtrack', 'dfs', 'bfs', 'dijkstra', 'binary search',
            'sort', 'hash', 'heap', 'stack', 'queue', 'array',
            'string', 'matrix', 'list', 'map', 'set',
            # Advanced algorithms
            'segment tree', 'fenwick', 'trie', 'suffix', 'kmp',
            'union find', 'disjoint', 'minimum spanning', 'mst',
            'shortest path', 'floyd', 'bellman', 'topological',
            'strongly connected', 'bipartite', 'matching',
            'flow', 'network', 'convex hull', 'geometry',
            'modular', 'prime', 'gcd', 'lcm', 'combinatorics',
            'probability', 'expected', 'polynomial', 'fft',
            'bitwise', 'bitmask', 'xor', 'simulation',
            'interactive', 'game theory', 'nim', 'sprague',
            # Complexity indicators
            'optimal', 'minimum', 'maximum', 'lexicograph',
            'permutation', 'subset', 'partition', 'counting'
        ]
        
        # Difficulty indicator words
        self.difficulty_words = {
            'easy': ['simple', 'basic', 'straightforward', 'trivial', 'direct'],
            'medium': ['standard', 'classic', 'typical', 'common'],
            'hard': ['complex', 'difficult', 'tricky', 'challenging', 'optimal', 'advanced']
        }
        
        self.math_symbols = set('+-*/=<>≤≥∑∏√^()[]{}')
    
    def extract_basic_features(self, text: str) -> dict:
        """Extract basic statistical features from text"""
        features = {}
        
        # Text length features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        
        # Average word length
        words = text.split()
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        
        # Count mathematical symbols
        features['math_symbol_count'] = sum(1 for char in text if char in self.math_symbols)
        
        # Count digits (often indicate constraints)
        features['digit_count'] = sum(1 for char in text if char.isdigit())
        
        # Count uppercase letters (often in variable names)
        features['uppercase_count'] = sum(1 for char in text if char.isupper())
        
        # New features for better prediction
        # Count of numbers in text (constraints like 10^9)
        import re
        numbers = re.findall(r'\d+', text)
        features['number_count'] = len(numbers)
        if numbers:
            num_values = [int(n) for n in numbers if len(n) <= 9]
            features['max_number'] = max(num_values) if num_values else 0
            features['avg_number'] = np.mean(num_values) if num_values else 0
        else:
            features['max_number'] = 0
            features['avg_number'] = 0
        
        # Power notation count (e.g., 10^9, 10^18)
        features['power_notation_count'] = len(re.findall(r'\d+\s*\^\s*\d+', text))
        
        # Sentence count
        features['sentence_count'] = len(re.findall(r'[.!?]+', text))
        
        # Question complexity (number of conditions)
        features['condition_count'] = text.lower().count(' if ') + text.lower().count(' when ')
        
        return features
    
    def extract_segmentation_features(self, text: str) -> dict:
        """Extract features using text segmentation"""
        features = {}
        
        # Get segmentation analysis
        seg_analysis = self.segmenter.segment_and_analyze(text)
        
        # Sentence and paragraph counts
        features['num_sentences'] = seg_analysis['num_sentences']
        features['num_paragraphs'] = seg_analysis['num_paragraphs']
        
        # Boolean features
        features['has_constraints'] = int(seg_analysis['has_constraints'])
        features['has_examples'] = int(seg_analysis['has_examples'])
        
        # Count algorithm keywords
        algo_indicators = seg_analysis['algorithm_indicators']
        features['num_algorithm_keywords'] = len(algo_indicators)
        features['total_algorithm_mentions'] = sum(algo_indicators.values())
        
        # Count complexity indicators
        complexity_indicators = seg_analysis['complexity_indicators']
        features['num_complexity_keywords'] = len(complexity_indicators)
        features['total_complexity_mentions'] = sum(complexity_indicators.values())
        
        # Segment-specific features
        semantic_segs = seg_analysis['semantic_segments']
        features['problem_statement_length'] = len(semantic_segs['problem_statement'])
        features['constraints_length'] = len(semantic_segs['constraints'])
        features['examples_length'] = len(semantic_segs['examples'])
        
        return features
    
    def extract_keyword_features(self, text: str) -> dict:
        """Extract features based on specific keywords"""
        features = {}
        text_lower = text.lower()
        
        for keyword in self.algorithm_keywords:
            feature_name = f'keyword_{keyword.replace(" ", "_")}'
            features[feature_name] = text_lower.count(keyword)
        
        # Add difficulty indicator word counts
        for difficulty, words in self.difficulty_words.items():
            feature_name = f'difficulty_{difficulty}_words'
            features[feature_name] = sum(text_lower.count(w) for w in words)
        
        return features
    
    def extract_all_features(self, text: str) -> dict:
        """
        Extract all features from text
        Returns a dictionary of feature_name: feature_value
        """
        all_features = {}
        
        # Basic features
        all_features.update(self.extract_basic_features(text))
        
        # Segmentation features
        all_features.update(self.extract_segmentation_features(text))
        
        # Keyword features
        all_features.update(self.extract_keyword_features(text))
        
        return all_features
    
    def fit_vectorizer(self, texts, max_features=500):
        """Fit TF-IDF vectorizer on training texts"""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),  # Include trigrams for better context
            min_df=2,
            max_df=0.95,  # Remove too common terms
            stop_words='english',
            sublinear_tf=True  # Apply log scaling to term frequencies
        )
        self.vectorizer.fit(texts)
    
    def extract_tfidf_features(self, texts):
        """Extract TF-IDF features"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
        return self.vectorizer.transform(texts).toarray()
    
    def create_feature_matrix(self, texts):
        """
        Create complete feature matrix combining:
        - Manual features (statistical, segmentation, keywords)
        - TF-IDF features
        """
        # Extract manual features for all texts
        manual_features_list = []
        for text in texts:
            features = self.extract_all_features(text)
            manual_features_list.append(features)
        
        # Convert to numpy array
        feature_names = sorted(manual_features_list[0].keys())
        manual_features_array = np.array([
            [features[name] for name in feature_names]
            for features in manual_features_list
        ])
        
        # Extract TF-IDF features
        tfidf_features = self.extract_tfidf_features(texts)
        
        # Combine both feature sets
        combined_features = np.hstack([manual_features_array, tfidf_features])
        
        return combined_features, feature_names