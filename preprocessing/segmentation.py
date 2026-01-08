"""
Text Segmentation Module
Segments problem descriptions into meaningful components for better feature extraction
"""

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Dict, List

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class TextSegmenter:
    """
    Segments programming problem text into different semantic components
    to extract more meaningful features
    """
    
    def __init__(self):
        self.algorithm_keywords = {
            'graph', 'tree', 'dp', 'dynamic programming', 'greedy', 
            'recursion', 'backtrack', 'dfs', 'bfs', 'dijkstra',
            'binary search', 'sort', 'hash', 'heap', 'stack', 'queue'
        }
        
        self.complexity_indicators = {
            'constraint', 'limit', 'maximum', 'minimum', 'optimize',
            'efficient', 'time complexity', 'space complexity'
        }
        
        self.math_symbols = set('+-*/=<>≤≥∑∏√^()[]{}')
    
    def segment_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        try:
            return sent_tokenize(text)
        except:
            # Fallback to simple split
            return text.split('.')
    
    def segment_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def extract_constraint_segment(self, text: str) -> str:
        """Extract the constraints/limits segment from problem description"""
        constraint_pattern = r'(constraint|limit|condition|note)s?[\s:]+(.*?)(?=\n\n|$)'
        match = re.search(constraint_pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(2) if match else ""
    
    def extract_example_segment(self, text: str) -> str:
        """Extract example/test case segments"""
        example_pattern = r'(example|sample|test case)s?[\s:]+(.*?)(?=\n\n|$)'
        match = re.search(example_pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(2) if match else ""
    
    def segment_by_semantic_type(self, text: str) -> Dict[str, str]:
        """
        Segment text into semantic components:
        - problem_statement: core problem description
        - constraints: limits and conditions
        - examples: sample inputs/outputs
        - explanation: additional clarifications
        """
        segments = {
            'problem_statement': '',
            'constraints': '',
            'examples': '',
            'explanation': ''
        }
        
        # Extract constraints
        segments['constraints'] = self.extract_constraint_segment(text)
        
        # Extract examples
        segments['examples'] = self.extract_example_segment(text)
        
        # Get paragraphs
        paragraphs = self.segment_by_paragraphs(text)
        
        if paragraphs:
            # First paragraph usually contains main problem statement
            segments['problem_statement'] = paragraphs[0]
            
            # Remaining paragraphs are explanations
            if len(paragraphs) > 1:
                segments['explanation'] = ' '.join(paragraphs[1:])
        
        return segments
    
    def extract_algorithmic_indicators(self, text: str) -> Dict[str, int]:
        """
        Count occurrences of algorithm-related keywords in text
        Useful for understanding problem complexity
        """
        text_lower = text.lower()
        indicators = {}
        
        for keyword in self.algorithm_keywords:
            count = text_lower.count(keyword)
            if count > 0:
                indicators[keyword] = count
        
        return indicators
    
    def extract_complexity_indicators(self, text: str) -> Dict[str, int]:
        """Extract indicators related to problem complexity"""
        text_lower = text.lower()
        indicators = {}
        
        for keyword in self.complexity_indicators:
            count = text_lower.count(keyword)
            if count > 0:
                indicators[keyword] = count
        
        return indicators
    
    def segment_and_analyze(self, text: str) -> Dict:
        """
        Complete segmentation and analysis of problem text
        Returns a dictionary with all segmented components and features
        """
        result = {
            'sentences': self.segment_by_sentences(text),
            'paragraphs': self.segment_by_paragraphs(text),
            'semantic_segments': self.segment_by_semantic_type(text),
            'algorithm_indicators': self.extract_algorithmic_indicators(text),
            'complexity_indicators': self.extract_complexity_indicators(text),
            'num_sentences': len(self.segment_by_sentences(text)),
            'num_paragraphs': len(self.segment_by_paragraphs(text)),
            'has_constraints': bool(self.extract_constraint_segment(text)),
            'has_examples': bool(self.extract_example_segment(text))
        }
        
        return result