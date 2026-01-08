"""
Text Cleaning Module
Handles cleaning and preprocessing of problem descriptions
"""

import re
import string
from typing import Optional

class TextCleaner:
    """Cleans and preprocesses text data"""
    
    def __init__(self):
        self.math_symbols = set('+-*/=<>≤≥∑∏√^')
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def normalize_newlines(self, text: str) -> str:
        """Normalize different types of newlines"""
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        return text
    
    def preserve_math_symbols(self, text: str) -> str:
        """Ensure mathematical symbols are preserved"""
        # Add spaces around math symbols to ensure they're tokenized
        for symbol in self.math_symbols:
            text = text.replace(symbol, f' {symbol} ')
        return text
    
    def clean_text(self, text: Optional[str]) -> str:
        """
        Complete cleaning pipeline
        """
        if text is None or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Normalize newlines
        text = self.normalize_newlines(text)
        
        # Preserve mathematical symbols
        text = self.preserve_math_symbols(text)
        
        # Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        return text
    
    def combine_fields(self, title: str, description: str, 
                       input_desc: str, output_desc: str) -> str:
        """
        Combine all text fields into a single text
        """
        combined = []
        
        if title:
            combined.append(f"Title: {self.clean_text(title)}")
        
        if description:
            combined.append(f"Description: {self.clean_text(description)}")
        
        if input_desc:
            combined.append(f"Input: {self.clean_text(input_desc)}")
        
        if output_desc:
            combined.append(f"Output: {self.clean_text(output_desc)}")
        
        return " ".join(combined)