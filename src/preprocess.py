"""
Text preprocessing module for legal document classification.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import pandas as pd

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_text(text, stem=True):
    """
    Comprehensive text preprocessing pipeline.
    
    Args:
        text (str): Raw text document
        stem (bool): Whether to apply stemming (True) or lemmatization (False)
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # 1. HTML tag removal
    # Justification: Legal notices may contain HTML formatting from email/websites
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # 2. Lowercasing
    # Justification: Case doesn't matter for classification (e.g., "Contract" vs "contract")
    text = text.lower()
    
    # 3. Punctuation and special character removal
    # Justification: Removes noise while preserving alphanumeric content
    # Keeps hyphens as they may be relevant in legal terms (e.g., "non-compete")
    text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Tokenization
    # Justification: Split text into individual tokens for further processing
    tokens = nltk.word_tokenize(text)
    
    # 5. Stopword removal
    # Justification: Remove common words that carry little meaning for classification
    # Using NLTK's default list as it's comprehensive and well-tested
    stop_words = set(stopwords.words('english'))
    # Adding domain-specific stopwords
    legal_stopwords = {'hereinafter', 'aforesaid', 'thereof', 'thereto', 'herein'}
    stop_words.update(legal_stopwords)
    
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # 6. Stemming (vs Lemmatization)
    # Justification: Using stemmer for speed and performance
    # PorterStemmer is faster than lemmatization and sufficient for legal terms
    # (e.g., "dispute", "disputed", "disputing" all stem to "disput")
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

def preprocess_dataframe(df, text_column='text', stem=True):
    """
    Apply preprocessing to a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of text column
        stem (bool): Whether to apply stemming
        
    Returns:
        pd.DataFrame: DataFrame with preprocessed column
    """
    df = df.copy()
    df['cleaned_text'] = df[text_column].apply(lambda x: preprocess_text(x, stem))
    return df