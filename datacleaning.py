import re
import json
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'@[^\s]+', '', text)  
    text = re.sub(r'[^\w\s]', ' ', text) 
    text = text.lower()
    return text

# Function to tokenize text
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

# Function to normalize text
def normalize_text(tokens):
    porter = PorterStemmer()
    normalized_tokens = [porter.stem(token) for token in tokens]
    return normalized_tokens

# Function to categorize sentiment based on compound score
def categorize_sentiment(compound_score):
    if compound_score < 0:
        return 'Negative'
    elif compound_score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Function to filter entries
def filter_entries(entry):
    if not (1.0 <= entry['rating'] <= 5.0):
        return False
    if not entry['title'] or not entry['text']:
        return False
    entry.pop('images', None)
    entry.pop('timestamp', None)
    entry.pop('verified_purchase', None)
    entry.pop('helpful_vote', None)
    return True
