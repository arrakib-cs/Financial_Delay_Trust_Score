# Clean and preprocess NT narratives
"""
SCRIPT 02: Text Preprocessing
==============================
This script cleans and preprocesses the NT narrative text.

What this does:
1. Loads clean data
2. Cleans narrative text (remove HTML, special chars)
3. Tokenizes text
4. Extracts basic text features
5. Saves processed text data

HOW TO RUN THIS:
----------------
    python 02_text_preprocessing.py

OUTPUT:
-------
    ../data/processed/data_with_text_features.csv
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 02: TEXT PREPROCESSING")
print("=" * 80)

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

# Load clean data
print("\nLoading clean data...")
df = pd.read_csv(DATA_PROCESSED / 'clean_data.csv', low_memory=False)
print(f"Loaded {len(df):,} observations")

# Text cleaning functions
print("\n" + "=" * 80)
print("DEFINING TEXT CLEANING FUNCTIONS")
print("=" * 80)

def clean_text(text):
    """
    Clean narrative text by removing:
    - HTML tags
    - Extra whitespace
    - Special characters (keeping only letters, numbers, basic punctuation)
    
    Returns cleaned lowercase text
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\-\'\"]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def count_words(text):
    """Count words in text"""
    if not text:
        return 0
    return len(text.split())

def count_sentences(text):
    """Count sentences in text (approximation using periods)"""
    if not text:
        return 0
    return len(re.findall(r'[.!?]+', text))

def count_numbers(text):
    """Count numerical values in text"""
    if not text:
        return 0
    return len(re.findall(r'\b\d+[\d,\.]*%?\b', text))

def count_dates(text):
    """Count date-like patterns"""
    if not text:
        return 0
    date_patterns = [
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b',
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
    ]
    count = 0
    for pattern in date_patterns:
        count += len(re.findall(pattern, text.lower()))
    return count

def extract_text_features(text):
    """
    Extract multiple text features at once
    Returns dict with all features
    """
    features = {}
    
    features['word_count'] = count_words(text)
    features['sentence_count'] = count_sentences(text)
    
    if features['sentence_count'] > 0:
        features['avg_words_per_sentence'] = features['word_count'] / features['sentence_count']
    else:
        features['avg_words_per_sentence'] = 0
    
    features['number_count'] = count_numbers(text)
    features['date_count'] = count_dates(text)
    features['char_count'] = len(text)
    
    if features['word_count'] > 0:
        features['numerical_density'] = (features['number_count'] / features['word_count']) * 100
    else:
        features['numerical_density'] = 0
    
    return features

print("Text cleaning functions defined")

# Apply text cleaning
print("\n" + "=" * 80)
print("CLEANING NARRATIVE TEXT")
print("=" * 80)

print("\nCleaning narrative text...")
print("This may take a few minutes for large datasets...")

df['narrative_cleaned'] = df['narrative_clean'].apply(clean_text)

# Show example
print("\nEXAMPLE - BEFORE CLEANING:")
print("-" * 80)
sample_idx = df[df['narrative_clean'].notna()].index[0]
print(df.loc[sample_idx, 'narrative_clean'][:300] + "...")

print("\nEXAMPLE - AFTER CLEANING:")
print("-" * 80)
print(df.loc[sample_idx, 'narrative_cleaned'][:300] + "...")

# Extract text features
print("\n" + "=" * 80)
print("EXTRACTING TEXT FEATURES")
print("=" * 80)

print("\nExtracting features from all narratives...")
print("This will take a few minutes...")

text_features = df['narrative_cleaned'].apply(extract_text_features)

text_features_df = pd.DataFrame(text_features.tolist())
df = pd.concat([df, text_features_df], axis=1)

print("Text features extracted")

# Summary statistics
print("\n" + "=" * 80)
print("TEXT FEATURE SUMMARY")
print("=" * 80)

text_vars = [
    'word_count', 'sentence_count', 'avg_words_per_sentence',
    'number_count', 'date_count', 'numerical_density'
]

text_summary = pd.DataFrame({
    'N': df[text_vars].count(),
    'Mean': df[text_vars].mean(),
    'Median': df[text_vars].median(),
    'Std': df[text_vars].std(),
    'Min': df[text_vars].min(),
    'Max': df[text_vars].max()
})

print(text_summary.round(2))

# Distribution of text length
print("\nTEXT LENGTH DISTRIBUTION:")
print("-" * 80)
length_bins = [0, 50, 100, 200, 500, 1000, 10000]
length_labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
df['text_length_bin'] = pd.cut(df['word_count'], bins=length_bins, labels=length_labels)
length_dist = df['text_length_bin'].value_counts().sort_index()
for bin_label, count in length_dist.items():
    pct = count / len(df) * 100
    print(f"{bin_label:12s}: {count:6,} ({pct:5.1f}%)")

# Create log transformations for regression use
print("\n" + "=" * 80)
print("CREATING LOG TRANSFORMATIONS")
print("=" * 80)

df['ln_word_count'] = np.log(df['word_count'] + 1)
print("Created ln_word_count")

df['ln_sentence_count'] = np.log(df['sentence_count'] + 1)
print("Created ln_sentence_count")

# Save processed data
print("\n" + "=" * 80)
print("SAVING PROCESSED DATA")
print("=" * 80)

output_file = DATA_PROCESSED / 'data_with_text_features.csv'
df.to_csv(output_file, index=False)
print(f"Saved to: {output_file}")
print(f"Dataset now has {len(df.columns)} columns (added {len(text_vars)} text features)")

# Create text feature dictionary
print("\nTEXT FEATURE DEFINITIONS:")
print("-" * 80)
feature_defs = {
    'word_count': 'Total number of words in narrative',
    'sentence_count': 'Total number of sentences',
    'avg_words_per_sentence': 'Average sentence length',
    'number_count': 'Count of numerical values',
    'date_count': 'Count of date mentions',
    'numerical_density': 'Numbers per 100 words',
    'ln_word_count': 'Natural log of word count',
    'ln_sentence_count': 'Natural log of sentence count'
}

for feat, definition in feature_defs.items():
    print(f"{feat:25s}: {definition}")

print("\n" + "=" * 80)
print("TEXT PREPROCESSING COMPLETE")
print("=" * 80)
print("\nNext step: Run python 03_trust_components.py")
print("=" * 80)
