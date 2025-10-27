# --- open_feedback.py ---

# 1️⃣ Import necessary libraries
from openai import OpenAI
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from nltk.corpus import stopwords
import nltk
from collections import Counter

# Download stopwords list if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# 2️⃣ Load your API key securely from the .env file
load_dotenv()  # Loads variables from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- CONFIGURATION ---
DB_NAME = 'feedback.db'
TABLE_NAME = 'reviews'
TEXT_COLUMN = 'review_text'  # Confirm this is your correct column name
# ---------------------

# --- STEP 2: CLEANING FUNCTION ---
def clean_text(text):
    """
    Cleans text by removing punctuation, special characters, and stopwords.
    """
    text = str(text).lower() 
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# --- STEP 3: SENTIMENT ANALYSIS FUNCTIONS ---
def get_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(str(text))['compound']

def get_sentiment_textblob(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # 1. Load Data
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()
        print(f"Successfully loaded {len(df)} reviews.")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # 2. Clean the Text Data
    print("Starting data cleaning...")
    df['cleaned_review'] = df[TEXT_COLUMN].apply(clean_text)
    
    # 3. Perform Sentiment Analysis
    print("Performing sentiment analysis (VADER & TextBlob)...")
    df['vader_compound'] = df['cleaned_review'].apply(get_sentiment_vader)
    df[['textblob_polarity', 'textblob_subjectivity']] = df['cleaned_review'].apply(
        lambda x: pd.Series(get_sentiment_textblob(x))
    )

    # Classify final sentiment based on VADER
    def classify_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['final_sentiment'] = df['vader_compound'].apply(classify_sentiment)

    # --- STEP 4: IDENTIFY KEY THEMES (Simplified Word Frequency) ---
    print("Identifying key themes in NEGATIVE feedback...")
    negative_reviews = df[df['final_sentiment'] == 'Negative']
    all_negative_text = ' '.join(negative_reviews['cleaned_review'])
    word_counts = Counter(all_negative_text.split())

    # --- STEP 5: SUMMARIZE FINDINGS ---
    print("\n\n--- SUMMARY OF FINDINGS ---")
    print("\n1. Sentiment Distribution:")
    sentiment_summary = df['final_sentiment'].value_counts(normalize=True).mul(100).round(2)
    print(sentiment_summary.to_string() + '%')
    
    print("\n2. Top 10 Themes in NEGATIVE Feedback (Ignoring common words):")
    filter_words = ['apple', 'vision', 'pro', 'headset', 'device', 'really', 'much', 'time', 'like', 'get']
    filtered_words = {word: count for word, count in word_counts.items() if word not in filter_words}
    top_filtered_words = pd.DataFrame(Counter(filtered_words).most_common(10), columns=['Key Theme', 'Frequency'])
    print(top_filtered_words.to_string())
