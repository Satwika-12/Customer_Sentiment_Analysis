import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Download required NLTK data components
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load raw dataset
input_path = os.path.join('Data', 'raw', 'customer_tickets_raw.csv')
df = pd.read_csv(input_path)

print("Processing 5,000 customer support logs using NLTK & VADER...")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# 1. Feature Engineering: Text Length & Token Count
df['Char_Length'] = df['Ticket_Text'].apply(len)
df['Word_Count'] = df['Ticket_Text'].apply(lambda x: len(word_tokenize(x)))

# 2. Text Preprocessing: Cleaned Tokens
def clean_text(text):
    tokens = word_tokenize(text.lower())
    cleaned = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(cleaned)

df['Cleaned_Text'] = df['Ticket_Text'].apply(clean_text)

# 3. NLTK VADER Sentiment Analysis
def get_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return pd.Series([compound, sentiment], index=['Sentiment_Score', 'Sentiment_Label'])

df[['Sentiment_Score', 'Sentiment_Label']] = df['Ticket_Text'].apply(get_sentiment)

# 4. TF-IDF Keyword Extraction (Top Keyword Features)
tfidf = TfidfVectorizer(max_features=10)
tfidf_matrix = tfidf.fit_transform(df['Cleaned_Text'])
top_keywords = tfidf.get_feature_names_out()
print(f"Top Extracted Keywords Across Support Tickets: {list(top_keywords)}")

# Save processed dataframe
output_path = os.path.join('Data', 'processed', 'customer_tickets_processed.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"NLP Pipeline Execution Complete! Cleaned dataset saved to {output_path}")