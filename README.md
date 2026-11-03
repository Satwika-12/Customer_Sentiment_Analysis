# Automated Customer Sentiment & Support Ticket Analytics 

An end-to-end NLP and text analytics pipeline designed to evaluate, tokenize, and classify unstructured customer support logs and tickets at scale.

## Project Overview
- **Data Ingestion & Simulation**: Pipeline processes 5,000+ unstructured customer support logs across categories (Technical, Billing, Product Inquiry, Account Management).
- **NLP Feature Engineering**: Text cleaning, tokenization, word count, character length extraction, and TF-IDF top-keyword analysis using NLTK and Scikit-Learn.
- **Sentiment Classification**: Integrated NLTK VADER Sentiment Intensity Analyzer to categorize tickets into Positive, Negative, and Neutral polarity buckets.
- **Visual Analytics**: Interactive visual insights generated via Seaborn and Matplotlib heatmaps, sentiment category breakdowns, and word length distributions.

## Tech Stack & Libraries
- **Language**: Python 3.14
- **NLP & Mining**: NLTK (VADER Lexicon, Tokenization, Stopwords)
- **Data Wrangling**: Pandas, NumPy
- **Machine Learning & Extraction**: Scikit-Learn (TF-IDF Vectorizer)
- **Visualizations**: Seaborn, Matplotlib

## Project Structure
Customer_Sentiment_Analysis/
├── Data/
│   ├── raw/
│   │   └── customer_tickets_raw.csv
│   └── processed/
│       └── customer_tickets_processed.csv
├── Scripts/
│   ├── generate_ticket_data.py
│   ├── preprocess_nlp.py
│   └── visualize_sentiment.py
├── Dashboards/
│   ├── sentiment_by_category.png
│   ├── priority_sentiment_heatmap.png
│   └── word_count_distribution.png
├── Models/
└── README.md