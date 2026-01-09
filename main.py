import streamlit as st
import pandas as pd
import re
import nltk
import emoji
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Twitter Sentiment Dashboard", layout="wide")

@st.cache_resource
def initialize_nltk():
    nltk.download('punkt_tab')
    nltk.download('stopwords')

initialize_nltk()

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_models():
    vader = SentimentIntensityAnalyzer()
    roberta = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=True
    )
    return vader, roberta

vader, roberta = load_models()

# --- 3. CLEANING ---
def clean_tweet(text):
    text = emoji.demojize(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z#\s]', '', text)
    return text.lower().strip()

# --- 4. SENTIMENT FUNCTIONS ---
def get_roberta_sentiment(text):
    results = roberta(text[:512])[0]
    best = max(results, key=lambda x: x['score'])
    return best['label'].capitalize(), best['score']

def get_vader_sentiment(text):
    score = vader.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive', score
    elif score <= -0.05:
        return 'Negative', score
    else:
        return 'Neutral', score

# --- 5. UI ---
st.title("📊 Twitter Sentiment Analysis Dashboard")
st.markdown("Analyze sentiments using **VADER** and **RoBERTa** models.")

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["VADER (Fast)", "RoBERTa (Accurate)"]
)

tab1, tab2 = st.tabs(["Single Tweet", "CSV Analysis"])

# --- TAB 1 ---
with tab1:
    tweet = st.text_area(
        "Enter a tweet:",
        "I love the new features of this app! 😍 #tech"
    )

    if st.button("Analyze Sentiment", type="primary"):
        cleaned = clean_tweet(tweet)

        if model_choice == "RoBERTa (Accurate)":
            label, score = get_roberta_sentiment(cleaned)
        else:
            label, score = get_vader_sentiment(cleaned)

        col1, col2 = st.columns(2)
        col1.metric("Sentiment", label)
        col2.metric("Confidence", f"{score:.2f}")

# --- TAB 2 ---
with tab2:
    file = st.file_uploader("Upload CSV (must contain 'text' column)", type="csv")

    if file:
        df = pd.read_csv(file)

        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            with st.spinner("Analyzing tweets..."):
                df['clean_text'] = df['text'].astype(str).apply(clean_tweet)

                if model_choice == "RoBERTa (Accurate)":
                    df[['Sentiment', 'Score']] = df['clean_text'].apply(
                        lambda x: pd.Series(get_roberta_sentiment(x))
                    )
                else:
                    df[['Sentiment', 'Score']] = df['clean_text'].apply(
                        lambda x: pd.Series(get_vader_sentiment(x))
                    )

            st.subheader("Sentiment Distribution")
            fig = px.pie(
                df,
                names='Sentiment',
                color='Sentiment',
                color_discrete_map={
                    'Positive': 'green',
                    'Negative': 'red',
                    'Neutral': 'gray'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Word Cloud")
            text_blob = " ".join(df['clean_text'])
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white'
            ).generate(text_blob)

            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

            st.subheader("Sample Results")
            st.dataframe(df.head(10))

st.sidebar.info(
    "ℹ️ RoBERTa is more accurate but significantly slower than VADER on large datasets."
)
