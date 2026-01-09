📊 Twitter Sentiment Analysis Dashboard

An interactive Streamlit web application for analyzing Twitter text sentiment using both lexicon-based and deep learning–based models.
The app supports single tweet analysis and bulk CSV processing with rich visualizations.

🚀 Features

🔍 Single Tweet Sentiment Analysis

📁 Bulk CSV Sentiment Analysis

🤖 Two sentiment models:

VADER – Fast, rule-based (ideal for large datasets)

RoBERTa – Transformer-based, high accuracy

📊 Interactive Pie Chart Visualization

☁️ Word Cloud for frequent terms

⚡ Optimized with Streamlit caching

🧠 Models Used
Model	Type	Description
VADER	Lexicon-based	Fast, optimized for social media text
RoBERTa	Transformer	High accuracy Twitter-trained model
🛠️ Tech Stack

Python 3.10+

Streamlit

Pandas

NLTK

VADER Sentiment

HuggingFace Transformers

Plotly

WordCloud

Matplotlib

📂 Project Structure
twitter-sentiment-dashboard/
│
├── main.py              # Streamlit application
├── README.md            # Project documentation
├── requirements.txt     # Dependencies
└── sample.csv           # Example dataset (optional)

📑 CSV File Format

Your CSV file must contain a column named text.

Example:

text
"I love this product!"
"This update is terrible"
"Not bad, could be better"

▶️ How to Run the App
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Streamlit App (IMPORTANT)
streamlit run main.py

3️⃣ Open in Browser
http://localhost:8501

📈 Output Examples

Sentiment Label: Positive / Negative / Neutral

Confidence Score

Sentiment Distribution Pie Chart

Word Cloud of frequent words

Preview of analyzed dataset

⚠️ Notes

RoBERTa is slower than VADER for large datasets.

First run may take time to download models.

Windows users may see HuggingFace cache warnings — these can be safely ignored.

📌 Use Cases

Social Media Monitoring

Brand Sentiment Analysis

Customer Feedback Analysis

Academic / Final Year Project

Resume & Portfolio Project

🧾 Resume Project Description (Optional)

Developed an interactive Twitter Sentiment Analysis dashboard using Streamlit, implementing both VADER and RoBERTa models for real-time and batch sentiment classification with data visualization and NLP preprocessing.

👨‍💻 Author

Hemant
Bachelor of Computer Applications (BCA)
Interested in Data Science & Machine Learning

⭐ Future Enhancements

Twitter API v2 integration

Model accuracy comparison

Deployment on Streamlit Cloud

Language detection & multilingual support

If you want, I can also:

✔️ Create requirements.txt

✔️ Make a GitHub-ready project

✔️ Help deploy it online

✔️ Write project explanation for viva/interview

Just tell me 👍

Is this conversation helpful so far?