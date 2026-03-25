# CineInsight 🎬

An AI-powered movie sentiment analysis and recommendation web app built during the Microsoft Elevate Internship.

## Live Demo

https://michael-unova-cineinsight-app-cunno4.streamlit.app/

## What it does

- **Sentiment Analyzer** — Paste any movie review and the AI classifies it as positive or negative with confidence score
- **Movie Recommender** — Search any movie and get similar recommendations based on genre and ratings
- **Data Explorer** — Interactive charts on 9,700+ movies from the MovieLens dataset

## ML Models

- Sentiment Analysis: Logistic Regression + TF-IDF vectorizer trained on 25,000 IMDB reviews — **89.76% accuracy**
- Recommendation Engine: Content-based filtering using TF-IDF + cosine similarity

## Tech Stack

- Python, Scikit-learn, Pandas, NumPy
- Streamlit, Plotly
- Datasets: IMDB 50k reviews, MovieLens latest-small

## Run locally

```bash
git clone https://github.com/michael-unova/cineinsight
cd cineinsight
pip install -r requirements.txt
python sentiment_model.py
streamlit run app.py
```

## Author

Rovaniaina Michael Ratsimbazafy — Microsoft Elevate Intern
