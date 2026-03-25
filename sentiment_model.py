import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

def load_imdb_from_folders(path="data/aclImdb/train"):
    reviews, labels = [], []
    for label in ['pos', 'neg']:
        folder = os.path.join(path, label)
        for fname in os.listdir(folder):
            if fname.endswith('.txt'):
                with open(os.path.join(folder, fname), encoding='utf-8') as f:
                    reviews.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    return pd.DataFrame({'text': reviews, 'label': labels})

def train_sentiment_model():
    print("Loading IMDB data...")
    df = load_imdb_from_folders("data/aclImdb/train")
    print(f"Loaded {len(df)} reviews")

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42)

    print("Training model...")
    vectorizer = TfidfVectorizer(max_features=50000, stop_words='english', ngram_range=(1,3))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=5.0)
    model.fit(X_train_tfidf, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_tfidf))
    print(f"Accuracy: {acc:.2%}")

    os.makedirs("models", exist_ok=True)
    with open("models/sentiment_model.pkl", "wb") as f: pickle.dump(model, f)
    with open("models/tfidf_vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)
    print("Model saved to models/")

def predict_sentiment(text, model=None, vectorizer=None):
    if model is None:
        with open("models/sentiment_model.pkl", "rb") as f: model = pickle.load(f)
    if vectorizer is None:
        with open("models/tfidf_vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    conf = model.predict_proba(vec)[0][pred]
    return ("Positive" if pred == 1 else "Negative"), conf

if __name__ == "__main__":
    train_sentiment_model()