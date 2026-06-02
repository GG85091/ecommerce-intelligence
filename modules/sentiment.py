import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


def load_reviews():
    return pd.read_csv("data/reviews.csv")


def train_model(df):
    le = LabelEncoder()
    y = le.fit_transform(df["sentiment"])
    vectorizer = CountVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(df["review_text"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    return model, vectorizer, le, accuracy, report


def predict_sentiment(text, model, vectorizer, le):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = le.inverse_transform([pred])[0]
    confidence = round(max(proba) * 100, 1)
    all_probas = {le.inverse_transform([i])[0]: round(p * 100, 1) for i, p in enumerate(proba)}
    return label, confidence, all_probas


def get_sentiment_stats(df):
    return df["sentiment"].value_counts().to_dict()
