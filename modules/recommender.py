import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_DATA_DIR = Path(__file__).parent.parent / "data"


def load_products():
    return pd.read_csv(_DATA_DIR / "products.csv")


def build_similarity_matrix(df):
    df = df.copy()
    df["combined"] = df["name"] + " " + df["category"] + " " + df["description"]
    tfidf = TfidfVectorizer(stop_words="english", max_features=500)
    matrix = tfidf.fit_transform(df["combined"])
    return cosine_similarity(matrix)


def get_recommendations(product_name, df, sim_matrix, top_n=5):
    matches = df[df["name"].str.lower().str.contains(product_name.lower())]
    if matches.empty:
        return None, None
    idx = matches.index[0]
    selected = df.iloc[idx]
    scores = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)
    scores = [(i, s) for i, s in scores if i != idx][:top_n]
    rec_indices = [i for i, _ in scores]
    rec_scores = [round(s, 3) for _, s in scores]
    result = df.iloc[rec_indices].copy()
    result["similarity_score"] = rec_scores
    return selected, result[["name", "category", "price", "rating", "similarity_score"]]


def get_top_rated(df, category=None, top_n=10):
    if category and category != "All":
        df = df[df["category"] == category]
    return df.nlargest(top_n, "rating")[["name", "category", "price", "rating"]]
