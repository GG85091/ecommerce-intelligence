# E-commerce Intelligence — ML Analytics Platform

**Live Demo:** [link]  
**Tech Stack:** Python · Streamlit · scikit-learn · NLP · Machine Learning

---

## Overview

Two-module ML platform for e-commerce analytics:

- **Product Recommendations** — Content-based filtering using TF-IDF vectorization and Cosine Similarity. Analyzes product descriptions to find the most similar items.
- **Sentiment Analysis** — NLP pipeline using Naive Bayes classifier to categorize customer reviews as Positive / Negative / Neutral with confidence scores.

---

## Modules

### 🎯 Recommendations (Cosine Similarity)

- TF-IDF vectorization of product descriptions
- Cosine similarity matrix (200×200)
- Top-N similar products with similarity scores
- Category-based filtering and top-rated products

### 💬 Sentiment Analysis (Naive Bayes NLP)

- CountVectorizer with 1000 features
- MultinomialNB classifier (3 classes)
- Real-time text classification with confidence %
- Dataset statistics and model accuracy report

---

## Dataset

Synthetic retail dataset: **200 products × 8 categories**, **600 customer reviews**

Categories: Electronics, Clothing, Books, Home & Garden, Sports, Beauty, Food, Toys

---

## Run Locally

```bash
pip install -r requirements.txt
python data/generate_data.py
streamlit run app.py
```

---

## Use Cases

- E-commerce product recommendation engines
- Customer review monitoring and brand sentiment tracking
- Retail analytics dashboards
