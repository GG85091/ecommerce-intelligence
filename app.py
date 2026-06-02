import os
if not os.path.exists("data/products.csv") or not os.path.exists("data/reviews.csv"):
    import subprocess
    subprocess.run(["python", "data/generate_data.py"])

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modules.recommender import load_products, build_similarity_matrix, get_recommendations, get_top_rated
from modules.sentiment import load_reviews, train_model, predict_sentiment, get_sentiment_stats

st.set_page_config(
    page_title="E-commerce Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { background-color: #0e1117; }
.stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
.metric-card { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 E-commerce Intelligence")
    st.caption("ML-powered retail analytics")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📦 Product Recommendations", "💬 Sentiment Analysis"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Python & Streamlit")

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def get_products():
    return load_products()

@st.cache_data
def get_reviews():
    return load_reviews()

@st.cache_resource
def get_sim_matrix(df_hash):
    df = get_products()
    return build_similarity_matrix(df)

@st.cache_resource
def get_model():
    df = get_reviews()
    return train_model(df)

df_products = get_products()
df_reviews = get_reviews()
sim_matrix = get_sim_matrix(len(df_products))
model, vectorizer, le, accuracy, report = get_model()

CATEGORIES = ["All"] + sorted(df_products["category"].unique().tolist())

# ── Tab: Recommendations ──────────────────────────────────────────────────────
if page == "📦 Product Recommendations":
    st.header("🎯 Product Recommendations")
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Find Similar Products")
        product_input = st.text_input("Enter product name", placeholder="e.g. Wireless Headphones")
        top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
        search_btn = st.button("Get Recommendations", type="primary")

        st.divider()
        category_filter = st.selectbox("Category filter", CATEGORIES)
        top_rated_btn = st.button("Show Top Rated")

    with right:
        if search_btn and product_input:
            selected, recs = get_recommendations(product_input, df_products, sim_matrix, top_n)
            if selected is None:
                st.warning(f"No products found matching '{product_input}'.")
            else:
                st.subheader("Selected Product")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Name", selected["name"][:20] + ("…" if len(selected["name"]) > 20 else ""))
                c2.metric("Category", selected["category"])
                c3.metric("Price", f"${selected['price']:.2f}")
                c4.metric("Rating", f"{selected['rating']} ⭐")

                st.subheader(f"Top {top_n} Similar Products")
                st.dataframe(
                    recs.style.background_gradient(subset=["similarity_score"], cmap="Blues"),
                    use_container_width=True,
                )

                # Bar chart — top 10 by rating in same category
                cat_df = df_products[df_products["category"] == selected["category"]].nlargest(10, "rating")
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor("#1a1f2e")
                ax.set_facecolor("#1a1f2e")
                bars = ax.barh(cat_df["name"].str[:25], cat_df["rating"], color="#00c2ff")
                ax.set_xlabel("Rating", color="white")
                ax.set_title(f"Top 10 in {selected['category']}", color="white")
                ax.tick_params(colors="white")
                ax.spines[:].set_color("#2d3748")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        elif top_rated_btn:
            st.subheader(f"Top Rated — {category_filter}")
            top_df = get_top_rated(df_products, category_filter, top_n=10)
            st.dataframe(top_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("#1a1f2e")
            ax.set_facecolor("#1a1f2e")
            ax.barh(top_df["name"].str[:25], top_df["rating"], color="#00c2ff")
            ax.set_xlabel("Rating", color="white")
            ax.set_title("Top 10 Products by Rating", color="white")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#2d3748")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Enter a product name and click **Get Recommendations**, or choose a category and click **Show Top Rated**.")

# ── Tab: Sentiment Analysis ───────────────────────────────────────────────────
else:
    st.header("💬 Sentiment Analysis")
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Analyze Review Sentiment")
        review_text = st.text_area(
            "Enter review text",
            height=150,
            placeholder="e.g. This product is amazing! Highly recommend it to everyone.",
        )
        analyze_btn = st.button("Analyze Sentiment", type="primary")

        if analyze_btn and review_text.strip():
            label, confidence, all_probas = predict_sentiment(review_text, model, vectorizer, le)
            color_map = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
            icon = color_map.get(label, "⚪")
            st.markdown(f"### {icon} {label.capitalize()}")
            st.markdown(f"**Confidence: {confidence}%**")
            st.divider()
            for sent_label, prob in sorted(all_probas.items()):
                st.text(f"{sent_label.capitalize()}: {prob}%")
                st.progress(int(prob))
        elif analyze_btn:
            st.warning("Please enter some review text first.")

    with right:
        st.subheader("Dataset Statistics")
        stats = get_sentiment_stats(df_reviews)
        total = len(df_reviews)
        most_common = max(stats, key=stats.get)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews", total)
        c2.metric("Model Accuracy", f"{accuracy * 100:.1f}%")
        c3.metric("Most Common", most_common.capitalize())

        # Pie chart
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#1a1f2e")
        ax.set_facecolor("#1a1f2e")
        labels = list(stats.keys())
        sizes = list(stats.values())
        colors = {"positive": "#4caf50", "negative": "#f44336", "neutral": "#ffeb3b"}
        pie_colors = [colors.get(l, "#9e9e9e") for l in labels]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%", startangle=140
        )
        for t in texts + autotexts:
            t.set_color("white")
        ax.set_title("Sentiment Distribution", color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Sample Reviews")
        st.dataframe(
            df_reviews[["review_text", "sentiment"]].tail(10).reset_index(drop=True),
            use_container_width=True,
        )

        with st.expander("Model Classification Report"):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
