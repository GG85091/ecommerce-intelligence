import os
import sys
from pathlib import Path

# Always work relative to this file's directory (critical for Streamlit Cloud)
BASE_DIR = Path(__file__).parent.resolve()
os.chdir(BASE_DIR)

if not (BASE_DIR / "data" / "products.csv").exists() or not (BASE_DIR / "data" / "reviews.csv").exists():
    import subprocess
    subprocess.run(
        [sys.executable, str(BASE_DIR / "data" / "generate_data.py")],
        cwd=str(BASE_DIR),
        check=True,
    )

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modules.recommender import load_products, build_similarity_matrix, get_recommendations, get_top_rated
from modules.sentiment import load_reviews, train_model, predict_sentiment, get_sentiment_stats
from translations import t

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Language switcher — first element
    lang = st.selectbox(
        "🌐 Language / Язык",
        options=["en", "ru"],
        format_func=lambda x: "🇬🇧 English" if x == "en" else "🇷🇺 Русский",
        key="language",
    )

    st.title(f"🛒 {t('app_title', lang)}")
    st.caption(t("app_subtitle", lang))
    st.divider()

    page = st.radio(
        "Navigate",
        [t("tab_recommendations", lang), t("tab_sentiment", lang)],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Python & Streamlit")

# ── Cached loaders ─────────────────────────────────────────────────────────────
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
df_reviews  = get_reviews()
sim_matrix  = get_sim_matrix(len(df_products))
model, vectorizer, le, accuracy, report = get_model()

CATEGORIES = [t("category_all", lang)] + sorted(df_products["category"].unique().tolist())

# ── Tab: Recommendations ───────────────────────────────────────────────────────
if page == t("tab_recommendations", lang):
    st.header(t("tab_recommendations", lang))
    left, right = st.columns([1, 2])

    with left:
        st.subheader(t("find_similar", lang))
        product_input = st.text_input(
            t("enter_product", lang),
            placeholder=t("product_placeholder", lang),
        )
        top_n = st.slider(t("num_recommendations", lang), min_value=3, max_value=10, value=5)
        search_btn = st.button(t("get_recommendations", lang), type="primary")

        st.divider()
        category_filter = st.selectbox(t("category_filter", lang), CATEGORIES)
        top_rated_btn = st.button(t("show_top_rated", lang))

    with right:
        if search_btn and product_input:
            selected, recs = get_recommendations(product_input, df_products, sim_matrix, top_n)
            if selected is None:
                st.warning(t("no_product_found", lang))
            else:
                st.subheader(t("selected_product", lang))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(t("name", lang),     selected["name"][:20] + ("…" if len(selected["name"]) > 20 else ""))
                c2.metric(t("category", lang), selected["category"])
                c3.metric(t("price", lang),    f"${selected['price']:.2f}")
                c4.metric(t("rating", lang),   f"{selected['rating']} ⭐")

                st.subheader(f"Top {top_n} {t('recommendations', lang)}")
                recs_display = recs.rename(columns={"similarity_score": t("similarity_score", lang)})
                st.dataframe(
                    recs_display.style.background_gradient(
                        subset=[t("similarity_score", lang)], cmap="Blues"
                    ),
                    use_container_width=True,
                )

                cat_df = df_products[df_products["category"] == selected["category"]].nlargest(10, "rating")
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor("#1a1f2e")
                ax.set_facecolor("#1a1f2e")
                ax.barh(cat_df["name"].str[:25], cat_df["rating"], color="#00c2ff")
                ax.set_xlabel(t("rating", lang), color="white")
                ax.set_title(f"Top 10 — {selected['category']}", color="white")
                ax.tick_params(colors="white")
                ax.spines[:].set_color("#2d3748")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        elif top_rated_btn:
            cat_label = category_filter if category_filter != t("category_all", lang) else None
            st.subheader(f"{t('top_rated_products', lang)} — {category_filter}")
            top_df = get_top_rated(df_products, cat_label, top_n=10)
            st.dataframe(top_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("#1a1f2e")
            ax.set_facecolor("#1a1f2e")
            ax.barh(top_df["name"].str[:25], top_df["rating"], color="#00c2ff")
            ax.set_xlabel(t("rating", lang), color="white")
            ax.set_title(t("top_rated_products", lang), color="white")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#2d3748")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(
                f"{t('enter_product', lang)} → **{t('get_recommendations', lang)}**  "
                f"| {t('category_filter', lang)} → **{t('show_top_rated', lang)}**"
            )

# ── Tab: Sentiment Analysis ────────────────────────────────────────────────────
else:
    st.header(t("tab_sentiment", lang))
    left, right = st.columns([1, 2])

    sentiment_labels = {
        "positive": t("positive", lang),
        "negative": t("negative", lang),
        "neutral":  t("neutral",  lang),
    }

    with left:
        st.subheader(t("analyze_sentiment", lang))
        review_text = st.text_area(
            t("enter_review", lang),
            height=150,
            placeholder=t("review_placeholder", lang),
        )
        analyze_btn = st.button(t("analyze_btn", lang), type="primary")

        if analyze_btn and review_text.strip():
            label, confidence, all_probas = predict_sentiment(review_text, model, vectorizer, le)
            color_map = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
            icon = color_map.get(label, "⚪")
            localized_label = sentiment_labels.get(label, label.capitalize())
            st.markdown(f"### {icon} {localized_label}")
            st.markdown(f"**{t('confidence', lang)}: {confidence}%**")
            st.divider()
            for sent_key, prob in sorted(all_probas.items()):
                display = sentiment_labels.get(sent_key, sent_key.capitalize())
                st.text(f"{display}: {prob}%")
                st.progress(int(prob))
        elif analyze_btn:
            st.warning(t("enter_review", lang))

    with right:
        st.subheader(t("dataset_stats", lang))
        stats = get_sentiment_stats(df_reviews)
        total = len(df_reviews)
        most_common = max(stats, key=stats.get)

        c1, c2, c3 = st.columns(3)
        c1.metric(t("total_reviews", lang),  total)
        c2.metric(t("model_accuracy", lang), f"{accuracy * 100:.1f}%")
        c3.metric(t("top_sentiment", lang),  sentiment_labels.get(most_common, most_common.capitalize()))

        # Pie chart with localized labels
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#1a1f2e")
        ax.set_facecolor("#1a1f2e")
        pie_labels = [sentiment_labels.get(k, k) for k in stats.keys()]
        sizes      = list(stats.values())
        colors_map = {"positive": "#4caf50", "negative": "#f44336", "neutral": "#ffeb3b"}
        pie_colors = [colors_map.get(k, "#9e9e9e") for k in stats.keys()]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=pie_labels, colors=pie_colors, autopct="%1.1f%%", startangle=140
        )
        for txt in texts + autotexts:
            txt.set_color("white")
        ax.set_title(t("sentiment_distribution", lang), color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader(t("sample_reviews", lang))
        st.dataframe(
            df_reviews[["review_text", "sentiment"]].tail(10).reset_index(drop=True),
            use_container_width=True,
        )

        with st.expander(t("model_info", lang)):
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
