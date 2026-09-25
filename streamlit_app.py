"""
FARFETCH Luxury Fashion Recommendation System
Streamlit Web App

Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from recommender import FarfetchRecommender

st.set_page_config(
    page_title="FARFETCH Recommender | BICT242",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.1rem; font-weight: 700; color: #111; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #666; margin-bottom: 1.5rem; }
    .stButton>button {
        width: 100%; background-color: #000; color: white; border-radius: 6px;
        font-weight: 600; border: none;
    }
    .stButton>button:hover { background-color: #333; color: white; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training FARFETCH recommendation models… (~15–20 sec)")
def load_recommender():
    rec = FarfetchRecommender()
    rec.load_data()
    rec.preprocess(min_interactions=3)
    rec.train_collaborative()
    rec.train_matrix_factorization(n_factors=15, n_epochs=18)
    return rec


with st.sidebar:
    st.markdown("### 🛍️ FARFETCH")
    st.caption("Luxury Fashion Recommender")
    st.markdown("---")

    rec = load_recommender()

    all_users = sorted(rec.user_id_to_idx.keys())
    selected_user = st.selectbox("Select Customer ID", options=all_users, index=0)

    method = st.selectbox(
        "Method",
        options=["hybrid", "mf", "user", "item"],
        format_func=lambda x: {
            "hybrid": "Hybrid (Recommended)",
            "mf": "Matrix Factorization",
            "user": "User-based CF",
            "item": "Item-based CF"
        }[x]
    )

    n_recs = st.slider("Number of recommendations", 3, 15, 8)

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    Personalised product recommendations for FARFETCH-style luxury fashion.

    **Algorithms**
    - User-based CF  
    - Item-based CF  
    - Matrix Factorization  
    - Hybrid ensemble  

    BICT242 Course Project
    """)


# Main
st.markdown('<p class="main-header">🛍️ FARFETCH Product Recommender</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Luxury fashion recommendations powered by hybrid collaborative filtering</p>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(rec.user_id_to_idx):,}")
c2.metric("Products", f"{len(rec.product_id_to_idx):,}")
c3.metric("Interactions", f"{len(rec.interactions):,}")
density = np.count_nonzero(rec.user_item_matrix) / rec.user_item_matrix.size
c4.metric("Density", f"{density:.2%}")

st.markdown("---")

# User info
user_row = rec.users[rec.users["user_id"] == selected_user]
if not user_row.empty:
    u = user_row.iloc[0]
    st.subheader(f"Recommendations for **{selected_user}**")
    cols = st.columns(4)
    cols[0].markdown(f"**Age:** {u['age']}")
    cols[1].markdown(f"**Gender:** {u['gender']}")
    cols[2].markdown(f"**Location:** {u['location']}")
    cols[3].markdown(f"**Styles:** {u['preferred_styles']}")

# Recommendations
with st.spinner("Generating recommendations…"):
    recs = rec.recommend(selected_user, n=n_recs, method=method)

if recs:
    df = pd.DataFrame(recs)
    df = df.rename(columns={
        "product_id": "Product ID",
        "name": "Name",
        "brand": "Brand",
        "category": "Category",
        "price": "Price ($)",
        "style": "Style",
        "predicted_rating": "Predicted Score"
    })

    def color_score(val):
        if val >= 4.0:
            return "background-color: #dcfce7"
        elif val >= 3.5:
            return "background-color: #fef9c3"
        return "background-color: #fee2e2"

    styled = df.style.map(color_score, subset=["Predicted Score"])\
                     .format({"Predicted Score": "{:.2f}", "Price ($)": "{:.0f}"})

    st.dataframe(styled, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv,
                       file_name=f"farfetch_recs_{selected_user}.csv", mime="text/csv")
else:
    st.warning("No recommendations available.")

with st.expander("How it works"):
    st.markdown("""
    This system uses **collaborative filtering** and **matrix factorization** to recommend 
    luxury fashion products similar to those preferred by customers with comparable taste.

    - **User-based CF**: finds customers with similar purchase/rating patterns  
    - **Item-based CF**: finds products similar to ones the customer already liked  
    - **Matrix Factorization**: learns hidden taste factors  
    - **Hybrid**: combines all three for best results  
    """)

st.markdown("---")
st.caption("BICT242 · Data Scalability and Analytics · FARFETCH Luxury Fashion Recommendation System")
