from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Journal Explorer", layout="wide")

BASE_DIR = Path("/Users/huss/data-processing/Github/Feminist-Literature-Engine-ML-Web-Scraping-App-/2_notebooks/1_1_25k_pkl")

FILES = {
    "journals_display": BASE_DIR / "journals_display.csv",
    "journals_display2": BASE_DIR / "journals_display2.csv",
    "journals_model_final": BASE_DIR / "journals_model_final.csv",
    "tfidf": BASE_DIR / "tfidf_vectorizer.pkl",
    "pca": BASE_DIR / "pca_model.pkl",
    "scaler": BASE_DIR / "scaler.pkl",
    "mlb": BASE_DIR / "mlb.pkl",
}

st.title("Journal Explorer")
st.write("Explore journals by year, citations, and semantic similarity.")

required = ["journals_display", "journals_model_final", "tfidf", "pca", "scaler", "mlb"]
missing = [name for name in required if not FILES[name].exists()]
if missing:
    st.error("Missing required files:")
    for name in missing:
        st.write(f"- {name}: {FILES[name]}")
    st.stop()

@st.cache_data
def load_data():
    journals_display = pd.read_csv(FILES["journals_display"], index_col=0)
    journals_display2 = pd.read_csv(FILES["journals_display2"], index_col=0) if FILES["journals_display2"].exists() else None
    journals_model_final = pd.read_csv(FILES["journals_model_final"], index_col=0)
    return journals_display, journals_display2, journals_model_final

@st.cache_resource
def load_models():
    tfidf = joblib.load(FILES["tfidf"])
    pca = joblib.load(FILES["pca"])
    scaler = joblib.load(FILES["scaler"])
    mlb = joblib.load(FILES["mlb"])
    return tfidf, pca, scaler, mlb

journals_display, journals_display2, journals_model_final = load_data()
tfidf, pca, scaler, mlb = load_models()

df = journals_display.copy()

for col in ["year", "citations"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "year" not in df.columns:
    st.error("Missing year column in journals_display.csv")
    st.stop()

df_years = df.dropna(subset=["year"]).copy()
if df_years.empty:
    st.error("No valid year values found.")
    st.stop()

df_years["year"] = df_years["year"].astype(int)
year_min = int(df_years["year"].min())
year_max = int(df_years["year"].max())

st.sidebar.header("Filters")
selected_years = st.sidebar.slider(
    "Select year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1,
)

search_text = st.sidebar.text_input("Search text")
show_open_access = st.sidebar.checkbox("Only open access", value=False)

min_citations = 0
if "citations" in df.columns:
    max_citations = int(df["citations"].fillna(0).max())
    min_citations = st.sidebar.slider("Minimum citations", 0, max_citations, 0)

filtered = df.copy()
filtered = filtered[filtered["year"].between(selected_years[0], selected_years[1], inclusive="both")]

if show_open_access and "openaccess" in filtered.columns:
    filtered = filtered[filtered["openaccess"] == 1]

if "citations" in filtered.columns:
    filtered = filtered[filtered["citations"].fillna(0) >= min_citations]

if search_text:
    text_cols = filtered.select_dtypes(include=["object"]).columns.tolist()
    if text_cols:
        mask = pd.Series(False, index=filtered.index)
        for col in text_cols:
            mask = mask | filtered[col].astype(str).str.contains(search_text, case=False, na=False)
        filtered = filtered[mask]

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Year trend", "Citations", "Results"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Papers shown", len(filtered))
    c2.metric("Year range", f"{selected_years[0]}–{selected_years[1]}")
    if "citations" in filtered.columns and len(filtered) > 0:
        c3.metric("Median citations", f"{filtered['citations'].median():.0f}")
    else:
        c3.metric("Median citations", "n/a")

    yearly = df_years.groupby("year").size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(yearly["year"], yearly["count"], marker="o", linewidth=2, color="#7F77DD")
    ax.fill_between(yearly["year"], yearly["count"], alpha=0.15, color="#7F77DD")
    ax.axvspan(selected_years[0], selected_years[1], color="orange", alpha=0.12)
    ax.set_title("Publications per year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of articles")
    st.pyplot(fig)

with tab2:
    yearly = df_years.groupby("year").size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(yearly["year"], yearly["count"], color="#7F77DD")
    ax.set_title("Publications per year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of articles")
    st.pyplot(fig)
    st.dataframe(yearly, use_container_width=True)

with tab3:
    if "citations" in df.columns:
        citation_series = df["citations"].dropna()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median citations", f"{citation_series.median():.1f}" if len(citation_series) else "0")
        c2.metric("Zero citations", int((citation_series == 0).sum()))
        c3.metric(">100 citations", int((citation_series > 100).sum()))
        c4.metric("Max citations", int(citation_series.max()) if len(citation_series) else 0)

        fig, ax = plt.subplots(figsize=(10, 4))
        citation_series.clip(upper=200).hist(bins=50, color="#1D9E75", edgecolor="white", ax=ax)
        ax.set_title("Citation distribution clipped at 200")
        ax.set_xlabel("Citations")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.scatter(df["year"], df["citations"], alpha=0.35, s=18, color="#4444aa")
        ax2.set_title("Citations by year")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Citations")
        st.pyplot(fig2)
    else:
        st.info("No citations column found.")

with tab4:
    st.subheader("Filtered journals")
    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data",
        data=csv,
        file_name="filtered_journals.csv",
        mime="text/csv",
    )

st.divider()
st.subheader("Model transform test")

query = st.text_input("Enter a title or abstract")
if query:
    q_tfidf = tfidf.transform([query])
    q_pca = pca.transform(q_tfidf.toarray())
    st.write("TF-IDF shape:", q_tfidf.shape)
    st.write("PCA shape:", q_pca.shape)