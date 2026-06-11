# =============================================================================
# 🌸 FEMINIST SCIENTIFIC PAPER RECOMMENDER (Journal Explorer)
# =============================================================================
# A modern Streamlit app exploring 25,000 feminist scientific papers (1814-2026)
# Compatible with Streamlit Community Cloud
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import joblib
from wordcloud import WordCloud

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="📚 Journal Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# FEMINIST COLOR PALETTE
# =============================================================================
FEMINIST_COLORS = {
    'primary': '#9B59B6',      # Purple
    'secondary': '#E91E63',    # Pink
    'success': '#16A085',      # Teal
    'warning': '#E67E22',      # Orange
    'info': '#3498DB',         # Blue
    'danger': '#E74C3C',       # Red
}

PASTEL_COLORS = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', 
                 '#D4A5A5', '#9B59B6', '#E91E63', '#16A085', '#E67E22']

# =============================================================================
# FEMINIST HISTORY CONTEXT
# =============================================================================
FEMINIST_PERIODS = {
    "All Periods": (1814, 2026),
    "First Wave (Suffrage 1848-1920)": (1848, 1920),
    "Second Wave (Workplace 1963-1980)": (1963, 1980),
    "Third Wave (Intersectionality 1990-2012)": (1990, 2012),
    "Fourth Wave (Digital 2012-2026)": (2012, 2026),
}

FEMINIST_CONTEXT = {
    "First Wave (Suffrage 1848-1920)": """
    **First Wave Feminism (1848-1920)**: Focused on women's suffrage and legal rights.
    
    Key events: 1848 Seneca Falls Convention, 1920 19th Amendment (voting rights).
    """,
    
    "Second Wave (Workplace 1963-1980)": """
    **Second Wave Feminism (1963-1980)**: Workplace discrimination, reproductive rights.
    
    Key events: 1963 Equal Pay Act, 1973 Roe v. Wade, Betty Friedan's *The Feminine Mystique*.
    """,
    
    "Third Wave (Intersectionality 1990-2012)": """
    **Third Wave Feminism (1990-2012)**: Intersectionality, diversity, LGBTQ+ rights.
    
    Key concepts: Kimberlé Crenshaw's intersectionality, Judith Butler's gender theory.
    """,
    
    "Fourth Wave (Digital 2012-2026)": """
    **Fourth Wave Feminism (2012-2026)**: Digital activism, #MeToo movement.
    
    Key features: Social media mobilization, systemic sexism focus, global equality fight.
    """
}

# =============================================================================
# FILE PATHS (FIXED FOR STREAMLIT CLOUD - RELATIVE PATHS)
# =============================================================================
BASE_DIR = Path("data")

FILES = {
    "journals_display": BASE_DIR / "journals_display.csv",
    "journals_display2": BASE_DIR / "journals_display2.csv",
    "journals_model_final": BASE_DIR / "journals_model_final.csv",
    "tfidf": BASE_DIR / "tfidf_vectorizer.pkl",
    "pca": BASE_DIR / "pca_model.pkl",
    "scaler": BASE_DIR / "scaler.pkl",
    "mlb": BASE_DIR / "mlb.pkl",
}

# =============================================================================
# DATA LOADING (CACHED)
# =============================================================================
@st.cache_data
def load_data():
    """Load journal data with error handling"""
    try:
        journals_display = pd.read_csv(FILES["journals_display"], index_col=0)
        journals_display2 = None
        if FILES["journals_display2"].exists():
            journals_display2 = pd.read_csv(FILES["journals_display2"], index_col=0)
        journals_model_final = pd.read_csv(FILES["journals_model_final"], index_col=0)
        return journals_display, journals_display2, journals_model_final
    except FileNotFoundError as e:
        st.error("❌ Data files not found in 'data/' folder!")
        st.error(f"Error: {e}")
        st.info("📁 Required files:")
        for name, path in FILES.items():
            if path.suffix == '.csv':
                st.write(f"  - {path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading  {e}")
        st.stop()

@st.cache_resource
def load_models():
    """Load ML models with error handling"""
    try:
        tfidf = joblib.load(FILES["tfidf"])
        pca = joblib.load(FILES["pca"])
        scaler = joblib.load(FILES["scaler"])
        mlb = joblib.load(FILES["mlb"])
        return tfidf, pca, scaler, mlb
    except FileNotFoundError as e:
        st.error("❌ Model files not found in 'data/' folder!")
        st.error(f"Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Load data
    journals_display, journals_display2, journals_model_final = load_data()
    if journals_display is None:
        st.stop()
    
    # Load models
    tfidf, pca, scaler, mlb = load_models()
    if tfidf is None:
        st.stop()
    
    # Initialize dataframe
    df = journals_display.copy()
    
    # Data preprocessing
    for col in ["year", "citations"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if "year" not in df.columns:
        st.error("Missing 'year' column in journals_display.csv")
        st.stop()
    
    df_years = df.dropna(subset=["year"]).copy()
    if df_years.empty:
        st.error("No valid year values found")
        st.stop()
    
    df_years["year"] = df_years["year"].astype(int)
    year_min = int(df_years["year"].min())
    year_max = int(df_years["year"].max())
    
    # =============================================================================
    # PAGE HEADER
    # =============================================================================
    st.title("📚 Journal Explorer")
    st.markdown("""
    **Explore journals by year, citations, and semantic similarity.**
    
    This app connects feminist scholarship with feminist history, making academic research 
    accessible while contextualizing it within the broader movement for gender equality.
    """)
    
    st.divider()
    
    # =============================================================================
    # SIDEBAR FILTERS
    # =============================================================================
    with st.sidebar:
        st.header("🔍 Filters")
        
        # 1. Feminist Period Preset
        st.subheader("📜 Feminist History Period")
        preset_period = st.selectbox(
            "Quick select:",
            options=["None - Use slider"] + list(FEMINIST_PERIODS.keys()),
            index=0
        )
        
        if preset_period != "None - Use slider":
            year_range = FEMINIST_PERIODS[preset_period]
            if preset_period in FEMINIST_CONTEXT:
                st.info(FEMINIST_CONTEXT[preset_period])
        else:
            year_range = st.slider(
                "Year range:",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
                step=1
            )
        
        # 2. Open Access Filter
        st.subheader("🔓 Access")
        open_access_only = st.checkbox("Only Open Access ✓", value=False)
        
        # 3. Citation Filter
        st.subheader("📊 Citations")
        min_citations = 0
        if "citations" in df.columns:
            max_citations = int(df["citations"].fillna(0).max())
            min_citations = st.slider("Minimum citations", 0, max_citations, 0)
        
        # 4. Search
        st.subheader("🔎 Search")
        search_text = st.text_input("Search text")
        
        st.divider()
        st.info("💡 Tip: Use feminist period presets to see scholarship evolution!")
    
    # =============================================================================
    # FILTERING
    # =============================================================================
    filtered = df.copy()
    filtered = filtered[filtered["year"].between(year_range[0], year_range[1], inclusive="both")]
    
    if open_access_only and "openaccess" in filtered.columns:
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
    
    # =============================================================================
    # TOP METRICS
    # =============================================================================
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("Papers shown", f"{len(filtered)}")
    c2.metric("Year range", f"{year_range[0]}–{year_range[1]}")
    
    if "citations" in filtered.columns and len(filtered) > 0:
        c3.metric("Median citations", f"{filtered['citations'].median():.0f}")
        c4.metric("Open Access %", 
                 f"{(filtered['openaccess'] == 1).sum() / len(filtered) * 100:.1f}%")
    
    st.divider()
    
    # =============================================================================
    # MAIN TABS
    # =============================================================================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🌊 WordCloud", "📈 Trends", "📚 Results"])
    
    # =============================================================================
    # TAB 1: OVERVIEW
    # =============================================================================
    with tab1:
        st.subheader("📊 Publication Overview")
        
        c1, c2 = st.columns(2)
        
        with c1:
            # Year trend line plot
            yearly = df_years.groupby("year").size().reset_index(name="count")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(yearly["year"], yearly["count"], marker="o", linewidth=2, 
                   color=FEMINIST_COLORS['primary'])
            ax.fill_between(yearly["year"], yearly["count"], alpha=0.15, 
                           color=FEMINIST_COLORS['primary'])
            ax.axvspan(year_range[0], year_range[1], color="orange", alpha=0.12)
            ax.set_title("Publications per Year (1814-2026)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of articles")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with c2:
            # Citation distribution
            if "citations" in df.columns:
                citation_series = df["citations"].dropna()
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Median", f"{citation_series.median():.1f}" if len(citation_series) else "0")
                c2.metric("Zero citations", int((citation_series == 0).sum()))
                c3.metric(">100 citations", int((citation_series > 100).sum()))
                c4.metric("Max", int(citation_series.max()) if len(citation_series) else 0)
                
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                citation_series.clip(upper=200).hist(bins=50, color=FEMINIST_COLORS['success'], 
                                                    edgecolor="white", ax=ax2)
                ax2.set_title("Citation distribution (clipped at 200)")
                ax2.set_xlabel("Citations")
                ax2.set_ylabel("Count")
                st.pyplot(fig2)
        
        st.divider()
        
        # Scatter plot: Citations by Year
        if "citations" in df.columns:
            st.subheader("🔍 Citations by Year: Open vs Closed Access")
            
            fig3, ax3 = plt.subplots(figsize=(14, 6))
            
            open_access = df[df['openaccess'] == 1]
            not_open = df[df['openaccess'] != 1]
            
            ax3.scatter(open_access['year'], open_access['citations'], 
                      alpha=0.4, s=30, color=FEMINIST_COLORS['success'], 
                      label='Open Access ✓', edgecolors='white')
            ax3.scatter(not_open['year'], not_open['citations'], 
                      alpha=0.4, s=30, color=FEMINIST_COLORS['danger'], 
                      label='Closed Access ✗', edgecolors='white')
            
            ax3.set_title("Citations by Year: Access Gap")
            ax3.set_xlabel("Year")
            ax3.set_ylabel("Citations")
            ax3.legend()
            ax3.grid(alpha=0.3)
            
            st.pyplot(fig3)
    
    # =============================================================================
    # TAB 2: WORDCLOUD
    # =============================================================================
    with tab2:
        st.subheader("🌊 WordCloud: Title Categories")
        
        st.markdown("""
        **This word cloud shows the most frequent title categories.**
        
        Use the search below to filter papers containing a specific term!
        """)
        
        if 'title_category' in filtered_df.columns if 'title_category' in filtered.columns else False:
            # Create word frequency dictionary
            word_freq = filtered['title_category'].value_counts().to_dict()
            
            # Create word cloud
            wc = WordCloud(width=1000, height=400, background_color='white', 
                          colormap='viridis', min_font_size=10)
            wc.generate_from_frequencies(word_freq)
            
            # Display word cloud
            st.image(wc.to_array(), use_container_width=True)
            
            # Word search
            selected_word = st.text_input(
                "🔍 Search for papers containing this term:",
                placeholder="Type a word from the word cloud..."
            )
            
            if selected_word:
                word_papers = filtered[filtered['title_category'].str.contains(
                    selected_word, case=False, na=False)]
                
                num_papers = st.slider(
                    f"Number of papers (10-{len(word_papers)}):",
                    min_value=10,
                    max_value=len(word_papers) if len(word_papers) > 0 else 10,
                    value=15,
                    step=5
                )
                
                st.success(f"Found {len(word_papers)} papers containing '{selected_word}'")
                
                display_papers = word_papers.head(num_papers)
                
                for idx, paper in display_papers.iterrows():
                    with st.container():
                        access_badge = "✓ Open Access" if paper['openaccess'] == 1 else "✗ Closed Access"
                        st.markdown(f"""
                        **{paper['title']}** [{paper['year']}]
                        
                        **Author(s)**: {paper['authors']}
                        
                        **Citations**: {paper['citations']} | **Domain**: {paper['domain']}
                        
                        {access_badge}
                        
                        ---
                        """)
                
                if st.button(f"📚 Show all {len(word_papers)} papers"):
                    st.dataframe(word_papers, use_container_width=True)
        else:
            st.info("title_category column not found in dataset")
    
    # =============================================================================
    # TAB 3: TRENDS
    # =============================================================================
    with tab3:
        st.subheader("📈 Topic Trends: Last 20 Years")
        
        st.markdown("""
        **Interactive chart showing topic frequency changes (2006-2026).**
        
        Click legend items to toggle topics. Hover for exact counts.
        """)
        
        # Filter last 20 years
        last_20_years = filtered[filtered['year'] >= 2006]
        
        if 'topics' in last_20_years.columns and len(last_20_years) > 0:
            topic_trends = last_20_years.groupby(['year', 'topics']).size().reset_index(name='count')
            top_topics = topic_trends['topics'].value_counts().head(10).index.tolist()
            topic_trends = topic_trends[topic_trends['topics'].isin(top_topics)]
            
            # Create interactive plotly chart
            fig_trends = go.Figure()
            
            for i, topic in enumerate(top_topics):
                topic_data = topic_trends[topic_trends['topics'] == topic]
                fig_trends.add_trace(go.Scatter(
                    x=topic_data['year'],
                    y=topic_data['count'],
                    name=topic,
                    line=dict(color=PASTEL_COLORS[i % len(PASTEL_COLORS)], width=3),
                    mode='lines+markers',
                    marker=dict(size=6)
                ))
            
            fig_trends.update_layout(
                title="Topic Frequency Trends (2006-2026)",
                xaxis_title="Year",
                yaxis_title="Frequency",
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white",
                height=600
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
            
            st.divider()
            
            st.info("""
            **Feminist Context**: Fourth wave (2012-2026) features digital activism, #MeToo movement,
            systemic sexism focus, and intersectional approach.
            """)
        else:
            st.info("topics column not found or insufficient data")
    
    # =============================================================================
    # TAB 4: RESULTS
    # =============================================================================
    with tab4:
        st.subheader("📚 Filtered Papers")
        
        papers_per_page = st.slider(
            "Papers to display:",
            min_value=10,
            max_value=100,
            value=15,
            step=5
        )
        
        sort_by = st.selectbox(
            "Sort by:",
            options=["Most Citations", "Newest Year", "Oldest Year", "Title (A-Z)"],
            index=0
        )
        
        if sort_by == "Most Citations":
            sorted_df = filtered.sort_values('citations', ascending=False)
        elif sort_by == "Newest Year":
            sorted_df = filtered.sort_values('year', ascending=False)
        elif sort_by == "Oldest Year":
            sorted_df = filtered.sort_values('year', ascending=True)
        elif sort_by == "Title (A-Z)":
            sorted_df = filtered.sort_values('title', ascending=True)
        
        display_df = sorted_df.head(papers_per_page)
        
        if len(display_df) > 0:
            for idx, paper in display_df.iterrows():
                with st.container():
                    access_badge = "✓ Open Access" if paper['openaccess'] == 1 else "✗ Closed Access"
                    st.markdown(f"""
                    **{paper['title']}** [{paper['year']}]
                    
                    **Author(s)**: {paper['authors']}
                    
                    **Citations**: {paper['citations']} | **Domain**: {paper['domain']} | **Field**: {paper['field']}
                    
                    **Language**: {paper['languages']} | **Type**: {paper['worktype']}
                    
                    {access_badge}
                    
                    ---
                    """)
            
            st.divider()
            
            csv_data = filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Papers (CSV)",
                data=csv_data,
                file_name=f"feminist_papers_{year_range[0]}_{year_range[1]}.csv",
                mime="text/csv"
            )
        else:
            st.info("No papers match filters. Try adjusting year range or citation minimum.")
    
    # =============================================================================
    # MODEL TRANSFORM TEST
    # =============================================================================
    st.divider()
    st.subheader("🔬 Semantic Similarity Search")
    
    query = st.text_input("Enter a title or abstract")
    if query and tfidf is not None:
        try:
            q_tfidf = tfidf.transform([query])
            q_pca = pca.transform(q_tfidf.toarray())
            
            st.write("TF-IDF shape:", q_tfidf.shape)
            st.write("PCA shape:", q_pca.shape)
            
            if journals_model_final is not None:
                distances = np.linalg.norm(journals_model_final.values - q_pca, axis=1)
                top_indices = np.argsort(distances)[:5]
                
                st.subheader("📚 Most Similar Papers")
                for idx in top_indices:
                    similar_paper = journals_model_final.iloc[idx]
                    st.write(f"- {similar_paper}")
        except Exception as e:
            st.error(f"Error: {e}")

# =============================================================================
# RUN APP
# =============================================================================
if __name__ == "__main__":
    main()
