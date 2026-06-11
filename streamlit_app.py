# =============================================================================
# 🌸 FEMINIST SCIENTIFIC PAPER RECOMMENDER
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
import base64

# =============================================================================
# PAGE CONFIGURATION (FIXED)
# =============================================================================
st.set_page_config(
    page_title="🌸 Journal Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# FILE PATHS CONFIGURATION (FIXED FOR STREAMLIT CLOUD)
# =============================================================================
# Use relative path - files must be in data/ folder in your GitHub repo
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
# LOAD DATA (FIXED ERROR HANDLING)
# =============================================================================
@st.cache_data
def load_data():
    """Load and cache journal data"""
    try:
        journals_display = pd.read_csv(FILES["journals_display"], index_col=0)
        journals_display2 = pd.read_csv(FILES["journals_display2"], index_col=0) if FILES["journals_display2"].exists() else None
        journals_model_final = pd.read_csv(FILES["journals_model_final"], index_col=0)
        return journals_display, journals_display2, journals_model_final
    except FileNotFoundError as e:
        st.error(f"❌ Data files not found! Please upload files to data/ folder in GitHub:")
        st.error(f"Error: {e}")
        st.info("📁 Files needed:")
        for name, path in FILES.items():
            st.write(f"  - {path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading  {e}")
        st.stop()

@st.cache_resource
def load_models():
    """Load and cache ML models"""
    try:
        tfidf = joblib.load(FILES["tfidf"])
        pca = joblib.load(FILES["pca"])
        scaler = joblib.load(FILES["scaler"])
        mlb = joblib.load(FILES["mlb"])
        return tfidf, pca, scaler, mlb
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Please upload .pkl files to data/ folder:")
        st.error(f"Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Title (simplified)
    st.title("📚 Journal Explorer")
    st.write("Explore journals by year, citations, and semantic similarity.")
    
    st.divider()
    
    # Load data
    journals_display, journals_display2, journals_model_final = load_data()
    
    if journals_display is None:
        st.stop()
    
    # Load models
    tfidf, pca, scaler, mlb = load_models()
    
    if tfidf is None:
        st.stop()
    
    # Rest of your code continues...
    # ... (keep all the rest of your original code from the merged script)
