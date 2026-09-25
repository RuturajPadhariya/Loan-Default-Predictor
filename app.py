"""
Loan Default Prediction — Streamlit Dashboard
===============================================
Industry-grade frontend for `Loan_Default.pkl` (Decision Tree Classifier)
with `Scaler.pkl` (StandardScaler).

Sections
--------
1. 🏠 Dashboard          – KPI cards & dataset overview
2. 🔮 Predict            – Interactive loan-default predictor
3. 📊 Model Performance  – Accuracy, confusion matrix, ROC-AUC, classification report
4. 🧠 How It Works       – Decision Tree explainer + feature importance
5. 📂 Data Explorer      – Browse / filter the raw dataset
6. ℹ️ About              – Project information
"""

# ──────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
)
from sklearn.tree import export_text

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="LoanGuard AI — Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Theme state management
# ──────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

IS_DARK = st.session_state.theme == "dark"

# ──────────────────────────────────────────────
# CSS — Dynamic theme (Light default + Dark toggle)
# ──────────────────────────────────────────────
if IS_DARK:
    css_vars = """
    :root {
        --bg-primary: #0B0F19;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.85);
        --bg-glass: rgba(30, 41, 72, 0.55);
        --bg-input: #1e293b;
        --border-main: rgba(99, 128, 255, 0.12);
        --border-hover: rgba(99, 128, 255, 0.3);
        --accent-primary: #818CF8;
        --accent-secondary: #6366F1;
        --accent-cyan: #22d3ee;
        --accent-emerald: #34d399;
        --accent-rose: #fb7185;
        --accent-amber: #fbbf24;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --shadow-card: 0 4px 24px rgba(0,0,0,0.25);
        --shadow-hover: 0 12px 40px rgba(99,102,241,0.18);
        --gradient-1: linear-gradient(135deg, #6366f1 0%, #818cf8 50%, #a78bfa 100%);
        --gradient-2: linear-gradient(135deg, #06b6d4 0%, #22d3ee 100%);
        --gradient-3: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        --gradient-4: linear-gradient(135deg, #f43f5e 0%, #fb7185 100%);
        --gradient-bg: linear-gradient(135deg, #0B0F19 0%, #111827 100%);
        --navbar-bg: rgba(11, 15, 25, 0.92);
        --plotly-bg: rgba(0,0,0,0);
        --plotly-font: #e2e8f0;
        --plotly-grid: rgba(148,163,184,0.08);
    }
    """
else:
    css_vars = """
    :root {
        --bg-primary: #F8FAFC;
        --bg-secondary: #FFFFFF;
        --bg-card: rgba(255, 255, 255, 0.92);
        --bg-glass: rgba(241, 245, 249, 0.85);
        --bg-input: #F1F5F9;
        --border-main: rgba(148, 163, 184, 0.2);
        --border-hover: rgba(99, 102, 241, 0.35);
        --accent-primary: #6366F1;
        --accent-secondary: #4F46E5;
        --accent-cyan: #0891B2;
        --accent-emerald: #059669;
        --accent-rose: #E11D48;
        --accent-amber: #D97706;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --shadow-card: 0 4px 24px rgba(0,0,0,0.06);
        --shadow-hover: 0 12px 40px rgba(99,102,241,0.12);
        --gradient-1: linear-gradient(135deg, #6366f1 0%, #818cf8 50%, #a78bfa 100%);
        --gradient-2: linear-gradient(135deg, #0891B2 0%, #06B6D4 100%);
        --gradient-3: linear-gradient(135deg, #059669 0%, #10B981 100%);
        --gradient-4: linear-gradient(135deg, #E11D48 0%, #F43F5E 100%);
        --gradient-bg: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%);
        --navbar-bg: rgba(255, 255, 255, 0.92);
        --plotly-bg: rgba(0,0,0,0);
        --plotly-font: #0F172A;
        --plotly-grid: rgba(148,163,184,0.15);
    }
    """

st.markdown(f"""
<style>
/* ── Import Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

{css_vars}

/* ─────────── ANIMATIONS ─────────── */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(24px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes slideDown {{
    from {{ opacity: 0; transform: translateY(-16px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulseGlow {{
    0%, 100% {{ box-shadow: 0 0 20px rgba(99,102,241,0.1); }}
    50% {{ box-shadow: 0 0 40px rgba(99,102,241,0.25); }}
}}
@keyframes shimmer {{
    0% {{ background-position: -200% center; }}
    100% {{ background-position: 200% center; }}
}}
@keyframes scaleIn {{
    from {{ opacity: 0; transform: scale(0.92); }}
    to   {{ opacity: 1; transform: scale(1); }}
}}

/* ─────────── GLOBAL ─────────── */
html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary);
}}

/* Force Streamlit internal theme variables */
.stApp {{
    --primary-color: var(--accent-primary) !important;
    --background-color: var(--bg-primary) !important;
    --secondary-background-color: var(--bg-secondary) !important;
    --text-color: var(--text-primary) !important;
}}

[data-testid="stAppViewContainer"] {{
    background: var(--gradient-bg);
}}

/* All main content text */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] td,
[data-testid="stAppViewContainer"] th,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] div {{
    color: inherit;
}}

/* ── Remove Streamlit top gap ── */
[data-testid="stHeader"] {{ background: transparent !important; height: 0 !important; }}
.stApp > header {{ display: none !important; }}
[data-testid="stAppViewContainer"] > div:first-child {{ padding-top: 0 !important; }}
.block-container {{ padding-top: 1rem !important; }}

/* hide sidebar completely — using top navbar */
[data-testid="stSidebar"] {{ display: none !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
button[kind="header"] {{ display: none !important; }}

/* ─────────── PREMIUM UNIFIED FLOATING NAVBAR ─────────── */
div[data-testid="stAppViewContainer"] .st-key-navbar {{
    background: var(--navbar-bg) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 18px !important;
    padding: 6px 10px !important;
    margin: 0 auto 1.5rem auto !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    position: sticky !important;
    top: 0.5rem !important;
    z-index: 99999 !important;
    animation: slideDown 0.4s ease both !important;
    max-width: 1280px !important;
    width: 100% !important;
    overflow: visible !important;
}}

/* Navbar columns alignment & compact spacing — ALWAYS single row */
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="stHorizontalBlock"] {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: flex-start !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
    width: 100% !important;
    gap: 6px !important;
    padding: 2px 2px !important;
}}
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="stHorizontalBlock"]::-webkit-scrollbar {{
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}}

/* Navbar column: sizes to its natural content without stretching or 100% mobile stacking */
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: max-content !important;
    max-width: none !important;
    padding: 0 !important;
    margin: 0 !important;
}}

/* Inner column wrappers */
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] > div,
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] [data-testid="stVerticalBlock"],
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] [data-testid="element-container"],
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] .stButton,
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] .stMarkdown {{
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: auto !important;
    min-width: max-content !important;
    margin: 0 !important;
    padding: 0 !important;
}}

/* Navbar Brand Icon Badge */
.navbar-brand-icon {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 38px !important;
    height: 38px !important;
    min-width: 38px !important;
    border-radius: 12px !important;
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-main) !important;
    font-size: 1.35rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: default !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    margin: 0 !important;
    flex-shrink: 0 !important;
}}
.navbar-brand-icon:hover {{
    transform: scale(1.08) !important;
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.3) !important;
}}

/* Navbar Buttons — Seamless segmented navigation */
div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button {{
    height: 38px !important;
    min-height: 38px !important;
    padding: 0 12px !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    border-radius: 12px !important;
    letter-spacing: 0.2px !important;
    white-space: nowrap !important;
    width: auto !important;
    min-width: max-content !important;
    flex-shrink: 0 !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 4px !important;
}}

/* Inactive nav button — clean, transparent tab */
div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[kind="secondary"],
div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[data-testid="baseButton-secondary"] {{
    background: transparent !important;
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    transform: none !important;
}}

div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[kind="secondary"]:hover,
div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[data-testid="baseButton-secondary"]:hover {{
    background: var(--bg-glass) !important;
    background-color: var(--bg-glass) !important;
    border-color: var(--border-hover) !important;
    color: var(--accent-primary) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.12) !important;
}}

div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[kind="secondary"] *,
div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[data-testid="baseButton-secondary"] * {{
    color: var(--text-secondary) !important;
    -webkit-text-fill-color: var(--text-secondary) !important;
    font-weight: 500 !important;
}}

div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[kind="secondary"]:hover *,
div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[data-testid="baseButton-secondary"]:hover * {{
    color: var(--accent-primary) !important;
    -webkit-text-fill-color: var(--accent-primary) !important;
}}

/* Active nav button — vibrant indigo pill */
div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[kind="primary"],
div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[data-testid="baseButton-primary"] {{
    background: var(--gradient-1) !important;
    background-color: var(--accent-primary) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
    font-weight: 700 !important;
    transform: translateY(-1px) !important;
}}

div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[kind="primary"]:hover,
div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[data-testid="baseButton-primary"]:hover {{
    box-shadow: 0 6px 22px rgba(99, 102, 241, 0.5) !important;
    color: #FFFFFF !important;
    transform: translateY(-2px) !important;
}}

div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[kind="primary"] *,
div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button[data-testid="baseButton-primary"] * {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
}}

/* Theme toggle inside navbar */
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"]:last-child .stButton > button {{
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03) !important;
}}
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"]:last-child .stButton > button:hover {{
    border-color: var(--accent-primary) !important;
    color: var(--accent-primary) !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.2) !important;
}}
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"]:last-child .stButton > button * {{
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}}
div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"]:last-child .stButton > button:hover * {{
    color: var(--accent-primary) !important;
    -webkit-text-fill-color: var(--accent-primary) !important;
}}

/* ─────────── KPI CARDS ─────────── */
.kpi-card {{
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-main);
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.6s ease both;
    box-shadow: var(--shadow-card);
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}}
.kpi-card:hover {{
    transform: translateY(-6px);
    box-shadow: var(--shadow-hover);
    border-color: var(--border-hover);
}}
.kpi-card .kpi-icon {{ font-size: 2.2rem; margin-bottom: 10px; }}
.kpi-card .kpi-value {{
    font-size: 2.1rem; font-weight: 800;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}}
.kpi-card .kpi-label {{
    font-size: 0.78rem; font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-top: 8px;
}}

/* KPI colour variants */
.kpi-cyan .kpi-value  {{ background: var(--gradient-2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.kpi-cyan::before     {{ background: var(--gradient-2); }}
.kpi-green .kpi-value {{ background: var(--gradient-3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.kpi-green::before    {{ background: var(--gradient-3); }}
.kpi-rose .kpi-value  {{ background: var(--gradient-4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.kpi-rose::before     {{ background: var(--gradient-4); }}
.kpi-purple::before   {{ background: var(--gradient-1); }}

/* Animation delays for staggered entry */
.kpi-delay-1 {{ animation-delay: 0.1s; }}
.kpi-delay-2 {{ animation-delay: 0.2s; }}
.kpi-delay-3 {{ animation-delay: 0.3s; }}
.kpi-delay-4 {{ animation-delay: 0.4s; }}

/* ─────────── SECTION TITLE ─────────── */
.section-title {{
    font-size: 1.45rem; font-weight: 700;
    margin: 2rem 0 1rem;
    display: flex; align-items: center; gap: 10px;
    color: var(--text-primary);
    animation: fadeIn 0.5s ease;
}}

/* ─────────── GLASS PANEL ─────────── */
.glass-panel {{
    background: var(--bg-card);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border-main);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-card);
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease both;
}}
.glass-panel:hover {{
    box-shadow: var(--shadow-hover);
    border-color: var(--border-hover);
}}

/* ─────────── PREDICTION RESULT CARDS ─────────── */
.prediction-safe {{
    background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(52,211,153,0.04) 100%);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 20px; padding: 36px; text-align: center;
    animation: scaleIn 0.5s ease both;
    box-shadow: 0 4px 24px rgba(16,185,129,0.08);
}}
.prediction-risk {{
    background: linear-gradient(135deg, rgba(244,63,94,0.08) 0%, rgba(251,113,133,0.04) 100%);
    border: 1px solid rgba(251,113,133,0.3);
    border-radius: 20px; padding: 36px; text-align: center;
    animation: scaleIn 0.5s ease both;
    box-shadow: 0 4px 24px rgba(244,63,94,0.08);
}}

/* ─────────── PAGE HEADER ─────────── */
.page-header {{
    text-align: center;
    margin: 1.2rem 0 2rem;
    animation: fadeInUp 0.5s ease both;
}}
.page-header h1 {{
    font-size: clamp(1.5rem, 3.2vw + 0.4rem, 2.4rem) !important;
    font-weight: 900;
    margin: 0;
    line-height: 1.25;
}}
.page-header p {{
    color: var(--text-secondary);
    font-size: clamp(0.88rem, 1.2vw + 0.2rem, 1.05rem);
    margin-top: 8px;
}}

/* ─────────── HERO SECTION ─────────── */
.hero-section {{
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    width: 100% !important;
    padding: 1.8rem 1rem 1.4rem !important;
    margin: 0 auto !important;
    animation: fadeInUp 0.6s ease both;
}}
.hero-title {{
    font-size: clamp(1.65rem, 4vw + 0.4rem, 2.8rem) !important;
    font-weight: 900 !important;
    background: var(--gradient-1) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 auto !important;
    text-align: center !important;
    line-height: 1.2 !important;
    width: 100% !important;
}}
.hero-subtitle {{
    color: var(--text-secondary) !important;
    font-size: clamp(0.92rem, 1.5vw + 0.3rem, 1.1rem) !important;
    margin: 14px auto 0 auto !important;
    max-width: 760px !important;
    width: 100% !important;
    line-height: 1.7 !important;
    text-align: center !important;
    display: block !important;
}}
.hero-badges {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 10px !important;
    margin: 20px auto 0 auto !important;
    flex-wrap: wrap !important;
    width: 100% !important;
}}
.hero-badge {{
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 24px !important;
    padding: 6px 16px !important;
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    box-shadow: var(--shadow-card) !important;
    transition: all 0.3s ease !important;
}}
.hero-badge:hover {{
    border-color: var(--border-hover) !important;
    transform: translateY(-2px) !important;
}}

/* ─────────── FORM SECTION ─────────── */
.form-section-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-primary);
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(99,102,241,0.15);
}}

/* ─────────── STREAMLIT WIDGET OVERRIDES ─────────── */

/* === INPUTS & TEXT CONTROLS === */
input,
textarea,
[data-baseweb="input"] input,
[data-testid="stNumberInput-Input"],
[data-testid="stTextInput-Input"],
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {{
    background-color: var(--bg-input) !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    caret-color: var(--accent-primary) !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
}}

[data-baseweb="input"],
[data-baseweb="base-input"] {{
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 10px !important;
}}
[data-baseweb="input"]:focus-within,
[data-baseweb="base-input"]:focus-within {{
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}}

/* Number input step (+ / -) buttons */
.stNumberInput button,
[data-testid="stNumberInput"] button {{
    background-color: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: none !important;
}}
.stNumberInput button svg,
[data-testid="stNumberInput"] button svg {{
    fill: var(--text-primary) !important;
}}

/* Selectbox */
[data-baseweb="select"],
[data-baseweb="select"] > div,
[data-baseweb="select"] [role="combobox"],
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}}
[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-baseweb="select"] p {{
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}}
[data-baseweb="select"] svg {{
    fill: var(--text-secondary) !important;
}}

/* Dropdown popover menu */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[data-baseweb="menu"] > ul {{
    background-color: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 10px !important;
}}
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"] {{
    background-color: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
}}
[data-baseweb="menu"] li span,
[data-baseweb="menu"] [role="option"] span,
[data-baseweb="menu"] li div,
[data-baseweb="menu"] [role="option"] div {{
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [aria-selected="true"] {{
    background-color: var(--bg-glass) !important;
    color: var(--accent-primary) !important;
}}

/* MultiSelect tags (chips) */
[data-baseweb="tag"],
span[data-baseweb="tag"],
div[data-baseweb="tag"] {{
    background-color: var(--bg-glass) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 6px !important;
}}
[data-baseweb="tag"] span,
[data-baseweb="tag"] div {{
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}}
[data-baseweb="tag"] svg,
[data-baseweb="tag"] [role="presentation"] {{
    fill: var(--text-secondary) !important;
}}

/* Sliders */
.stSlider > div > div > div {{ color: var(--text-primary) !important; }}
.stSlider [data-baseweb="slider"] div {{
    color: var(--text-primary) !important;
}}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {{
    color: var(--text-secondary) !important;
}}

/* All input labels */
.stSelectbox label, .stMultiSelect label, .stNumberInput label,
.stTextInput label, .stTextArea label, .stSlider label,
.stRadio label, .stCheckbox label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {{
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}}

/* Code blocks and inline code */
code,
pre,
[data-testid="stMarkdownContainer"] code,
[data-testid="stMarkdownContainer"] pre,
.stMarkdown code,
.stMarkdown pre {{
    background-color: var(--bg-glass) !important;
    color: var(--accent-primary) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 6px !important;
    padding: 3px 8px !important;
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 0.9em !important;
}}

/* === FORM CONTAINER === */
[data-testid="stForm"] {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: var(--shadow-card) !important;
}}

/* === GENERIC CONTAINERS === */
[data-testid="stVerticalBlock"] > div,
[data-testid="stHorizontalBlock"] > div {{
    color: var(--text-primary);
}}
[data-testid="column"] {{
    color: var(--text-primary);
}}

/* === METRICS === */
[data-testid="stMetric"],
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {{
    color: var(--text-primary) !important;
}}

/* Tabs */
button[data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    transition: all 0.25s ease !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--accent-primary) !important;
    border-bottom-color: var(--accent-primary) !important;
}}

/* Expander */
details {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 12px !important;
}}

/* Divider */
hr {{ border-color: var(--border-main) !important; }}

/* ─────────── BUTTONS ─────────── */
.stButton > button,
.stDownloadButton > button,
.stDownloadButton a,
.stDownloadButton a > button,
[data-testid="stDownloadButton"] > button,
[data-testid="stDownloadButton"] a,
[data-testid="stDownloadButton"] a > button,
.stFormSubmitButton > button,
[data-testid="stFormSubmitButton"] > button,
button[kind*="FormSubmit"],
button[data-testid*="FormSubmit"],
button[data-testid="baseButton-primary"],
button[data-testid="baseButton-secondary"],
button[kind="secondary"] {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

/* All Secondary & Download buttons */
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"],
.stDownloadButton > button,
.stDownloadButton a,
.stDownloadButton a > button,
[data-testid="stDownloadButton"] > button,
[data-testid="stDownloadButton"] a,
[data-testid="stDownloadButton"] a > button,
button[kind="secondary"],
button[data-testid="baseButton-secondary"] {{
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-main) !important;
    box-shadow: var(--shadow-card) !important;
}}

.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover,
.stDownloadButton > button:hover,
.stDownloadButton a:hover,
.stDownloadButton a > button:hover,
[data-testid="stDownloadButton"] > button:hover,
[data-testid="stDownloadButton"] a:hover,
[data-testid="stDownloadButton"] a > button:hover,
button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {{
    background-color: var(--bg-glass) !important;
    border-color: var(--border-hover) !important;
    color: var(--accent-primary) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.2) !important;
}}

/* Force visible text on secondary & download buttons */
.stDownloadButton > button *,
.stDownloadButton a *,
[data-testid="stDownloadButton"] > button *,
[data-testid="stDownloadButton"] a *,
.stButton > button[kind="secondary"] *,
.stButton > button[data-testid="baseButton-secondary"] * {{
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}}
.stDownloadButton > button:hover *,
.stDownloadButton a:hover *,
[data-testid="stDownloadButton"] > button:hover *,
[data-testid="stDownloadButton"] a:hover *,
.stButton > button[kind="secondary"]:hover *,
.stButton > button[data-testid="baseButton-secondary"]:hover * {{
    color: var(--accent-primary) !important;
    -webkit-text-fill-color: var(--accent-primary) !important;
}}

/* Nav Primary (active) button */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {{
    background: var(--gradient-1) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.25) !important;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {{
    box-shadow: 0 6px 24px rgba(99,102,241,0.35) !important;
    color: #FFFFFF !important;
    transform: translateY(-2px) !important;
}}

/* Form Submit Button — Prominent Primary CTA (Predict Default Risk) */
.stFormSubmitButton > button,
[data-testid="stFormSubmitButton"] > button,
button[kind="primaryFormSubmit"],
button[kind="secondaryFormSubmit"],
button[data-testid="baseButton-primaryFormSubmit"],
button[data-testid="baseButton-secondaryFormSubmit"] {{
    background: var(--gradient-1) !important;
    background-color: var(--accent-primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-size: 1.08rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    cursor: pointer !important;
}}
.stFormSubmitButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
button[kind="primaryFormSubmit"]:hover,
button[kind="secondaryFormSubmit"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.55) !important;
    color: #FFFFFF !important;
}}
.stFormSubmitButton > button *,
[data-testid="stFormSubmitButton"] > button *,
button[kind="primaryFormSubmit"] *,
button[kind="secondaryFormSubmit"] * {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
}}

/* ─────────── CUSTOM HIGH-END TABLES ─────────── */
.custom-table-wrapper {{
    background: var(--bg-card);
    border: 1px solid var(--border-main);
    border-radius: 14px;
    overflow-x: auto;
    overflow-y: auto;
    box-shadow: var(--shadow-card);
    margin: 1rem 0;
    transition: all 0.3s ease;
}}
.custom-table-wrapper:hover {{
    box-shadow: var(--shadow-hover);
    border-color: var(--border-hover);
}}
.custom-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    text-align: left;
    background: transparent;
}}
.custom-table thead {{
    position: sticky;
    top: 0;
    z-index: 5;
}}
.custom-table th {{
    background: var(--bg-secondary);
    color: var(--text-primary);
    font-weight: 700;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 14px 18px;
    border-bottom: 2px solid rgba(99,102,241,0.3);
    white-space: nowrap;
}}
.custom-table td {{
    padding: 12px 18px;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border-main);
    white-space: nowrap;
}}
.custom-table tbody tr {{
    transition: background 0.2s ease;
}}
.custom-table tbody tr:nth-child(even) {{
    background: var(--bg-glass);
}}
.custom-table tbody tr:hover {{
    background: rgba(99,102,241,0.08) !important;
}}

/* Table Badges */
.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}}
.badge-purple {{
    background: rgba(99,102,241,0.12);
    color: var(--accent-primary);
    border: 1px solid rgba(99,102,241,0.25);
}}
.badge-cyan {{
    background: rgba(8,145,178,0.12);
    color: var(--accent-cyan);
    border: 1px solid rgba(8,145,178,0.25);
}}
.badge-green {{
    background: rgba(5,150,105,0.12);
    color: var(--accent-emerald);
    border: 1px solid rgba(5,150,105,0.25);
}}
.badge-rose {{
    background: rgba(225,29,72,0.12);
    color: var(--accent-rose);
    border: 1px solid rgba(225,29,72,0.25);
}}

/* Fallbacks for generic HTML and Streamlit tables */
[data-testid="stTable"],
[data-testid="stTable"] table,
[data-testid="stTable"] thead,
[data-testid="stTable"] tbody,
[data-testid="stTable"] tr,
[data-testid="stTable"] th,
[data-testid="stTable"] td,
table, thead, tbody, tr, th, td {{
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-main) !important;
}}
[data-testid="stTable"] th {{
    background-color: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
}}

[data-testid="stDataFrame"],
.stDataFrame {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 14px !important;
}}

/* ─────────── FOOTER ─────────── */
.site-footer {{
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border-main);
    animation: fadeIn 0.5s ease both;
}}
.footer-brand {{
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-primary);
    margin-bottom: 8px;
}}
.footer-text {{
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.8;
}}
.footer-badges {{
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-top: 12px;
    flex-wrap: wrap;
}}
.footer-badge {{
    background: var(--bg-glass);
    border: 1px solid var(--border-main);
    border-radius: 6px;
    padding: 3px 12px;
    font-size: 0.7rem;
    color: var(--text-muted);
    font-weight: 500;
}}

/* ─────────── ABOUT PAGE ─────────── */
.about-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-main);
    border-radius: 16px;
    padding: 24px;
    box-shadow: var(--shadow-card);
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease both;
    height: 100%;
}}
.about-card:hover {{
    transform: translateY(-4px);
    box-shadow: var(--shadow-hover);
    border-color: var(--border-hover);
}}

/* ─────────── SMOOTH SCROLLBAR ─────────── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border-main); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent-primary); }}

/* ── chart container animation ── */
.chart-container {{
    animation: fadeInUp 0.6s ease both;
}}

/* ─────────── RESPONSIVE SYSTEM (DESKTOP RESIZE, TABLET, MOBILE / iOS & ANDROID) ─────────── */

/* Touch & tap optimization */
* {{
    -webkit-tap-highlight-color: transparent !important;
}}

/* 1. Large screens (> 1100px): Centered, spacious dock */
@media (min-width: 1100px) {{
    div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="stHorizontalBlock"] {{
        justify-content: center !important;
        gap: 8px !important;
    }}
    div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button {{
        padding: 0 15px !important;
        font-size: 0.86rem !important;
    }}
}}

/* 2. Resized PC window & Tablets (769px to 1099px) */
@media (max-width: 1099px) and (min-width: 769px) {{
    div[data-testid="stAppViewContainer"] .st-key-navbar {{
        padding: 5px 8px !important;
    }}
    div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="stHorizontalBlock"] {{
        justify-content: flex-start !important;
        gap: 5px !important;
    }}
    div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button {{
        padding: 0 10px !important;
        font-size: 0.81rem !important;
        height: 36px !important;
        min-height: 36px !important;
    }}

    /* 4-column blocks (KPI cards, metrics, filters, form row 5) adapt into 2x2 grid */
    div[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:not(.st-key-navbar [data-testid="stHorizontalBlock"]) {{
        flex-wrap: wrap !important;
        gap: 12px !important;
    }}
    div[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:not(.st-key-navbar [data-testid="stHorizontalBlock"]) > [data-testid="column"]:nth-last-child(4),
    div[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:not(.st-key-navbar [data-testid="stHorizontalBlock"]) > [data-testid="column"]:nth-last-child(4) ~ [data-testid="column"] {{
        flex: 1 1 calc(50% - 12px) !important;
        min-width: calc(50% - 12px) !important;
        max-width: calc(50% - 12px) !important;
    }}

    /* 3-column blocks (Predict inputs, About cards) wrap gracefully */
    div[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:not(.st-key-navbar [data-testid="stHorizontalBlock"]) > [data-testid="column"]:nth-last-child(3),
    div[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:not(.st-key-navbar [data-testid="stHorizontalBlock"]) > [data-testid="column"]:nth-last-child(3) ~ [data-testid="column"] {{
        flex: 1 1 calc(50% - 12px) !important;
        min-width: calc(50% - 12px) !important;
    }}
}}

/* 3. Medium Screens & Side-by-Side Content (<= 992px) */
@media (max-width: 992px) {{
    /* 2-column blocks (Charts, Prediction Result + Gauge) stack into full-width cards */
    div[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:not(.st-key-navbar [data-testid="stHorizontalBlock"]) > [data-testid="column"]:nth-last-child(2),
    div[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:not(.st-key-navbar [data-testid="stHorizontalBlock"]) > [data-testid="column"]:nth-last-child(2) ~ [data-testid="column"] {{
        flex: 1 1 100% !important;
        min-width: 100% !important;
        width: 100% !important;
    }}
}}

/* 4. Mobile & Small Tablet (<= 768px): iOS / Android optimized */
@media (max-width: 768px) {{
    /* Page margins: flush and clean for phone screens */
    .block-container {{
        padding-top: 0.3rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
        max-width: 100% !important;
    }}

    /* Floating Navbar on mobile — Single-row swipeable pill dock that NEVER stacks */
    div[data-testid="stAppViewContainer"] .st-key-navbar {{
        border-radius: 14px !important;
        padding: 4px 6px !important;
        top: 0.2rem !important;
        margin-bottom: 0.8rem !important;
        overflow: visible !important;
        position: sticky !important;
    }}

    div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="stHorizontalBlock"] {{
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        justify-content: flex-start !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
        gap: 5px !important;
        padding: 2px 2px !important;
    }}

    div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] {{
        display: flex !important;
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: max-content !important;
        max-width: none !important;
    }}

    div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] > div,
    div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] [data-testid="stVerticalBlock"],
    div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] [data-testid="element-container"],
    div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] .stButton,
    div[data-testid="stAppViewContainer"] .st-key-navbar [data-testid="column"] .stMarkdown {{
        display: inline-flex !important;
        width: auto !important;
        min-width: max-content !important;
        flex: 0 0 auto !important;
    }}

    div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button {{
        height: 36px !important;
        min-height: 36px !important;
        padding: 0 11px !important;
        font-size: 0.80rem !important;
        white-space: nowrap !important;
        min-width: max-content !important;
        border-radius: 10px !important;
        flex-shrink: 0 !important;
    }}

    .navbar-brand-icon {{
        width: 36px !important;
        height: 36px !important;
        min-width: 36px !important;
        font-size: 1.15rem !important;
        border-radius: 10px !important;
    }}

    /* ALL other content columns stack vertically to 100% width on mobile */
    div[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:not(.st-key-navbar [data-testid="stHorizontalBlock"]) {{
        flex-direction: column !important;
        flex-wrap: wrap !important;
        gap: 12px !important;
    }}
    div[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"]:not(.st-key-navbar [data-testid="stHorizontalBlock"]) > [data-testid="column"] {{
        flex: 1 1 100% !important;
        min-width: 100% !important;
        width: 100% !important;
        max-width: 100% !important;
    }}

    /* Hero section */
    .hero-section {{
        padding: 1.2rem 0.4rem 0.8rem !important;
    }}
    .hero-badges {{
        gap: 6px !important;
        margin-top: 12px !important;
    }}
    .hero-badge {{
        padding: 4px 10px !important;
        font-size: 0.73rem !important;
    }}

    /* Section titles */
    .section-title {{
        font-size: 1.15rem !important;
        margin: 1.2rem 0 0.6rem !important;
    }}

    /* KPI Cards */
    .kpi-card {{
        padding: 16px 12px !important;
        border-radius: 12px !important;
        margin-bottom: 6px !important;
    }}
    .kpi-card .kpi-icon {{
        font-size: 1.5rem !important;
        margin-bottom: 4px !important;
    }}
    .kpi-card .kpi-value {{
        font-size: 1.6rem !important;
    }}
    .kpi-card .kpi-label {{
        font-size: 0.70rem !important;
        letter-spacing: 0.8px !important;
        margin-top: 4px !important;
    }}

    /* Input controls (prevent iOS Safari 16px auto-zoom + touch target 46px) */
    input,
    textarea,
    select,
    [data-baseweb="input"] input,
    [data-testid="stNumberInput-Input"],
    [data-testid="stTextInput-Input"] {{
        font-size: 16px !important;
        min-height: 46px !important;
    }}
    [data-baseweb="input"],
    [data-baseweb="base-input"],
    [data-baseweb="select"],
    [data-baseweb="select"] > div {{
        min-height: 46px !important;
    }}

    /* Form container */
    [data-testid="stForm"] {{
        padding: 16px 12px !important;
        border-radius: 14px !important;
    }}
    .stFormSubmitButton > button,
    [data-testid="stFormSubmitButton"] > button {{
        padding: 0.85rem 1.2rem !important;
        font-size: 1rem !important;
        min-height: 48px !important;
        width: 100% !important;
    }}

    /* Prediction Result Cards */
    .prediction-safe,
    .prediction-risk {{
        padding: 22px 14px !important;
        border-radius: 16px !important;
    }}

    /* Glass Panels & About Cards */
    .glass-panel {{
        padding: 18px 14px !important;
        border-radius: 14px !important;
    }}
    .about-card {{
        padding: 18px 14px !important;
        border-radius: 14px !important;
        margin-bottom: 10px !important;
    }}

    /* Custom Tables */
    .custom-table-wrapper {{
        border-radius: 10px !important;
        margin: 0.6rem 0 !important;
    }}
    .custom-table th {{
        padding: 10px 12px !important;
        font-size: 0.72rem !important;
    }}
    .custom-table td {{
        padding: 9px 12px !important;
        font-size: 0.82rem !important;
    }}

    /* Footer */
    .site-footer {{
        padding: 1.8rem 0.5rem 1.2rem !important;
        margin-top: 2rem !important;
    }}
}}

/* 5. Small phones (<= 480px: iPhone SE, compact Androids) */
@media (max-width: 480px) {{
    .kpi-card .kpi-value {{
        font-size: 1.4rem !important;
    }}
    .custom-table th, .custom-table td {{
        padding: 8px 10px !important;
    }}
    div[data-testid="stAppViewContainer"] .st-key-navbar .stButton > button {{
        padding: 0 9px !important;
        font-size: 0.78rem !important;
    }}
}}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Data & model loading (cached)
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("Loan_default.csv")
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("Loan_Default.pkl")


@st.cache_resource(show_spinner=False)
def load_scaler():
    return joblib.load("Scaler.pkl")


FEATURE_COLUMNS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DebtToIncomeRatio",
    "Education_High School", "Education_Master's", "Education_PhD",
    "EmploymentType_Part-time", "EmploymentType_Self-employed",
    "EmploymentType_Unemployed", "MaritalStatus_Married",
    "MaritalStatus_Single", "HasMortgage_Yes", "HasDependents_Yes",
    "LoanPurpose_Business", "LoanPurpose_Education", "LoanPurpose_Home",
    "LoanPurpose_Other", "HasCoSigner_Yes",
]

NUMERIC_COLS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
]

SCALED_COLS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DebtToIncomeRatio",
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals and scale numerics to match training pipeline."""
    X = df.drop(columns=["LoanID", "Default"], errors="ignore")

    # Rename DTIRatio → DebtToIncomeRatio to match model training
    if "DTIRatio" in X.columns:
        X = X.rename(columns={"DTIRatio": "DebtToIncomeRatio"})

    X = pd.get_dummies(X, drop_first=True)

    # Ensure column order matches training
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLUMNS]

    # Apply StandardScaler to numeric features
    scaler = load_scaler()
    X[SCALED_COLS] = scaler.transform(X[SCALED_COLS])

    return X


@st.cache_data(show_spinner=False)
def get_metrics():
    """Compute train/test metrics once and cache."""
    df = load_data()
    X = preprocess(df)
    y = df["Default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    model = load_model()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "cm": cm, "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "report": report, "y_test": y_test, "y_prob": y_prob,
        "X_test": X_test, "y_pred": y_pred,
    }


# ──────────────────────────────────────────────
# Plotly theme helper — adapts to light/dark
# ──────────────────────────────────────────────
def get_plotly_layout():
    """Return Plotly layout dict that matches the current theme."""
    if IS_DARK:
        return dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#e2e8f0"),
            margin=dict(l=24, r=24, t=44, b=32),
            xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
        )
    else:
        return dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#1e293b"),
            margin=dict(l=24, r=24, t=44, b=32),
            xaxis=dict(gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.15)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.15)"),
        )


PLOTLY_LAYOUT = get_plotly_layout()
PLOTLY_CONFIG = {"responsive": True, "displayModeBar": False}


# ──────────────────────────────────────────────
# Theme-aware color palettes
# ──────────────────────────────────────────────
CLR_PRIMARY = "#6366f1"
CLR_DANGER = "#fb7185" if IS_DARK else "#E11D48"
CLR_SUCCESS = "#34d399" if IS_DARK else "#059669"
CLR_WARN = "#fbbf24" if IS_DARK else "#D97706"
CLR_CYAN = "#22d3ee" if IS_DARK else "#0891B2"
CLR_TEXT_SEC = "#94a3b8" if IS_DARK else "#64748b"
CLR_CARD_TEXT = "#cbd5e1" if IS_DARK else "#334155"
CLR_HEADING = "#a78bfa" if IS_DARK else "#4F46E5"


# ──────────────────────────────────────────────
# Styled HTML Table Generator (Theme Reactive)
# ──────────────────────────────────────────────
def render_styled_table(df: pd.DataFrame, max_height: int = 420, hide_index: bool = True) -> str:
    """Generate a clean, high-end responsive HTML table matching the active theme."""
    headers = list(df.columns)
    show_idx = not hide_index or (df.index.name is not None) or (len(df.index) > 0 and isinstance(df.index[0], str))

    th_cells = []
    if show_idx:
        idx_label = df.index.name if df.index.name else "Index"
        th_cells.append(f"<th>{idx_label}</th>")
    for col in headers:
        th_cells.append(f"<th>{col}</th>")
    th_html = "".join(th_cells)

    rows_html = []
    for idx, row in df.iterrows():
        td_cells = []
        if show_idx:
            td_cells.append(f"<td style='font-weight:600; color:var(--accent-primary);'>{idx}</td>")
        for col in headers:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                val_str = f"{val:.4f}" if (0 < abs(val) < 0.01) else f"{val:.2f}"
            elif isinstance(val, (int, np.integer)):
                val_str = f"{val:,}"
            else:
                val_str = str(val)
                if val_str in ["Numeric"]:
                    val_str = '<span class="badge badge-purple">Numeric</span>'
                elif val_str in ["Categorical"]:
                    val_str = '<span class="badge badge-cyan">Categorical</span>'
                elif val_str in ["Low", "0", "unlikely to default", "No"]:
                    val_str = f'<span class="badge badge-green">{val_str}</span>'
                elif val_str in ["High", "1", "likely to default", "Yes"]:
                    val_str = f'<span class="badge badge-rose">{val_str}</span>'
            td_cells.append(f"<td>{val_str}</td>")
        rows_html.append(f"<tr>{''.join(td_cells)}</tr>")

    tbody_html = "".join(rows_html)

    return f"""<div class="custom-table-wrapper" style="max-height:{max_height}px;">
<table class="custom-table">
<thead><tr>{th_html}</tr></thead>
<tbody>{tbody_html}</tbody>
</table>
</div>"""


# ──────────────────────────────────────────────
# TOP NAVBAR (Unified Floating Glass Dock)
# ──────────────────────────────────────────────
NAV_PAGES = ["🏠 Dashboard", "🔮 Predict", "📊 Model Performance",
             "🧠 How It Works", "📂 Data Explorer", "ℹ️ About"]

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🏠 Dashboard"

with st.container(key="navbar"):
    logo_col, *nav_button_cols, theme_col = st.columns(
        [0.45, 1.1, 0.95, 1.35, 1.15, 1.15, 0.85, 0.85],
        vertical_alignment="center",
        gap="small",
    )

    with logo_col:
        st.markdown(
            '<div class="navbar-brand-icon" title="Loan Default Predictor">🏦</div>',
            unsafe_allow_html=True,
        )

    for i, p in enumerate(NAV_PAGES):
        with nav_button_cols[i]:
            if st.button(p, use_container_width=True, key=f"nav_{p}",
                         type="primary" if st.session_state.nav_page == p else "secondary"):
                st.session_state.nav_page = p
                st.rerun()

    with theme_col:
        theme_label = "🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"
        st.button(theme_label, on_click=toggle_theme, use_container_width=True, key="theme_toggle")

page = st.session_state.nav_page


# ═══════════════════════════════════════════════
# PAGE: Dashboard
# ═══════════════════════════════════════════════
if page == "🏠 Dashboard":
    # Hero section — strictly centered, no inner multi-line indentation
    st.markdown("""<div class="hero-section">
<h1 class="hero-title">Loan Default Prediction Dashboard</h1>
<p class="hero-subtitle">Real-time analytics & predictions powered by a Decision Tree Classifier with StandardScaler preprocessing on 255K+ loan records.</p>
<div class="hero-badges">
<span class="hero-badge">🌲 Decision Tree</span>
<span class="hero-badge">📏 StandardScaler</span>
<span class="hero-badge">📊 255K+ Records</span>
<span class="hero-badge">🎯 24 Features</span>
</div>
</div>""", unsafe_allow_html=True)

    df = load_data()
    m = get_metrics()

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card kpi-purple kpi-delay-1">
            <div class="kpi-icon">📋</div>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-label">Total Records</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card kpi-cyan kpi-delay-2">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-value">{m['acc']:.1%}</div>
            <div class="kpi-label">Model Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        default_rate = df["Default"].mean()
        st.markdown(f"""
        <div class="kpi-card kpi-rose kpi-delay-3">
            <div class="kpi-icon">⚠️</div>
            <div class="kpi-value">{default_rate:.1%}</div>
            <div class="kpi-label">Default Rate</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card kpi-green kpi-delay-4">
            <div class="kpi-icon">📈</div>
            <div class="kpi-value">{m['roc_auc']:.3f}</div>
            <div class="kpi-label">ROC-AUC Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-title">📊 Default Distribution</div>', unsafe_allow_html=True)
        counts = df["Default"].value_counts().reset_index()
        counts.columns = ["Default", "Count"]
        counts["Label"] = counts["Default"].map({0: "No Default", 1: "Default"})
        fig_pie = px.pie(
            counts, values="Count", names="Label",
            color_discrete_sequence=[CLR_PRIMARY, CLR_DANGER],
            hole=0.55,
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
        fig_pie.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=380)
        st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)

    with col_right:
        st.markdown('<div class="section-title">💰 Income Distribution by Default</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            df, x="Income", color="Default",
            nbins=60,
            color_discrete_map={0: CLR_PRIMARY, 1: CLR_DANGER},
            barmode="overlay", opacity=0.7,
            labels={"Default": "Default Status"},
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=380, legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ))
        st.plotly_chart(fig_hist, use_container_width=True, config=PLOTLY_CONFIG)

    # Charts row 2
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">📈 Credit Score vs Loan Amount</div>', unsafe_allow_html=True)
        sample = df.sample(n=min(3000, len(df)), random_state=42)
        fig_sc = px.scatter(
            sample, x="CreditScore", y="LoanAmount", color="Default",
            color_discrete_map={0: CLR_PRIMARY, 1: CLR_DANGER},
            opacity=0.5, labels={"Default": "Default"},
        )
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=380, legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ))
        st.plotly_chart(fig_sc, use_container_width=True, config=PLOTLY_CONFIG)

    with col_b:
        st.markdown('<div class="section-title">🏷️ Default Rate by Education</div>', unsafe_allow_html=True)
        edu_rate = df.groupby("Education")["Default"].mean().reset_index()
        edu_rate.columns = ["Education", "Default Rate"]
        fig_bar = px.bar(
            edu_rate.sort_values("Default Rate", ascending=True),
            x="Default Rate", y="Education", orientation="h",
            color="Default Rate",
            color_continuous_scale=[CLR_PRIMARY, CLR_DANGER],
        )
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=380, coloraxis_showscale=False)
        fig_bar.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)

    # Charts row 3 — additional insights
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-title">📊 Default Rate by Employment Type</div>', unsafe_allow_html=True)
        emp_rate = df.groupby("EmploymentType")["Default"].mean().reset_index()
        emp_rate.columns = ["EmploymentType", "Default Rate"]
        fig_emp = px.bar(
            emp_rate.sort_values("Default Rate", ascending=False),
            x="EmploymentType", y="Default Rate",
            color="Default Rate",
            color_continuous_scale=[CLR_SUCCESS, CLR_DANGER],
        )
        fig_emp.update_layout(**PLOTLY_LAYOUT, height=380, coloraxis_showscale=False,
                              xaxis_title="Employment Type", yaxis_title="Default Rate")
        fig_emp.update_traces(marker_line_width=0)
        st.plotly_chart(fig_emp, use_container_width=True, config=PLOTLY_CONFIG)

    with col_d:
        st.markdown('<div class="section-title">📉 Interest Rate Distribution</div>', unsafe_allow_html=True)
        fig_ir = px.histogram(
            df, x="InterestRate", color="Default",
            nbins=50, barmode="overlay", opacity=0.7,
            color_discrete_map={0: CLR_PRIMARY, 1: CLR_DANGER},
            labels={"Default": "Default Status"},
        )
        fig_ir.update_layout(**PLOTLY_LAYOUT, height=380, legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ), xaxis_title="Interest Rate (%)", yaxis_title="Count")
        st.plotly_chart(fig_ir, use_container_width=True, config=PLOTLY_CONFIG)


# ═══════════════════════════════════════════════
# PAGE: Predict
# ═══════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown(f"""
    <div class="page-header">
        <h1 style="color:{CLR_HEADING};">🔮 Loan Default Predictor</h1>
        <p>Fill in the applicant details below to predict default risk using our Decision Tree model</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()

    with st.form("prediction_form"):
        st.markdown('<div class="form-section-title">👤 Applicant Information</div>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        with r1c2:
            income = st.number_input("Annual Income ($)", min_value=0, max_value=500000, value=55000, step=1000)
        with r1c3:
            education = st.selectbox("Education", ["Bachelor's", "Master's", "High School", "PhD"])

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        with r2c2:
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        with r2c3:
            months_employed = st.number_input("Months Employed", min_value=0, max_value=500, value=60, step=1)

        st.markdown('<div class="form-section-title">💳 Loan Details</div>', unsafe_allow_html=True)
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=500000, value=25000, step=500)
        with r3c2:
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=8.5, step=0.1)
        with r3c3:
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

        r4c1, r4c2, r4c3 = st.columns(3)
        with r4c1:
            credit_score = st.slider("Credit Score", 300, 850, 650)
        with r4c2:
            num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=20, value=3, step=1)
        with r4c3:
            dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

        st.markdown('<div class="form-section-title">📋 Additional Information</div>', unsafe_allow_html=True)
        r5c1, r5c2, r5c3, r5c4 = st.columns(4)
        with r5c1:
            loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
        with r5c2:
            has_mortgage = st.selectbox("Has Mortgage?", ["No", "Yes"])
        with r5c3:
            has_dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        with r5c4:
            has_cosigner = st.selectbox("Has Co-Signer?", ["No", "Yes"])

        submitted = st.form_submit_button("⚡ Predict Default Risk", type="primary", use_container_width=True)

    if submitted:
        # Build a single-row DataFrame matching the training schema
        row = {col: 0 for col in FEATURE_COLUMNS}
        row["Age"] = age
        row["Income"] = income
        row["LoanAmount"] = loan_amount
        row["CreditScore"] = credit_score
        row["MonthsEmployed"] = months_employed
        row["NumCreditLines"] = num_credit_lines
        row["InterestRate"] = interest_rate
        row["LoanTerm"] = loan_term
        row["DebtToIncomeRatio"] = dti_ratio

        # One-hot flags (drop_first → Bachelor's, Full-time, Divorced, Auto are base)
        if education == "High School":
            row["Education_High School"] = 1
        elif education == "Master's":
            row["Education_Master's"] = 1
        elif education == "PhD":
            row["Education_PhD"] = 1

        if employment_type == "Part-time":
            row["EmploymentType_Part-time"] = 1
        elif employment_type == "Self-employed":
            row["EmploymentType_Self-employed"] = 1
        elif employment_type == "Unemployed":
            row["EmploymentType_Unemployed"] = 1

        if marital_status == "Married":
            row["MaritalStatus_Married"] = 1
        elif marital_status == "Single":
            row["MaritalStatus_Single"] = 1

        if has_mortgage == "Yes":
            row["HasMortgage_Yes"] = 1
        if has_dependents == "Yes":
            row["HasDependents_Yes"] = 1
        if has_cosigner == "Yes":
            row["HasCoSigner_Yes"] = 1

        if loan_purpose == "Business":
            row["LoanPurpose_Business"] = 1
        elif loan_purpose == "Education":
            row["LoanPurpose_Education"] = 1
        elif loan_purpose == "Home":
            row["LoanPurpose_Home"] = 1
        elif loan_purpose == "Other":
            row["LoanPurpose_Other"] = 1

        input_df = pd.DataFrame([row])[FEATURE_COLUMNS]

        # Apply scaler to numeric features
        scaler = load_scaler()
        input_df[SCALED_COLS] = scaler.transform(input_df[SCALED_COLS])

        model = load_model()
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.markdown("<br>", unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1.3, 1])
        with res_col1:
            if prediction == 0:
                st.markdown(f"""
                <div class="prediction-safe">
                    <div style="font-size:3.5rem; margin-bottom:8px;">✅</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{CLR_SUCCESS};">Low Default Risk</div>
                    <div style="color:{CLR_TEXT_SEC}; margin-top:8px; font-size:1rem;">
                        The model predicts this applicant is <strong>unlikely to default</strong>.
                    </div>
                    <div style="margin-top:16px; font-size:1.6rem; font-weight:700; color:{CLR_SUCCESS};">
                        Confidence: {proba[0]*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-risk">
                    <div style="font-size:3.5rem; margin-bottom:8px;">🚨</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{CLR_DANGER};">High Default Risk</div>
                    <div style="color:{CLR_TEXT_SEC}; margin-top:8px; font-size:1rem;">
                        The model predicts this applicant is <strong>likely to default</strong>.
                    </div>
                    <div style="margin-top:16px; font-size:1.6rem; font-weight:700; color:{CLR_DANGER};">
                        Default Probability: {proba[1]*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            # Probability gauge
            gauge_bar_color = CLR_PRIMARY
            gauge_bg = "rgba(30,41,72,0.45)" if IS_DARK else "rgba(241,245,249,0.9)"
            gauge_border = "rgba(99,128,255,0.12)" if IS_DARK else "rgba(148,163,184,0.2)"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1] * 100,
                number={"suffix": "%", "font": {"size": 36, "color": PLOTLY_LAYOUT["font"]["color"]}},
                title={"text": "Default Probability", "font": {"size": 16, "color": CLR_TEXT_SEC}},
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(color=CLR_TEXT_SEC)),
                    bar=dict(color=gauge_bar_color),
                    bgcolor=gauge_bg,
                    bordercolor=gauge_border,
                    steps=[
                        dict(range=[0, 30], color="rgba(16,185,129,0.12)"),
                        dict(range=[30, 60], color="rgba(251,191,36,0.12)"),
                        dict(range=[60, 100], color="rgba(244,63,94,0.12)"),
                    ],
                    threshold=dict(line=dict(color=CLR_DANGER, width=3), thickness=0.8, value=50),
                ),
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color=PLOTLY_LAYOUT["font"]["color"]),
                height=280,
                margin=dict(l=30, r=30, t=60, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config=PLOTLY_CONFIG)

        # Risk factors summary
        st.markdown(f'<div class="section-title">📋 Input Summary</div>', unsafe_allow_html=True)
        summary_data = {
            "Parameter": ["Age", "Income", "Loan Amount", "Credit Score", "Months Employed",
                          "Interest Rate", "Loan Term", "DTI Ratio", "Education",
                          "Employment", "Marital Status", "Loan Purpose"],
            "Value": [f"{age}", f"${income:,}", f"${loan_amount:,}", f"{credit_score}",
                      f"{months_employed}", f"{interest_rate}%", f"{loan_term} months",
                      f"{dti_ratio:.2f}", education, employment_type, marital_status, loan_purpose],
        }
        st.markdown(render_styled_table(pd.DataFrame(summary_data), max_height=480), unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE: Model Performance
# ═══════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown(f"""
    <div class="page-header">
        <h1 style="color:{CLR_CYAN};">📊 Model Performance Metrics</h1>
        <p>Comprehensive evaluation of the Decision Tree Classifier on a held-out 20% test set (stratified)</p>
    </div>
    """, unsafe_allow_html=True)

    m = get_metrics()

    # Metric cards
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, label, val, cls, delay in [
        (mc1, "Accuracy",  m["acc"],  "kpi-purple", "kpi-delay-1"),
        (mc2, "Precision", m["prec"], "kpi-cyan",   "kpi-delay-2"),
        (mc3, "Recall",    m["rec"],  "kpi-green",  "kpi-delay-3"),
        (mc4, "F1 Score",  m["f1"],   "kpi-rose",   "kpi-delay-4"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls} {delay}">
                <div class="kpi-value">{val:.3f}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔲 Confusion Matrix", "📈 ROC Curve",
        "📋 Classification Report", "📊 Probability Distribution",
    ])

    with tab1:
        cm = m["cm"]
        labels = ["No Default (0)", "Default (1)"]
        cm_text_color = "white" if IS_DARK else "#1e293b"
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0, "#EEF2FF" if not IS_DARK else "#1e1b4b"],
                        [0.5, "#818CF8" if not IS_DARK else "#6366f1"],
                        [1, "#4F46E5" if not IS_DARK else "#a78bfa"]],
            text=[[f"{v:,}" for v in row] for row in cm],
            texttemplate="%{text}",
            textfont=dict(size=18, color=cm_text_color),
            showscale=False,
        ))
        layout_overrides = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"}
        fig_cm.update_layout(
            **layout_overrides, height=450,
            xaxis_title="Predicted", yaxis_title="Actual",
            yaxis=dict(autorange="reversed", gridcolor=PLOTLY_LAYOUT["xaxis"]["gridcolor"],
                       zerolinecolor=PLOTLY_LAYOUT["xaxis"]["zerolinecolor"]),
        )
        st.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CONFIG)

    with tab2:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=m["fpr"], y=m["tpr"], mode="lines",
            name=f'ROC (AUC = {m["roc_auc"]:.3f})',
            line=dict(color=CLR_PRIMARY, width=3),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Classifier",
            line=dict(color=CLR_TEXT_SEC, width=1, dash="dash"),
        ))
        fig_roc.update_layout(
            **PLOTLY_LAYOUT, height=450,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.55, y=0.05),
        )
        st.plotly_chart(fig_roc, use_container_width=True, config=PLOTLY_CONFIG)

    with tab3:
        report = m["report"]
        report_df = pd.DataFrame(report).T
        report_df = report_df.round(3)
        st.markdown(render_styled_table(report_df, max_height=320, hide_index=False), unsafe_allow_html=True)

    with tab4:
        y_test = m["y_test"]
        y_prob = m["y_prob"]
        prob_df = pd.DataFrame({"Probability": y_prob, "Actual": y_test.values})
        fig_prob = px.histogram(
            prob_df, x="Probability", color="Actual",
            nbins=60, barmode="overlay", opacity=0.7,
            color_discrete_map={0: CLR_PRIMARY, 1: CLR_DANGER},
            labels={"Actual": "Actual Class"},
        )
        fig_prob.update_layout(**PLOTLY_LAYOUT, height=420, legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ))
        st.plotly_chart(fig_prob, use_container_width=True, config=PLOTLY_CONFIG)


# ═══════════════════════════════════════════════
# PAGE: How It Works
# ═══════════════════════════════════════════════
elif page == "🧠 How It Works":
    st.markdown(f"""
    <div class="page-header">
        <h1 style="color:{CLR_SUCCESS};">🧠 How the Model Works</h1>
        <p>Understanding the Decision Tree Classifier for loan default prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Explainer panels ──
    st.markdown('<div class="section-title">📖 Algorithm Overview</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass-panel">
        <h3 style="color:{CLR_HEADING}; margin-top:0;">What is a Decision Tree?</h3>
        <p style="color:{CLR_CARD_TEXT}; line-height:1.8;">
            A Decision Tree Classifier is a <strong>supervised machine learning algorithm</strong> that
            makes predictions by learning simple decision rules from training data. It recursively
            partitions the feature space into regions by selecting the feature and threshold that
            best separates the classes at each node. The tree grows from a <strong>root node</strong>
            down to <strong>leaf nodes</strong>, where each leaf represents a class prediction.
        </p>
        <div style="text-align:center; margin: 20px 0 10px;">
            <div style="display:inline-block; background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.25);
                        border-radius:12px; padding:16px 32px;">
                <span style="font-size:1.2rem; font-family:'Courier New',monospace; color:{CLR_HEADING}; font-weight:700;">
                    If Feature_i ≤ threshold → Left Branch, else → Right Branch
                </span>
            </div>
        </div>
        <p style="color:{CLR_TEXT_SEC}; font-size:0.9rem; text-align:center; margin-top:12px;">
            At each leaf node, the majority class of training samples determines the prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_hw1, col_hw2 = st.columns(2)

    with col_hw1:
        st.markdown(f"""
        <div class="glass-panel">
            <h4 style="color:{CLR_CYAN}; margin-top:0;">🔄 Training Pipeline</h4>
            <ol style="color:{CLR_CARD_TEXT}; line-height:2.2;">
                <li><strong>Data Loading</strong> — Load 255K+ loan records from CSV.</li>
                <li><strong>Feature Engineering</strong> — One-hot encode categorical features (Education, Employment, etc.).</li>
                <li><strong>Standard Scaling</strong> — Apply StandardScaler to 9 numeric features for normalisation.</li>
                <li><strong>Train/Test Split</strong> — 80/20 stratified split (random_state=42).</li>
                <li><strong>Decision Tree Training</strong> — Fit DecisionTreeClassifier on training data.</li>
                <li><strong>Evaluation</strong> — Compute accuracy, precision, recall, F1, and ROC-AUC on test set.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_hw2:
        model = load_model()
        st.markdown(f"""
        <div class="glass-panel">
            <h4 style="color:{CLR_WARN}; margin-top:0;">⚙️ Model Configuration</h4>
            <table style="width:100%; color:{CLR_CARD_TEXT}; border-collapse:separate; border-spacing:0 10px;">
                <tr><td style="color:{CLR_TEXT_SEC};">Algorithm</td><td style="font-weight:600;">Decision Tree Classifier</td></tr>
                <tr><td style="color:{CLR_TEXT_SEC};">Criterion</td><td style="font-weight:600;">Gini Impurity</td></tr>
                <tr><td style="color:{CLR_TEXT_SEC};">Max Depth</td><td style="font-weight:600;">{model.max_depth if model.max_depth else "None (unlimited)"}</td></tr>
                <tr><td style="color:{CLR_TEXT_SEC};">Tree Depth (actual)</td><td style="font-weight:600;">{model.get_depth()}</td></tr>
                <tr><td style="color:{CLR_TEXT_SEC};">Number of Leaves</td><td style="font-weight:600;">{model.get_n_leaves():,}</td></tr>
                <tr><td style="color:{CLR_TEXT_SEC};">Random State</td><td style="font-weight:600;">42</td></tr>
                <tr><td style="color:{CLR_TEXT_SEC};">Input Features</td><td style="font-weight:600;">24 (9 scaled + 15 encoded)</td></tr>
                <tr><td style="color:{CLR_TEXT_SEC};">Scaler</td><td style="font-weight:600;">StandardScaler (9 numeric features)</td></tr>
                <tr><td style="color:{CLR_TEXT_SEC};">Classes</td><td style="font-weight:600;">0 (No Default), 1 (Default)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Decision Tree conceptual visualization
    st.markdown('<div class="section-title">🌲 Decision Tree Conceptual View</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass-panel">
        <p style="color:{CLR_CARD_TEXT}; line-height:1.8; margin-bottom:16px;">
            The Decision Tree splits the data at each node based on the feature and threshold that
            maximises the <strong>Gini impurity reduction</strong>. Below is a simplified conceptual view
            of how the tree makes decisions:
        </p>
        <div style="text-align:center; padding: 20px;">
            <div style="display:inline-block; background:rgba(99,102,241,0.1); border:2px solid rgba(99,102,241,0.3);
                        border-radius:12px; padding:12px 24px; margin-bottom:8px;">
                <span style="color:{CLR_HEADING}; font-weight:700; font-size:0.95rem;">🌲 Root: InterestRate ≤ threshold?</span>
            </div>
            <div style="display:flex; justify-content:center; gap:40px; margin-top:16px;">
                <div>
                    <div style="color:{CLR_SUCCESS}; font-weight:600; margin-bottom:8px;">← Yes</div>
                    <div style="display:inline-block; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25);
                                border-radius:10px; padding:10px 20px;">
                        <span style="color:{CLR_CARD_TEXT}; font-size:0.88rem;">Income ≤ threshold?</span>
                    </div>
                </div>
                <div>
                    <div style="color:{CLR_DANGER}; font-weight:600; margin-bottom:8px;">→ No</div>
                    <div style="display:inline-block; background:rgba(244,63,94,0.08); border:1px solid rgba(244,63,94,0.25);
                                border-radius:10px; padding:10px 20px;">
                        <span style="color:{CLR_CARD_TEXT}; font-size:0.88rem;">CreditScore ≤ threshold?</span>
                    </div>
                </div>
            </div>
            <p style="color:{CLR_TEXT_SEC}; font-size:0.85rem; margin-top:20px;">
                This process continues recursively until reaching leaf nodes with final predictions.
                <br>The actual tree has <strong>{model.get_depth()} levels</strong> and <strong>{model.get_n_leaves():,} leaf nodes</strong>.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="section-title">📊 Feature Importance (Gini Importance)</div>', unsafe_allow_html=True)

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    colors = [CLR_PRIMARY if imp > 0.05 else (CLR_CYAN if imp > 0.02 else CLR_TEXT_SEC)
              for imp in feat_imp["Importance"]]
    fig_fi = go.Figure(go.Bar(
        x=feat_imp["Importance"], y=feat_imp["Feature"],
        orientation="h", marker_color=colors,
    ))
    fig_fi.update_layout(
        **PLOTLY_LAYOUT, height=650,
        xaxis_title="Gini Importance",
        yaxis_title="",
    )
    st.plotly_chart(fig_fi, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown(f"""
    <div class="glass-panel" style="margin-top:0.5rem;">
        <p style="color:{CLR_CARD_TEXT}; margin:0; line-height:1.8;">
            <strong>Gini Importance</strong> measures how much each feature contributes to reducing
            impurity across all splits in the tree. Higher values indicate features that the model
            relies on more heavily for making decisions. The top features —
            <span style="color:{CLR_HEADING}; font-weight:700;">Income</span>,
            <span style="color:{CLR_HEADING}; font-weight:700;">Interest Rate</span>, and
            <span style="color:{CLR_HEADING}; font-weight:700;">Loan Amount</span> —
            are the most influential in predicting loan default.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Scaler information
    st.markdown('<div class="section-title">📏 StandardScaler Details</div>', unsafe_allow_html=True)
    scaler = load_scaler()
    scaler_df = pd.DataFrame({
        "Feature": SCALED_COLS,
        "Mean (μ)": scaler.mean_,
        "Std Dev (σ)": scaler.scale_,
    }).round(4)
    st.markdown(render_styled_table(scaler_df, max_height=420), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass-panel">
        <p style="color:{CLR_CARD_TEXT}; margin:0; line-height:1.8;">
            <strong>StandardScaler</strong> transforms each numeric feature by subtracting its mean and
            dividing by its standard deviation: <code style="color:{CLR_HEADING};">z = (x - μ) / σ</code>.
            This ensures all numeric features are on the same scale, which can improve model stability.
            The scaler was fit on the training data and is applied to both training and test sets,
            as well as any new prediction inputs.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE: Data Explorer
# ═══════════════════════════════════════════════
elif page == "📂 Data Explorer":
    st.markdown(f"""
    <div class="page-header">
        <h1 style="color:{CLR_WARN};">📂 Data Explorer</h1>
        <p>Browse, filter, and understand the Loan Default dataset</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()

    # Filters
    st.markdown('<div class="section-title">🔍 Filters</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        edu_filter = st.multiselect("Education", df["Education"].unique(), default=df["Education"].unique())
    with fc2:
        emp_filter = st.multiselect("Employment", df["EmploymentType"].unique(), default=df["EmploymentType"].unique())
    with fc3:
        default_filter = st.multiselect("Default Status", [0, 1], default=[0, 1])
    with fc4:
        age_range = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()),
                              (int(df["Age"].min()), int(df["Age"].max())))

    filtered = df[
        (df["Education"].isin(edu_filter)) &
        (df["EmploymentType"].isin(emp_filter)) &
        (df["Default"].isin(default_filter)) &
        (df["Age"].between(*age_range))
    ]

    st.markdown(f"""
    <div class="glass-panel" style="padding:14px 24px;">
        <span style="color:{CLR_TEXT_SEC};">Showing</span>
        <span style="color:{CLR_HEADING}; font-weight:700;"> {len(filtered):,} </span>
        <span style="color:{CLR_TEXT_SEC};">of {len(df):,} records</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(render_styled_table(filtered.head(100), max_height=420), unsafe_allow_html=True)
    st.download_button(
        "📥 Download Filtered Dataset as CSV",
        data=filtered.to_csv(index=False),
        file_name="filtered_loan_default_data.csv",
        mime="text/csv",
        type="secondary",
        use_container_width=True,
        key="download_filtered_csv",
    )

    # Descriptive statistics
    st.markdown('<div class="section-title">📈 Descriptive Statistics</div>', unsafe_allow_html=True)
    st.markdown(render_styled_table(filtered[NUMERIC_COLS].describe().round(2), max_height=380, hide_index=False), unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown('<div class="section-title">🔗 Correlation Heatmap (Numeric Features)</div>', unsafe_allow_html=True)
    corr = filtered[NUMERIC_COLS].corr().round(2)
    corr_text_color = "white" if IS_DARK else "#1e293b"
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, "#EEF2FF" if not IS_DARK else "#1e1b4b"],
                    [0.5, "#818CF8" if not IS_DARK else "#312e81"],
                    [1, "#4F46E5" if not IS_DARK else "#6366f1"]],
        text=corr.values, texttemplate="%{text}",
        textfont=dict(size=11, color=corr_text_color),
    ))
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=500)
    st.plotly_chart(fig_corr, use_container_width=True, config=PLOTLY_CONFIG)

    # Additional: box plots for key features
    st.markdown('<div class="section-title">📦 Feature Distributions by Default Status</div>', unsafe_allow_html=True)
    box_cols = st.columns(3)
    for i, feat in enumerate(["Income", "CreditScore", "InterestRate"]):
        with box_cols[i]:
            fig_box = px.box(
                filtered, x="Default", y=feat,
                color="Default",
                color_discrete_map={0: CLR_PRIMARY, 1: CLR_DANGER},
                labels={"Default": "Default Status"},
            )
            fig_box.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False,
                                  xaxis_title="Default Status", yaxis_title=feat)
            st.plotly_chart(fig_box, use_container_width=True, config=PLOTLY_CONFIG)


# ═══════════════════════════════════════════════
# PAGE: About
# ═══════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown(f"""
    <div class="page-header">
        <h1 style="color:{CLR_HEADING};">ℹ️ About This Project</h1>
        <p>Loan Default Prediction using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    # Project overview
    st.markdown(f"""
    <div class="glass-panel">
        <h3 style="color:{CLR_HEADING}; margin-top:0;">🎯 Project Objective</h3>
        <p style="color:{CLR_CARD_TEXT}; line-height:1.9; font-size:1.02rem;">
            This project aims to predict whether a loan applicant is likely to <strong>default</strong>
            on their loan using a <strong>Decision Tree Classifier</strong>. The model analyses
            24 features including financial metrics, credit history, employment details, and
            personal information to provide a risk assessment. This tool can help financial
            institutions make more informed lending decisions and manage credit risk effectively.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tech cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="about-card">
            <div style="font-size:2.5rem; margin-bottom:12px;">📊</div>
            <h4 style="color:{CLR_HEADING}; margin-top:0;">Dataset</h4>
            <ul style="color:{CLR_CARD_TEXT}; line-height:2; padding-left:18px;">
                <li><strong>255,347</strong> loan records</li>
                <li><strong>18</strong> original features</li>
                <li><strong>9</strong> numeric + <strong>7</strong> categorical features</li>
                <li>Binary target: Default (0/1)</li>
                <li>Source: Loan_default.csv</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="about-card" style="animation-delay:0.15s;">
            <div style="font-size:2.5rem; margin-bottom:12px;">🤖</div>
            <h4 style="color:{CLR_HEADING}; margin-top:0;">Model</h4>
            <ul style="color:{CLR_CARD_TEXT}; line-height:2; padding-left:18px;">
                <li><strong>Decision Tree Classifier</strong></li>
                <li>StandardScaler preprocessing</li>
                <li>One-hot encoded categoricals</li>
                <li><strong>24</strong> input features after encoding</li>
                <li>Saved as Loan_Default.pkl + Scaler.pkl</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="about-card" style="animation-delay:0.3s;">
            <div style="font-size:2.5rem; margin-bottom:12px;">🛠️</div>
            <h4 style="color:{CLR_HEADING}; margin-top:0;">Tech Stack</h4>
            <ul style="color:{CLR_CARD_TEXT}; line-height:2; padding-left:18px;">
                <li><strong>Python</strong> — Core language</li>
                <li><strong>scikit-learn</strong> — ML model & scaler</li>
                <li><strong>Streamlit</strong> — Web framework</li>
                <li><strong>Plotly</strong> — Interactive charts</li>
                <li><strong>Pandas / NumPy</strong> — Data processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features explanation
    st.markdown('<div class="section-title">📋 Feature Descriptions</div>', unsafe_allow_html=True)

    feature_descriptions = pd.DataFrame({
        "Feature": ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
                     "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
                     "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
                     "HasDependents", "LoanPurpose", "HasCoSigner"],
        "Type": ["Numeric", "Numeric", "Numeric", "Numeric", "Numeric",
                  "Numeric", "Numeric", "Numeric", "Numeric",
                  "Categorical", "Categorical", "Categorical", "Categorical",
                  "Categorical", "Categorical", "Categorical"],
        "Description": [
            "Applicant's age in years",
            "Annual income in dollars",
            "Total loan amount requested",
            "Credit score (300–850)",
            "Number of months employed",
            "Number of open credit lines",
            "Loan interest rate (%)",
            "Loan term in months",
            "Debt-to-Income ratio (0–1)",
            "Education level (High School, Bachelor's, Master's, PhD)",
            "Employment type (Full-time, Part-time, Self-employed, Unemployed)",
            "Marital status (Single, Married, Divorced)",
            "Whether the applicant has a mortgage (Yes/No)",
            "Whether the applicant has dependents (Yes/No)",
            "Purpose of the loan (Auto, Business, Education, Home, Other)",
            "Whether the loan has a co-signer (Yes/No)",
        ],
    })
    st.markdown(render_styled_table(feature_descriptions, max_height=580), unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown('<div class="section-title">🔄 ML Pipeline</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="glass-panel" style="text-align:center; padding:32px;">
        <div style="display:flex; justify-content:center; align-items:center; gap:8px; flex-wrap:wrap;">
            <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.25);
                        border-radius:10px; padding:12px 20px;">
                <span style="color:{CLR_HEADING}; font-weight:600; font-size:0.9rem;">📥 Raw Data</span>
            </div>
            <span style="color:{CLR_TEXT_SEC}; font-size:1.4rem;">→</span>
            <div style="background:rgba(6,182,212,0.1); border:1px solid rgba(6,182,212,0.25);
                        border-radius:10px; padding:12px 20px;">
                <span style="color:{CLR_CYAN}; font-weight:600; font-size:0.9rem;">🔄 One-Hot Encoding</span>
            </div>
            <span style="color:{CLR_TEXT_SEC}; font-size:1.4rem;">→</span>
            <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.25);
                        border-radius:10px; padding:12px 20px;">
                <span style="color:{CLR_SUCCESS}; font-weight:600; font-size:0.9rem;">📏 StandardScaler</span>
            </div>
            <span style="color:{CLR_TEXT_SEC}; font-size:1.4rem;">→</span>
            <div style="background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.25);
                        border-radius:10px; padding:12px 20px;">
                <span style="color:{CLR_WARN}; font-weight:600; font-size:0.9rem;">🌲 Decision Tree</span>
            </div>
            <span style="color:{CLR_TEXT_SEC}; font-size:1.4rem;">→</span>
            <div style="background:rgba(244,63,94,0.1); border:1px solid rgba(244,63,94,0.25);
                        border-radius:10px; padding:12px 20px;">
                <span style="color:{CLR_DANGER}; font-weight:600; font-size:0.9rem;">🎯 Prediction</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# FOOTER (shown on all pages)
# ═══════════════════════════════════════════════
st.markdown(f"""
<div class="site-footer">
    <div class="footer-brand">🏦 Loan Default Prediction</div>
    <div class="footer-text">
        Dashboard · Decision Tree Classifier · StandardScaler<br>
        Built for ML Project · SEM 5
    </div>
    <div class="footer-badges">
        <span class="footer-badge">Python</span>
        <span class="footer-badge">Streamlit</span>
        <span class="footer-badge">scikit-learn</span>
        <span class="footer-badge">Plotly</span>
        <span class="footer-badge">Pandas</span>
    </div>
</div>
""", unsafe_allow_html=True)
