"""

app.py  –  Smart Crop Yield Prediction using AI

──────────────────────────────────────────────────

Features:

  • 8 branded AI models for yield prediction

  • Crop image scanning & analysis

  • Live weather API integration (Open-Meteo – no key needed)

  • Rich interactive dashboard & analytics

  • Dataset explorer with filters & downloads

Run:  python -m streamlit run app.py

"""

import streamlit as st

import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from PIL import Image

import os, warnings, time, hashlib

warnings.filterwarnings("ignore")

# Increase PIL decompression limit for high-res farm photos

Image.MAX_IMAGE_PIXELS = 200_000_000

from dataset        import (load_dataset, CROPS, SEASONS, SOIL_TYPES, STATES,

                             get_crop_names, get_soil_names, get_state_names, get_season_names,

                             get_crop_info, get_state_info)

from model          import (train_all_models, predict_yield, load_meta, is_trained,

                             get_model_names, get_model_info, MODEL_REGISTRY,

                             increment_open_count, check_retrain_needed,

                             mark_training_done, get_retrain_status)

from weather_api    import fetch_weather, fetch_weather_for_area, weather_code_text, detect_user_location

# ── Heavy modules: lazy-loaded per-page for faster startup ──
# image_analyzer, soil_analyzer, recommendation_engine, disease_detector,
# satellite_ndvi, utils  — imported on demand inside page blocks

@st.cache_resource(show_spinner=False)
def _get_recommendation_engine():
    from recommendation_engine import RecommendationEngine
    return RecommendationEngine(use_online=True)

@st.cache_resource(show_spinner=False)
def _get_disease_detector():
    from disease_detector import CropDiseaseDetector
    return CropDiseaseDetector()

# ──────────────────────────────────────────────────────────────────────────── #

#  Page Config                                                                  #

# ──────────────────────────────────────────────────────────────────────────── #

st.set_page_config(

    page_title="Smart Crop Yield AI",

    page_icon="🌾",

    layout="wide",

    initial_sidebar_state="expanded",

)

# ──────────────────────────────────────────────────────────────────────────── #

#  Custom CSS – Picturebook Forestry Theme with 3D effects                      #

# ──────────────────────────────────────────────────────────────────────────── #

def _get_theme_css(theme="light"):
    is_dark = theme == "dark"

    # ── Theme variable values ──
    vars_light = """
        --bg-primary: #F8F6F0;
        --bg-card: #FFFFFF;
        --bg-sidebar: linear-gradient(180deg, #F1EDE4 0%, #EAE5DA 100%);
        --bg-sidebar-border: #E0DDD5;
        --bg-input: #FFFFFF;
        --text-primary: #1B4332;
        --text-body: #344E41;
        --text-muted: #6B7F6F;
        --text-sidebar: #344E41;
        --border-card: #E8E5DD;
        --border-input: #E0DDD5;
        --border-section: #E8E5DD;
        --accent: #2D6A4F;
        --accent-light: #52B788;
        --accent-lighter: #95D5B2;
        --accent-bg: rgba(45, 106, 79, 0.08);
        --accent-hover: rgba(45, 106, 79, 0.12);
        --card-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
        --card-hover-shadow: 0 12px 40px rgba(45, 106, 79, 0.12), 0 4px 12px rgba(0, 0, 0, 0.04);
        --btn-bg: #2D6A4F;
        --btn-hover: #1B4332;
        --btn-text: #FFFFFF;
        --btn-shadow: 0 2px 8px rgba(45, 106, 79, 0.2);
        --weather-inline-bg: #F0F8F4;
        --weather-inline-border: #B7D7C8;
        --gradient-subtle-1: rgba(82, 183, 136, 0.06);
        --gradient-subtle-2: rgba(212, 163, 115, 0.06);
        --scrollbar-track: #F1EDE4;
        --scrollbar-thumb: #C5C0B6;
        --progress-track: #E8E5DD;
        --expander-bg: #FFFFFF;
    """

    vars_dark = """
        --bg-primary: #0F1A14;
        --bg-card: rgba(20, 35, 28, 0.85);
        --bg-sidebar: linear-gradient(180deg, #0D1710 0%, #0A130E 100%);
        --bg-sidebar-border: rgba(82, 183, 136, 0.15);
        --bg-input: rgba(15, 26, 20, 0.9);
        --text-primary: #D4E7D9;
        --text-body: #B0CCBA;
        --text-muted: #7A9B87;
        --text-sidebar: #B0CCBA;
        --border-card: rgba(82, 183, 136, 0.15);
        --border-input: rgba(82, 183, 136, 0.2);
        --border-section: rgba(82, 183, 136, 0.12);
        --accent: #52B788;
        --accent-light: #74C99B;
        --accent-lighter: #95D5B2;
        --accent-bg: rgba(82, 183, 136, 0.08);
        --accent-hover: rgba(82, 183, 136, 0.15);
        --card-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
        --card-hover-shadow: 0 12px 40px rgba(82, 183, 136, 0.12), 0 4px 12px rgba(0, 0, 0, 0.2);
        --btn-bg: #2D6A4F;
        --btn-hover: #52B788;
        --btn-text: #FFFFFF;
        --btn-shadow: 0 2px 8px rgba(45, 106, 79, 0.4);
        --weather-inline-bg: rgba(20, 40, 30, 0.8);
        --weather-inline-border: rgba(82, 183, 136, 0.3);
        --gradient-subtle-1: rgba(82, 183, 136, 0.04);
        --gradient-subtle-2: rgba(45, 106, 79, 0.04);
        --scrollbar-track: #0D1710;
        --scrollbar-thumb: #2D4A3A;
        --progress-track: rgba(82, 183, 136, 0.12);
        --expander-bg: rgba(20, 35, 28, 0.85);
    """

    theme_vars = vars_dark if is_dark else vars_light

    return f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {{
    {theme_vars}
}}

/* ══════════════════════════════════════════════════════════════════
   ROOT
   ══════════════════════════════════════════════════════════════════ */

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes floatLeaf {{
    0%   {{ transform: translateY(0) rotate(0deg); }}
    50%  {{ transform: translateY(-6px) rotate(2deg); }}
    100% {{ transform: translateY(0) rotate(0deg); }}
}}

@keyframes resultGlow {{
    0%   {{ box-shadow: 0 8px 32px rgba(45, 106, 79, 0.15); }}
    100% {{ box-shadow: 0 12px 48px rgba(45, 106, 79, 0.25); }}
}}

[data-testid="stAppViewContainer"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-body);
    background: var(--bg-primary);
    background-attachment: fixed;
}}

[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(ellipse at 20% 0%, var(--gradient-subtle-1) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, var(--gradient-subtle-2) 0%, transparent 50%);
    opacity: 1;
}}

[data-testid="stAppViewContainer"]::after {{ display: none; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: var(--bg-sidebar);
    border-right: 1px solid var(--bg-sidebar-border);
    box-shadow: 2px 0 12px rgba(0, 0, 0, 0.04);
}}

[data-testid="stSidebar"] * {{ color: var(--text-sidebar) !important; }}

/* ══════════════════════════════════════════════════════════════════
   HERO BANNER
   ══════════════════════════════════════════════════════════════════ */
.hero {{
    background: linear-gradient(135deg, #2D6A4F 0%, #40916C 50%, #52B788 100%);
    border: none;
    border-radius: 20px;
    padding: 48px 44px;
    margin-bottom: 32px;
    box-shadow: 0 8px 32px rgba(45, 106, 79, 0.20), 0 2px 8px rgba(0, 0, 0, 0.06);
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.8s cubic-bezier(0.25, 0.1, 0.25, 1) both;
}}

.hero::after {{
    content: "";
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.06);
    pointer-events: none;
}}

.hero::before {{
    content: "🌾🌿🌱☀️";
    position: absolute;
    top: 20px; right: 24px;
    font-size: 2rem;
    opacity: 0.3;
    letter-spacing: 8px;
    animation: floatLeaf 4s ease-in-out infinite;
}}

.hero h1 {{
    margin: 0;
    font-size: 2.6rem;
    font-weight: 800;
    font-family: 'Inter', sans-serif;
    color: #FFFFFF;
    letter-spacing: -0.5px;
    line-height: 1.15;
}}

.hero p {{
    margin: 12px 0 0;
    font-size: 1.1rem;
    color: rgba(255, 255, 255, 0.85);
    font-weight: 400;
    letter-spacing: 0.2px;
    line-height: 1.6;
}}

/* ══════════════════════════════════════════════════════════════════
   CARDS
   ══════════════════════════════════════════════════════════════════ */
.card {{
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: var(--card-shadow);
    transition: all 0.4s cubic-bezier(0.25, 0.1, 0.25, 1);
    animation: fadeInUp 0.6s cubic-bezier(0.25, 0.1, 0.25, 1) both;
    color: var(--text-body);
}}

.card:hover {{
    transform: translateY(-4px);
    box-shadow: var(--card-hover-shadow);
}}

.metric-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: var(--card-shadow);
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.6s cubic-bezier(0.25, 0.1, 0.25, 1) both;
}}

.metric-card::after {{
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent-light), var(--accent-lighter));
    border-radius: 16px 16px 0 0;
}}

.metric-card:hover {{
    transform: translateY(-6px) scale(1.02);
    box-shadow: var(--card-hover-shadow);
}}

.metric-value {{
    font-family: 'DM Sans', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px;
}}

.metric-label {{
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 8px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}}

/* ── Result box ── */
.result-box {{
    background: linear-gradient(135deg, #2D6A4F, #40916C);
    border-radius: 20px;
    padding: 48px;
    text-align: center;
    border: none;
    animation: resultGlow 3s ease-in-out infinite alternate;
    position: relative;
    overflow: hidden;
}}

.result-box::before {{
    content: '🌾 Predicted Yield';
    position: absolute;
    top: 14px; left: 20px;
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    color: rgba(255,255,255,0.6);
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 600;
}}

.result-box::after {{
    content: "";
    position: absolute;
    top: -50%; right: -30%;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
    pointer-events: none;
}}

.result-value {{
    font-family: 'DM Sans', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -1px;
}}

.result-label {{
    font-size: 1.1rem;
    color: rgba(255, 255, 255, 0.8);
    margin-top: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 3px;
}}

/* ── Model card ── */
.model-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-left: 4px solid var(--accent-light);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: var(--card-shadow);
    transition: all 0.35s cubic-bezier(0.25, 0.1, 0.25, 1);
    color: var(--text-body);
}}

.model-card:hover {{
    border-left-color: var(--accent);
    transform: translateX(6px);
    box-shadow: var(--card-hover-shadow);
}}

/* ── Weather card ── */
.weather-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: var(--card-shadow);
    transition: all 0.35s cubic-bezier(0.25, 0.1, 0.25, 1);
}}

.weather-card:hover {{
    box-shadow: var(--card-hover-shadow);
    transform: translateY(-4px);
}}

.weather-value {{
    font-family: 'DM Sans', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text-primary);
}}

.weather-label {{
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 8px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}}

/* ── Scan card ── */
.scan-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 24px;
    box-shadow: var(--card-shadow);
    color: var(--text-body);
}}

/* ══════════════════════════════════════════════════════════════════
   SECTION HEADERS
   ══════════════════════════════════════════════════════════════════ */
.section-header {{
    font-family: 'Inter', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    padding-bottom: 14px;
    border-bottom: 2px solid var(--border-section);
    margin-bottom: 28px;
    letter-spacing: -0.3px;
    position: relative;
    animation: fadeInUp 0.5s cubic-bezier(0.25, 0.1, 0.25, 1) both;
}}

.section-header::after {{
    content: "";
    position: absolute;
    bottom: -2px; left: 0;
    width: 60px; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent-light));
    border-radius: 2px;
}}

.sub-header {{
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 16px;
    letter-spacing: 0.3px;
}}

/* ══════════════════════════════════════════════════════════════════
   BUTTONS – Refined (Apple-style)
   ══════════════════════════════════════════════════════════════════ */
[data-testid="stAppViewContainer"] div.stButton > button {{
    background: var(--btn-bg) !important;
    color: var(--btn-text) !important;
    border: none !important;
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    padding: 10px 24px;
    font-size: 0.9rem;
    letter-spacing: 0.2px;
    box-shadow: var(--btn-shadow);
    transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
    cursor: pointer;
}}

[data-testid="stAppViewContainer"] div.stButton > button p,
[data-testid="stAppViewContainer"] div.stButton > button span,
[data-testid="stAppViewContainer"] div.stButton > button div {{
    color: var(--btn-text) !important;
}}

[data-testid="stAppViewContainer"] div.stButton > button:hover {{
    background: var(--btn-hover) !important;
    color: var(--btn-text) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(45, 106, 79, 0.3);
}}

[data-testid="stAppViewContainer"] div.stButton > button:active {{
    transform: translateY(0);
    box-shadow: 0 2px 6px rgba(45, 106, 79, 0.2);
}}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] div.stButton > button {{
    background: transparent !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    border-radius: 0 10px 10px 0 !important;
    color: var(--text-sidebar) !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 10px 16px !important;
    text-align: left !important;
    margin-bottom: 4px !important;
    box-shadow: none !important;
    transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1) !important;
    letter-spacing: 0.2px;
}}

[data-testid="stSidebar"] div.stButton > button:hover {{
    background: var(--accent-bg) !important;
    color: var(--text-primary) !important;
    border-left: 3px solid var(--accent-light) !important;
    transform: translateX(4px) !important;
}}

/* ── Active nav (primary) ── */
[data-testid="stSidebar"] div.stButton > button[kind="primary"],
[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-primary"] {{
    background: var(--accent-hover) !important;
    border: none !important;
    border-left: 3px solid var(--accent) !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}}

[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover,
[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background: var(--accent-hover) !important;
    transform: translateX(4px) !important;
}}

/* ── Retrain button ── */
[data-testid="stSidebar"] button[key*="retrain"] {{
    background: rgba(212, 163, 115, 0.12) !important;
    border: 1px solid #D4A373 !important;
    border-left: 3px solid #D4A373 !important;
    color: {'#D4A373' if is_dark else '#8B6914'} !important;
    border-radius: 0 10px 10px 0 !important;
}}
[data-testid="stSidebar"] button[key*="retrain"]:hover {{
    background: rgba(212, 163, 115, 0.2) !important;
}}

/* ══════════════════════════════════════════════════════════════════
   FORM ELEMENTS
   ══════════════════════════════════════════════════════════════════ */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea {{
    background: var(--bg-input) !important;
    border: 1.5px solid var(--border-input) !important;
    color: var(--text-body) !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:focus-within {{
    border-color: var(--accent-light) !important;
    box-shadow: 0 0 0 3px rgba(82, 183, 136, 0.12), 0 1px 4px rgba(0, 0, 0, 0.04) !important;
}}

label {{
    color: var(--accent) !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    font-size: 0.88rem !important;
}}

/* ── Sliders ── */
div[data-baseweb="slider"] div[role="slider"] {{
    background: var(--accent) !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(45, 106, 79, 0.3) !important;
    border-radius: 50% !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 2px solid var(--border-section);
    padding-bottom: 0;
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 10px 10px 0 0;
    color: var(--text-muted);
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    padding: 10px 20px;
    transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
    font-size: 0.9rem;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: var(--accent);
    background: var(--accent-bg);
}}

.stTabs [aria-selected="true"] {{
    background: var(--accent-bg) !important;
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent);
    border-radius: 10px 10px 0 0;
}}

/* ── Progress Bar ── */
[data-testid="stProgress"] > div > div > div {{
    background-color: var(--progress-track) !important;
    border-radius: 10px !important;
}}
[data-testid="stProgress"] > div > div > div > div {{
    background: linear-gradient(90deg, var(--accent), var(--accent-light)) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}}

/* ── Weather inline ── */
.weather-inline {{
    background: var(--weather-inline-bg);
    border: 1px solid var(--weather-inline-border);
    border-left: 4px solid var(--accent-light);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 15px;
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text-body);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}}

/* ── Streamlit element overrides ── */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 {{
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}}

[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] div {{
    color: var(--text-body);
}}

[data-testid="stExpander"] {{
    border: 1px solid var(--border-card) !important;
    border-radius: 12px !important;
    background: var(--expander-bg) !important;
}}

[data-testid="stMetric"] {{
    background: var(--bg-card);
    border-radius: 12px;
    padding: 12px;
    border: 1px solid var(--border-card);
}}

/* ── Theme toggle styling ── */
[data-testid="stSidebar"] .stToggle label span {{
    color: var(--text-sidebar) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}}

/* ── Hide branding ── */
#MainMenu, footer, header {{ visibility: hidden; }}
html {{ scroll-behavior: smooth; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: var(--scrollbar-track); }}
::-webkit-scrollbar-thumb {{ background: var(--scrollbar-thumb); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ opacity: 0.8; }}

</style>
"""

# ── Apply theme ──
if "app_theme" not in st.session_state:
    st.session_state.app_theme = "light"
st.markdown(_get_theme_css(st.session_state.app_theme), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────── #

#  Session state initialisation                                                 #

# ──────────────────────────────────────────────────────────────────────────── #

if "weather" not in st.session_state:

    st.session_state.weather = None

if "scan_result" not in st.session_state:

    st.session_state.scan_result = None

if "df" not in st.session_state:

    st.session_state.df = None

if "meta" not in st.session_state:

    st.session_state.meta = None

if "initialized" not in st.session_state:

    st.session_state.initialized = False

if "data_info" not in st.session_state:

    st.session_state.data_info = None

if "user_location" not in st.session_state:

    st.session_state.user_location = None

# ──────────────────────────────────────────────────────────────────────────── #

#  Auto-initialization: download data + train models on first run               #

#  Now with auto-retrain on 5 opens / new dataset / new PC                      #

# ──────────────────────────────────────────────────────────────────────────── #

if not st.session_state.initialized:

    # Track app opens for auto-retrain

    _open_num = increment_open_count()

    _init = st.empty()

    with _init.container():

        st.markdown(

            "<div style='text-align:center;padding:60px 20px;'>"

            "<div style='font-size:4rem;'>🌾</div>"

            "<h1 style='color:#2D6A4F;margin:12px 0;font-family:Inter,sans-serif;'>Smart Crop Yield AI</h1>"

            "<p style='color:#6B7F6F;font-size:1.05rem;'>Initializing — please wait…</p>"

            "</div>", unsafe_allow_html=True

        )

        prog = st.progress(0, text="Preparing…")

        # Step 1: Load / download dataset

        prog.progress(5, text="📥 Loading crop production data from Government of India…")

        try:

            df, data_info = load_dataset()

            st.session_state.df = df

            st.session_state.data_info = data_info

        except Exception as e:

            st.error(f"Failed to load dataset: {e}")

            st.stop()

        prog.progress(30, text=f"✅ Dataset loaded — {len(df):,} records ({data_info.get('source','')[:40]}…)")

        # Step 2: Detect user location

        prog.progress(32, text="\U0001f4cd Detecting your location\u2026")

        try:

            loc = detect_user_location()

            st.session_state.user_location = loc

            if loc and loc.get("city"):

                prog.progress(34, text=f"\u2705 Location: {loc['city']}, {loc.get('state','')}")

        except Exception:

            pass

        # Step 3: Smart auto-retrain — checks open count, dataset, PC

        _needs_retrain, _retrain_reason = check_retrain_needed(df)

        _reason_msgs = {

            "first_run": "\U0001f9e0 First run \u2014 training 8 AI models \u00d7 3 rounds\u2026",

            "open_count": "\U0001f504 Auto-retrain triggered (5+ app opens since last train)\u2026",

            "new_dataset": "\U0001f4ca Dataset change detected \u2014 retraining models\u2026",

            "new_machine": "\U0001f4bb New PC detected \u2014 retraining models for this machine\u2026",

        }

        if is_trained() and not _needs_retrain:

            prog.progress(80, text="\U0001f9e0 Loading previously trained AI models\u2026")

            st.session_state.meta = load_meta()

        else:

            _msg = _reason_msgs.get(_retrain_reason, "\U0001f9e0 Training 8 AI models \u00d7 3 rounds\u2026")

            prog.progress(35, text=_msg)

            def _train_progress(rnd, total, avg_r2):

                pct = 35 + int(50 * rnd / total)

                prog.progress(pct, text=f"\U0001f504 Round {rnd}/{total} complete (avg R\u00b2 = {avg_r2})")

            st.session_state.meta = train_all_models(df, n_rounds=3,

                                                      progress_cb=_train_progress)

            mark_training_done(df)

        prog.progress(100, text="✅ All systems ready!")

        st.session_state.initialized = True

    _init.empty()

    st.rerun()

# ──────────────────────────────────────────────────────────────────────────── #

#  Sidebar                                                                      #

# ──────────────────────────────────────────────────────────────────────────── #

if "page" not in st.session_state:

    st.session_state.page = "🏠 Dashboard"

NAV_ITEMS = [

    ("🏠", "Dashboard",      "🏠 Dashboard"),

    ("🔮", "Predict Yield",  "🔮 Predict Yield"),

    ("📷", "Image Scanner",  "📷 Image Scanner"),

    ("🌦️", "Weather Intel",  "🌦️ Weather Intel"),

    ("📊", "Model Hub",      "📊 Model Hub"),

    ("🗄️", "Data Explorer",  "🗄️ Data Explorer"),

]

def set_page(p):

    st.session_state.page = p

with st.sidebar:

    st.markdown(

        "<div style='text-align:center;padding:16px 0 8px;'>"

        "<span style='font-size:2rem;'>🌾</span><br>"

        "<span style='font-size:1.15rem;font-weight:800;color:#2D6A4F;letter-spacing:0.5px;font-family:Inter,sans-serif;'>"

        "Smart Crop Yield AI</span>"

        "</div>"

        "<div style='border-bottom:1px solid var(--bg-sidebar-border);margin:4px 0 8px;'></div>",

        unsafe_allow_html=True,

    )

    # ── Theme toggle ──
    def _toggle_theme():
        st.session_state.app_theme = "dark" if st.session_state.app_theme == "light" else "light"

    _is_dark = st.session_state.get("app_theme", "light") == "dark"
    st.toggle("🌙 Dark Mode", value=_is_dark, key="theme_toggle", on_change=_toggle_theme)

    st.markdown("<div style='border-bottom:1px solid var(--bg-sidebar-border);margin:4px 0 12px;'></div>",
                unsafe_allow_html=True)

    # ── Navigation buttons (on_click avoids double-rerun) ──

    for icon, label, page_key in NAV_ITEMS:

        is_active = st.session_state.page == page_key

        btn_type = "primary" if is_active else "secondary"

        st.button(f"{icon}  {label}", key=f"nav_{label}",

                  width='stretch', type=btn_type,

                  on_click=set_page, args=(page_key,))

    st.markdown("<div style='border-bottom:1px solid var(--bg-sidebar-border);margin:8px 0 12px;'></div>",

                unsafe_allow_html=True)

    # ── Data source badge ──

    info = st.session_state.get("data_info")

    if info:

        src_short = "🏛️ Govt of India" if "Government" in info.get("source", "") else "📊 Synthetic"

        yr = f" ({info['year_min']}–{info['year_max']})" if info.get("year_min") else ""

        st.markdown(

            f"<div style='font-size:0.72rem;color:#2D6A4F;padding:4px 8px;"

            f"background:rgba(45,106,79,0.08);border-radius:8px;margin-bottom:8px;'>"

            f"{src_short}{yr}<br>"

            f"{info.get('n_records',0):,} records · {info.get('n_crops',0)} crops</div>",

            unsafe_allow_html=True,

        )

    # ── Retrain button ──

    if st.button("🔄  Retrain Models", key="retrain_models", width='stretch'):

        with st.spinner("Refreshing data & retraining…"):

            df, data_info = load_dataset(force_refresh=True)

            st.session_state.df = df

            st.session_state.data_info = data_info

            st.session_state.meta = train_all_models(df)

            mark_training_done(df)

        st.success("Retrained on fresh data! ✅")

        st.session_state._retrain_status = get_retrain_status()

        st.rerun()

    # ── Auto-retrain status (cached in session to avoid disk read every rerun) ──

    if "_retrain_status" not in st.session_state:

        st.session_state._retrain_status = get_retrain_status()

    _rs = st.session_state._retrain_status

    _next_in = _rs["opens_until_retrain"]

    _next_lbl = f"in {_next_in} opens" if _next_in > 0 else "next start"

    st.markdown(

        f"<div style='font-size:0.68rem;color:#6B7F6F;padding:4px 8px;"

        f"background:rgba(45,106,79,0.06);border-radius:8px;margin:6px 0;'>"

        f"🔄 Auto-retrain: <b>{_next_lbl}</b><br>"

        f"Opens: {_rs['total_opens']} | Since train: {_rs['opens_since_train']}<br>"

        f"💻 {'Same PC ✅' if _rs['same_machine'] else 'Different PC ⚠️'}</div>"

        "<div style='border-bottom:1px solid var(--bg-sidebar-border);margin:10px 0;'></div>",

        unsafe_allow_html=True,

    )

    st.markdown(

        "<div style='font-size:0.68rem;color:#6B7F6F;text-align:center;padding-top:4px;'>"

        "Smart Crop Yield AI v4.1 ML-Trained<br>Semester Project · Auto-Retrain Enabled</div>",

        unsafe_allow_html=True,

    )

page = st.session_state.page

# ──────────────────────────────────────────────────────────────────────────── #

#  Helpers                                                                      #

# ──────────────────────────────────────────────────────────────────────────── #

def need_data():

    if st.session_state.df is None:

        st.warning("⏳ Dataset is loading… If this persists, click **🔄 Retrain Models** in the sidebar.")

        return True

    return False

def need_model():

    if st.session_state.meta is None:

        st.warning("⏳ Models are loading… If this persists, click **🔄 Retrain Models** in the sidebar.")

        return True

    return False

@st.cache_data(show_spinner=False)
def metric_card(value, label):

    return (f"<div class='metric-card'>"

            f"<div class='metric-value'>{value}</div>"

            f"<div class='metric-label'>{label}</div></div>")

@st.cache_data(ttl=600, show_spinner=False)

def _cached_weather(*args, **kwargs):

    """Cache weather API calls for 10 minutes to avoid repeated requests."""

    return fetch_weather(*args, **kwargs)

@st.cache_data(ttl=600, show_spinner=False)

def _cached_weather_area(*args, **kwargs):

    """Cache area-based weather for 10 minutes."""

    return fetch_weather_for_area(*args, **kwargs)

def _prepare_image_for_analysis(img):

    """Downscale very large images to avoid slow analysis (preserves EXIF via separate extraction)."""

    MAX_DIM = 2048

    w, h = img.size

    if max(w, h) > MAX_DIM:

        ratio = MAX_DIM / max(w, h)

        new_size = (int(w * ratio), int(h * ratio))

        img = img.resize(new_size, Image.LANCZOS)

    return img

# ╔══════════════════════════════════════════════════════════════════════════╗ #

# ║  PAGE 1 – DASHBOARD                                                      ║ #

# ╚══════════════════════════════════════════════════════════════════════════╝ #

if page == "🏠 Dashboard":

    st.markdown(

        "<div class='hero'>"

        "<h1>🌾 Smart Crop Yield Prediction</h1>"

        "<p>AI-powered crop yield forecasting with 8 advanced machine-learning models, "

        "image scanning, and live weather intelligence.</p>"

        "</div>", unsafe_allow_html=True

    )

    if need_data():

        st.stop()

    df = st.session_state.df

    # ── Data source banner ──

    info = st.session_state.get("data_info")

    if info:

        yr_text = f" &nbsp;|&nbsp; Years: {info['year_min']}–{info['year_max']}" if info.get("year_min") else ""

        st.markdown(

            f"<div class='card' style='border-color:#B7D7C8;padding:14px 20px;'>"

            f"<span style='font-size:0.82rem;color:#2D6A4F;'>📂 <b>Data Source:</b> {info.get('source','')}</span><br>"

            f"<span style='font-size:0.76rem;color:#6B7F6F;'>"

            f"{info.get('n_records',0):,} records &nbsp;|&nbsp; "

            f"{info.get('n_crops',0)} crops &nbsp;|&nbsp; "

            f"{info.get('n_states',0)} states{yr_text}</span>"

            f"</div>", unsafe_allow_html=True

        )

    # ── KPI row ──

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1: st.markdown(metric_card(f"{len(df):,}", "Total Records"), unsafe_allow_html=True)

    with c2: st.markdown(metric_card(df["Crop"].nunique(), "Crop Types"), unsafe_allow_html=True)

    with c3: st.markdown(metric_card(df["State"].nunique(), "Indian States"), unsafe_allow_html=True)

    with c4: st.markdown(metric_card(f"{df['Yield_ton_per_ha'].mean():.2f}", "Avg Yield (t/ha)"), unsafe_allow_html=True)

    with c5:

        n_models = len(MODEL_REGISTRY)

        st.markdown(metric_card(n_models, "AI Models"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dashboard Charts (cached to avoid rebuilding on every rerun) ──
    @st.cache_data(show_spinner=False)
    def _build_dashboard_charts(_n_records):
        _df = st.session_state.df
        avg_y = _df.groupby("Crop")["Yield_ton_per_ha"].mean().reset_index().sort_values("Yield_ton_per_ha")
        f1 = px.bar(avg_y, x="Yield_ton_per_ha", y="Crop", orientation="h",
                     color="Yield_ton_per_ha", color_continuous_scale="Greens",
                     title="📊 Average Yield by Crop (t/ha)", template="plotly_white")
        f1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          coloraxis_showscale=False, showlegend=False)

        cc = _df["Crop"].value_counts().reset_index()
        cc.columns = ["Crop", "Count"]
        f2 = px.pie(cc, values="Count", names="Crop", title="🥧 Crop Distribution",
                      color_discrete_sequence=px.colors.sequential.Greens_r,
                      template="plotly_white")
        f2.update_layout(paper_bgcolor="rgba(0,0,0,0)")

        f3 = px.box(_df, x="Season", y="Yield_ton_per_ha", color="Season",
                       title="📦 Yield by Season", template="plotly_white",
                       color_discrete_sequence=px.colors.qualitative.Pastel)
        f3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           showlegend=False)

        samp = _df.sample(min(1000, len(_df)), random_state=42)
        f4 = px.scatter(samp, x="Temperature", y="Yield_ton_per_ha",
                          color="Crop", opacity=0.55, title="🌡️ Temperature vs Yield",
                          template="plotly_white")
        f4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        num_cols = ["Nitrogen","Phosphorus","Potassium","pH","Temperature",
                    "Humidity","Rainfall","Fertilizer","Pesticide","Yield_ton_per_ha"]
        corr = _df[num_cols].corr()
        f5 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                         title="Correlation Heatmap", template="plotly_white")
        f5.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        return f1, f2, f3, f4, f5

    _f1, _f2, _f3, _f4, _f5 = _build_dashboard_charts(len(df))

    # ── Charts row 1 ──

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(_f1, width='stretch')

    with col2:

        st.plotly_chart(_f2, width='stretch')

    # ── Charts row 2 ──

    col3, col4 = st.columns(2)

    with col3:

        st.plotly_chart(_f3, width='stretch')

    with col4:

        st.plotly_chart(_f4, width='stretch')

    # ── Correlation heatmap ──

    st.markdown("#### 🔗 Feature Correlation Matrix")

    st.plotly_chart(_f5, width='stretch')

# ╔══════════════════════════════════════════════════════════════════════════╗ #

# ║  PAGE 2 – PREDICT YIELD                                                  ║ #

# ╚══════════════════════════════════════════════════════════════════════════╝ #

elif page == "🔮 Predict Yield":

    from soil_analyzer import analyze_soil, get_crop_suitability, get_soil_suggestions

    from utils import get_yield_category

    _recommendation_engine = _get_recommendation_engine()

    st.markdown("<div class='section-header'>🔮 AI Crop Yield Prediction</div>",

                unsafe_allow_html=True)

    if need_model():

        st.stop()

    meta = st.session_state.meta

    # ── Model selector ──

    st.markdown("<div class='sub-header'>🤖 Choose an AI Model</div>", unsafe_allow_html=True)

    model_names = get_model_names()

    best = meta.get("best_name", model_names[0])

    model_cols = st.columns(4)

    for i, name in enumerate(model_names):

        info = get_model_info(name)

        r2 = meta["results"].get(name, {}).get("R2", "—")

        with model_cols[i % 4]:

            badge = " 🏆" if name == best else ""

            st.markdown(

                f"<div class='model-card'>"

                f"<b>{info.get('icon','')} {name}{badge}</b><br>"

                f"<span style='font-size:0.75rem;color:#40916C;'>{info.get('family','')}</span><br>"

                f"<span style='font-size:0.8rem;'>R² = <b style='color:#8bc34a;'>{r2}</b></span>"

                f"</div>", unsafe_allow_html=True

            )

    selected_model = st.selectbox(

        "Select model for prediction",

        model_names,

        index=model_names.index(best) if best in model_names else 0,

        format_func=lambda x: f"{get_model_info(x).get('icon','')} {x} — {get_model_info(x).get('family','')}",

    )

    sel_info = get_model_info(selected_model)

    st.markdown(f"<div class='card'>ℹ️ <b>{selected_model}</b>: {sel_info.get('desc','')}</div>",

                unsafe_allow_html=True)

    st.markdown("---")

    # ── Location & Weather (outside form so it auto-updates) ──

    st.markdown("<div class='sub-header'>🌍 Location & Live Weather</div>",

                unsafe_allow_html=True)

    st.markdown("<span style='font-size:0.82rem;color:#bbdefb;'>Select a state and optionally "

                "enter a city/area — weather will auto-fill the Climate sliders below.</span>",

                unsafe_allow_html=True)

    # Auto-detect state/area from IP geolocation

    _all_states = get_state_names()

    _auto_idx = 0

    _auto_area = ""

    _loc = st.session_state.get("user_location")

    if _loc:

        ms = _loc.get("matched_state", "")

        if ms and ms in _all_states:

            _auto_idx = _all_states.index(ms)

        _auto_area = _loc.get("city", "")

    loc_c1, loc_c2, loc_c3 = st.columns([2, 3, 1])

    with loc_c1:

        pred_state = st.selectbox("State", _all_states, index=_auto_idx,

                                  key="pred_form_state")

    with loc_c2:

        pred_area = st.text_input("Area / City / Village (optional)",

                                  value=_auto_area,

                                  placeholder="e.g. Kanpur, Nagpur, Coimbatore\u2026",

                                  key="pred_form_area")

    with loc_c3:

        st.markdown("<br>", unsafe_allow_html=True)

        fetch_wx = st.button("🌍 Fetch", key="pred_fetch_wx")

    # Auto-fetch weather for the selected state/area

    _cache_key = f"{pred_state}_{pred_area.strip()}"

    if "pred_wx_cache_key" not in st.session_state:

        st.session_state.pred_wx_cache_key = ""

    if "pred_wx" not in st.session_state:

        st.session_state.pred_wx = None

    if fetch_wx or st.session_state.pred_wx_cache_key != _cache_key:

        try:

            if pred_area.strip():

                w = _cached_weather_area(pred_area.strip(), pred_state)

            else:

                w = _cached_weather(pred_state)

                w["location"] = pred_state

            st.session_state.pred_wx = w

            st.session_state.pred_wx_cache_key = _cache_key

        except Exception:

            st.session_state.pred_wx = None

    # Defaults from weather or state profile

    w = st.session_state.pred_wx

    if w:

        weather_temp = float(w["temperature"])

        weather_hum  = float(w["humidity"])

        weather_rain = int(w["rainfall"])

        loc_label = w.get("location", pred_state)

        wx_code = weather_code_text(w.get("weather_code", 0))

        st.markdown(

            f"<div class='weather-inline'>"

            f"<b>📍 {loc_label}</b> — {wx_code} &nbsp;&nbsp;"

            f"🌡️ {w['temperature']}°C &nbsp;|&nbsp; 💧 {w['humidity']}% &nbsp;|&nbsp; "

            f"🌧️ ~{w['rainfall']} mm/yr &nbsp;|&nbsp; 💨 {w.get('wind_speed',0)} km/h"

            f"</div>", unsafe_allow_html=True

        )

    else:

        si = get_state_info(pred_state)

        weather_temp = float(si["temp"]) if si else 25.0

        weather_hum  = 60.0

        weather_rain = int(si["rain"]) if si else 800

    st.markdown("---")

    # ── Input form ──

    with st.form("predict_form"):

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown("##### 🌱 Crop Info")

            crop   = st.selectbox("Crop", get_crop_names())

            season = st.selectbox("Season", get_season_names())

            soil   = st.selectbox("Soil Type", get_soil_names())

            area   = st.number_input("Area (ha)", 0.5, 500.0, 10.0, 0.5)

        with c2:

            st.markdown("##### 🌤️ Climate  <span style='font-size:0.7rem;color:#90caf9;'>(auto-filled from weather)</span>",

                        unsafe_allow_html=True)

            temp = st.slider("Temperature (°C)", 5.0, 50.0, min(50.0, max(5.0, weather_temp)), 0.5)

            hum  = st.slider("Humidity (%)", 20.0, 99.0, min(99.0, max(20.0, weather_hum)), 0.5)

            rain = st.slider("Rainfall (mm/year)", 100, 4000, min(4000, max(100, weather_rain)), 50)

        with c3:

            st.markdown("##### 🧪 Soil Nutrients")

            nitrogen   = st.slider("Nitrogen (kg/ha)", 5, 300, 80, 5)

            phosphorus = st.slider("Phosphorus (kg/ha)", 5, 150, 45, 5)

            potassium  = st.slider("Potassium (kg/ha)", 5, 400, 40, 5)

            ph         = st.slider("pH", 3.5, 9.5, 6.5, 0.1)

        with c4:

            st.markdown("##### 🧴 Inputs")

            fert = st.slider("Fertilizer (kg/ha)", 10, 400, 120, 5)

            pest = st.slider("Pesticide (kg/ha)", 0.5, 25.0, 5.0, 0.5)

            irr  = st.radio("Irrigation", ["Yes", "No"], horizontal=True)

        submitted = st.form_submit_button("🌾 Predict Yield")

    if submitted:

        inputs = {

            "Crop": crop, "Season": season, "State": pred_state, "Soil_Type": soil,

            "Area_ha": area, "Nitrogen": nitrogen, "Phosphorus": phosphorus,

            "Potassium": potassium, "pH": ph, "Temperature": temp,

            "Humidity": hum, "Rainfall": rain, "Fertilizer": fert,

            "Pesticide": pest, "Irrigation": 1 if irr == "Yes" else 0,

        }

        with st.spinner(f"Running **{selected_model}**…"):

            result = predict_yield(inputs, model_name=selected_model)

        total = result * area

        st.markdown("<br>", unsafe_allow_html=True)

        _, rc, _ = st.columns([1, 1.5, 1])

        with rc:

            st.markdown(

                f"<div class='result-box'>"

                f"<div style='color:#a5d6a7;font-size:0.95rem;'>Predicted by <b>{selected_model}</b></div>"

                f"<div class='result-value'>{result:.3f}</div>"

                f"<div class='result-label'>tonnes per hectare &nbsp;|&nbsp; "

                f"Total: <b>{total:,.1f} t</b> for {area} ha</div>"

                f"</div>", unsafe_allow_html=True

            )

        # ── Gauge ──

        crop_info = get_crop_info(crop)

        if crop_info:

            y_lo, y_hi = crop_info["yield_range"]

            fig = go.Figure(go.Indicator(

                mode="gauge+number+delta",

                value=result,

                delta={"reference": (y_lo + y_hi) / 2, "valueformat": ".2f"},

                title={"text": f"{crop} — Yield vs Average", "font": {"color": "#a5d6a7"}},

                gauge=dict(

                    axis=dict(range=[0, y_hi * 1.3], tickcolor="#a5d6a7"),

                    bar=dict(color="#43a047"),

                    steps=[

                        {"range": [0, y_lo], "color": "#1a2a1a"},

                        {"range": [y_lo, y_hi], "color": "#1b5e20"},

                        {"range": [y_hi, y_hi * 1.3], "color": "#2e7d32"},

                    ],

                    threshold=dict(line=dict(color="#8bc34a", width=3),

                                   thickness=0.8, value=(y_lo + y_hi) / 2),

                ),

                number=dict(suffix=" t/ha", font=dict(color="#a5d6a7")),

            ))

            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#a5d6a7"), height=300)

            st.plotly_chart(fig, width='stretch')

        # ── Feature importance ──

        res = meta["results"].get(selected_model, {})

        fi  = res.get("feat_import")

        if fi:

            fi_df = pd.DataFrame({"Feature": list(fi.keys()),

                                  "Importance": list(fi.values())}).sort_values("Importance")

            fig2 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",

                          color="Importance", color_continuous_scale="Greens",

                          title=f"Feature Importance — {selected_model}", template="plotly_white")

            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",

                               coloraxis_showscale=False)

            st.plotly_chart(fig2, width='stretch')

        # ── 🧪 Soil Health Analysis (from soil_analyzer module) ──
        st.markdown("---")
        st.markdown("#### 🧪 Soil Health Analysis")
        try:
            soil_result = analyze_soil(
                nitrogen=nitrogen, phosphorus=phosphorus, potassium=potassium,
                ph=ph, organic_carbon=2.5, soil_type=soil, crop=crop
            )
            soil_score = soil_result.get("overall_score", 50)
            soil_label = soil_result.get("overall_label", soil_result.get("overall_status", "Unknown"))
            soil_color = "#4CAF50" if soil_score >= 70 else "#FF9800" if soil_score >= 40 else "#f44336"
            sc1, sc2 = st.columns(2)
            with sc1:
                fig_soil = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=soil_score,
                    title={"text": "Soil Health Score", "font": {"color": "#a5d6a7", "size": 16}},
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="#a5d6a7"),
                        bar=dict(color=soil_color),
                        steps=[
                            {"range": [0, 40],  "color": "#1a0e0e"},
                            {"range": [40, 70], "color": "#1a2a1a"},
                            {"range": [70, 100], "color": "#1b5e20"},
                        ],
                    ),
                    number=dict(suffix="/100", font=dict(color=soil_color)),
                ))
                fig_s.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#a5d6a7"),
                                       height=250, margin=dict(t=60, b=10))
                st.plotly_chart(fig_s, width='stretch')
            with sc2:
                st.markdown(
                    f"<div style='background:rgba(45,106,79,0.04);border-radius:14px;padding:20px;margin-top:10px;'>"
                    f"<div style='font-size:1.3rem;font-weight:700;color:{soil_color};'>{soil_label}</div>"
                    f"<div style='color:#b0bec5;font-size:0.88rem;margin-top:8px;'>"
                    f"<b>Nitrogen:</b> {soil_result.get('nutrient_ratings',{}).get('Nitrogen (N)',{}).get('rating','—')}<br>"
                    f"<b>Phosphorus:</b> {soil_result.get('nutrient_ratings',{}).get('Phosphorus (P)',{}).get('rating','—')}<br>"
                    f"<b>Potassium:</b> {soil_result.get('nutrient_ratings',{}).get('Potassium (K)',{}).get('rating','—')}<br>"
                    f"<b>pH:</b> {soil_result.get('ph_status',{}).get('rating','—') if isinstance(soil_result.get('ph_status'),dict) else str(soil_result.get('ph_status','—'))}"
                    f"</div></div>", unsafe_allow_html=True
                )
            suggestions = get_soil_suggestions(soil_result, crop)
            if suggestions:
                st.markdown("**💡 Soil Improvement Suggestions:**")
                for s in suggestions:
                    st.markdown(
                        f"<div style='color:#A5D6A7;margin:4px 0;border-left:3px solid #4CAF50;"
                        f"padding:4px 12px;'>🌱 {s}</div>", unsafe_allow_html=True
                    )
        except Exception:
            st.info("💡 Detailed soil analysis available for: Wheat, Rice, Maize, Cotton, Sugarcane.")

        # ── 🌾 Crop Suitability Ranking ──
        st.markdown("---")
        st.markdown("#### 🌾 Crop Suitability for Your Soil")
        try:
            suitability = get_crop_suitability(
                nitrogen=nitrogen, phosphorus=phosphorus, potassium=potassium,
                ph=ph, soil_type=soil, top_n=5
            )
            if suitability:
                suit_df = pd.DataFrame(suitability)
                if "crop" in suit_df.columns and "score" in suit_df.columns:
                    suit_df = suit_df.sort_values("score", ascending=False)
                    fig_s = px.bar(suit_df, x="score", y="crop", orientation="h",
                                     color="score", color_continuous_scale="YlGn",
                                     title="Top 5 Crops for Your Soil Conditions",
                                     template="plotly_white")
                    fig_s.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                           coloraxis_showscale=False, height=300)
                    st.plotly_chart(fig_s, width='stretch')
        except Exception:
            pass

        # ── 🤖 AI Smart Recommendations ──
        st.markdown("---")
        st.markdown("#### 🤖 AI Smart Recommendations")
        try:
            _wd = {"temperature": temp, "humidity": hum, "precipitation": rain}
            _sd = soil_result if 'soil_result' in locals() else None
            recs = _recommendation_engine.get_recommendations(
                crop=crop, soil_data=_sd,
                weather_data=_wd, yield_prediction=result
            )
            if recs.get("crop_management"):
                st.markdown("##### 🌱 Crop Management")
                for r in recs["crop_management"]:
                    st.markdown(
                        f"<div style='background:rgba(76,175,80,0.08);border-left:3px solid #4CAF50;"
                        f"border-radius:8px;padding:10px 14px;margin:6px 0;color:#344E41;'>{r}</div>",
                        unsafe_allow_html=True
                    )
            if recs.get("soil_improvement"):
                st.markdown("##### 🧪 Soil Improvement")
                for r in recs["soil_improvement"]:
                    st.markdown(
                        f"<div style='background:rgba(255,152,0,0.08);border-left:3px solid #FF9800;"
                        f"border-radius:8px;padding:10px 14px;margin:6px 0;color:#344E41;'>{r}</div>",
                        unsafe_allow_html=True
                    )
            if recs.get("weather_advisory"):
                st.markdown("##### 🌦️ Weather Advisory")
                for r in recs["weather_advisory"]:
                    st.markdown(
                        f"<div style='background:rgba(33,150,243,0.08);border-left:3px solid #2196F3;"
                        f"border-radius:8px;padding:10px 14px;margin:6px 0;color:#344E41;'>{r}</div>",
                        unsafe_allow_html=True
                    )
            if recs.get("yield_optimization"):
                st.markdown("##### 📈 Yield Optimization")
                for r in recs["yield_optimization"]:
                    st.markdown(
                        f"<div style='background:rgba(139,195,74,0.08);border-left:3px solid #8BC34A;"
                        f"border-radius:8px;padding:10px 14px;margin:6px 0;color:#344E41;'>{r}</div>",
                        unsafe_allow_html=True
                    )
            if recs.get("online_insights"):
                with st.expander("🌐 Online Knowledge Insights"):
                    for r in recs["online_insights"]:
                        st.markdown(r)
        except Exception:
            pass

        # ── Yield Category Badge ──
        try:
            y_cat, y_color = get_yield_category(result, crop)
            st.markdown(
                f"<div style='text-align:center;margin-top:16px;'>"
                f"<span style='background:{y_color}22;color:{y_color};border:1px solid {y_color}44;"
                f"padding:6px 18px;border-radius:20px;font-weight:700;font-size:0.9rem;'>"
                f"Yield Category: {y_cat}</span></div>",
                unsafe_allow_html=True
            )
        except Exception:
            pass

# ╔══════════════════════════════════════════════════════════════════════════╗ #

# ║  PAGE 3 – IMAGE SCANNER                                                  ║ #

# ╚══════════════════════════════════════════════════════════════════════════╝ #

elif page == "📷 Image Scanner":

    from image_analyzer import analyze_crop_image

    from utils import CROP_DISEASES

    _disease_detector = _get_disease_detector()

    st.markdown("<div class='section-header'>📷 Crop Image Scanner & Analysis</div>",

                unsafe_allow_html=True)

    st.markdown(

        "<div class='card'>Upload a photo of your crop or field. "

        "The AI will analyze the image to identify the crop type, assess plant health, "

        "and (if models are trained) predict the expected yield.</div>",

        unsafe_allow_html=True

    )

    uploaded = st.file_uploader("Upload a crop / field image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:

        img = Image.open(uploaded)

        # Keep original for EXIF extraction, prepare optimized copy for analysis

        img_for_analysis = _prepare_image_for_analysis(img)

        ic1, ic2 = st.columns([1, 1.2])

        with ic1:

            st.image(img, caption="Uploaded Image", width='stretch')

        _img_hash = hashlib.md5(uploaded.getvalue()).hexdigest()

        if st.session_state.get("_last_img_hash") == _img_hash and st.session_state.get("scan_result"):

            result = st.session_state.scan_result

        else:

            with st.spinner("🔍 Analyzing image…"):

                result = analyze_crop_image(img)

                st.session_state.scan_result = result

                st.session_state["_last_img_hash"] = _img_hash

        with ic2:

            st.markdown("<div class='scan-card'>", unsafe_allow_html=True)

            # ── GPS Location from Photo EXIF (optional) ──

            _gps = result.get("gps_info")

            _pmeta = result.get("photo_metadata", {})

            if _gps:

                st.markdown(

                    f"<div style='background:rgba(33,150,243,0.10);border:1px solid rgba(33,150,243,0.3);"

                    f"border-radius:12px;padding:12px 16px;margin-bottom:12px;'>"

                    f"<div style='font-weight:700;color:#64B5F6;font-size:1rem;'>📍 Photo Location Detected</div>"

                    f"<div style='color:#6B7F6F;font-size:0.88rem;margin-top:4px;'>"

                    f"<b>Location:</b> {_gps.get('location_name', 'Unknown')}<br>"

                    f"<b>GPS:</b> {_gps['latitude']:.4f}°N, {_gps['longitude']:.4f}°E"

                    f"{'  |  <b>Altitude:</b> ' + str(_gps['altitude_m']) + ' m' if _gps.get('altitude_m') else ''}"

                    f"{'  |  <b>Date:</b> ' + str(_gps['gps_timestamp']) if _gps.get('gps_timestamp') else ''}</div>"

                    f"<div style='color:#78909C;font-size:0.72rem;margin-top:3px;'>Source: {_gps.get('source','EXIF')}</div>"

                    f"</div>", unsafe_allow_html=True)

            elif _pmeta.get("has_exif"):

                _cam = ""

                if _pmeta.get("camera_make") or _pmeta.get("camera_model"):

                    _cam = f" | Camera: {_pmeta.get('camera_make','')} {_pmeta.get('camera_model','')}"

                if _pmeta.get("capture_time"):

                    st.markdown(

                        f"<div style='color:#78909C;font-size:0.78rem;margin-bottom:8px;'>"

                        f"📷 Photo taken: {_pmeta['capture_time']}{_cam}</div>",

                        unsafe_allow_html=True)

            else:

                st.markdown(

                    "<div style='color:#546E7A;font-size:0.72rem;margin-bottom:8px;'>"

                    "📍 No GPS data found in photo. For location-based advice, use a phone camera with GPS enabled.</div>",

                    unsafe_allow_html=True)

            if result.get("is_barren", False):

                # Non-crop or barren image — show scene-aware rejection

                scene = result.get("scene_type", "soil")

                scene_icons = {

                    "soil": "🏜️", "water": "🌊", "sky": "☁️",

                    "urban": "🏢", "indoor": "🏠", "selfie": "🤳",

                    "document": "📄", "night": "🌙", "unknown": "❓",

                }

                scene_labels = {

                    "soil": "🚜 Barren / Plowed / Fallow Land Detected",

                    "water": "Water Body Detected",

                    "sky": "Sky Image — No Ground Content",

                    "urban": "Urban / Built Environment",

                    "indoor": "Indoor Photo — Not a Field",

                    "selfie": "Portrait / Selfie Detected",

                    "document": "Document / Screenshot",

                    "night": "Very Dark / Night Image",

                    "unknown": "Unrecognised Image Type",

                }

                sc_icon = scene_icons.get(scene, "🚫")

                sc_label = scene_labels.get(scene, "No Crop Detected")

                sc_conf = result.get("scene_confidence", 0)

                st.markdown(f"### {sc_icon} Scene Analysis")

                st.markdown(

                    f"<div style='text-align:center;padding:30px 0;'>"

                    f"<div style='font-size:3.5rem;'>{sc_icon}</div>"

                    f"<div style='font-size:1.1rem;color:#ffcc80;font-weight:600;margin-top:8px;'>"

                    f"{sc_label}</div>"

                    f"<div style='font-size:0.85rem;color:#bdbdbd;margin-top:4px;'>"

                    f"Scene confidence: {sc_conf:.0%}</div>"

                    f"</div>",

                    unsafe_allow_html=True,

                )

            else:

                st.markdown("### Crop Health Analysis")

                # Health gauge

                fig = go.Figure(go.Indicator(

                    mode="gauge+number",

                    value=result["health_score"],

                    title={"text": "Crop Health Score", "font": {"color": "#00E5FF", "size": 16}},

                    gauge=dict(

                        axis=dict(range=[0, 100], tickcolor="#00E5FF"),

                        bar=dict(color="#00E676"),

                        steps=[

                            {"range": [0, 35],  "color": "#1A0000"},

                            {"range": [35, 65], "color": "#1A1A00"},

                            {"range": [65, 100], "color": "#001A0D"},

                        ],

                    ),

                    number=dict(suffix="%", font=dict(color="#00E5FF")),

                ))

                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#00E5FF"), height=220,

                                  margin=dict(t=60, b=10))

                st.plotly_chart(fig, width='stretch')

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Detailed analysis ──

        st.markdown("#### 📋 Health & Vegetation Report")

        hs = result['health_score']; ht = 'Excellent' if hs > 75 else 'Good' if hs > 55 else 'Fair' if hs > 35 else 'Poor'; st.markdown(f"**Health Score:** {hs:.1f}% — {ht}\n\n"f"**Vegetation Indices:** ExG={result['features']['ExG']:.3f}, "f"VARI={result['features']['VARI']:.3f}, "f"GLI={result['features']['GLI']:.3f}\n\n"f"**Canopy Coverage:** {result['features']['canopy_coverage']:.1%}")

        # —— Methodology (how the scanner works) ——

        if result.get("methodology"):

            with st.expander("🔬 How does the scanner work?"):

                st.markdown(result["methodology"])

        # ── If non-crop / barren, stop here ──

        if result.get("is_barren", False):

            scene = result.get("scene_type", "soil")

            sc_icons2 = {

                "soil": "🏜️", "water": "🌊", "sky": "☁️",

                "urban": "🏢", "indoor": "🏠", "selfie": "🤳",

                "document": "📄", "night": "🌙", "unknown": "❓",

            }

            scene_tips = {

                "soil": "Upload an image showing a field with visible crop growth.",

                "water": "This looks like a water body. Upload a crop field photo instead.",

                "sky": "The image is mostly sky. Point the camera at the field.",

                "urban": "This looks like an urban area. Upload a photo of agricultural land.",

                "indoor": "This appears to be indoors. Take a photo outdoors in a crop field.",

                "selfie": "This looks like a portrait. Upload a photo of a crop/field.",

                "document": "This looks like a document. Upload a real field photo.",

                "night": "The image is too dark. Take a photo in daylight.",

                "unknown": "Upload a clearer photo of an agricultural field.",

            }

            scene_labels2 = {
                "soil": "🚜 Barren / Plowed / Fallow Land",
                "water": "🌊 Water Body", "sky": "☁️ Sky Image",
                "urban": "🏢 Urban Area", "indoor": "🏠 Indoor Photo",
                "selfie": "🤳 Portrait / Selfie", "document": "📄 Document",
                "night": "🌙 Night Image", "unknown": "❓ Unknown Image",
            }
            tip = scene_tips.get(scene, "Upload a clear agricultural field photo.")

            st.markdown(

                f"<div style='background:rgba(255,152,0,0.15);border:1px solid rgba(255,152,0,0.4);"

                f"border-radius:14px;padding:22px 28px;margin-top:16px;text-align:center;'>"

                f"<div style='font-size:2.2rem;'>{sc_icons2.get(scene, '🚫')}</div>"

                f"<div style='font-size:1.15rem;font-weight:700;color:#ffcc80;margin:8px 0;'>"

                f"{scene_labels2.get(scene, 'Scene Detected')}</div>"

                f"<div style='font-size:0.88rem;color:#344E41;'>"

                f"{tip}</div>"

                f"</div>",

                unsafe_allow_html=True,

            )

            # ── SOIL FERTILITY ANALYSIS (only for soil scene) ──

            soil_info = result.get("soil_fertility")

            if scene == "soil" and soil_info:

                st.markdown("---")

                st.markdown("## 🧪 Soil Fertility Analysis")

                # Fertility gauge

                fert_score = soil_info["fertility_score"]

                fert_label = soil_info["fertility_label"]

                fert_color = soil_info["fertility_color"]

                st.markdown(

                    f"<div style='background:rgba(45,106,79,0.04);border-radius:14px;padding:20px;margin:12px 0;text-align:center;'>"

                    f"<div style='font-size:2.8rem;font-weight:800;color:{fert_color};'>{fert_score:.0f}/100</div>"

                    f"<div style='font-size:1.1rem;font-weight:700;color:{fert_color};'>{fert_label}</div>"

                    f"</div>",

                    unsafe_allow_html=True,

                )

                st.markdown(soil_info["summary"])

                # Soil colour interpretation

                st.markdown("### 🎨 Soil Colour Analysis")

                st.markdown(soil_info["color_analysis"])

                # Recommended crops

                st.markdown("### 🌱 Recommended Crops for This Soil")

                rec_crops = soil_info["recommended_crops"]

                crop_emojis = {"Wheat": "🌾", "Rice": "🌾", "Sugarcane": "🍬", "Maize": "🌽",

                    "Cotton": "🏳️", "Tomato": "🍅", "Banana": "🍌", "Mustard": "🌼",

                    "Chickpea": "🫘", "Soybean": "🫘", "Groundnut": "🥜", "Potato": "🥔",

                    "Barley": "🌾", "Jute": "🌿", "Tea": "🍵", "Coffee": "☕"}

                cols = st.columns(min(len(rec_crops), 4))

                for i, crop in enumerate(rec_crops):

                    with cols[i % min(len(rec_crops), 4)]:

                        emoji = crop_emojis.get(crop, "🌱")

                        st.markdown(

                            f"<div style='background:rgba(76,175,80,0.15);border-radius:10px;padding:12px;text-align:center;margin:4px 0;'>"

                            f"<div style='font-size:1.5rem;'>{emoji}</div>"

                            f"<div style='font-weight:600;color:#a5d6a7;'>{crop}</div>"

                            f"</div>", unsafe_allow_html=True)

                # Chemical methods

                st.markdown("### 🧪 Chemical / Fertilizer Methods")

                for title, desc in soil_info["chemical_methods"]:

                    with st.expander(title):

                        st.markdown(desc)

                # Natural methods

                st.markdown("### 🌿 Natural / Organic Methods")

                for title, desc in soil_info["natural_methods"]:

                    with st.expander(title):

                        st.markdown(desc)

            # ── SEASONAL CROP SUGGESTIONS ──

            seasonal = result.get("seasonal_suggestions")

            if seasonal:

                st.markdown("---")

                st.markdown("## 📅 What to Grow Right Now?")

                season = seasonal["current_season"]

                season_colors = {"Kharif": "#4CAF50", "Rabi": "#2196F3", "Zaid": "#FF9800"}

                s_color = season_colors.get(season, "#9C27B0")

                st.markdown(

                    f"<div style='background:rgba(45,106,79,0.04);border-radius:14px;padding:18px;margin:12px 0;'>"

                    f"<div style='font-size:1.6rem;font-weight:800;color:{s_color};text-align:center;'>"

                    f"🌾 {season} Season</div>"

                    f"<div style='color:#e0e0e0;text-align:center;margin-top:8px;'>{seasonal['season_desc']}</div>"

                    f"</div>", unsafe_allow_html=True)

                st.markdown("### 🌱 Crops You Can Plant Now")

                for crop_item in seasonal["crops_to_plant"]:

                    with st.expander(f"{crop_item['emoji']} {crop_item['name']}  —  Sow: {crop_item['sowing']}  |  Harvest: {crop_item['harvest']}"):

                        st.markdown(f"💡 **Tip:** {crop_item['tip']}")

                st.markdown("### 🛠️ Field Preparation Tips")

                for i, tip in enumerate(seasonal["preparation_tips"], 1):

                    st.markdown(f"{i}. {tip}")

                st.markdown(

                    f"<div style='background:rgba(33,150,243,0.10);border:1px solid rgba(33,150,243,0.3);"

                    f"border-radius:12px;padding:14px 18px;margin-top:14px;'>"

                    f"<b style='color:#90CAF9;'>📆 Next Season:</b> {seasonal['next_season']}</div>",

                    unsafe_allow_html=True)

            st.stop()

        # ── Crop Rotation & Soil Nutrition Advice ──

        crop_advice = result.get("crop_advice")

        if crop_advice:

            top_crop_name = result["top_crop"]

            st.markdown("---")

            # Crop Rotation

            st.markdown("#### 🔄 Crop Rotation Recommendations")

            rot_after  = crop_advice.get("rotation_after", [])

            rot_before = crop_advice.get("rotation_before", [])

            avoid_seq  = crop_advice.get("avoid_sequence", [])

            companion  = crop_advice.get("companion", [])

            family     = crop_advice.get("family", "")

            soil_imp   = crop_advice.get("soil_impact", "")

            nut_demand = crop_advice.get("nutrient_demand", "")

            st.markdown(

                f"<div style='background:rgba(0, 230, 118, 0.05);border:1px solid rgba(0, 230, 118, 0.3);"

                f"border-radius:12px;padding:18px 22px;margin-bottom:12px;box-shadow:inset 0 0 10px rgba(0,230,118,0.1);'>"

                f"<b style='color:#00E676;'>Crop Family:</b> {family} &nbsp;|&nbsp; "

                f"<b style='color:#00E676;'>Nutrient Demand:</b> {nut_demand} &nbsp;|&nbsp; "

                f"<b style='color:#00E676;'>Soil Impact:</b> {soil_imp}</div>",

                unsafe_allow_html=True,

            )

            rot_col1, rot_col2 = st.columns(2)

            with rot_col1:

                st.markdown("**✅ Plant AFTER** " + top_crop_name + ":")

                for c in rot_after:

                    st.markdown(f"&nbsp;&nbsp;&nbsp;🌱 {c}")

                st.markdown("**✅ Plant BEFORE** " + top_crop_name + ":")

                for c in rot_before:

                    st.markdown(f"&nbsp;&nbsp;&nbsp;🌱 {c}")

            with rot_col2:

                st.markdown("**❌ Avoid in Sequence:**")

                for c in avoid_seq:

                    st.markdown(f"&nbsp;&nbsp;&nbsp;⚠️ {c}")

                st.markdown("**🤝 Companion Crops:**")

                for c in companion:

                    st.markdown(f"&nbsp;&nbsp;&nbsp;🌿 {c}")

            # Soil Nutrition Analysis

            st.markdown("#### 🧪 Soil Nutrition Analysis for " + top_crop_name)

            soil_nutrition = crop_advice.get("soil_nutrition", [])

            for title, desc in soil_nutrition:

                with st.expander(title):

                    st.markdown(desc)

            # Crop Tips

            tips = crop_advice.get("tips", "")

            if tips:

                st.markdown(

                    f"<div style='background:rgba(255,193,7,0.10);border:1px solid rgba(255,193,7,0.3);"

                    f"border-radius:14px;padding:16px 20px;margin-top:12px;'>"

                    f"<b style='color:#ffe082;'>💡 Expert Tips for {top_crop_name}:</b><br>"

                    f"<span style='color:#e0e0e0;font-size:0.92rem;'>{tips}</span></div>",

                    unsafe_allow_html=True,

                )

        # ── Visual Yield Estimation ──

        yield_est = result.get("yield_estimation")

        if yield_est:

            st.markdown("---")

            st.markdown("#### 📊 Visual Yield Estimation")

            yr = yield_est["yield_rating"]

            y_label = yield_est["yield_label"]

            if yr >= 75:

                y_color = "#4CAF50"

            elif yr >= 55:

                y_color = "#8BC34A"

            elif yr >= 35:

                y_color = "#FF9800"

            else:

                y_color = "#f44336"

            est_range = yield_est["estimated_range_qtl_per_ha"]

            typ_range = yield_est["typical_range_qtl_per_ha"]

            yc1, yc2 = st.columns(2)

            with yc1:

                st.markdown(

                    f"<div style='background:rgba(45,106,79,0.04);border-radius:14px;padding:20px;text-align:center;'>"

                    f"<div style='font-size:2.5rem;font-weight:800;color:{y_color};'>{yr:.0f}/100</div>"

                    f"<div style='font-size:1.0rem;font-weight:700;color:{y_color};'>{y_label}</div>"

                    f"</div>", unsafe_allow_html=True)

            with yc2:

                st.markdown(

                    f"<div style='background:rgba(45,106,79,0.04);border-radius:14px;padding:20px;text-align:center;'>"

                    f"<div style='font-size:1.8rem;font-weight:800;color:#81D4FA;'>"

                    f"{est_range[0]:.0f} - {est_range[1]:.0f} qtl/ha</div>"

                    f"<div style='font-size:0.85rem;color:#b0bec5;'>Estimated Yield Range</div>"

                    f"<div style='font-size:0.75rem;color:#78909c;margin-top:4px;'>"

                    f"Typical for {yield_est['crop']}: {typ_range[0]}-{typ_range[1]} qtl/ha</div>"

                    f"</div>", unsafe_allow_html=True)

            # Growth stage

            st.markdown(

                f"<div style='background:rgba(156,39,176,0.10);border:1px solid rgba(156,39,176,0.3);"

                f"border-radius:12px;padding:14px 18px;margin:12px 0;'>"

                f"<b style='color:#CE93D8;'>🌿 Growth Stage:</b> {yield_est['growth_stage_desc']}</div>",

                unsafe_allow_html=True)

            # Yield factors

            st.markdown("**Yield Factors:**")

            for factor in yield_est["factors"]:

                fc = "#4CAF50" if factor["score"] >= 65 else "#FF9800" if factor["score"] >= 40 else "#f44336"

                st.markdown(

                    f"<div style='display:flex;align-items:center;margin:6px 0;'>"

                    f"<div style='width:150px;font-weight:600;color:#344E41;'>{factor['name']}</div>"

                    f"<div style='flex:1;background:rgba(255,255,255,0.1);border-radius:8px;height:22px;margin:0 10px;overflow:hidden;'>"

                    f"<div style='width:{factor['score']}%;background:{fc};height:100%;border-radius:8px;'></div></div>"

                    f"<div style='width:80px;text-align:right;color:{fc};font-weight:700;'>{factor['score']}/100</div>"

                    f"<div style='width:80px;text-align:right;font-size:0.8rem;color:#b0bec5;'>{factor['status']}</div>"

                    f"</div>", unsafe_allow_html=True)

            st.markdown("")

        # ── Disease & Stress Analysis ──
        disease_data = result.get("disease_stress")
        if disease_data:
            st.markdown("---")
            st.markdown("#### 🦠 Disease & Stress Analysis")

            ds = disease_data["stress_score"]
            ds_status = disease_data["overall_status"]
            ds_color = disease_data["status_color"]
            if ds <= 10:
                ds_icon = "✅"
            elif ds <= 30:
                ds_icon = "⚠️"
            elif ds <= 50:
                ds_icon = "🔶"
            else:
                ds_icon = "🔴"

            st.markdown(
                f"<div style='background:rgba(45,106,79,0.04);border-left:4px solid {ds_color};"
                f"border-radius:12px;padding:18px;margin:10px 0;'>"
                f"<div style='display:flex;align-items:center;gap:12px;'>"
                f"<span style='font-size:2.2rem;'>{ds_icon}</span>"
                f"<div>"
                f"<div style='font-size:1.6rem;font-weight:800;color:{ds_color};'>"
                f"Stress Score: {ds}/100</div>"
                f"<div style='font-size:1rem;color:#b0bec5;'>{ds_status}</div>"
                f"</div></div></div>", unsafe_allow_html=True)

            issues = disease_data.get("issues", [])
            if issues:
                st.markdown("**Detected Issues:**")
                for iss in issues:
                    sev = iss.get("severity", "Low")
                    sev_c = "#f44336" if sev == "Severe" else "#FF9800" if sev == "Moderate" else "#8BC34A"
                    causes_text = "  •  ".join(iss.get("likely_causes", []))
                    st.markdown(
                        f"<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:12px 16px;"
                        f"margin:6px 0;border-left:3px solid {sev_c};'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                        f"<span style='font-weight:700;color:#344E41;'>{iss.get('icon','')} {iss['name']}</span>"
                        f"<span style='background:{sev_c};color:#fff;padding:2px 10px;border-radius:8px;"
                        f"font-size:0.75rem;font-weight:700;text-transform:uppercase;'>{sev}</span></div>"
                        f"<div style='color:#6B7F6F;font-size:0.85rem;margin-top:4px;'>{iss.get('detail','')}</div>"
                        f"<div style='color:#90A4AE;font-size:0.82rem;margin-top:3px;'>"
                        f"<b>Likely causes:</b> {causes_text}</div></div>",
                        unsafe_allow_html=True)

            recs = disease_data.get("recommendations", [])
            if recs:
                st.markdown("**Recommendations:**")
                for r in recs:
                    st.markdown(f"<div style='color:#81C784;margin:3px 0;'>💡 {r}</div>",
                                unsafe_allow_html=True)

        # ── Irrigation & Water Needs ──
        irrig_data = result.get("irrigation")
        if irrig_data:
            st.markdown("---")
            st.markdown("#### 💧 Irrigation & Water Needs")

            wsi = irrig_data["water_stress_index"]
            urgency = irrig_data["irrigation_urgency"]
            urg_color = irrig_data["urgency_color"]
            urg_icon = irrig_data.get("urgency_icon", "💧")

            wc1, wc2 = st.columns(2)
            with wc1:
                st.markdown(
                    f"<div style='background:rgba(45,106,79,0.04);border-radius:14px;padding:20px;text-align:center;'>"
                    f"<div style='font-size:2.4rem;font-weight:800;color:{urg_color};'>{wsi}/100</div>"
                    f"<div style='font-size:0.9rem;color:#b0bec5;'>Water Stress Index</div>"
                    f"</div>", unsafe_allow_html=True)
            with wc2:
                st.markdown(
                    f"<div style='background:rgba(45,106,79,0.04);border-radius:14px;padding:20px;text-align:center;'>"
                    f"<div style='font-size:2rem;'>{urg_icon}</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:{urg_color};'>{urgency}</div>"
                    f"</div>", unsafe_allow_html=True)

            wn_mm = irrig_data.get("water_need_mm")
            wn_cat = irrig_data.get("water_need_category", "")
            if wn_mm:
                st.markdown(
                    f"<div style='background:rgba(33,150,243,0.08);border:1px solid rgba(33,150,243,0.25);"
                    f"border-radius:12px;padding:14px 18px;margin:10px 0;'>"
                    f"<b style='color:#64B5F6;'>💦 {irrig_data.get('crop','')} Water Requirement:</b> "
                    f"{wn_mm[0]}-{wn_mm[1]} mm/season ({wn_cat})</div>",
                    unsafe_allow_html=True)

            indicators = irrig_data.get("indicators", [])
            if indicators:
                st.markdown("**Moisture Indicators:**")
                for ind in indicators:
                    ind_s = ind.get("status", "").lower()
                    ind_c = "#4CAF50" if ind_s in ("adequate","normal","good") else "#FF9800" if ind_s in ("faded","moderate") else "#f44336"
                    ind_icon = "🟢" if ind_s in ("adequate","normal","good") else "🟡" if ind_s in ("faded","moderate") else "🔴"
                    st.markdown(
                        f"<div style='color:{ind_c};margin:4px 0;'>"
                        f"{ind_icon} <b>{ind['name']}:</b> {ind.get('detail', '')}</div>",
                        unsafe_allow_html=True)

            irr_recs = irrig_data.get("recommendations", [])
            if irr_recs:
                st.markdown("**Irrigation Advice:**")
                for ir in irr_recs:
                    st.markdown(f"<div style='color:#81D4FA;margin:3px 0;'>💡 {ir}</div>",
                                unsafe_allow_html=True)

        # ── Pest & Disease Risk ──
        pest_data = result.get("pest_risk")
        if pest_data:
            st.markdown("---")
            st.markdown("#### 🐛 Pest & Disease Risk Assessment")

            p_season = pest_data.get("season_context", "unknown").title()
            p_level = pest_data.get("risk_level", "Low")
            p_color = pest_data.get("risk_color", "#9E9E9E")
            active_pests = pest_data.get("active_pests", [])
            active_diseases = pest_data.get("active_diseases", [])

            st.markdown(
                f"<div style='background:rgba(45,106,79,0.04);border-radius:14px;padding:16px;text-align:center;"
                f"border:1px solid {p_color};margin-bottom:12px;'>"
                f"<div style='font-size:1.3rem;font-weight:700;color:{p_color};'>{p_level}</div>"
                f"<div style='font-size:0.85rem;color:#b0bec5;'>Current Season: {p_season} "
                f"| {pest_data.get('total_threats',0)} Active Threats</div>"
                f"</div>", unsafe_allow_html=True)

            all_threats = active_pests + active_diseases
            if all_threats:
                for threat in all_threats[:6]:
                    is_pest = threat in active_pests
                    t_icon = "🦟" if is_pest else "🍄"
                    t_type = "PEST" if is_pest else "DISEASE"
                    sev = threat.get("severity", "Moderate")
                    sev_c = "#f44336" if sev in ("Very High","High") else "#FF9800" if sev == "Moderate" else "#8BC34A"

                    st.markdown(
                        f"<div style='background:rgba(0,0,0,0.2);border-radius:10px;padding:14px 16px;"
                        f"margin:8px 0;border-left:3px solid {sev_c};'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                        f"<span style='font-weight:700;color:#e0e0e0;font-size:1.05rem;'>"
                        f"{t_icon} {threat.get('name', 'Unknown')}</span>"
                        f"<span style='background:{sev_c};color:#fff;padding:2px 10px;border-radius:8px;"
                        f"font-size:0.72rem;font-weight:700;text-transform:uppercase;'>{t_type} — {sev}</span>"
                        f"</div>"
                        f"<div style='color:#6B7F6F;font-size:0.82rem;margin-top:5px;'>"
                        f"<b>Symptoms:</b> {threat.get('symptom', 'N/A')}</div>"
                        f"<div style='color:#81C784;font-size:0.82rem;margin-top:3px;'>"
                        f"<b>Control:</b> {threat.get('control', 'N/A')}</div>"
                        f"</div>", unsafe_allow_html=True)

                preventive = pest_data.get("preventive_measures", [])
                if preventive:
                    st.markdown("**Preventive Measures:**")
                    for pm in preventive:
                        st.markdown(f"<div style='color:#A5D6A7;margin:3px 0;'>🛡️ {pm}</div>",
                                    unsafe_allow_html=True)
            else:
                st.success("✅ No major pest/disease risks for this crop in the current season.")

        # ── Weed Detection ──
        weed_data = result.get("weed_detection")
        if weed_data:
            st.markdown("---")
            st.markdown("#### 🌿 Weed Detection Analysis")

            wr = weed_data["weed_risk"]
            wr_label = weed_data.get("risk_label", "Unknown")
            wr_color = weed_data.get("risk_color", "#9E9E9E")

            st.markdown(
                f"<div style='background:rgba(45,106,79,0.04);border-radius:14px;padding:18px;text-align:center;'>"
                f"<div style='font-size:2.2rem;font-weight:800;color:{wr_color};'>{wr}/100</div>"
                f"<div style='font-size:1rem;font-weight:600;color:{wr_color};'>{wr_label}</div>"
                f"<div style='font-size:0.8rem;color:#90A4AE;margin-top:4px;'>Weed Presence Score</div>"
                f"</div>", unsafe_allow_html=True)

            w_indicators = weed_data.get("indicators", [])
            if w_indicators:
                st.markdown("**Indicators:**")
                for wi in w_indicators:
                    st.markdown(
                        f"<div style='color:#FFB74D;margin:4px 0;'>⚠️ {wi}</div>",
                        unsafe_allow_html=True)

            w_advice = weed_data.get("advice", [])
            if w_advice:
                st.markdown("**Management Advice:**")
                for wa in w_advice:
                    st.markdown(f"<div style='color:#A5D6A7;margin:3px 0;'>🌱 {wa}</div>",
                                unsafe_allow_html=True)

        # ── 🦠 ML Disease Detection (from disease_detector module) ──
        if result.get("top_crop") and not result.get("is_barren", False):
            detected_crop_name = result["top_crop"]
            supported_crops_dd = list(CROP_DISEASES.keys())
            if detected_crop_name in supported_crops_dd:
                st.markdown("---")
                st.markdown("#### 🔬 ML Disease Detection")
                st.markdown(
                    f"<div style='background:rgba(156,39,176,0.08);border:1px solid rgba(156,39,176,0.3);"
                    f"border-radius:12px;padding:12px 16px;margin-bottom:12px;'>"
                    f"<span style='color:#CE93D8;font-size:0.88rem;'>"
                    f"Running MLP neural network disease classifier for <b>{detected_crop_name}</b>…</span></div>",
                    unsafe_allow_html=True
                )
                try:
                    if not _disease_detector.is_trained:
                        with st.spinner("Training disease detection model…"):
                            _disease_detector.train()
                    dd_result = _disease_detector.predict(img_for_analysis, crop=detected_crop_name)
                    dd_disease = dd_result.get("disease", "Unknown")
                    dd_conf = dd_result.get("confidence", 0)
                    dd_severity = dd_result.get("severity", "N/A")
                    if dd_disease == "Healthy":
                        dd_color = "#4CAF50"
                        dd_icon = "✅"
                    elif dd_severity in ("Critical", "High"):
                        dd_color = "#f44336"
                        dd_icon = "🔴"
                    elif dd_severity == "Moderate":
                        dd_color = "#FF9800"
                        dd_icon = "🟠"
                    else:
                        dd_color = "#8BC34A"
                        dd_icon = "🟢"
                    ddc1, ddc2 = st.columns(2)
                    with ddc1:
                        st.markdown(
                            f"<div style='background:rgba(45,106,79,0.04);border-radius:14px;padding:20px;text-align:center;'>"
                            f"<div style='font-size:2rem;'>{dd_icon}</div>"
                            f"<div style='font-size:1.4rem;font-weight:800;color:{dd_color};margin-top:8px;'>"
                            f"{dd_disease}</div>"
                            f"<div style='font-size:0.85rem;color:#b0bec5;margin-top:4px;'>"
                            f"Confidence: {dd_conf:.1%} | Severity: {dd_severity}</div>"
                            f"</div>", unsafe_allow_html=True
                        )
                    with ddc2:
                        # Show probability chart for all diseases
                        all_probs = dd_result.get("all_probs", {})
                        if all_probs:
                            prob_df = pd.DataFrame(list(all_probs.items()),
                                                   columns=["Disease", "Probability"]).sort_values("Probability")
                            fig_dd = px.bar(prob_df, x="Probability", y="Disease", orientation="h",
                                           color="Probability", color_continuous_scale="RdYlGn_r",
                                           title="Disease Probability Distribution",
                                           template="plotly_white")
                            fig_dd.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                                 coloraxis_showscale=False, height=250,
                                                 margin=dict(t=40, b=10))
                            st.plotly_chart(fig_dd, width='stretch')
                    # Show recommendations
                    dd_recs = dd_result.get("recommendations", [])
                    if dd_recs:
                        st.markdown("**🩺 Treatment Recommendations:**")
                        for rec in dd_recs:
                            st.markdown(
                                f"<div style='background:rgba(156,39,176,0.06);border-left:3px solid #AB47BC;"
                                f"border-radius:8px;padding:10px 14px;margin:5px 0;color:#344E41;'>{rec}</div>",
                                unsafe_allow_html=True
                            )
                except Exception as e:
                    st.warning(f"Disease detection encountered an issue: {e}")

        # ── Auto yield prediction from scan ──

        if st.session_state.meta is not None:

            st.markdown("---")

            st.markdown("#### ⚡ Auto Yield Prediction")

            # If photo has GPS, show auto-detected location

            _photo_gps = result.get("gps_info")

            if _photo_gps:

                _gps_loc = _photo_gps.get("location_name", "")

                st.markdown(

                    f"<div style='background:rgba(33,150,243,0.08);border:1px solid rgba(33,150,243,0.25);"

                    f"border-radius:10px;padding:10px 14px;margin-bottom:8px;'>"

                    f"📍 <b>Photo location auto-detected:</b> {_gps_loc} "

                    f"({_photo_gps['latitude']:.3f}°, {_photo_gps['longitude']:.3f}°)</div>",

                    unsafe_allow_html=True)

            else:

                st.markdown("Select your location to get area-specific weather and yield prediction.")

            detected_crop = result["top_crop"]

            crop_info = get_crop_info(detected_crop)

            # Pick smart defaults from crop profile

            default_season = crop_info["seasons"][0] if crop_info else "Kharif"

            default_soils  = crop_info["soils"][0]   if crop_info else "Loamy"

            default_n  = int(np.mean(crop_info["N"]))  if crop_info else 80

            default_p  = int(np.mean(crop_info["P"]))  if crop_info else 45

            default_k  = int(np.mean(crop_info["K"]))  if crop_info else 40

            default_ph = float(np.mean(crop_info["pH"])) if crop_info else 6.5

            sc1, sc2 = st.columns(2)

            with sc1:

                scan_state = st.selectbox("State", get_state_names(), key="scan_state")

            with sc2:

                scan_area = st.text_input("Area / City / Village",

                                          placeholder="e.g. Gorakhpur, Bareilly…",

                                          key="scan_area")

            if st.button("🌾 Fetch Weather & Predict Yield", key="scan_predict_btn",

                         width='stretch'):

                # Fetch area-level weather

                if scan_area.strip():

                    with st.spinner(f"🌍 Fetching weather for {scan_area}, {scan_state}…"):

                        w = _cached_weather_area(scan_area.strip(), scan_state)

                else:

                    with st.spinner(f"🌍 Fetching weather for {scan_state}…"):

                        w = _cached_weather(scan_state)

                        w["location"] = scan_state

                loc_label = w.get("location", scan_state)

                # Show weather card

                st.markdown(

                    f"<div class='weather-card' style='margin:12px 0;'>"

                    f"<b>📍 {loc_label}</b> — {weather_code_text(w.get('weather_code', 0))}<br>"

                    f"<span style='font-size:0.82rem;color:#90caf9;'>"

                    f"🌡️ {w['temperature']}°C &nbsp;|&nbsp; 💧 {w['humidity']}% &nbsp;|&nbsp; "

                    f"🌧️ ~{w['rainfall']} mm/year &nbsp;|&nbsp; 💨 {w.get('wind_speed',0)} km/h"

                    f"</span></div>", unsafe_allow_html=True

                )

                default_area = 5.0

                inputs = {

                    "Crop": detected_crop, "Season": default_season,

                    "State": scan_state, "Soil_Type": default_soils,

                    "Area_ha": default_area, "Nitrogen": default_n,

                    "Phosphorus": default_p, "Potassium": default_k,

                    "pH": default_ph, "Temperature": w["temperature"],

                    "Humidity": w["humidity"], "Rainfall": w["rainfall"],

                    "Fertilizer": 120, "Pesticide": 5.0, "Irrigation": 1,

                }

                pred = predict_yield(inputs)

                total = pred * default_area

                st.markdown(

                    f"<div class='result-box'>"

                    f"<div style='color:#a5d6a7;font-size:0.9rem;'>"

                    f"🌾 {detected_crop} — Predicted Yield at {loc_label}</div>"

                    f"<div class='result-value'>{pred:.3f}</div>"

                    f"<div class='result-label'>tonnes/ha &nbsp;|&nbsp; Total for {default_area:.0f} ha: "

                    f"<b>{total:,.1f} t</b></div>"

                    f"</div>", unsafe_allow_html=True

                )

                # Show assumptions used

                with st.expander("📋 Prediction Parameters Used"):

                    pc1, pc2, pc3 = st.columns(3)

                    with pc1:

                        st.markdown(f"**Crop:** {detected_crop}")

                        st.markdown(f"**Season:** {default_season}")

                        st.markdown(f"**State:** {scan_state}")

                        st.markdown(f"**Soil:** {default_soils}")

                    with pc2:

                        st.markdown(f"**Area:** {default_area} ha")

                        st.markdown(f"**N / P / K:** {default_n} / {default_p} / {default_k}")

                        st.markdown(f"**pH:** {default_ph:.1f}")

                    with pc3:

                        st.markdown(f"**Temperature:** {w['temperature']:.1f} °C")

                        st.markdown(f"**Humidity:** {w['humidity']:.0f}%")

                        st.markdown(f"**Rainfall:** {w['rainfall']:.0f} mm")

                    st.info("💡 For custom parameters, use the **Predict Yield** page.")

# ╔══════════════════════════════════════════════════════════════════════════╗ #

# ║  PAGE 4 – WEATHER INTELLIGENCE                                           ║ #

# ╚══════════════════════════════════════════════════════════════════════════╝ #

elif page == "🌦️ Weather Intel":

    from satellite_ndvi import generate_ndvi_data, get_ndvi_analysis, get_ndvi_change_analysis

    st.markdown("<div class='section-header'>🌦️ Live Weather Intelligence</div>",

                unsafe_allow_html=True)

    st.markdown(

        "<div class='card'>Fetches <b>real-time weather</b> from the <b>Open-Meteo API</b> "

        "(100 % free, no API key required). Select a state, enter your area/city, "

        "and get hyper-local weather conditions.</div>",

        unsafe_allow_html=True

    )

    wc_s, wc_a, wc_b = st.columns([2, 3, 1])

    with wc_s:

        wstate = st.selectbox("Select State", get_state_names(), key="weather_page_state")

    with wc_a:

        warea = st.text_input("Enter Area / City / Village",

                              placeholder="e.g. Lucknow, Varanasi, Siliguri…",

                              key="weather_page_area")

    with wc_b:

        st.markdown("<br>", unsafe_allow_html=True)

        fetch_w_btn = st.button("🌍 Fetch Weather")

    if fetch_w_btn:

        if warea.strip():

            with st.spinner(f"Fetching weather for {warea}, {wstate}…"):

                w = _cached_weather_area(warea.strip(), wstate)

                st.session_state.weather = w

        else:

            with st.spinner(f"Fetching weather for {wstate}…"):

                w = _cached_weather(wstate)

                w["location"] = wstate

                st.session_state.weather = w

    w = st.session_state.weather

    if w:

        loc = w.get("location", "")

        st.markdown(f"#### 📍 {loc} — {weather_code_text(w.get('weather_code', 0))}")

        st.markdown(f"<span style='color:#90caf9;font-size:0.85rem;'>Source: {w.get('source','')}</span>",

                    unsafe_allow_html=True)

        wc1, wc2, wc3, wc4 = st.columns(4)

        with wc1:

            st.markdown(

                f"<div class='weather-card'>"

                f"<div class='weather-value'>{w['temperature']}°C</div>"

                f"<div class='weather-label'>Temperature</div></div>",

                unsafe_allow_html=True)

        with wc2:

            st.markdown(

                f"<div class='weather-card'>"

                f"<div class='weather-value'>{w['humidity']}%</div>"

                f"<div class='weather-label'>Humidity</div></div>",

                unsafe_allow_html=True)

        with wc3:

            st.markdown(

                f"<div class='weather-card'>"

                f"<div class='weather-value'>{w['rainfall']} mm</div>"

                f"<div class='weather-label'>Est. Annual Rain</div></div>",

                unsafe_allow_html=True)

        with wc4:

            st.markdown(

                f"<div class='weather-card'>"

                f"<div class='weather-value'>{w.get('wind_speed', 0)} km/h</div>"

                f"<div class='weather-label'>Wind Speed</div></div>",

                unsafe_allow_html=True)

        st.markdown("---")

        # ── Crop recommendation based on weather ──

        st.markdown("#### 🌿 Recommended Crops for Current Conditions")

        recs = []

        for crop_name, info in CROPS.items():

            t_lo, t_hi = info["optimal_temp"]

            r_lo, r_hi = info["optimal_rain"]

            h_lo, h_hi = info["humidity"]

            t_fit = 1 - min(abs(w["temperature"] - (t_lo + t_hi) / 2) / 15, 1)

            r_fit = 1 - min(abs(w["rainfall"]    - (r_lo + r_hi) / 2) / 1500, 1)

            h_fit = 1 - min(abs(w["humidity"]     - (h_lo + h_hi) / 2) / 40, 1)

            score = round((t_fit * 0.4 + r_fit * 0.35 + h_fit * 0.25) * 100, 1)

            recs.append({"Crop": crop_name, "Suitability %": score})

        rec_df = pd.DataFrame(recs).sort_values("Suitability %", ascending=False)

        fig = px.bar(rec_df.head(10), x="Suitability %", y="Crop", orientation="h",

                     color="Suitability %", color_continuous_scale="YlGn",

                     title="Top 10 Crops for Current Weather", template="plotly_white")

        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",

                          coloraxis_showscale=False, height=400)

        st.plotly_chart(fig, width='stretch')

        # ── Multi-state comparison ──

        st.markdown("#### 🗺️ Multi-State Weather Comparison")

        if st.button("Fetch All States", help="Fetches weather for all states — may take 15-30 seconds"):
            all_w = []
            prog = st.progress(0)
            states = get_state_names()
            for i, s in enumerate(states):
                wd = _cached_weather(s)
                all_w.append({"State": s, **{k: v for k, v in wd.items() if k not in ("source", "weather_code", "area", "location")}})
                prog.progress((i + 1) / len(states))
            all_df = pd.DataFrame(all_w)

            # 1. Temperature comparison (sorted)
            fig_t = px.bar(all_df.sort_values("temperature"), x="State", y="temperature",
                           color="temperature", color_continuous_scale="RdYlBu_r",
                           title="🌡️ Temperature Across States (°C)", template="plotly_white")
            fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                coloraxis_showscale=False)
            st.plotly_chart(fig_t, width='stretch')

            # 2. Humidity comparison
            fig_h = px.bar(all_df.sort_values("humidity"), x="State", y="humidity",
                           color="humidity", color_continuous_scale="Blues",
                           title="💧 Humidity Across States (%)", template="plotly_white")
            fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                coloraxis_showscale=False)
            st.plotly_chart(fig_h, width='stretch')

            mc1, mc2 = st.columns(2)
            with mc1:
                # 3. Rainfall comparison
                fig_r = px.bar(all_df.sort_values("rainfall"), x="State", y="rainfall",
                               color="rainfall", color_continuous_scale="Teal",
                               title="🌧️ Annual Rainfall (mm)", template="plotly_white")
                fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    coloraxis_showscale=False, height=380)
                st.plotly_chart(fig_r, width='stretch')
            with mc2:
                # 4. Wind speed comparison
                fig_w = px.bar(all_df.sort_values("wind_speed"), x="State", y="wind_speed",
                               color="wind_speed", color_continuous_scale="Purples",
                               title="💨 Wind Speed (km/h)", template="plotly_white")
                fig_w.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    coloraxis_showscale=False, height=380)
                st.plotly_chart(fig_w, width='stretch')

            # 5. Summary table with all parameters
            st.markdown("##### 📋 Complete Weather Summary")
            display_df = all_df.sort_values("temperature", ascending=False).copy()
            display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
            st.dataframe(display_df, width='stretch', hide_index=True)

            # 6. Hot vs Cold scatter comparison
            st.markdown("##### 🎯 Hot vs Cold States")
            hot5 = all_df.nlargest(5, "temperature")
            cold5 = all_df.nsmallest(5, "temperature")
            compare_df = pd.concat([hot5.assign(group="Hottest 5"), cold5.assign(group="Coldest 5")])
            fig_comp = px.scatter(compare_df, x="temperature", y="humidity", size="rainfall",
                                  color="group", hover_name="State",
                                  color_discrete_map={"Hottest 5": "#ef5350", "Coldest 5": "#42a5f5"},
                                  title="Temperature vs Humidity (bubble size = rainfall)",
                                  template="plotly_white",
                                  labels={"temperature": "Temperature (°C)", "humidity": "Humidity (%)"})
            fig_comp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_comp, width='stretch')

    else:

        st.info("Click **Fetch Live Weather** to load data.")

# ╔══════════════════════════════════════════════════════════════════════════╗ #

    # ── 🛰️ Satellite NDVI Vegetation Analysis ──
    st.markdown("---")
    st.markdown("#### 🛰️ Satellite NDVI — Vegetation Health Analysis")
    st.markdown(
        "<div class='card' style='border-color:rgba(76,175,80,0.3);'>"
        "<span style='color:#40916C;font-size:0.88rem;'>"
        "Simulated NDVI (Normalized Difference Vegetation Index) analysis — "
        "in production this connects to Sentinel-2/Landsat satellite imagery.</span></div>",
        unsafe_allow_html=True
    )
    ndvi_c1, ndvi_c2, ndvi_c3, ndvi_c4 = st.columns([2, 2, 2, 1])
    with ndvi_c1:
        ndvi_crop = st.selectbox("Crop for NDVI", get_crop_names(), key="ndvi_crop")
    with ndvi_c2:
        ndvi_state = st.selectbox("State", get_state_names(), key="ndvi_state")
    with ndvi_c3:
        ndvi_area = st.text_input("Area / Village / City (optional)", key="ndvi_area",
                                   placeholder="e.g. Ludhiana, Amritsar...")
    with ndvi_c4:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_ndvi = st.button("🛰️ Generate NDVI", key="gen_ndvi_btn")

    if gen_ndvi:
        # Resolve location from area or state
        _ndvi_lat, _ndvi_lon = None, None
        if ndvi_area and ndvi_area.strip():
            from weather_api import geocode_area
            geo = geocode_area(ndvi_area.strip())
            if geo:
                _ndvi_lat, _ndvi_lon = geo["lat"], geo["lon"]
                st.caption(f"📍 Resolved: {geo['name']}, {geo.get('admin1', ndvi_state)}")
        if _ndvi_lat is None:
            si = get_state_info(ndvi_state)
            _ndvi_lat = si["lat"] if si else 25.0
            _ndvi_lon = si["lon"] if si else 80.0
        lat, lon = _ndvi_lat, _ndvi_lon
        with st.spinner("Generating NDVI satellite data…"):
            ndvi_data = generate_ndvi_data(lat, lon, crop=ndvi_crop, grid_size=20)
            ndvi_analysis = get_ndvi_analysis(ndvi_data, crop=ndvi_crop)
            ndvi_change = get_ndvi_change_analysis(ndvi_data)

        # NDVI Health banner
        h_color = ndvi_data["health_color"]
        st.markdown(
            f"<div style='background:rgba(45,106,79,0.04);border-radius:14px;padding:20px;margin:12px 0;"
            f"border-left:4px solid {h_color};'>"
            f"<div style='display:flex;align-items:center;gap:16px;'>"
            f"<div style='font-size:2.5rem;font-weight:800;color:{h_color};'>{ndvi_data['mean']:.3f}</div>"
            f"<div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:{h_color};'>{ndvi_data['health']}</div>"
            f"<div style='font-size:0.85rem;color:#b0bec5;'>Mean NDVI — {ndvi_crop} in {(ndvi_area + ', ' if ndvi_area and ndvi_area.strip() else '') + ndvi_state}</div>"
            f"</div></div></div>",
            unsafe_allow_html=True
        )

        nc1, nc2, nc3, nc4 = st.columns(4)
        with nc1:
            st.markdown(metric_card(f"{ndvi_data['veg_fraction']*100:.1f}%", "Vegetation Cover"),
                        unsafe_allow_html=True)
        with nc2:
            st.markdown(metric_card(f"{ndvi_data['bare_fraction']*100:.1f}%", "Bare Soil"),
                        unsafe_allow_html=True)
        with nc3:
            st.markdown(metric_card(f"{ndvi_data['std']:.3f}", "Spatial Variability"),
                        unsafe_allow_html=True)
        with nc4:
            ch_icon = ndvi_change["icon"]
            st.markdown(metric_card(f"{ch_icon} {ndvi_change['change_pct']:+.1f}%", ndvi_change["trend"]),
                        unsafe_allow_html=True)

        ndvi_r1, ndvi_r2 = st.columns(2)
        with ndvi_r1:
            # NDVI Heatmap
            fig_ndvi = px.imshow(ndvi_data["ndvi_grid"], color_continuous_scale="RdYlGn",
                                 zmin=-0.1, zmax=0.95,
                                 title="NDVI Spatial Heatmap", template="plotly_white")
            fig_ndvi.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig_ndvi, width='stretch')
        with ndvi_r2:
            # Monthly NDVI trend
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            fig_trend = px.line(x=months, y=ndvi_data["temporal"],
                                title="12-Month NDVI Trend", template="plotly_white",
                                labels={"x": "Month", "y": "NDVI"})
            fig_trend.update_traces(line_color="#4CAF50", line_width=3)
            fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    height=350)
            st.plotly_chart(fig_trend, width='stretch')

        # NDVI Analysis insights
        if ndvi_analysis.get("insights"):
            st.markdown("##### 📋 NDVI Insights")
            for ins in ndvi_analysis["insights"]:
                st.markdown(ins)
        if ndvi_analysis.get("recommendations"):
            st.markdown("##### 💡 NDVI Recommendations")
            for rec in ndvi_analysis["recommendations"]:
                st.markdown(
                    f"<div style='background:rgba(76,175,80,0.08);border-left:3px solid #4CAF50;"
                    f"border-radius:8px;padding:10px 14px;margin:6px 0;color:#344E41;'>{rec}</div>",
                    unsafe_allow_html=True
                )

        # Change analysis
        st.markdown(
            f"<div style='background:rgba(0,0,0,0.25);border-radius:12px;padding:14px 18px;margin:10px 0;'>"
            f"<b style='color:#90caf9;'>📊 Change Analysis:</b> "
            f"{ndvi_change['icon']} NDVI changed by <b>{ndvi_change['change']:+.4f}</b> "
            f"({ndvi_change['change_pct']:+.1f}%) — Trend: <b>{ndvi_change['trend']}</b></div>",
            unsafe_allow_html=True
        )

        # ── Current Season Detection & Farming Suggestions ──
        st.markdown("---")
        import datetime as _dt
        _month = _dt.date.today().month
        if _month in (6, 7, 8, 9, 10):
            _season, _s_color, _s_desc = "Kharif", "#4CAF50", "Monsoon season (June–October) — ideal for rain-fed crops like Rice, Maize, Cotton, Soybean."
            _crops_now = [("🌾 Rice", "Transplant June–July, harvest Oct–Nov"),
                          ("🌽 Maize", "Plant after first rains, harvest Sep–Oct"),
                          ("🏳️ Cotton", "Sow April–June, harvest Oct–Jan"),
                          ("🪸 Soybean", "Sow June–July, harvest Sep–Oct"),
                          ("🥜 Groundnut", "Sow June–July, harvest Oct–Nov")]
            _tips = ["Ensure drainage to prevent waterlogging.", "Apply FYM 10-15 tonnes/ha before sowing.",
                     "Get soil tested before planting.", "Consider green manuring with dhaincha."]
            _next = "Rabi (Oct–March): Wheat, Mustard, Chickpea, Potato, Barley"
        elif _month in (11, 12, 1, 2, 3):
            _season, _s_color, _s_desc = "Rabi", "#2196F3", "Winter season (October–March) — cool temperatures ideal for Wheat, Mustard, Chickpea, Potato."
            _crops_now = [("🌾 Wheat", "Sow Oct–Nov, harvest March–April"),
                          ("🌼 Mustard", "Sow Oct–Nov, harvest Feb–March"),
                          ("🪸 Chickpea", "Sow Oct–Nov, harvest Feb–March"),
                          ("🥔 Potato", "Plant Oct–Nov, harvest Jan–Feb"),
                          ("🍅 Tomato", "Transplant Sep–Oct, harvest Jan–March")]
            _tips = ["Apply compost 5-10 tonnes/ha during field prep.", "Laser-level for uniform irrigation.",
                     "Apply DAP 100 kg/ha + Potash 60 kg/ha at sowing.", "Plan wheat-mustard intercropping."]
            _next = "Zaid (March–June): Watermelon, Muskmelon, Cucumber, Moong, Maize"
        else:
            _season, _s_color, _s_desc = "Zaid", "#FF9800", "Summer season (March–June) — hot & dry, irrigated short-duration crops perform best."
            _crops_now = [("🌽 Maize", "Short-duration hybrids, March–April"),
                          ("🥜 Groundnut", "Summer crop, frequent light irrigation"),
                          ("🍅 Tomato", "Use shade nets in extreme heat"),
                          ("🍌 Banana", "March–April planting, drip irrigation"),
                          ("🍬 Sugarcane", "Spring planting gives highest yield")]
            _tips = ["Ensure reliable irrigation — summer crops need frequent watering.",
                     "Apply mulch 5-7cm to reduce evaporation.", "Consider drip irrigation for water savings.",
                     "Plan windbreaks for hot dry winds."]
            _next = "Kharif (June–October): Rice, Maize, Cotton, Soybean, Groundnut"

        st.markdown(
            f"<div style='background:rgba(45,106,79,0.04);border-radius:14px;padding:18px;margin:12px 0;'>"
            f"<div style='font-size:1.5rem;font-weight:800;color:{_s_color};text-align:center;'>"
            f"🌾 Current Season: {_season}</div>"
            f"<div style='color:#e0e0e0;text-align:center;margin-top:6px;font-size:0.92rem;'>{_s_desc}</div>"
            f"</div>", unsafe_allow_html=True)

        st.markdown("##### 🌱 What to Grow Right Now")
        _scols = st.columns(min(len(_crops_now), 5))
        for _ci, (_cn, _cd) in enumerate(_crops_now):
            with _scols[_ci % len(_scols)]:
                st.markdown(
                    f"<div style='background:rgba(76,175,80,0.10);border-radius:10px;padding:12px;text-align:center;margin:4px 0;'>"
                    f"<div style='font-size:1.2rem;font-weight:700;color:#a5d6a7;'>{_cn}</div>"
                    f"<div style='font-size:0.78rem;color:#b0bec5;margin-top:4px;'>{_cd}</div>"
                    f"</div>", unsafe_allow_html=True)

        with st.expander("🛠️ Field Preparation Tips"):
            for _ti, _t in enumerate(_tips, 1):
                st.markdown(f"{_ti}. {_t}")

        st.markdown(
            f"<div style='background:rgba(33,150,243,0.10);border:1px solid rgba(33,150,243,0.3);"
            f"border-radius:12px;padding:12px 16px;margin-top:10px;'>"
            f"<b style='color:#90CAF9;'>📆 Next Season:</b> {_next}</div>",
            unsafe_allow_html=True)

# ║  PAGE 5 – MODEL HUB                                                      ║ #

# ╚══════════════════════════════════════════════════════════════════════════╝ #

elif page == "📊 Model Hub":

    st.markdown("<div class='section-header'>📊 AI Model Hub — Performance & Comparison</div>",

                unsafe_allow_html=True)

    if need_model():

        st.stop()

    meta    = st.session_state.meta

    results = meta["results"]

    best    = meta["best_name"]

    # ── Best model banner ──

    bm = results[best]

    st.markdown(

        f"<div class='card' style='border-color:#8bc34a;'>"

        f"🏆 <span style='font-size:1.3rem;font-weight:700;color:#a5d6a7;'>"

        f"Best Model: {best}</span><br>"

        f"<span style='font-size:0.95rem;'>R² = <b>{bm['R2']}</b> &nbsp;|&nbsp; "

        f"CV R² = <b>{bm['CV_R2']}</b> &nbsp;|&nbsp; "

        f"MAE = <b>{bm['MAE']}</b> &nbsp;|&nbsp; RMSE = <b>{bm['RMSE']}</b></span><br>"

        f"<span style='font-size:0.78rem;color:#40916C;'>"

        f"Trained in {meta.get('training_rounds', 1)} rounds &mdash; "

        f"best from round {meta.get('best_round', 1)}</span>"

        f"</div>", unsafe_allow_html=True

    )

    # ── Model cards grid ──

    st.markdown("#### 🤖 All Models")

    for i in range(0, len(results), 4):

        cols = st.columns(4)

        for j, (name, vals) in enumerate(list(results.items())[i:i+4]):

            info = get_model_info(name)

            with cols[j]:

                badge = " 🏆" if name == best else ""

                acc_color = "#8bc34a" if vals["R2"] > 0.85 else ("#ffee58" if vals["R2"] > 0.6 else "#ef5350")

                st.markdown(

                    f"<div class='model-card'>"

                    f"<b>{info.get('icon','')} {name}{badge}</b><br>"

                    f"<span style='font-size:0.72rem;color:#40916C;'>{info.get('family','')}</span><br>"

                    f"<span style='font-size:0.78rem;'>{info.get('desc','')[:80]}…</span><br>"

                    f"<br>R² = <span style='color:{acc_color};font-weight:700;'>{vals['R2']}</span>"

                    f" &nbsp;|&nbsp; MAE = {vals['MAE']}"

                    f"</div>", unsafe_allow_html=True

                )

    st.markdown("---")

    # ── Side-by-side comparison charts ──

    df_res = pd.DataFrame([{"Model": k, **{m: v for m, v in vals.items() if m != "feat_import"}}

                           for k, vals in results.items()])

    tab1, tab2, tab3 = st.tabs(["📊 R² Comparison", "📉 Error Metrics", "🌿 Feature Importance"])

    with tab1:

        fig = px.bar(df_res.sort_values("R2"), x="R2", y="Model", orientation="h",

                     color="R2", color_continuous_scale="Greens",

                     title="R² Score by Model (higher = better)", template="plotly_white")

        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",

                          coloraxis_showscale=False)

        st.plotly_chart(fig, width='stretch')

    with tab2:

        fig2 = make_subplots(rows=1, cols=2, subplot_titles=("MAE (lower = better)",

                                                              "RMSE (lower = better)"))

        colors = px.colors.qualitative.Pastel

        for i, row in df_res.iterrows():

            fig2.add_trace(go.Bar(name=row["Model"], x=[row["Model"]], y=[row["MAE"]],

                                  marker_color=colors[i % len(colors)], showlegend=(i < len(df_res))),

                           1, 1)

            fig2.add_trace(go.Bar(x=[row["Model"]], y=[row["RMSE"]],

                                  marker_color=colors[i % len(colors)], showlegend=False),

                           1, 2)

        fig2.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",

                           plot_bgcolor="rgba(0,0,0,0)", height=400)

        st.plotly_chart(fig2, width='stretch')

    with tab3:

        model_sel = st.selectbox("Select model", list(results.keys()),

                                 index=list(results.keys()).index(best))

        fi = results[model_sel].get("feat_import")

        if fi:

            fi_df = pd.DataFrame({"Feature": list(fi.keys()),

                                  "Importance": list(fi.values())}).sort_values("Importance")

            fig3 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",

                          color="Importance", color_continuous_scale="Greens",

                          title=f"Feature Importances — {model_sel}", template="plotly_white")

            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",

                               coloraxis_showscale=False)

            st.plotly_chart(fig3, width='stretch')

        else:

            st.info(f"{model_sel} does not expose feature importances (linear/SVM/KNN models).")

    # ── Detailed metrics table ──

    st.markdown("#### 📋 Full Metrics Table")

    styled = df_res.set_index("Model")

    st.dataframe(

        styled.style.background_gradient(cmap="Greens", subset=["R2", "CV_R2"])

                    .background_gradient(cmap="Reds_r", subset=["MAE", "RMSE"])

                    .format("{:.4f}"),

        width='stretch', height=340,

    )

# ╔══════════════════════════════════════════════════════════════════════════╗ #

# ║  PAGE 6 – DATA EXPLORER                                                  ║ #

# ╚══════════════════════════════════════════════════════════════════════════╝ #

elif page == "🗄️ Data Explorer":

    st.markdown("<div class='section-header'>🗄️ Dataset Explorer</div>", unsafe_allow_html=True)

    if need_data():

        st.stop()

    df = st.session_state.df.copy()

    # ── Filters ──

    with st.expander("🔍 Filter Data", expanded=True):

        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:

            sel_crop = st.multiselect("Crop", sorted(df["Crop"].unique()),

                                      default=sorted(df["Crop"].unique()))

        with fc2:

            sel_season = st.multiselect("Season", df["Season"].unique().tolist(),

                                        default=df["Season"].unique().tolist())

        with fc3:

            sel_state = st.multiselect("State", sorted(df["State"].unique()),

                                       default=sorted(df["State"].unique()))

        with fc4:

            min_y = float(df["Yield_ton_per_ha"].min())

            max_y = float(df["Yield_ton_per_ha"].max())

            yield_range = st.slider("Yield range", min_y, max_y, (min_y, max_y), 0.1)

    filtered = df[

        df["Crop"].isin(sel_crop) &

        df["Season"].isin(sel_season) &

        df["State"].isin(sel_state) &

        df["Yield_ton_per_ha"].between(*yield_range)

    ]

    st.markdown(f"**{len(filtered):,}** records match")

    # ── Summary ──

    st.markdown("#### 📐 Summary Statistics")

    st.dataframe(filtered.describe().round(3), width='stretch')

    # ── Table ──

    st.markdown("#### 📄 Raw Data (first 500)")

    st.dataframe(filtered.head(500), width='stretch', height=340)

    # ── Download ──

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button("⬇️ Download Filtered CSV", csv, "crop_data_filtered.csv", "text/csv")

    st.markdown("---")

    # ── Charts ──

    ch1, ch2 = st.columns(2)

    with ch1:

        st.markdown("#### 📈 Yield Distribution by State")

        top_st = (filtered.groupby("State")["Yield_ton_per_ha"].mean()

                  .sort_values(ascending=False).head(10).index)

        fig = px.violin(filtered[filtered["State"].isin(top_st)],

                        x="State", y="Yield_ton_per_ha", color="State",

                        box=True, template="plotly_white",

                        color_discrete_sequence=px.colors.qualitative.Pastel)

        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",

                          showlegend=False)

        st.plotly_chart(fig, width='stretch')

    with ch2:

        st.markdown("#### 🌧️ Rainfall vs Yield")

        fig2 = px.scatter(filtered.sample(min(1200, len(filtered))),

                          x="Rainfall", y="Yield_ton_per_ha", color="Crop",

                          opacity=0.5, template="plotly_white")

        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        st.plotly_chart(fig2, width='stretch')
