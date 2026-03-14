"""
=============================================================================
 SMART CROP YIELD PREDICTION USING ARTIFICIAL INTELLIGENCE
 Module: utils.py — Shared Constants, Theme Engine, CSS & Helper Functions
=============================================================================

 PURPOSE:
   Central utility module providing application-wide constants, color themes
   for both Light and Dark modes, CSS stylesheet generation with 3D glassmorphism
   effects, animated wallpapers, and reusable HTML rendering functions.

 EXPORTS:
   - APP constants (title, version, crops, states, seasons, soil types)
   - CROP_PARAMETERS: optimal agronomic ranges per crop (India-specific)
   - Theme engine: get_custom_css(mode), render_metric_card(), render_glass_card()
   - Helper functions: interpret_ndvi(), get_yield_category(), format_yield()

 AUTHOR : AgriTech AI Solutions
 VERSION: 3.0.0  (Production — March 2026)
=============================================================================
"""

# ── Standard library imports ──────────────────────────────────────────────
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
#  1. APPLICATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

APP_TITLE      = "🌾 Smart Crop Yield Prediction Using AI"
APP_SUBTITLE   = "Precision Agriculture Intelligence Platform"
APP_VERSION    = "3.0.0"
APP_AUTHOR     = "AgriTech AI Solutions"

SUPPORTED_CROPS = ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane"]

INDIAN_STATES = [
    "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Odisha", "Punjab", "Rajasthan",
    "Tamil Nadu", "Telangana", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

SEASONS            = ["Kharif", "Rabi", "Zaid", "Whole Year"]
SOIL_TYPES         = ["Alluvial", "Black", "Red", "Laterite", "Sandy", "Clay", "Loamy", "Silty"]
IRRIGATION_METHODS = ["Rainfed", "Canal", "Tubewell", "Drip", "Sprinkler", "Flood"]

# Common crop diseases for the disease-detection module
CROP_DISEASES = {
    "Wheat":     ["Leaf Rust", "Powdery Mildew", "Septoria Leaf Blotch", "Yellow Rust", "Karnal Bunt"],
    "Rice":      ["Blast", "Brown Spot", "Sheath Blight", "Bacterial Leaf Blight", "False Smut"],
    "Maize":     ["Northern Leaf Blight", "Gray Leaf Spot", "Common Rust", "Stalk Rot", "Maize Dwarf Mosaic"],
    "Cotton":    ["Bacterial Blight", "Alternaria Leaf Spot", "Grey Mildew", "Root Rot", "Cotton Leaf Curl"],
    "Sugarcane": ["Red Rot", "Smut", "Wilt", "Leaf Scald", "Grassy Shoot Disease"],
}


# ═══════════════════════════════════════════════════════════════════════════
#  2. CROP PARAMETER RANGES  (India-specific agronomic data)
# ═══════════════════════════════════════════════════════════════════════════

CROP_PARAMETERS = {
    "Wheat": {
        "optimal_temp": (15, 25), "optimal_humidity": (40, 70),
        "optimal_rainfall": (400, 650), "optimal_ph": (6.0, 7.5),
        "season": "Rabi", "growing_days": (120, 150),
        "optimal_N": (100, 150), "optimal_P": (50, 80), "optimal_K": (40, 60),
        "avg_yield": 3.5, "max_yield": 6.5,
    },
    "Rice": {
        "optimal_temp": (22, 32), "optimal_humidity": (60, 90),
        "optimal_rainfall": (1000, 2000), "optimal_ph": (5.5, 7.0),
        "season": "Kharif", "growing_days": (100, 150),
        "optimal_N": (80, 120), "optimal_P": (40, 60), "optimal_K": (40, 80),
        "avg_yield": 4.0, "max_yield": 8.0,
    },
    "Maize": {
        "optimal_temp": (20, 30), "optimal_humidity": (50, 80),
        "optimal_rainfall": (500, 800), "optimal_ph": (5.5, 7.5),
        "season": "Kharif", "growing_days": (90, 120),
        "optimal_N": (120, 180), "optimal_P": (60, 80), "optimal_K": (40, 60),
        "avg_yield": 3.0, "max_yield": 7.0,
    },
    "Cotton": {
        "optimal_temp": (25, 35), "optimal_humidity": (50, 70),
        "optimal_rainfall": (600, 1000), "optimal_ph": (6.0, 8.0),
        "season": "Kharif", "growing_days": (150, 200),
        "optimal_N": (80, 120), "optimal_P": (40, 60), "optimal_K": (40, 60),
        "avg_yield": 1.8, "max_yield": 4.0,
    },
    "Sugarcane": {
        "optimal_temp": (25, 38), "optimal_humidity": (60, 85),
        "optimal_rainfall": (1500, 2500), "optimal_ph": (6.0, 7.5),
        "season": "Whole Year", "growing_days": (270, 365),
        "optimal_N": (150, 250), "optimal_P": (60, 100), "optimal_K": (80, 120),
        "avg_yield": 70.0, "max_yield": 120.0,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  3. NDVI INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════

def interpret_ndvi(ndvi_value):
    """Translate a raw NDVI value into a human-readable vegetation health assessment."""
    if ndvi_value >= 0.6:
        return {"category": "Healthy Vegetation", "color": "#22c55e", "icon": "🌿",
                "description": "Dense, healthy vegetation — excellent crop growth.",
                "yield_potential": "High", "health_score": min(100, int(ndvi_value * 110))}
    elif ndvi_value >= 0.3:
        return {"category": "Moderate Vegetation", "color": "#f59e0b", "icon": "🌱",
                "description": "Moderate density — crop may need attention.",
                "yield_potential": "Medium", "health_score": int(ndvi_value * 100)}
    elif ndvi_value >= 0.1:
        return {"category": "Sparse Vegetation", "color": "#f97316", "icon": "🍂",
                "description": "Sparse cover — possible crop stress.",
                "yield_potential": "Low", "health_score": int(ndvi_value * 80)}
    else:
        return {"category": "Barren Land", "color": "#ef4444", "icon": "🏜️",
                "description": "Minimal vegetation — barren or fallow land.",
                "yield_potential": "Very Low", "health_score": max(5, int(ndvi_value * 50))}


# ═══════════════════════════════════════════════════════════════════════════
#  4. DATA / VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def validate_range(value, min_val, max_val, name="value"):
    """Return (is_valid, message) for a numeric range check."""
    if value < min_val or value > max_val:
        return False, f"{name} ({value}) outside expected [{min_val}, {max_val}]"
    return True, "Valid"

def safe_float(value, default=0.0):
    """Safely coerce value to float; return default on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def format_yield(yield_value, crop=""):
    """Format a yield number with appropriate units."""
    if crop == "Sugarcane":
        return f"{yield_value:.1f} tons/ha"
    return f"{yield_value:.2f} tons/ha"

def get_yield_category(predicted, crop):
    """Classify predicted yield as High / Medium / Low relative to crop average."""
    params = CROP_PARAMETERS.get(crop, {})
    avg = params.get("avg_yield", 3.0)
    if predicted >= avg * 1.2:
        return "High", "#22c55e"
    elif predicted >= avg * 0.8:
        return "Medium", "#f59e0b"
    else:
        return "Low", "#ef4444"

def get_timestamp():
    """Current date-time string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_season_from_month(month=None):
    """Map calendar month to Indian crop season."""
    if month is None:
        month = datetime.now().month
    if month in (6, 7, 8, 9, 10):
        return "Kharif"
    elif month in (11, 12, 1, 2, 3):
        return "Rabi"
    return "Zaid"


# ═══════════════════════════════════════════════════════════════════════════
#  5. COLOUR THEME DICTIONARIES
# ═══════════════════════════════════════════════════════════════════════════

THEME_COLORS_DARK = {
    "bg": "#0a0f1a", "bg2": "#111827",
    "card_bg": "rgba(17, 24, 39, 0.85)",
    "card_border": "rgba(76, 175, 80, 0.30)",
    "text": "#e2e8f0", "text2": "#94a3b8",
    "primary": "#22c55e", "primary_dark": "#16a34a",
    "accent": "#4caf50", "surface": "rgba(30, 41, 59, 0.8)",
    "success": "#22c55e", "warning": "#f59e0b",
    "danger": "#ef4444", "info": "#3b82f6", "gold": "#fbbf24",
}

THEME_COLORS_LIGHT = {
    "bg": "#f0fdf4", "bg2": "#ffffff",
    "card_bg": "rgba(255,255,255,0.88)",
    "card_border": "rgba(22,163,74,0.25)",
    "text": "#1e293b", "text2": "#64748b",
    "primary": "#16a34a", "primary_dark": "#15803d",
    "accent": "#22c55e", "surface": "rgba(255,255,255,0.9)",
    "success": "#16a34a", "warning": "#d97706",
    "danger": "#dc2626", "info": "#2563eb", "gold": "#d97706",
}

THEME_COLORS = THEME_COLORS_DARK


# ═══════════════════════════════════════════════════════════════════════════
#  6. CSS STYLESHEET GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def get_custom_css(mode="dark"):
    """
    Return a complete <style> block with animated wallpaper, 3-D glassmorphism
    cards, modern gradient buttons, and neon-glow metric cards.
    mode: 'dark' or 'light'
    """
    t = THEME_COLORS_DARK if mode == "dark" else THEME_COLORS_LIGHT

    if mode == "dark":
        wallpaper = "background: linear-gradient(-45deg, #0a0f1a, #0d1f0d, #0a1628, #111827); background-size: 400% 400%; animation: gradientShift 20s ease infinite;"
        sidebar_bg = "linear-gradient(180deg, #0f1729 0%, #0d1f0d 60%, #0a0f1a 100%)"
    else:
        wallpaper = "background: linear-gradient(-45deg, #f0fdf4, #ecfdf5, #f0f9ff, #fefce8); background-size: 400% 400%; animation: gradientShift 20s ease infinite;"
        sidebar_bg = "linear-gradient(180deg, #f0fdf4 0%, #ecfdf5 60%, #ffffff 100%)"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @keyframes gradientShift {{ 0%{{background-position:0% 50%;}} 50%{{background-position:100% 50%;}} 100%{{background-position:0% 50%;}} }}
    @keyframes float3d {{ 0%,100%{{transform:translateY(0px);}} 50%{{transform:translateY(-6px);}} }}
    @keyframes fadeSlideUp {{ from{{opacity:0;transform:translateY(24px);}} to{{opacity:1;transform:translateY(0);}} }}
    @keyframes pulseGlow {{ 0%,100%{{box-shadow:0 0 5px {t['primary']}33;}} 50%{{box-shadow:0 0 20px {t['primary']}55;}} }}

    .stApp {{ font-family:'Inter',-apple-system,sans-serif !important; {wallpaper} color:{t['text']}; }}
    section[data-testid="stSidebar"] {{ background:{sidebar_bg} !important; }}
    section[data-testid="stSidebar"] * {{ color:{t['text']} !important; }}

    .glass-card {{
        background:{t['card_bg']}; backdrop-filter:blur(24px); -webkit-backdrop-filter:blur(24px);
        border:1px solid {t['card_border']}; border-radius:20px; padding:28px; margin:10px 0;
        box-shadow:0 8px 32px rgba(0,0,0,0.25), 0 2px 8px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.06);
        transition:all 0.4s cubic-bezier(.4,0,.2,1); animation:fadeSlideUp 0.5s ease-out;
        transform-style:preserve-3d; perspective:800px;
    }}
    .glass-card:hover {{ transform:translateY(-6px) scale(1.01);
        box-shadow:0 20px 60px rgba(0,0,0,0.35),0 4px 16px {t['primary']}22; border-color:{t['primary']}66; }}

    .metric-card {{
        background:{t['card_bg']}; backdrop-filter:blur(16px); border:1px solid {t['card_border']};
        border-radius:18px; padding:22px 16px; text-align:center; position:relative; overflow:hidden;
        box-shadow:0 10px 40px rgba(0,0,0,0.2),0 2px 8px rgba(0,0,0,0.12),inset 0 1px 0 rgba(255,255,255,0.05);
        transition:all 0.35s cubic-bezier(.4,0,.2,1); animation:fadeSlideUp 0.5s ease-out, float3d 6s ease-in-out infinite;
    }}
    .metric-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px;
        background:linear-gradient(90deg,{t['primary']},{t['accent']},{t['info']}); border-radius:18px 18px 0 0; }}
    .metric-card:hover {{ transform:translateY(-8px) scale(1.03); box-shadow:0 24px 64px rgba(0,0,0,0.3),0 0 30px {t['primary']}20; }}
    .metric-icon {{ font-size:2.4rem; margin-bottom:6px; }}
    .metric-label {{ font-size:0.78rem; font-weight:700; color:{t['text2']}; text-transform:uppercase; letter-spacing:1.8px; margin-bottom:4px; }}
    .metric-value {{ font-size:1.8rem; font-weight:800; margin:6px 0; }}

    .main-header {{
        background:{t['card_bg']}; backdrop-filter:blur(20px); border:1px solid {t['card_border']};
        border-radius:24px; padding:44px 40px; text-align:center; margin-bottom:28px;
        position:relative; overflow:hidden; box-shadow:0 12px 48px rgba(0,0,0,0.25); animation:fadeSlideUp 0.6s ease-out;
    }}
    .main-title {{ font-size:2.6rem; font-weight:900; color:{t['primary']}; margin-bottom:8px; text-shadow:0 2px 12px {t['primary']}33; }}
    .main-subtitle {{ font-size:1.1rem; color:{t['text2']}; font-weight:400; }}

    .section-header {{ font-size:1.5rem; font-weight:700; color:{t['primary']}; margin:28px 0 14px;
        padding-bottom:10px; border-bottom:2px solid {t['card_border']}; animation:fadeSlideUp 0.4s ease-out; }}

    .stButton > button {{
        background:linear-gradient(135deg,{t['primary']},{t['primary_dark']}) !important;
        color:#ffffff !important; border:none !important; border-radius:14px !important;
        padding:12px 28px !important; font-weight:700 !important; font-size:0.95rem !important;
        letter-spacing:0.5px !important;
        box-shadow:0 4px 16px {t['primary']}44,0 2px 4px rgba(0,0,0,0.15),inset 0 1px 0 rgba(255,255,255,0.15) !important;
        transition:all 0.3s cubic-bezier(.4,0,.2,1) !important;
    }}
    .stButton > button:hover {{
        transform:translateY(-3px) scale(1.02) !important;
        box-shadow:0 8px 28px {t['primary']}66,0 4px 12px rgba(0,0,0,0.2) !important;
    }}
    .stButton > button:active {{ transform:translateY(0) scale(0.98) !important; }}

    .rec-card {{
        background:{t['card_bg']}; backdrop-filter:blur(12px); border-left:4px solid {t['primary']};
        border-radius:14px; padding:18px 22px; margin:10px 0;
        box-shadow:0 4px 16px rgba(0,0,0,0.12); transition:all 0.3s ease; animation:fadeSlideUp 0.4s ease-out;
    }}
    .rec-card:hover {{ transform:translateX(8px); box-shadow:0 8px 24px rgba(0,0,0,0.18); }}
    .rec-card.warning {{ border-left-color:{t['warning']}; }}
    .rec-card.danger {{ border-left-color:{t['danger']}; }}
    .rec-card.info {{ border-left-color:{t['info']}; }}

    .badge {{ display:inline-block; padding:4px 14px; border-radius:20px; font-size:0.78rem; font-weight:700; letter-spacing:.5px; }}
    .badge-success {{ background:{t['success']}22; color:{t['success']}; border:1px solid {t['success']}44; }}
    .badge-warning {{ background:{t['warning']}22; color:{t['warning']}; border:1px solid {t['warning']}44; }}
    .badge-danger {{ background:{t['danger']}22; color:{t['danger']}; border:1px solid {t['danger']}44; }}

    .progress-container {{ background:{t['card_bg']}; border-radius:12px; overflow:hidden; height:14px; margin:8px 0;
        box-shadow:inset 0 2px 4px rgba(0,0,0,0.15); }}
    .progress-bar {{ height:100%; border-radius:12px; background:linear-gradient(90deg,{t['primary']},{t['accent']});
        transition:width 1.2s cubic-bezier(.4,0,.2,1); box-shadow:0 0 10px {t['primary']}44; }}

    .stTabs [data-baseweb="tab-list"] {{ gap:8px; }}
    .stTabs [data-baseweb="tab"] {{ border-radius:12px; padding:10px 20px; background:{t['card_bg']}; border:1px solid {t['card_border']}; }}
    .stTabs [aria-selected="true"] {{ background:linear-gradient(135deg,{t['primary']}33,{t['primary']}22) !important;
        border-color:{t['primary']}88 !important; box-shadow:0 4px 12px {t['primary']}22 !important; }}

    .stDataFrame {{ border-radius:14px; overflow:hidden; }}
    .streamlit-expanderHeader {{ background:{t['card_bg']} !important; border-radius:12px !important; border:1px solid {t['card_border']} !important; }}
    [data-testid="stFileUploader"] {{ border:2px dashed {t['card_border']} !important; border-radius:16px !important; background:{t['card_bg']} !important; }}

    .splash-container {{ display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:60vh; text-align:center; animation:fadeSlideUp 0.8s ease-out; }}
    .splash-title {{ font-size:2.8rem; font-weight:900; color:{t['primary']}; margin-bottom:12px; text-shadow:0 4px 20px {t['primary']}44; }}
    .splash-subtitle {{ font-size:1.15rem; color:{t['text2']}; margin-bottom:32px; }}

    .disease-card {{ background:{t['card_bg']}; backdrop-filter:blur(12px); border:1px solid {t['danger']}44;
        border-radius:16px; padding:20px; margin:8px 0; box-shadow:0 6px 24px rgba(0,0,0,0.15); transition:all 0.3s ease; }}
    .disease-card:hover {{ transform:translateY(-4px); box-shadow:0 12px 36px rgba(0,0,0,0.2); }}

    .footer {{ text-align:center; padding:24px; color:{t['text2']}; font-size:0.82rem;
        border-top:1px solid {t['card_border']}; margin-top:48px; }}

    #MainMenu {{visibility:hidden;}}
    header {{visibility:hidden;}}
    </style>
    """


# ═══════════════════════════════════════════════════════════════════════════
#  7. HTML RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def render_metric_card(icon, label, value, delta=None, color="#22c55e"):
    """Build HTML for a 3-D metric card with neon glow text."""
    delta_html = ""
    if delta:
        try:
            delta_num = float(str(delta).replace('%', '').replace('+', ''))
            d_color = "#22c55e" if delta_num >= 0 else "#ef4444"
        except ValueError:
            d_color = "#94a3b8"
        delta_html = f'<div style="color:{d_color};font-size:0.82rem;margin-top:4px;">{delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};text-shadow:0 0 20px {color}44;">{value}</div>
        {delta_html}
    </div>
    """

def render_glass_card(content, extra_class=""):
    """Wrap arbitrary HTML content inside a glassmorphism card."""
    return f'<div class="glass-card {extra_class}">{content}</div>'

def render_recommendation_card(icon, title, description, category=""):
    """Render a styled recommendation / alert card."""
    cls = category if category in ("warning", "danger", "info") else ""
    return f"""
    <div class="rec-card {cls}">
        <div style="font-size:1.05rem;font-weight:700;margin-bottom:6px;">{icon} {title}</div>
        <div style="color:#94a3b8;font-size:0.9rem;line-height:1.65;">{description}</div>
    </div>
    """

def render_badge(text, badge_type="success"):
    """Small inline badge — badge_type: success | warning | danger."""
    return f'<span class="badge badge-{badge_type}">{text}</span>'

def render_progress_bar(value, max_val=100, color="#22c55e"):
    """Horizontal progress bar (0-100%)."""
    pct = min(100, max(0, (value / max_val) * 100))
    return f"""
    <div class="progress-container">
        <div class="progress-bar" style="width:{pct}%;background:linear-gradient(90deg,{color},{color}cc);"></div>
    </div>
    """
