"""
=============================================================================
 SMART CROP YIELD PREDICTION USING ARTIFICIAL INTELLIGENCE
 Module: soil_analyzer.py — Soil Health & Crop Suitability Analyser
=============================================================================

 PURPOSE:
   Analyse soil parameters (N, P, K, pH, organic carbon, texture) and
   provide crop-specific suitability scores plus integrated suggestions
   from the recommendation engine.

 KEY FUNCTIONS:
   • analyze_soil()           — full soil health report
   • get_crop_suitability()   — rank crops by soil fit
   • get_soil_suggestions()   — integrated recommendation queries

 AUTHOR : AgriTech AI Solutions
 VERSION: 3.0.0
=============================================================================
"""

import numpy as np
from utils import CROP_PARAMETERS


# ═══════════════════════════════════════════════════════════════════════════
#  SOIL HEALTH THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

SOIL_RATINGS = {
    "nitrogen": [
        (280, "High",     "#22c55e"),
        (200, "Medium",   "#f59e0b"),
        (0,   "Low",      "#ef4444"),
    ],
    "phosphorus": [
        (50, "High",      "#22c55e"),
        (25, "Medium",    "#f59e0b"),
        (0,  "Low",       "#ef4444"),
    ],
    "potassium": [
        (280, "High",     "#22c55e"),
        (150, "Medium",   "#f59e0b"),
        (0,   "Low",      "#ef4444"),
    ],
    "ph": [
        (7.5, "Alkaline", "#3b82f6"),
        (6.5, "Neutral",  "#22c55e"),
        (5.5, "Slightly Acidic", "#f59e0b"),
        (0.0, "Acidic",   "#ef4444"),
    ],
    "organic_carbon": [
        (0.75, "High",    "#22c55e"),
        (0.50, "Medium",  "#f59e0b"),
        (0.0,  "Low",     "#ef4444"),
    ],
}

SOIL_TYPES = {
    "Alluvial":  {"fertility": "High",   "drainage": "Good",     "crops": ["Rice", "Wheat", "Sugarcane", "Maize"]},
    "Black":     {"fertility": "High",   "drainage": "Poor",     "crops": ["Cotton", "Soybean", "Wheat", "Chickpea"]},
    "Red":       {"fertility": "Medium", "drainage": "Good",     "crops": ["Groundnut", "Millet", "Maize", "Potato"]},
    "Laterite":  {"fertility": "Low",    "drainage": "Excessive","crops": ["Cashew", "Tea", "Coffee", "Rubber"]},
    "Sandy":     {"fertility": "Low",    "drainage": "Excessive","crops": ["Barley", "Millet", "Groundnut"]},
    "Clay":      {"fertility": "High",   "drainage": "Poor",     "crops": ["Rice", "Wheat", "Cotton"]},
    "Loamy":     {"fertility": "High",   "drainage": "Good",     "crops": ["Wheat", "Sugarcane", "Rice", "Maize", "Soybean"]},
    "Saline":    {"fertility": "Low",    "drainage": "Variable", "crops": ["Barley", "Mustard", "Cotton"]},
}


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_soil(nitrogen, phosphorus, potassium, ph, organic_carbon=0.5,
                 soil_type="Loamy", crop=None):
    """
    Comprehensive soil health analysis.

    Parameters
    ----------
    nitrogen       : float — available N (kg/ha)
    phosphorus     : float — available P (kg/ha)
    potassium      : float — available K (kg/ha)
    ph             : float — soil pH (0-14)
    organic_carbon : float — organic carbon %
    soil_type      : str   — soil texture class
    crop           : str   — target crop (optional)

    Returns
    -------
    dict with overall_score, overall_label, overall_color,
         nutrient_ratings, ph_status, oc_status, soil_info,
         deficiencies, recommendations, crop_suitability (if crop given)
    """
    # Rate each nutrient
    def _rate(param, value):
        for threshold, label, color in SOIL_RATINGS.get(param, []):
            if value >= threshold:
                return {"value": value, "rating": label, "color": color}
        return {"value": value, "rating": "Unknown", "color": "#94a3b8"}

    n_info = _rate("nitrogen", nitrogen)
    p_info = _rate("phosphorus", phosphorus)
    k_info = _rate("potassium", potassium)
    ph_info = _rate("ph", ph)
    oc_info = _rate("organic_carbon", organic_carbon)

    # Identify deficiencies
    deficiencies = []
    if n_info["rating"] == "Low":
        deficiencies.append("Nitrogen")
    if p_info["rating"] == "Low":
        deficiencies.append("Phosphorus")
    if k_info["rating"] == "Low":
        deficiencies.append("Potassium")
    if oc_info["rating"] == "Low":
        deficiencies.append("Organic Carbon")

    # Overall health score (0-100)
    score = 50  # baseline
    for info in [n_info, p_info, k_info, oc_info]:
        if info["rating"] == "High":
            score += 10
        elif info["rating"] == "Medium":
            score += 5
        else:
            score -= 5

    # pH penalty: farther from neutral (6.5-7.0) → lower score
    ph_deviation = abs(ph - 6.75)
    score -= ph_deviation * 5

    score = max(0, min(100, round(score)))

    if score >= 75:
        label, color = "Excellent", "#22c55e"
    elif score >= 55:
        label, color = "Good", "#84cc16"
    elif score >= 35:
        label, color = "Moderate", "#f59e0b"
    else:
        label, color = "Poor", "#ef4444"

    soil_info = SOIL_TYPES.get(soil_type, {
        "fertility": "Unknown", "drainage": "Unknown", "crops": []
    })

    # Recommendations
    recs = _build_soil_recommendations(
        nitrogen, phosphorus, potassium, ph, organic_carbon,
        soil_type, deficiencies, crop
    )

    result = {
        "overall_score":     score,
        "overall_label":     label,
        "overall_color":     color,
        "nutrient_ratings":  {
            "Nitrogen (N)":    n_info,
            "Phosphorus (P)":  p_info,
            "Potassium (K)":   k_info,
        },
        "ph_status":         ph_info,
        "oc_status":         oc_info,
        "soil_type":         soil_type,
        "soil_info":         soil_info,
        "deficiencies":      deficiencies,
        "recommendations":   recs,
    }

    # Crop suitability overlay if crop specified
    if crop:
        result["crop_suitability"] = _crop_soil_fit(
            crop, nitrogen, phosphorus, potassium, ph, soil_type
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  CROP SUITABILITY RANKING
# ═══════════════════════════════════════════════════════════════════════════

def get_crop_suitability(nitrogen, phosphorus, potassium, ph,
                          soil_type="Loamy", top_n=8):
    """
    Rank all known crops by suitability for the given soil.
    Returns list of dicts sorted by score descending.
    """
    rankings = []
    for crop in CROP_PARAMETERS:
        fit = _crop_soil_fit(crop, nitrogen, phosphorus, potassium, ph, soil_type)
        rankings.append({"crop": crop, **fit})

    rankings.sort(key=lambda x: x["score"], reverse=True)
    return rankings[:top_n]


def _crop_soil_fit(crop, n, p, k, ph, soil_type):
    """Score how well soil parameters match a crop's requirements."""
    params = CROP_PARAMETERS.get(crop, {})
    score = 60  # baseline

    # pH preference
    ph_lo, ph_hi = params.get("optimal_ph", (5.5, 7.5))
    if ph_lo <= ph <= ph_hi:
        score += 15
    elif ph < ph_lo - 1 or ph > ph_hi + 1:
        score -= 15
    else:
        score -= 5

    # Nutrient needs (use lower bound of optimal range as minimum requirement)
    n_range = params.get("optimal_N", (80, 120))
    p_range = params.get("optimal_P", (40, 60))
    k_range = params.get("optimal_K", (40, 60))
    n_need = n_range[0] if isinstance(n_range, (list, tuple)) else n_range
    p_need = p_range[0] if isinstance(p_range, (list, tuple)) else p_range
    k_need = k_range[0] if isinstance(k_range, (list, tuple)) else k_range
    if n >= n_need: score += 8
    elif n >= n_need * 0.6: score += 3
    else: score -= 5
    if p >= p_need: score += 5
    if k >= k_need: score += 5

    # Soil type affinity
    soil_crops = SOIL_TYPES.get(soil_type, {}).get("crops", [])
    if crop in soil_crops:
        score += 10

    score = max(0, min(100, score))
    if score >= 75:
        suitability, color = "Highly Suitable", "#22c55e"
    elif score >= 55:
        suitability, color = "Suitable", "#84cc16"
    elif score >= 40:
        suitability, color = "Marginal", "#f59e0b"
    else:
        suitability, color = "Not Recommended", "#ef4444"

    return {"score": score, "suitability": suitability, "color": color}


# ═══════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def _build_soil_recommendations(n, p, k, ph, oc, soil_type, deficiencies, crop):
    """Generate targeted soil improvement recommendations."""
    recs = []

    if "Nitrogen" in deficiencies:
        recs.append("🔴 **Nitrogen deficient** — Apply Urea (46-0-0) at 50-80 kg/ha or incorporate green manure (Dhaincha / Sesbania).")
    if "Phosphorus" in deficiencies:
        recs.append("🔴 **Phosphorus deficient** — Apply DAP (18-46-0) at 40-60 kg/ha or Single Super Phosphate (SSP).")
    if "Potassium" in deficiencies:
        recs.append("🔴 **Potassium deficient** — Apply Muriate of Potash (MOP 0-0-60) at 30-50 kg/ha.")
    if "Organic Carbon" in deficiencies:
        recs.append("🟡 **Low organic carbon** — Add FYM (10-15 t/ha) or vermicompost; practise crop-residue incorporation.")

    if ph < 5.5:
        recs.append(f"⚠️ Soil is acidic (pH {ph:.1f}) — apply agricultural lime (CaCO₃) at 2-4 t/ha to raise pH.")
    elif ph > 8.0:
        recs.append(f"⚠️ Soil is alkaline (pH {ph:.1f}) — apply gypsum (CaSO₄) at 2-5 t/ha and incorporate organic matter.")

    if soil_type == "Sandy":
        recs.append("🏜️ Sandy soil — add organic matter and clay amendments to improve water retention.")
    elif soil_type == "Clay":
        recs.append("🧱 Heavy clay — improve drainage with organic matter and avoid compaction during wet conditions.")

    if crop:
        params = CROP_PARAMETERS.get(crop, {})
        if params:
            t_lo, t_hi = params.get("optimal_temp", (20, 35))
            recs.append(f"🌱 **{crop}** grows best at {t_lo}-{t_hi}°C with pH {params.get('optimal_ph', (5.5, 7.5))[0]}-{params.get('optimal_ph', (5.5, 7.5))[1]}.")

    if not recs:
        recs.append("✅ Soil parameters are within healthy ranges — maintain current management practices.")

    return recs


# ═══════════════════════════════════════════════════════════════════════════
#  INTEGRATED SUGGESTIONS (for combined Soil & Crop page)
# ═══════════════════════════════════════════════════════════════════════════

def get_soil_suggestions(soil_result, crop=None):
    """
    Generate high-level suggestions combining soil analysis with
    crop-specific advice.  Used by the combined Soil & Crop Analyzer page.
    """
    suggestions = []
    deficiencies = soil_result.get("deficiencies", [])
    score = soil_result.get("overall_score", 50)

    if score >= 75:
        suggestions.append("🟢 Soil is in excellent condition for farming.")
    elif score >= 55:
        suggestions.append("🟡 Soil health is acceptable but could be improved.")
    else:
        suggestions.append("🔴 Soil needs significant improvement before planting.")

    if deficiencies:
        suggestions.append(f"Address deficiencies in: **{', '.join(deficiencies)}**.")

    if crop:
        suit = soil_result.get("crop_suitability", {})
        s = suit.get("suitability", "Unknown")
        sc = suit.get("score", 0)
        suggestions.append(f"Suitability for **{crop}**: {s} (score {sc}/100).")
        if sc < 55:
            suggestions.append(f"Consider alternative crops better suited to current soil conditions.")

    # Seasonal tip
    from datetime import datetime
    month = datetime.now().month
    if 6 <= month <= 9:
        suggestions.append("🌧️ Kharif season — good time for Rice, Maize, Cotton, Soybean.")
    elif 10 <= month <= 2:
        suggestions.append("❄️ Rabi season — ideal for Wheat, Barley, Chickpea, Mustard.")
    else:
        suggestions.append("☀️ Zaid/summer season — consider Watermelon, Cucumber, Moong.")

    return suggestions
