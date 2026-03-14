"""
=============================================================================
 SMART CROP YIELD PREDICTION USING ARTIFICIAL INTELLIGENCE
 Module: satellite_ndvi.py — Simulated Satellite & NDVI Analysis
=============================================================================

 PURPOSE:
   Generate realistic NDVI (Normalized Difference Vegetation Index) data
   that mimics satellite imagery analysis.  In a production setting this
   would connect to Sentinel-2 / Landsat APIs; here we simulate values
   based on geographic, seasonal, and crop-specific parameters.

 NDVI SCALE:
   -1.0 → 0.0  : Water / bare soil / non-vegetation
    0.0 → 0.2  : Sparse / stressed vegetation
    0.2 → 0.5  : Moderate canopy (young crops / grassland)
    0.5 → 0.8  : Dense healthy vegetation (prime agriculture)
    0.8 → 1.0  : Very dense vegetation (tropical forest)

 AUTHOR : AgriTech AI Solutions
 VERSION: 3.0.0
=============================================================================
"""

import numpy as np
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

NDVI_HEALTH = {
    "Excellent": {"min": 0.60, "max": 1.00, "color": "#22c55e"},
    "Good":      {"min": 0.40, "max": 0.60, "color": "#84cc16"},
    "Moderate":  {"min": 0.25, "max": 0.40, "color": "#f59e0b"},
    "Poor":      {"min": 0.10, "max": 0.25, "color": "#f97316"},
    "Critical":  {"min":-0.10, "max": 0.10, "color": "#ef4444"},
}


def _season_factor():
    """Return 0–1 seasonal multiplier (India: Kharif Jun-Oct peak)."""
    month = datetime.now().month
    return {1: 0.55, 2: 0.50, 3: 0.45, 4: 0.40, 5: 0.42,
            6: 0.60, 7: 0.80, 8: 0.90, 9: 0.85, 10: 0.75,
            11: 0.65, 12: 0.58}.get(month, 0.6)


def _crop_vigor(crop):
    """Return base NDVI vigor for a given crop type."""
    vigor = {
        "Rice": 0.72, "Wheat": 0.68, "Maize": 0.65, "Sugarcane": 0.78,
        "Cotton": 0.52, "Soybean": 0.60, "Barley": 0.58, "Millet": 0.50,
        "Sorghum": 0.48, "Groundnut": 0.55, "Chickpea": 0.50,
        "Mustard": 0.52, "Potato": 0.62, "Onion": 0.45, "Tomato": 0.58,
    }
    return vigor.get(crop, 0.55)


# ═══════════════════════════════════════════════════════════════════════════
#  NDVI GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_ndvi_data(latitude, longitude, crop=None, grid_size=20):
    """
    Generate a simulated NDVI raster grid and summary statistics.

    Parameters
    ----------
    latitude, longitude : float — centre point
    crop : str | None — crop name for base vigor
    grid_size : int — spatial grid dimension (N×N pixels)

    Returns
    -------
    dict with keys: ndvi_grid, mean, median, std, health, health_color,
                    veg_fraction, water_fraction, bare_fraction,
                    histogram (dict of bins → counts),
                    temporal (list of 12 monthly values)
    """
    np.random.seed(int(abs(latitude * 100 + longitude * 100)) % 100000)

    base = _crop_vigor(crop) if crop else 0.55
    season = _season_factor()
    base *= season

    # Perlin-like spatial noise (using additive sine waves)
    x = np.linspace(0, 4 * np.pi, grid_size)
    y = np.linspace(0, 4 * np.pi, grid_size)
    xx, yy = np.meshgrid(x, y)
    spatial = (
        0.30 * np.sin(xx * 0.8 + yy * 0.5)
        + 0.15 * np.sin(xx * 1.5 - yy * 1.2)
        + 0.10 * np.cos(xx * 2.1 + yy * 0.3)
    )
    # Scale spatial pattern to tight range
    spatial = 0.12 * (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-9)

    noise = np.random.normal(0, 0.04, (grid_size, grid_size))
    ndvi_grid = np.clip(base + spatial + noise, -0.1, 0.95)

    mean_val = float(np.mean(ndvi_grid))
    median_val = float(np.median(ndvi_grid))
    std_val = float(np.std(ndvi_grid))

    # Map mean to health category
    health = "Critical"
    health_color = "#ef4444"
    for label, info in NDVI_HEALTH.items():
        if info["min"] <= mean_val < info["max"]:
            health = label
            health_color = info["color"]
            break
    if mean_val >= 0.60:
        health, health_color = "Excellent", "#22c55e"

    # Fractional cover estimates
    veg   = float(np.mean(ndvi_grid > 0.2))
    water = float(np.mean(ndvi_grid < 0.0))
    bare  = 1 - veg - water

    # Histogram for charts
    bins = np.arange(-0.1, 1.05, 0.05)
    counts, edges = np.histogram(ndvi_grid.flatten(), bins=bins)
    histogram = {f"{edges[i]:.2f}": int(counts[i]) for i in range(len(counts))}

    # 12-month temporal NDVI series
    monthly_factors = [0.55, 0.50, 0.45, 0.40, 0.42, 0.60,
                       0.80, 0.90, 0.85, 0.75, 0.65, 0.58]
    base_raw = _crop_vigor(crop) if crop else 0.55
    temporal = [round(base_raw * f + np.random.normal(0, 0.02), 3)
                for f in monthly_factors]

    return {
        "ndvi_grid":      ndvi_grid,
        "mean":           round(mean_val, 4),
        "median":         round(median_val, 4),
        "std":            round(std_val, 4),
        "health":         health,
        "health_color":   health_color,
        "veg_fraction":   round(veg, 3),
        "water_fraction": round(water, 3),
        "bare_fraction":  round(bare, 3),
        "histogram":      histogram,
        "temporal":       temporal,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  NDVI ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def get_ndvi_analysis(ndvi_data, crop=None):
    """
    Interpret NDVI results into human-friendly insights & recommendations.
    """
    if ndvi_data is None:
        return {"status": "No data", "insights": [], "recommendations": []}

    mean  = ndvi_data.get("mean", 0)
    health = ndvi_data.get("health", "Unknown")
    veg   = ndvi_data.get("veg_fraction", 0)
    bare  = ndvi_data.get("bare_fraction", 0)

    insights = [
        f"Mean NDVI: **{mean:.3f}** — vegetation health rated **{health}**.",
        f"Vegetated area: **{veg*100:.1f}%** of field.",
        f"Bare / sparse soil: **{bare*100:.1f}%**.",
    ]
    recs = []

    if health == "Excellent":
        recs.append("✅ Vegetation is thriving — maintain current practices.")
    elif health == "Good":
        recs.append("🟢 Healthy canopy detected. Monitor for any stress patches.")
    elif health == "Moderate":
        recs.append("🟡 Some areas show reduced vigour — scout field for nutrient deficiency or pest pressure.")
        if crop:
            recs.append(f"Consider foliar nutrient spray targeted at low-NDVI zones for {crop}.")
    elif health == "Poor":
        recs.append("🟠 Significant stress detected — check irrigation, soil health, disease.")
        recs.append("Targeted soil sampling in low-NDVI zones recommended.")
    else:
        recs.append("🔴 Critical vegetation stress — immediate action required.")
        recs.append("Inspect for water stress, disease, or pest infestation.")

    if bare > 0.4:
        recs.append("⚠️ Large bare-soil fraction — consider cover-cropping or mulch to prevent erosion.")

    return {
        "status":          health,
        "insights":        insights,
        "recommendations": recs,
    }


def get_ndvi_change_analysis(current, previous_mean=None):
    """
    Compare current NDVI with a previous period to detect trends.
    """
    if previous_mean is None:
        previous_mean = current["mean"] - np.random.uniform(-0.08, 0.08)

    change = current["mean"] - previous_mean
    pct = (change / max(abs(previous_mean), 0.01)) * 100

    if change > 0.05:
        trend, icon = "Improving", "📈"
    elif change < -0.05:
        trend, icon = "Declining", "📉"
    else:
        trend, icon = "Stable", "➡️"

    return {
        "previous_mean": round(previous_mean, 4),
        "current_mean":  current["mean"],
        "change":        round(change, 4),
        "change_pct":    round(pct, 1),
        "trend":         trend,
        "icon":          icon,
    }
