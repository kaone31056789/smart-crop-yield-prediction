"""
dataset.py  –  Indian Crop Production Dataset (Government of India)
──────────────────────────────────────────────────────────────────────
Primary source: Government of India, Ministry of Agriculture & Farmers' Welfare
               Crop Production Statistics — data.gov.in

Downloads real crop production records, computes yield per hectare,
and augments with soil / climate features for ML training.
Falls back to model-based synthetic generation when offline.
"""

import numpy as np
import pandas as pd
import os, warnings

np.random.seed(42)
warnings.filterwarnings("ignore")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  File paths & Data source URLs                                                #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
RAW_CACHE  = "raw_govt_crop_data.csv"
DATA_CACHE = "crop_data.csv"
INFO_CACHE = "data_info.pkl"

# Mirrors of the Government of India crop production statistics (originally from data.gov.in)
DATA_URLS = [
    "https://raw.githubusercontent.com/dishika123/Crop_analysis_dashboard/main/crop.csv",
    "https://raw.githubusercontent.com/dishika123/Crop_analysis_dashboard/main/India%20Agriculture%20Crop%20Production.csv",
]

DATA_SOURCE_GOV  = "Government of India — Ministry of Agriculture (data.gov.in)"
DATA_SOURCE_SYN  = "Synthetic (modelled on Government of India agricultural statistics)"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Crop profiles (realistic yield ranges in tonnes / hectare)                   #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
CROPS = {
    "Rice": {
        "yield_range": (2.0, 5.5), "optimal_temp": (22, 30), "optimal_rain": (1200, 2000),
        "N": (60, 120), "P": (20, 60), "K": (20, 60), "pH": (5.5, 7.0), "humidity": (70, 90),
        "seasons": ["Kharif", "Whole Year"],
        "soils": ["Alluvial", "Clay", "Loamy"],
    },
    "Wheat": {
        "yield_range": (2.0, 5.0), "optimal_temp": (14, 22), "optimal_rain": (400, 700),
        "N": (80, 150), "P": (30, 60), "K": (20, 50), "pH": (6.0, 7.5), "humidity": (40, 65),
        "seasons": ["Rabi"],
        "soils": ["Loamy", "Alluvial", "Clay"],
    },
    "Maize": {
        "yield_range": (2.5, 7.0), "optimal_temp": (20, 30), "optimal_rain": (600, 1000),
        "N": (80, 160), "P": (30, 60), "K": (20, 50), "pH": (5.8, 7.0), "humidity": (50, 80),
        "seasons": ["Kharif", "Rabi"],
        "soils": ["Loamy", "Sandy Loam", "Alluvial"],
    },
    "Sugarcane": {
        "yield_range": (55, 100), "optimal_temp": (24, 33), "optimal_rain": (1100, 1600),
        "N": (150, 300), "P": (50, 100), "K": (60, 120), "pH": (6.0, 7.5), "humidity": (65, 90),
        "seasons": ["Whole Year"],
        "soils": ["Loamy", "Alluvial", "Black"],
    },
    "Cotton": {
        "yield_range": (1.0, 3.5), "optimal_temp": (25, 35), "optimal_rain": (600, 1000),
        "N": (40, 100), "P": (20, 50), "K": (20, 40), "pH": (6.0, 8.0), "humidity": (40, 70),
        "seasons": ["Kharif"],
        "soils": ["Black", "Loamy", "Red"],
    },
    "Soybean": {
        "yield_range": (1.0, 3.0), "optimal_temp": (20, 28), "optimal_rain": (500, 800),
        "N": (20, 50), "P": (40, 80), "K": (20, 40), "pH": (6.0, 7.0), "humidity": (50, 75),
        "seasons": ["Kharif"],
        "soils": ["Black", "Loamy", "Red"],
    },
    "Potato": {
        "yield_range": (15, 40), "optimal_temp": (14, 22), "optimal_rain": (500, 750),
        "N": (100, 200), "P": (50, 100), "K": (80, 150), "pH": (5.5, 6.5), "humidity": (60, 80),
        "seasons": ["Rabi"],
        "soils": ["Sandy Loam", "Loamy", "Alluvial"],
    },
    "Tomato": {
        "yield_range": (15, 45), "optimal_temp": (18, 27), "optimal_rain": (400, 650),
        "N": (80, 150), "P": (40, 80), "K": (60, 120), "pH": (6.0, 7.0), "humidity": (50, 75),
        "seasons": ["Rabi", "Kharif"],
        "soils": ["Loamy", "Sandy Loam", "Red"],
    },
    "Mustard": {
        "yield_range": (0.8, 2.5), "optimal_temp": (12, 22), "optimal_rain": (350, 550),
        "N": (40, 80), "P": (20, 50), "K": (10, 30), "pH": (6.0, 7.5), "humidity": (40, 60),
        "seasons": ["Rabi"],
        "soils": ["Loamy", "Alluvial", "Sandy Loam"],
    },
    "Chickpea": {
        "yield_range": (0.7, 2.2), "optimal_temp": (18, 28), "optimal_rain": (400, 650),
        "N": (15, 40), "P": (30, 60), "K": (15, 35), "pH": (6.0, 7.5), "humidity": (35, 55),
        "seasons": ["Rabi"],
        "soils": ["Black", "Loamy", "Red"],
    },
    "Groundnut": {
        "yield_range": (1.0, 3.5), "optimal_temp": (22, 30), "optimal_rain": (500, 800),
        "N": (10, 30), "P": (30, 60), "K": (30, 50), "pH": (5.5, 7.0), "humidity": (50, 70),
        "seasons": ["Kharif"],
        "soils": ["Sandy Loam", "Red", "Loamy"],
    },
    "Jute": {
        "yield_range": (2.0, 4.5), "optimal_temp": (25, 35), "optimal_rain": (1200, 1800),
        "N": (40, 80), "P": (15, 40), "K": (15, 40), "pH": (5.5, 7.0), "humidity": (70, 90),
        "seasons": ["Kharif"],
        "soils": ["Alluvial", "Loamy", "Clay"],
    },
    "Banana": {
        "yield_range": (20, 60), "optimal_temp": (24, 32), "optimal_rain": (1000, 1800),
        "N": (150, 300), "P": (30, 80), "K": (200, 400), "pH": (6.0, 7.5), "humidity": (70, 90),
        "seasons": ["Whole Year"],
        "soils": ["Loamy", "Alluvial", "Clay"],
    },
    "Barley": {
        "yield_range": (2.0, 4.5), "optimal_temp": (10, 20), "optimal_rain": (300, 550),
        "N": (50, 100), "P": (25, 50), "K": (15, 35), "pH": (6.5, 8.0), "humidity": (35, 55),
        "seasons": ["Rabi"],
        "soils": ["Loamy", "Sandy Loam", "Alluvial"],
    },
    "Tea": {
        "yield_range": (1.5, 3.5), "optimal_temp": (16, 28), "optimal_rain": (1500, 2500),
        "N": (80, 200), "P": (30, 60), "K": (30, 80), "pH": (4.5, 5.5), "humidity": (70, 95),
        "seasons": ["Whole Year"],
        "soils": ["Red", "Laterite", "Loamy"],
    },
    "Coffee": {
        "yield_range": (0.5, 2.5), "optimal_temp": (18, 28), "optimal_rain": (1200, 2000),
        "N": (60, 150), "P": (20, 60), "K": (40, 100), "pH": (5.0, 6.5), "humidity": (65, 85),
        "seasons": ["Whole Year"],
        "soils": ["Laterite", "Red", "Loamy"],
    },
}

SEASONS = ["Kharif", "Rabi", "Zaid", "Whole Year"]

SOIL_TYPES = {
    "Sandy":      {"quality": 0.60, "water_ret": 0.3},
    "Sandy Loam": {"quality": 0.80, "water_ret": 0.5},
    "Clay":       {"quality": 0.75, "water_ret": 0.9},
    "Loamy":      {"quality": 1.00, "water_ret": 0.7},
    "Black":      {"quality": 0.95, "water_ret": 0.85},
    "Red":        {"quality": 0.78, "water_ret": 0.55},
    "Alluvial":   {"quality": 0.95, "water_ret": 0.65},
    "Laterite":   {"quality": 0.65, "water_ret": 0.45},
}

STATES = {
    "Punjab":           {"lat": 30.73, "lon": 76.78, "temp": 24.0, "rain": 650},
    "Haryana":          {"lat": 29.06, "lon": 76.09, "temp": 25.5, "rain": 550},
    "Uttar Pradesh":    {"lat": 26.85, "lon": 80.91, "temp": 26.0, "rain": 900},
    "Bihar":            {"lat": 25.60, "lon": 85.14, "temp": 26.5, "rain": 1100},
    "West Bengal":      {"lat": 22.57, "lon": 88.36, "temp": 27.0, "rain": 1500},
    "Madhya Pradesh":   {"lat": 23.26, "lon": 77.41, "temp": 26.5, "rain": 1100},
    "Rajasthan":        {"lat": 26.92, "lon": 75.79, "temp": 27.5, "rain": 400},
    "Maharashtra":      {"lat": 19.08, "lon": 72.88, "temp": 27.0, "rain": 1200},
    "Karnataka":        {"lat": 12.97, "lon": 77.59, "temp": 26.0, "rain": 1200},
    "Tamil Nadu":       {"lat": 13.08, "lon": 80.27, "temp": 28.5, "rain": 950},
    "Andhra Pradesh":   {"lat": 15.91, "lon": 79.74, "temp": 28.0, "rain": 900},
    "Gujarat":          {"lat": 23.02, "lon": 72.57, "temp": 28.0, "rain": 800},
    "Odisha":           {"lat": 20.30, "lon": 85.83, "temp": 27.5, "rain": 1400},
    "Assam":            {"lat": 26.14, "lon": 91.74, "temp": 25.0, "rain": 2200},
    "Telangana":        {"lat": 17.39, "lon": 78.49, "temp": 27.5, "rain": 950},
    "Kerala":           {"lat": 10.85, "lon": 76.27, "temp": 27.0, "rain": 2800},
    "Chhattisgarh":     {"lat": 21.25, "lon": 81.63, "temp": 26.5, "rain": 1300},
    "Jharkhand":        {"lat": 23.36, "lon": 85.33, "temp": 26.0, "rain": 1200},
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Name mappings (government dataset → our labels)                              #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
CROP_NAME_MAP = {
    "rice": "Rice", "wheat": "Wheat", "maize": "Maize",
    "sugarcane": "Sugarcane",
    "cotton(lint)": "Cotton", "cotton": "Cotton",
    "soyabean": "Soybean", "soybean": "Soybean",
    "potato": "Potato",
    "tomato": "Tomato", "tomatoes": "Tomato",
    "rapeseed &mustard": "Mustard", "mustard ": "Mustard", "mustard": "Mustard",
    "rapeseed": "Mustard", "rapeseeds": "Mustard",
    "gram": "Chickpea", "chickpea": "Chickpea", "bengal gram": "Chickpea",
    "groundnut": "Groundnut", "ground nut": "Groundnut",
    "jute": "Jute", "jute & mesta": "Jute", "mesta": "Jute",
    "banana": "Banana",
    "barley": "Barley",
    "tea": "Tea",
    "coffee": "Coffee",
}

STATE_NAME_MAP = {
    "punjab": "Punjab", "haryana": "Haryana",
    "uttar pradesh": "Uttar Pradesh", "bihar": "Bihar",
    "west bengal": "West Bengal", "madhya pradesh": "Madhya Pradesh",
    "rajasthan": "Rajasthan", "maharashtra": "Maharashtra",
    "karnataka": "Karnataka", "tamil nadu": "Tamil Nadu",
    "andhra pradesh": "Andhra Pradesh", "gujarat": "Gujarat",
    "odisha": "Odisha", "orissa": "Odisha",
    "assam": "Assam", "telangana": "Telangana",
    "kerala": "Kerala", "chhattisgarh": "Chhattisgarh",
    "jharkhand": "Jharkhand", "chattisgarh": "Chhattisgarh",
}

SEASON_NAME_MAP = {
    "kharif": "Kharif", "rabi": "Rabi",
    "whole year": "Whole Year", "annual": "Whole Year",
    "summer": "Zaid", "zaid": "Zaid",
    "winter": "Rabi", "autumn": "Kharif",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Download real government data                                                #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
def _download_raw_data() -> pd.DataFrame | None:
    """Download crop production stats from government data mirrors."""
    if not HAS_REQUESTS:
        print("[dataset] requests module not available — skipping download")
        return None

    # Check local raw cache first
    if os.path.exists(RAW_CACHE):
        try:
            df = pd.read_csv(RAW_CACHE)
            if len(df) > 1000:
                print(f"[dataset] Loaded {len(df):,} raw records from local cache")
                return df
        except Exception:
            pass

    for url in DATA_URLS:
        try:
            print(f"[dataset] Downloading from: {url}")
            resp = requests.get(url, timeout=45)
            resp.raise_for_status()

            with open(RAW_CACHE, "wb") as f:
                f.write(resp.content)

            df = pd.read_csv(RAW_CACHE)
            if len(df) > 1000:
                print(f"[dataset] ✓ Downloaded {len(df):,} records from government data source")
                return df
        except Exception as e:
            print(f"[dataset] ✗ {url} — {e}")
            continue

    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Process raw government data                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
def _identify_columns(df: pd.DataFrame) -> dict:
    """Flexible column identification for varying CSV formats."""
    mapping = {}
    for c in df.columns:
        cl = c.strip().lower().replace("_", " ")
        # Skip unit columns
        if "unit" in cl:
            continue
        if "state" in cl and "district" not in cl:
            mapping["State"] = c
        elif "crop year" in cl or (cl == "year") or ("crop_year" in c.lower()):
            mapping["Year"] = c
        elif "season" in cl:
            mapping["Season"] = c
        elif "crop" in cl and "year" not in cl and "production" not in cl:
            mapping["Crop"] = c
        elif cl in ("area", "area_ha", "area ha"):
            mapping["Area"] = c
        elif cl in ("production", "production_tonnes"):
            mapping["Production"] = c
        elif cl in ("yield", "yield_ton_per_ha"):
            mapping["Yield"] = c
        elif "district" in cl:
            mapping["District"] = c
    return mapping


def _augment_features(crop: str, state: str, season: str) -> dict:
    """Generate augmented soil / weather / input features for a real data row."""
    crop_info  = CROPS.get(crop, CROPS["Rice"])
    state_info = STATES.get(state, {"temp": 26, "rain": 1000})

    # Soil type
    if np.random.random() < 0.8:
        soil_type = np.random.choice(crop_info["soils"])
    else:
        soil_type = np.random.choice(list(SOIL_TYPES.keys()))

    # NPK from crop requirements + natural variation
    nitrogen   = np.round(np.random.uniform(*crop_info["N"]) * np.random.uniform(0.7, 1.3), 1)
    phosphorus = np.round(np.random.uniform(*crop_info["P"]) * np.random.uniform(0.7, 1.3), 1)
    potassium  = np.round(np.random.uniform(*crop_info["K"]) * np.random.uniform(0.7, 1.3), 1)

    # pH
    ph = np.round(np.random.uniform(crop_info["pH"][0] - 0.5, crop_info["pH"][1] + 0.5), 2)
    ph = np.clip(ph, 3.5, 9.5)

    # Climate (state profile + natural variation)
    temperature = np.round(state_info["temp"] + np.random.normal(0, 3), 1)
    temperature = np.clip(temperature, 5, 50)

    h_lo, h_hi = crop_info["humidity"]
    humidity = np.round(np.random.uniform(h_lo - 10, h_hi + 10), 1)
    humidity = np.clip(humidity, 20, 99)

    rainfall = np.round(state_info["rain"] * np.random.uniform(0.5, 1.5), 1)
    rainfall = np.clip(rainfall, 100, 4000)

    # Farming inputs
    fertilizer = np.round(np.random.uniform(30, 350), 1)
    pesticide  = np.round(np.random.uniform(0.5, 25), 2)
    irrigation = np.random.choice([0, 1], p=[0.35, 0.65])

    return {
        "Soil_Type": soil_type,
        "Nitrogen": nitrogen, "Phosphorus": phosphorus, "Potassium": potassium,
        "pH": ph, "Temperature": temperature, "Humidity": humidity,
        "Rainfall": rainfall, "Fertilizer": fertilizer,
        "Pesticide": pesticide, "Irrigation": irrigation,
    }


def _process_raw_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Process government crop production data into model-ready format."""
    df = raw_df.copy()
    col_map = _identify_columns(df)

    required = {"State", "Crop"}
    has_yield = "Yield" in col_map
    has_area_prod = "Area" in col_map and "Production" in col_map
    if not required.issubset(col_map.keys()) or (not has_yield and not has_area_prod):
        print(f"[dataset] Missing columns. Found: {col_map}. Need: State, Crop + (Yield or Area+Production)")
        return pd.DataFrame(), {}

    # Rename to standard columns
    rename_dict = {v: k for k, v in col_map.items()}
    df = df.rename(columns=rename_dict)

    # ── Clean & filter ──
    # Use pre-computed Yield column if available, else compute from Area/Production
    if "Yield" in df.columns:
        df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
        df = df.dropna(subset=["Yield"])
        df = df[df["Yield"] > 0]
        df["Yield_raw"] = df["Yield"]
        # Still need Area for the final dataset
        if "Area" in df.columns:
            df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
            df["Area"] = df["Area"].fillna(100)
        else:
            df["Area"] = 100.0
    else:
        df = df.dropna(subset=["Area", "Production"])
        df["Area"]       = pd.to_numeric(df["Area"], errors="coerce")
        df["Production"] = pd.to_numeric(df["Production"], errors="coerce")
        df = df.dropna(subset=["Area", "Production"])
        df = df[(df["Area"] > 0) & (df["Production"] > 0)]
        df["Yield_raw"] = df["Production"] / df["Area"]

    # Map crop names
    df["Crop_clean"]  = df["Crop"].str.strip().str.lower()
    df["Crop_mapped"] = df["Crop_clean"].map(CROP_NAME_MAP)
    df = df.dropna(subset=["Crop_mapped"])

    # Map state names
    df["State_clean"]  = df["State"].str.strip().str.lower()
    df["State_mapped"] = df["State_clean"].map(STATE_NAME_MAP)
    df = df.dropna(subset=["State_mapped"])

    # Map seasons
    if "Season" in df.columns:
        df["Season_clean"]  = df["Season"].str.strip().str.lower()
        df["Season_mapped"] = df["Season_clean"].map(SEASON_NAME_MAP)
        df["Season_mapped"] = df["Season_mapped"].fillna("Kharif")
    else:
        df["Season_mapped"] = "Kharif"

    # Filter outliers per crop
    records = []
    for crop_name, crop_info in CROPS.items():
        ylo, yhi = crop_info["yield_range"]
        mask = (
            (df["Crop_mapped"] == crop_name) &
            (df["Yield_raw"] >= ylo * 0.3) &
            (df["Yield_raw"] <= yhi * 2.5)
        )
        crop_df = df[mask]
        if len(crop_df) > 0:
            records.append(crop_df)

    if not records:
        return pd.DataFrame(), {}

    filtered = pd.concat(records, ignore_index=True)

    # Prefer recent years
    year_min, year_max = None, None
    if "Year" in filtered.columns:
        filtered["Year"] = pd.to_numeric(filtered["Year"], errors="coerce")
        filtered = filtered.dropna(subset=["Year"])
        year_min = int(filtered["Year"].min())
        year_max = int(filtered["Year"].max())
        recent_cutoff = year_max - 10
        recent = filtered[filtered["Year"] >= recent_cutoff]
        if len(recent) > 2000:
            filtered = recent

    # ── Build final dataset with augmented features ──
    final_records = []
    for _, row in filtered.iterrows():
        crop   = row["Crop_mapped"]
        state  = row["State_mapped"]
        season = row.get("Season_mapped", "Kharif")
        y      = row["Yield_raw"]

        area = np.clip(row["Area"] / max(1, row["Area"] // 500 + 1), 0.5, 500)
        area = np.round(area, 2)

        aug = _augment_features(crop, state, season)
        final_records.append({
            "Crop": crop, "Season": season, "State": state,
            "Soil_Type": aug["Soil_Type"],
            "Area_ha": area,
            "Nitrogen": aug["Nitrogen"], "Phosphorus": aug["Phosphorus"],
            "Potassium": aug["Potassium"], "pH": aug["pH"],
            "Temperature": aug["Temperature"], "Humidity": aug["Humidity"],
            "Rainfall": aug["Rainfall"], "Fertilizer": aug["Fertilizer"],
            "Pesticide": aug["Pesticide"], "Irrigation": aug["Irrigation"],
            "Yield_ton_per_ha": np.round(y, 3),
        })

    result = pd.DataFrame(final_records)

    # Supplement missing crops with synthetic records so all 16 are available
    present_crops = set(result["Crop"].unique()) if len(result) > 0 else set()
    missing_crops = set(CROPS.keys()) - present_crops
    if missing_crops:
        print(f"[dataset] Supplementing {len(missing_crops)} missing crops with synthetic data: {missing_crops}")
        supplement = []
        state_names = list(STATES.keys())
        for crop in missing_crops:
            c_info = CROPS[crop]
            for _ in range(200):
                state = np.random.choice(state_names)
                season = np.random.choice(c_info["seasons"]) if np.random.random() < 0.85 \
                         else np.random.choice(SEASONS)
                aug = _augment_features(crop, state, season)
                # Compute realistic yield from crop profile
                y_lo, y_hi = c_info["yield_range"]
                base = (y_lo + y_hi) / 2
                cy = base * np.random.uniform(0.6, 1.4)
                cy = np.round(max(cy, y_lo * 0.1), 3)
                supplement.append({
                    "Crop": crop, "Season": season, "State": state,
                    "Soil_Type": aug["Soil_Type"],
                    "Area_ha": np.round(np.clip(np.random.lognormal(2.5, 1.2), 0.5, 500), 2),
                    "Nitrogen": aug["Nitrogen"], "Phosphorus": aug["Phosphorus"],
                    "Potassium": aug["Potassium"], "pH": aug["pH"],
                    "Temperature": aug["Temperature"], "Humidity": aug["Humidity"],
                    "Rainfall": aug["Rainfall"], "Fertilizer": aug["Fertilizer"],
                    "Pesticide": aug["Pesticide"], "Irrigation": aug["Irrigation"],
                    "Yield_ton_per_ha": cy,
                })
        result = pd.concat([result, pd.DataFrame(supplement)], ignore_index=True)

    # Cap to ~10 000 records for training speed
    if len(result) > 10000:
        result = result.sample(n=10000, random_state=42).reset_index(drop=True)

    data_info = {
        "source": DATA_SOURCE_GOV,
        "n_records": len(result),
        "n_crops": result["Crop"].nunique(),
        "n_states": result["State"].nunique(),
        "year_min": year_min,
        "year_max": year_max,
        "raw_records_total": len(raw_df),
        "raw_records_matched": len(filtered),
    }

    print(f"[dataset] Processed {data_info['n_records']:,} records "
          f"({data_info['n_crops']} crops, {data_info['n_states']} states)")
    if year_min and year_max:
        print(f"[dataset] Year range: {year_min}–{year_max}")

    return result, data_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Synthetic fallback (based on official Indian agri statistics)                 #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
def _generate_synthetic(n_samples: int = 8000) -> tuple[pd.DataFrame, dict]:
    """Generate synthetic dataset modelled on government agriculture statistics."""
    records = []
    state_names = list(STATES.keys())

    for _ in range(n_samples):
        crop = np.random.choice(list(CROPS.keys()))
        info = CROPS[crop]

        season = np.random.choice(info["seasons"]) if np.random.random() < 0.85 \
                 else np.random.choice(SEASONS)
        state = np.random.choice(state_names)
        st_info = STATES[state]

        soil_type = np.random.choice(info["soils"]) if np.random.random() < 0.8 \
                    else np.random.choice(list(SOIL_TYPES.keys()))
        soil_info = SOIL_TYPES[soil_type]

        area = np.round(np.clip(np.random.lognormal(2.5, 1.2), 0.5, 500), 2)

        n_lo, n_hi = info["N"]; p_lo, p_hi = info["P"]; k_lo, k_hi = info["K"]
        nitrogen   = np.round(np.random.uniform(n_lo * 0.6, n_hi * 1.3), 1)
        phosphorus = np.round(np.random.uniform(p_lo * 0.6, p_hi * 1.3), 1)
        potassium  = np.round(np.random.uniform(k_lo * 0.6, k_hi * 1.3), 1)

        ph_lo, ph_hi = info["pH"]
        ph = np.round(np.clip(np.random.uniform(ph_lo - 0.8, ph_hi + 0.8), 3.5, 9.5), 2)

        opt_t_lo, opt_t_hi = info["optimal_temp"]
        temperature = np.round(np.clip(st_info["temp"] + np.random.normal(0, 4),
                                       opt_t_lo - 10, opt_t_hi + 12), 1)

        h_lo, h_hi = info["humidity"]
        humidity = np.round(np.clip(np.random.uniform(h_lo - 15, h_hi + 10), 20, 99), 1)

        rainfall = np.round(np.clip(st_info["rain"] * np.random.uniform(0.5, 1.5), 100, 4000), 1)
        fertilizer = np.round(np.random.uniform(30, 350), 1)
        pesticide  = np.round(np.random.uniform(0.5, 25), 2)
        irrigation = np.random.choice([0, 1], p=[0.35, 0.65])

        y_lo, y_hi = info["yield_range"]
        base = (y_lo + y_hi) / 2
        t_mid = (opt_t_lo + opt_t_hi) / 2
        r_mid = (info["optimal_rain"][0] + info["optimal_rain"][1]) / 2
        hum_mid = (h_lo + h_hi) / 2
        n_mid, p_mid, k_mid = (n_lo+n_hi)/2, (p_lo+p_hi)/2, (k_lo+k_hi)/2
        ph_mid = (ph_lo + ph_hi) / 2

        temp_sc = np.clip(1 - abs(temperature - t_mid) / 18, 0.15, 1.0)
        rain_sc = np.clip(1 - abs(rainfall - r_mid) / (r_mid * 2), 0.15, 1.0)
        hum_sc  = np.clip(1 - abs(humidity - hum_mid) / 60, 0.4, 1.0)
        npk_sc  = np.clip(((1-abs(nitrogen-n_mid)/(n_mid*2)) +
                            (1-abs(phosphorus-p_mid)/(p_mid*2)) +
                            (1-abs(potassium-k_mid)/(k_mid*2))) / 3, 0.3, 1.0)
        ph_sc   = np.clip(1 - abs(ph - ph_mid) / 3, 0.3, 1.0)
        soil_q  = soil_info["quality"]
        irr_b   = 1.12 if irrigation else 1.0
        fert_sc = np.clip(np.log1p(fertilizer) / np.log1p(250), 0.5, 1.15)
        pest_sc = np.clip(1 - pesticide / 50, 0.65, 1.0)
        seas_b  = 1.08 if season in info["seasons"] else 0.85

        cy = base * temp_sc * rain_sc * hum_sc * npk_sc * ph_sc * soil_q * irr_b * fert_sc * pest_sc * seas_b
        cy = np.round(max(cy + np.random.normal(0, base * 0.06), y_lo * 0.1), 3)

        records.append({
            "Crop": crop, "Season": season, "State": state, "Soil_Type": soil_type,
            "Area_ha": area, "Nitrogen": nitrogen, "Phosphorus": phosphorus,
            "Potassium": potassium, "pH": ph, "Temperature": temperature,
            "Humidity": humidity, "Rainfall": rainfall, "Fertilizer": fertilizer,
            "Pesticide": pesticide, "Irrigation": irrigation,
            "Yield_ton_per_ha": cy,
        })

    df = pd.DataFrame(records)
    data_info = {
        "source": DATA_SOURCE_SYN,
        "n_records": len(df),
        "n_crops": df["Crop"].nunique(),
        "n_states": df["State"].nunique(),
        "year_min": None, "year_max": None,
        "raw_records_total": 0, "raw_records_matched": 0,
    }
    return df, data_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Main entry point                                                             #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
def load_dataset(force_refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Load the crop production dataset.
    Priority: 1. Local cache → 2. Download real govt data → 3. Synthetic fallback.
    Returns (DataFrame, data_info_dict).
    """
    import joblib

    # 1. Check local cache
    if not force_refresh and os.path.exists(DATA_CACHE) and os.path.exists(INFO_CACHE):
        try:
            df = pd.read_csv(DATA_CACHE)
            info = joblib.load(INFO_CACHE)
            if len(df) > 100:
                print(f"[dataset] Loaded {len(df):,} records from local cache")
                return df, info
        except Exception:
            pass

    # 2. Try downloading real government data
    raw = _download_raw_data()
    if raw is not None and len(raw) > 0:
        df, info = _process_raw_data(raw)
        if len(df) > 500:
            df.to_csv(DATA_CACHE, index=False)
            joblib.dump(info, INFO_CACHE)
            return df, info

    # 3. Fallback to synthetic generation
    print("[dataset] Using synthetic data (modelled on government statistics)")
    df, info = _generate_synthetic(8000)
    df.to_csv(DATA_CACHE, index=False)
    joblib.dump(info, INFO_CACHE)
    return df, info


# Backward compatibility alias
def generate_dataset(n_samples: int = 8000, save_path: str = "crop_data.csv") -> pd.DataFrame:
    """Legacy wrapper — now calls load_dataset()."""
    df, _ = load_dataset(force_refresh=True)
    df.to_csv(save_path, index=False)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Helper functions                                                             #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
def get_crop_names():
    return sorted(CROPS.keys())

def get_season_names():
    return SEASONS

def get_soil_names():
    return sorted(SOIL_TYPES.keys())

def get_state_names():
    return sorted(STATES.keys())

def get_crop_info(crop_name):
    return CROPS.get(crop_name)

def get_state_info(state_name):
    return STATES.get(state_name)


if __name__ == "__main__":
    df, info = load_dataset(force_refresh=True)
    print(f"\nDataset: {info['source']}")
    print(f"Records: {info['n_records']:,}")
    print(f"Crops:   {info['n_crops']}")
    print(f"States:  {info['n_states']}")
    if info.get("year_min"):
        print(f"Years:   {info['year_min']}–{info['year_max']}")
    print(df.head())
