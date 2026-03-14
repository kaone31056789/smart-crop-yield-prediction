"""
=============================================================================
 SMART CROP YIELD PREDICTION USING ARTIFICIAL INTELLIGENCE
 Module: recommendation_engine.py — API-Based Smart Recommendation Engine
=============================================================================

 PURPOSE:
   Provide intelligent, context-aware farming recommendations by combining
   local analysis with online knowledge queries.

   Online sources (best-effort, graceful fallback):
     • Wikipedia API  — crop/agronomic background articles
     • Open-Meteo     — weather-aware advice (via weather_api module)
     • REST Countries — regional agricultural context

   When internet is unavailable the engine falls back to a comprehensive
   built-in knowledge base so the app never shows empty results.

 AUTHOR : AgriTech AI Solutions
 VERSION: 3.0.0
=============================================================================
"""

import requests
import numpy as np
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════
#  BUILT-IN KNOWLEDGE BASE  (offline fallback)
# ═══════════════════════════════════════════════════════════════════════════

_CROP_TIPS = {
    "Rice": {
        "planting":    "Transplant 21-25 day old seedlings at 20×15 cm spacing in puddled field.",
        "irrigation":  "Maintain 5 cm standing water from transplanting to grain filling; drain 15 days before harvest.",
        "fertilizer":  "Apply 120:60:40 kg NPK/ha — split N in 3 doses (basal, tillering, panicle initiation).",
        "pest":        "Monitor for stem borer, BPH, and leaf folder. Use pheromone traps and need-based spraying.",
        "harvest":     "Harvest at 80% grain maturity (golden panicle); thresh within 24 hours.",
    },
    "Wheat": {
        "planting":    "Sow at 100 kg seed/ha with row spacing of 20-22.5 cm; optimal sowing Nov 1-25.",
        "irrigation":  "5-6 irrigations at Crown Root Initiation, Tillering, Late Jointing, Flowering, Milk, Dough stages.",
        "fertilizer":  "Apply 120:60:40 NPK/ha; half N basal, quarter at first irrigation, quarter at tillering.",
        "pest":        "Watch for aphids, termites, and rust. Seed treatment with Imidacloprid for aphids.",
        "harvest":     "Harvest at hard-dough stage when grain moisture is ~14%; avoid delays to prevent shattering.",
    },
    "Maize": {
        "planting":    "Sow at 20 kg seed/ha; spacing 60×20 cm. Treat seed with Thiram (3g/kg).",
        "irrigation":  "Critical stages: knee-high, tasseling, grain filling. Avoid waterlogging.",
        "fertilizer":  "120:60:40 NPK/ha; apply 1/3 N basal, 1/3 at knee-high, 1/3 at tasseling.",
        "pest":        "Fall armyworm is major threat — use Emamectin Benzoate 5% SG; scout weekly.",
        "harvest":     "Harvest when husks turn brown and grain moisture drops below 20%.",
    },
    "Sugarcane": {
        "planting":    "Plant 2-3 budded setts at 75 cm row spacing; treat setts with Carbendazim.",
        "irrigation":  "Frequent light irrigations during formative phase; reduce before harvest for sugar accumulation.",
        "fertilizer":  "250:60:60 NPK/ha; side-dress N at 45 and 90 days after planting.",
        "pest":        "Early shoot borer control with Chlorantraniliprole; trash mulching for moisture conservation.",
        "harvest":     "Harvest at 10-12 months; avoid burning trash — use mechanical harvester if available.",
    },
    "Cotton": {
        "planting":    "Sow Bt-cotton at 2.5 kg seed/ha; spacing 90×60 cm.",
        "irrigation":  "Drip irrigation preferred; critical at flowering and boll development.",
        "fertilizer":  "120:60:60 NPK/ha + 25 kg ZnSO₄; apply N in 3 splits.",
        "pest":        "Monitor bollworm, whitefly, pink bollworm. Refuge crop (non-Bt) mandatory in 20% area.",
        "harvest":     "Pick bolls at full opening; 3-4 pickings at 15-day intervals.",
    },
    "Soybean": {
        "planting":    "Sow at 65-75 kg/ha; row spacing 30-45 cm. Inoculate seed with Rhizobium.",
        "irrigation":  "Rain-fed mostly; supplement at flowering and pod-fill if dry spell > 15 days.",
        "fertilizer":  "20:60:40 NPK/ha (low N due to N-fixation); apply all basal.",
        "pest":        "Watch for girdle beetle, stem fly, and soybean rust. Neem-based sprays for early pest management.",
        "harvest":     "Harvest at R8 (95% pods brown); thresh at 12-13% moisture.",
    },
}

# Generic tips for crops not in the detailed DB
_GENERIC_TIPS = {
    "planting":    "Follow local extension recommendations for seed rate, spacing, and sowing time.",
    "irrigation":  "Irrigate at critical growth stages; avoid both waterlogging and prolonged drought stress.",
    "fertilizer":  "Conduct soil test and apply balanced NPK based on crop removal; split nitrogen applications.",
    "pest":        "Adopt IPM: regular scouting, biological control agents, need-based chemical application.",
    "harvest":     "Harvest at physiological maturity; dry to safe moisture level before storage.",
}


# ═══════════════════════════════════════════════════════════════════════════
#  ONLINE KNOWLEDGE FETCHERS
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_wikipedia_summary(query, sentences=3):
    """
    Fetch a short summary from Wikipedia's REST API.
    Returns str (may be empty on failure).
    """
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
        resp = requests.get(url, timeout=6, headers={"User-Agent": "AgriAI/3.0"})
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract", "")
            # Trim to requested sentence count
            parts = extract.split(". ")
            return ". ".join(parts[:sentences]) + "." if parts else ""
    except Exception:
        pass
    return ""


def _fetch_crop_wiki_info(crop):
    """Get Wikipedia background for a crop."""
    return _fetch_wikipedia_summary(f"{crop} (plant)")


def _search_web_suggestions(query):
    """
    Search DuckDuckGo Instant Answer API for quick farming advice.
    Returns list of suggestion strings (may be empty).
    """
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=6,
        )
        data = resp.json()
        results = []
        abstract = data.get("AbstractText", "")
        if abstract:
            results.append(abstract)
        for topic in data.get("RelatedTopics", [])[:3]:
            text = topic.get("Text", "")
            if text:
                results.append(text)
        return results
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class RecommendationEngine:
    """
    Smart, API-enhanced recommendation engine for precision agriculture.

    Usage
    -----
    >>> engine = RecommendationEngine()
    >>> recs = engine.get_recommendations(
    ...     crop="Rice", soil_data={...}, weather_data={...}
    ... )
    """

    def __init__(self, use_online=True):
        """
        Parameters
        ----------
        use_online : bool — attempt online API queries (True) or offline only
        """
        self.use_online = use_online
        self._cache = {}   # simple in-memory cache

    def get_recommendations(self, crop=None, soil_data=None, weather_data=None,
                            yield_prediction=None, ndvi_data=None, image_analysis=None):
        """
        Generate comprehensive, context-aware recommendations.

        All parameters are optional — the engine adapts to whatever context
        is available.

        Returns
        -------
        dict with sections: crop_management, soil_improvement,
             weather_advisory, yield_optimization, online_insights
        """
        recs = {
            "crop_management":    [],
            "soil_improvement":   [],
            "weather_advisory":   [],
            "yield_optimization": [],
            "online_insights":    [],
        }

        # ── Crop management tips (offline KB + online) ───────────
        if crop:
            tips = _CROP_TIPS.get(crop, _GENERIC_TIPS)
            for key in ("planting", "irrigation", "fertilizer", "pest", "harvest"):
                tip = tips.get(key, _GENERIC_TIPS.get(key, ""))
                if tip:
                    recs["crop_management"].append(f"**{key.title()}:** {tip}")

            # Online enrichment
            if self.use_online:
                wiki = self._cached_fetch("wiki_" + crop, _fetch_crop_wiki_info, crop)
                if wiki:
                    recs["online_insights"].append(f"📚 **About {crop}:** {wiki}")

                web = self._cached_fetch(
                    "web_" + crop,
                    _search_web_suggestions,
                    f"best farming practices for {crop} in India"
                )
                for text in web[:2]:
                    recs["online_insights"].append(f"🌐 {text}")

        # ── Soil improvement ─────────────────────────────────────
        if soil_data:
            for rec in soil_data.get("recommendations", []):
                recs["soil_improvement"].append(rec)
            deficiencies = soil_data.get("deficiencies", [])
            if deficiencies and self.use_online:
                query = f"how to fix {' and '.join(deficiencies)} deficiency in agricultural soil"
                web = self._cached_fetch("soil_" + "_".join(deficiencies),
                                          _search_web_suggestions, query)
                for text in web[:2]:
                    recs["online_insights"].append(f"🌐 {text}")

        # ── Weather advisory ─────────────────────────────────────
        if weather_data:
            temp = weather_data.get("temperature", 25)
            hum  = weather_data.get("humidity", 60)
            rain = weather_data.get("precipitation", 0)

            if temp > 40:
                recs["weather_advisory"].append("🌡️ Extreme heat expected — irrigate early morning/late evening; use mulch.")
            elif temp < 10:
                recs["weather_advisory"].append("🥶 Cold conditions — protect sensitive crops with row covers.")

            if rain > 30:
                recs["weather_advisory"].append("🌧️ Heavy rain forecast — ensure drainage; delay fertiliser application.")
            elif rain == 0 and hum < 35:
                recs["weather_advisory"].append("☀️ Dry conditions — schedule irrigation; prioritise critical-stage crops.")

            total_rain = weather_data.get("total_forecast_rain", 0)
            if total_rain > 100:
                recs["weather_advisory"].append("🌊 Significant rain in 7-day forecast — postpone field operations.")
            elif total_rain < 10:
                recs["weather_advisory"].append("🏜️ Little rain in forecast — plan irrigation schedule accordingly.")

        # ── Yield optimisation ───────────────────────────────────
        if yield_prediction is not None:
            if yield_prediction < 30:
                recs["yield_optimization"].append("📉 Predicted yield is below average — review all input parameters.")
                recs["yield_optimization"].append("Consider soil amendment, variety change, or irrigation improvement.")
            elif yield_prediction > 70:
                recs["yield_optimization"].append("📈 High yield predicted — maintain current practices and monitor for pest/disease.")
            else:
                recs["yield_optimization"].append("🔄 Moderate yield expected — incremental improvements in nutrition or irrigation could boost output.")

        # ── NDVI-driven recommendations ──────────────────────────
        if ndvi_data:
            health = ndvi_data.get("health", "")
            if health in ("Poor", "Critical"):
                recs["yield_optimization"].append("🛰️ Satellite NDVI shows stress — scout fields for localised issues.")
            elif health == "Excellent":
                recs["yield_optimization"].append("🛰️ NDVI indicates excellent vegetation — crop is on track.")

        # ── Image-analysis driven ────────────────────────────────
        if image_analysis:
            for sug in image_analysis.get("suggestions", []):
                recs["yield_optimization"].append(sug)

        # ── Seasonal general ─────────────────────────────────────
        month = datetime.now().month
        if 6 <= month <= 7:
            recs["crop_management"].append("🌧️ **Monsoon prep:** Ensure bunding, drainage channels, and seed/input availability.")
        elif month in (11, 12):
            recs["crop_management"].append("❄️ **Rabi sowing window:** Complete wheat/barley sowing by end-November for best yield.")
        elif month in (3, 4):
            recs["crop_management"].append("☀️ **Summer:** Plan Zaid crops or undertake soil preparation for Kharif.")

        return recs

    def _cached_fetch(self, key, func, *args):
        """Simple in-memory cache wrapper for API calls."""
        if key not in self._cache:
            self._cache[key] = func(*args)
        return self._cache[key]

    def get_quick_tips(self, crop):
        """Return a short list of the most important tips for a crop."""
        tips = _CROP_TIPS.get(crop, _GENERIC_TIPS)
        return [f"**{k.title()}:** {v}" for k, v in tips.items()]

    def search_online(self, query):
        """
        Direct online search — returns list of text snippets.
        For use in the UI when user requests custom queries.
        """
        if not self.use_online:
            return ["Online search disabled."]
        results = _search_web_suggestions(query)
        if not results:
            wiki = _fetch_wikipedia_summary(query.split()[0] if query else "Agriculture")
            if wiki:
                results.append(wiki)
        if not results:
            results.append("No online results found. Try a different query.")
        return results
