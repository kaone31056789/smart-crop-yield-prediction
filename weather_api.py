"""
weather_api.py  –  Fetch live weather data using the free Open-Meteo API (no key needed).
Supports area/city-level geocoding + state-level fallback.
"""

import requests

# Open-Meteo endpoints (completely free, no API key)
BASE_URL     = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL  = "https://geocoding-api.open-meteo.com/v1/search"

# State capital coordinates for weather lookup
STATE_COORDS = {
    "Andhra Pradesh":   (15.91, 79.74),
    "Assam":            (26.14, 91.74),
    "Bihar":            (25.60, 85.14),
    "Chhattisgarh":     (21.25, 81.63),
    "Gujarat":          (23.02, 72.57),
    "Haryana":          (29.06, 76.09),
    "Jharkhand":        (23.36, 85.33),
    "Karnataka":        (12.97, 77.59),
    "Kerala":           (10.85, 76.27),
    "Madhya Pradesh":   (23.26, 77.41),
    "Maharashtra":      (19.08, 72.88),
    "Odisha":           (20.30, 85.83),
    "Punjab":           (30.73, 76.78),
    "Rajasthan":        (26.92, 75.79),
    "Tamil Nadu":       (13.08, 80.27),
    "Telangana":        (17.39, 78.49),
    "Uttar Pradesh":    (26.85, 80.91),
    "West Bengal":      (22.57, 88.36),
}

# Fallback average climate (if API is unavailable)
FALLBACK_CLIMATE = {
    "Andhra Pradesh":   {"temperature": 28.0, "humidity": 65, "rainfall": 900},
    "Assam":            {"temperature": 25.0, "humidity": 80, "rainfall": 2200},
    "Bihar":            {"temperature": 26.5, "humidity": 68, "rainfall": 1100},
    "Chhattisgarh":     {"temperature": 26.5, "humidity": 60, "rainfall": 1300},
    "Gujarat":          {"temperature": 28.0, "humidity": 55, "rainfall": 800},
    "Haryana":          {"temperature": 25.5, "humidity": 48, "rainfall": 550},
    "Jharkhand":        {"temperature": 26.0, "humidity": 62, "rainfall": 1200},
    "Karnataka":        {"temperature": 26.0, "humidity": 65, "rainfall": 1200},
    "Kerala":           {"temperature": 27.0, "humidity": 78, "rainfall": 2800},
    "Madhya Pradesh":   {"temperature": 26.5, "humidity": 52, "rainfall": 1100},
    "Maharashtra":      {"temperature": 27.0, "humidity": 60, "rainfall": 1200},
    "Odisha":           {"temperature": 27.5, "humidity": 70, "rainfall": 1400},
    "Punjab":           {"temperature": 24.0, "humidity": 50, "rainfall": 650},
    "Rajasthan":        {"temperature": 27.5, "humidity": 38, "rainfall": 400},
    "Tamil Nadu":       {"temperature": 28.5, "humidity": 70, "rainfall": 950},
    "Telangana":        {"temperature": 27.5, "humidity": 58, "rainfall": 950},
    "Uttar Pradesh":    {"temperature": 26.0, "humidity": 55, "rainfall": 900},
    "West Bengal":      {"temperature": 27.0, "humidity": 72, "rainfall": 1500},
}


def fetch_weather(state: str, timeout: int = 6) -> dict:
    """
    Fetch current weather for a given Indian state using Open-Meteo API.
    Returns dict with temperature, humidity, rainfall.
    Falls back to average climate data if network fails.
    """
    coords = STATE_COORDS.get(state)
    if not coords:
        return FALLBACK_CLIMATE.get(state, {"temperature": 26, "humidity": 60, "rainfall": 900})

    lat, lon = coords
    return _fetch_weather_for_coords(lat, lon, state, timeout)


def geocode_area(area_name: str, timeout: int = 5) -> dict | None:
    """
    Geocode an area/city/village name using the free Open-Meteo Geocoding API.
    Returns {"name": ..., "lat": ..., "lon": ..., "admin1": ..., "country": ...} or None.
    """
    try:
        params = {"name": area_name, "count": 5, "language": "en", "format": "json"}
        resp = requests.get(GEOCODE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        # Prefer Indian results
        for r in results:
            if r.get("country_code", "").upper() == "IN":
                return {
                    "name": r.get("name", area_name),
                    "lat": r["latitude"],
                    "lon": r["longitude"],
                    "admin1": r.get("admin1", ""),
                    "country": r.get("country", "India"),
                }
        # Fallback to first result
        r = results[0]
        return {
            "name": r.get("name", area_name),
            "lat": r["latitude"],
            "lon": r["longitude"],
            "admin1": r.get("admin1", ""),
            "country": r.get("country", ""),
        }
    except Exception:
        return None


def fetch_weather_for_area(area_name: str, state: str | None = None, timeout: int = 6) -> dict:
    """
    Fetch weather for a specific area/city/village.
    First geocodes the area, then fetches weather for those coordinates.
    Falls back to state-level weather if geocoding fails.
    """
    # Try geocoding with state hint for better results
    search_query = f"{area_name}, {state}, India" if state else f"{area_name}, India"
    geo = geocode_area(search_query, timeout)

    # If state-qualified search failed, retry with just the area name + India
    if not geo and state:
        geo = geocode_area(f"{area_name}, India", timeout)

    if geo:
        result = _fetch_weather_for_coords(geo["lat"], geo["lon"], state, timeout)
        result["area"] = geo["name"]
        result["location"] = f"{geo['name']}, {geo.get('admin1', state or '')}"
        return result

    # Fallback to state-level
    if state:
        result = fetch_weather(state, timeout)
        result["area"] = state + " (state avg)"
        result["location"] = state
        return result

    return {"temperature": 26, "humidity": 60, "rainfall": 900,
            "wind_speed": 5.0, "weather_code": 0,
            "source": "Offline (default)", "area": area_name, "location": area_name}


def _fetch_weather_for_coords(lat: float, lon: float, state: str | None = None,
                               timeout: int = 6) -> dict:
    """Internal: fetch weather for given lat/lon coordinates."""

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,rain,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,rain_sum",
            "timezone": "Asia/Kolkata",
            "forecast_days": 7,
        }
        resp = requests.get(BASE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        daily   = data.get("daily", {})

        temperature = current.get("temperature_2m", 26)
        humidity    = current.get("relative_humidity_2m", 60)
        wind_speed  = current.get("wind_speed_10m", 5)
        weather_code = current.get("weather_code", 0)

        # Estimate annual rainfall from 7-day sum × 52
        week_rain = sum(daily.get("rain_sum", [0]) or [0])
        est_annual_rain = round(week_rain * 52, 1)
        # If estimate is unrealistically low, use fallback rain
        fallback = FALLBACK_CLIMATE.get(state, {})
        if est_annual_rain < 100:
            est_annual_rain = fallback.get("rainfall", 900)

        return {
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "rainfall": est_annual_rain,
            "wind_speed": round(wind_speed, 1),
            "weather_code": weather_code,
            "source": "Live (Open-Meteo API)",
        }

    except Exception:
        fb = FALLBACK_CLIMATE.get(state, {"temperature": 26, "humidity": 60, "rainfall": 900})
        return {**fb, "wind_speed": 5.0, "weather_code": 0, "source": "Offline (average climate)"}


def weather_code_text(code: int) -> str:
    """Convert WMO weather code to human-readable text."""
    codes = {
        0: "Clear sky ☀️", 1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅",
        3: "Overcast ☁️", 45: "Foggy 🌫️", 48: "Rime fog 🌫️",
        51: "Light drizzle 🌦️", 53: "Moderate drizzle 🌦️", 55: "Dense drizzle 🌧️",
        61: "Slight rain 🌧️", 63: "Moderate rain 🌧️", 65: "Heavy rain 🌧️",
        71: "Light snow ❄️", 73: "Moderate snow 🌨️", 75: "Heavy snow 🌨️",
        80: "Rain showers 🌦️", 81: "Moderate showers 🌧️", 82: "Violent showers ⛈️",
        95: "Thunderstorm ⛈️", 96: "Thunderstorm + hail ⛈️",
    }
    return codes.get(code, f"Code {code}")


def detect_user_location(timeout: int = 5) -> dict | None:
    """
    Auto-detect user's approximate location via free IP geolocation.
    Uses ip-api.com (no key required, free for non-commercial use).
    Returns dict with lat, lon, city, state, matched_state, country — or None.
    """
    # Union Territories & common city → nearest agricultural state mapping
    UT_STATE_MAP = {
        "chandigarh": "Punjab",
        "delhi": "Uttar Pradesh",
        "new delhi": "Uttar Pradesh",
        "national capital territory of delhi": "Uttar Pradesh",
        "puducherry": "Tamil Nadu",
        "pondicherry": "Tamil Nadu",
        "goa": "Maharashtra",
        "jammu and kashmir": "Punjab",
        "ladakh": "Punjab",
        "andaman and nicobar": "West Bengal",
        "dadra and nagar haveli": "Gujarat",
        "daman and diu": "Gujarat",
        "lakshadweep": "Kerala",
        "sikkim": "West Bengal",
        "meghalaya": "Assam",
        "mizoram": "Assam",
        "manipur": "Assam",
        "nagaland": "Assam",
        "tripura": "Assam",
        "arunachal pradesh": "Assam",
        "uttarakhand": "Uttar Pradesh",
        "himachal pradesh": "Punjab",
    }

    try:
        resp = requests.get(
            "http://ip-api.com/json/?fields=status,lat,lon,city,regionName,country",
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            return None

        region = data.get("regionName", "")
        city   = data.get("city", "")

        # Try to match to our known Indian states
        matched_state = None
        for state in STATE_COORDS:
            if state.lower() == region.lower():
                matched_state = state
                break
        if not matched_state:
            for state in STATE_COORDS:
                if state.lower() in region.lower() or region.lower() in state.lower():
                    matched_state = state
                    break
        # Fallback: UT / small-state mapping
        if not matched_state:
            matched_state = UT_STATE_MAP.get(region.lower())
        # Last resort: try city name as region hint
        if not matched_state:
            matched_state = UT_STATE_MAP.get(city.lower())

        return {
            "lat": data["lat"],
            "lon": data["lon"],
            "city": city,
            "state": region,
            "matched_state": matched_state,
            "country": data.get("country", ""),
        }
    except Exception:
        return None
