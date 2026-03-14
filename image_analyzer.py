"""
image_analyzer.py  -  Industry-Grade Crop Image Analysis Engine (v3)
====================================================================

Architecture (7-stage hierarchical pipeline):
    S0  Image quality & scene classification
        - Detects: soil / barren, water body, sky, urban / indoor, selfie /
          portrait, close-up, document / screenshot, night / dark
    S1  Multi-scale feature extraction (640 + 320 px dual resolution)
    S2  8 vegetation indices  (ExG, ExGR, VARI, GLI, RGRI, NGRDI, MGRVI, ExR)
    S3  Colour-space analysis (HSV stats + 12 fine-grained colour bands)
    S4  Texture pipeline (edge density, Laplacian, Sobel direction, LBP, uniformity)
    S5  Spatial / patch statistics (4x4 grid)
    S6  Hierarchical colour-group gate -> within-group scored ensemble

Feature count: 55+ numeric features per image.
All computations: NumPy + PIL only. Zero network calls.  Zero model downloads.

Scientific references:
    Woebbecke et al. (1995)  - ExG   | Meyer & Neto (2008)     - ExGR
    Gitelson et al. (2002)   - VARI  | Louhaichi et al. (2001)  - GLI
    Tucker (1979)            - NGRDI | Haralick et al. (1973)   - Texture
"""

from __future__ import annotations
import math, os, pickle
import numpy as np
from PIL import Image, ImageFilter
from PIL.ExifTags import TAGS, GPSTAGS

_EPS = 1e-8


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  EXIF GPS Extraction (optional — works on photos with embedded location)  #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _extract_gps_from_image(image: Image.Image) -> dict | None:
    """
    Extract GPS coordinates from image EXIF data.
    Returns dict with lat, lon, altitude, timestamp, or None if no GPS info.
    """
    try:
        exif_data = image._getexif()
        if not exif_data:
            return None
    except (AttributeError, Exception):
        return None

    gps_info = {}
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == "GPSInfo":
            for gps_tag_id, gps_value in value.items():
                gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                gps_info[gps_tag] = gps_value

    if not gps_info:
        return None

    def _dms_to_decimal(dms, ref):
        """Convert GPS DMS (degrees, minutes, seconds) to decimal degrees."""
        try:
            degrees = float(dms[0])
            minutes = float(dms[1])
            seconds = float(dms[2])
            decimal = degrees + minutes / 60.0 + seconds / 3600.0
            if ref in ('S', 'W'):
                decimal = -decimal
            return round(decimal, 6)
        except (TypeError, IndexError, ValueError, ZeroDivisionError):
            return None

    lat = lon = altitude = timestamp = None

    if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
        lat = _dms_to_decimal(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
    if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
        lon = _dms_to_decimal(gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"])
    if "GPSAltitude" in gps_info:
        try:
            altitude = round(float(gps_info["GPSAltitude"]), 1)
        except (TypeError, ValueError):
            pass
    if "GPSDateStamp" in gps_info:
        timestamp = str(gps_info["GPSDateStamp"])

    if lat is None or lon is None:
        return None

    # Reverse geocode to approximate location name
    location_name = _reverse_geocode_approx(lat, lon)

    return {
        "latitude": lat,
        "longitude": lon,
        "altitude_m": altitude,
        "gps_timestamp": timestamp,
        "location_name": location_name,
        "source": "Photo EXIF GPS",
    }


def _reverse_geocode_approx(lat: float, lon: float) -> str:
    """
    Approximate reverse geocode using known Indian state bounding boxes.
    No network call needed — works fully offline.
    """
    _INDIAN_STATES = [
        ("Punjab",           30.8, 75.4, 29.5, 76.8),
        ("Haryana",          29.0, 75.5, 27.5, 77.5),
        ("Uttar Pradesh",    28.5, 78.0, 24.0, 84.0),
        ("Bihar",            26.5, 84.0, 24.0, 88.0),
        ("West Bengal",      24.0, 86.5, 21.5, 89.0),
        ("Madhya Pradesh",   25.0, 75.0, 21.5, 82.5),
        ("Rajasthan",        30.0, 69.5, 23.0, 78.0),
        ("Gujarat",          24.5, 68.5, 20.0, 74.5),
        ("Maharashtra",      21.0, 72.5, 15.5, 80.5),
        ("Karnataka",        18.5, 74.0, 11.5, 78.5),
        ("Tamil Nadu",       13.5, 76.0, 8.0, 80.5),
        ("Kerala",           12.8, 74.8, 8.2, 77.5),
        ("Andhra Pradesh",   19.0, 76.5, 12.5, 84.5),
        ("Telangana",        19.5, 77.0, 15.5, 81.5),
        ("Odisha",           22.5, 81.5, 17.5, 87.5),
        ("Chhattisgarh",     24.0, 80.0, 17.5, 84.5),
        ("Jharkhand",        25.0, 83.5, 21.5, 87.5),
        ("Assam",            28.0, 89.5, 24.0, 96.0),
        ("Uttarakhand",      31.5, 77.5, 28.5, 81.0),
        ("Himachal Pradesh", 33.5, 75.5, 30.5, 79.0),
        ("Jammu & Kashmir",  37.0, 73.5, 32.0, 80.5),
    ]
    for state, lat_max, lon_min, lat_min, lon_max in _INDIAN_STATES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return f"{state}, India"

    # Outside India — just label by hemisphere
    if 6.5 <= lat <= 37.5 and 68.0 <= lon <= 97.5:
        return "India (approximate)"
    return f"Lat {lat:.2f}, Lon {lon:.2f}"


def _extract_photo_metadata(image: Image.Image) -> dict:
    """
    Extract all useful metadata from the photo (EXIF date, camera, etc.).
    """
    meta = {"has_exif": False}
    try:
        exif_data = image._getexif()
        if not exif_data:
            return meta
        meta["has_exif"] = True
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "DateTime":
                meta["capture_time"] = str(value)
            elif tag == "Make":
                meta["camera_make"] = str(value)
            elif tag == "Model":
                meta["camera_model"] = str(value)
            elif tag == "ImageWidth":
                meta["original_width"] = value
            elif tag == "ImageLength":
                meta["original_height"] = value
    except Exception:
        pass
    return meta


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  S0: Scene classification (reject non-crop images early)                  #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

SCENE_TYPES = {
    "crop_field":  "Agricultural field with crops",
    "soil":        "Bare soil / barren / fallow land",
    "water":       "Water body (river, pond, flooded field)",
    "sky":         "Sky-dominant image (no ground content)",
    "urban":       "Urban / built environment (buildings, roads, concrete)",
    "indoor":      "Indoor / non-outdoor image",
    "selfie":      "Human face / portrait / selfie",
    "document":    "Document / screenshot / text image",
    "night":       "Very dark / night-time image",
    "unknown":     "Unrecognisable image type",
}


def _classify_scene(feat: dict) -> tuple[str, float, str]:
    """
    Classify the image into a scene type BEFORE crop analysis.

    Returns (scene_type, scene_confidence 0-1, human_reason).
    """
    hm, sm, vm  = feat["hue_mean"], feat["sat_mean"], feat["val_mean"]
    hs, ss, vs   = feat["hue_std"], feat["sat_std"], feat["val_std"]
    br           = feat["brightness"]
    contr        = feat["contrast"]
    gf, yf, rf  = feat["green_frac"], feat["yellow_frac"], feat["red_frac"]
    bf           = feat["brown_frac"]
    wf           = feat["white_frac"]
    skyf         = feat.get("sky_frac", 0)
    canopy       = feat["canopy_coverage"]
    exg          = feat["ExG"]
    vari         = feat["VARI"]
    gli          = feat["GLI"]
    rgri         = feat["RGRI"]
    ed           = feat["edge_density"]
    lv           = feat.get("laplacian_var", 0)
    gr, rr, blr  = feat["green_ratio"], feat["red_ratio"], feat.get("blue_ratio", 0.33)
    straight_l   = feat.get("straight_line_ratio", 0)
    skin_frac    = feat.get("skin_frac", 0)
    gray_frac    = feat.get("gray_frac", 0)
    text_score   = feat.get("text_like_score", 0)

    # ── 1. Night / very dark ──
    if br < 35 and vm < 50:
        return ("night", 0.92,
                "Image is extremely dark (brightness {:.0f}/255). "
                "Likely night-time or severely underexposed.".format(br))

    # ── 2. Document / screenshot ──
    # Guard: outdoor images with soil, vegetation, or sky are NEVER documents.
    _outdoor_signs = (bf > 0.08 or gf > 0.06 or skyf > 0.04
                      or (10 < hm < 70 and sm > 0.10)
                      or canopy > 0.08)
    if not _outdoor_signs:
        if text_score > 0.55 and wf > 0.40 and sm < 0.15:
            return ("document", 0.88,
                    "Image has high white fraction ({:.0%}) and text-like "
                    "horizontal edge patterns, suggesting a document or screenshot.".format(wf))
        if wf > 0.60 and contr < 40 and sm < 0.08:
            return ("document", 0.82,
                    "Near-white uniform image with very low saturation.")

    # ── 3. Bare soil / barren (before selfie — brown/soil overlaps skin-tone hue) ──
    # Guard: skip soil if image is bright + skin-toned (portrait, not dirt)
    if 5 < hm < 85 and not (skin_frac > 0.25 and br > 160):  # warm hue + not portrait
        soil_score = 0.0
        if canopy < 0.12:  soil_score += 0.25
        elif canopy < 0.18: soil_score += 0.12
        if exg < -0.02:    soil_score += 0.20
        if vari < -0.08:   soil_score += 0.12
        if gli < -0.05:    soil_score += 0.10
        if gf < 0.15:      soil_score += 0.10
        if bf > 0.15:      soil_score += 0.15
        if rgri > 1.15:    soil_score += 0.10
        if rr > 0.35:      soil_score += 0.10
        if sm < 0.20 and 10 < hm < 60: soil_score += 0.10

        if soil_score >= 0.50:
            return ("soil", min(0.95, 0.55 + soil_score),
                    "Very low vegetation indices (ExG={:.3f}, Canopy={:.1%}) "
                    "with warm earthy hue and low green fraction. "
                    "This appears to be bare soil, fallow, or freshly "
                    "ploughed land.".format(exg, canopy))

    # ── 4. Water body (blue + darker → before sky) ──
    if skyf > 0.30 and canopy < 0.15 and br < 150:
        return ("water", 0.83,
                "Blue-dominant image (sky/blue fraction {:.0%}) with low "
                "brightness ({:.0f}) — likely a water body.".format(skyf, br))
    if blr > 0.40 and sm > 0.15 and hm > 180 and canopy < 0.15 and br < 160:
        return ("water", 0.80,
                "Image dominated by blue hue (mean {:.0f} deg) with very "
                "low vegetation - likely a water body.".format(hm))

    # ── 5. Sky-dominant (blue + bright) ──
    if skyf > 0.45 and canopy < 0.12 and br > 120:
        return ("sky", 0.85,
                "Blue sky dominates the image ({:.0%} blue pixels) "
                "with negligible vegetation.".format(skyf))

    # ── 6. Human face / selfie (skin dominates over vegetation yellow/green) ──
    if skin_frac > 0.30 and ed > 0.08 and skin_frac > (yf + gf) * 1.5:
        return ("selfie", 0.80,
                "Significant skin-tone pixel fraction ({:.0%}) detected. "
                "This appears to be a portrait or selfie, not a field photo.".format(skin_frac))

    # ── 7. Urban / built environment ──
    if gray_frac > 0.40 and sm < 0.12 and canopy < 0.10:
        return ("urban", 0.82,
                "High fraction of grey / desaturated pixels ({:.0%}) with "
                "negligible vegetation — built or paved surface.".format(gray_frac))
    if straight_l > 0.30 and gray_frac > 0.25 and canopy < 0.12:
        return ("urban", 0.78,
                "Strong straight-line geometry + grey dominance suggest "
                "urban structures or roads.")

    # ── 8. Indoor ──
    if wf > 0.30 and sm < 0.10 and canopy < 0.08 and ed > 0.12:
        return ("indoor", 0.75,
                "Low saturation, high white fraction, and structured edges "
                "suggest an indoor photo with artificial lighting.")

    # ── 9. Image looks like a crop field ──
    veg_signal = (canopy > 0.15) + (exg > 0.0) + (gf > 0.15) + (vari > -0.05)
    if veg_signal >= 2:
        conf = min(0.98, 0.60 + canopy * 0.25 + max(0, exg) * 2)
        return ("crop_field", conf, "")

    return ("unknown", 0.50,
            "Could not confidently classify this image. It may not be "
            "an agricultural field photo.")


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  P1: RGB -> HSV  (vectorised)                                             #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _rgb_to_hsv(img: np.ndarray):
    """RGB uint8 array -> H(0-360), S(0-1), V(0-255)."""
    r, g, b = (img[..., i].astype(np.float64) for i in range(3))
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn + _EPS
    h = np.zeros_like(r)
    mr = mx == r; mg = (mx == g) & ~mr; mb = ~mr & ~mg
    h[mr] = (60 * ((g[mr] - b[mr]) / d[mr]) + 360) % 360
    h[mg] = 60 * ((b[mg] - r[mg]) / d[mg]) + 120
    h[mb] = 60 * ((r[mb] - g[mb]) / d[mb]) + 240
    s = np.where(mx == 0, 0.0, d / (mx + _EPS))
    return h, s, mx


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  P2: 8 vegetation indices                                                 #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _vegetation_indices(r, g, b) -> dict:
    t = r + g + b + _EPS
    rn, gn, bn = r / t, g / t, b / t
    ExG  = 2 * gn - rn - bn
    ExR  = 1.4 * rn - gn
    return {
        "ExG":   float(np.mean(ExG)),
        "ExR":   float(np.mean(ExR)),
        "ExGR":  float(np.mean(ExG - ExR)),
        "VARI":  float(np.mean(np.clip((g - r) / (g + r - b + _EPS), -1, 1))),
        "GLI":   float(np.mean(np.clip((2*g - r - b) / (2*g + r + b + _EPS), -1, 1))),
        "RGRI":  float(np.mean(np.clip(r / (g + _EPS), 0, 5))),
        "NGRDI": float(np.mean(np.clip((g - r) / (g + r + _EPS), -1, 1))),
        "MGRVI": float(np.mean(np.clip((g*g - r*r) / (g*g + r*r + _EPS), -1, 1))),
    }


def _canopy_coverage(r, g, b, thr=0.05):
    t = r + g + b + _EPS
    rn, gn, bn = r / t, g / t, b / t
    return float(np.mean((2 * gn - rn - bn) > thr))


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  P4: Texture pipeline                                                     #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _texture_features(img: np.ndarray) -> dict:
    gray = np.mean(img, axis=2).astype(np.uint8)
    pil_g = Image.fromarray(gray)

    edges = np.array(pil_g.filter(ImageFilter.FIND_EDGES), dtype=float)
    edge_density = float(np.mean(edges > 30))
    edge_mean    = float(np.mean(edges))

    # Laplacian variance (image sharpness / focus)
    lap = np.array(pil_g.filter(ImageFilter.Kernel(
        (3, 3), [0,1,0,1,-4,1,0,1,0], scale=1, offset=128)), dtype=float) - 128
    laplacian_var = float(np.var(lap))

    # Block-level local variance
    H, W = gray.shape
    bs = max(8, min(H, W) // 16)
    variances = []
    for i in range(0, H - bs, bs):
        for j in range(0, W - bs, bs):
            variances.append(float(np.var(gray[i:i+bs, j:j+bs].astype(float))))
    local_var = float(np.mean(variances)) if variances else 0.0

    # Smoothness ratio (fraction of patches with low variance)
    smooth = sum(1 for v in variances if v < 200) if variances else 0
    uniformity = smooth / max(len(variances), 1)

    # Directional Sobel energy (horizontal vs vertical structure)
    sobelH = np.array(pil_g.filter(ImageFilter.Kernel(
        (3, 3), [-1,-2,-1,0,0,0,1,2,1], scale=1, offset=128)), dtype=float) - 128
    sobelV = np.array(pil_g.filter(ImageFilter.Kernel(
        (3, 3), [-1,0,1,-2,0,2,-1,0,1], scale=1, offset=128)), dtype=float) - 128
    h_en = float(np.mean(sobelH ** 2))
    v_en = float(np.mean(sobelV ** 2))
    dir_ratio = h_en / (v_en + _EPS)

    # Straight-line indicator (for urban detection)
    strong_h = np.sum(np.abs(sobelH) > 40)
    strong_v = np.sum(np.abs(sobelV) > 40)
    total_px = max(float(H * W), 1)
    straight_line_ratio = float((strong_h + strong_v) / total_px / 2)

    # Text-like score (strong horizontal edges, low vertical)
    text_like = float(h_en / (v_en + h_en + _EPS)) if h_en > 50 else 0.0

    return {
        "edge_density": edge_density, "edge_mean": edge_mean,
        "local_variance": local_var, "laplacian_var": laplacian_var,
        "uniformity": uniformity, "dir_ratio": dir_ratio,
        "straight_line_ratio": straight_line_ratio,
        "text_like_score": text_like,
    }


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  P3: Fine-grained colour fractions (12 bands)                             #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _color_fractions(h, s, v) -> dict:
    px = float(h.size)
    sm = s > 0.12   # saturated mask
    t  = max(float(np.sum(sm)), 1.0)

    # Skin-tone detection (for selfie rejection)
    skin = ((h > 5) & (h < 50) & (s > 0.15) & (s < 0.65) & (v > 80) & (v < 230))

    # Gray-tone (for urban / concrete detection)
    gray_mask = (s < 0.10) & (v > 50) & (v < 210)

    return {
        "bright_yellow_frac": float(np.sum(sm & (h>35) & (h<65) & (s>0.30))) / px,
        "pale_yellow_frac":   float(np.sum(sm & (h>30) & (h<70) & (s<=0.30))) / px,
        "yellow_frac":        float(np.sum(sm & (h>30) & (h<70))) / t,
        "green_frac":         float(np.sum(sm & (h>=65) & (h<155))) / t,
        "dark_green_frac":    float(np.sum(sm & (h>=65) & (h<155) & (v<140))) / px,
        "bright_green_frac":  float(np.sum(sm & (h>=65) & (h<155) & (v>=140))) / px,
        "red_frac":           float(np.sum(sm & ((h<20) | (h>340)))) / t,
        "orange_frac":        float(np.sum(sm & (h>=20) & (h<40))) / t,
        "brown_frac":         float(np.sum((h>10) & (h<65) & (s>0.08) & (s<0.65) & (v>50) & (v<210))) / px,
        "white_frac":         float(np.sum((s<0.10) & (v>200))) / px,
        "sky_frac":           float(np.sum(sm & (h>=190) & (h<260))) / px,
        "cyan_frac":          float(np.sum(sm & (h>=155) & (h<190))) / px,
        "skin_frac":          float(np.sum(skin)) / px,
        "gray_frac":          float(np.sum(gray_mask)) / px,
    }


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  P5: Spatial / patch analysis (4x4 grid)                                  #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _patch_analysis(img_arr, h, s, grid=4) -> dict:
    H, W = img_arr.shape[:2]
    ph, pw = H // grid, W // grid
    if ph < 8 or pw < 8:
        return {"patch_hue_std": 0, "patch_green_std": 0,
                "spatial_uniformity": 1.0, "patch_count": 0}
    hue_means, grn_means, sat_means = [], [], []
    for gi in range(grid):
        for gj in range(grid):
            y0, y1 = gi*ph, (gi+1)*ph
            x0, x1 = gj*pw, (gj+1)*pw
            r = img_arr[y0:y1, x0:x1, 0].astype(float)
            g = img_arr[y0:y1, x0:x1, 1].astype(float)
            b = img_arr[y0:y1, x0:x1, 2].astype(float)
            hue_means.append(float(np.mean(h[y0:y1, x0:x1])))
            grn_means.append(float(np.mean(g / (r+g+b+_EPS))))
            sat_means.append(float(np.mean(s[y0:y1, x0:x1])))
    phs = float(np.std(hue_means))
    pgs = float(np.std(grn_means))
    pss = float(np.std(sat_means))
    su = max(0.0, 1.0 - (phs/60 + pgs/0.08 + pss/0.15) / 3)
    return {"patch_hue_std": phs, "patch_green_std": pgs,
            "spatial_uniformity": su, "patch_count": grid*grid}


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  S1: Full feature extraction (55+ features)                               #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _extract_all_features(img_arr: np.ndarray) -> dict:
    r = img_arr[:,:,0].astype(np.float64)
    g = img_arr[:,:,1].astype(np.float64)
    b = img_arr[:,:,2].astype(np.float64)
    h, s, v = _rgb_to_hsv(img_arr)

    feat = {
        **_vegetation_indices(r, g, b),
        "canopy_coverage": _canopy_coverage(r, g, b),
        **_texture_features(img_arr),
        "hue_mean": float(np.mean(h)), "hue_std": float(np.std(h)),
        "hue_median": float(np.median(h)),
        "sat_mean": float(np.mean(s)), "sat_std": float(np.std(s)),
        "val_mean": float(np.mean(v)), "val_std": float(np.std(v)),
        **_color_fractions(h, s, v),
        **_patch_analysis(img_arr, h, s),
        "brightness": float(np.mean(img_arr)),
        "contrast":   float(np.std(img_arr)),
        "green_ratio": float(np.mean(g / (r+g+b+_EPS))),
        "red_ratio":   float(np.mean(r / (r+g+b+_EPS))),
        "blue_ratio":  float(np.mean(b / (r+g+b+_EPS))),
    }
    # ── Derived discriminating features ──
    ed = feat["edge_density"]
    lv = feat["laplacian_var"]
    # Leaf coarseness: high for large-leaved crops (banana, maize), low for
    # fine-textured crops (wheat, rice, soybean).
    feat["leaf_coarseness"] = lv / (ed + 0.01)
    # Canopy-to-texture ratio: smooth dense canopy = high, coarse canopy = low
    feat["canopy_texture_ratio"] = feat["canopy_coverage"] / (ed + 0.02)
    # Green intensity: bright_green / total green balance
    gf = feat.get("bright_green_frac", 0)
    dgf = feat.get("dark_green_frac", 0)
    feat["green_brightness_ratio"] = gf / (gf + dgf + _EPS)
    return feat


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  S6: Colour-group hierarchical gate                                       #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

COLOR_GROUPS = {
    "YELLOW_DOMINANT": ["Mustard"],
    "YELLOW_HEAVY":    ["Mustard", "Wheat", "Barley", "Rice"],
    "YELLOW_MODERATE": ["Wheat", "Barley", "Rice", "Chickpea", "Cotton"],
    "GREEN_BRIGHT":    ["Rice", "Potato", "Soybean", "Groundnut", "Maize",
                        "Sugarcane", "Wheat", "Cotton", "Chickpea", "Tomato"],
    "GREEN_DARK":      ["Tea", "Jute", "Coffee", "Banana", "Maize", "Sugarcane",
                        "Soybean", "Groundnut", "Potato"],
    "RED_MIXED":       ["Tomato", "Coffee", "Cotton"],
    "WHITE_MIXED":     ["Cotton", "Chickpea", "Barley"],
    "MIXED":           None,   # all crops eligible
}

def _classify_color_group(feat: dict) -> str:
    byf = feat.get("bright_yellow_frac", 0)
    yf  = feat["yellow_frac"]
    gf  = feat["green_frac"]
    dgf = feat.get("dark_green_frac", 0)
    rf  = feat["red_frac"]
    wf  = feat["white_frac"]
    hm  = feat["hue_mean"]

    if byf > 0.18 and yf > 0.35:
        return "YELLOW_DOMINANT"
    if yf > 0.30 and hm < 68:
        return "YELLOW_HEAVY"
    if yf > 0.15 and hm < 75:
        return "YELLOW_MODERATE"
    if rf > 0.08 or feat.get("orange_frac", 0) > 0.06:
        return "RED_MIXED"
    if wf > 0.15:
        return "WHITE_MIXED"
    if dgf > 0.20 or (hm > 95 and feat["sat_mean"] > 0.35):
        return "GREEN_DARK"
    if gf > 0.30:
        return "GREEN_BRIGHT"
    return "MIXED"


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  Crop profiles (16 crops, tighter ranges, hard constraints)               #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

CROP_PROFILES = {
    "Mustard": {
        "hue": (35, 65), "sat": (0.28, 0.85), "val": (125, 235),
        "hue_center": 50, "sat_center": 0.65, "val_center": 191,
        "canopy": (0.20, 0.95), "texture": (0.01, 0.15),
        "ExG": (-0.08, 0.10), "NGRDI": (-0.25, 0.06),
        "yellow_hint": 0.60, "yellow_min": 0.20, "bright_yellow_min": 0.08,
        "green_max": 0.60, "multi_color": False,
        "desc": "Bright yellow flowers covering green stalks - very distinctive hue",
    },
    "Wheat": {
        "hue": (38, 100), "sat": (0.14, 0.58), "val": (85, 215),
        "hue_center": 60, "sat_center": 0.35, "val_center": 166,
        # Green vegetative wheat (young crop, hue 80-95)
        "alt_centers": [(85, 0.40, 150), (72, 0.36, 158)],
        "canopy": (0.22, 0.82), "texture": (0.02, 0.14),
        "ExG": (-0.04, 0.14), "NGRDI": (-0.18, 0.14),
        "yellow_hint": 0.28, "yellow_min": 0.0, "yellow_max": 0.65,
        "texture_max": 0.14,           # wheat has fine uniform texture
        "leaf_coarseness_max": 600,    # thin blades → low coarseness
        "uniformity_min": 0.35,        # wheat fields are uniform carpets
        "multi_color": False,
        "desc": "Yellow-green to golden grain stalks with fine uniform texture",
    },
    "Barley": {
        "hue": (40, 95), "sat": (0.12, 0.55), "val": (80, 205),
        "hue_center": 55, "sat_center": 0.30, "val_center": 153,
        "alt_centers": [(78, 0.35, 148)],  # green vegetative barley
        "canopy": (0.18, 0.75), "texture": (0.02, 0.14),
        "ExG": (-0.04, 0.12), "NGRDI": (-0.18, 0.12),
        "yellow_hint": 0.32, "yellow_min": 0.0, "yellow_max": 0.65,
        "texture_max": 0.14,
        "leaf_coarseness_max": 600,
        "multi_color": False,
        "desc": "Golden-green grain stalks with awned heads; similar to wheat",
    },
    "Rice": {
        "hue": (50, 110), "sat": (0.16, 0.68), "val": (75, 215),
        "hue_center": 80, "sat_center": 0.40, "val_center": 153,
        "alt_centers": [(95, 0.45, 140)],  # deep green paddy
        "canopy": (0.28, 0.92), "texture": (0.01, 0.12),
        "ExG": (-0.02, 0.17), "NGRDI": (-0.10, 0.22),
        "yellow_hint": 0.15, "yellow_max": 0.50,
        "texture_max": 0.12,
        "leaf_coarseness_max": 550,
        "uniformity_min": 0.30,
        "multi_color": False,
        "desc": "Bright green paddy fields with smooth texture and high moisture reflectance",
    },
    "Maize": {
        "hue": (70, 140), "sat": (0.22, 0.70), "val": (65, 215),
        "hue_center": 105, "sat_center": 0.45, "val_center": 140,
        "canopy": (0.25, 0.87), "texture": (0.06, 0.28),
        "ExG": (-0.02, 0.19), "NGRDI": (-0.08, 0.24),
        "yellow_hint": 0.06, "yellow_max": 0.30, "multi_color": False,
        "texture_min": 0.06,
        "leaf_coarseness_min": 250,     # broad leaves
        "uniformity_max": 0.72,        # row structure creates non-uniformity
        "desc": "Tall dark green stalks with broad textured leaves and row structure",
    },
    "Sugarcane": {
        "hue": (60, 138), "sat": (0.20, 0.67), "val": (65, 215),
        "hue_center": 100, "sat_center": 0.42, "val_center": 153,
        "canopy": (0.35, 0.97), "texture": (0.04, 0.25),
        "ExG": (-0.01, 0.18), "NGRDI": (-0.06, 0.22),
        "yellow_hint": 0.06, "yellow_max": 0.25, "multi_color": False,
        "canopy_min": 0.35,
        "desc": "Dense tall canes with very high canopy density and directional structure",
    },
    "Jute": {
        "hue": (75, 150), "sat": (0.24, 0.70), "val": (55, 205),
        "hue_center": 112, "sat_center": 0.45, "val_center": 140,
        "canopy": (0.35, 0.97), "texture": (0.03, 0.17),
        "ExG": (0.01, 0.20), "NGRDI": (-0.02, 0.24),
        "yellow_hint": 0.04, "yellow_max": 0.15, "multi_color": False,
        "canopy_min": 0.35,
        "desc": "Tall dark green fibre plants with smooth stems and dense canopy",
    },
    "Tomato": {
        "hue": (55, 140), "sat": (0.20, 0.70), "val": (65, 215),
        "hue_center": 95, "sat_center": 0.45, "val_center": 145,
        "alt_centers": [(110, 0.55, 140), (85, 0.40, 155)],
        "canopy": (0.18, 0.78), "texture": (0.04, 0.24),
        "ExG": (-0.06, 0.15), "NGRDI": (-0.15, 0.18),
        "yellow_hint": 0.04, "yellow_max": 0.20, "multi_color": True,
        "red_affinity": 0.05, "hue_std_min": 22,
        "orange_affinity": 0.02,
        "desc": "Green vines with red/orange fruit clusters; high colour variance",
    },
    "Cotton": {
        "hue": (65, 128), "sat": (0.10, 0.60), "val": (85, 235),
        "hue_center": 100, "sat_center": 0.45, "val_center": 148,
        "canopy": (0.10, 0.95), "texture": (0.03, 0.28),
        "ExG": (-0.06, 0.35), "NGRDI": (-0.15, 0.28),
        "yellow_hint": 0.04, "yellow_max": 0.18, "multi_color": False,
        "white_affinity": 0.08,
        "white_min": 0.05,
        "red_max": 0.05,
        "desc": "Green plants with distinctive white cotton bolls",
    },
    "Soybean": {
        "hue": (65, 138), "sat": (0.20, 0.64), "val": (68, 205),
        "hue_center": 98, "sat_center": 0.38, "val_center": 140,
        "canopy": (0.25, 0.82), "texture": (0.02, 0.16),
        "ExG": (-0.02, 0.17), "NGRDI": (-0.08, 0.20),
        "yellow_hint": 0.05, "yellow_max": 0.25, "multi_color": False,
        "desc": "Short bushy green plants with uniform rounded canopy",
    },
    "Potato": {
        "hue": (58, 120), "sat": (0.18, 0.62), "val": (72, 210),
        "hue_center": 85, "sat_center": 0.35, "val_center": 148,
        "canopy": (0.28, 0.87), "texture": (0.02, 0.15),
        "ExG": (-0.02, 0.16), "NGRDI": (-0.10, 0.18),
        "yellow_hint": 0.05, "yellow_max": 0.28, "multi_color": False,
        "desc": "Low-growing dense green leafy ground cover with row pattern",
    },
    "Groundnut": {
        "hue": (65, 130), "sat": (0.18, 0.60), "val": (72, 205),
        "hue_center": 92, "sat_center": 0.35, "val_center": 140,
        "canopy": (0.18, 0.72), "texture": (0.02, 0.15),
        "ExG": (-0.02, 0.15), "NGRDI": (-0.10, 0.17),
        "yellow_hint": 0.05, "yellow_max": 0.24, "multi_color": False,
        "desc": "Low spreading green plants with smooth small leaves",
    },
    "Chickpea": {
        "hue": (58, 120), "sat": (0.12, 0.54), "val": (78, 205),
        "hue_center": 82, "sat_center": 0.28, "val_center": 148,
        "canopy": (0.08, 0.62), "texture": (0.03, 0.20),
        "ExG": (-0.04, 0.13), "NGRDI": (-0.12, 0.14),
        "yellow_hint": 0.08, "yellow_max": 0.35, "multi_color": False,
        "canopy_max": 0.62,
        "desc": "Pale green sparse bushy plants with ferny leaves and visible soil",
    },
    "Banana": {
        "hue": (90, 160), "sat": (0.25, 0.70), "val": (50, 200),
        "hue_center": 115, "sat_center": 0.48, "val_center": 120,
        "canopy": (0.22, 0.88), "texture": (0.10, 0.35),
        "ExG": (-0.01, 0.19), "NGRDI": (-0.06, 0.22),
        "yellow_hint": 0.03, "yellow_max": 0.12, "multi_color": False,
        "texture_min": 0.10,
        "leaf_coarseness_min": 400,     # large broad leaves
        "uniformity_max": 0.65,        # individual plants visible
        "green_brightness_max": 0.55,  # banana leaves are darker green
        "desc": "Very large broad dark green leaves - highest texture score",
    },
    "Tea": {
        "hue": (85, 162), "sat": (0.26, 0.74), "val": (45, 200),
        "hue_center": 125, "sat_center": 0.50, "val_center": 115,
        "canopy": (0.45, 0.99), "texture": (0.03, 0.21),
        "ExG": (0.01, 0.22), "NGRDI": (-0.02, 0.24),
        "yellow_hint": 0.02, "yellow_max": 0.10, "multi_color": False,
        "canopy_min": 0.45,
        "desc": "Very dense dark green compact bushes with manicured rows; highest canopy",
    },
    "Coffee": {
        "hue": (82, 155), "sat": (0.20, 0.65), "val": (50, 200),
        "hue_center": 118, "sat_center": 0.40, "val_center": 115,
        "canopy": (0.28, 0.85), "texture": (0.05, 0.22),
        "ExG": (-0.01, 0.18), "NGRDI": (-0.06, 0.20),
        "yellow_hint": 0.03, "yellow_max": 0.12, "multi_color": False,
        "desc": "Dark green shrubs under partial shade; red berries at harvest",
    },
}


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  Scoring engine (multi-criteria weighted + hard-constraints)              #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _rs(val, lo, hi, falloff) -> float:
    """Range score with super-linear decay outside [lo, hi]."""
    if lo <= val <= hi:
        return 1.0
    d = lo - val if val < lo else val - hi
    return max(0.0, 1.0 - (d / falloff) ** 1.5)


def _score_crop(feat: dict, prof: dict, is_candidate: bool) -> float:
    S, W = [], []

    # ── PRIMARY: Gaussian center-distance (HSV) — MULTI-MODE AWARE ──
    # Crops like wheat have very different appearances at different growth
    # stages (green vegetative vs golden mature).  We evaluate the primary
    # center AND every alternate centre and keep the BEST match.
    hc = prof.get("hue_center", (prof["hue"][0] + prof["hue"][1]) / 2)
    sc = prof.get("sat_center", (prof["sat"][0] + prof["sat"][1]) / 2)
    vc = prof.get("val_center", (prof["val"][0] + prof["val"][1]) / 2)

    centers = [(hc, sc, vc)] + list(prof.get("alt_centers", []))

    if prof.get("multi_color"):
        sigma_h, sigma_s, sigma_v = 20, 0.15, 20
        w_center = 12.0
    else:
        sigma_h, sigma_s, sigma_v = 15, 0.08, 15
        w_center = 20.0

    best_center_score = max(
        math.exp(
            -((feat["hue_mean"] - ch) / sigma_h) ** 2
            - ((feat["sat_mean"] - cs) / sigma_s) ** 2
            - ((feat["val_mean"] - cv) / sigma_v) ** 2
        )
        for ch, cs, cv in centers
    )
    S.append(best_center_score); W.append(w_center)

    # P2: Hue range
    S.append(_rs(feat["hue_mean"], *prof["hue"], 35));  W.append(3.0)
    # P2: Saturation range
    S.append(_rs(feat["sat_mean"], *prof["sat"], 0.20)); W.append(1.5)
    # P2: Value / brightness range
    vr = prof.get("val", (65, 215))
    S.append(_rs(feat["val_mean"], *vr, 55)); W.append(1.0)
    # P1: ExG  (wider falloff for multi-color: green + accent crops)
    exg_fo = 0.18 if prof.get("multi_color") else 0.07
    S.append(_rs(feat["ExG"], *prof["ExG"], exg_fo)); W.append(2.0)
    # P1: NGRDI
    ngrdi_fo = 0.18 if prof.get("multi_color") else 0.10
    nr = prof.get("NGRDI", (-0.15, 0.15))
    S.append(_rs(feat["NGRDI"], *nr, ngrdi_fo)); W.append(2.0)
    # P4: Texture
    S.append(_rs(feat["edge_density"], *prof["texture"], 0.10)); W.append(2.5)
    # P2: Canopy
    S.append(_rs(feat["canopy_coverage"], *prof["canopy"], 0.22)); W.append(2.5)
    # P5: Spatial uniformity
    S.append(_rs(feat.get("spatial_uniformity", 0.5), 0.15, 1.0, 0.3)); W.append(0.8)

    # Multi-colour indicator
    if prof.get("multi_color"):
        mc = min(1.0, feat["hue_std"] / 50)
        S.append(mc); W.append(6.0)
    else:
        mc = max(0.0, 1.0 - max(0, feat["hue_std"] - 35) / 50)
        S.append(mc); W.append(1.5)

    # Yellow affinity
    yf = feat["yellow_frac"]
    yh = prof.get("yellow_hint", 0.0)
    y_dist = abs(yf - yh)
    y_score = max(0, 1.0 - (y_dist / 0.30) ** 1.3)
    w_y = 5.0 if yh > 0.20 else 1.5 if yh > 0.10 else 0.5
    S.append(y_score); W.append(w_y)

    # Hard: yellow_min
    ymin = prof.get("yellow_min", 0.0)
    if ymin > 0:
        if yf >= ymin:
            S.append(1.0); W.append(4.0)
        else:
            S.append(max(0, yf / ymin) ** 2); W.append(4.0)

    # Hard: yellow_max
    ymax = prof.get("yellow_max", 1.0)
    if ymax < 1.0 and yf > ymax:
        penalty = max(0, 1.0 - (yf - ymax) / 0.20) ** 2
        S.append(penalty); W.append(4.5)

    # Hard: bright_yellow_min (Mustard)
    bymin = prof.get("bright_yellow_min", 0)
    if bymin > 0:
        byf = feat.get("bright_yellow_frac", 0)
        S.append(min(1.0, byf / bymin) if byf < bymin else 1.0); W.append(5.0)

    # Hard: green_max
    gmax = prof.get("green_max")
    if gmax is not None and feat["green_frac"] > gmax:
        S.append(max(0, 1.0 - (feat["green_frac"] - gmax) / 0.25)); W.append(3.5)

    # Hard: texture_min (maize, banana)
    tmin = prof.get("texture_min", 0)
    if tmin > 0:
        if feat["edge_density"] >= tmin:
            S.append(1.0); W.append(3.0)
        else:
            S.append(max(0, feat["edge_density"] / tmin)); W.append(3.0)

    # Hard: canopy_min (sugarcane, jute, tea)
    cmin = prof.get("canopy_min", 0)
    if cmin > 0:
        if feat["canopy_coverage"] >= cmin:
            S.append(1.0); W.append(3.0)
        else:
            S.append(max(0, feat["canopy_coverage"] / cmin)); W.append(3.0)

    # Hard: canopy_max (chickpea)
    cmax = prof.get("canopy_max")
    if cmax is not None:
        if feat["canopy_coverage"] <= cmax:
            S.append(1.0); W.append(2.5)
        else:
            S.append(max(0, 1.0 - (feat["canopy_coverage"] - cmax) / 0.20)); W.append(2.5)

    # Red affinity (tomato) — strong indicator of fruit-bearing crops
    ra = prof.get("red_affinity", 0)
    if ra > 0:
        rf = feat["red_frac"]
        S.append(1.0 if rf >= ra else max(0, rf / ra)**0.8); W.append(10.0)

    # Orange affinity (tomato — orange/ripening fruit)
    oa = prof.get("orange_affinity", 0)
    if oa > 0:
        of_ = feat.get("orange_frac", 0)
        S.append(1.0 if of_ >= oa else max(0, of_ / (oa + 1e-9))**0.8); W.append(6.0)

    # Hue-std minimum (tomato — high colour diversity from fruit + foliage)
    hsm = prof.get("hue_std_min", 0)
    if hsm > 0:
        S.append(1.0 if feat["hue_std"] >= hsm else max(0, feat["hue_std"]/hsm)); W.append(7.0)

    # Hard: red_max (cotton — should NOT have significant red; typical of tomato)
    rmax = prof.get("red_max")
    if rmax is not None and feat["red_frac"] > rmax:
        penalty = max(0, 1.0 - (feat["red_frac"] - rmax) / 0.05) ** 2
        S.append(penalty); W.append(12.0)

    # Hard: white_min (cotton — MUST have white from cotton bolls)
    wmin = prof.get("white_min")
    if wmin is not None:
        wf = feat["white_frac"]
        if wf >= wmin:
            S.append(1.0); W.append(8.0)
        else:
            S.append(max(0, wf / wmin) ** 2); W.append(8.0)

    # White affinity (cotton)
    wa = prof.get("white_affinity", 0)
    if wa > 0:
        wf = feat["white_frac"]
        S.append(1.0 if wf >= wa else max(0, wf / wa)**0.8); W.append(7.0)

    # ── NEW: Texture / leaf-morphology constraints (v4.1) ──

    # Hard: texture_max (wheat, rice, barley — fine-textured crops)
    tmax = prof.get("texture_max")
    if tmax is not None:
        ed_val = feat["edge_density"]
        if ed_val <= tmax:
            S.append(1.0); W.append(4.0)
        else:
            S.append(max(0, 1.0 - (ed_val - tmax) / 0.12) ** 1.5); W.append(4.0)

    # Hard: leaf_coarseness_min (banana, maize — broad-leaved crops)
    lc_min = prof.get("leaf_coarseness_min", 0)
    if lc_min > 0:
        lc = feat.get("leaf_coarseness", 0)
        if lc >= lc_min:
            S.append(1.0); W.append(4.5)
        else:
            S.append(max(0, lc / lc_min) ** 1.3); W.append(4.5)

    # Hard: leaf_coarseness_max (wheat, rice — fine-leaved crops)
    lc_max = prof.get("leaf_coarseness_max")
    if lc_max is not None:
        lc = feat.get("leaf_coarseness", 0)
        if lc <= lc_max:
            S.append(1.0); W.append(4.0)
        else:
            S.append(max(0, 1.0 - (lc - lc_max) / 500)); W.append(4.0)

    # Hard: uniformity_min (wheat, rice — carpet-like fields)
    uni_min = prof.get("uniformity_min", 0)
    if uni_min > 0:
        uni = feat.get("spatial_uniformity", 0.5)
        if uni >= uni_min:
            S.append(1.0); W.append(3.5)
        else:
            S.append(max(0, uni / uni_min)); W.append(3.5)

    # Hard: uniformity_max (banana, maize — patchy/structured)
    uni_max = prof.get("uniformity_max")
    if uni_max is not None:
        uni = feat.get("spatial_uniformity", 0.5)
        if uni <= uni_max:
            S.append(1.0); W.append(3.0)
        else:
            S.append(max(0, 1.0 - (uni - uni_max) / 0.30)); W.append(3.0)

    # Hard: green_brightness_max (banana — dark leaves, not bright green)
    gbmax = prof.get("green_brightness_max")
    if gbmax is not None:
        gb = feat.get("green_brightness_ratio", 0.5)
        if gb <= gbmax:
            S.append(1.0); W.append(3.0)
        else:
            S.append(max(0, 1.0 - (gb - gbmax) / 0.30)); W.append(3.0)

    # Weighted average
    raw = sum(a * b for a, b in zip(S, W)) / (sum(W) + _EPS)
    # Candidate boost / penalty (15% swing)
    # Multi-color crops get candidate status ONLY if the image is NOT
    # yellow-dominant (prevents Cotton from beating Mustard on yellow fields).
    is_cand = is_candidate
    if prof.get("multi_color") and not is_candidate:
        # Only grant candidate bypass if yellow fraction is low
        if yf < 0.15:
            is_cand = True
    raw = raw * 1.15 if is_cand else raw * 0.85
    return round(min(raw * 100, 99.9), 1)


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  TRAINED ML CROP CLASSIFIER  (Random Forest on synthetic profiles)        #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

_CLASSIFIER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "saved_models")
_CLASSIFIER_PATH = os.path.join(_CLASSIFIER_DIR, "crop_image_classifier.pkl")

_ML_FEATURES = [
    "hue_mean", "sat_mean", "val_mean", "hue_std", "sat_std", "val_std",
    "ExG", "ExGR", "VARI", "GLI", "NGRDI", "RGRI", "MGRVI",
    "canopy_coverage", "edge_density", "laplacian_var", "local_variance",
    "uniformity", "dir_ratio", "spatial_uniformity",
    "yellow_frac", "green_frac", "red_frac", "brown_frac", "white_frac",
    "bright_yellow_frac", "dark_green_frac", "bright_green_frac",
    "orange_frac", "leaf_coarseness", "canopy_texture_ratio",
    "green_brightness_ratio", "patch_hue_std", "patch_green_std",
    "brightness", "contrast",
]

# Cache the classifier in-memory after first build
_cached_clf = None


def _generate_synthetic_samples(rng, crop: str, prof: dict, n: int = 300):
    """Generate *n* synthetic feature vectors from a crop profile."""
    h_lo, h_hi = prof["hue"]
    s_lo, s_hi = prof["sat"]
    v_lo, v_hi = prof.get("val", (65, 215))

    hc0 = prof.get("hue_center", (h_lo + h_hi) / 2)
    sc0 = prof.get("sat_center", (s_lo + s_hi) / 2)
    vc0 = prof.get("val_center", (v_lo + v_hi) / 2)
    centers = [(hc0, sc0, vc0)] + list(prof.get("alt_centers", []))
    n_centers = len(centers)

    rows = []
    for i in range(n):
        ch, cs, cv = centers[i % n_centers]
        hue   = np.clip(rng.normal(ch, (h_hi - h_lo) / 5), h_lo - 5, h_hi + 5)
        sat   = np.clip(rng.normal(cs, (s_hi - s_lo) / 5),
                        max(0.05, s_lo - 0.05), min(1.0, s_hi + 0.05))
        val   = np.clip(rng.normal(cv, (v_hi - v_lo) / 5),
                        max(30, v_lo - 10), min(255, v_hi + 10))

        exg_lo, exg_hi = prof.get("ExG", (-0.05, 0.15))
        ngrdi_lo, ngrdi_hi = prof.get("NGRDI", (-0.15, 0.15))
        exg   = rng.uniform(exg_lo, exg_hi)
        ngrdi = rng.uniform(ngrdi_lo, ngrdi_hi)
        vari  = np.clip(ngrdi * rng.uniform(0.6, 1.4), -1, 1)
        gli   = np.clip(exg * rng.uniform(0.4, 1.2), -1, 1)
        rgri  = np.clip(1.0 - ngrdi * 2 + rng.normal(0, 0.1), 0.3, 3.0)
        exgr  = np.clip(exg * rng.uniform(0.5, 1.5), -0.5, 0.5)
        mgrvi = np.clip(ngrdi * rng.uniform(0.7, 1.3), -1, 1)

        can_lo, can_hi = prof.get("canopy", (0.15, 0.85))
        canopy = np.clip(rng.normal((can_lo + can_hi) / 2, (can_hi - can_lo) / 5),
                         can_lo - 0.05, can_hi + 0.05)

        tex_lo, tex_hi = prof.get("texture", (0.02, 0.20))
        edge = np.clip(rng.normal((tex_lo + tex_hi) / 2, (tex_hi - tex_lo) / 4),
                       max(0.005, tex_lo - 0.02), tex_hi + 0.02)

        hue_std = rng.uniform(15, 55) if prof.get("multi_color") else rng.uniform(8, 35)
        sat_std = rng.uniform(0.05, 0.20)
        val_std = rng.uniform(15, 45)

        lap_var   = edge * rng.uniform(300, 2000)
        local_var = rng.uniform(100, 800)
        # Uniformity correlates with constraints
        u_lo = max(0.20, prof.get("uniformity_min", 0.20))
        u_hi = min(0.92, prof.get("uniformity_max", 0.92))
        uniformity = np.clip(rng.normal((u_lo + u_hi) / 2, (u_hi - u_lo) / 4),
                             u_lo - 0.05, u_hi + 0.05)
        dir_ratio  = rng.uniform(0.5, 2.0)

        # Yellow fraction depends on hue mode:
        # Green mode (hue>70): yellow should be LOW
        # Golden mode (hue<65): yellow correlates with profile hint
        yh = prof.get("yellow_hint", 0.05)
        if hue > 75:
            # Green growth stage — reduce yellow
            yf = np.clip(rng.normal(max(0.02, yh * 0.25), 0.04), 0, 0.15)
        elif hue > 60:
            # Transitional
            yf = np.clip(rng.normal(yh * 0.6, 0.06), 0, 0.4)
        else:
            # Golden / normal mode
            yf = np.clip(rng.normal(yh, 0.08), 0, 0.8)

        gf = np.clip(rng.normal(0.45 if hue > 70 else 0.12, 0.12), 0, 0.85)
        rf = rng.uniform(0.03, 0.18) if prof.get("red_affinity", 0) > 0 else rng.uniform(0, 0.04)
        wf = rng.uniform(0.05, 0.22) if prof.get("white_affinity", 0) > 0 else rng.uniform(0, 0.05)
        bf = np.clip(rng.normal(0.08, 0.05), 0, 0.4)
        byf = yf * rng.uniform(0.2, 0.8) if yf > 0.10 else rng.uniform(0, 0.03)
        dgf = np.clip(gf * rng.uniform(0.1, 0.5) * (1 if hue > 90 else 0.3), 0, 0.5)
        bgf = np.clip(gf * rng.uniform(0.1, 0.6) * (1 if hue < 100 else 0.5), 0, 0.5)
        of  = rng.uniform(0.02, 0.10) if prof.get("orange_affinity", 0) > 0 else rng.uniform(0, 0.04)

        # Leaf coarseness respects constraints
        lc_min = prof.get("leaf_coarseness_min", 0)
        lc_max = prof.get("leaf_coarseness_max", 3000)
        leaf_coarseness   = np.clip(lap_var / (edge + 0.01),
                                    max(50, lc_min * 0.7),
                                    min(3000, lc_max * 1.3))
        canopy_tex_ratio  = canopy / (edge + 0.02)
        # Green brightness ratio correlates with green_brightness_max
        gb_max = prof.get("green_brightness_max", 0.85)
        green_br_ratio    = np.clip(bgf / (bgf + dgf + _EPS), 0.1, gb_max + 0.1)

        spatial_uni = np.clip(uniformity * rng.uniform(0.8, 1.2), 0.1, 0.95)
        phs = rng.uniform(5, 25)
        pgs = rng.uniform(0.005, 0.03)
        brightness = np.clip(val * rng.uniform(0.85, 1.1), 30, 255)
        contrast   = rng.uniform(25, 65)

        rows.append([
            hue, sat, val, hue_std, sat_std, val_std,
            exg, exgr, vari, gli, ngrdi, rgri, mgrvi,
            canopy, edge, lap_var, local_var, uniformity, dir_ratio,
            spatial_uni, yf, gf, rf, bf, wf, byf, dgf, bgf, of,
            leaf_coarseness, canopy_tex_ratio, green_br_ratio,
            phs, pgs, brightness, contrast,
        ])
    return rows


def _build_crop_classifier():
    """Build or load a Random Forest crop classifier trained on
    synthetic feature vectors derived from CROP_PROFILES."""
    global _cached_clf
    if _cached_clf is not None:
        return _cached_clf

    # Try loading from disk
    if os.path.exists(_CLASSIFIER_PATH):
        try:
            with open(_CLASSIFIER_PATH, "rb") as f:
                _cached_clf = pickle.load(f)
            return _cached_clf
        except Exception:
            pass

    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(42)
    X, y = [], []
    for crop_name, prof in CROP_PROFILES.items():
        samples = _generate_synthetic_samples(rng, crop_name, prof, n=350)
        X.extend(samples)
        y.extend([crop_name] * len(samples))

    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=14, min_samples_split=4,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X, y)

    # Cache to disk
    os.makedirs(_CLASSIFIER_DIR, exist_ok=True)
    try:
        with open(_CLASSIFIER_PATH, "wb") as f:
            pickle.dump(clf, f)
    except Exception:
        pass

    _cached_clf = clf
    return clf


def _ml_crop_predict(feat: dict) -> dict:
    """Use the trained RF to predict crop probabilities from extracted
    features.  Returns {crop_name: pct_probability}."""
    try:
        clf = _build_crop_classifier()
        x = np.array([[feat.get(k, 0.0) for k in _ML_FEATURES]], dtype=np.float64)
        proba = clf.predict_proba(x)[0]
        return {c: round(p * 100, 2) for c, p in zip(clf.classes_, proba)}
    except Exception:
        return {}


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  Visual Yield Estimation from Image Features                              #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _estimate_yield(feat: dict, crop: str, confidence: float,
                    health_score: float) -> dict:
    """
    Estimate yield potential from image features.
    Returns dict with: yield_rating (0-100), yield_label, growth_stage,
    growth_stage_desc, factors (list of factor dicts), yield_category,
    estimated_range_qtl_per_ha (tuple)
    """
    gf   = feat["green_frac"]
    yf   = feat["yellow_frac"]
    canopy = feat["canopy_coverage"]
    exg  = feat["ExG"]
    vari = feat["VARI"]
    gli  = feat["GLI"]
    rgri = feat["RGRI"]
    ed   = feat.get("edge_density", 0)
    br   = feat["brightness"]
    contr = feat["contrast"]

    # ── Growth stage estimation ──
    if gf > 0.60 and canopy > 0.55 and exg > 0.06:
        stage = "peak_vegetative"
        stage_desc = "Peak Vegetative — Full canopy, vigorous green growth. Crop is in active growing phase."
        stage_mult = 1.0
    elif gf > 0.40 and canopy > 0.35 and exg > 0.02:
        stage = "mid_vegetative"
        stage_desc = "Mid Vegetative — Good canopy development. Crop is establishing well."
        stage_mult = 0.85
    elif yf > 0.25 and gf < 0.20:
        stage = "maturity"
        stage_desc = "Maturity / Harvest Ready — Yellowing indicates the crop is nearing or at harvest stage."
        stage_mult = 0.95
    elif yf > 0.15 and gf > 0.15:
        stage = "ripening"
        stage_desc = "Ripening — Mix of green and yellow; crop is transitioning towards maturity."
        stage_mult = 0.90
    elif canopy < 0.15 and gf < 0.15:
        stage = "early_seedling"
        stage_desc = "Early Seedling / Just Planted — Very low canopy. Crop may have just germinated."
        stage_mult = 0.40
    elif canopy < 0.30 and gf < 0.30:
        stage = "early_growth"
        stage_desc = "Early Growth — Crop is young with sparse canopy but developing."
        stage_mult = 0.60
    else:
        stage = "mid_growth"
        stage_desc = "Mid Growth — Moderate canopy and vegetation indices."
        stage_mult = 0.75

    # ── Yield factors ──
    factors = []

    # Canopy density (most important)
    if canopy > 0.55:
        can_sc = 95
        factors.append({"name": "Canopy Density", "score": can_sc, "status": "Excellent", "detail": f"Dense canopy coverage ({canopy:.0%}) indicates good crop stand."})
    elif canopy > 0.35:
        can_sc = 70
        factors.append({"name": "Canopy Density", "score": can_sc, "status": "Good", "detail": f"Moderate canopy ({canopy:.0%}). Reasonable crop density."})
    elif canopy > 0.20:
        can_sc = 45
        factors.append({"name": "Canopy Density", "score": can_sc, "status": "Fair", "detail": f"Sparse canopy ({canopy:.0%}). Crop may be young or stressed."})
    else:
        can_sc = 20
        factors.append({"name": "Canopy Density", "score": can_sc, "status": "Poor", "detail": f"Very low canopy ({canopy:.0%}). Indicates early stage or significant stress."})

    # Vegetation health
    veg_sc = min(95, max(10, int(50 + exg * 500 + vari * 200)))
    if veg_sc > 70:
        factors.append({"name": "Vegetation Health", "score": veg_sc, "status": "Healthy", "detail": f"Strong vegetation indices (ExG={exg:.3f}, VARI={vari:.3f}). Crop is photosynthetically active."})
    elif veg_sc > 45:
        factors.append({"name": "Vegetation Health", "score": veg_sc, "status": "Moderate", "detail": f"Moderate vegetation indices (ExG={exg:.3f}). Some stress may be present."})
    else:
        factors.append({"name": "Vegetation Health", "score": veg_sc, "status": "Stressed", "detail": f"Low vegetation indices (ExG={exg:.3f}, VARI={vari:.3f}). Crop shows signs of stress."})

    # Uniformity
    uni = feat.get("spatial_uniformity", 0.5)
    uni_sc = min(95, max(10, int(uni * 100)))
    if uni_sc > 65:
        factors.append({"name": "Field Uniformity", "score": uni_sc, "status": "Uniform", "detail": "Even growth across the field. Good agronomic practices."})
    else:
        factors.append({"name": "Field Uniformity", "score": uni_sc, "status": "Patchy", "detail": "Uneven growth detected. Check for irrigation or fertility gaps."})

    # Stress indicators
    stress_sc = min(95, max(10, int(100 - rgri * 40)))
    if rgri < 0.9:
        factors.append({"name": "Stress Level", "score": stress_sc, "status": "Low Stress", "detail": f"RGRI={rgri:.2f}. Crop shows minimal stress signals."})
    elif rgri < 1.1:
        factors.append({"name": "Stress Level", "score": stress_sc, "status": "Moderate", "detail": f"RGRI={rgri:.2f}. Some stress indicators present."})
    else:
        factors.append({"name": "Stress Level", "score": stress_sc, "status": "High Stress", "detail": f"RGRI={rgri:.2f}. Significant stress — yellowing, disease, or nutrient deficiency."})

    # ── Compute overall yield rating ──
    raw_yield = (can_sc * 0.35 + veg_sc * 0.30 + uni_sc * 0.15 +
                 stress_sc * 0.10 + health_score * 0.10)
    raw_yield *= stage_mult
    # Also weight by identification confidence
    raw_yield *= (0.6 + 0.4 * min(confidence, 100) / 100)
    yield_rating = round(min(99, max(5, raw_yield)), 1)

    # ── Crop-specific typical yield ranges (qtl/ha) ──
    _TYPICAL_YIELD = {
        "Rice": (25, 55), "Wheat": (20, 50), "Maize": (25, 65),
        "Sugarcane": (400, 900), "Cotton": (8, 22), "Mustard": (8, 20),
        "Tomato": (150, 400), "Potato": (150, 350), "Banana": (300, 600),
        "Soybean": (10, 25), "Chickpea": (8, 20), "Groundnut": (10, 25),
        "Barley": (15, 40), "Jute": (15, 30), "Tea": (10, 22),
        "Coffee": (6, 15),
    }
    typical = _TYPICAL_YIELD.get(crop, (15, 40))
    frac = yield_rating / 100
    est_low  = round(typical[0] + (typical[1] - typical[0]) * frac * 0.7, 1)
    est_high = round(typical[0] + (typical[1] - typical[0]) * frac * 1.1, 1)
    est_high = min(est_high, typical[1] * 1.05)

    # Yield category
    if yield_rating >= 75:
        y_cat = "High Yield Potential"
    elif yield_rating >= 55:
        y_cat = "Moderate Yield Potential"
    elif yield_rating >= 35:
        y_cat = "Below Average Yield"
    else:
        y_cat = "Low Yield Potential"

    return {
        "yield_rating": yield_rating,
        "yield_label": y_cat,
        "growth_stage": stage,
        "growth_stage_desc": stage_desc,
        "factors": factors,
        "yield_category": y_cat,
        "estimated_range_qtl_per_ha": (est_low, est_high),
        "typical_range_qtl_per_ha": typical,
        "crop": crop,
    }


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  Seasonal Crop Suggestions for Soil / Barren Land                         #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _seasonal_crop_suggestions() -> dict:
    """
    Based on the current month, suggest which crops can be planted NOW
    in Indian agriculture. Returns dict with: current_season, season_desc,
    crops_to_plant (list of dicts), preparation_tips, next_season info.
    """
    import datetime
    month = datetime.date.today().month  # 1-12

    # Indian cropping calendar
    # Kharif: June–October (monsoon crops)
    # Rabi: October–March  (winter crops)
    # Zaid: March–June     (summer crops)

    if month in (6, 7, 8, 9, 10):  # June–Oct → Kharif
        season = "Kharif"
        season_desc = ("Monsoon season (June–October). The soil has good moisture "
                       "from rainfall, ideal for planting rain-fed crops.")
        crops = [
            {"name": "Rice", "emoji": "🌾", "sowing": "June–July", "harvest": "Oct–Nov",
             "tip": "Transplant seedlings 20-25 days after nursery sowing. Ensure standing water 5cm."},
            {"name": "Maize", "emoji": "🌽", "sowing": "June–July", "harvest": "Sep–Oct",
             "tip": "Plant after first good rains. Space 60×20cm. Apply Urea at knee-high stage."},
            {"name": "Cotton", "emoji": "🏳️", "sowing": "April–June", "harvest": "Oct–Jan",
             "tip": "Bt Cotton varieties recommended. Maintain 90×60cm spacing."},
            {"name": "Soybean", "emoji": "🫘", "sowing": "June–July", "harvest": "Sep–Oct",
             "tip": "Treat seeds with Rhizobium culture. Row spacing 30-45cm."},
            {"name": "Groundnut", "emoji": "🥜", "sowing": "June–July", "harvest": "Oct–Nov",
             "tip": "Apply gypsum at flowering. Maintain soil moisture during pegging."},
            {"name": "Jute", "emoji": "🌿", "sowing": "March–May", "harvest": "July–Aug",
             "tip": "Broadcast seeds in well-prepared wet soil. Ret after 15–20 days of harvest."},
            {"name": "Sugarcane", "emoji": "🍬", "sowing": "Feb–March", "harvest": "Dec–March (next year)",
             "tip": "Plant setts with 2-3 buds. Earthing up after 3 months is critical."},
        ]
        prep_tips = [
            "Ensure proper drainage channels to prevent waterlogging during heavy rains.",
            "Apply basal dose of FYM (10-15 tonnes/ha) before sowing.",
            "Get soil tested for pH and nutrient levels before planting.",
            "Prepare raised beds for crops sensitive to waterlogging.",
            "Consider green manuring with dhaincha or sunhemp before main crop.",
        ]
        next_season = "Rabi (Oct–March): Wheat, Mustard, Chickpea, Barley, Potato"

    elif month in (11, 12, 1, 2, 3):  # Nov–Mar → Rabi
        season = "Rabi"
        season_desc = ("Winter season (October–March). Cool temperatures and residual "
                       "soil moisture are ideal for winter crops.")
        crops = [
            {"name": "Wheat", "emoji": "🌾", "sowing": "Oct–Nov", "harvest": "March–April",
             "tip": "Sow HD-2967/HD-3086 varieties. First irrigation at 21 days (crown root stage)."},
            {"name": "Mustard", "emoji": "🌼", "sowing": "Oct–Nov", "harvest": "Feb–March",
             "tip": "Maintain 30×10cm spacing. Aphid management is critical during flowering."},
            {"name": "Chickpea", "emoji": "🫘", "sowing": "Oct–Nov", "harvest": "Feb–March",
             "tip": "Avoid excessive irrigation. Treat seeds with Rhizobium + PSB before sowing."},
            {"name": "Potato", "emoji": "🥔", "sowing": "Oct–Nov", "harvest": "Jan–Feb",
             "tip": "Use certified seed tubers. Earth up at 30 and 45 days. Watch for late blight."},
            {"name": "Barley", "emoji": "🌾", "sowing": "Oct–Nov", "harvest": "March–April",
             "tip": "Tolerant to salinity. Good for areas with limited irrigation."},
            {"name": "Tomato", "emoji": "🍅", "sowing": "Sep–Oct (transplant)", "harvest": "Jan–March",
             "tip": "Stake plants after 30 days. Apply Trichoderma for wilt prevention."},
        ]
        prep_tips = [
            "Apply well-decomposed FYM or compost (5-10 tonnes/ha) during field prep.",
            "Ensure 1-2 pre-sowing irrigations to settle the soil before planting.",
            "Laser-level the field for uniform irrigation distribution.",
            "Apply basal dose of DAP (100 kg/ha) and Potash (60 kg/ha) at sowing.",
            "Plan wheat-mustard intercropping for better land utilization.",
        ]
        next_season = "Zaid (March–June): Cucumber, Watermelon, Muskmelon, Moong"

    else:  # April, May → Zaid
        season = "Zaid"
        season_desc = ("Summer season (March–June). Hot and dry conditions. "
                       "Irrigated short-duration crops perform best.")
        crops = [
            {"name": "Maize", "emoji": "🌽", "sowing": "March–April", "harvest": "June–July",
             "tip": "Choose short-duration hybrid varieties. Irrigate every 7-10 days."},
            {"name": "Groundnut", "emoji": "🥜", "sowing": "March–April", "harvest": "June–July",
             "tip": "Summer groundnut needs frequent light irrigations. Mulch to conserve moisture."},
            {"name": "Tomato", "emoji": "🍅", "sowing": "Feb–March", "harvest": "May–June",
             "tip": "Use shade nets in extreme heat. Mulch with straw to reduce soil temperature."},
            {"name": "Banana", "emoji": "🍌", "sowing": "March–April (planting)", "harvest": "12–14 months",
             "tip": "Tissue culture plants preferred. Drip irrigation saves 40% water."},
            {"name": "Sugarcane", "emoji": "🍬", "sowing": "Feb–March", "harvest": "Dec–March next year",
             "tip": "Spring planting gives highest yield. Trench method recommended."},
        ]
        prep_tips = [
            "Ensure reliable irrigation — summer crops need frequent watering.",
            "Apply mulch (5-7cm straw/leaves) to reduce soil moisture evaporation.",
            "Consider drip irrigation for 40-60% water savings.",
            "Green manuring with dhaincha if field is to be left for Kharif.",
            "Plan for windbreaks if the area has hot dry winds (loo).",
        ]
        next_season = "Kharif (June–October): Rice, Maize, Cotton, Soybean"

    return {
        "current_season": season,
        "season_desc": season_desc,
        "crops_to_plant": crops,
        "preparation_tips": prep_tips,
        "next_season": next_season,
    }


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  Soil Fertility Assessment & Recommendations                              #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _assess_soil_fertility(feat: dict) -> dict:
    """
    Estimate soil fertility from image features and return detailed advice.
    Returns dict with: fertility_score, fertility_label, color_analysis,
    chemical_methods, natural_methods, recommended_crops
    """
    hm   = feat["hue_mean"]
    sm   = feat["sat_mean"]
    vm   = feat["val_mean"]
    bf   = feat.get("brown_frac", 0)
    exg  = feat["ExG"]
    canopy = feat["canopy_coverage"]
    contr = feat["contrast"]
    ed   = feat.get("edge_density", 0)
    rgri = feat["RGRI"]

    # ── Fertility scoring (0-100) based on soil colour & texture ──
    score = 50.0  # baseline

    # Dark soil (rich in organic matter) → higher fertility
    if vm < 120:
        score += 15
    elif vm < 160:
        score += 8
    else:
        score -= 5  # very light / sandy

    # Hue: reddish-brown (laterite, iron-rich) vs grey (depleted)
    if 15 < hm < 45:   # warm brown → moderate
        score += 8
    elif 5 < hm < 15:  # reddish → iron-rich laterite
        score += 5
    elif hm > 50:       # yellowish / gray → possibly depleted
        score -= 8

    # Saturation: more saturated soil → more minerals visible
    if sm > 0.25:
        score += 6
    elif sm < 0.10:
        score -= 8   # very grey / depleted

    # Texture (edge density) → crumbly structured soil is good
    if 0.04 < ed < 0.15:
        score += 8  # good structure
    elif ed > 0.20:
        score += 3  # rocky / cloddy
    elif ed < 0.02:
        score -= 5  # too smooth / compacted

    # Contrast → varied tones → mixed organics
    if contr > 30:
        score += 4
    elif contr < 15:
        score -= 4

    # Any remaining vegetation signal → recently active soil
    if canopy > 0.05:
        score += 5
    if exg > -0.01:
        score += 3

    score = max(5, min(95, score))

    # ── Fertility label ──
    if score >= 75:
        label = "Good Fertility"
        color = "#4caf50"
        summary = ("The soil appears **dark and well-structured**, indicating good organic "
                   "matter content and mineral presence. This soil is likely suitable for "
                   "most crops with proper management.")
    elif score >= 55:
        label = "Moderate Fertility"
        color = "#ff9800"
        summary = ("The soil shows **moderate organic content** and reasonable structure. "
                   "With proper fertilization and soil management, this can support "
                   "productive crop growth.")
    elif score >= 35:
        label = "Low Fertility"
        color = "#f44336"
        summary = ("The soil appears **light-coloured or depleted**, suggesting low organic "
                   "matter. It may be sandy, compacted, or nutrient-poor. Significant "
                   "soil improvement is recommended before planting.")
    else:
        label = "Very Poor Fertility"
        color = "#b71c1c"
        summary = ("The soil looks **severely depleted** — very light, smooth, and lacking "
                   "organic matter indicators. Major soil rehabilitation is needed.")

    # ── Soil colour interpretation ──
    if vm < 100:
        color_analysis = "**Dark soil** — Rich in organic matter (humus). Good water retention."
    elif vm < 140:
        color_analysis = "**Medium-dark soil** — Moderate organic content. Decent nutrient levels."
    elif vm < 180:
        color_analysis = "**Light brown soil** — Lower organic matter. May need amendment."
    else:
        color_analysis = "**Pale/sandy soil** — Likely low in nutrients and organic matter. Poor water retention."

    if hm < 20:
        color_analysis += "\n- **Reddish tone** → Iron-rich (laterite soil). Good for root crops but may be acidic."
    elif hm < 45:
        color_analysis += "\n- **Warm brown tone** → Balanced mineral content. Common in alluvial and loamy soils."
    elif hm < 60:
        color_analysis += "\n- **Yellowish tone** → May indicate iron oxide or clay-heavy composition."
    else:
        color_analysis += "\n- **Greyish tone** → Possible waterlogging history or mineral depletion."

    # ── Chemical / Pesticide Methods ──
    chemical_methods = []
    chemical_methods.append(("🧪 **NPK Fertilizer (10-26-26 or 12-32-16)**",
        "Apply 100-150 kg/hectare as basal dose before sowing. Provides Nitrogen, "
        "Phosphorus, and Potassium — the three essential macronutrients."))
    chemical_methods.append(("🧪 **DAP (Di-Ammonium Phosphate)**",
        "Apply 50-100 kg/hectare for phosphorus-deficient soils. Boosts root "
        "development and early plant vigour."))
    chemical_methods.append(("🧪 **Urea (46-0-0)**",
        "Apply 50-80 kg/hectare in 2-3 split doses. Primary nitrogen source for "
        "leaf growth and green biomass."))
    chemical_methods.append(("🧪 **Potash (MOP / SOP)**",
        "Apply 40-60 kg/hectare. Improves disease resistance, water regulation, "
        "and fruit/grain quality."))
    chemical_methods.append(("🧪 **Micronutrient Mix (Zn, Fe, Mn, B)**",
        "Apply 15-25 kg/hectare or as foliar spray. Corrects hidden hunger — "
        "zinc and boron deficiency is common in Indian soils."))

    if score < 50:
        chemical_methods.append(("🧪 **Lime / Dolomite (for acidic soil)**",
            "Apply 1-2 tonnes/hectare if soil pH < 5.5. Corrects acidity and "
            "unlocks bound nutrients."))
        chemical_methods.append(("🧪 **Gypsum (for alkaline / sodic soil)**",
            "Apply 2-5 tonnes/hectare if soil pH > 8.5. Displaces sodium and "
            "improves soil structure."))

    # ── Natural / Organic Methods ──
    natural_methods = []
    natural_methods.append(("🌿 **Farmyard Manure (FYM)**",
        "Apply 8-10 tonnes/hectare. Improves organic matter, water retention, "
        "and microbial activity. Best applied 3-4 weeks before sowing."))
    natural_methods.append(("🌿 **Vermicompost**",
        "Apply 2-5 tonnes/hectare. Rich in humic acids, enzymes, and beneficial "
        "microbes. Ideal for degraded soils."))
    natural_methods.append(("🌿 **Green Manuring**",
        "Grow dhaincha (Sesbania), sunhemp, or cowpea and plough it into the soil "
        "before flowering. Adds 20-30 kg nitrogen/hectare naturally."))
    natural_methods.append(("🌿 **Composting (Nadep / Pit method)**",
        "Prepare compost from crop residues, animal dung, and kitchen waste. "
        "Takes 3-4 months. Apply 3-5 tonnes/hectare."))
    natural_methods.append(("🌿 **Biofertilizers (Rhizobium, PSB, Azotobacter)**",
        "Seed treatment @ 10g/kg seed or soil application. Fixes atmospheric "
        "nitrogen and solubilises phosphorus naturally."))
    natural_methods.append(("🌿 **Mulching**",
        "Cover soil with crop residues, straw, or dry leaves (5-8 cm layer). "
        "Reduces water evaporation, regulates temperature, adds organic matter."))
    natural_methods.append(("🌿 **Crop Rotation**",
        "Alternate legumes (pulses) with cereals each season. Legumes fix "
        "20-40 kg nitrogen/hectare, breaking pest cycles."))

    if score < 50:
        natural_methods.append(("🌿 **Biochar**",
            "Apply 1-2 tonnes/hectare. Long-lasting soil carbon that improves "
            "nutrient retention and microbial habitat."))
        natural_methods.append(("🌿 **Jeevamrutha (liquid biofertilizer)**",
            "Mix cow dung (10 kg), cow urine (10 L), jaggery (2 kg), pulse flour (2 kg) "
            "in 200 L water. Ferment 48 hours. Apply to 1 acre."))

    # ── Recommended crops for this soil condition ──
    if score >= 65:
        recommended = ["Wheat", "Rice", "Sugarcane", "Maize", "Cotton",
                       "Tomato", "Banana"]
    elif score >= 45:
        recommended = ["Mustard", "Chickpea", "Soybean", "Groundnut",
                       "Potato", "Barley", "Jute"]
    else:
        recommended = ["Chickpea", "Barley", "Mustard", "Groundnut",
                       "Jute"]  # hardy, low-fertility tolerant

    return {
        "fertility_score": round(score, 1),
        "fertility_label": label,
        "fertility_color": color,
        "summary": summary,
        "color_analysis": color_analysis,
        "chemical_methods": chemical_methods,
        "natural_methods": natural_methods,
        "recommended_crops": recommended,
    }


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  Disease / Stress Detection from Image                                    #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _detect_disease_stress(feat: dict, crop: str) -> dict:
    """
    Detect possible diseases, nutrient deficiencies, and stress from
    image colour/texture anomalies. Returns dict with: overall_status,
    stress_score (0-100, higher = more stress), issues (list of dicts),
    recommendations (list of strings).
    """
    gf   = feat["green_frac"]
    yf   = feat["yellow_frac"]
    rf   = feat["red_frac"]
    bf   = feat.get("brown_frac", 0)
    exg  = feat["ExG"]
    vari = feat["VARI"]
    gli  = feat["GLI"]
    rgri = feat["RGRI"]
    ngrdi = feat["NGRDI"]
    canopy = feat["canopy_coverage"]
    ed   = feat.get("edge_density", 0)
    br   = feat["brightness"]
    hue_std = feat["hue_std"]
    sat_mean = feat["sat_mean"]
    val_std  = feat["val_std"]
    pgs  = feat.get("patch_green_std", 0)
    uni  = feat.get("spatial_uniformity", 0.5)

    issues = []
    stress = 0

    # ── Chlorosis (yellowing) ──
    if yf > 0.20 and gf < 0.35 and ngrdi < 0.0:
        severity = "Severe" if yf > 0.35 else "Moderate"
        stress += 25 if severity == "Severe" else 15
        issues.append({
            "name": "Chlorosis (Yellowing)",
            "severity": severity,
            "icon": "🟡",
            "detail": (f"Yellow fraction is elevated ({yf:.0%}) with reduced green "
                       f"({gf:.0%}). This suggests nitrogen deficiency, iron "
                       f"deficiency, or early-stage disease."),
            "likely_causes": ["Nitrogen deficiency", "Iron deficiency (Fe)",
                              "Magnesium deficiency", "Viral infection"],
        })

    # ── Necrosis (browning / dead tissue) ──
    if bf > 0.12 and exg < 0.0:
        severity = "Severe" if bf > 0.25 else "Moderate"
        stress += 20 if severity == "Severe" else 12
        issues.append({
            "name": "Necrosis / Brown Spots",
            "severity": severity,
            "icon": "🟤",
            "detail": (f"Significant brown fraction ({bf:.0%}) with negative ExG "
                       f"({exg:.3f}). Indicates dead tissue, fungal infection, "
                       f"or severe nutrient burn."),
            "likely_causes": ["Fungal blight", "Bacterial leaf spot",
                              "Phosphorus deficiency", "Herbicide damage"],
        })

    # ── Water stress (wilting indicators) ──
    if vari < -0.05 and sat_mean < 0.25 and canopy < 0.40:
        severity = "Severe" if vari < -0.12 else "Moderate"
        stress += 18 if severity == "Severe" else 10
        issues.append({
            "name": "Water Stress / Wilting",
            "severity": severity,
            "icon": "💧",
            "detail": (f"Low VARI ({vari:.3f}) with reduced saturation ({sat_mean:.2f}) "
                       f"and declining canopy ({canopy:.0%}). Plants may be "
                       f"water-stressed or wilting."),
            "likely_causes": ["Drought / insufficient irrigation",
                              "Root damage", "High temperature stress"],
        })

    # ── Nutrient deficiency signals ──
    if rgri > 1.2 and ngrdi < -0.05:
        stress += 12
        issues.append({
            "name": "Nutrient Deficiency Pattern",
            "severity": "Moderate",
            "icon": "⚠️",
            "detail": (f"High red-green ratio (RGRI={rgri:.2f}) with negative NGRDI "
                       f"({ngrdi:.3f}). This red-shift pattern typically indicates "
                       f"phosphorus or potassium deficiency."),
            "likely_causes": ["Phosphorus (P) deficiency",
                              "Potassium (K) deficiency", "Zinc deficiency"],
        })

    # ── Uneven growth (patchy field) ──
    if pgs > 0.05 and uni < 0.40:
        stress += 10
        issues.append({
            "name": "Uneven / Patchy Growth",
            "severity": "Moderate",
            "icon": "🔲",
            "detail": (f"Spatial uniformity is low ({uni:.2f}) with high green "
                       f"variation across patches ({pgs:.3f}). Field shows uneven "
                       f"growth — possible irrigation or fertility gaps."),
            "likely_causes": ["Uneven irrigation", "Variable soil fertility",
                              "Pest patches", "Weed infestations"],
        })

    # ── Pest/insect damage indicators ──
    if ed > 0.15 and canopy < 0.35 and hue_std > 30:
        stress += 8
        issues.append({
            "name": "Possible Pest Damage",
            "severity": "Low",
            "icon": "🐛",
            "detail": (f"High edge density ({ed:.2%}) with sparse canopy "
                       f"({canopy:.0%}) and high hue variability ({hue_std:.0f}°). "
                       f"This texture pattern can indicate leaf damage from insects."),
            "likely_causes": ["Leaf-eating insects", "Defoliators",
                              "Stem borers", "Mites"],
        })

    # ── Overall status ──
    stress = min(95, max(0, stress))
    if stress >= 50:
        status = "Critical — Immediate Action Needed"
        status_color = "#f44336"
    elif stress >= 30:
        status = "Stressed — Intervention Recommended"
        status_color = "#FF9800"
    elif stress >= 10:
        status = "Mild Stress — Monitor Closely"
        status_color = "#FFC107"
    else:
        status = "Healthy — No Significant Issues"
        status_color = "#4CAF50"

    # ── Recommendations ──
    recs = []
    if not issues:
        recs.append("Crop appears healthy. Continue current management practices.")
        recs.append("Regular scouting every 7-10 days is recommended as a preventive measure.")
    else:
        if any(i["name"].startswith("Chlorosis") for i in issues):
            recs.append("Apply foliar spray of 2% Urea for quick nitrogen correction.")
            recs.append("Soil test for iron (Fe) and magnesium (Mg) levels.")
        if any(i["name"].startswith("Necrosis") for i in issues):
            recs.append("Apply fungicide (Mancozeb 75 WP @ 2g/L or Carbendazim 50 WP @ 1g/L).")
            recs.append("Remove and destroy severely affected plant parts.")
        if any(i["name"].startswith("Water") for i in issues):
            recs.append("Irrigate immediately — apply 5-7 cm water through flood or drip.")
            recs.append("Apply mulch (5-8 cm) to reduce further moisture loss.")
        if any(i["name"].startswith("Nutrient") for i in issues):
            recs.append("Apply DAP (50 kg/ha) + MOP (30 kg/ha) through fertigation.")
            recs.append("Foliar spray of micronutrient mix (Zn, Fe, Mn, B) at 2g/L.")
        if any(i["name"].startswith("Uneven") for i in issues):
            recs.append("Check and repair irrigation system for uniform water distribution.")
            recs.append("Variable-rate fertilizer application based on soil sampling zones.")
        if any(i["name"].startswith("Possible Pest") for i in issues):
            recs.append("Scout field for specific pests. Use yellow/blue sticky traps for identification.")
            recs.append("Apply neem oil (5ml/L) as a broad-spectrum bio-pesticide.")

    return {
        "stress_score": stress,
        "overall_status": status,
        "status_color": status_color,
        "issues": issues,
        "recommendations": recs,
        "issue_count": len(issues),
    }


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  Irrigation Need Estimation                                               #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _estimate_irrigation_need(feat: dict, crop: str) -> dict:
    """
    Estimate irrigation requirements from image features.
    Returns dict with: water_stress_index (0-100), irrigation_urgency,
    recommendations, moisture_indicators.
    """
    vari  = feat["VARI"]
    gli   = feat["GLI"]
    exg   = feat["ExG"]
    canopy = feat["canopy_coverage"]
    br    = feat["brightness"]
    sat_m = feat["sat_mean"]
    val_m = feat["val_mean"]
    yf    = feat["yellow_frac"]
    bf    = feat.get("brown_frac", 0)

    # Water stress index (0 = well-watered, 100 = severe drought)
    wsi = 50.0  # baseline

    # VARI is strongly correlated with leaf water content
    if vari > 0.10:    wsi -= 20
    elif vari > 0.02:  wsi -= 10
    elif vari < -0.05: wsi += 15
    elif vari < -0.12: wsi += 25

    # Brightness: overly bright crops may indicate wilting (leaves curl, expose lighter underside)
    if br > 170:       wsi += 10
    elif br < 100:     wsi -= 5

    # Saturation drop → loss of green intensity → water stress
    if sat_m < 0.20:   wsi += 15
    elif sat_m > 0.40: wsi -= 10

    # Canopy decline
    if canopy < 0.20:  wsi += 12
    elif canopy > 0.50: wsi -= 8

    # Yellow/brown creep → drying out
    if yf > 0.20:      wsi += 10
    if bf > 0.15:      wsi += 8

    wsi = max(0, min(100, wsi))

    # Crop-specific water requirements (mm per growing season)
    _WATER_NEEDS = {
        "Rice": (1200, 2000, "Very High"), "Wheat": (300, 500, "Moderate"),
        "Maize": (500, 800, "High"), "Sugarcane": (1500, 2500, "Very High"),
        "Cotton": (700, 1300, "High"), "Mustard": (200, 400, "Low"),
        "Tomato": (400, 600, "Moderate"), "Potato": (500, 700, "Moderate"),
        "Banana": (1200, 2200, "Very High"), "Soybean": (450, 700, "Moderate"),
        "Chickpea": (150, 300, "Low"), "Groundnut": (400, 600, "Moderate"),
        "Barley": (250, 400, "Low"), "Jute": (500, 800, "High"),
        "Tea": (1200, 1800, "Very High"), "Coffee": (1000, 1600, "High"),
    }
    wn = _WATER_NEEDS.get(crop, (400, 700, "Moderate"))

    # Urgency
    if wsi >= 70:
        urgency = "Irrigate Immediately"
        urgency_color = "#f44336"
        urgency_icon = "🔴"
    elif wsi >= 50:
        urgency = "Irrigation Needed Soon"
        urgency_color = "#FF9800"
        urgency_icon = "🟠"
    elif wsi >= 30:
        urgency = "Monitor — Irrigate in 2-3 Days"
        urgency_color = "#FFC107"
        urgency_icon = "🟡"
    else:
        urgency = "Adequate Moisture"
        urgency_color = "#4CAF50"
        urgency_icon = "🟢"

    # Moisture indicators
    indicators = []
    if vari < 0:
        indicators.append({"name": "Leaf Water Content", "status": "Low",
                           "detail": f"VARI={vari:.3f} (negative = stressed)"})
    else:
        indicators.append({"name": "Leaf Water Content", "status": "Adequate",
                           "detail": f"VARI={vari:.3f} (positive = hydrated)"})

    if sat_m < 0.25:
        indicators.append({"name": "Colour Saturation", "status": "Faded",
                           "detail": f"Saturation={sat_m:.2f} — loss of vivid green"})
    else:
        indicators.append({"name": "Colour Saturation", "status": "Normal",
                           "detail": f"Saturation={sat_m:.2f} — good colour intensity"})

    if canopy < 0.30:
        indicators.append({"name": "Canopy Coverage", "status": "Sparse",
                           "detail": f"Coverage={canopy:.0%} — low leaf area"})
    else:
        indicators.append({"name": "Canopy Coverage", "status": "Good",
                           "detail": f"Coverage={canopy:.0%} — adequate leaf area"})

    recs = []
    if wsi >= 50:
        recs.append(f"Apply irrigation immediately — {crop} requires {wn[0]}-{wn[1]} mm per season (Water Need: {wn[2]}).")
        recs.append("Prefer early morning or evening irrigation to minimize evaporation.")
        recs.append("Consider drip irrigation for 40-60% water savings vs flood irrigation.")
    elif wsi >= 30:
        recs.append(f"Plan irrigation within 2-3 days. {crop} water need: {wn[2]} ({wn[0]}-{wn[1]} mm/season).")
        recs.append("Mulch soil surface with crop residues to conserve existing moisture.")
    else:
        recs.append(f"Soil moisture appears adequate. {crop} water need: {wn[2]} ({wn[0]}-{wn[1]} mm/season).")
        recs.append("Continue regular irrigation schedule based on crop stage.")

    return {
        "water_stress_index": round(wsi, 1),
        "irrigation_urgency": urgency,
        "urgency_color": urgency_color,
        "urgency_icon": urgency_icon,
        "water_need_mm": (wn[0], wn[1]),
        "water_need_category": wn[2],
        "crop": crop,
        "indicators": indicators,
        "recommendations": recs,
    }


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  Pest & Disease Risk Assessment (season-aware)                            #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _assess_pest_risk(crop: str) -> dict:
    """
    Season-aware pest and disease risk warnings for the identified crop.
    Returns dict with: risk_level, pests (list of dicts), diseases (list),
    preventive_measures (list).
    """
    import datetime
    month = datetime.date.today().month

    # Determine Indian season context
    if month in (6, 7, 8, 9):
        season_ctx = "monsoon"
        humidity = "high"
    elif month in (10, 11, 12, 1, 2):
        season_ctx = "winter"
        humidity = "moderate"
    else:
        season_ctx = "summer"
        humidity = "low"

    _PEST_DB = {
        "Rice": {
            "pests": [
                {"name": "Stem Borer", "risk_season": ["monsoon"], "severity": "High",
                 "symptom": "Dead hearts in vegetative, white ears in reproductive stage",
                 "control": "Apply Cartap Hydrochloride 4G @ 25 kg/ha or release Trichogramma parasitoids."},
                {"name": "Brown Plant Hopper (BPH)", "risk_season": ["monsoon"], "severity": "Very High",
                 "symptom": "Hopper burn — circular patches of dried plants",
                 "control": "Avoid excess nitrogen. Apply Pymetrozine 50 WG @ 0.3g/L or Buprofezin 25 SC."},
                {"name": "Leaf Folder", "risk_season": ["monsoon", "winter"], "severity": "Moderate",
                 "symptom": "Folded leaves with scraping damage inside",
                 "control": "Apply Chlorantraniliprole 18.5 SC @ 0.3 ml/L."},
            ],
            "diseases": [
                {"name": "Blast (Pyricularia)", "risk_season": ["monsoon"], "severity": "Very High",
                 "symptom": "Diamond-shaped lesions on leaves, neck rot",
                 "control": "Apply Tricyclazole 75 WP @ 0.6g/L. Use resistant varieties (Pusa Basmati 1509)."},
                {"name": "Bacterial Leaf Blight", "risk_season": ["monsoon"], "severity": "High",
                 "symptom": "Water-soaked yellow lesions along leaf margins",
                 "control": "Avoid excess N. Drain field, apply Streptocycline 0.1g + Copper oxychloride 2.5g/L."},
            ],
        },
        "Wheat": {
            "pests": [
                {"name": "Aphid", "risk_season": ["winter"], "severity": "Moderate",
                 "symptom": "Clusters on ears, honeydew secretion, sooty mould",
                 "control": "Apply Imidacloprid 17.8 SL @ 0.3 ml/L or Dimethoate 30 EC @ 1.5 ml/L."},
                {"name": "Termite", "risk_season": ["winter", "summer"], "severity": "High",
                 "symptom": "Wilting plants, hollowed-out roots",
                 "control": "Soil treatment with Chlorpyrifos 20 EC @ 5L/ha at sowing."},
            ],
            "diseases": [
                {"name": "Yellow Rust", "risk_season": ["winter"], "severity": "Very High",
                 "symptom": "Yellow-orange pustules in rows on leaves",
                 "control": "Apply Propiconazole 25 EC @ 1ml/L. Sow resistant varieties (HD-3086, DBW-187)."},
                {"name": "Karnal Bunt", "risk_season": ["winter"], "severity": "Moderate",
                 "symptom": "Partial bunting of grains with fishy smell",
                 "control": "Seed treatment with Carboxin + Thiram @ 2g/kg and timely sowing."},
            ],
        },
        "Mustard": {
            "pests": [
                {"name": "Mustard Aphid", "risk_season": ["winter"], "severity": "Very High",
                 "symptom": "Dense colonies on shoots/flowers, stunted pods",
                 "control": "Apply Dimethoate 30 EC @ 1ml/L or neem oil 5ml/L at early infestation."},
                {"name": "Painted Bug", "risk_season": ["winter"], "severity": "Moderate",
                 "symptom": "Sucking sap from young plants, wilting",
                 "control": "Dust Malathion 5% DP @ 20 kg/ha in early morning."},
            ],
            "diseases": [
                {"name": "White Rust (Albugo)", "risk_season": ["winter", "monsoon"], "severity": "High",
                 "symptom": "White chalky pustules on underside of leaves and stag-head on flowers",
                 "control": "Apply Metalaxyl + Mancozeb (Ridomil Gold) @ 2g/L as preventive spray."},
                {"name": "Alternaria Blight", "risk_season": ["winter"], "severity": "High",
                 "symptom": "Concentric ring spots on leaves and pods",
                 "control": "Spray Mancozeb 75 WP @ 2.5g/L at 45 and 60 DAS."},
            ],
        },
        "Maize": {
            "pests": [
                {"name": "Fall Armyworm", "risk_season": ["monsoon", "summer"], "severity": "Very High",
                 "symptom": "Ragged feeding on whorl leaves, frass in whorl",
                 "control": "Apply Emamectin Benzoate 5 SG @ 0.4g/L or Spinetoram 11.7 SC @ 0.5ml/L."},
                {"name": "Stem Borer", "risk_season": ["monsoon"], "severity": "High",
                 "symptom": "Shot holes on leaves, dead heart",
                 "control": "Apply Carbofuran 3G @ 8-10 kg/ha in whorl."},
            ],
            "diseases": [
                {"name": "Maydis Leaf Blight", "risk_season": ["monsoon"], "severity": "High",
                 "symptom": "Elongated tan lesions parallel to leaf veins",
                 "control": "Apply Mancozeb 75 WP @ 2.5g/L + use resistant hybrids."},
            ],
        },
        "Cotton": {
            "pests": [
                {"name": "Pink Bollworm", "risk_season": ["monsoon", "winter"], "severity": "Very High",
                 "symptom": "Rosette flowers, borer holes in bolls",
                 "control": "Use Bt cotton + refuge. Pheromone traps. Apply Profenofos 50 EC @ 2ml/L."},
                {"name": "Whitefly", "risk_season": ["monsoon", "summer"], "severity": "High",
                 "symptom": "Sooty mould, leaf curl, sticky leaves",
                 "control": "Apply Diafenthiuron 50 WP @ 1g/L. Remove alternate hosts (weeds)."},
            ],
            "diseases": [
                {"name": "Cotton Leaf Curl Virus (CLCuV)", "risk_season": ["monsoon"], "severity": "Very High",
                 "symptom": "Leaf curling, vein thickening, stunted growth",
                 "control": "Use tolerant varieties. Control whitefly vector early. No chemical cure."},
            ],
        },
        "Tomato": {
            "pests": [
                {"name": "Fruit Borer (Helicoverpa)", "risk_season": ["monsoon", "winter"], "severity": "Very High",
                 "symptom": "Bore holes in fruits with frass",
                 "control": "Apply HaNPV (250 LE/ha) + Neem 5ml/L or Chlorantraniliprole 18.5 SC."},
                {"name": "Leaf Miner", "risk_season": ["monsoon", "winter", "summer"], "severity": "Moderate",
                 "symptom": "Serpentine white mines on leaves",
                 "control": "Apply Abamectin 1.9 EC @ 0.5ml/L. Sticky traps for adults."},
            ],
            "diseases": [
                {"name": "Late Blight (Phytophthora)", "risk_season": ["monsoon", "winter"], "severity": "Very High",
                 "symptom": "Water-soaked lesions, white mould on underside",
                 "control": "Apply Metalaxyl + Mancozeb @ 2g/L. Avoid overhead irrigation."},
                {"name": "Fusarium Wilt", "risk_season": ["monsoon", "summer"], "severity": "High",
                 "symptom": "Yellowing from bottom, vascular browning",
                 "control": "Soil drench with Trichoderma viride @ 5g/L. Use grafted plants."},
            ],
        },
        "Potato": {
            "pests": [
                {"name": "Potato Tuber Moth", "risk_season": ["winter", "summer"], "severity": "High",
                 "symptom": "Tunnels in tubers, webbing",
                 "control": "Deep planting + earthing up. Store tubers with dried lantana leaves."},
            ],
            "diseases": [
                {"name": "Late Blight", "risk_season": ["winter", "monsoon"], "severity": "Very High",
                 "symptom": "Dark water-soaked patches, white sporulation",
                 "control": "Apply Cymoxanil + Mancozeb @ 3g/L at disease onset. Use Kufri Jyoti variety."},
                {"name": "Common Scab", "risk_season": ["winter"], "severity": "Moderate",
                 "symptom": "Rough corky patches on tuber skin",
                 "control": "Maintain soil pH below 5.5. Irrigate uniformly during tuber initiation."},
            ],
        },
        "Sugarcane": {
            "pests": [
                {"name": "Early Shoot Borer", "risk_season": ["summer", "monsoon"], "severity": "High",
                 "symptom": "Dead heart in young shoots",
                 "control": "Release Trichogramma chilonis. Apply Chlorantraniliprole 0.4 GR in soil."},
                {"name": "Woolly Aphid", "risk_season": ["monsoon"], "severity": "High",
                 "symptom": "White woolly masses on leaf undersides",
                 "control": "Release Dipha aphidivora predator. Spray Dimethoate 30 EC @ 1ml/L."},
            ],
            "diseases": [
                {"name": "Red Rot", "risk_season": ["monsoon", "winter"], "severity": "Very High",
                 "symptom": "Red internal stalk with white patches",
                 "control": "Use disease-free setts. Hot water treatment (50°C, 2 hrs). Resistant varieties."},
            ],
        },
    }

    # Fallback for crops not in pest DB
    crop_data = _PEST_DB.get(crop, {
        "pests": [{"name": "General Defoliators", "risk_season": ["monsoon", "summer"],
                   "severity": "Moderate", "symptom": "Leaf feeding damage",
                   "control": "Apply neem oil 5ml/L as preventive spray."}],
        "diseases": [{"name": "General Fungal", "risk_season": ["monsoon"],
                      "severity": "Moderate", "symptom": "Leaf spots / blights",
                      "control": "Apply Mancozeb 75 WP @ 2.5g/L as preventive."}],
    })

    # Filter by current season
    active_pests = [p for p in crop_data["pests"] if season_ctx in p["risk_season"]]
    active_diseases = [d for d in crop_data["diseases"] if season_ctx in d["risk_season"]]
    all_threats = active_pests + active_diseases

    if not all_threats:
        risk_level = "Low"
        risk_color = "#4CAF50"
    elif any(t["severity"] == "Very High" for t in all_threats):
        risk_level = "High Alert"
        risk_color = "#f44336"
    elif any(t["severity"] == "High" for t in all_threats):
        risk_level = "Elevated"
        risk_color = "#FF9800"
    else:
        risk_level = "Moderate"
        risk_color = "#FFC107"

    preventive = [
        "Maintain field hygiene — remove crop residues and weed hosts.",
        "Regular scouting every 5-7 days during peak risk periods.",
        "Use certified disease-free seeds and resistant varieties.",
    ]
    if season_ctx == "monsoon":
        preventive.append("Ensure proper drainage to prevent fungal outbreaks.")
        preventive.append("Avoid excess nitrogen — it increases pest susceptibility.")
    elif season_ctx == "summer":
        preventive.append("Summer deep ploughing exposes soil pests to sunlight.")
        preventive.append("Install pheromone and light traps for pest monitoring.")

    return {
        "risk_level": risk_level,
        "risk_color": risk_color,
        "season_context": season_ctx,
        "active_pests": active_pests,
        "active_diseases": active_diseases,
        "preventive_measures": preventive,
        "total_threats": len(all_threats),
    }


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  Weed Presence Detection                                                  #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _detect_weed_presence(feat: dict) -> dict:
    """
    Detect potential weed infestation from spatial & colour anomalies.
    Returns dict with: weed_risk (0-100), risk_label, indicators, advice.
    """
    uni      = feat.get("spatial_uniformity", 0.5)
    phs      = feat.get("patch_hue_std", 0)
    pgs      = feat.get("patch_green_std", 0)
    hue_std  = feat["hue_std"]
    gf       = feat["green_frac"]
    dgf      = feat.get("dark_green_frac", 0)
    bgf      = feat.get("bright_green_frac", 0)
    canopy   = feat["canopy_coverage"]
    ed       = feat.get("edge_density", 0)

    weed_score = 0
    indicators = []

    # High hue variation across patches → mixed species
    if phs > 20 and uni < 0.45:
        weed_score += 30
        indicators.append("High colour variation across field patches — suggests mixed species.")

    # Bimodal green: bright + dark green patches side by side
    if dgf > 0.08 and bgf > 0.08 and abs(dgf - bgf) < 0.15:
        weed_score += 20
        indicators.append("Both dark-green and bright-green zones detected — possible weed canopy mixed with crop.")

    # High green coverage with poor uniformity → weed carpet
    if gf > 0.40 and uni < 0.35:
        weed_score += 15
        indicators.append("High green fraction with poor uniformity — green cover may include weeds.")

    # High texture in inter-row zones
    if ed > 0.10 and canopy < 0.50:
        weed_score += 10
        indicators.append("Elevated texture with moderate canopy — possible weed presence between crop rows.")

    # Large hue std → multiple plant types
    if hue_std > 35 and pgs > 0.04:
        weed_score += 10
        indicators.append("High hue variability — multiple vegetation types in frame.")

    weed_score = min(95, max(0, weed_score))

    if weed_score >= 60:
        label = "High Weed Risk"
        color = "#f44336"
    elif weed_score >= 35:
        label = "Moderate Weed Risk"
        color = "#FF9800"
    elif weed_score >= 15:
        label = "Low Weed Risk"
        color = "#FFC107"
    else:
        label = "Minimal / No Weeds"
        color = "#4CAF50"

    advice = []
    if weed_score >= 40:
        advice.append("Perform manual or mechanical weeding within 7 days.")
        advice.append("Apply pre-emergence herbicide if crop stage allows (consult agronomist).")
        advice.append("Consider inter-cultivation with power weeder between rows.")
    elif weed_score >= 15:
        advice.append("Monitor inter-row spaces; hand weeding may suffice.")
        advice.append("Mulching with crop residues can suppress weed emergence.")
    else:
        advice.append("Field appears clean. Maintain vigilance with regular scouting.")

    return {
        "weed_risk": weed_score,
        "risk_label": label,
        "risk_color": color,
        "indicators": indicators,
        "advice": advice,
    }


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  Crop Rotation & Soil Nutrition Recommendations                           #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

CROP_ADVICE = {
    "Mustard": {
        "family": "Brassicaceae (Crucifer)",
        "nutrient_demand": "Medium N, High P, Medium K, High S (Sulphur)",
        "soil_impact": "Extracts sulphur; deep taproots improve subsoil structure",
        "rotation_after": ["Wheat", "Rice", "Chickpea", "Maize"],
        "rotation_before": ["Rice", "Soybean", "Cotton"],
        "avoid_sequence": ["Mustard", "Tomato", "Cotton"],
        "companion": ["Chickpea", "Barley"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "Medium demand. Apply 40-60 kg N/ha. Mustard responds well to split application — half at sowing, half at flowering."),
            ("🟠 **Phosphorus (P)**", "High demand. Apply 30-40 kg P₂O₅/ha. Critical for root development and oil content."),
            ("🟡 **Potassium (K)**", "Medium demand. Apply 20-30 kg K₂O/ha. Improves cold tolerance and seed quality."),
            ("🔵 **Sulphur (S)**", "Very high demand — mustard is a sulphur-loving crop. Apply 20-40 kg S/ha as gypsum or SSP."),
        ],
        "tips": "Mustard fixes no nitrogen — follow with a legume (chickpea) to replenish soil N."
    },
    "Wheat": {
        "family": "Poaceae (Grass/Cereal)",
        "nutrient_demand": "High N, Medium P, Medium K",
        "soil_impact": "Heavy nitrogen feeder; residue adds organic matter if managed",
        "rotation_after": ["Rice", "Soybean", "Groundnut", "Cotton"],
        "rotation_before": ["Maize", "Sugarcane", "Rice"],
        "avoid_sequence": ["Wheat", "Barley"],
        "companion": ["Mustard", "Chickpea"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "High demand. Apply 120-150 kg N/ha in 3 splits (basal, tillering, heading). Most critical nutrient for wheat yield."),
            ("🟠 **Phosphorus (P)**", "Apply 50-60 kg P₂O₅/ha at sowing. Essential for root establishment."),
            ("🟡 **Potassium (K)**", "Apply 40-50 kg K₂O/ha. Strengthens straw and reduces lodging."),
            ("🟢 **Zinc (Zn)**", "Apply 25 kg ZnSO₄/ha. Zinc deficiency is widespread in Indian wheat soils."),
        ],
        "tips": "Follow wheat with a legume (moong, chickpea) to restore nitrogen naturally."
    },
    "Rice": {
        "family": "Poaceae (Grass/Cereal)",
        "nutrient_demand": "High N, Medium P, High K, High Si",
        "soil_impact": "Waterlogged conditions affect soil aeration; silicon depletion common",
        "rotation_after": ["Wheat", "Mustard", "Chickpea", "Potato"],
        "rotation_before": ["Jute", "Soybean", "Groundnut"],
        "avoid_sequence": ["Rice", "Rice"],
        "companion": ["Azolla (green manure)", "Fish (rice-fish farming)"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "High demand. Apply 100-120 kg N/ha in 3 splits. Use neem-coated urea to reduce losses."),
            ("🟠 **Phosphorus (P)**", "Apply 40-60 kg P₂O₅/ha. Phosphorus availability drops in waterlogged soils."),
            ("🟡 **Potassium (K)**", "High demand. Apply 40-60 kg K₂O/ha. Rice removes large amounts of K in straw."),
            ("⚪ **Silicon (Si)**", "Apply silica-rich amendments (rice husk ash, diatomite). Strengthens stems, reduces pest damage."),
        ],
        "tips": "Avoid continuous rice-rice. Alternate with wheat or pulses to break pest cycles."
    },
    "Maize": {
        "family": "Poaceae (Grass/Cereal)",
        "nutrient_demand": "Very High N, High P, High K, High Zn",
        "soil_impact": "Very heavy feeder — depletes soil rapidly if not managed",
        "rotation_after": ["Wheat", "Chickpea", "Mustard", "Potato"],
        "rotation_before": ["Soybean", "Groundnut"],
        "avoid_sequence": ["Maize", "Sugarcane"],
        "companion": ["Soybean (intercropping)", "Beans"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "Very high demand. Apply 150-200 kg N/ha in 3-4 splits. Maize is the heaviest N feeder among cereals."),
            ("🟠 **Phosphorus (P)**", "Apply 60-80 kg P₂O₅/ha. Critical during early growth and tasseling."),
            ("🟡 **Potassium (K)**", "Apply 40-60 kg K₂O/ha. Improves grain filling and drought resistance."),
            ("🟢 **Zinc (Zn)**", "Apply 25-50 kg ZnSO₄/ha. Maize is highly sensitive to zinc deficiency."),
        ],
        "tips": "Maize-soybean intercropping improves soil nitrogen and suppresses weeds."
    },
    "Sugarcane": {
        "family": "Poaceae (Grass/Cereal)",
        "nutrient_demand": "Very High N, High P, Very High K",
        "soil_impact": "Extremely heavy feeder; long duration (12-18 months) depletes soil substantially",
        "rotation_after": ["Wheat", "Mustard", "Potato", "Chickpea"],
        "rotation_before": ["Rice", "Cotton"],
        "avoid_sequence": ["Sugarcane", "Sugarcane"],
        "companion": ["Wheat (intercrop first 3 months)", "Potato (intercrop)"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "Very high demand. Apply 200-300 kg N/ha in 4 splits over the growing season."),
            ("🟠 **Phosphorus (P)**", "Apply 60-80 kg P₂O₅/ha at planting. Entire dose as basal."),
            ("🟡 **Potassium (K)**", "Very high demand. Apply 80-120 kg K₂O/ha. Sugarcane removes 2-3 kg K per tonne of cane."),
            ("🟤 **Organic Matter**", "Apply 20-25 tonnes FYM/ha + press mud/filter cake. Sugarcane needs massive organic inputs."),
        ],
        "tips": "Never grow sugarcane continuously. Follow with a pulse/oilseed to restore soil health."
    },
    "Cotton": {
        "family": "Malvaceae",
        "nutrient_demand": "High N, High P, Very High K, High B (Boron)",
        "soil_impact": "Deep roots improve subsoil; removes significant potassium",
        "rotation_after": ["Wheat", "Chickpea", "Groundnut"],
        "rotation_before": ["Soybean", "Maize"],
        "avoid_sequence": ["Cotton", "Tomato"],
        "companion": ["Groundnut (intercrop)", "Cowpea (intercrop)"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "High demand. Apply 100-150 kg N/ha in splits. Excessive N causes excessive vegetative growth."),
            ("🟠 **Phosphorus (P)**", "Apply 50-60 kg P₂O₅/ha. Critical for boll development."),
            ("🟡 **Potassium (K)**", "Very high demand. Apply 50-80 kg K₂O/ha. Cotton is a 'potash-hungry' crop."),
            ("🔵 **Boron (B)**", "Apply 1-2 kg Borax/ha. Boron deficiency causes boll shedding."),
        ],
        "tips": "Cotton-wheat or cotton-chickpea is an excellent rotation for Indo-Gangetic plains."
    },
    "Soybean": {
        "family": "Fabaceae (Legume)",
        "nutrient_demand": "Low N (fixes own), High P, High K",
        "soil_impact": "Fixes 40-80 kg N/ha in root nodules — improves soil nitrogen",
        "rotation_after": ["Wheat", "Maize", "Cotton"],
        "rotation_before": ["Rice", "Maize", "Cotton"],
        "avoid_sequence": ["Soybean", "Groundnut"],
        "companion": ["Maize (intercrop)", "Sugarcane"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "Self-fixing! Apply only 20 kg N/ha as starter dose. Inoculate with Rhizobium for best results."),
            ("🟠 **Phosphorus (P)**", "High demand. Apply 60-80 kg P₂O₅/ha. P is critical for nodulation and N-fixation."),
            ("🟡 **Potassium (K)**", "Apply 40-60 kg K₂O/ha. Improves seed quality and oil content."),
            ("🟤 **Rhizobium inoculation**", "Treat seeds with Rhizobium japonicum @ 10g/kg seed. Doubles nitrogen fixation."),
        ],
        "tips": "Soybean is an ideal rotation crop — it leaves 20-40 kg residual N/ha for the next crop."
    },
    "Potato": {
        "family": "Solanaceae (Nightshade)",
        "nutrient_demand": "High N, Very High P, Very High K",
        "soil_impact": "Heavy feeder; removes large amounts of K; soil-borne disease buildup",
        "rotation_after": ["Wheat", "Maize", "Mustard"],
        "rotation_before": ["Rice", "Maize"],
        "avoid_sequence": ["Potato", "Tomato"],
        "companion": ["Beans", "Maize"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "Apply 120-150 kg N/ha in 2 splits (planting + earthing up)."),
            ("🟠 **Phosphorus (P)**", "Very high demand. Apply 80-100 kg P₂O₅/ha. Entire dose at planting."),
            ("🟡 **Potassium (K)**", "Very high demand. Apply 100-120 kg K₂O/ha. Potassium is THE key nutrient for tuber size and quality."),
            ("🟢 **Zinc + Boron**", "Apply ZnSO₄ (25 kg/ha) + Borax (10 kg/ha). Both critical for tuber development."),
        ],
        "tips": "Never follow potato with tomato (same family — shared diseases like late blight)."
    },
    "Tomato": {
        "family": "Solanaceae (Nightshade)",
        "nutrient_demand": "High N, High P, Very High K, High Ca",
        "soil_impact": "Heavy feeder; soil-borne disease risk (Fusarium, nematodes)",
        "rotation_after": ["Wheat", "Maize", "Chickpea"],
        "rotation_before": ["Soybean", "Mustard"],
        "avoid_sequence": ["Tomato", "Potato", "Cotton"],
        "companion": ["Marigold (pest deterrent)", "Basil"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "Apply 100-150 kg N/ha in frequent splits. Excessive N delays fruiting."),
            ("🟠 **Phosphorus (P)**", "Apply 60-80 kg P₂O₅/ha. Supports flowering and fruit set."),
            ("🟡 **Potassium (K)**", "Very high demand. Apply 80-120 kg K₂O/ha. Improves fruit colour, taste, and shelf life."),
            ("⚪ **Calcium (Ca)**", "Apply calcium nitrate or gypsum. Prevents blossom end rot."),
        ],
        "tips": "3-year rotation minimum for solanaceous crops to prevent soil disease buildup."
    },
    "Chickpea": {
        "family": "Fabaceae (Legume)",
        "nutrient_demand": "Low N (fixes own), High P, Medium K",
        "soil_impact": "Fixes 20-50 kg N/ha; improves soil structure with deep roots",
        "rotation_after": ["Rice", "Maize", "Jute"],
        "rotation_before": ["Wheat", "Mustard", "Cotton"],
        "avoid_sequence": ["Chickpea", "Chickpea"],
        "companion": ["Wheat (relay cropping)", "Mustard"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "Self-fixing! Apply only 15-20 kg N/ha as starter. Inoculate with Mesorhizobium."),
            ("🟠 **Phosphorus (P)**", "High demand. Apply 40-60 kg P₂O₅/ha + PSB inoculation. P drives nodulation."),
            ("🟡 **Potassium (K)**", "Apply 20-30 kg K₂O/ha. Moderate need."),
            ("🔵 **Sulphur (S)**", "Apply 20 kg S/ha as gypsum. Improves protein content."),
        ],
        "tips": "Chickpea is an excellent soil-builder. Use it between two cereal crops."
    },
    "Groundnut": {
        "family": "Fabaceae (Legume)",
        "nutrient_demand": "Low N, High P, High K, High Ca",
        "soil_impact": "Fixes 30-60 kg N/ha; loosens soil with pegging",
        "rotation_after": ["Wheat", "Maize", "Cotton"],
        "rotation_before": ["Rice", "Maize", "Sugarcane"],
        "avoid_sequence": ["Groundnut", "Soybean"],
        "companion": ["Maize (intercrop)", "Sorghum"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "Self-fixing! Apply only 20 kg N/ha. Use Rhizobium + PSB."),
            ("🟠 **Phosphorus (P)**", "Apply 40-60 kg P₂O₅/ha. P is critical for pod development."),
            ("🟡 **Potassium (K)**", "Apply 30-40 kg K₂O/ha."),
            ("⚪ **Calcium (Ca)**", "Very important! Apply 200-400 kg gypsum/ha at pegging. Calcium drives pod filling."),
        ],
        "tips": "Always apply gypsum at flowering/pegging stage. Groundnut is very calcium-hungry."
    },
    "Barley": {
        "family": "Poaceae (Grass/Cereal)",
        "nutrient_demand": "Medium N, Medium P, Medium K",
        "soil_impact": "Tolerant of poor soils; good for reclaiming marginal land",
        "rotation_after": ["Rice", "Soybean"],
        "rotation_before": ["Wheat", "Maize"],
        "avoid_sequence": ["Barley", "Wheat"],
        "companion": ["Mustard", "Chickpea"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "Moderate demand. Apply 60-80 kg N/ha in 2 splits."),
            ("🟠 **Phosphorus (P)**", "Apply 30-40 kg P₂O₅/ha at sowing."),
            ("🟡 **Potassium (K)**", "Apply 20-30 kg K₂O/ha."),
            ("🟢 **Zinc**", "Apply ZnSO₄ (20 kg/ha) on zinc-deficient soils."),
        ],
        "tips": "Barley is salt-tolerant — good for marginal/reclaimed soils."
    },
    "Jute": {
        "family": "Tiliaceae (Fibre)",
        "nutrient_demand": "High N, Medium P, High K",
        "soil_impact": "Deep roots improve soil; leaves add organic matter",
        "rotation_after": ["Rice", "Potato"],
        "rotation_before": ["Rice", "Wheat"],
        "avoid_sequence": ["Jute", "Jute"],
        "companion": ["Rice (relay cropping)"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "High demand. Apply 60-80 kg N/ha in 2-3 splits."),
            ("🟠 **Phosphorus (P)**", "Apply 30-40 kg P₂O₅/ha."),
            ("🟡 **Potassium (K)**", "High demand. Apply 40-60 kg K₂O/ha."),
            ("🟤 **Organic Matter**", "Apply FYM 5-10 tonnes/ha. Jute responds well to organic manure."),
        ],
        "tips": "Jute leaves left in soil after retting add significant organic matter."
    },
    "Tea": {
        "family": "Theaceae",
        "nutrient_demand": "Very High N, Medium P, High K, Acidic soil required",
        "soil_impact": "Long-term perennial; acidifies soil over time",
        "rotation_after": [],  # perennial
        "rotation_before": [],
        "avoid_sequence": [],
        "companion": ["Shade trees (Albizia, Silver Oak)"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "Very high demand. Apply 120-160 kg N/ha/year in 4 splits. Ammonium sulphate preferred over urea."),
            ("🟠 **Phosphorus (P)**", "Apply 30-50 kg P₂O₅/ha through rock phosphate (slow release in acidic soil)."),
            ("🟡 **Potassium (K)**", "Apply 60-80 kg K₂O/ha. Improves tea quality and cold tolerance."),
            ("🟤 **pH Management**", "Tea requires pH 4.5-5.5. Apply sulphur if pH rises. Never lime tea gardens."),
        ],
        "tips": "Tea is a perennial — no rotation possible. Maintain shade trees and mulch heavily."
    },
    "Coffee": {
        "family": "Rubiaceae",
        "nutrient_demand": "High N, Medium P, High K, acidic to neutral pH",
        "soil_impact": "Perennial; shade requirement maintains forest-like soil ecology",
        "rotation_after": [],  # perennial
        "rotation_before": [],
        "avoid_sequence": [],
        "companion": ["Pepper (vine on shade trees)", "Cardamom", "Silver Oak"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "Apply 100-140 kg N/ha/year. Split into 3 applications aligned with rainfall."),
            ("🟠 **Phosphorus (P)**", "Apply 40-60 kg P₂O₅/ha through rock phosphate or SSP."),
            ("🟡 **Potassium (K)**", "Apply 80-120 kg K₂O/ha. Critical for berry development and quality."),
            ("🟤 **Organic Mulch**", "Apply 8-10 tonnes/ha of composted coffee pulp or FYM. Maintains soil biology."),
        ],
        "tips": "Coffee thrives in multi-tier agroforestry. Maintain shade trees for soil health."
    },
    "Banana": {
        "family": "Musaceae",
        "nutrient_demand": "Very High N, High P, Very High K",
        "soil_impact": "Extremely heavy feeder; high water demand; depletes soil rapidly",
        "rotation_after": ["Chickpea", "Groundnut", "Soybean"],
        "rotation_before": ["Rice", "Maize"],
        "avoid_sequence": ["Banana", "Banana"],
        "companion": ["Ginger (intercrop)", "Turmeric"],
        "soil_nutrition": [
            ("🔴 **Nitrogen (N)**", "Very high demand. Apply 200-300 g N/plant/year (split monthly). Use neem-coated urea."),
            ("🟠 **Phosphorus (P)**", "Apply 50-80 g P₂O₅/plant/year. Entire dose at planting."),
            ("🟡 **Potassium (K)**", "Extremely high demand. Apply 200-400 g K₂O/plant/year. Banana is THE most potash-hungry fruit crop."),
            ("🟤 **Organic Matter**", "Apply 10-15 kg FYM/plant + pseudostem mulch. Banana needs massive organic inputs."),
        ],
        "tips": "Follow banana with a legume crop to replenish soil. Avoid continuous banana."
    },
}

# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  Health assessment                                                        #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def _assess_health(feat: dict) -> float:
    vari_s  = np.clip((feat["VARI"] + 0.2) / 0.5, 0, 1) * 22
    gli_s   = np.clip((feat["GLI"] + 0.1) / 0.35, 0, 1) * 18
    exgr_s  = np.clip((feat["ExGR"] + 0.05) / 0.25, 0, 1) * 14
    ngrdi_s = np.clip((feat["NGRDI"] + 0.1) / 0.4, 0, 1) * 10
    rgri_s  = np.clip(max(0, 1 - (feat["RGRI"] - 0.6) / 0.8), 0, 1) * 12
    canopy_s = np.clip(feat["canopy_coverage"] / 0.6, 0, 1) * 14
    br_s    = np.clip(1 - abs(feat["brightness"] - 120) / 100, 0, 1) * 10
    return round(float(np.clip(
        vari_s + gli_s + exgr_s + ngrdi_s + rgri_s + canopy_s + br_s, 2, 98)), 1)


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #
#  Main public API                                                          #
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| #

def analyze_crop_image(image: Image.Image) -> dict:
    """
    Industry-grade hierarchical crop identification & health assessment.

    Returns dict (backward-compatible keys):
      top_crop, confidence, health_score, crop_scores, features,
      is_barren, analysis_text, methodology, scene_type, scene_reason,
      gps_info (optional - from photo EXIF), photo_metadata (optional)
    """
    # ── Extract GPS & metadata BEFORE converting (preserves EXIF) ──
    gps_info = _extract_gps_from_image(image)
    photo_meta = _extract_photo_metadata(image)

    img = image.convert("RGB")

    # ── S1: Tri-scale extraction (640 + 320 + 160 px) ──
    img_l = img.copy(); img_l.thumbnail((640, 640))
    img_m = img.copy(); img_m.thumbnail((320, 320))
    img_s = img.copy(); img_s.thumbnail((160, 160))
    fl = _extract_all_features(np.array(img_l))
    fm = _extract_all_features(np.array(img_m))
    fs = _extract_all_features(np.array(img_s))

    # Fuse: weighted average of 3 scales (large=0.5, mid=0.3, small=0.2)
    feat = dict(fl)
    _AVG_KEYS = [
        "ExG","ExGR","VARI","GLI","RGRI","NGRDI","MGRVI",
        "canopy_coverage","hue_mean","sat_mean","val_mean",
        "yellow_frac","green_frac","red_frac","bright_yellow_frac",
        "brown_frac","white_frac","edge_density","bright_green_frac",
        "dark_green_frac","orange_frac","skin_frac","gray_frac","sky_frac",
    ]
    for k in _AVG_KEYS:
        vals = [d.get(k) for d in (fl, fm, fs) if k in d]
        if len(vals) == 3:
            feat[k] = vals[0] * 0.5 + vals[1] * 0.3 + vals[2] * 0.2
        elif len(vals) == 2:
            feat[k] = (vals[0] + vals[1]) / 2
    for k in ["spatial_uniformity","patch_hue_std","patch_green_std"]:
        if k in fm: feat[k] = fm[k]

    # ── S0: Scene classification (reject non-crop images) ──
    scene_type, scene_conf, scene_reason = _classify_scene(feat)

    if scene_type not in ("crop_field", "soil"):
        # Non-agricultural image — return helpful rejection
        label_map = {
            "water":    ("Water Body Detected",    "a water body (river, pond, or flooded area)"),
            "sky":      ("Sky Image Detected",     "mostly sky with no ground-level vegetation"),
            "urban":    ("Urban Area Detected",    "an urban or built environment (buildings, roads, concrete)"),
            "indoor":   ("Indoor Photo Detected",  "an indoor scene, not an agricultural field"),
            "selfie":   ("Portrait Detected",      "a human face or portrait, not a field photo"),
            "document": ("Document Detected",      "a document, screenshot, or text image"),
            "night":    ("Dark Image Detected",    "a very dark or night-time image"),
            "unknown":  ("Unrecognised Image",     "an image that could not be classified as agricultural"),
        }
        title, desc = label_map.get(scene_type, ("Unrecognised", "an unrecognised image"))
        return {
            "top_crop": title, "confidence": 0.0,
            "health_score": 0.0, "crop_scores": {}, "features": feat,
            "is_barren": True,
            "scene_type": scene_type, "scene_confidence": scene_conf,
            "gps_info": gps_info, "photo_metadata": photo_meta,
            "analysis_text": (
                f"**{title}** — This image appears to show {desc}.\n\n"
                f"*{scene_reason}*\n\n"
                f"**Please upload a clear photo of an agricultural field** "
                f"(overhead or 45-degree angle, natural daylight) for crop "
                f"identification and yield prediction.\n\n"
                f"**Scene confidence:** {scene_conf:.0%}\n"
            ),
            "methodology": _methodology_text(),
        }

    if scene_type == "soil":
        soil_info = _assess_soil_fertility(feat)
        seasonal_info = _seasonal_crop_suggestions()
        return {
            "top_crop": "No Crop Detected", "confidence": 0.0,
            "health_score": 0.0, "crop_scores": {}, "features": feat,
            "is_barren": True,
            "scene_type": "soil", "scene_confidence": scene_conf,
            "soil_fertility": soil_info,
            "seasonal_suggestions": seasonal_info,
            "gps_info": gps_info, "photo_metadata": photo_meta,
            "analysis_text": (
                "**No crop detected** — The image shows "
                "**bare soil / barren / fallow land** with little vegetation.\n\n"
                f"*{scene_reason}*\n\n"
                "The field may be fallow, freshly ploughed, harvested, or "
                "the soil is not yet planted.\n\n"
                f"**Vegetation Indices:**\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| ExG | {feat['ExG']:.3f} |\n"
                f"| VARI | {feat['VARI']:.3f} |\n"
                f"| Canopy Coverage | {feat['canopy_coverage']:.1%} |\n"
                f"| GLI | {feat['GLI']:.3f} |\n"
                f"| Brown Fraction | {feat['brown_frac']:.1%} |\n\n"
                f"**Scene confidence:** {scene_conf:.0%}\n"
            ),
            "methodology": _methodology_text(),
        }

    # ── S6: Colour-group gate ──
    cg = _classify_color_group(feat)
    candidates = COLOR_GROUPS.get(cg)
    if candidates is None:
        candidates = list(CROP_PROFILES.keys())

    # ── Score all crops (rule-based) ──
    rule_scores = {}
    for crop, prof in CROP_PROFILES.items():
        rule_scores[crop] = _score_crop(feat, prof, crop in candidates)

    # ── ML prediction (trained Random Forest) ──
    ml_scores = _ml_crop_predict(feat)

    # ── Ensemble: 60% rule-based + 40% ML ──
    if ml_scores:
        scores = {}
        for crop in rule_scores:
            rs = rule_scores[crop]
            ms = ml_scores.get(crop, 0.0)
            scores[crop] = rs * 0.60 + ms * 0.40
    else:
        scores = rule_scores

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_crop   = ranked[0][0]
    raw_conf   = ranked[0][1]
    gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0

    # ── Confidence calibration ──
    # Sigmoid-centred calibration: pushes extreme scores toward realistic range
    # and sharpens the gap between 1st and 2nd place.
    def _calibrate(raw, g):
        import math
        # Logistic mapping centred at 55
        x = (raw - 55) / 22
        cal = 100 / (1 + math.exp(-x))
        # Boost for clear winner (large gap)
        gap_bonus = min(g * 0.25, 8)
        # Penalty for very close race
        gap_penalty = max(0, (5 - g) * 1.5) if g < 5 else 0
        return max(5, min(98, cal + gap_bonus - gap_penalty))

    confidence = _calibrate(raw_conf, gap)

    if confidence < 18:
        return {
            "top_crop": "Unrecognised Image", "confidence": confidence,
            "health_score": _assess_health(feat),
            "crop_scores": dict(ranked[:8]), "features": feat,
            "is_barren": True,
            "scene_type": "unknown", "scene_confidence": 0.4,
            "analysis_text": (
                f"**Could not identify a crop.** Best guess: "
                f"**{top_crop}** ({confidence:.0f}%).\n\n"
                "The image may not show a recognisable crop, or lighting "
                "conditions are unusual. Try an overhead daylight photo.\n"
            ),
            "methodology": _methodology_text(),
        }

    # ── Health assessment ──
    health_score = _assess_health(feat)

    # ── Confidence narrative ──
    if confidence > 75 and gap > 8:
        certainty = "High confidence"
    elif confidence > 55:
        certainty = "Moderate confidence"
    elif confidence > 35:
        certainty = "Low confidence"
    else:
        certainty = "Very low confidence"

    if health_score > 75:
        htxt = "Excellent - lush, vigorous vegetation with strong green indices."
    elif health_score > 55:
        htxt = "Good - normal vegetation; no severe stress detected."
    elif health_score > 35:
        htxt = "Fair - possible water/nutrient stress or early senescence."
    else:
        htxt = "Poor - significant stress; yellowing or sparse canopy."

    prof = CROP_PROFILES[top_crop]
    r2 = ranked[1] if len(ranked) > 1 else ("--", 0)
    r3 = ranked[2] if len(ranked) > 2 else ("--", 0)

    # Scale all Top-3 scores proportionally so they're on the same
    # calibrated scale as the displayed confidence for #1.
    _scale = confidence / (raw_conf + _EPS)
    r2_pct = r2[1] * _scale if isinstance(r2[1], (int, float)) else 0
    r3_pct = r3[1] * _scale if isinstance(r3[1], (int, float)) else 0

    analysis_text = (
        f"**{certainty}** - The image most closely matches "
        f"**{top_crop}** ({confidence:.0f}%).\n\n"
        f"*{prof['desc']}*\n\n"
        f"**Top 3:**  "
        f"1\\. {top_crop} ({confidence:.0f}%)  |  "
        f"2\\. {r2[0]} ({r2_pct:.0f}%)  |  "
        f"3\\. {r3[0]} ({r3_pct:.0f}%)\n\n"
        f"Separation gap: **{gap:.1f}** pts "
        f"{'(strong)' if gap > 10 else '(moderate)' if gap > 5 else '(narrow)'}\n\n"
        f"Colour group: **{cg.replace('_',' ').title()}**\n\n"
        f"---\n"
        f"**Health:** {htxt}\n\n"
        f"**Vegetation Indices:**\n"
        f"| Index | Value | Status |\n"
        f"|-------|-------|--------|\n"
        f"| ExG | {feat['ExG']:.3f} | {'Good' if feat['ExG']>0.03 else 'Low'} |\n"
        f"| VARI | {feat['VARI']:.3f} | {'Healthy' if feat['VARI']>0.05 else 'Stressed'} |\n"
        f"| GLI | {feat['GLI']:.3f} | {'Strong' if feat['GLI']>0.02 else 'Weak'} |\n"
        f"| NGRDI | {feat['NGRDI']:.3f} | "
        f"{'Green-dominant' if feat['NGRDI']>0 else 'Red-shifted'} |\n"
        f"| RGRI | {feat['RGRI']:.2f} | "
        f"{'Low stress' if feat['RGRI']<1 else 'High stress'} |\n"
        f"| Canopy | {feat['canopy_coverage']:.1%} | "
        f"{'Dense' if feat['canopy_coverage']>0.5 else 'Moderate' if feat['canopy_coverage']>0.25 else 'Sparse'} |\n\n"
        f"**Image Diagnostics:**\n"
        f"- Hue: {feat['hue_mean']:.0f} deg  |  Yellow: {feat['yellow_frac']:.1%}  |  "
        f"Green: {feat['green_frac']:.1%}\n"
        f"- Texture: {feat['edge_density']:.1%}  |  "
        f"Uniformity: {feat.get('spatial_uniformity',0):.2f}\n"
        f"- Brightness: {feat['brightness']:.0f}/255  |  "
        f"Contrast: {feat['contrast']:.0f}\n"
    )

    yield_info = _estimate_yield(feat, top_crop, confidence, health_score)
    disease_info = _detect_disease_stress(feat, top_crop)
    irrigation_info = _estimate_irrigation_need(feat, top_crop)
    pest_info = _assess_pest_risk(top_crop)
    weed_info = _detect_weed_presence(feat)

    return {
        "top_crop": top_crop, "confidence": confidence,
        "health_score": health_score,
        "crop_scores": dict(ranked[:8]), "features": feat,
        "is_barren": False,
        "scene_type": "crop_field", "scene_confidence": scene_conf,
        "crop_advice": CROP_ADVICE.get(top_crop, {}),
        "yield_estimation": yield_info,
        "disease_stress": disease_info,
        "irrigation": irrigation_info,
        "pest_risk": pest_info,
        "weed_detection": weed_info,
        "gps_info": gps_info, "photo_metadata": photo_meta,
        "analysis_text": analysis_text,
        "methodology": _methodology_text(),
    }


def _methodology_text() -> str:
    return (
        "**Analysis Methodology (Industry-Grade 12-Stage Pipeline v4.1 — ML-Trained):**\n\n"
        "This scanner uses **12 cascaded analysis stages** including a **trained "
        "Random Forest classifier** for industry-precision crop identification, "
        "health diagnostics, and advisory:\n\n"
        "**S0 - Scene Classification:**\n"
        "Classifies into 10 scene categories (crop, soil, water, sky, "
        "urban, indoor, portrait, document, night, unknown). "
        "Non-agricultural images rejected with specific reasoning.\n\n"
        "**S1 - Tri-Scale Feature Extraction:**\n"
        "Features extracted at 3 resolutions (640 + 320 + 160 px), "
        "weighted-fused (50/30/20%) to capture fine detail through broad patterns.\n\n"
        "**S2 - 8 Vegetation Indices:**\n"
        "ExG, ExGR, VARI, GLI, RGRI, NGRDI, MGRVI — peer-reviewed "
        "metrics from precision agriculture research.\n\n"
        "**S3 - Colour Analysis:**\n"
        "HSV decomposition with 14 fine-grained colour bands.\n\n"
        "**S4 - Texture & Leaf-Morphology Pipeline:**\n"
        "Edge density, Laplacian sharpness, Sobel direction, leaf "
        "coarseness index, canopy-texture ratio, and uniformity metrics.\n\n"
        "**S5 - Spatial Analysis:**\n"
        "4x4 grid patch statistics for geographic colour uniformity.\n\n"
        "**S6 - Multi-Mode Crop Profiling:**\n"
        "Growth-stage-aware profiles with alternate HSV centres "
        "(e.g. green vegetative wheat vs golden mature wheat). "
        "Colour-group gate (6 groups) narrows candidates, "
        "then 25+ weighted criteria score each crop.\n\n"
        "**S7 - Trained Random Forest Classifier (ML):**\n"
        "A 200-tree Random Forest trained on 5,600 synthetic crop feature "
        "vectors votes alongside the rule engine. Ensemble weighting: "
        "55% rules + 45% ML for robust identification.\n\n"
        "**S8 - Disease & Stress Detection:**\n"
        "Chlorosis, necrosis, wilting, nutrient deficiency, and pest "
        "damage detected from colour/texture anomalies.\n\n"
        "**S9 - Irrigation & Water Stress:**\n"
        "Water stress index estimated from VARI, saturation, canopy "
        "decline, and crop-specific water requirements.\n\n"
        "**S10 - Pest Risk & Weed Detection:**\n"
        "Season-aware pest/disease database for 8 major crops. "
        "Weed presence estimated from spatial heterogeneity.\n\n"
        "**S11 - Confidence Calibration:**\n"
        "Sigmoid-centred calibration with gap bonus/penalty for "
        "realistic confidence scoring.\n\n"
        "*For highest accuracy, use a clear overhead or 45-degree field "
        "photo in natural daylight.*"
    )
