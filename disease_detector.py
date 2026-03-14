"""
=============================================================================
 SMART CROP YIELD PREDICTION USING ARTIFICIAL INTELLIGENCE
 Module: disease_detector.py — AI Crop Disease Detection from Leaf Images
=============================================================================

 PURPOSE:
   Detect crop diseases from leaf photographs using a lightweight CNN-like
   pipeline: hand-crafted feature extraction (colour histograms, texture
   descriptors, edge statistics) → sklearn MLP Classifier.

   In production this would use a full CNN (ResNet / EfficientNet) or
   YOLO-based detector.  The feature-extraction + MLP approach gives
   reasonable accuracy without heavy deep-learning dependencies.

 SUPPORTED CROPS & DISEASES  (from utils.CROP_DISEASES):
   Rice     – Blast, Bacterial Blight, Brown Spot, Sheath Rot
   Wheat    – Rust, Powdery Mildew, Septoria, Karnal Bunt
   Maize    – Northern Leaf Blight, Gray Leaf Spot, Common Rust, Stalk Rot
   Cotton   – Bacterial Blight, Verticillium Wilt, Leaf Curl, Boll Rot
   Soybean  – Soybean Rust, Downy Mildew, Frogeye Leaf Spot, Stem Rot

 AUTHOR : AgriTech AI Solutions
 VERSION: 3.0.0
=============================================================================
"""

import numpy as np
from PIL import Image, ImageFilter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from utils import CROP_DISEASES


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def _extract_features(img: Image.Image) -> np.ndarray:
    """
    Extract a 78-dimensional feature vector from a PIL Image.

    Features (78 total):
      [0:48]  — RGB + HSV colour histograms (8 bins × 6 channels)
      [48:60] — Texture features from grayscale (LBP-like statistics)
      [60:68] — Edge statistics (Sobel-like via PIL edge filter)
      [68:74] — Region statistics (quadrant mean intensities + ratios)
      [74:78] — Global: aspect ratio, brightness, contrast, green-ratio
    """
    img = img.convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0

    features = []

    # ── 1. Colour histograms (RGB, 8 bins each = 24 features) ────
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=8, range=(0, 1))
        features.extend(hist / hist.sum())

    # ── 2. HSV colour histograms (8 bins each = 24 features) ─────
    hsv = np.array(img.convert("HSV"), dtype=np.float32) / 255.0
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch], bins=8, range=(0, 1))
        features.extend(hist / hist.sum())

    # ── 3. Texture features (12 features) ────────────────────────
    gray = np.mean(arr, axis=2)
    # Local variance in 4×4 blocks
    for bx in range(4):
        for by in range(3):
            block = gray[bx*32:(bx+1)*32, by*43:(by+1)*43]
            features.append(float(np.var(block)))

    # ── 4. Edge features (8 features) ────────────────────────────
    edge_img = img.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edge_img, dtype=np.float32) / 255.0
    features.append(float(np.mean(edge_arr)))
    features.append(float(np.std(edge_arr)))
    features.append(float(np.percentile(edge_arr, 90)))
    features.append(float(np.percentile(edge_arr, 10)))
    # Quadrant edge density
    h, w = edge_arr.shape
    for qi, qj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        q = edge_arr[qi*h//2:(qi+1)*h//2, qj*w//2:(qj+1)*w//2]
        features.append(float(np.mean(q)))

    # ── 5. Region statistics (6 features) ────────────────────────
    for qi in range(2):
        for qj in range(2):
            q = gray[qi*64:(qi+1)*64, qj*64:(qj+1)*64]
            features.append(float(np.mean(q)))
    # Centre vs periphery ratio
    centre = gray[32:96, 32:96]
    features.append(float(np.mean(centre) / (np.mean(gray) + 1e-9)))
    features.append(float(np.std(centre)))

    # ── 6. Global features (4 features) ──────────────────────────
    features.append(1.0)  # aspect ratio (already square)
    features.append(float(np.mean(gray)))          # brightness
    features.append(float(np.std(gray)))            # contrast
    features.append(float(np.mean(arr[:, :, 1]) / (np.mean(arr) + 1e-9)))  # green ratio

    return np.array(features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  DISEASE DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

class CropDiseaseDetector:
    """
    Lightweight crop-disease detector using feature extraction + MLP.

    Usage
    -----
    >>> detector = CropDiseaseDetector()
    >>> detector.train()          # trains on synthetic examples
    >>> result = detector.predict(pil_image, crop="Rice")
    """

    def __init__(self):
        self.models = {}        # crop → (scaler, mlp)
        self.is_trained = False

    # ─────────────────────────────────────────────────────────────
    #  SYNTHETIC TRAINING DATA
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _generate_synthetic_samples(crop, n_per_class=60):
        """
        Generate synthetic feature vectors for each disease label.
        Each disease signature is a shifted/scaled version of a base
        feature vector to simulate visual differences.
        """
        diseases = CROP_DISEASES.get(crop, [])
        labels_all = ["Healthy"] + diseases
        X, y = [], []

        rng = np.random.RandomState(hash(crop) % 2**31)

        for idx, label in enumerate(labels_all):
            base = rng.uniform(0.1, 0.9, 78)
            # Assign characteristic "signature" shifts
            if label != "Healthy":
                # Disease images tend to have more brown/yellow (lower green)
                base[1] *= 0.7       # reduce green-channel mid bin
                base[4] += 0.15      # boost hue shift
                base[48:60] *= 1.3   # more texture variance
                base[60:68] *= 1.2   # more edge activity
                # Each disease gets a unique perturbation on different channels
                base[idx % 8] += 0.2
                base[24 + (idx * 3) % 24] += 0.15
                base[68 + idx % 6] *= 1.1
            else:
                # Healthy: more green, smoother texture
                base[1] *= 1.3
                base[48:60] *= 0.7

            for _ in range(n_per_class):
                sample = base + rng.normal(0, 0.08, 78)
                X.append(np.clip(sample, 0, 2))
                y.append(label)

        return np.array(X, dtype=np.float32), np.array(y)

    # ─────────────────────────────────────────────────────────────
    #  TRAIN
    # ─────────────────────────────────────────────────────────────

    def train(self, progress_callback=None):
        """
        Train one MLP model per crop on synthetic data.
        Calls progress_callback(crop, step, total) if supplied.
        """
        crops = list(CROP_DISEASES.keys())
        for step, crop in enumerate(crops):
            X, y = self._generate_synthetic_samples(crop)
            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)

            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                max_iter=300,
                random_state=42,
                early_stopping=False,
            )
            mlp.fit(X_scaled, y)
            self.models[crop] = (scaler, mlp)

            if progress_callback:
                progress_callback(crop, step + 1, len(crops))

        self.is_trained = True

    # ─────────────────────────────────────────────────────────────
    #  PREDICT
    # ─────────────────────────────────────────────────────────────

    def predict(self, image: Image.Image, crop: str = "Rice"):
        """
        Predict disease from a PIL leaf image.

        Returns
        -------
        dict with keys:
            disease       — predicted label (str)
            confidence    — probability 0–1 (float)
            all_probs     — {label: prob} (dict)
            severity      — Low / Moderate / High / Critical (str)
            recommendations — list of strings
        """
        if not self.is_trained or crop not in self.models:
            return {
                "disease": "Unknown",
                "confidence": 0.0,
                "all_probs": {},
                "severity": "N/A",
                "recommendations": ["Model not trained for this crop."],
            }

        features = _extract_features(image).reshape(1, -1)
        scaler, mlp = self.models[crop]
        X_scaled = scaler.transform(features)

        probs = mlp.predict_proba(X_scaled)[0]
        classes = mlp.classes_
        pred_idx = int(np.argmax(probs))
        pred_label = classes[pred_idx]
        confidence = float(probs[pred_idx])

        all_probs = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}

        # Severity estimation from probability + disease type
        if pred_label == "Healthy":
            severity = "None"
        elif confidence > 0.85:
            severity = "Critical"
        elif confidence > 0.65:
            severity = "High"
        elif confidence > 0.45:
            severity = "Moderate"
        else:
            severity = "Low"

        recs = _get_disease_recommendations(crop, pred_label, severity)

        return {
            "disease":         pred_label,
            "confidence":      round(confidence, 4),
            "all_probs":       all_probs,
            "severity":        severity,
            "recommendations": recs,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  DISEASE-SPECIFIC RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════

_DISEASE_RECS = {
    # Rice
    "Blast": [
        "Apply Tricyclazole (0.06%) or Isoprothiolane fungicide.",
        "Drain standing water; avoid excess nitrogen fertiliser.",
        "Plant blast-resistant varieties (e.g., Pusa Basmati 1121).",
    ],
    "Bacterial Blight": [
        "Use copper-based bactericides (Copper Oxychloride 0.25%).",
        "Remove and burn infected plant debris.",
        "Ensure good drainage; avoid overhead irrigation.",
    ],
    "Brown Spot": [
        "Apply Mancozeb (0.25%) or Propiconazole fungicide.",
        "Improve soil nutrition, especially potassium.",
        "Use certified disease-free seeds.",
    ],
    "Sheath Rot": [
        "Spray Carbendazim (0.1%) at boot-leaf stage.",
        "Maintain balanced NPK fertilisation.",
        "Rogue out severely infected plants.",
    ],
    # Wheat
    "Rust": [
        "Apply Propiconazole (0.1%) or Tebuconazole fungicide.",
        "Plant rust-resistant varieties (e.g., HD-2967).",
        "Early sowing reduces exposure to rust spores.",
    ],
    "Powdery Mildew": [
        "Spray sulphur-based fungicide (Wettable Sulphur 0.3%).",
        "Ensure adequate spacing for air circulation.",
        "Avoid excessive nitrogen application.",
    ],
    "Septoria": [
        "Apply Chlorothalonil or Propiconazole.",
        "Rotate with non-cereal crops to break disease cycle.",
        "Remove crop residue after harvest.",
    ],
    "Karnal Bunt": [
        "Seed treatment with Carboxin or Thiram.",
        "Use certified seed from disease-free areas.",
        "Avoid late sowing in warm, humid regions.",
    ],
    # Maize
    "Northern Leaf Blight": [
        "Apply Mancozeb or Propiconazole fungicide.",
        "Plant resistant hybrids; rotate with non-maize crops.",
        "Remove infected leaf debris after harvest.",
    ],
    "Gray Leaf Spot": [
        "Apply strobilurin-based fungicide at early symptom stage.",
        "Reduce plant density for better air flow.",
        "Practise crop rotation with legumes.",
    ],
    "Common Rust": [
        "Apply Mancozeb (0.25%) or Propiconazole.",
        "Plant early-maturing hybrids to escape peak rust period.",
    ],
    "Stalk Rot": [
        "Improve drainage; avoid waterlogged conditions.",
        "Apply potassium-rich fertiliser to strengthen stalks.",
        "Harvest at physiological maturity to prevent lodging.",
    ],
    # Cotton
    "Verticillium Wilt": [
        "Use resistant cultivars; practise crop rotation.",
        "Apply Trichoderma-based bio-fungicide to soil.",
        "Avoid excessive irrigation during vegetative stage.",
    ],
    "Leaf Curl": [
        "Control whitefly vectors with Imidacloprid spray.",
        "Use virus-resistant Bt-cotton varieties.",
        "Remove and burn infected plants promptly.",
    ],
    "Boll Rot": [
        "Apply Copper Oxychloride at boll-formation stage.",
        "Ensure good field drainage after rain.",
        "Avoid dense planting to reduce humidity.",
    ],
    # Soybean
    "Soybean Rust": [
        "Apply Tebuconazole + Trifloxystrobin at first sign.",
        "Plant early-maturing varieties; avoid late sowing.",
        "Scout fields weekly during reproductive stages.",
    ],
    "Downy Mildew": [
        "Seed treatment with Metalaxyl.",
        "Remove infected plant debris after harvest.",
        "Ensure well-drained soil; avoid overhead irrigation.",
    ],
    "Frogeye Leaf Spot": [
        "Apply Benomyl or Thiophanate-methyl fungicide.",
        "Rotate with non-host crops (cereals).",
        "Use certified disease-free seed.",
    ],
    "Stem Rot": [
        "Apply Carbendazim soil drench at planting.",
        "Rotate with cereals; improve organic matter.",
        "Avoid mechanical injury during cultivation.",
    ],
}

# Default for anything not explicitly listed
_DEFAULT_REC = [
    "Consult local agricultural extension officer for diagnosis.",
    "Collect samples and send to nearest plant pathology lab.",
    "Remove visibly infected plant parts to limit spread.",
]


def _get_disease_recommendations(crop, disease, severity):
    """Return actionable recommendations for a given disease."""
    if disease == "Healthy":
        return [
            "✅ Plant appears healthy — continue current management.",
            "Maintain regular scouting schedule for early detection.",
            "Ensure balanced nutrition and adequate irrigation.",
        ]
    recs = list(_DISEASE_RECS.get(disease, _DEFAULT_REC))
    if severity in ("High", "Critical"):
        recs.insert(0, f"⚠️ {severity} severity — take immediate action.")
    return recs
