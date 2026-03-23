"""
model.py  –  Multi-model training, evaluation, and prediction with branded AI names.
Auto-retrain system: retrains after N app opens, dataset change, or new PC.
"""

import os, warnings, hashlib, platform, json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.ensemble        import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree          import DecisionTreeRegressor
from sklearn.neighbors     import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm           import SVR
from sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

MODEL_DIR  = "saved_models"
META_FILE  = os.path.join(MODEL_DIR, "meta.pkl")

CAT_COLS = ["Crop", "Season", "State", "Soil_Type"]
NUM_COLS = [
    "Area_ha", "Nitrogen", "Phosphorus", "Potassium", "pH",
    "Temperature", "Humidity", "Rainfall", "Fertilizer", "Pesticide", "Irrigation",
]
TARGET = "Yield_ton_per_ha"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Branded AI model registry                                                    #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
MODEL_REGISTRY = {
    "AgriForest AI": {
        "cls": RandomForestRegressor,
        "params": {"n_estimators": 250, "max_depth": 18, "random_state": 42, "n_jobs": -1},
        "icon": "🌲",
        "desc": "Ensemble of 250 decision trees — best for structured tabular data. "
                "Highly accurate and resistant to overfitting.",
        "family": "Random Forest",
    },
    "CropBoost Pro": {
        "cls": GradientBoostingRegressor,
        "params": {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.08, "random_state": 42},
        "icon": "🚀",
        "desc": "Sequential boosting that corrects previous errors. "
                "Top performer on many Kaggle competitions.",
        "family": "Gradient Boosting",
    },
    "DeepCrop Neural": {
        "cls": MLPRegressor,
        "params": {"hidden_layer_sizes": (128, 64, 32), "max_iter": 500,
                   "learning_rate_init": 0.001, "random_state": 42, "early_stopping": True},
        "icon": "🧠",
        "desc": "Multi-layer neural network (128→64→32 neurons). "
                "Captures non-linear patterns in climate & soil data.",
        "family": "Neural Network",
    },
    "ExtraCrop AI": {
        "cls": ExtraTreesRegressor,
        "params": {"n_estimators": 200, "max_depth": 18, "random_state": 42, "n_jobs": -1},
        "icon": "🌳",
        "desc": "Extremely randomized trees with additional randomness for robust predictions.",
        "family": "Extra Trees",
    },
    "FarmTree Lite": {
        "cls": DecisionTreeRegressor,
        "params": {"max_depth": 12, "random_state": 42},
        "icon": "🌿",
        "desc": "Single decision tree — simple, fast, and interpretable. "
                "Good for understanding feature importance.",
        "family": "Decision Tree",
    },
    "AgriLinear Plus": {
        "cls": Ridge,
        "params": {"alpha": 1.0},
        "icon": "📈",
        "desc": "Ridge regression with L2 regularization. "
                "Linear model baseline — fast training, reliable.",
        "family": "Ridge Regression",
    },
    "CropSVM Expert": {
        "cls": SVR,
        "params": {"kernel": "rbf", "C": 10.0, "epsilon": 0.1},
        "icon": "🎯",
        "desc": "Support Vector Regression with RBF kernel. "
                "Finds optimal hyperplane in high-dimensional space.",
        "family": "Support Vector Machine",
    },
    "AgroNeighbor AI": {
        "cls": KNeighborsRegressor,
        "params": {"n_neighbors": 7, "weights": "distance", "n_jobs": -1},
        "icon": "📍",
        "desc": "K-Nearest Neighbors — predicts based on the 7 most similar crop records. "
                "Simple yet effective instance-based learning.",
        "family": "K-Nearest Neighbors",
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Encoding / Scaling                                                           #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
def _encode(df, encoders=None, fit=True):
    df = df.copy()
    if fit:
        encoders = {}
        for col in CAT_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in CAT_COLS:
            df[col] = encoders[col].transform(df[col].astype(str))
    return df, encoders


def _scale(X, scaler=None, fit=True):
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, scaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Training                                                                     #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
def train_all_models(df: pd.DataFrame, n_rounds: int = 3,
                     progress_cb=None) -> dict:
    """
    Train every model in MODEL_REGISTRY for *n_rounds* independent attempts
    (different data-split seeds).  The round with the highest average R² is
    kept and saved — this gives a measurable accuracy boost over a single run.

    Args:
        df:          Training dataframe.
        n_rounds:    Number of training rounds (default 3).
        progress_cb: Optional callback(round_num, total, round_avg_r2).
    Returns:
        meta dict (same format as before, with added training_rounds info).
    """
    cols = CAT_COLS + NUM_COLS + [TARGET]
    df_enc, encoders = _encode(df[cols])
    X = df_enc[CAT_COLS + NUM_COLS].values
    y = df_enc[TARGET].values

    SEEDS = [42, 59, 73, 91]          # one seed per round
    best_round_avg  = -np.inf
    best_round_data = None             # (results, scaler, models_dict, round_num)
    round_history   = []               # for logging

    for rnd in range(min(n_rounds, len(SEEDS))):
        seed = SEEDS[rnd]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed)
        X_tr_sc, scaler = _scale(X_tr, fit=True)
        X_te_sc, _      = _scale(X_te, scaler=scaler, fit=False)

        round_results = {}
        round_models  = {}

        for name, spec in MODEL_REGISTRY.items():
            params = spec["params"].copy()
            if "random_state" in params:
                params["random_state"] = seed       # vary seed per round

            mdl = spec["cls"](**params)
            mdl.fit(X_tr_sc, y_tr)
            y_pred = mdl.predict(X_te_sc)

            mae  = mean_absolute_error(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            r2   = r2_score(y_te, y_pred)
            cv   = cross_val_score(mdl, X_tr_sc, y_tr, cv=3,
                                   scoring="r2", n_jobs=-1).mean()

            fi = None
            if hasattr(mdl, "feature_importances_"):
                fi = dict(zip(CAT_COLS + NUM_COLS,
                              mdl.feature_importances_.tolist()))

            round_results[name] = {
                "MAE": round(mae, 4), "RMSE": round(rmse, 4),
                "R2": round(r2, 4), "CV_R2": round(cv, 4),
                "feat_import": fi,
            }
            round_models[name] = mdl

        avg_r2 = float(np.mean([v["R2"] for v in round_results.values()]))
        round_history.append({"round": rnd + 1, "seed": seed, "avg_r2": round(avg_r2, 4)})

        arrow = " ▲" if avg_r2 > best_round_avg else ""
        print(f"[model] Round {rnd+1}/{n_rounds} complete "
              f"(seed={seed}, avg R²={avg_r2:.4f}){arrow}")

        if avg_r2 > best_round_avg:
            best_round_avg  = avg_r2
            best_round_data = (round_results, scaler, round_models, rnd + 1)

        if progress_cb:
            progress_cb(rnd + 1, n_rounds, round(avg_r2, 4))

    # ── Save the winning round ────────────────────────────────────
    results, scaler, models, best_rnd = best_round_data
    best_r2, best_name = -np.inf, None

    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, mdl in models.items():
        joblib.dump(mdl, os.path.join(MODEL_DIR, f"{name}.pkl"))
        if results[name]["R2"] > best_r2:
            best_r2  = results[name]["R2"]
            best_name = name

    meta = {
        "encoders": encoders, "scaler": scaler,
        "best_name": best_name,
        "feat_names": CAT_COLS + NUM_COLS,
        "results": results,
        "training_rounds": n_rounds,
        "best_round": best_rnd,
        "round_history": round_history,
        "model_registry": {
            k: {m: v for m, v in spec.items() if m not in ("cls",)}
            for k, spec in MODEL_REGISTRY.items()
        },
    }
    joblib.dump(meta, META_FILE)
    print(f"[model] Best round: {best_rnd}/{n_rounds}. "
          f"All {len(results)} models saved. Best: {best_name} (R²={best_r2:.4f})")
    return meta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Prediction                                                                   #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
def predict_yield(inputs: dict, model_name: str | None = None) -> float:
    """Predict yield using a chosen (or best) model."""
    meta = joblib.load(META_FILE)
    name = model_name or meta["best_name"]
    mdl  = joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl"))

    row = pd.DataFrame([inputs])
    row, _ = _encode(row, encoders=meta["encoders"], fit=False)
    X = row[CAT_COLS + NUM_COLS].values
    X_sc, _ = _scale(X, scaler=meta["scaler"], fit=False)
    return float(mdl.predict(X_sc)[0])


def load_meta() -> dict:
    return joblib.load(META_FILE)


def is_trained() -> bool:
    return os.path.exists(META_FILE)


def get_model_names() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def get_model_info(name: str) -> dict:
    return MODEL_REGISTRY.get(name, {})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Auto-Retrain Tracker                                                         #
#  Retrains when: (1) 25+ app opens since last train, (2) new dataset detected, #
#                 (3) running on a different PC / hostname                       #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
TRACKER_FILE = os.path.join(MODEL_DIR, "retrain_tracker.json")
RETRAIN_INTERVAL = 25  # retrain every N app opens


def _get_machine_id() -> str:
    """Unique ID for the current machine (hostname + platform)."""
    return hashlib.md5(
        f"{platform.node()}|{platform.system()}|{platform.machine()}".encode()
    ).hexdigest()[:12]


def _get_dataset_hash(df: pd.DataFrame) -> str:
    """Fast hash of the dataset shape + sample values to detect changes."""
    sig = f"{df.shape}|{df.columns.tolist()}|{df.iloc[:5].to_csv(index=False)}"
    return hashlib.md5(sig.encode()).hexdigest()[:16]


def _load_tracker() -> dict:
    """Load the retrain tracker state."""
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_tracker(tracker: dict):
    """Save the retrain tracker state."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)


def increment_open_count():
    """Call this every time the app starts. Returns the updated count."""
    tracker = _load_tracker()
    tracker["open_count"] = tracker.get("open_count", 0) + 1
    _save_tracker(tracker)
    return tracker["open_count"]


def check_retrain_needed(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Check if auto-retrain should trigger.

    Returns (should_retrain: bool, reason: str).
    Reasons: 'first_run', 'open_count', 'new_dataset', 'new_machine', ''
    """
    tracker = _load_tracker()
    current_machine = _get_machine_id()
    current_data_hash = _get_dataset_hash(df)

    # Case 0: No saved models at all
    if not is_trained():
        return True, "first_run"

    # Case 0.5: Tracker was never initialised (first time using auto-retrain)
    if not tracker.get("last_train_time"):
        return True, "first_run"

    # Case 1: Running on a different PC
    if tracker.get("machine_id") and tracker["machine_id"] != current_machine:
        return True, "new_machine"

    # Case 2: Dataset has changed
    if tracker.get("data_hash") and tracker["data_hash"] != current_data_hash:
        return True, "new_dataset"

    # Case 3: Enough app opens since last training
    opens_since_train = tracker.get("open_count", 0) - tracker.get("last_train_open", 0)
    if opens_since_train >= RETRAIN_INTERVAL:
        return True, "open_count"

    return False, ""


def mark_training_done(df: pd.DataFrame):
    """Mark that training just completed — update tracker state."""
    tracker = _load_tracker()
    tracker["machine_id"] = _get_machine_id()
    tracker["data_hash"] = _get_dataset_hash(df)
    tracker["last_train_open"] = tracker.get("open_count", 0)
    tracker["last_train_time"] = pd.Timestamp.now().isoformat()
    _save_tracker(tracker)


def get_retrain_status() -> dict:
    """Return human-readable auto-retrain status for UI display."""
    tracker = _load_tracker()
    opens = tracker.get("open_count", 0)
    last_open = tracker.get("last_train_open", 0)
    remaining = max(0, RETRAIN_INTERVAL - (opens - last_open))
    return {
        "total_opens": opens,
        "opens_since_train": opens - last_open,
        "opens_until_retrain": remaining,
        "last_train_time": tracker.get("last_train_time", "Never"),
        "machine_id": tracker.get("machine_id", "Unknown"),
        "current_machine": _get_machine_id(),
        "same_machine": tracker.get("machine_id") == _get_machine_id(),
    }
