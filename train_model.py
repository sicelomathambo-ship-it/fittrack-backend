"""
train_model.py — FitTrack AI
Run this once to produce calories_model.joblib (the same model from the Colab notebook).
Place calories_model.joblib in the same directory as main.py before starting the server.

Usage:
    python train_model.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "fittrack_ai_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "calories_model.joblib")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH}")

# ── Feature engineering ────────────────────────────────────────────────────────
# Drop columns that are targets or would cause data leakage
drop_cols = [
    "calories_burned",      # target
    "session_date",         # not predictive
    "muscles_trained",      # free-text, handled separately if needed
    "workout_completed",    # outcome, not input
    "protein_target_g",     # derived output
    "fatigue_score",        # also a target (predict separately)
    "recovery_ready",       # also a target
    "completion_pct",       # also a target
    "cumulative_load_7d_kcal",  # computed from target — leakage risk
]

feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"Features used: {feature_cols}")

X = df[feature_cols].copy()
y = df["calories_burned"]

# One-hot encode categoricals (matches notebook's pd.get_dummies approach)
X = pd.get_dummies(X, drop_first=True)
print(f"Feature matrix shape: {X.shape}")

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ── Train Random Forest (same as notebook) ─────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"\nModel performance:")
print(f"  MAE : {mae:.2f} kcal")
print(f"  R²  : {r2:.4f}")

# ── Feature importance (top 10) ────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 10 feature importances:")
print(importances.nlargest(10).to_string())

# ── Save model ─────────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
print(f"\n✓ Model saved to {MODEL_PATH}")

# Save feature names so main.py can align inference vectors
import json
feature_names_path = os.path.join(BASE_DIR, "feature_names.json")
with open(feature_names_path, "w") as f:
    json.dump(list(X.columns), f, indent=2)
print(f"✓ Feature names saved to {feature_names_path}")
