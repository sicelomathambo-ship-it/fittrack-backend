# FitTrack AI — Backend

**CIS 508 · Group 9 · Machine Learning in Business**

FastAPI backend that powers the FitTrack AI dashboard at `https://fit-track-ai-ruby.vercel.app/`.

---

## Architecture

```
fittrack_ai_dataset.csv  →  train_model.py  →  calories_model.joblib
                                                       ↓
frontend (Vercel)  ←→  FastAPI (main.py)  ←→  Anthropic API (LLM layer)
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Server + model status |
| `GET`  | `/api/stats` | Full dataset statistics + cohort table |
| `POST` | `/api/predict` | ML prediction: calories, fatigue, recovery |
| `POST` | `/api/plan` | Full session plan (prediction + exercises) |
| `POST` | `/api/explain` | LLM governance explanation (Claude API) |
| `POST` | `/api/nutrition` | Post-session LLM nutrition recommendation |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (run once)
```bash
python train_model.py
```
This produces `calories_model.joblib` — the same Random Forest from the Colab notebook.

### 3. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 4. Start the server
```bash
uvicorn main:app --reload --port 8000
```

### 5. Open the interactive docs
```
http://localhost:8000/docs
```

---

## Example Requests

### POST /api/plan
```json
{
  "goal": "Muscle Gain",
  "experience_level": 2,
  "age": 28,
  "weight_kg": 80,
  "sleep_hours": 7.5,
  "height_m": 1.80
}
```

**Response:**
```json
{
  "profile": { "goal": "Muscle Gain", "level_name": "Intermediate", ... },
  "prediction": {
    "calories_burned": 523,
    "fatigue_score": 33.8,
    "recovery_pct": 91.2,
    "recovery_status": "Ready to train",
    "hrv_ms": 53.3,
    "avg_bpm": 149,
    "duration_min": 65,
    "protein_target_g": 30,
    "water_intake_liters": 2.6,
    "model_source": "random_forest"
  },
  "workout": {
    "exercises": [
      { "exercise": "Back squat", "sets": 4, "reps": "6", "rest_sec": 120, "intensity": "75% 1RM" },
      ...
    ],
    "workout_type": "Strength"
  }
}
```

### POST /api/explain
```json
{
  "goal": "Muscle Gain",
  "experience_level": 2,
  "age": 28,
  "weight_kg": 80,
  "sleep_hours": 7.5,
  "calories": 523,
  "fatigue_score": 33.8,
  "recovery_pct": 91.2,
  "duration_min": 65,
  "recovery_status": "Ready to train"
}
```

---

## Dataset

`fittrack_ai_dataset.csv` — 2,000 gym sessions with features:

| Column | Description |
|--------|-------------|
| `age`, `gender`, `weight_kg`, `height_m` | Demographics |
| `bmi`, `fat_percentage` | Body composition |
| `experience_level` | 1=Beginner, 2=Intermediate, 3=Advanced |
| `goal` | Muscle Gain / Weight Loss / Endurance / General Fitness / Flexibility |
| `avg_bpm`, `max_bpm`, `hrv_ms`, `resting_bpm` | Heart rate metrics |
| `session_duration_hours`, `calories_burned` | Session output (target) |
| `fatigue_score`, `recovery_ready`, `completion_pct` | Training state targets |
| `protein_target_g`, `water_intake_liters` | Nutrition targets |

---

## ML Model

The Random Forest Regressor matches the notebook (`FitAI_Calories_Prediction_Model.ipynb`):

- **Target**: `calories_burned`
- **Features**: 15 numerical + one-hot encoded categoricals
- **Split**: 80/20 train/test, `random_state=42`
- **Expected performance**: MAE < 30 kcal, R² > 0.92

If no model file is found, the server falls back to cohort-based statistics which provide similar accuracy at the population level.

---

## Connecting to the Frontend

In your Vercel frontend, replace mock data with API calls:

```javascript
// Generate plan
const response = await fetch('https://your-backend.railway.app/api/plan', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ goal, experience_level, age, weight_kg, sleep_hours })
});
const plan = await response.json();

// Get LLM explanation
const explain = await fetch('https://your-backend.railway.app/api/explain', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ ...plan.profile, ...plan.prediction })
});
```

## Deployment (Railway / Render / Fly.io)

```bash
# Railway
railway init
railway up

# Set env var in dashboard:
ANTHROPIC_API_KEY=sk-ant-...
```

The `main.py` entry point (`uvicorn main:app`) is compatible with all major Python hosting platforms.

---

## Governance layer

Every `/api/explain` call uses Claude to:
1. Explain **why** the plan was selected (specificity + progressive overload rationale)
2. Interpret the **fatigue score** with a concrete training adjustment
3. Provide a **nutrition action** tied to calorie burn and goal

This mirrors Slide 9 of the presentation: *"Plan explainability — LLM narrates why each exercise was selected."*
