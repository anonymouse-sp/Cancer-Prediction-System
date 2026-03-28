from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Any
import csv
import numpy as np

app = FastAPI(title="OncoPredict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: int = Field(..., ge=0, le=1)
    height_cm: float = Field(..., ge=120.0, le=230.0)
    weight_kg: float = Field(..., ge=30.0, le=250.0)
    smoke: int = Field(..., ge=0, le=1)
    genetic_risk: int = Field(..., ge=0, le=2)
    early_onset: int = Field(1, ge=0, le=2)
    known_gene: int = Field(1, ge=0, le=2)
    physical_activity: float = Field(..., ge=0.0, le=5.0)
    alcohol_intake: float = Field(..., ge=0.0, le=5.0)
    family_history: int = Field(..., ge=0, le=1)
    diet_pattern: int | None = Field(None, ge=0, le=2)
    pollution_exposure: int | None = Field(None, ge=0, le=1)
    uv_radiation_exposure: int | None = Field(None, ge=0, le=1)
    occupational_hazard: int | None = Field(None, ge=0, le=1)
    diabetes: int | None = Field(None, ge=0, le=1)
    inflammation_marker: float | None = Field(None, ge=0.0, le=20.0)
    bp_level: int | None = Field(None, ge=0, le=2)
    fatigue_level: int | None = Field(None, ge=0, le=3)
    symptom_flags: list[str] | None = None


FEATURE_COLUMNS = [
    "age",
    "gender",
    "bmi",
    "smoke",
    "genetic_risk",
    "physical_activity",
    "alcohol_intake",
    "family_history",
]

FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "age": ("Age", "age", "AGE"),
    "gender": ("Gender", "gender", "sex", "Sex"),
    "bmi": ("BMI", "bmi", "body_mass_index", "BodyMassIndex"),
    "smoke": (
        "Smoking",
        "smoking",
        "smoke",
        "Smokes",
        "Smokes (years)",
        "tobacco_use",
    ),
    "genetic_risk": (
        "GeneticRisk",
        "genetic_risk",
        "family_genetic_risk",
        "hereditary_risk",
    ),
    "physical_activity": (
        "PhysicalActivity",
        "physical_activity",
        "exercise_level",
        "activity_level",
    ),
    "alcohol_intake": (
        "AlcoholIntake",
        "alcohol_intake",
        "alcohol",
        "Alcohol",
        "drinking_level",
    ),
    "family_history": (
        "CancerHistory",
        "family_history",
        "FamilyHistory",
        "Family history",
        "Dx:Cancer",
    ),
}

TARGET_ALIASES = (
    "Diagnosis",
    "diagnosis",
    "target",
    "Target",
    "Risk",
    "risk",
    "Biopsy",
    "cancer",
    "Cancer",
)

FEATURE_DEFAULTS = {
    "age": 45.0,
    "gender": 0.0,
    "bmi": 25.0,
    "smoke": 0.0,
    "genetic_risk": 1.0,
    "physical_activity": 5.0,
    "alcohol_intake": 2.0,
    "family_history": 0.0,
}


def _get_dataset_paths() -> list[Path]:
    backend_dir = Path(__file__).resolve().parent
    candidate_paths = [
        backend_dir.parent / "The_Cancer_data_1500_V2.csv",
        backend_dir.parent / "risk_factors_cervical_cancer.csv",
        backend_dir / "The_Cancer_data_1500_V2.csv",
        backend_dir / "data" / "The_Cancer_data_1500_V2.csv",
        backend_dir / "data" / "risk_factors_cervical_cancer.csv",
        Path.home() / "Downloads" / "The_Cancer_data_1500_V2.csv",
        Path.home() / "Downloads" / "risk_factors_cervical_cancer.csv",
    ]

    for folder in [backend_dir.parent, backend_dir, backend_dir / "data"]:
        if folder.exists():
            candidate_paths.extend(sorted(folder.glob("*.csv")))

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in candidate_paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def _find_value(row: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for key in aliases:
        if key in row and str(row[key]).strip() != "":
            return row[key]
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"", "nan", "na", "none", "null", "?"}:
        return None
    if text in {"yes", "true", "y"}:
        return 1.0
    if text in {"no", "false", "n"}:
        return 0.0

    try:
        return float(text)
    except ValueError:
        return None


def _to_binary(value: Any) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return 1 if parsed > 0 else 0


def _extract_training_arrays(rows: list[dict[str, str]], fieldnames: list[str]) -> "tuple[np.ndarray, np.ndarray, float]":
    target_col = next((col for col in TARGET_ALIASES if col in fieldnames), None)
    if target_col is None:
        return np.empty((0, 8)), np.empty((0,)), 0.0

    feature_rows: list[list[float]] = []
    labels: list[int] = []
    mapped_count: int = 0
    usable_count: int = 0

    for row in rows:
        target_raw = row.get(target_col)
        target = _to_binary(target_raw)
        if target is None:
            continue

        values: dict[str, float | None] = {}
        row_mapped = 0

        for feature in FEATURE_COLUMNS:
            raw = _find_value(row, FEATURE_ALIASES[feature])
            parsed = _to_float(raw)
            if parsed is not None:
                row_mapped += 1
            values[feature] = parsed

        if values["bmi"] is None:
            h_raw = _find_value(row, ("height_cm", "HeightCm", "height"))
            w_raw = _find_value(row, ("weight_kg", "WeightKg", "weight"))
            h_val = _to_float(h_raw)
            w_val = _to_float(w_raw)
            if h_val and h_val > 0 and w_val is not None:
                h_m = h_val / 100.0
                values["bmi"] = w_val / (h_m * h_m)
                row_mapped += 1

        row_values = []
        for feature in FEATURE_COLUMNS:
            val = values[feature]
            row_values.append(float(val) if val is not None else np.nan)

        if row_mapped >= 4:
            usable_count += 1
            mapped_count += row_mapped
            feature_rows.append(row_values)
            labels.append(target)

    if not feature_rows:
        return np.empty((0, 8)), np.empty((0,)), 0.0

    x = np.array(feature_rows, dtype=float)
    y = np.array(labels, dtype=int)

    for idx, feature in enumerate(FEATURE_COLUMNS):
        col = x[:, idx]
        mask = np.isfinite(col)
        if np.any(mask):
            fill = float(np.median(col[mask]))
        else:
            fill = FEATURE_DEFAULTS[feature]
        col[~mask] = fill
        x[:, idx] = col

    coverage = mapped_count / float(max(1, usable_count * len(FEATURE_COLUMNS)))
    return x, y, coverage


def _load_training_data() -> "tuple[np.ndarray, np.ndarray, str, float]":
    best_candidate: "tuple[np.ndarray, np.ndarray, str, float, float] | None" = None

    for candidate in _get_dataset_paths():
        if not candidate.exists():
            continue

        with candidate.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue

            rows = list(reader)
            fieldnames: list[str] = list(reader.fieldnames)

        if not rows:
            continue

        x, y, coverage = _extract_training_arrays(rows, fieldnames)
        if len(y) < 20 or len(np.unique(y)) < 2:
            continue

        score = (coverage * 0.7) + (min(len(y), 5000) / 5000.0 * 0.3)
        current = (x, y, str(candidate), coverage, score)
        if best_candidate is None or current[4] > best_candidate[4]:
            best_candidate = current

    if best_candidate is not None:
        return best_candidate[0], best_candidate[1], best_candidate[2], best_candidate[3]

    x = np.array(
        [
            [25, 0, 20.1, 0, 0, 8.5, 1.0, 0],
            [30, 1, 24.8, 0, 0, 7.9, 1.2, 0],
            [35, 1, 31.5, 1, 2, 2.4, 4.0, 1],
            [40, 0, 33.0, 1, 2, 2.1, 3.7, 1],
            [45, 1, 29.2, 1, 1, 3.2, 3.0, 1],
            [50, 0, 34.4, 1, 2, 1.8, 4.3, 1],
            [55, 1, 36.1, 1, 2, 1.5, 4.6, 1],
            [60, 0, 35.7, 1, 2, 1.2, 4.5, 1],
            [65, 1, 37.4, 1, 2, 0.9, 4.8, 1],
            [28, 0, 21.9, 0, 0, 8.1, 1.1, 0],
            [33, 1, 26.8, 0, 1, 6.8, 1.8, 1],
            [48, 1, 32.1, 1, 1, 3.4, 3.2, 1],
            [52, 0, 30.7, 1, 1, 3.7, 2.5, 0],
            [37, 0, 27.8, 0, 1, 5.9, 2.9, 0],
            [42, 1, 28.9, 0, 1, 5.4, 2.2, 1],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1], dtype=int)
    return x, y, "synthetic", 1.0


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> "tuple[float, float]":
    best_threshold = 0.5
    best_balanced_accuracy = -1.0

    for threshold in np.linspace(0.3, 0.8, 51):
        y_pred = (y_prob >= threshold).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        balanced_accuracy = (tpr + tnr) / 2.0

        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_threshold = float(threshold)

    return best_threshold, best_balanced_accuracy


def _evaluate_split(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division="warn")),
        "recall": float(recall_score(y_true, y_pred, zero_division="warn")),
        "f1": float(f1_score(y_true, y_pred, zero_division="warn")),
    }


X_data, y_data, TRAINING_SOURCE, FEATURE_COVERAGE = _load_training_data()

X_train, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

VAL_PROB = model.predict_proba(X_val)[:, 1]
HIGH_RISK_THRESHOLD, VAL_BALANCED_ACCURACY = _find_best_threshold(np.asarray(y_val), VAL_PROB)
HIGH_RISK_THRESHOLD = max(0.45, HIGH_RISK_THRESHOLD)
VAL_METRICS = _evaluate_split(np.asarray(y_val), VAL_PROB, HIGH_RISK_THRESHOLD)
TEST_PROB = model.predict_proba(X_test)[:, 1]
TEST_METRICS = _evaluate_split(np.asarray(y_test), TEST_PROB, HIGH_RISK_THRESHOLD)

DOCTORS = [
    "Local General Physician",
    "Oncology Specialist",
    "Diagnostic Lab",
]


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "training_source": TRAINING_SOURCE,
        "samples": int(len(y_data)),
        "split": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "model_threshold": round(float(HIGH_RISK_THRESHOLD), 3),
        "validation_balanced_accuracy": round(float(VAL_BALANCED_ACCURACY), 4),
        "validation_metrics": {k: round(float(v), 4) for k, v in VAL_METRICS.items()},
        "test_metrics": {k: round(float(v), 4) for k, v in TEST_METRICS.items()},
        "feature_count": 8,
        "feature_coverage": round(float(FEATURE_COVERAGE), 4),
    }


def _questionnaire_risk_score(
    payload: PredictRequest,
    bmi: float,
    symptom_count: int,
    red_flag_count: int,
    symptom_score: float,
) -> float:
    score = 0.0

    if payload.age >= 60:
        score += 0.08
    elif payload.age >= 45:
        score += 0.05
    elif payload.age >= 35:
        score += 0.02

    if bmi >= 35:
        score += 0.07
    elif bmi >= 30:
        score += 0.05
    elif bmi < 18.5:
        score += 0.03

    score += 0.1 if payload.smoke == 1 else 0.0
    score += 0.06 * payload.genetic_risk
    score += 0.08 if payload.family_history == 1 else 0.0
    score += min(0.05, (payload.alcohol_intake / 5.0) * 0.05)
    score += min(0.05, ((5.0 - payload.physical_activity) / 5.0) * 0.05)

    if payload.diet_pattern is not None:
        score += 0.03 * float(payload.diet_pattern)
    if payload.pollution_exposure == 1:
        score += 0.04
    if payload.uv_radiation_exposure == 1:
        score += 0.04
    if payload.occupational_hazard == 1:
        score += 0.05
    if payload.diabetes == 1:
        score += 0.04

    if payload.inflammation_marker is not None:
        inflammation: float = float(payload.inflammation_marker)
        if inflammation >= 10.0:
            score += 0.08
        elif inflammation >= 6.0:
            score += 0.05
        elif inflammation >= 3.0:
            score += 0.02

    if payload.bp_level is not None:
        score += 0.03 * float(payload.bp_level)
    if payload.fatigue_level is not None:
        fatigue_level_int: int = int(payload.fatigue_level)
        fatigue_weights: dict[int, float] = {0: 0.0, 1: 0.02, 2: 0.04, 3: 0.06}
        score += fatigue_weights.get(fatigue_level_int, 0.0)

    early_onset_weights = {0: 0.0, 1: 0.03, 2: 0.06}
    known_gene_weights = {0: 0.0, 1: 0.04, 2: 0.1}
    score += early_onset_weights.get(payload.early_onset, 0.0)
    score += known_gene_weights.get(payload.known_gene, 0.0)

    score += min(0.18, symptom_count * 0.02)
    score += min(0.12, red_flag_count * 0.04)
    score += min(0.1, symptom_score * 0.01)

    return float(min(1.0, score))


@app.post("/predict")
def predict(payload: PredictRequest):
    height_m = payload.height_cm / 100.0
    bmi = payload.weight_kg / (height_m * height_m)
    physical_activity_model_scale = payload.physical_activity * 2.0

    features = np.array(
        [
            [
                payload.age,
                payload.gender,
                bmi,
                payload.smoke,
                payload.genetic_risk,
                physical_activity_model_scale,
                payload.alcohol_intake,
                payload.family_history,
            ]
        ]
    )

    base_risk_score = float(model.predict_proba(features)[0][1])

    symptom_weights = {
        "unexplained_weight_loss": 2.5,
        "new_lumps": 4.0,
        "unusual_bleeding": 3.0,
        "non_healing_sores": 2.0,
        "changed_bathroom_habits": 2.0,
        "persistent_cough": 2.0,
        "changing_moles": 2.0,
        "difficulty_swallowing": 2.5,
    }
    selected_symptoms = set(payload.symptom_flags or [])
    symptom_score = float(sum(symptom_weights.get(symptom, 0.0) for symptom in selected_symptoms))
    symptom_count = len(selected_symptoms)
    red_flag_symptoms = {
        "new_lumps",
        "unusual_bleeding",
        "difficulty_swallowing",
        "persistent_cough",
    }
    red_flag_count = len(selected_symptoms.intersection(red_flag_symptoms))

    questionnaire_risk = _questionnaire_risk_score(
        payload,
        bmi,
        symptom_count,
        red_flag_count,
        symptom_score,
    )
    risk_score = min(1.0, (0.6 * base_risk_score) + (0.4 * questionnaire_risk))

    severe_symptom_pattern = symptom_count >= 6 or red_flag_count >= 3
    if severe_symptom_pattern:
        risk_score = max(risk_score, 0.72)

    high_risk = risk_score >= HIGH_RISK_THRESHOLD
    specialist_alert = payload.known_gene == 2

    if high_risk:
        response = {
            "risk_score": round(float(risk_score), 3),
            "risk_level": "high",
            "message": "Your risk is High. Please see a doctor soon.",
            "scoring": {
                "model_probability": round(float(base_risk_score), 3),
                "questionnaire_score": round(float(questionnaire_risk), 3),
                "decision_threshold": round(float(HIGH_RISK_THRESHOLD), 3),
            },
            "doctors": DOCTORS,
            "remedies": [
                "Schedule a screening test",
                "Talk to a health worker",
                "Avoid tobacco immediately",
            ],
            "disclaimer": "THIS IS NOT A MEDICAL DIAGNOSIS. CONSULT A DOCTOR.",
        }
        if specialist_alert:
            response["specialist_recommendation"] = (
                "Known hereditary cancer gene reported. Consult an oncology/genetics specialist promptly."
            )
        return response

    response = {
        "risk_score": round(float(risk_score), 3),
        "risk_level": "low",
        "message": "You are doing great! Stay healthy.",
        "scoring": {
            "model_probability": round(float(base_risk_score), 3),
            "questionnaire_score": round(float(questionnaire_risk), 3),
            "decision_threshold": round(float(HIGH_RISK_THRESHOLD), 3),
        },
        "maintenance": [
            "Eat 2 fruits every day",
            "Walk for 30 minutes",
            "Drink plenty of water",
        ],
        "improvements": [
            "Try to sleep 8 hours",
            "Reduce sugar intake",
        ],
        "disclaimer": "THIS IS NOT A MEDICAL DIAGNOSIS. CONSULT A DOCTOR.",
    }
    if specialist_alert:
        response["specialist_recommendation"] = (
            "Known hereditary cancer gene reported. Consult an oncology/genetics specialist promptly."
        )
    return response
