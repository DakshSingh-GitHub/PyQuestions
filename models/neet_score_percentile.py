from __future__ import annotations

"""
NEET marks + rank prediction pipeline.

Dataset strategy:
1) Student performance dataset: candidate features -> neet_marks.
2) Rank reference dataset: marks -> rank mapping.

If files are missing, this script generates realistic starter datasets so the
pipeline can run immediately. Replace them with real yearly NEET data for
production quality.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "study_hours_daily",
    "mock_test_avg",
    "physics_accuracy_pct",
    "chemistry_accuracy_pct",
    "biology_accuracy_pct",
    "revision_cycles",
    "attendance_pct",
    "sleep_hours",
    "dropper_years",
]

CATEGORICAL_FEATURES = [
    "coaching_type",
    "exam_medium",
    "school_board",
]

TARGET_MARKS = "neet_marks"
TARGET_RANK = "neet_rank"


@dataclass
class Paths:
    student_data: Path
    rank_data: Path
    artifact_dir: Path


def rank_from_marks(marks: np.ndarray | float, total_candidates: int) -> np.ndarray:
    marks_arr = np.asarray(marks, dtype=float)
    normalized_gap = np.clip((720.0 - marks_arr) / 720.0, 0.0, 1.0)
    return 1 + total_candidates * np.power(normalized_gap, 3.75)


def generate_student_dataset(path: Path, rows: int = 12000) -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    ability = rng.beta(2.1, 2.0, rows)
    dropper_years = rng.choice([0, 1, 2], size=rows, p=[0.68, 0.24, 0.08])

    study_hours_daily = np.clip(
        rng.normal(3.4 + 4.6 * ability + 0.45 * dropper_years, 1.1),
        0.8,
        12.5,
    )
    mock_test_avg = np.clip(
        rng.normal(165 + 440 * ability + 16 * dropper_years, 36),
        50,
        710,
    )
    physics_accuracy_pct = np.clip(rng.normal(30 + 63 * ability, 9), 8, 99)
    chemistry_accuracy_pct = np.clip(rng.normal(35 + 60 * ability, 8), 10, 99)
    biology_accuracy_pct = np.clip(rng.normal(40 + 57 * ability, 7), 12, 99)
    revision_cycles = np.clip(np.round(rng.normal(1.5 + 4.8 * ability, 1.3)), 0, 11)
    attendance_pct = np.clip(rng.normal(62 + 34 * ability, 7), 35, 100)
    sleep_hours = np.clip(rng.normal(7.25 - 0.11 * (study_hours_daily - 5), 0.7), 4.5, 9.3)

    coaching_type = rng.choice(
        ["none", "offline", "online", "hybrid"], size=rows, p=[0.22, 0.37, 0.19, 0.22]
    )
    exam_medium = rng.choice(["english", "hindi", "regional"], size=rows, p=[0.58, 0.29, 0.13])
    school_board = rng.choice(["cbse", "state", "icse"], size=rows, p=[0.62, 0.31, 0.07])

    coaching_bonus = np.select(
        [coaching_type == "hybrid", coaching_type == "offline", coaching_type == "online"],
        [14, 9, 6],
        default=0,
    )
    sleep_penalty = np.maximum(0, 6.0 - sleep_hours) * 5.5

    neet_marks = (
        0.46 * mock_test_avg
        + 1.05 * physics_accuracy_pct
        + 1.2 * chemistry_accuracy_pct
        + 1.35 * biology_accuracy_pct
        + 6.7 * revision_cycles
        + 2.1 * study_hours_daily
        + 0.35 * attendance_pct
        + coaching_bonus
        - sleep_penalty
        + rng.normal(0, 22, rows)
        - 78
    )
    neet_marks = np.clip(np.round(neet_marks), 0, 720).astype(int)

    base_rank = rank_from_marks(neet_marks, total_candidates=2_350_000)
    rank_noise = rng.normal(0, 1500, rows)
    neet_rank = np.clip(np.round(base_rank + rank_noise), 1, 2_400_000).astype(int)

    df = pd.DataFrame(
        {
            "study_hours_daily": study_hours_daily.round(2),
            "mock_test_avg": mock_test_avg.round(1),
            "physics_accuracy_pct": physics_accuracy_pct.round(1),
            "chemistry_accuracy_pct": chemistry_accuracy_pct.round(1),
            "biology_accuracy_pct": biology_accuracy_pct.round(1),
            "revision_cycles": revision_cycles.astype(int),
            "attendance_pct": attendance_pct.round(1),
            "sleep_hours": sleep_hours.round(2),
            "dropper_years": dropper_years.astype(int),
            "coaching_type": coaching_type,
            "exam_medium": exam_medium,
            "school_board": school_board,
            "neet_marks": neet_marks,
            "neet_rank": neet_rank,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def generate_rank_reference_dataset(path: Path) -> None:
    years = [2022, 2023, 2024, 2025]
    candidates = {2022: 1_870_000, 2023: 2_040_000, 2024: 2_330_000, 2025: 2_380_000}
    rows = []
    for year in years:
        for marks in range(0, 721, 5):
            year_adjustment = 1 + ((year - 2022) * 0.02)
            rank = rank_from_marks(marks, int(candidates[year] * year_adjustment))
            rows.append((year, marks, int(round(float(rank)))))

    df = pd.DataFrame(rows, columns=["year", "marks", "rank"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def ensure_datasets(paths: Paths, rebuild_sample_data: bool) -> None:
    if rebuild_sample_data or not paths.student_data.exists():
        generate_student_dataset(paths.student_data)
    if rebuild_sample_data or not paths.rank_data.exists():
        generate_rank_reference_dataset(paths.rank_data)


def load_and_validate_student_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_MARKS])
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Student dataset is missing columns: {sorted(missing)}. "
            f"Required columns: {sorted(expected_columns)}"
        )
    return df


def load_and_validate_rank_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"marks", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Rank dataset is missing columns: {sorted(missing)}. "
            f"Required columns: {sorted(required)}"
        )
    return df


def train_marks_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_MARKS]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=160,
                    max_depth=12,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    metrics = {
        "marks_mae": float(mean_absolute_error(y_test, preds)),
        "marks_r2": float(r2_score(y_test, preds)),
    }
    return model, metrics


def train_rank_mapper(rank_df: pd.DataFrame) -> IsotonicRegression:
    grouped = (
        rank_df.groupby("marks", as_index=False)["rank"]
        .median()
        .sort_values("marks", ascending=True)
    )
    mapper = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=1)
    mapper.fit(grouped["marks"], grouped["rank"])
    return mapper


def evaluate_rank_pipeline(
    student_df: pd.DataFrame, marks_model: Pipeline, rank_mapper: IsotonicRegression
) -> dict[str, float]:
    if TARGET_RANK not in student_df.columns:
        return {}

    x = student_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_rank = student_df[TARGET_RANK].to_numpy()
    pred_marks = marks_model.predict(x)
    pred_rank = rank_mapper.predict(pred_marks)

    return {
        "rank_mae": float(mean_absolute_error(y_rank, pred_rank)),
        "rank_r2": float(r2_score(y_rank, pred_rank)),
    }


def predict_candidate(
    marks_model: Pipeline, rank_mapper: IsotonicRegression, candidate: dict[str, object]
) -> tuple[float, int]:
    df = pd.DataFrame([candidate])
    predicted_marks = float(np.clip(marks_model.predict(df)[0], 0, 720))
    predicted_rank = int(max(1, round(float(rank_mapper.predict([predicted_marks])[0]))))
    return predicted_marks, predicted_rank


def save_artifacts(paths: Paths, marks_model: Pipeline, rank_mapper: IsotonicRegression) -> None:
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": marks_model,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET_MARKS,
        },
        paths.artifact_dir / "neet_marks_model.joblib",
    )
    joblib.dump(rank_mapper, paths.artifact_dir / "neet_rank_mapper.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a NEET marks + rank predictor using student and rank datasets."
    )
    parser.add_argument(
        "--student-data",
        type=Path,
        default=Path("data/neet_student_profiles.csv"),
        help="CSV with student features and `neet_marks` (optional `neet_rank`).",
    )
    parser.add_argument(
        "--rank-data",
        type=Path,
        default=Path("data/neet_marks_rank_reference.csv"),
        help="CSV with `marks` and `rank` columns (can include extra columns like year).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("models/artifacts"),
        help="Directory where trained models are saved.",
    )
    parser.add_argument(
        "--rebuild-sample-data",
        action="store_true",
        help="Regenerate starter datasets even if files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(
        student_data=args.student_data,
        rank_data=args.rank_data,
        artifact_dir=args.artifact_dir,
    )

    ensure_datasets(paths, rebuild_sample_data=args.rebuild_sample_data)
    student_df = load_and_validate_student_data(paths.student_data)
    rank_df = load_and_validate_rank_data(paths.rank_data)

    marks_model, marks_metrics = train_marks_model(student_df)
    rank_mapper = train_rank_mapper(rank_df)
    rank_metrics = evaluate_rank_pipeline(student_df, marks_model, rank_mapper)
    save_artifacts(paths, marks_model, rank_mapper)

    sample_candidate = {
        "study_hours_daily": 7.1,
        "mock_test_avg": 540,
        "physics_accuracy_pct": 74,
        "chemistry_accuracy_pct": 79,
        "biology_accuracy_pct": 84,
        "revision_cycles": 6,
        "attendance_pct": 89,
        "sleep_hours": 7.0,
        "dropper_years": 1,
        "coaching_type": "hybrid",
        "exam_medium": "english",
        "school_board": "cbse",
    }

    predicted_marks, predicted_rank = predict_candidate(
        marks_model, rank_mapper, sample_candidate
    )

    print(f"Student dataset: {paths.student_data}")
    print(f"Rank reference dataset: {paths.rank_data}")
    print(f"Saved artifacts in: {paths.artifact_dir}")
    print(f"Marks model -> MAE: {marks_metrics['marks_mae']:.2f}, R2: {marks_metrics['marks_r2']:.3f}")
    if rank_metrics:
        print(f"Rank pipeline -> MAE: {rank_metrics['rank_mae']:.0f}, R2: {rank_metrics['rank_r2']:.3f}")
    print(
        "Sample candidate prediction -> "
        f"Marks: {predicted_marks:.1f}/720, Estimated Rank: {predicted_rank}"
    )


if __name__ == "__main__":
    main()
