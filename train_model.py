
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

FAKE_PATH = BASE_DIR / "fake_rental_same_format_100x12 (1).csv"
TRUE_PATH = BASE_DIR / "real_rental_dataset_100_rows (1).csv"

MODEL_PATH = BASE_DIR / "rental_model.joblib"

NUMERIC_COLS = [
    "BHK",
    "Rent",
    "Size",
    "Rent_per_sqft",
    "Log_rent"
]

CATEGORICAL_COLS = [
    "Area Locality",
    "City",
    "Point of Contact"
]

REQUIRED_COLS = [
    "BHK",
    "Rent",
    "Size",
    "Area Locality",
    "City",
    "Point of Contact"
]


# ==========================================
# 2. LOAD AND CLEAN DATA
# ==========================================

def load_dataset():

    if not FAKE_PATH.exists():
        raise FileNotFoundError(f"Fake dataset not found: {FAKE_PATH}")

    if not TRUE_PATH.exists():
        raise FileNotFoundError(f"Real dataset not found: {TRUE_PATH}")

    fake = pd.read_csv(FAKE_PATH)
    real = pd.read_csv(TRUE_PATH)

    # Clean column names
    fake.columns = fake.columns.str.strip()
    real.columns = real.columns.str.strip()

    # Check required columns
    for name, df in [("Fake", fake), ("Real", real)]:

        missing = [
            col for col in REQUIRED_COLS
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} dataset is missing columns: {missing}"
            )

    fake["class"] = 0
    real["class"] = 1

    data = pd.concat(
        [fake, real],
        ignore_index=True
    )

    print("\nOriginal dataset counts:")
    print(data["class"].value_counts())

    return data


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

def build_features(df):

    out = pd.DataFrame(index=df.index)

    out["BHK"] = pd.to_numeric(
        df["BHK"], errors="coerce"
    )

    out["Rent"] = pd.to_numeric(
        df["Rent"], errors="coerce"
    )

    out["Size"] = pd.to_numeric(
        df["Size"], errors="coerce"
    )

    # Avoid division by zero
    out["Rent_per_sqft"] = (
        out["Rent"] /
        out["Size"].replace(0, np.nan)
    )

    # Reduce the effect of very large rent values
    out["Log_rent"] = np.log1p(
        out["Rent"].clip(lower=0)
    )

    # Categorical features
    for col in CATEGORICAL_COLS:

        out[col] = (
            df[col]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace("", "unknown")
        )

    return out


# ==========================================
# 4. REMOVE DUPLICATE / AMBIGUOUS RECORDS
# ==========================================

def clean_duplicates(X, y):

    # Hash identical feature records
    hashes = pd.util.hash_pandas_object(
        X, index=False
    )

    temp = pd.DataFrame({
        "hash": hashes,
        "class": y.to_numpy()
    })

    # Same listing features with conflicting labels
    conflicting = temp.groupby("hash")["class"].nunique()
    conflicting_hashes = conflicting[
        conflicting > 1
    ].index

    if len(conflicting_hashes) > 0:

        print(
            f"\nRemoving {len(conflicting_hashes)} "
            "ambiguous feature groups."
        )

    valid = ~hashes.isin(conflicting_hashes)

    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    # Remove duplicate feature records with the same label
    combined = X.copy()
    combined["_class"] = y

    before = len(combined)

    combined = combined.drop_duplicates(
        keep="first"
    )

    print(
        f"Duplicate records removed: "
        f"{before - len(combined)}"
    )

    y = combined.pop("_class")
    X = combined

    return X, y


# ==========================================
# 5. PREPROCESSING PIPELINE
# ==========================================

numeric_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    )

])


categorical_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="unknown"
        )
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )

])


preprocessor = ColumnTransformer([

    (
        "numeric",
        numeric_pipeline,
        NUMERIC_COLS
    ),

    (
        "categorical",
        categorical_pipeline,
        CATEGORICAL_COLS
    )

])


# ==========================================
# 6. MACHINE LEARNING MODEL
# ==========================================

model = Pipeline([

    (
        "preprocessing",
        preprocessor
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            random_state=42
        )
    )

])


# ==========================================
# 7. TRAINING
# ==========================================

def main():

    print("\nLoading rental datasets...")

    data = load_dataset()

    X = build_features(data)
    y = data["class"]

    # Remove records without essential numeric inputs
    valid = (
        X["Rent"].notna()
        & X["Size"].notna()
        & X["BHK"].notna()
        & (X["Rent"] >= 0)
        & (X["Size"] > 0)
        & (X["BHK"] >= 0)
    )

    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    # Remove duplicates and conflicting labels
    X, y = clean_duplicates(X, y)

    print("\nFinal class counts:")
    print(
        y.value_counts()
        .rename(index={0: "Fake", 1: "Real"})
    )

    if y.nunique() != 2:
        raise ValueError(
            "Both fake and real classes are required."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Not enough examples in one class."
        )

    # ======================================
    # 8. TRAIN / TEST SPLIT
    # ======================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.25,

        random_state=42,

        stratify=y

    )

    print("\nTraining model...")

    model.fit(X_train, y_train)

    # ======================================
    # 9. MODEL EVALUATION
    # ======================================

    predictions = model.predict(X_test)

    print("\n========== TEST RESULTS ==========")

    print(
        "Accuracy:",
        round(
            accuracy_score(y_test, predictions),
            4
        )
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Fake", "Real"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )
    )

    # ======================================
    # 10. CROSS VALIDATION
    # ======================================

    min_class_count = y.value_counts().min()

    n_splits = min(5, int(min_class_count))

    if n_splits >= 2:

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=[
                "accuracy",
                "precision",
                "recall",
                "f1"
            ],
            zero_division=0
        )

        print(
            f"\n========== {n_splits}-FOLD "
            "CROSS VALIDATION =========="
        )

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]:

            values = scores[f"test_{metric}"]

            print(
                f"{metric.title()}: "
                f"{values.mean():.3f} "
                f"(+/- {values.std():.3f})"
            )

    # ======================================
    # 11. TRAIN FINAL MODEL
    # ======================================

    print("\nTraining final model on all data...")

    model.fit(X, y)

    # ======================================
    # 12. SAVE MODEL AND METADATA
    # ======================================

    bundle = {

        "pipeline": model,

        "version": 2,

        "cities": sorted(
            X["City"].unique().tolist()
        ),

        "localities": sorted(
            X["Area Locality"].unique().tolist()
        ),

        "contacts": sorted(
            X["Point of Contact"].unique().tolist()
        ),

        "ranges": {

            "Rent": (
                float(X["Rent"].min()),
                float(X["Rent"].max())
            ),

            "Size": (
                float(X["Size"].min()),
                float(X["Size"].max())
            ),

            "BHK": (
                float(X["BHK"].min()),
                float(X["BHK"].max())
            )

        },

        "training_records": int(len(X)),

        "class_counts": {
            "fake": int((y == 0).sum()),
            "real": int((y == 1).sum())
        }

    }

    joblib.dump(bundle, MODEL_PATH)

    print("\n===================================")
    print("MODEL TRAINING COMPLETED")
    print("===================================")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Total records: {len(X)}")


if __name__ == "__main__":
    main()
