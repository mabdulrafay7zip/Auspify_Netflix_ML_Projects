"""
Task 3 (Medium) - Netflix Audience Rating Classification
Auspify Technologies ML Internship

Goal: predict a title's audience rating (TV-MA, TV-14, PG-13, R, ...)
from its type, release year, duration, main genre, country and a few
engineered features. Decision Tree and Random Forest are trained and
compared, and a small grid search tunes the Random Forest.

Ratings that appear only a handful of times (e.g. 'NC-17', 'TV-Y') are
merged into an "Other" group so every class has enough examples to learn
from. Rows with a missing rating are dropped.

Author: Muhammad Abdul Rafay - Auspify Technologies ML Intern
"""

from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parents[1] / "data" / "netflix_dataset.csv"
RANDOM_STATE = 42
MIN_CLASS_COUNT = 100  # rarer ratings are grouped into "Other"


def parse_duration(value) -> tuple:
    """'90 min' -> (90, 'min'); '2 Seasons' -> (2, 'season')."""
    if pd.isna(value):
        return (float("nan"), "unknown")
    m = re.match(r"(\d+)", str(value))
    number = float(m.group(1)) if m else float("nan")
    return (number, "season" if "Season" in str(value) else "min")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["rating"]).copy()
    counts = d["rating"].value_counts()
    d["rating_group"] = d["rating"].where(d["rating"].isin(
        counts[counts >= MIN_CLASS_COUNT].index), "Other")
    d["primary_genre"] = d["listed_in"].fillna("Unknown").str.split(",").str[0].str.strip()
    d["country_main"] = d["country"].fillna("Unknown").str.split(",").str[0].str.strip()
    top_countries = d["country_main"].value_counts().head(10).index
    d["country_main"] = d["country_main"].where(d["country_main"].isin(top_countries), "Other")
    parsed = d["duration"].apply(parse_duration)
    d["duration_value"] = [p[0] for p in parsed]
    d["duration_unit"] = [p[1] for p in parsed]
    d["director_known"] = (d["director"].notna()).astype(int)
    d["content_age"] = 2021 - d["release_year"]  # dataset snapshot ends in 2021
    return d


NUMERIC = ["release_year", "duration_value", "director_known", "content_age"]
CATEGORICAL = ["type", "primary_genre", "country_main", "duration_unit"]


def make_pipeline(model) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    d = build_features(raw)
    print(f"Rows with a usable rating: {len(d)} of {len(raw)}")
    print("Rating classes:", d["rating_group"].value_counts().to_dict())

    X, y = d[NUMERIC + CATEGORICAL], d["rating_group"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Classes: {y.nunique()}")

    tree = make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE))
    tree.fit(X_train, y_train)
    tree_acc = accuracy_score(y_test, tree.predict(X_test))
    print(f"\nDecision Tree accuracy: {tree_acc:.4f}")

    forest = make_pipeline(RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))
    forest.fit(X_train, y_train)
    forest_acc = accuracy_score(y_test, forest.predict(X_test))
    print(f"Random Forest accuracy: {forest_acc:.4f}")

    # Light tuning of the forest (small grid, 3-fold CV on the train split).
    grid = GridSearchCV(
        make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        {"model__n_estimators": [200, 400],
         "model__max_depth": [None, 20],
         "model__min_samples_leaf": [1, 2]},
        cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    tuned_acc = accuracy_score(y_test, grid.predict(X_test))
    print(f"Tuned Random Forest accuracy: {tuned_acc:.4f} | best params: {grid.best_params_}")

    best_pipe, best_name, best_acc = grid, "Random Forest (tuned)", tuned_acc
    if forest_acc > best_acc:
        best_pipe, best_name, best_acc = forest, "Random Forest", forest_acc
    if tree_acc > best_acc:
        best_pipe, best_name, best_acc = tree, "Decision Tree", tree_acc

    report = classification_report(y_test, best_pipe.predict(X_test))
    print(f"\nBest model: {best_name} ({best_acc:.4f})\n{report}")
    (HERE / "classification_report.txt").write_text(
        f"Best model: {best_name} | test accuracy: {best_acc:.4f}\n\n{report}",
        encoding="utf-8")

    # Accuracy comparison plot.
    labels = ["Decision Tree", "Random Forest", "Random Forest (tuned)"]
    values = [tree_acc, forest_acc, tuned_acc]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color="#4C72B0")
    plt.ylabel("Test accuracy")
    plt.ylim(0, max(values) * 1.25)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.title("Rating Classification - accuracy comparison")
    plt.tight_layout()
    plt.savefig(HERE / "accuracy_comparison.png", dpi=150)
    plt.close()

    # Confusion matrix for the best model.
    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, best_pipe.predict(X_test), labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(cm, display_labels=labels_sorted).plot(
        ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(HERE / "confusion_matrix.png", dpi=150)
    plt.close()

    (HERE / "results.json").write_text(json.dumps(
        {"dataset_rows": int(len(raw)), "rows_used": int(len(d)),
         "classes": int(y.nunique()),
         "accuracy": {"Decision Tree": round(float(tree_acc), 4),
                      "Random Forest": round(float(forest_acc), 4),
                      "Random Forest (tuned)": round(float(tuned_acc), 4)},
         "best_model": best_name,
         "best_params": {k: (None if v is None else v)
                         for k, v in grid.best_params_.items()}},
        indent=2), encoding="utf-8")
    print("Saved: classification_report.txt, accuracy_comparison.png, "
          "confusion_matrix.png, results.json")


if __name__ == "__main__":
    main()
