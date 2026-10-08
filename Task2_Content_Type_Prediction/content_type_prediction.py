"""
Task 2 (Easy) - Netflix Content Type Prediction Model
Machine learning project portfolio

Goal: predict whether a catalogue entry is a Movie or a TV Show using
only descriptive features (release year, rating, main genre, country,
whether a director is credited, when Netflix added it, ...).

Note on leakage (important): three columns would give the answer away,
so they are neutralised on purpose -
* `duration`: movies are "90 min", shows are "2 Seasons" -> not used.
* `listed_in`: genres literally say "TV Dramas" vs "Dramas, Movies",
  so only the genre *theme* (Drama, Comedy, Crime, ...) is kept.
* `rating`: "TV-MA" vs "R" reveals the type, so ratings are grouped
  into maturity bands (Kids / Family / Teens / Adult) instead.
A first version without these fixes scored a meaningless 100%.

Author: Muhammad Abdul Rafay - ML Intern
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parents[1] / "data" / "netflix_dataset.csv"
RANDOM_STATE = 42


MATURITY = {
    "TV-Y": "Kids", "TV-Y7": "Kids", "TV-Y7-FV": "Kids", "TV-G": "Kids",
    "G": "Kids", "PG": "Family", "TV-PG": "Family",
    "PG-13": "Teens", "TV-14": "Teens",
    "R": "Adult", "TV-MA": "Adult", "NC-17": "Adult", "NR": "Adult",
    "UR": "Adult",
}

THEMES = [  # (keyword in the first genre label, clean theme name)
    ("documentar", "Documentaries"), ("stand-up", "Stand-Up"),
    ("children", "Children & Family"), ("family", "Children & Family"),
    ("comed", "Comedies"), ("horror", "Horror"), ("crime", "Crime"),
    ("romantic", "Romance"), ("thriller", "Thrillers"),
    ("anime", "Anime"), ("reality", "Reality"), ("action", "Action"),
    ("dram", "Dramas"), ("international", "International"),
    ("music", "Music"), ("sports", "Sports"), ("sci", "Sci-Fi & Fantasy"),
    ("fantasy", "Sci-Fi & Fantasy"), ("independent", "Independent"),
    ("classic", "Classics"), ("cult", "Classics"), ("faith", "Faith"),
    ("lgbtq", "LGBTQ"), ("myster", "Mysteries"), ("teen", "Teen"),
    ("docuseries", "Documentaries"), ("talk", "Talk"),
]


def genre_theme(listed_in) -> str:
    """Map the first genre label to a theme with no Movie/TV wording."""
    first = str(listed_in).split(",")[0].strip().lower()
    for keyword, theme in THEMES:
        if keyword in first:
            return theme
    return "Other"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a clean feature table from the raw catalogue."""
    d = df.copy()
    d["primary_genre"] = d["listed_in"].apply(genre_theme)
    d["rating_band"] = d["rating"].map(MATURITY).fillna("Adult")
    d["country_main"] = d["country"].fillna("Unknown").str.split(",").str[0].str.strip()
    # Keep only the frequent countries, group the long tail as "Other".
    top_countries = d["country_main"].value_counts().head(10).index
    d["country_main"] = d["country_main"].where(d["country_main"].isin(top_countries), "Other")
    d["director_known"] = (d["director"].notna()).astype(int)
    added = pd.to_datetime(d["date_added"], format="mixed", errors="coerce")
    d["added_year"] = added.dt.year.fillna(d["release_year"])
    d["added_month"] = added.dt.month.fillna(0).astype(int)
    d["title_words"] = d["title"].str.split().str.len()
    d["genres_count"] = d["listed_in"].fillna("").str.count(",") + 1
    return d


NUMERIC = ["release_year", "added_year", "added_month",
           "director_known", "title_words", "genres_count"]
CATEGORICAL = ["rating_band", "primary_genre", "country_main"]


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
    print(f"Dataset rows: {len(raw)}")
    d = build_features(raw)
    X, y = d[NUMERIC + CATEGORICAL], d["type"]
    print("Class balance:", y.value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300,
                                                random_state=RANDOM_STATE, n_jobs=-1),
    }

    results, best_name, best_pipe, best_acc = {}, None, None, -1.0
    for name, model in candidates.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results[name] = round(float(acc), 4)
        print(f"\n{name}: accuracy = {acc:.4f}")
        print(classification_report(y_test, pred))
        if acc > best_acc:
            best_name, best_pipe, best_acc = name, pipe, acc

    print(f"\nBest model: {best_name} ({best_acc:.4f})")

    # Confusion matrix for the best model - saved as the screenshot.
    pred = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=["Movie", "TV Show"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Movie", "TV Show"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(HERE / "confusion_matrix.png", dpi=150)
    plt.close()

    # Accuracy comparison chart.
    plt.figure(figsize=(7, 4))
    plt.bar(results.keys(), results.values(), color=["#999999", "#999999", "#4C72B0"])
    plt.ylabel("Test accuracy")
    plt.ylim(0, 1.05)
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.title("Content Type Prediction - model comparison")
    plt.tight_layout()
    plt.savefig(HERE / "model_comparison.png", dpi=150)
    plt.close()

    (HERE / "results.json").write_text(json.dumps(
        {"dataset_rows": int(len(raw)), "test_rows": int(len(X_test)),
         "accuracy": results, "best_model": best_name}, indent=2), encoding="utf-8")
    print("Saved: confusion_matrix.png, model_comparison.png, results.json")


if __name__ == "__main__":
    main()
