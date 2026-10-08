"""
Task 4 (Medium) - Netflix Content Segmentation (K-Means Clustering)
Machine learning project portfolio

Goal: group Netflix titles into meaningful segments without using any
label. Features mix numeric values (release year, number of genres) and
encoded categorical values (rating maturity, genre theme, main country).

Format features (Movie-vs-TV flag, raw duration in minutes vs seasons)
are deliberately excluded: a first run with them included simply split
the catalogue into "movies" and "TV shows" (k=2), which is already
known and tells us nothing about the content itself.
The number of clusters is chosen by looking at BOTH the elbow method
(inertia) and silhouette scores, and the segments are visualised with a
2-D PCA projection.

Author: Muhammad Abdul Rafay - ML Intern
"""

from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parents[1] / "data" / "netflix_dataset.csv"
RANDOM_STATE = 42

# Coarse maturity grouping keeps the rating signal without 17 sparse levels.
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
    ("talk", "Talk"),
]


def genre_theme(listed_in) -> str:
    """First genre label -> clean theme (no 'TV'/'Movies' wording)."""
    first = str(listed_in).split(",")[0].strip().lower()
    for keyword, theme in THEMES:
        if keyword in first:
            return theme
    return "Other"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["primary_genre"] = d["listed_in"].apply(genre_theme)
    d["country_main"] = d["country"].fillna("Unknown").str.split(",").str[0].str.strip()
    top_countries = d["country_main"].value_counts().head(6).index
    d["country_main"] = d["country_main"].where(d["country_main"].isin(top_countries), "Other")
    d["maturity"] = d["rating"].map(MATURITY).fillna("Adult")
    nums = d["duration"].astype(str).str.extract(r"(\d+)")[0]
    d["duration_value"] = pd.to_numeric(nums, errors="coerce")
    d["is_movie"] = (d["type"] == "Movie").astype(int)
    d["genres_count"] = d["listed_in"].fillna("").str.count(",") + 1
    return d


# is_movie / duration_value are still computed above, but only to
# *describe* the finished clusters - they are NOT clustering features.
NUMERIC = ["release_year", "genres_count"]
CATEGORICAL = ["maturity", "primary_genre", "country_main"]


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    d = build_features(raw)
    print(f"Rows clustered: {len(d)}")

    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    X = pre.fit_transform(d[NUMERIC + CATEGORICAL])
    if hasattr(X, "toarray"):
        X = X.toarray()
    print(f"Feature matrix after encoding: {X.shape}")

    # --- Choosing k: elbow (inertia) + silhouette -----------------------
    ks = list(range(2, 11))
    inertias, silhouettes = [], []
    rng_sample = np.random.RandomState(RANDOM_STATE).choice(
        len(d), size=min(2000, len(d)), replace=False)  # silhouette is O(n^2)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X[rng_sample], labels[rng_sample])))
        print(f"k={k}: inertia={km.inertia_:.0f}  silhouette={silhouettes[-1]:.4f}")

    best_k = ks[int(np.argmax(silhouettes))]
    print(f"\nChosen k = {best_k} (highest silhouette score)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ks, inertias, "o-")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow method")
    axes[1].plot(ks, silhouettes, "o-", color="#4C72B0")
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"chosen k={best_k}")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette score by k"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(HERE / "elbow_and_silhouette.png", dpi=150)
    plt.close()

    # --- Final model + PCA visualisation ---------------------------------
    final = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    d["cluster"] = final.fit_predict(X)

    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=d["cluster"],
                          cmap="tab10", s=8, alpha=0.6)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PCA component 1"); plt.ylabel("PCA component 2")
    plt.title(f"Netflix content segments (K-Means, k={best_k}) - PCA view")
    plt.tight_layout()
    plt.savefig(HERE / "cluster_visualisation.png", dpi=150)
    plt.close()

    # --- Describe each segment in plain numbers, then interpret ----------
    profile = d.groupby("cluster").agg(
        titles=("title", "count"),
        pct_movies=("is_movie", "mean"),
        avg_release_year=("release_year", "mean"),
        avg_duration=("duration_value", "mean"),
        top_genre=("primary_genre", lambda s: s.mode().iloc[0]),
        top_country=("country_main", lambda s: s.mode().iloc[0]),
        top_maturity=("maturity", lambda s: s.mode().iloc[0]),
    ).round(2)
    print("\nCluster profile:\n", profile.to_string())
    profile.to_csv(HERE / "cluster_profile.csv")

    lines = [f"Netflix Content Segmentation - k={best_k} clusters, "
             f"silhouette={max(silhouettes):.4f}\n"]
    for c, row in profile.iterrows():
        share = 100 * row["pct_movies"]
        kind = "mostly movies" if share >= 60 else ("mostly TV shows" if share <= 40 else "a mix of movies and TV shows")
        lines.append(
            f"Cluster {c} ({int(row['titles'])} titles): {kind} "
            f"({share:.0f}% movies), centred on {row['top_genre']} from "
            f"{row['top_country']}, maturity '{row['top_maturity']}', "
            f"average release year {row['avg_release_year']:.0f}, "
            f"average duration value {row['avg_duration']:.0f}.")
    interpretation = "\n".join(lines)
    (HERE / "cluster_interpretation.txt").write_text(interpretation, encoding="utf-8")
    print("\n" + interpretation)

    (HERE / "results.json").write_text(json.dumps(
        {"rows": int(len(d)), "features": int(X.shape[1]), "chosen_k": int(best_k),
         "silhouette_by_k": {str(k): round(s, 4) for k, s in zip(ks, silhouettes)},
         "cluster_sizes": {str(int(k)): int(v) for k, v in
                           d["cluster"].value_counts().sort_index().items()}},
        indent=2), encoding="utf-8")
    print("\nSaved: elbow_and_silhouette.png, cluster_visualisation.png, "
          "cluster_profile.csv, cluster_interpretation.txt, results.json")


if __name__ == "__main__":
    main()
