"""
Task 1 (Easy) - Netflix Content Recommendation System
Machine learning project portfolio

Idea
----
A content-based recommender: every title is turned into one short
"profile" string built from its genres (listed_in), country, director,
type and rating. TF-IDF converts those profiles into vectors and cosine
similarity tells us which titles are closest to a title the user liked.

Author: Muhammad Abdul Rafay - ML Intern
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parents[1] / "data" / "netflix_dataset.csv"


def load_catalogue(path: Path) -> pd.DataFrame:
    """Load the Netflix catalogue and fill the gaps we care about."""
    df = pd.read_csv(path)
    for col in ["director", "country", "rating", "listed_in"]:
        df[col] = df[col].fillna("Unknown")
    # A duplicated title would confuse the lookup, keep the first entry.
    df = df.drop_duplicates(subset="title").reset_index(drop=True)
    return df


def build_profiles(df: pd.DataFrame) -> pd.Series:
    """Combine the descriptive columns into a single text profile.

    People and genre names have their spaces removed (e.g. 'Mike Flanagan'
    becomes 'MikeFlanagan') so TF-IDF treats them as one token instead of
    splitting them into common first/last names.
    """
    def squash(text: str) -> str:
        return str(text).replace(" ", "").replace(",", " ")

    profiles = (
        df["listed_in"].apply(squash) + " "
        + df["country"].apply(squash) + " "
        + df["director"].apply(squash) + " "
        + df["type"].str.replace(" ", "") + " "
        + df["rating"]
    )
    return profiles


class NetflixRecommender:
    """Simple content-based recommender over the Netflix catalogue."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(build_profiles(df))
        # Title -> row position lookup (case-insensitive).
        self.index_of = {t.lower(): i for i, t in enumerate(df["title"])}

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """Return the top_n titles most similar to `title`."""
        key = title.strip().lower()
        if key not in self.index_of:
            raise ValueError(f"'{title}' was not found in the catalogue.")
        pos = self.index_of[key]
        scores = cosine_similarity(self.matrix[pos], self.matrix).ravel()
        # Rank everything, then drop the title itself (score 1.0).
        ranked = scores.argsort()[::-1]
        ranked = [i for i in ranked if i != pos][:top_n]
        out = self.df.iloc[ranked][["title", "type", "country",
                                     "release_year", "rating", "listed_in"]].copy()
        out.insert(1, "similarity", scores[ranked].round(4))
        return out.reset_index(drop=True)


def main() -> None:
    df = load_catalogue(DATA_PATH)
    print(f"Catalogue loaded: {len(df)} unique titles")
    print(f"Columns used for profiles: listed_in, country, director, type, rating")

    recommender = NetflixRecommender(df)
    print(f"TF-IDF matrix shape: {recommender.matrix.shape} "
          f"(titles x vocabulary terms)")

    examples = ["Midnight Mass", "Ganglands", "The Great British Baking Show"]
    report_lines, plot_data = [], {}
    for title in examples:
        if title.lower() not in recommender.index_of:
            continue
        recs = recommender.recommend(title, top_n=10)
        plot_data[title] = recs
        report_lines.append(f"\nTop 10 recommendations for '{title}':")
        report_lines.append(recs.to_string(index=False))
        print(report_lines[-2])
        print(report_lines[-1])

    # Save the example outputs as the submission evidence file.
    (HERE / "recommendations_example.txt").write_text(
        "Netflix Content Recommendation System - example outputs\n"
        f"Catalogue size: {len(df)} titles | TF-IDF features: {recommender.matrix.shape[1]}\n"
        + "\n".join(report_lines), encoding="utf-8")

    summary = {"catalogue_titles": int(len(df)),
               "tfidf_features": int(recommender.matrix.shape[1]),
               "examples": {t: r["title"].tolist() for t, r in plot_data.items()}}
    (HERE / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Screenshot-style plot: similarity of the top picks for the first example.
    first = examples[0]
    if first in plot_data:
        recs = plot_data[first].iloc[::-1]  # ascending for a horizontal bar chart
        plt.figure(figsize=(9, 5))
        plt.barh(recs["title"], recs["similarity"], color="#4C72B0")
        plt.xlabel("Cosine similarity")
        plt.title(f"Top 10 titles most similar to '{first}'")
        plt.tight_layout()
        plt.savefig(HERE / "recommendations_plot.png", dpi=150)
        plt.close()
    print("\nSaved: recommendations_example.txt, results.json, recommendations_plot.png")


if __name__ == "__main__":
    main()
