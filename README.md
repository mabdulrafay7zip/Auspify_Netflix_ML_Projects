# Netflix Machine Learning Projects — Portfolio

**Author:** Muhammad Abdul Rafay — ML Intern
**Internship period:** 1 Oct 2026 – 1 Nov 2026 (remote)

Four Machine Learning tasks (2 Easy + 2 Medium) completed on the
Netflix titles dataset
(8,790 titles; `data/netflix_dataset.csv`). All code is original,
written for this portfolio, and every result quoted below comes from
actually running the scripts in this repository.

| Task | Level | Project | Headline result |
|---|---|---|---|
| 1 | Easy | Netflix Content Recommendation System | TF-IDF + cosine similarity over 8,787 titles; e.g. Midnight Mass → The Haunting of Hill House (0.7267) |
| 2 | Easy | Content Type Prediction (Movie vs TV Show) | Random Forest **91.92%** test accuracy (leakage-controlled features) |
| 3 | Medium | Audience Rating Classification (10 classes) | Tuned Random Forest **50.85%** test accuracy vs 36.5% majority baseline |
| 4 | Medium | Content Segmentation (K-Means) | k = 3 segments chosen by silhouette (0.2082): modern US, modern international, older US catalogue |

## Structure
```
netflix-ml-projects/
├── data/netflix_dataset.csv
├── requirements.txt
└── Projects/
    ├── Task1_Netflix_Recommendation_System/
    ├── Task2_Content_Type_Prediction/
    ├── Task3_Rating_Classification/
    └── Task4_Content_Segmentation/
```
Each task folder contains its runnable `.py` script, a README with the
real results, PNG screenshots of its plots, a `results.json` and the
full `run_output.txt` log.

## How to Run Everything
```bash
pip install -r requirements.txt
cd Projects/Task1_Netflix_Recommendation_System && python recommendation_system.py
cd ../Task2_Content_Type_Prediction && python content_type_prediction.py
cd ../Task3_Rating_Classification && python rating_classification.py   # ~2-3 min (grid search)
cd ../Task4_Content_Segmentation && python content_segmentation.py
```

---
Muhammad Abdul Rafay — ML Intern
