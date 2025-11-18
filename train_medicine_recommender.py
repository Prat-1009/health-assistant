"""
train_medicine_recommender.py

- Expects:
    Dataset/drug_review_test.csv
    Dataset/drugs_side_effects.csv

- Produces:
    recommender_assets/medicine_recommender_model.pkl
    recommender_assets/condition_drug_map.json
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------- Config -----------------
REVIEWS_CSV = "Dataset/drug_review_test.csv"
SIDE_CSV = "Dataset/drugs_side_effects.csv"
OUT_DIR = "recommender_assets"
TOP_N_CONDITIONS = 30   # how many distinct conditions to keep for model
TOP_DRUGS_PER_CONDITION = 7

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------- Load CSVs -----------------
print("Loading CSVs...")
df_rev = pd.read_csv(REVIEWS_CSV, low_memory=False)
df_side = pd.read_csv(SIDE_CSV, low_memory=False)

print("Review columns:", df_rev.columns.tolist())
print("Side-effect columns:", df_side.columns.tolist())

# ----------------- Normalize and select columns -----------------
# Reviews CSV: we expect columns 'drugName', 'condition', 'review', 'rating' (rating optional)
rev_drug_col = "drugName" if "drugName" in df_rev.columns else None
rev_cond_col = "condition" if "condition" in df_rev.columns else None
rev_review_col = "review" if "review" in df_rev.columns else None
rev_rating_col = "rating" if "rating" in df_rev.columns else None

if not (rev_drug_col and rev_cond_col and rev_review_col):
    raise RuntimeError("Reviews CSV must contain columns: drugName, condition, review")

df_rev = df_rev[[rev_drug_col, rev_cond_col, rev_review_col] + ([rev_rating_col] if rev_rating_col else [])].dropna(subset=[rev_cond_col, rev_review_col])
df_rev.columns = ["drugName", "condition", "review"] + (["rating"] if rev_rating_col else [])

# Side-effect CSV: common columns are 'drug_name' and 'side_effects'
side_drug_col = "drug_name" if "drug_name" in df_side.columns else None
side_se_col = None
for c in ["side_effects", "sideEffects", "adverse_reactions"]:
    if c in df_side.columns:
        side_se_col = c
        break

# We'll rename side_effects col to 'side_effects' if present
if side_drug_col:
    rename_map = {side_drug_col: "drug_name"}
    if side_se_col:
        rename_map[side_se_col] = "side_effects"
    df_side = df_side.rename(columns=rename_map)
else:
    raise RuntimeError("Side-effects CSV must have a column with drug names (e.g., 'drug_name').")

# Normalize text
df_rev["condition"] = df_rev["condition"].astype(str).str.lower().str.strip()
df_rev["review"] = df_rev["review"].astype(str).str.replace('"', " ").str.strip()
df_side["drug_name"] = df_side["drug_name"].astype(str).str.strip()

# ----------------- Keep top conditions for manageable model -----------------
top_conditions = df_rev["condition"].value_counts().nlargest(TOP_N_CONDITIONS).index.tolist()
df_train = df_rev[df_rev["condition"].isin(top_conditions)].copy()

if df_train.shape[0] < 50:
    print("Warning: training size is small:", df_train.shape)

# ----------------- Train TF-IDF + LogisticRegression -----------------
print(f"Training classifier on {len(top_conditions)} conditions and {df_train.shape[0]} samples...")
X = df_train["review"]
y = df_train["condition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=42, stratify=y)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
acc = pipeline.score(X_test, y_test)
print(f"Validation accuracy (review→condition): {acc:.3f}")

# Save model
model_path = os.path.join(OUT_DIR, "medicine_recommender_model.pkl")
joblib.dump(pipeline, model_path)
print("Saved model to:", model_path)

# ----------------- Build condition -> top drugs mapping -----------------
# Combine rating info where available. Use df_rev (which has drugName + condition + optional rating)
frames = []

if "rating" in df_rev.columns:
    frames.append(df_rev[["drugName", "condition", "rating"]])
else:
    frames.append(df_rev[["drugName", "condition"]].assign(rating=None))

# side effects may include rating info too (column 'rating' sometimes)
if "rating" in df_side.columns:
    side_subset = df_side[["drug_name", "rating"]].rename(columns={"drug_name":"drugName"})
    # side file doesn't always have condition; we'll combine only drug-level rating
    # but to keep mapping per condition we mostly rely on review dataset
    # we append side info with null condition so it won't affect condition->drug avg.
    frames.append(side_subset.assign(condition=None))
else:
    # nothing extra
    pass

combined = pd.concat(frames, ignore_index=True, sort=False)
combined = combined.dropna(subset=["drugName"])

# Calculate per-condition, per-drug average rating. If rating missing, we fallback to count
if "rating" in combined.columns and combined["rating"].notnull().any():
    avg = combined.dropna(subset=["condition","rating"]).groupby(["condition","drugName"]).rating.mean().reset_index().rename(columns={"rating":"avg_rating"})
else:
    # fallback: count of occurrences in review dataset per condition/drug
    counts = df_rev.groupby(["condition","drugName"]).size().reset_index().rename(columns={0:"avg_rating"})
    avg = counts

# Side effects lookup map (drug_name -> side_effects)
side_map = {}
if "side_effects" in df_side.columns:
    for _, row in df_side.dropna(subset=["drug_name"]).iterrows():
        dn = row["drug_name"]
        if dn not in side_map and pd.notna(row.get("side_effects")):
            side_map[dn] = row.get("side_effects")

# Build mapping dict
condition_drug_map = {}
for cond, grp in avg.groupby("condition"):
    if cond is None or pd.isna(cond):
        continue
    top = grp.sort_values("avg_rating", ascending=False).head(TOP_DRUGS_PER_CONDITION)
    recs = []
    for _, r in top.iterrows():
        drug = r["drugName"]
        recs.append({
            "drugName": drug,
            "avg_rating": float(r["avg_rating"]) if not pd.isna(r["avg_rating"]) else None,
            "side_effects": side_map.get(drug) or side_map.get(drug.lower())
        })
    condition_drug_map[cond] = recs

# Save mapping
map_path = os.path.join(OUT_DIR, "condition_drug_map.json")
with open(map_path, "w", encoding="utf-8") as f:
    json.dump(condition_drug_map, f, ensure_ascii=False, indent=2)

print("Saved condition->drug map to:", map_path)
print("Number of conditions in mapping:", len(condition_drug_map))
