import streamlit as st
import joblib
import json
import os
import pandas as pd

def medicine_advisor_page():

    st.set_page_config(page_title="Medicine Advisor", page_icon="💊", layout="wide")
    st.title("💊 Medicine Advisor")

    # Paths
    MODEL_PATH = os.path.join("recommender_assets", "medicine_recommender_model.pkl")
    MAP_PATH = os.path.join("recommender_assets", "condition_drug_map.json")
    REVIEWS_CSV = "Dataset/drug_review_test.csv"
    SIDE_CSV = "Dataset/drugs_side_effects.csv"

    @st.cache_data(ttl=3600)
    def load_assets():
        if not os.path.exists(MODEL_PATH) or not os.path.exists(MAP_PATH):
            return None, None, None, None
        model = joblib.load(MODEL_PATH)
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            cond_map = json.load(f)
        rev_df = pd.read_csv(REVIEWS_CSV, low_memory=False)
        side_df = pd.read_csv(SIDE_CSV, low_memory=False)
        return model, cond_map, rev_df, side_df

    model, cond_map, rev_df, side_df = load_assets()

    if model is None or cond_map is None:
        st.error("Model or mapping not found. Run `python train_medicine_recommender.py` first.")
        st.stop()

    tab1, tab2 = st.tabs(["💊 Recommend by Symptoms / Condition", "⚠️ Side-effects & Drug Lookup"])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("Type symptoms or paste doctor's notes — model will predict a condition and recommend medicines.")

        sym = st.text_area("Describe symptoms (e.g. 'severe headache and fever for 2 days')", height=130)
        manual_cond = st.text_input("Or enter a condition directly (optional, e.g. 'acne')").strip().lower()

        if st.button("Get Recommendations"):
            if manual_cond:
                predicted = manual_cond
                st.info(f"Using manual condition: **{predicted}**")

            elif sym.strip():
                try:
                    predicted = model.predict([sym])[0]
                    st.info(f"Model predicted condition: **{predicted}**")
                except Exception as e:
                    st.error("Prediction failed: " + str(e))
                    predicted = None
            else:
                st.warning("Enter symptoms or a condition first.")
                predicted = None

            if predicted:
                recs = cond_map.get(predicted.lower()) or cond_map.get(predicted)
                if not recs:
                    st.warning("No recommendations available for this condition in the dataset.")
                else:
                    st.markdown("### Top recommended medicines")
                    for r in recs:
                        med = r.get("drugName")
                        rating = r.get("avg_rating")
                        se = r.get("side_effects")

                        st.subheader(med)
                        if rating:
                            st.write(f"Average rating: **{rating:.2f}**")

                        if se:
                            st.write("**Side effects:**")
                            st.write(se)
                        else:
                            # fallback lookup
                            if "drug_name" in side_df.columns:
                                lookup = side_df[
                                    side_df['drug_name'].str.contains(str(med), case=False, na=False)
                                ]
                                if not lookup.empty and 'side_effects' in side_df.columns:
                                    st.write("**Side effects (from dataset):**")
                                    st.write(lookup['side_effects'].iloc[0])
                                else:
                                    st.write("**Side effects:** Not available")
                            else:
                                st.write("**Side effects:** Not available")

    # ---------------- TAB 2 ----------------
    with tab2:
        st.subheader("Search side-effects by medicine name")
        med_search = st.text_input("Enter medicine name (partial ok):")

        if st.button("Search Side Effects"):
            if not med_search.strip():
                st.warning("Enter medicine name first.")
            else:
                possible_drug_col = None
                for c in ["drug_name", "drugName", "medicine", "medicine_name"]:
                    if c in side_df.columns:
                        possible_drug_col = c
                        break

                se_col = "side_effects" if "side_effects" in side_df.columns else None

                if not possible_drug_col:
                    st.error("Drug-name column not found in side-effects CSV.")
                else:
                    matches = side_df[
                        side_df[possible_drug_col].str.contains(med_search, case=False, na=False)
                    ]

                    if matches.empty:
                        st.warning("No matches found for that medicine.")
                    else:
                        for _, row in matches.iterrows():
                            name = row[possible_drug_col]
                            st.write(f"### {name}")

                            if se_col and pd.notna(row.get(se_col)):
                                st.write(row.get(se_col))
                            else:
                                st.write(row.to_dict())
