import streamlit as st
import pandas as pd

st.title("💊 Medicine Advisor")

# Load datasets
reviews = pd.read_csv("Dataset/drug_review_test.csv")
effects = pd.read_csv("Dataset/drugs_side_effects.csv")

tab1, tab2 = st.tabs(["💊 Medicine Recommendation", "⚠️ Side Effects Checker"])

# -------------------------- MEDICINE RECOMMENDATION ------------------------
with tab1:
    st.subheader("Get Best Medicines For Your Condition")

    user_condition = st.text_input("Enter Condition (ex: diabetes, depression, headache)")

    if st.button("Recommend"):
        if user_condition.strip() == "":
            st.warning("Please enter a condition.")
        else:
            # filter by condition
            data = reviews[reviews['condition'].str.lower() == user_condition.lower()]

            if data.empty:
                st.warning("No medicines found for this condition.")
            else:
                top = (
                    data.groupby("drugName")['rating']
                        .mean()
                        .sort_values(ascending=False)
                        .head(5)
                )

                st.success("Top Recommended Medicines:")
                st.table(top)

# -------------------------- SIDE EFFECTS CHECKER ---------------------------
with tab2:
    st.subheader("Check Side Effects of Medicine")

    med_name = st.text_input("Enter Medicine Name")

    if st.button("Show Side Effects"):
        # FIXED COLUMN NAME: drug_name
        data = effects[effects['drug_name'].str.contains(med_name, case=False, na=False)]

        if data.empty:
            st.warning("No side effects found.")
        else:
            for _, row in data.iterrows():
                st.write(f"### {row['drug_name']}")
                st.write(row['side_effects'])
