import streamlit as st
import pickle

def disease_prediction_page():

    st.title("🧪 Multiple Disease Prediction")

    # Load Models
    diabetes_model = pickle.load(open('Saved Models/diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open('Saved Models/heart_disease_model.sav', 'rb'))
    parkinsons_model = pickle.load(open('Saved Models/parkinsons_model.sav', 'rb'))

    choice = st.selectbox("Select Disease to Predict", 
                        ["Diabetes", "Heart Disease", "Parkinsons"])

    # -------------------------- DIABETES ------------------------------
    if choice == "Diabetes":
        st.header("🩸 Diabetes Prediction")

        features = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]

        inputs = []
        for f in features:
            inputs.append(st.number_input(f))

        if st.button("Predict Diabetes"):
            pred = diabetes_model.predict([inputs])[0]

            if pred == 1:
                st.error("⚠️ The person is Diabetic")
            else:
                st.success("✔ The person is NOT Diabetic")

    # -------------------------- HEART DISEASE -------------------------
    if choice == "Heart Disease":
        st.header("❤️ Heart Disease Prediction")

        features = [
            "Age","Sex","CP","Trestbps","Chol","FBS","Restecg",
            "Thalach","Exang","Oldpeak","Slope","CA","Thal"
        ]

        values = []
        for f in features:
            values.append(st.number_input(f))

        if st.button("Predict Heart Disease"):
            pred = heart_disease_model.predict([values])[0]

            if pred == 1:
                st.error("⚠️ Heart Disease Detected")
            else:
                st.success("✔ No Heart Disease")

    # -------------------------- PARKINSONS ---------------------------
    if choice == "Parkinsons":
        st.header("🧠 Parkinson's Prediction")

        feature_names = [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
            "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
            "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
            "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
            "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
        ]

        inputs = []
        for f in feature_names:
            inputs.append(st.number_input(f))

        if st.button("Predict Parkinson's"):
            pred = parkinsons_model.predict([inputs])[0]

            if pred == 1:
                st.error("⚠️ Parkinson's Detected")
            else:
                st.success("✔ No Parkinson's")
