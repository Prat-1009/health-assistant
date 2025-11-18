import streamlit as st
from streamlit_option_menu import option_menu

from screens.disease_prediction import disease_prediction_page
from screens.medicine_advisor import medicine_advisor_page

st.set_page_config(
    page_title="Health Assistant",
    page_icon="🧑‍⚕️",
    layout="wide"
)


# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: -10px;'>🧑‍⚕️ Health Assistant</h1>
    <p style='text-align: center; font-size: 18px; color: #cccccc;'>
        A smart medical support system that helps you assess health risks 
        and discover safe, suitable medicines instantly.
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- Sidebar Menu ----------------
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Disease Prediction", "Medicine Advisor"],
        icons=["house", "activity", "capsule"],
        default_index=0
    )


# ---------------- Content Rendering ----------------
if selected == "Home":

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align: center;">
            <h2 style="margin-bottom: 5px;">Welcome 👋</h2>
            <p style="font-size: 17px; color: #cccccc; max-width: 700px; margin: auto;">
                This system helps you check disease predictions and get proper medicine suggestions instantly.
                Use the left navigation menu to explore features.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- Add Professional Line Here ----
    st.markdown(
        """
        <p style="text-align:center; font-size:16px; color:#bbbbbb; margin-top:25px;">
            Our AI-powered system analyzes medical patterns to support early detection and better health decisions.
        </p>
        """,
        unsafe_allow_html=True
    )

    # ---- Add Feature Highlights ----
    st.markdown(
        """
        <div style="text-align:center; margin-top:30px;">
            <h3 style="color:#4CC9F0;">✨ Key Features</h3>
            <p style="color:#cccccc;">✔ Instant Disease Prediction<br>
            ✔ Smart Medicine Recommendations<br>
            ✔ Side Effects Detection<br>
            ✔ Fast & Secure Analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- Add Footer ----
    st.markdown(
        """
        <p style="text-align:center; color:#888888; margin-top:40px;">
            Start by selecting any option from the sidebar to continue.
        </p>
        """,
        unsafe_allow_html=True
    )


    

elif selected == "Disease Prediction":
    disease_prediction_page()

elif selected == "Medicine Advisor":
    medicine_advisor_page()
