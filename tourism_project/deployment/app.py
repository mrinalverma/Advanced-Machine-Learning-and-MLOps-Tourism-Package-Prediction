import streamlit as st
import pandas as pd
import joblib
import os

# Configure the page
st.set_page_config(page_title="Wellness Package Predictor", page_icon="🌴", layout="centered")

st.title("🌴 Visit with Us: Wellness Package Predictor")
st.write("Enter the customer's details below to predict if they will purchase the Wellness Tourism Package.")

# 1. Load the model committed to the repository
@st.cache_resource
def load_model():
    model_path = "artifacts/model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("❌ Model not found! Please ensure GitHub Actions has finished running and committed 'artifacts/model.pkl' to the repository.")
else:
    # Create an input form
    with st.form("customer_form"):
        st.subheader("Customer Demographics")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        with col2:
            occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
            designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, max_value=500000, value=50000)
        with col3:
            city_tier = st.selectbox("City Tier", [1, 2, 3])
            passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

        st.subheader("Travel & Interaction History")
        col4, col5 = st.columns(2)
        with col4:
            number_of_person_visiting = st.number_input("Total Persons Visiting", min_value=1, max_value=10, value=2)
            number_of_children_visiting = st.number_input("Children (<5 yrs)", min_value=0, max_value=5, value=0)
            number_of_trips = st.number_input("Trips Annually", min_value=0, max_value=20, value=2)
            preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
        with col5:
            type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
            product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
            pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
            number_of_followups = st.slider("Number of Follow-ups", 1, 10, 3)
            duration_of_pitch = st.number_input("Pitch Duration (mins)", min_value=5, max_value=120, value=15)

        # Submit button
        submit_button = st.form_submit_button(label="Predict Purchase Likelihood")

    if submit_button:
        # 2. Get the inputs and save them into a dataframe
        input_dict = {
            'Age': [age],
            'TypeofContact': [type_of_contact],
            'CityTier': [city_tier],
            'Occupation': [occupation],
            'Gender': [gender],
            'NumberOfPersonVisiting': [number_of_person_visiting],
            'PreferredPropertyStar': [preferred_property_star],
            'MaritalStatus': [marital_status],
            'NumberOfTrips': [number_of_trips],
            'Passport': [passport],
            'OwnCar': [own_car],
            'NumberOfChildrenVisiting': [number_of_children_visiting],
            'Designation': [designation],
            'MonthlyIncome': [monthly_income],
            'PitchSatisfactionScore': [pitch_satisfaction_score],
            'ProductPitched': [product_pitched],
            'NumberOfFollowups': [number_of_followups],
            'DurationOfPitch': [duration_of_pitch]
        }

        input_df = pd.DataFrame(input_dict)

        # Predict using the loaded model
        with st.spinner("Analyzing customer profile..."):
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

        # Display Results
        st.markdown("---")
        if prediction == 1:
            st.success(f"### Likely to Purchase!")
            st.write(f"**Confidence Probability:** {probability * 100:.1f}%")
            st.balloons()
        else:
            st.error(f"### Unlikely to Purchase.")
            st.write(f"**Confidence Probability:** {probability * 100:.1f}%")
