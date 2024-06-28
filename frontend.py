import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_filename = "ml_models/random_forest_model.pkl"
model = pickle.load(open(model_filename, 'rb'))

# Feature mappings
gender_d = {"Female": 0, "Male": 1}
yes_no_d = {"No": 0, "Yes": 1}
caec_d = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
calc_d = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
transport_d = {
    "Automobile": 0,
    "Bike": 1,
    "Motorbike": 2,
    "Public_Transportation": 3,
    "Walking": 4,
}

def encode_features(df):
    le_gender = LabelEncoder()
    le_family_history = LabelEncoder()
    le_highcal_intake = LabelEncoder()
    le_snacking = LabelEncoder()
    le_smoke = LabelEncoder()
    le_track_cal_intake = LabelEncoder()
    le_alcohol_intake = LabelEncoder()
    le_transport_mode = LabelEncoder()

    df['gender'] = le_gender.fit_transform(df['gender'])
    df['family_history_with_overweight'] = le_family_history.fit_transform(df['family_history_with_overweight'])
    df['highcal_intake'] = le_highcal_intake.fit_transform(df['highcal_intake'])
    df['snacking'] = le_snacking.fit_transform(df['snacking'])
    df['smoke'] = le_smoke.fit_transform(df['smoke'])
    df['track_cal_intake'] = le_track_cal_intake.fit_transform(df['track_cal_intake'])
    df['alcohol_intake'] = le_alcohol_intake.fit_transform(df['alcohol_intake'])
    df['transport_mode'] = le_transport_mode.fit_transform(df['transport_mode'])

    return df

def main():
    st.set_page_config(page_title="Obesity Prediction App")
    st.title("Obesity Prediction")

    # CSS for green background
    st.markdown(
        f"""
        <style>
        .appview-container {{
           background: radial-gradient(circle, #3d7c6c, #1b4b53);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    

    # Input fields
    gender = st.radio("Gender", list(gender_d.keys()), format_func=lambda x: x)
    age = st.slider("Age", min_value=1, max_value=100, value=25)
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.75)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    family_history_with_overweight = st.radio("Family History with Overweight", list(yes_no_d.keys()), format_func=lambda x: x)
    FAVC = st.radio("High Caloric Food Consumption", list(yes_no_d.keys()), format_func=lambda x: x)
    FCVC = st.slider("Vegetable Consumption Frequency", min_value=1, max_value=3, value=2)
    NCP = st.slider("Number of Main Meals", min_value=1, max_value=5, value=3)
    CAEC = st.radio("Food Consumption Between Meals", list(caec_d.keys()), format_func=lambda x: x)
    SMOKE = st.radio("Do you smoke?", list(yes_no_d.keys()), format_func=lambda x: x)
    CH2O = st.slider("Water Intake (liters)", min_value=1, max_value=3, value=2)
    SCC = st.radio("Calorie Consumption Monitoring", list(yes_no_d.keys()), format_func=lambda x: x)
    FAF = st.slider("Physical Activity Frequency (days per week)", min_value=0, max_value=7, value=2)
    TUE = st.slider("Time using technology devices (hours per day)", min_value=0, max_value=24, value=5)
    CALC = st.radio("Alcohol Consumption Frequency", list(calc_d.keys()), format_func=lambda x: x)
    MTRANS = st.radio("Transportation Mode", list(transport_d.keys()), format_func=lambda x: x)

    # Prepare input data
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'height': [height],
        'weight': [weight],
        'family_history_with_overweight': [family_history_with_overweight],
        'highcal_intake': [FAVC],
        'veg_intake': [FCVC],
        'meals_daily': [NCP],
        'snacking': [CAEC],
        'smoke': [SMOKE],
        'water_intake_daily': [CH2O],
        'track_cal_intake': [SCC],
        'physical_weekly': [FAF],
        'tech_usage_daily': [TUE],
        'alcohol_intake': [CALC],
        'transport_mode': [MTRANS]
    })

    # Encode categorical features
    input_data = encode_features(input_data)

    # Display input data
    st.write("### Input Data")
    st.write(input_data)

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        obesity_levels = {
            0: "Insufficient Weight",
            1: "Normal Weight",
            2: "Overweight Level I",
            3: "Overweight Level II",
            4: "Obesity Type I",
            5: "Obesity Type II",
            6: "Obesity Type III"
        }

        result = obesity_levels[prediction[0]]
        confidence = prediction_proba[0][prediction[0]] * 100

        # Display prediction
        st.write(f"## Predicted Obesity Level: **{result}**")
        st.write(f"### Confidence: **{confidence:.2f}%**")

if __name__ == "__main__":
    main()
