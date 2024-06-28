import random
import streamlit as st
import base64
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model_filename = "ml_models/random_forest_model.pkl"
model = pickle.load(open(model_filename, 'rb'))

# Feature mappings
gender_d = {"Female": 0, "Male": 1}
yes_no_d = {"No": 0, "Yes": 1}
caec_d = {"Never": 0, "Sometimes": 1, "Frequently": 2, "Daily": 3}
calc_d = {"Never": 0, "Sometimes": 1, "Frequently": 2, "Daily": 3}
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

def get_image_as_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

def main():
    st.set_page_config(page_title="HealthMate", page_icon="🥑")
    st.title("HealthMate")

    image_base64 = get_image_as_base64("avocado.png")

    st.markdown(
        f'<div class="img-container"><img src="data:image/png;base64,{image_base64}" alt="avocado" style="width:100%;"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .appview-container {
            background: radial-gradient(circle, #5EAA9A, #34746c);
        }
        .img-container img {
            max-height: 600px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=1, max_value=100, value=25)
        gender = st.radio("Gender", list(gender_d.keys()), format_func=lambda x: x)
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.75)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)

    with col2:
        FCVC = st.slider("Vegetable Consumption Frequency", min_value=1, max_value=3, value=2)
        NCP = st.slider("Number of Main Meals", min_value=1, max_value=5, value=3)
        CH2O = st.slider("Water Intake (liters)", min_value=1, max_value=3, value=2)
        FAF = st.slider("Physical Activity Frequency (days per week)", min_value=0, max_value=7, value=2)
        TUE = st.slider("Time using technology devices (hours per day)", min_value=0, max_value=24, value=5)

    col3, col4 = st.columns(2)

    with col3:
        SCC = st.radio("Calorie Consumption Monitoring", list(yes_no_d.keys()), format_func=lambda x: x)
        FAVC = st.radio("High Caloric Food Consumption", list(yes_no_d.keys()), format_func=lambda x: x)
        SMOKE = st.radio("Do you smoke?", list(yes_no_d.keys()), format_func=lambda x: x)
        family_history_with_overweight = st.radio("Family History with Overweight", list(yes_no_d.keys()), format_func=lambda x: x)
        

    with col4:
        CAEC = st.radio("Food Consumption Between Meals", list(caec_d.keys()), format_func=lambda x: x)
        CALC = st.radio("Alcohol Consumption Frequency", list(calc_d.keys()), format_func=lambda x: x)
        MTRANS = st.radio("Transportation", list(transport_d.keys()), format_func=lambda x: x)

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

    input_data = encode_features(input_data)

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

        st.write(f"## Predicted Obesity Level: **{result}**")
        st.write(f"### Confidence: **{confidence:.2f}%**")

        if result != "Normal Weight":
            health_tips = [
                "Try to eat more fruits and vegetables.",
                "Exercise regularly to maintain a healthy weight.",
                "Stay hydrated by drinking plenty of water.",
                "Reduce your intake of sugary drinks and snacks.",
                "Ensure you get enough sleep every night."
            ]
            st.write(f"### Health Tip: **{random.choice(health_tips)}**")

if __name__ == "__main__":
    main()
