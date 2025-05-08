import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from disease_preprocess import count_cases
from disease_outbreak import detect_outbreak_per_day, predict_future_outbreaks

# Load the prediction model and label encoder
model = joblib.load('models/disease_prediction_model.pkl')
le = joblib.load('encoders/label_encoder.pkl')

symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
    'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
    'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin',
    'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
    'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
    'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
    'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
    'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
    'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
    'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]


# Streamlit UI for symptom input and disease prediction
def symptom_checker():
    st.subheader("Disease Prediction")

    # Split symptoms into two parts
    half = len(symptoms) // 2
    first_half = symptoms[:half]
    second_half = symptoms[half:]

    # Display first half in two columns
    cols = st.columns(2)
    user_input = {}
    for i, symptom in enumerate(first_half):
        with cols[i % 2]:
            user_input[symptom] = st.checkbox(symptom)

    # Show More Symptoms button
    if st.button("Show More Symptoms"):
        st.write("**Additional Symptoms:**")
        more_cols = st.columns(2)
        for i, symptom in enumerate(second_half):
            with more_cols[i % 2]:
                user_input[symptom] = st.checkbox(symptom)

    # Prediction button
    if st.button("Predict Disease"):
        input_df = pd.DataFrame([user_input])
        
        # Check if at least one symptom is checked
        if not input_df.loc[:, input_df.columns != 'prognosis'].any(axis=None):
            st.warning("Please select at least one symptom.")
            return

        try:
            # Perform prediction
            input_df = input_df.reindex(columns=symptoms, fill_value=0)
            prediction = model.predict(input_df)
            predicted_disease = le.inverse_transform(prediction)[0]
            input_df['prognosis'] = predicted_disease

            # Save to CSV
            csv_path = 'dataset/sample_user_data.csv'
            file_exists = os.path.isfile(csv_path)
            input_df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

            # Display predicted disease
            st.success(f"Predicted Disease: {predicted_disease}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Disease outbreak detection and forecasting
def outbreak_forecasting():
    st.subheader("Outbreak Detection and Forecasting")

    if st.button("Detect and Forecast Outbreaks"):
        try:
            # Read data
            df = pd.read_csv('dataset/sample_user_data.csv')
            
            # Count cases and detect outbreaks
            disease_counts = count_cases(df)
            detected_outbreaks = detect_outbreak_per_day(disease_counts)
            forecasted_outbreaks = predict_future_outbreaks(disease_counts, days=7)
            
            # Display outbreaks and forecasts
            st.write("Detected Outbreaks:", detected_outbreaks)
            st.write("Forecasted Outbreaks:", forecasted_outbreaks)
            
            split_date = detected_outbreaks['date'].max()

            # Add 'source' column to forecasted_outbreaks based on date
            forecasted_outbreaks['source'] = forecasted_outbreaks['date'].apply(
                lambda d: 'historical' if pd.to_datetime(
                    d) <= pd.to_datetime(split_date) else 'forecasted'
            )

            total_outbreaks = forecasted_outbreaks.copy()

            # Ensure date is datetime
            total_outbreaks['date'] = pd.to_datetime(total_outbreaks['date'])
            # Plot outbreak data
            for disease in forecasted_outbreaks['prognosis'].unique():
                disease_data = forecasted_outbreaks[forecasted_outbreaks['prognosis'] == disease]
                fig, ax = plt.subplots(figsize=(10, 5))

                # Plot historical
                hist = disease_data[disease_data['source'] == 'historical']
                ax.plot(hist['date'], hist['cases'], label='Historical', marker='o', color='blue')

                # Plot forecasted
                forecast = disease_data[disease_data['source'] == 'forecasted']
                if not forecast.empty:
                    ax.plot(forecast['date'], forecast['cases'], label='Forecasted', marker='o', color='orange')

                # Mark outbreaks
                outbreak_dates = disease_data[disease_data['outbreak'] == 1]['date']
                outbreak_cases = disease_data[disease_data['outbreak'] == 1]['cases']
                ax.scatter(outbreak_dates, outbreak_cases, color='red', label='Outbreak', zorder=5)

                ax.set_title(f'Disease Cases and Outbreaks: {disease}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Cases')
                ax.legend()

                st.pyplot(fig)

        except Exception as e:
            st.error(f"Outbreak Detection failed: {e}")

# Main Streamlit App
def main():
    tab1, tab2 = st.tabs(["🩺 Disease Prediction", "📊 Outbreak Forecasting"])
    
    with tab1:
        symptom_checker()
    
    with tab2:
        outbreak_forecasting()

if __name__ == "__main__":
    main()
