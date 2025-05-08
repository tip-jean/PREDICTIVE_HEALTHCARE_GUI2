import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

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


class SymptomCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Disease Prediction - SymbiPredict')
        self.vars = {}
        self.df = pd.DataFrame(columns=symptoms)

        self.init_df()
        self.create_widgets()

    def init_df(self):
        for symptom in symptoms:
            self.df[symptom] = 0
        self.df['date'] = pd.to_datetime('today').normalize()
        self.df['prognosis'] = ''

    def create_widgets(self):
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(
            self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set, height=400)

        for i, symptom in enumerate(symptoms):
            var = tk.IntVar()
            cb = ttk.Checkbutton(scrollable_frame, text=symptom, variable=var)
            cb.grid(row=i, column=0, sticky='w')
            self.vars[symptom] = var

        canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')

        predict_btn = ttk.Button(
            self.root, text='Predict Disease', command=self.predict)
        predict_btn.grid(row=1, column=0, pady=10)

    def predict(self):
        import os
        from datetime import datetime
        user_input = {symptom: self.vars[symptom].get()
                      for symptom in symptoms}
        input_df = pd.DataFrame([user_input])

        try:
            prediction = model.predict(input_df)
            predicted_disease = le.inverse_transform(prediction)
            input_df['date'] = datetime.now().strftime('%Y-%m-%d')
            input_df['prognosis'] = predicted_disease[0]

            csv_path = 'dataset/sample_user_data.csv'
            file_exists = os.path.isfile(csv_path)
            input_df.to_csv(csv_path, mode='a',
                            header=not file_exists, index=False)

            last_row = pd.read_csv(csv_path).tail(1)
            messagebox.showinfo(
                'Prediction', f'Predicted Disease: {predicted_disease[0]}')

            print('Latest entry:\n', last_row)

            self.df = last_row
            self.df.reset_index(drop=True, inplace=True)
        except Exception as e:
            messagebox.showerror('Error', f'Prediction failed: {e}')
            return None


def check_symptoms():
    root = tk.Tk()
    app = SymptomCheckerApp(root)

    root.mainloop()

    return app.df