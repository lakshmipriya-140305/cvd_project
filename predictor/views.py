
import joblib
import pandas as pd

from pathlib import Path
from django.shortcuts import render


# Load the saved ML files

MODEL_DIR = Path(__file__).resolve().parent / 'ml_models'

model = joblib.load(MODEL_DIR / 'final_heart_disease_model.pkl')

preprocessor = joblib.load(MODEL_DIR / 'preprocessor.pkl')

interaction_generator = joblib.load(
    MODEL_DIR / 'interaction_generator.pkl'
)

selected_features = joblib.load(
    MODEL_DIR / 'selected_features.pkl'
)


# Home page

def home(request):
    return render(request, 'home.html')


# Prediction function

def predict(request):

    if request.method == 'POST':

        patient_data = {
            'Age': float(request.POST.get('age')),
            'Sex': request.POST.get('sex'),
            'ChestPainType': request.POST.get('chest_pain'),
            'RestingBP': float(request.POST.get('resting_bp')),
            'Cholesterol': float(request.POST.get('cholesterol')),
            'FastingBS': float(request.POST.get('fasting_bs')),
            'RestingECG': request.POST.get('resting_ecg'),
            'MaxHR': float(request.POST.get('max_hr')),
            'ExerciseAngina': request.POST.get('exercise_angina'),
            'Oldpeak': float(request.POST.get('oldpeak')),
            'ST_Slope': request.POST.get('st_slope'),
        }

        # Convert input into DataFrame

        input_data = pd.DataFrame([patient_data])

        # Apply preprocessing

        processed_data = preprocessor.transform(input_data)

        # Generate interaction features

        interaction_data = interaction_generator.transform(
            processed_data
        )

        # Select important features

        final_input = interaction_data[:, selected_features]

        # Make prediction

        prediction = int(model.predict(final_input).ravel()[0])

        probability = model.predict_proba(final_input)[0][1] * 100

        # Display result

        return render(
            request,
            'result.html',
            {
                'patient_data': patient_data,
                'prediction': prediction,
                'probability': round(float(probability), 2)
            }
        )

    return render(request, 'home.html')
