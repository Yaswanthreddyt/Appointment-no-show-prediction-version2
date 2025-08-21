from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import pickle
from config import NEIGHBOURHOOD_DISTANCE_KM, WEATHER_CATEGORIES

app = Flask(__name__)

# Load model and features
try:
    # Prefer unified model in project root
    model = joblib.load("model.pkl")
except Exception:
    # Fallback to legacy path if present
    model = joblib.load("models/random_forest_no_show_model.pkl")

# Load metadata for feature order and categories
try:
    with open("model_metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
        feature_columns = metadata.get('feature_columns')
        weather_categories = metadata.get('weather_categories', WEATHER_CATEGORIES)
        encoders = metadata.get('encoders', {})
except Exception:
    feature_columns = None
    weather_categories = WEATHER_CATEGORIES
    encoders = {}

@app.route('/')
def home():
    return "API is up. Use /predict with POST request."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Extract and compute engineered inputs
        # Gender
        if 'Gender_encoded' in data:
            gender_encoded = int(data.get('Gender_encoded'))
        else:
            gender_raw = str(data.get('Gender', 'F'))
            try:
                gender_encoded = int(encoders.get('gender').transform([gender_raw])[0])
            except Exception:
                gender_encoded = 1 if gender_raw.upper().startswith('M') else 0
        age = float(data.get('Age'))
        neighbourhood = str(data.get('Neighbourhood', ''))
        if 'SMS_received_encoded' in data:
            sms_encoded = int(data.get('SMS_received_encoded'))
        else:
            sms_raw = str(data.get('SMS_received', '0'))
            try:
                sms_encoded = int(encoders.get('sms').transform([sms_raw])[0])
            except Exception:
                sms_encoded = int(sms_raw in ['1', 'True', 'true'])
        hipertension = int(data.get('Hipertension', 0))
        diabetes = int(data.get('Diabetes', 0))
        alcoholism = int(data.get('Alcoholism', 0))
        handcap = int(data.get('Handcap', 0))
        appointment_gap = int(data.get('AppointmentGap', 0))
        scheduled_dow = int(data.get('ScheduledDayOfWeek', 0))
        appointment_dow = int(data.get('AppointmentDayOfWeek', 0))
        scheduled_month = int(data.get('ScheduledMonth', 1))
        appointment_month = int(data.get('AppointmentMonth', 1))
        scheduled_hour = int(data.get('ScheduledHour', 9))
        is_weekend = int(data.get('IsWeekend', 0))
        age_group = int(data.get('Age_Group', 0))
        medical_count = int(data.get('Medical_Conditions_Count', hipertension + diabetes + alcoholism + handcap))

        # Distance from neighbourhood mapping
        distance_map_upper = {k.upper(): v for k, v in NEIGHBOURHOOD_DISTANCE_KM.items()}
        default_distance = float(np.median(list(distance_map_upper.values()))) if len(distance_map_upper) else 3.0
        estimated_distance = float(data.get('EstimatedDistance', distance_map_upper.get(neighbourhood.upper(), default_distance)))

        # Weather one-hot
        weather = str(data.get('Weather', weather_categories[0]))
        weather_one_hot = {f"Weather_{wc}": 1 if weather == wc else 0 for wc in weather_categories}

        # Assemble feature dict
        # Neighbourhood encoded
        try:
            neighbourhood_encoded = int(encoders.get('neighbourhood').transform([neighbourhood])[0])
        except Exception:
            neighbourhood_encoded = hash(neighbourhood) % 100

        features_dict = {
            'Gender_encoded': gender_encoded,
            'Age': age,
            'Neighbourhood_encoded': int(data.get('Neighbourhood_encoded', neighbourhood_encoded)),
            'SMS_received_encoded': sms_encoded,
            'Hipertension': hipertension,
            'Diabetes': diabetes,
            'Alcoholism': alcoholism,
            'Handcap': handcap,
            'AppointmentGap': appointment_gap,
            'ScheduledDayOfWeek': scheduled_dow,
            'AppointmentDayOfWeek': appointment_dow,
            'ScheduledMonth': scheduled_month,
            'AppointmentMonth': appointment_month,
            'ScheduledHour': scheduled_hour,
            'IsWeekend': is_weekend,
            'Age_Group': age_group,
            'Medical_Conditions_Count': medical_count,
            'EstimatedDistance': estimated_distance,
        }
        features_dict.update(weather_one_hot)

        # Build input vector
        if feature_columns:
            input_vector = [features_dict.get(col, 0) for col in feature_columns]
        else:
            default_order = list(features_dict.keys()) + list(weather_one_hot.keys())
            input_vector = [features_dict.get(col, 0) for col in default_order]

        input_df = pd.DataFrame([input_vector], columns=feature_columns if feature_columns else default_order)

        prediction = model.predict(input_df)[0]
        result = "No-show" if prediction == 1 else "Showed up"
        proba = model.predict_proba(input_df)[0].tolist()
        return jsonify({'prediction': int(prediction), 'label': result, 'proba': proba})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
