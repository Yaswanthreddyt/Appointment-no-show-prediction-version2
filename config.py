"""
Configuration file for the AI-Based No-Show Appointment Prediction Tool
"""

# Model Configuration
MODEL_CONFIG = {
    'model_path': 'model.pkl',
    'algorithm': 'RandomForest',
    'n_estimators': 100,
    'random_state': 42,
    'test_size': 0.2
}

# Feature Configuration
FEATURES = {
    'categorical': ['Gender', 'Neighbourhood', 'SMS_received', 'Weather'],
    'numerical': ['Age', 'AppointmentGap', 'EstimatedDistance'],
    'binary': ['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap']
}

# Data Configuration
DATA_CONFIG = {
    'data_path': 'Data/KaggleV2-May-2016.csv',
    'target_column': 'No-show',
    'positive_class': 'Yes',  # Class indicating no-show
    'date_columns': ['ScheduledDay', 'AppointmentDay']
}

# Domain Constants
# Estimated distances (in km) from neighbourhood to clinic (mock values)
NEIGHBOURHOOD_DISTANCE_KM = {
    'JARDIM CAMBURI': 2.3,
    'MARUÍPE': 5.1,
    'SANTA LUCIA': 3.8,
    'CENTRO': 1.5,
    'ITARARÉ': 6.2,
    'ROMÃO': 4.7,
    'GOIABEIRAS': 3.2,
    'MARIA ORTIZ': 4.1,
    'PRAIA DO CANTO': 2.9,
    'ILHA DO PRÍNCIPE': 5.9,
    'JARDIM DA PENHA': 5.0,
    'MATA DA PRAIA': 5.0,
    'PONTAL DE CAMBURI': 5.0,
    'REPÚBLICA': 5.0,
    'ANDORINHAS': 5.0,
    'CONQUISTA': 5.0,
    'NOVA PALESTINA': 5.0,
    'DA PENHA': 5.0,
    'TABUAZEIRO': 5.0,
    'BENTO FERREIRA': 5.0,
    'SÃO PEDRO': 5.0,
    'SANTA MARTHA': 5.0,
    'SÃO CRISTÓVÃO': 5.0,
    'GRANDE VITÓRIA': 5.0,
    'RESISTÊNCIA': 5.0,
    'ILHA DE SANTA MARIA': 5.0,
    'ILHA DAS CAIEIRAS': 5.0,
    'JUCUTUQUARA': 5.0,
    'MONTE BELO': 5.0,
    'SÃO BENEDITO': 5.0,
    'SANTO ANDRÉ': 5.0,
    'SOLON BORGES': 5.0,
    'BONFIM': 5.0,
    'MÁRIO CYPRESTE': 5.0,
    'SANTO ANTÔNIO': 5.0,
    'BELA VISTA': 5.0,
    'PRAIA DO SUÁ': 5.0,
    'SANTA HELENA': 5.0,
    'REDENÇÃO': 5.0,
    'SÃO JOSÉ': 5.0,
    'GURIGICA': 5.0,
    'ARIOVALDO FAVALESSA': 5.0,
    'CARATOÍRA': 5.0,
    'CONSOLAÇÃO': 5.0,
    'PARQUE MOSCOSO': 5.0,
    'DO MOSCOSO': 5.0,
    'SANTOS DUMONT': 5.0,
    'SANTA CLARA': 5.0,
    'SANTA LUÍZA': 5.0,
    'BARRO VERMELHO': 5.0,
    'ESTRELINHA': 5.0,
    'FORTE SÃO JOÃO': 5.0,
    'FONTE GRANDE': 5.0,
    'PIEDADE': 5.0,
    'JESUS DE NAZARETH': 5.0,
    'ENSEADA DO SUÁ': 5.0,
    'MORADA DE CAMBURI': 5.0,
    'SANTOS REIS': 5.0
}

# Weather categories used for one-hot encoding
WEATHER_CATEGORIES = ['Sunny', 'Rainy', 'Cloudy']

# UI Configuration
UI_CONFIG = {
    'page_title': 'AI-Based No-Show Appointment Prediction Tool',
    'page_icon': '🏥',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'max_width': 1200
}

# Validation Rules
VALIDATION = {
    'age_range': (0, 120),
    'max_appointment_gap': 365,  # Maximum days between scheduled and appointment
    'required_fields': ['Gender', 'Age', 'ScheduledDate', 'AppointmentDate']
}

# Feedback Configuration
FEEDBACK_CONFIG = {
    'feedback_file': 'user_feedback.csv',
    'collect_feedback': True,
    'feedback_fields': ['timestamp', 'prediction', 'confidence', 'accurate', 'wrong', 'comments', 'features']
}

# Visualization Configuration
VIZ_CONFIG = {
    'chart_height': 400,
    'color_scheme': 'Blues',
    'show_feature_importance': True,
    'show_probability_charts': True
}

# Disclaimer Text
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"
DISCLAIMER_TEXT = """
⚠️ **Disclaimer:** This tool is for demonstration purposes only and not for medical use. 
Predictions are based on historical data patterns and should not be used to make medical decisions.
"""

# About Section Text
ABOUT_TEXT = """
**AI-Based No-Show Appointment Prediction Tool**

This application uses a Random Forest machine learning model to predict the likelihood of a patient 
showing up for their medical appointment based on various factors including:

- **Demographics**: Gender, Age, Neighbourhood
- **Appointment Details**: Scheduled date, Appointment date, Gap between dates
- **Communication**: Whether SMS reminders were sent
- **Medical Conditions**: Hipertension, Diabetes, Alcoholism, Handcap

**How it works:**
1. The model was trained on historical appointment data
2. It analyzes patterns in the input features
3. It provides a prediction with confidence score
4. Feature importance shows which factors most influence the prediction

**Model Performance:**
- Uses Random Forest algorithm for robust predictions
- Provides probability scores for both outcomes
- Shows feature importance for interpretability
"""
