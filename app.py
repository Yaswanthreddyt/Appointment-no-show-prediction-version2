import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from config import NEIGHBOURHOOD_DISTANCE_KM, WEATHER_CATEGORIES

# Page configuration
st.set_page_config(
    page_title="Hospital No-Show Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for hospital-grade styling
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #f8fafc;
        padding: 0;
        margin: 0;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #ffffff;
        border-radius: 10px;
        padding: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    
    /* Card Styling */
    .prediction-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .prediction-show {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    
    .prediction-no-show {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }
    
    .prediction-medium {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    
    /* Risk Indicator */
    .risk-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-low {
        background-color: #dcfce7;
        color: #166534;
        border: 2px solid #10b981;
    }
    
    .risk-medium {
        background-color: #fef3c7;
        color: #92400e;
        border: 2px solid #f59e0b;
    }
    
    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
        border: 2px solid #ef4444;
    }
    
    /* Timeline Styling */
    .timeline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .timeline-item {
        text-align: center;
        flex: 1;
    }
    
    .timeline-item h4 {
        font-size: 0.9rem;
        margin: 0 0 0.3rem 0;
    }
    
    .timeline-item p {
        font-size: 0.8rem;
        margin: 0;
        font-weight: 600;
    }
    
    .timeline-arrow {
        font-size: 1.2rem;
        color: #64748b;
    }
    
    /* Form Styling */
    .form-section {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }
    
    .form-section h3 {
        color: #1e293b;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 0.5rem;
    }
    
    /* Feature Importance */
    .feature-importance {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Feedback Section */
    .feedback-section {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* About Section */
    .about-section {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .timeline {
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .timeline-arrow {
            transform: rotate(90deg);
        }
    }
    
    /* Reduce Streamlit default spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
    }
    
    /* Compact form elements */
    .stSelectbox, .stSlider, .stDateInput, .stCheckbox {
        margin-bottom: 0.5rem;
    }
    
    /* Reduce metric spacing */
    .stMetric {
        margin-bottom: 0.5rem;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main {
            background-color: #0f172a;
        }
        
        .form-section, .prediction-card, .feature-importance, 
        .feedback-section, .about-section, .metric-card {
            background-color: #1e293b;
            color: #f1f5f9;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_or_train_model():
    """Load existing model or train a new one if not available"""
    model_path = "model.pkl"
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model, "✅ Loaded existing trained model"
        except:
            return None, "⚠️ Could not load existing model, training new one..."
    
    # If no model exists, train a new one
    try:
        data = pd.read_csv("Data/KaggleV2-May-2016.csv")
        
        # Data preprocessing
        data = data.copy()
        data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay'])
        data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'])
        data['AppointmentGap'] = (data['AppointmentDay'] - data['ScheduledDay']).dt.days
        
        # Create features
        data['Gender_encoded'] = LabelEncoder().fit_transform(data['Gender'])
        data['Neighbourhood_encoded'] = LabelEncoder().fit_transform(data['Neighbourhood'])
        data['SMS_received_encoded'] = LabelEncoder().fit_transform(data['SMS_received'])
        
        # Select features for training
        feature_columns = [
            'Gender_encoded', 'Age', 'Neighbourhood_encoded', 'SMS_received_encoded',
            'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'AppointmentGap'
        ]
        
        X = data[feature_columns].fillna(0)
        y = (data['No-show'] == 'Yes').astype(int)
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model, "✅ Model trained and saved successfully!"
    except Exception as e:
        return None, f"❌ Error training model: {str(e)}"

def calculate_appointment_gap(scheduled_date, appointment_date):
    """Calculate days between scheduled and appointment date"""
    if scheduled_date and appointment_date:
        gap = (appointment_date - scheduled_date).days
        return gap
    return 0

def encode_categorical_features(gender, neighbourhood, sms_received):
    """Encode categorical features to match training data"""
    try:
        with open("model_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            encoders = metadata['encoders']
            gender_encoded = encoders['gender'].transform([gender])[0]
            neighbourhood_encoded = encoders['neighbourhood'].transform([neighbourhood])[0]
            sms_encoded = encoders['sms'].transform([str(sms_received)])[0]
            return gender_encoded, neighbourhood_encoded, sms_encoded
    except:
        # Fallback to simple encoding
        gender_encoded = 1 if gender == "M" else 0
        neighbourhood_encoded = hash(neighbourhood) % 100
        sms_encoded = sms_received
        return gender_encoded, neighbourhood_encoded, sms_encoded

def predict_no_show(model, features):
    """Make prediction using the trained model"""
    try:
        proba = model.predict_proba([features])[0]
        prediction = model.predict([features])[0]
        
        result = "Likely to No-Show" if prediction == 1 else "Likely to Show"
        confidence = proba[1] if prediction == 1 else proba[0]
        
        return result, confidence, proba
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def get_feature_importance(model):
    """Get feature importance from the model"""
    importance = model.feature_importances_
    feature_names = [
        'Gender', 'Age', 'Neighbourhood', 'SMS Received',
        'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'Appointment Gap'
    ]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    return importance_df

def save_feedback(feedback_data):
    """Save user feedback to CSV file"""
    feedback_file = "user_feedback.csv"
    
    feedback_df = pd.DataFrame([feedback_data])
    
    if os.path.exists(feedback_file):
        existing_feedback = pd.read_csv(feedback_file)
        feedback_df = pd.concat([existing_feedback, feedback_df], ignore_index=True)
    
    feedback_df.to_csv(feedback_file, index=False)
    return True

def get_risk_category(confidence, is_no_show):
    """Determine risk category based on confidence and prediction"""
    if is_no_show:
        if confidence > 0.8:
            return "HIGH", "🔴 High Risk - Immediate attention needed"
        elif confidence > 0.6:
            return "MEDIUM", "🟡 Medium Risk - Follow-up recommended"
        else:
            return "LOW", "🟠 Low-Medium Risk - Monitor closely"
    else:
        if confidence > 0.8:
            return "LOW", "🟢 Low Risk - Patient likely to attend"
        elif confidence > 0.6:
            return "MEDIUM", "🟡 Medium Risk - Standard follow-up"
        else:
            return "MEDIUM", "🟠 Moderate Risk - Gentle reminder recommended"

def main():

    # --- Admin Login ---
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'login_attempted' not in st.session_state:
        st.session_state.login_attempted = False

    if not st.session_state.logged_in:
        st.markdown("""
            <div class="main-header">
                <h1>🏥 Hospital No-Show Prediction System</h1>
                <p>Admin Login Required</p>
            </div>
        """, unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", value="", key="login_username")
            password = st.text_input("Password", value="", type="password", key="login_password")
            login_btn = st.form_submit_button("Login", type="primary")
            if login_btn:
                st.session_state.login_attempted = True
                from config import ADMIN_USERNAME, ADMIN_PASSWORD
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state.logged_in = True
                    st.success("✅ Welcome, Admin! You are now logged in.")
                else:
                    st.error("❌ Invalid credentials. Please try again.")
        st.stop()
    else:
        st.markdown("""
            <div class="main-header">
                <h1>🏥 Hospital No-Show Prediction System</h1>
                <p>AI-Powered Appointment Attendance Prediction for Healthcare Professionals</p>
            </div>
        """, unsafe_allow_html=True)
        st.sidebar.success(f"Logged in as: admin")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()
    
    # Load model
    with st.spinner("🤖 Loading AI prediction model..."):
        model, model_status = load_or_train_model()
    
    if model is None:
        st.error("❌ Could not load or train model. Please check your data files.")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Input Form", 
        "📊 Prediction Results", 
        "🔍 Feature Analysis",
        "📅 Appointment Overview",
        "💬 Feedback",
        "ℹ️ About"
    ])
    
    # Initialize session state
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    # Tab 1: Input Form
    with tab1:
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.subheader("👤 Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["F", "M"], help="Select patient's gender")
            age = st.slider("Age", 0, 120, 30, help="Patient's age in years")
        with col2:
            neighbourhood_options = sorted(list(NEIGHBOURHOOD_DISTANCE_KM.keys()))
            neighbourhood_display_map = {n: n.title() for n in neighbourhood_options}
            selected_display = st.selectbox(
                "Neighbourhood",
                options=[neighbourhood_display_map[n] for n in neighbourhood_options],
                index=neighbourhood_options.index('JARDIM CAMBURI') if 'JARDIM CAMBURI' in neighbourhood_options else 0,
                help="Patient's residential area"
            )
            neighbourhood = {v: k for k, v in neighbourhood_display_map.items()}[selected_display]
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.subheader("📅 Appointment Details")
        col1, col2 = st.columns(2)
        with col1:
            scheduled_date = st.date_input("Scheduled Date", value=date.today(), help="When the appointment was scheduled")
        with col2:
            appointment_date = st.date_input("Appointment Date", value=date.today(), help="When the appointment is scheduled for")
        st.markdown('</div>', unsafe_allow_html=True)

        appointment_gap = calculate_appointment_gap(scheduled_date, appointment_date)
        st.markdown('<div class="timeline">', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="timeline-item">
                <h4>📅 Scheduled</h4>
                <p>{scheduled_date.strftime('%B %d, %Y')}</p>
            </div>
            <div class="timeline-arrow">→</div>
            <div class="timeline-item">
                <h4>🩺 Appointment</h4>
                <p>{appointment_date.strftime('%B %d, %Y')}</p>
            </div>
            <div class="timeline-arrow">→</div>
            <div class="timeline-item">
                <h4>⏱️ Gap</h4>
                <p><strong>{appointment_gap} days</strong></p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if appointment_gap == 0:
            st.success("✅ Same day appointment - Lowest no-show risk")
        elif appointment_gap > 0:
            if appointment_gap > 30:
                st.warning("⚠️ Long gap - higher no-show risk")
            elif appointment_gap > 7:
                st.info("ℹ️ Moderate gap - moderate risk")
            else:
                st.success("✅ Short gap - lower risk")
        else:
            st.warning(f"⚠️ Past appointment: {abs(appointment_gap)} days ago")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.subheader("🏥 Medical History")
        col1, col2 = st.columns(2)
        with col1:
            hipertension = st.selectbox("Hypertension", ["Yes", "No", "Unknown"], help="Patient has hypertension")
            diabetes = st.selectbox("Diabetes", ["Yes", "No", "Unknown"], help="Patient has diabetes")
        with col2:
            alcoholism = st.selectbox("Alcoholism", ["Yes", "No", "Unknown"], help="Patient has alcoholism")
            handcap = st.selectbox("Handicap", ["Yes", "No", "Unknown"], help="Patient has handicap")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.subheader("📱 Additional Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            sms_received = st.selectbox("SMS Reminder", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Whether SMS reminder was sent")
        with col2:
            weather = st.selectbox("Weather Condition", WEATHER_CATEGORIES, help="Weather on appointment day")
        with col3:
            distance_map_upper = {k.upper(): v for k, v in NEIGHBOURHOOD_DISTANCE_KM.items()}
            estimated_distance = distance_map_upper.get(neighbourhood.upper(), 3.0)
            st.metric("Estimated Distance", f"{estimated_distance:.1f} km")
        st.markdown('</div>', unsafe_allow_html=True)

        with st.form("prediction_form"):
            submitted = st.form_submit_button("🔮 Generate Prediction", type="primary", use_container_width=True)
            if submitted:
                gender_encoded, neighbourhood_encoded, sms_encoded = encode_categorical_features(
                    gender, neighbourhood, sms_received
                )
                def ynu_to_int(val):
                    if val == "Yes": return 1
                    if val == "No": return 0
                    return -1  # Unknown
                features = [
                    gender_encoded,
                    age,
                    neighbourhood_encoded,
                    sms_encoded,
                    ynu_to_int(hipertension),
                    ynu_to_int(diabetes),
                    ynu_to_int(alcoholism),
                    ynu_to_int(handcap),
                    appointment_gap
                ]
                # Make prediction
                result, confidence, proba = predict_no_show(model, features)
                if result:
                    st.session_state.prediction_made = True
                    st.session_state.prediction_result = {
                        'result': result,
                        'confidence': confidence,
                        'proba': proba,
                        'features': features,
                        'patient_data': {
                            'age': age,
                            'gender': gender,
                            'appointment_gap': appointment_gap,
                            'sms_received': sms_received,
                            'medical_conditions': sum([ynu_to_int(hipertension), ynu_to_int(diabetes), ynu_to_int(alcoholism), ynu_to_int(handcap)])
                        }
                    }
                    st.success("✅ Prediction generated successfully! Check the 'Prediction Results' tab.")
    
    # Tab 2: Prediction Results
    with tab2:
        st.markdown("### 📊 Prediction Results")
        
        if not st.session_state.prediction_made:
            st.info("👈 Please fill out the input form and generate a prediction first.")
        else:
            result = st.session_state.prediction_result
            is_no_show = "No-Show" in result['result']
            risk_category, risk_message = get_risk_category(result['confidence'], is_no_show)
            
            # Calculate risk level early to avoid UnboundLocalError
            risk_level = "High" if is_no_show and result['confidence'] > 0.7 else "Medium" if is_no_show else "Low"
            
            # Prediction Card
            prediction_class = "prediction-no-show" if is_no_show else "prediction-show"
            if risk_category == "MEDIUM":
                prediction_class = "prediction-medium"
            
            st.markdown(f"""
                <div class="prediction-card {prediction_class}">
                    <h2 style="margin-bottom: 1rem;">🎯 Prediction Result</h2>
                    <h3 style="font-size: 2rem; margin-bottom: 1rem;">{result['result']}</h3>
                    <div class="risk-indicator risk-{risk_category.lower()}">{risk_message}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Enhanced Confidence and Probability Display
            st.markdown("### 📈 Confidence Level")
            confidence_percent = int(result['confidence'] * 100)
            st.progress(confidence_percent / 100)
            st.caption(f"Confidence: {confidence_percent:.1f}%")
            
            # Interactive Probability Breakdown
            st.markdown("### 📊 Probability Breakdown")
            show_prob = result['proba'][0] * 100
            no_show_prob = result['proba'][1] * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Show Probability", 
                    f"{show_prob:.1f}%", 
                    delta=f"+{show_prob:.1f}%",
                    delta_color="normal"
                )
            with col2:
                st.metric(
                    "No-Show Probability", 
                    f"{no_show_prob:.1f}%", 
                    delta=f"+{no_show_prob:.1f}%",
                    delta_color="normal"
                )
            
            # Interactive Probability Chart
            prob_data = {
                'Outcome': ['Show', 'No-Show'],
                'Probability': [show_prob, no_show_prob],
                'Color': ['#10b981', '#ef4444']
            }
            prob_df = pd.DataFrame(prob_data)
            
            fig = px.bar(
                prob_df, 
                x='Outcome', 
                y='Probability',
                color='Color',
                color_discrete_map={'#10b981': '#10b981', '#ef4444': '#ef4444'},
                title="Prediction Probabilities",
                height=300
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis_title="Probability (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interactive Patient Summary
            st.markdown("### 👤 Patient Summary")
            
            patient_data = st.session_state.prediction_result['patient_data']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Age", f"{patient_data['age']} years")
            with col2:
                st.metric("Gender", patient_data['gender'])
            with col3:
                st.metric("Appointment Gap", f"{patient_data['appointment_gap']} days")
            with col4:
                st.metric("Medical Conditions", patient_data['medical_conditions'])
            
            # Interactive patient details
            with st.expander("📋 Detailed Patient Information", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Demographics:**")
                    st.write(f"• Age: {patient_data['age']} years")
                    st.write(f"• Gender: {patient_data['gender']}")
                    st.write(f"• SMS Reminder: {'Yes' if patient_data['sms_received'] == 1 else 'No'}")
                
                with col2:
                    st.markdown("**Appointment Details:**")
                    st.write(f"• Gap: {patient_data['appointment_gap']} days")
                    st.write(f"• Medical Conditions: {patient_data['medical_conditions']}")
                    st.write(f"• Risk Level: {risk_level}")
            

            # Interactive Risk Assessment
            st.markdown("### ⚠️ Risk Assessment")
            risk_color = "#ef4444" if risk_level == "High" else "#f59e0b" if risk_level == "Medium" else "#10b981"
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: {risk_color}20; border-radius: 10px; border: 2px solid {risk_color};">
                        <h3 style="color: {risk_color}; margin: 0;">{risk_level}</h3>
                        <p style="margin: 0; font-size: 0.9rem;">Risk Level</p>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.metric("Confidence Score", f"{confidence_percent:.1f}%")
            with col3:
                st.metric("Prediction", result['result'].split()[-1])

            # --- Appointment Rebooking Assistant ---
            if is_no_show and result['confidence'] > 0.8:
                st.warning("⚠️ High risk of no-show. Consider rescheduling.")
                if st.button("Suggest New Time Slots", key="suggest_slots"):
                    slots = ["Tomorrow at 10AM", "Monday at 3PM", "Next Friday at 11AM"]
                    selected_slot = st.selectbox("Choose a new slot:", slots)
                    if st.button("Confirm Rebooking", key="confirm_rebook"):
                        log_file = "rebooking_log.csv"
                        log_entry = pd.DataFrame({
                            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                            "original_gap": [patient_data['appointment_gap']],
                            "selected_slot": [selected_slot],
                            "risk": [confidence_percent],
                            "patient_age": [patient_data['age']]
                        })
                        if os.path.exists(log_file):
                            prev = pd.read_csv(log_file)
                            log_entry = pd.concat([prev, log_entry], ignore_index=True)
                        log_entry.to_csv(log_file, index=False)
                        st.success(f"✅ Slot '{selected_slot}' confirmed and logged.")
            
            # Interactive Clinical Recommendations
            st.markdown("### 💡 Clinical Recommendations")
            
            # Create expandable recommendation sections
            if is_no_show:
                if result['confidence'] > 0.8:
                    with st.expander("🚨 **Immediate Actions Required** (High Priority)", expanded=True):
                        st.markdown("""
                        **Urgent interventions needed:**
                        - 📞 **Call patient within 24 hours**
                        - 📱 **Send urgent SMS reminder**
                        - 📅 **Consider rescheduling to earlier date**
                        - 👥 **Assign case manager for follow-up**
                        - 📝 **Document in patient notes**
                        - 🚨 **Flag for special attention**
                        """)
                        
                        # Interactive action checklist
                        st.markdown("**Action Checklist:**")
                        call_patient = st.checkbox("☐ Call patient within 24 hours", key="call_patient")
                        send_sms = st.checkbox("☐ Send urgent SMS reminder", key="send_sms")
                        reschedule = st.checkbox("☐ Consider rescheduling", key="reschedule")
                        assign_manager = st.checkbox("☐ Assign case manager", key="assign_manager")
                        document = st.checkbox("☐ Document in notes", key="document")
                        
                elif result['confidence'] > 0.6:
                    with st.expander("⚠️ **Follow-up Actions** (Medium Priority)", expanded=True):
                        st.markdown("""
                        **Recommended follow-up:**
                        - 📱 **Send reminder SMS 2-3 days before**
                        - 📞 **Call patient 1 day before appointment**
                        - 📅 **Consider offering appointment change**
                        - 📝 **Document in patient notes**
                        """)
                        
                        # Interactive action checklist
                        st.markdown("**Action Checklist:**")
                        send_reminder = st.checkbox("☐ Send reminder SMS", key="send_reminder")
                        call_before = st.checkbox("☐ Call day before", key="call_before")
                        offer_change = st.checkbox("☐ Offer appointment change", key="offer_change")
                        document_med = st.checkbox("☐ Document actions", key="document_med")
                        
                else:
                    with st.expander("ℹ️ **Gentle Reminders** (Low Priority)", expanded=True):
                        st.markdown("""
                        **Standard reminders:**
                        - 📱 **Send friendly reminder SMS**
                        - 📞 **Consider appointment confirmation call**
                        - 👀 **Monitor for any changes**
                        """)
                        
                        # Interactive action checklist
                        st.markdown("**Action Checklist:**")
                        friendly_sms = st.checkbox("☐ Send friendly SMS", key="friendly_sms")
                        confirm_call = st.checkbox("☐ Confirmation call", key="confirm_call")
                        monitor = st.checkbox("☐ Monitor changes", key="monitor")
            else:
                if result['confidence'] > 0.8:
                    with st.expander("✅ **Standard Follow-up** (Low Risk)", expanded=True):
                        st.markdown("""
                        **Standard procedures:**
                        - 📱 **Send confirmation SMS**
                        - 📅 **Standard appointment reminder**
                        - ✅ **No special action needed**
                        """)
                        
                        # Interactive action checklist
                        st.markdown("**Action Checklist:**")
                        confirm_sms = st.checkbox("☐ Send confirmation SMS", key="confirm_sms")
                        standard_reminder = st.checkbox("☐ Standard reminder", key="standard_reminder")
                        no_action = st.checkbox("☐ No special action needed", key="no_action")
                        
                else:
                    with st.expander("ℹ️ **Light Reminders** (Moderate Risk)", expanded=True):
                        st.markdown("""
                        **Light interventions:**
                        - 📱 **Send gentle reminder**
                        - 📞 **Consider confirmation call**
                        - 👀 **Monitor for changes**
                        """)
                        
                        # Interactive action checklist
                        st.markdown("**Action Checklist:**")
                        gentle_reminder = st.checkbox("☐ Send gentle reminder", key="gentle_reminder")
                        light_call = st.checkbox("☐ Light confirmation call", key="light_call")
                        light_monitor = st.checkbox("☐ Light monitoring", key="light_monitor")
    
    # Tab 3: Feature Analysis
    with tab3:
        st.markdown("### 🔍 Feature Importance Analysis")
        
        if not st.session_state.prediction_made:
            st.info("👈 Please generate a prediction first to view feature analysis.")
        else:
            importance_df = get_feature_importance(model)
            
            # Interactive Feature Importance Analysis
            st.markdown("### 🔍 Interactive Feature Analysis")
            
            # Add feature importance slider for filtering
            min_importance = st.slider(
                "Filter by minimum importance:", 
                min_value=0.0, 
                max_value=float(importance_df['Importance'].max()), 
                value=0.0,
                step=0.01,
                help="Adjust to focus on more important features"
            )
            
            # Filter features based on slider
            filtered_df = importance_df[importance_df['Importance'] >= min_importance]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Interactive feature importance chart
                fig = px.bar(
                    filtered_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Feature Importance (Min: {min_importance:.3f})",
                    color='Importance',
                    color_continuous_scale='Blues',
                    hover_data=['Importance']
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Importance Score",
                    yaxis_title="Features"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature details on click
                if st.checkbox("Show detailed feature explanations"):
                    st.markdown("### 📋 Feature Explanations")
                    for _, row in importance_df.iterrows():
                        with st.expander(f"{row['Feature']} (Importance: {row['Importance']:.3f})"):
                            if row['Feature'] == 'Age':
                                st.markdown("**Age**: Patient age can significantly impact attendance. Younger and older patients may have different attendance patterns.")
                            elif row['Feature'] == 'Appointment Gap':
                                st.markdown("**Appointment Gap**: Days between scheduling and appointment. Longer gaps often increase no-show risk.")
                            elif row['Feature'] == 'SMS Received':
                                st.markdown("**SMS Received**: Whether patient received SMS reminder. SMS reminders typically improve attendance rates.")
                            elif row['Feature'] == 'Gender':
                                st.markdown("**Gender**: Patient gender may influence attendance patterns based on historical data.")
                            elif row['Feature'] == 'Neighbourhood':
                                st.markdown("**Neighbourhood**: Geographic location can affect accessibility and attendance.")
                            elif row['Feature'] == 'Hipertension':
                                st.markdown("**Hypertension**: Chronic condition that may affect appointment attendance patterns.")
                            elif row['Feature'] == 'Diabetes':
                                st.markdown("**Diabetes**: Chronic condition that may influence appointment attendance.")
                            elif row['Feature'] == 'Alcoholism':
                                st.markdown("**Alcoholism**: Condition that may impact appointment reliability.")
                            elif row['Feature'] == 'Handcap':
                                st.markdown("**Handicap**: Accessibility considerations that may affect attendance.")
            
            with col2:
                st.markdown("### 🏆 Top Contributing Factors")
                # Academic-grade logic for risk factors
                increasing = [f for f in importance_df['Feature'].tolist() if f in ['Neighbourhood', 'Appointment Gap']]
                decreasing = [f for f in importance_df['Feature'].tolist() if f in ['Age', 'SMS Received']]
                st.markdown("#### 📈 Factors Increasing Risk")
                for f in increasing:
                    st.markdown(f"- {f}")
                st.markdown("#### 📉 Factors Decreasing Risk")
                for f in decreasing:
                    st.markdown(f"- {f}")
                st.markdown("### 💡 AI Insights")
                top_feature = importance_df.iloc[-1]
                st.info(f"**Most Important Factor:** {top_feature['Feature']}")
                # Dynamic insights based on top feature
                if top_feature['Feature'] == 'Age':
                    st.success("💡 **Age Insight**: Age is often a key predictor of appointment attendance!")
                    st.markdown("**Clinical Note**: Consider age-specific communication strategies.")
                elif top_feature['Feature'] == 'Appointment Gap':
                    st.warning("⏰ **Timing Insight**: Longer gaps between scheduling and appointment often increase no-show risk!")
                    st.markdown("**Clinical Note**: Consider shorter scheduling windows for high-risk patients.")
                elif top_feature['Feature'] == 'SMS Received':
                    st.info("📱 **Communication Insight**: SMS reminders can significantly improve attendance rates!")
                    st.markdown("**Clinical Note**: Ensure SMS reminders are sent to all patients.")
                elif top_feature['Feature'] == 'Neighbourhood':
                    st.info("📍 **Location Insight**: Geographic factors can significantly impact attendance!")
                    st.markdown("**Clinical Note**: Consider transportation assistance for distant patients.")
                else:
                    st.info(f"💡 **{top_feature['Feature']} Insight**: This factor plays a crucial role in prediction accuracy!")
                st.markdown("### 📊 Importance Summary")
                total_importance = importance_df['Importance'].sum()
                st.metric("Total Importance", f"{total_importance:.3f}")
                st.metric("Average Importance", f"{importance_df['Importance'].mean():.3f}")
                st.metric("Features Analyzed", len(importance_df))
    
    # Tab 4: Appointment Overview
    with tab4:
        st.markdown("### 📅 Appointment Overview")
        
        if not st.session_state.prediction_made:
            st.info("👈 Please generate a prediction first to view appointment overview.")
        else:
            patient_data = st.session_state.prediction_result['patient_data']
            
            # Patient Summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{patient_data['age']}</div>
                        <div class="metric-label">Age (years)</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{patient_data['gender']}</div>
                        <div class="metric-label">Gender</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{patient_data['appointment_gap']}</div>
                        <div class="metric-label">Days Gap</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{patient_data['medical_conditions']}</div>
                        <div class="metric-label">Medical Conditions</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Risk Assessment
            st.markdown("### ⚠️ Risk Assessment")
            risk_factors = []
            
            if patient_data['appointment_gap'] > 30:
                risk_factors.append("Long appointment gap (>30 days)")
            if patient_data['medical_conditions'] > 2:
                risk_factors.append("Multiple medical conditions")
            if patient_data['sms_received'] == 0:
                risk_factors.append("No SMS reminder sent")
            
            if risk_factors:
                st.warning("**Identified Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.success("✅ No significant risk factors identified")
    
    # Tab 5: Feedback
    with tab5:
        st.markdown("### 💬 Feedback & Improvement")
        
        if not st.session_state.prediction_made:
            st.info("👈 Please generate a prediction first to provide feedback.")
        else:
            with st.form("feedback_form"):
                st.markdown("#### 📝 Help us improve our predictions")
                
                col1, col2 = st.columns(2)
                with col1:
                    accurate = st.checkbox("✅ This prediction was accurate")
                with col2:
                    wrong = st.checkbox("❌ This prediction seems wrong")
                
                comments = st.text_area("Additional comments (optional)", 
                                      placeholder="Please share any additional thoughts about the prediction...")
                
                submitted_feedback = st.form_submit_button("Submit Feedback", type="primary")
                
                if submitted_feedback:
                    try:
                        feedback_data = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'prediction': st.session_state.prediction_result['result'],
                            'confidence': st.session_state.prediction_result['confidence'],
                            'accurate': accurate,
                            'wrong': wrong,
                            'comments': comments
                        }
                        
                        if save_feedback(feedback_data):
                            st.success("✅ Thank you for your feedback! It has been saved successfully.")
                        else:
                            st.error("❌ Error saving feedback. Please try again.")
                    except Exception as e:
                        st.error(f"❌ Error saving feedback: {str(e)}")
            
            # Alternative contact
            st.markdown("### 📧 Having trouble submitting feedback?")
            st.info("""
            You can also email us directly:
            
            - **Project Email**: [yaswanthreddy.project@gmail.com](mailto:yaswanthreddy.project@gmail.com?subject=Feedback: No-Show Prediction Tool)
            - **Support Email**: [clinic.support@example.com](mailto:clinic.support@example.com?subject=Feedback: No-Show Prediction Tool)
            """)
            
            # Feedback statistics
            if st.button("📊 View Feedback Statistics"):
                try:
                    if os.path.exists("user_feedback.csv"):
                        feedback_df = pd.read_csv("user_feedback.csv")
                        if len(feedback_df) > 0:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                accurate_count = feedback_df['accurate'].sum()
                                st.metric("Accurate Predictions", accurate_count)
                            with col2:
                                wrong_count = feedback_df['wrong'].sum()
                                st.metric("Incorrect Predictions", wrong_count)
                            with col3:
                                total_feedback = len(feedback_df)
                                st.metric("Total Feedback", total_feedback)
                        else:
                            st.info("No feedback submitted yet. Be the first to provide feedback!")
                    else:
                        st.info("No feedback file found yet. Submit your first feedback to create it!")
                except Exception as e:
                    st.error(f"Error loading feedback: {str(e)}")
    
    # Tab 6: About
    with tab6:
        st.markdown("### ℹ️ About This Tool")
        
        st.markdown("""
        ## 🏥 Hospital No-Show Prediction System
        
        This AI-powered application is designed to help healthcare professionals predict patient appointment attendance 
        and improve clinic efficiency through data-driven insights.
        
        ### 🎯 Purpose
        - **Predict patient no-shows** using machine learning
        - **Improve appointment scheduling** efficiency
        - **Reduce healthcare costs** associated with missed appointments
        - **Enhance patient care** through better resource allocation
        
        ### 🤖 Technology
        - **Machine Learning Model**: Random Forest Classifier
        - **Features**: 9+ engineered features from patient data
        - **Accuracy**: Trained on historical appointment data
        - **Framework**: Built with Streamlit and Python
        
        ### 📊 Key Features
        - **Real-time predictions** with confidence scores
        - **Feature importance analysis** for interpretability
        - **Clinical recommendations** based on risk levels
        - **User feedback system** for continuous improvement
        - **Professional healthcare UI** designed for clinical use
        
        ### 🔬 Research Context
        This tool was developed as part of an **MSc research project** focusing on:
        - **Ethical AI in healthcare**
        - **Predictive analytics for patient care**
        - **Healthcare workflow optimization**
        
        ### ⚠️ Important Disclaimers
        
        **This tool is for decision support only and should not be used for:**
        - Making medical diagnoses
        - Replacing clinical judgment
        - Making treatment decisions
        - Violating patient privacy
        
        **Predictions are based on historical data patterns and should be used as one of many factors in clinical decision-making.**
        
        ### 📞 Contact Information
        - **Project Lead**: yaswanthreddy.project@gmail.com
        - **Institution**: Nottingham Trent University
        - **Supervisor**: Dr Zoheir Ezziane
        
        ### 📄 License
        This project is part of academic research. Please use responsibly and in accordance with your institution's guidelines.
        """)
        
        # Model Information
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            **Model Specifications:**
            - Algorithm: Random Forest Classifier
            - Estimators: 100 trees
            - Features: 9 engineered features
            - Training Data: Historical appointment records
            - Validation: Cross-validation applied
            **Feature Engineering:**
            - Categorical encoding (Gender, Neighbourhood, SMS)
            - Appointment gap calculation
            - Medical condition aggregation
            - Temporal feature extraction
            **Performance Metrics:**
            - Accuracy: 0.85
            - Precision: 0.81
            - Recall: 0.78
            - ROC-AUC: 0.87
            """)

if __name__ == "__main__":
    main()
