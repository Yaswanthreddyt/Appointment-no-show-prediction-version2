#!/usr/bin/env python3
"""
Advanced Model Training Script with SMOTE Analysis
for AI-Based No-Show Appointment Prediction Tool
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')
from config import NEIGHBOURHOOD_DISTANCE_KM, WEATHER_CATEGORIES

def load_and_preprocess_data():
    """Load and preprocess the appointment data"""
    print("📊 Loading data...")
    
    # Load data
    data_path = "Data/KaggleV2-May-2016.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"✅ Loaded {len(data):,} records")
    
    # Basic data info
    print(f"📋 Data shape: {data.shape}")
    print(f"📊 Columns: {list(data.columns)}")
    
    # Check target distribution
    print(f"\n🎯 Target distribution:")
    print(data['No-show'].value_counts())
    print(f"No-show rate: {(data['No-show'] == 'Yes').mean():.2%}")
    
    return data

def engineer_features(data):
    """Engineer features for the model"""
    print("\n🔧 Engineering features...")
    
    # Create a copy to avoid modifying original
    df = data.copy()
    
    # Convert dates
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    
    # Calculate appointment gap
    df['AppointmentGap'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    
    # Create day of week features
    df['ScheduledDayOfWeek'] = df['ScheduledDay'].dt.dayofweek
    df['AppointmentDayOfWeek'] = df['AppointmentDay'].dt.dayofweek
    
    # Create month features
    df['ScheduledMonth'] = df['ScheduledDay'].dt.month
    df['AppointmentMonth'] = df['AppointmentDay'].dt.month
    
    # Create time-based features
    df['ScheduledHour'] = df['ScheduledDay'].dt.hour
    df['IsWeekend'] = df['AppointmentDay'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_neighbourhood = LabelEncoder()
    le_sms = LabelEncoder()
    
    df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
    df['Neighbourhood_encoded'] = le_neighbourhood.fit_transform(df['Neighbourhood'])
    df['SMS_received_encoded'] = le_sms.fit_transform(df['SMS_received'])

    # Estimated distance to clinic (km) from neighbourhood mapping
    # Use uppercase keys for robust matching and fall back to median distance
    print("🚗 Adding EstimatedDistance feature from neighbourhood mapping...")
    distance_map_upper = {k.upper(): v for k, v in NEIGHBOURHOOD_DISTANCE_KM.items()}
    default_distance = float(np.median(list(distance_map_upper.values()))) if len(distance_map_upper) else 3.0
    df['EstimatedDistance'] = df['Neighbourhood'].astype(str).str.upper().map(distance_map_upper).fillna(default_distance)

    # Simulated weather conditions for appointment day
    print("⛅ Adding Weather feature and one-hot encoding...")
    rng = np.random.default_rng(seed=42)
    # Assign random weather if not available
    df['Weather'] = rng.choice(WEATHER_CATEGORIES, size=len(df))
    weather_dummies = pd.get_dummies(df['Weather'], prefix='Weather', dtype=int)
    # Ensure all expected weather categories exist as columns
    expected_weather_cols = [f"Weather_{w}" for w in WEATHER_CATEGORIES]
    weather_dummies = weather_dummies.reindex(columns=expected_weather_cols, fill_value=0)
    
    # Create interaction features
    # Handle NaN values in Age before creating age groups
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Create age groups safely
    try:
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 120], labels=[0, 1, 2, 3, 4])
        df['Age_Group'] = df['Age_Group'].cat.codes  # Convert to numeric codes
    except:
        # Fallback: simple age grouping
        df['Age_Group'] = 0  # Default group
        df.loc[df['Age'] <= 18, 'Age_Group'] = 0
        df.loc[(df['Age'] > 18) & (df['Age'] <= 35), 'Age_Group'] = 1
        df.loc[(df['Age'] > 35) & (df['Age'] <= 50), 'Age_Group'] = 2
        df.loc[(df['Age'] > 50) & (df['Age'] <= 65), 'Age_Group'] = 3
        df.loc[df['Age'] > 65, 'Age_Group'] = 4
    
    # Medical conditions count
    df['Medical_Conditions_Count'] = df[['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap']].sum(axis=1)
    
    # Handle all NaN values before creating feature matrix
    df = df.fillna(0)
    
    # Create feature matrix
    feature_columns = [
        'Gender_encoded', 'Age', 'Neighbourhood_encoded', 'SMS_received_encoded',
        'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'AppointmentGap',
        'ScheduledDayOfWeek', 'AppointmentDayOfWeek', 'ScheduledMonth', 'AppointmentMonth',
        'ScheduledHour', 'IsWeekend', 'Age_Group', 'Medical_Conditions_Count',
        'EstimatedDistance'
    ] + expected_weather_cols
    
    # Concatenate base features with weather one-hot columns
    X = pd.concat([df[feature_columns[:18]], weather_dummies], axis=1)
    y = (df['No-show'] == 'Yes').astype(int)  # 1 for no-show, 0 for show
    
    print(f"✅ Created {len(feature_columns)} features")
    print(f"📊 Feature names: {feature_columns}")
    
    # Update feature_columns to reflect full X order
    full_feature_columns = list(X.columns)
    return X, y, full_feature_columns, le_gender, le_neighbourhood, le_sms

def train_model_with_smote(X, y):
    """Train Random Forest model with SMOTE for handling imbalanced data"""
    print("\n🤖 Training Random Forest model with SMOTE...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {X_train.shape[0]:,} samples")
    print(f"📊 Test set: {X_test.shape[0]:,} samples")
    print(f"🎯 Training set no-show rate: {y_train.mean():.2%}")
    print(f"🎯 Test set no-show rate: {y_test.mean():.2%}")
    
    # Apply SMOTE to training data only
    print("🔄 Applying SMOTE to balance training data...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"📊 Balanced training set: {X_train_balanced.shape[0]:,} samples")
    print(f"🎯 Balanced training set no-show rate: {y_train_balanced.mean():.2%}")
    
    # Create Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Train model
    print("🚀 Training Random Forest...")
    rf_model.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    print("\n📊 Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Show', 'No-Show']))
    
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Cross-validation
    print("\n🔄 Cross-validation scores:")
    cv_scores = cross_val_score(rf_model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return rf_model, X_test, y_test, y_pred, y_pred_proba

def save_model_and_metadata(model, feature_columns, encoders, performance_metrics):
    """Save the trained model and metadata"""
    print("\n💾 Saving model and metadata...")
    
    # Save the model
    model_path = "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        'feature_columns': feature_columns,
        'encoders': {
            'gender': encoders['gender'],
            'neighbourhood': encoders['neighbourhood'],
            'sms': encoders['sms']
        },
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': 'RandomForestClassifier',
        'n_features': len(feature_columns),
        'performance_metrics': performance_metrics,
        'weather_categories': WEATHER_CATEGORIES
    }
    
    metadata_path = "model_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Metadata saved to: {metadata_path}")
    
    return model_path, metadata_path

def main():
    """Main training function"""
    print("🏥 AI-Based No-Show Appointment Prediction Tool")
    print("🤖 Advanced Model Training with SMOTE Analysis")
    print("=" * 60)
    
    try:
        # Load and preprocess data
        data = load_and_preprocess_data()
        
        # Engineer features
        X, y, feature_columns, le_gender, le_neighbourhood, le_sms = engineer_features(data)
        
        # Train model with SMOTE
        model, X_test, y_test, y_pred, y_pred_proba = train_model_with_smote(X, y)
        
        # Performance metrics
        performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'test_samples': len(X_test),
            'training_samples': len(X) - len(X_test)
        }
        
        # Save model and metadata
        encoders = {
            'gender': le_gender,
            'neighbourhood': le_neighbourhood,
            'sms': le_sms
        }
        
        model_path, metadata_path = save_model_and_metadata(
            model, feature_columns, encoders, performance_metrics
        )
        
        print("\n🎉 Model training completed successfully!")
        print(f"📊 Final model performance:")
        print(f"   - Accuracy: {performance_metrics['accuracy']:.4f}")
        print(f"   - ROC AUC: {performance_metrics['roc_auc']:.4f}")
        print(f"   - Features: {len(feature_columns)}")
        print(f"   - Training samples: {performance_metrics['training_samples']:,}")
        print(f"   - Test samples: {performance_metrics['test_samples']:,}")
        
        # Feature importance
        print(f"\n🔍 Top 5 Most Important Features:")
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head().iterrows()):
            print(f"   {i+1}. {row['Feature']}: {row['Importance']:.4f}")
        
        print(f"\n✅ Your model is ready to use!")
        print(f"🚀 Run the Streamlit app to test predictions")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
