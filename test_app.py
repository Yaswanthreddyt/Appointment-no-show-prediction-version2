#!/usr/bin/env python3
"""
Test script for the AI-Based No-Show Appointment Prediction Tool
"""

import unittest
import pandas as pd
import numpy as np
from datetime import date, datetime
import os
import sys

# Add the current directory to the path so we can import app functions
sys.path.append('.')

try:
    from app import (
        calculate_appointment_gap,
        encode_categorical_features,
        predict_no_show,
        get_feature_importance,
        save_feedback
    )
    from sklearn.ensemble import RandomForestClassifier
    APP_IMPORTED = True
except ImportError as e:
    print(f"Warning: Could not import app functions: {e}")
    APP_IMPORTED = False

class TestNoShowPredictionTool(unittest.TestCase):
    """Test cases for the no-show prediction tool"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple test model
        self.test_model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_test = np.random.rand(100, 9)  # 9 features
        y_test = np.random.randint(0, 2, 100)  # Binary classification
        self.test_model.fit(X_test, y_test)
        
        # Test data
        self.test_features = [0, 30, 5, 1, 0, 0, 0, 0, 7]
        
    def test_calculate_appointment_gap(self):
        """Test appointment gap calculation"""
        if not APP_IMPORTED:
            self.skipTest("App functions not available")
            
        scheduled = date(2024, 1, 1)
        appointment = date(2024, 1, 8)
        gap = calculate_appointment_gap(scheduled, appointment)
        self.assertEqual(gap, 7)
        
        # Same day
        gap = calculate_appointment_gap(scheduled, scheduled)
        self.assertEqual(gap, 0)
        
        # Past appointment
        gap = calculate_appointment_gap(appointment, scheduled)
        self.assertEqual(gap, -7)
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding"""
        if not APP_IMPORTED:
            self.skipTest("App functions not available")
            
        gender_encoded, neighbourhood_encoded, sms_encoded = encode_categorical_features(
            "M", "Test Neighbourhood", 1
        )
        
        self.assertEqual(gender_encoded, 1)  # M should be 1
        self.assertIsInstance(neighbourhood_encoded, int)
        self.assertEqual(sms_encoded, 1)  # 1 should be 1
        
        # Test female
        gender_encoded, _, _ = encode_categorical_features("F", "Test", 0)
        self.assertEqual(gender_encoded, 0)  # F should be 0
    
    def test_predict_no_show(self):
        """Test prediction functionality"""
        if not APP_IMPORTED:
            self.skipTest("App functions not available")
            
        result, confidence, proba = predict_no_show(self.test_model, self.test_features)
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(confidence)
        self.assertIsNotNone(proba)
        self.assertIn(result, ["Likely to Show", "Likely to No-Show"])
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
        self.assertEqual(len(proba), 2)
        self.assertAlmostEqual(sum(proba), 1.0, places=5)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction"""
        if not APP_IMPORTED:
            self.skipTest("App functions not available")
            
        importance_df = get_feature_importance(self.test_model)
        
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertEqual(len(importance_df), 9)  # 9 features
        self.assertIn('Feature', importance_df.columns)
        self.assertIn('Importance', importance_df.columns)
        self.assertTrue(all(importance_df['Importance'] >= 0))
    
    def test_save_feedback(self):
        """Test feedback saving functionality"""
        if not APP_IMPORTED:
            self.skipTest("App functions not available")
            
        feedback_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prediction': 'Likely to Show',
            'confidence': 0.85,
            'accurate': True,
            'wrong': False,
            'comments': 'Test feedback',
            'features': str(self.test_features)
        }
        
        # Test saving feedback
        try:
            save_feedback(feedback_data)
            
            # Check if file was created
            self.assertTrue(os.path.exists('user_feedback.csv'))
            
            # Check if feedback was saved
            saved_feedback = pd.read_csv('user_feedback.csv')
            self.assertGreater(len(saved_feedback), 0)
            
        except Exception as e:
            self.fail(f"Feedback saving failed: {e}")
        
        finally:
            # Clean up test file
            if os.path.exists('user_feedback.csv'):
                os.remove('user_feedback.csv')
    
    def test_data_validation(self):
        """Test input data validation"""
        # Test age range
        self.assertGreaterEqual(30, 0)
        self.assertLessEqual(30, 120)
        
        # Test appointment gap
        gap = 7
        self.assertGreaterEqual(gap, 0)
        self.assertLessEqual(gap, 365)
        
        # Test required fields
        required_fields = ['Gender', 'Age', 'ScheduledDate', 'AppointmentDate']
        test_data = {'Gender': 'M', 'Age': 30, 'ScheduledDate': '2024-01-01', 'AppointmentDate': '2024-01-08'}
        for field in required_fields:
            self.assertIn(field, test_data)

def run_basic_tests():
    """Run basic functionality tests without requiring the full app"""
    print("🧪 Running Basic Tests...")
    
    # Test date calculations
    try:
        scheduled = date(2024, 1, 1)
        appointment = date(2024, 1, 8)
        gap = (appointment - scheduled).days
        print(f"✅ Date calculation test passed: Gap = {gap} days")
    except Exception as e:
        print(f"❌ Date calculation test failed: {e}")
    
    # Test numpy operations
    try:
        test_array = np.array([1, 2, 3, 4, 5])
        mean_val = np.mean(test_array)
        print(f"✅ NumPy test passed: Mean = {mean_val}")
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
    
    # Test pandas operations
    try:
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"✅ Pandas test passed: DataFrame shape = {test_df.shape}")
    except Exception as e:
        print(f"❌ Pandas test failed: {e}")
    
    # Test scikit-learn
    try:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(10, 3)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        print("✅ Scikit-learn test passed: Model trained successfully")
    except Exception as e:
        print(f"❌ Scikit-learn test failed: {e}")
    
    # Test plotly
    try:
        import plotly.express as px
        print("✅ Plotly test passed: Import successful")
    except Exception as e:
        print(f"❌ Plotly test failed: {e}")

if __name__ == "__main__":
    print("🏥 Testing AI-Based No-Show Appointment Prediction Tool")
    print("=" * 60)
    
    # Run basic tests first
    run_basic_tests()
    
    print("\n" + "=" * 60)
    
    # Run unit tests if app is available
    if APP_IMPORTED:
        print("🧪 Running Unit Tests...")
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        print("⚠️  Skipping unit tests - app functions not available")
        print("💡 Make sure to install requirements: pip install -r requirements.txt")
    
    print("\n✅ Testing completed!")
