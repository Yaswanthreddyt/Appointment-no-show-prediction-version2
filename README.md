# AI-Based No-Show Appointment Prediction Tool

A Streamlit web application that uses machine learning to predict whether patients are likely to show up for their medical appointments.

## Features

- **User Input Form**: Collects patient information including demographics, appointment details, and medical conditions
- **AI Prediction**: Uses a trained Random Forest model to predict no-show probability
- **Confidence Scores**: Provides probability scores for both show and no-show outcomes
- **Feature Importance**: Visualizes which factors most influence the prediction
- **User Feedback**: Collects and stores user feedback on prediction accuracy
- **Responsive Design**: Clean, modern UI that works well on different screen sizes

## Input Features

1. **Basic Information**
   - Gender (Male/Female)
   - Age (0-120)

2. **Appointment Details**
   - Scheduled Date
   - Appointment Date
   - Appointment Gap (automatically calculated)

3. **Location & Communication**
   - Neighbourhood
   - SMS Received (Yes/No)

4. **Medical Conditions**
   - Hipertension
   - Diabetes
   - Alcoholism
   - Handcap

## Installation

1. **Clone or download** this repository to your local machine

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your data file** is in the correct location:
   - Place your `model.pkl` file in the root directory, OR
   - Ensure `Data/KaggleV2-May-2016.csv` exists for automatic model training

## Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Fill out the patient information** form in the sidebar

4. **Click "Predict No-Show Probability"** to get results

5. **View the prediction** along with confidence scores and feature importance

6. **Provide feedback** on prediction accuracy (optional)

## Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 9 engineered features from patient data
- **Output**: Binary classification (Show/No-Show) with probability scores
- **Training**: Automatically trains on your data if no pre-trained model exists

## File Structure

```
no_show_prediction_tool/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── model.pkl             # Trained model (will be created if not exists)
├── user_feedback.csv     # User feedback storage (created automatically)
├── Data/                 # Data directory
│   └── KaggleV2-May-2016.csv  # Training data
└── UI/                   # UI components (if any)
```

## Customization

### Using Your Own Model

If you have a pre-trained model:

1. Save it as `model.pkl` in the root directory
2. Ensure the model expects the same feature format:
   ```python
   [gender_encoded, age, neighbourhood_encoded, sms_encoded, 
    hipertension, diabetes, alcoholism, handcap, appointment_gap]
   ```

### Modifying Features

To add/remove features:

1. Update the `feature_columns` list in `load_or_train_model()`
2. Modify the `encode_categorical_features()` function
3. Update the `get_feature_importance()` function
4. Adjust the input form in the main function

## Output Files

- **user_feedback.csv**: Stores user feedback with timestamps and prediction details
- **model.pkl**: Trained model file (created automatically if needed)

## Disclaimer

⚠️ **This tool is for demonstration purposes only and not for medical use.** Predictions are based on historical data patterns and should not be used to make medical decisions.

## Technical Details

- **Framework**: Streamlit
- **Machine Learning**: scikit-learn
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Pickle

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port with `streamlit run app.py --server.port 8502`

2. **Model loading errors**: Ensure your `model.pkl` file is compatible or let the app train a new one

3. **Data loading errors**: Check that your CSV file path is correct

4. **Package conflicts**: Use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify your data files are in the correct locations

## License

This project is part of an MSc research project. Please use responsibly and in accordance with your institution's guidelines.
