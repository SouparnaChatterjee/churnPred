
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model with error handling
try:
    model = pickle.load(open('churn_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print("Current directory:", os.getcwd())
    print("Files in directory:", os.listdir())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from the form
        tenure = float(request.form['tenure'])
        monthly_charges = float(request.form['monthly_charges'])
        
        # Create features array
        features = np.array([[tenure, monthly_charges]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Prepare output
        output = "Likely to Churn" if prediction[0] == 1 else "Likely to Stay"
        
        return render_template('index.html', 
                             prediction_text=f'Customer is {output} (Churn Probability: {probability:.2%})')
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error making prediction: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
