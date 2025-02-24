from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the form
    features = [float(x) for x in request.form.values()]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)

    output = "Likely to Churn" if prediction[0] == 1 else "Likely to Stay"

    return render_template('index.html',
                          prediction_text=f'Customer is {output}')

if __name__ == "__main__":
    app.run(debug=True)
