@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all features from the form
        features = [
            float(request.form['tenure']),
            float(request.form['monthly_charges']),
            float(request.form['total_charges']),
            float(request.form['senior_citizen']),
            float(request.form['partner']),
            float(request.form['dependents']),
            float(request.form['phone_service']),
            float(request.form['multiple_lines']),
            float(request.form['internet_service']),
            float(request.form['online_security']),
            # Add all other features your model expects...
        ]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]
        
        output = "Likely to Churn" if prediction[0] == 1 else "Likely to Stay"
        
        return render_template('index.html', 
                             prediction_text=f'Customer is {output} (Churn Probability: {probability:.2%})')
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error making prediction: {str(e)}')
