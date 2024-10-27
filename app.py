import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained RandomForestClassifier model
model = joblib.load('crop_recommendation_model.joblib')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    pH = float(request.form['pH'])
    rainfall = float(request.form['rainfall'])

    # Create input dataframe for prediction
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [pH],
        'rainfall': [rainfall]
    })

    # Make prediction using the loaded model
    predicted_crop = model.predict(input_data)[0]

    # Render result template with the predicted crop
    return render_template('result.html', crop=predicted_crop)

if __name__ == '__main__':
    app.run(debug=True)
