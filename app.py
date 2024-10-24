from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)

# loading models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Convert form inputs to appropriate types
            Year = int(request.form['Year'])  # Convert to integer
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])  # Convert to float
            pesticides_tonnes = float(request.form['pesticides_tonnes'])  # Convert to float
            avg_temp = float(request.form['avg_temp'])  # Convert to float
            Area = request.form['Area']  # Area as string
            Item = request.form['Item']  # Crop type as string

            # Validate numerical inputs
            if Year < 1990:  # Year must be greater than or equal to 1990
                return "Error: Year must be greater than or equal to 1990."
            if average_rain_fall_mm_per_year <= 0:
                return "Error: Average Rainfall must be greater than 0."
            if pesticides_tonnes < 0:
                return "Error: Pesticides cannot be negative."
            if avg_temp <= 0:
                return "Error: Average Temperature must be greater than 0."

            # Validate Area and Item (you can adjust based on dataset constraints)
            if not Area or not Item:
                return "Error: Area and Crop Type cannot be empty."

            # Prepare input features
            features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

            # Preprocess the input features
            transformed_features = preprocessor.transform(features)

            # Make a prediction
            prediction = dtr.predict(transformed_features).reshape(1, -1)

            # Return the prediction
            return render_template('index.html', prediction=f"Predicted Yield: {prediction[0][0]}")

        except ValueError:
            return "Error: Please enter valid numerical values."

if __name__ == "__main__":
    app.run(debug=True)
