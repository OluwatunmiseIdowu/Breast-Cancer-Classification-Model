from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    input_data = [
        float(request.form['mean_radius']),
        float(request.form['mean_texture']),
        float(request.form['mean_perimeter']),
        float(request.form['mean_area']),
        float(request.form['mean_smoothness']),
        float(request.form['mean_compactness']),
        float(request.form['mean_concavity']),
        float(request.form['mean_concave_points']),
        float(request.form['mean_symmetry']),
        float(request.form['mean_fractal_dimension']),
        
    ]

    # Convert input data to numpy array and reshape for prediction
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data_reshaped)

    # Return the result
    result = 'Malignant' if prediction[0] == 0 else 'Benign'
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
