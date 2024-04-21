from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

app = Flask(__name__)

# Your Python code for data analysis
# Import necessary libraries and define functions here

# Load the trained model
database = pd.read_csv("../HackTech/data.csv", header=0)
database.drop('Unnamed: 32', axis=1, inplace=True)
database['diagnosis'] = database['diagnosis'].map({'M': 1, 'B': 0})

train_dataset, test_dataset = train_test_split(database, test_size=0.3)

features_mean = list(database.columns[2:11])
predictor_var = features_mean
outcome_var = 'diagnosis'
model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=2)
model.fit(train_dataset[predictor_var], train_dataset[outcome_var])


def cancer_diagnosis_predict_s(data, model=model, predictors=predictor_var):
    new_predictions = model.predict(data[predictors])
    new_res = [int(prediction) for prediction in new_predictions]
    return new_res


@app.route('/')
def index():
    return render_template('test.html')


@app.route('/analyze_data', methods=['POST'])
def analyze_data():
    # Extract data from the form submission
    patient_data = {
        'radius_mean': [request.form['radius_mean']],
        'texture_mean': [request.form['texture_mean']],
        'perimeter_mean': [request.form['perimeter_mean']],
        'area_mean': [request.form['area_mean']],
        'smoothness_mean': [request.form['smoothness_mean']],
        'compactness_mean': [request.form['compactness_mean']],
        'concavity_mean': [request.form['concavity_mean']],
        'concave_points_mean': [request.form['concave_points_mean']],
        'symmetry_mean': [request.form['symmetry_mean']],
        'fractal_dimension_mean': [request.form['fractal_dimension_mean']],
        'radius_se': [request.form['radius_se']],
        'texture_se': [request.form['texture_se']],
        'perimeter_se': [request.form['perimeter_se']],
        'area_se': [request.form['area_se']],
        'smoothness_se': [request.form['smoothness_se']],
        'compactness_se': [request.form['compactness_se']],
        'concavity_se': [request.form['concavity_se']],
        'concave_points_se': [request.form['concave_points_se']],
        'symmetry_se': [request.form['symmetry_se']],
        'fractal_dimension_se': [request.form['fractal_dimension_se']],
        'radius_worst': [request.form['radius_worst']],
        'texture_worst': [request.form['texture_worst']],
        'perimeter_worst': [request.form['perimeter_worst']],
        'area_worst': [request.form['area_worst']],
        'smoothness_worst': [request.form['smoothness_worst']],
        'compactness_worst': [request.form['compactness_worst']],
        'concavity_worst': [request.form['concavity_worst']],
        'concave_points_worst': [request.form['concave_points_worst']],
        'symmetry_worst': [request.form['symmetry_worst']],
        'fractal_dimension_worst': [request.form['fractal_dimension_worst']]
    }

    # Convert data to DataFrame
    patient_df = pd.DataFrame(patient_data)

    # Perform data analysis
    results = cancer_diagnosis_predict_s(patient_df)

    # Pass the results to the HTML template for rendering
    return render_template('result.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)