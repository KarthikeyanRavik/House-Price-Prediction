from flask import Flask, render_template,request
import pandas as pd
import pickle
import sklearn
import numpy as np
import locale

app = Flask(__name__)  # Initialize Flask app

# Read the CSV file
data = pd.read_csv("Cleaned_Bengaluru_House_Data.csv")
model_path = "C:\\Users\\karthikeyan\\Videos\\course\\Machine learning project\\House Price prediction\\House price prediction\\RidgeModel.pkl"
pipe = pickle.load(open(model_path,'rb'))

@app.route('/')  # Route to home page
def index():
    locations = sorted(data['location'].dropna().unique())  # Get unique locations
    return render_template('index.html', locations=locations)  # Render HTML from templates

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('BHK')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location,bhk,bath,sqft)

    input = pd.DataFrame([[location,sqft,bath,bhk]],columns = ['location','total_sqft','bath','BHK'])
    prediction = pipe.predict(input)[0] * 100000

    locale.setlocale(locale.LC_ALL, 'en_IN')

    formatted_price = locale.format_string("%.2f", prediction, grouping=True)

    return formatted_price

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Run Flask on port 5001
