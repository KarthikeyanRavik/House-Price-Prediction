from flask import Flask, render_template,request
import pandas as pd
import pickle
import sklearn
import numpy as np
import locale
import os


app = Flask(__name__)  # Initialize Flask app

# Read the CSV file
csv_path = os.path.join(os.path.dirname(__file__), "Cleaned_Bengaluru_House_Data.csv")
data = pd.read_csv(csv_path,encoding="utf-8")
model_path = "RidgeModel.pkl"
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
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)  # Run Flask on port 5001
