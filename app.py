from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)


model = joblib.load('LinearRegressionModel.pkl')


@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        visits = int(request.form['visits'])
        orders = int(request.form['orders'])
        
        input_data = np.array([[visits, orders]])  

        
        predicted_revenue = model.predict(input_data)[0]  

        
        return render_template('index.html', predicted_revenue=predicted_revenue, visits=visits, orders=orders)

if __name__ == '__main__':
    app.run(debug=True)
