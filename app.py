from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image

import joblib
model=joblib.load('model2.pkl')

app = Flask(__name__)

@app.route('/')
def crop_page():
    return render_template('health1.html')

@app.route('/predict', methods = ['POST'])
def recomend_crop():
    try:
        # Get values from form
        age = request.form.get('age')
        anaemia = request.form.get('anaemia')
        creatinine = request.form.get('creatinine')
        diabetes = request.form.get('diabetes')
        ejection = request.form.get('ejection')
        bp = request.form.get('bp')
        plateletes = request.form.get('plateletes')
        serum = request.form.get('serum')
        sodium = request.form.get('sodium')
        sex = request.form.get('sex')
        smoking = request.form.get('smoking')
        time = request.form.get('time')

        # ✅ Check for any missing inputs
        inputs = [age, anaemia, creatinine, diabetes, ejection, bp, plateletes,
                  serum, sodium, sex, smoking, time]

        if any(val == '' or val is None for val in inputs):
            return "Error: All fields must be filled.", 400

        # ✅ Convert all to float
        input_data = [float(val) for val in inputs]

        # Make prediction
        input_array = np.array(input_data).reshape(1, -1)
        result = model.predict(input_array)

        label = "No Risk" if result[0] == 0 else "At Risk"
        return render_template('health1.html',result=label)

    
    except ValueError:
        return "Error: Invalid input. Please enter numbers only.", 400


    



if __name__ == '__main__':
    app.run(debug = True,port=5001)
