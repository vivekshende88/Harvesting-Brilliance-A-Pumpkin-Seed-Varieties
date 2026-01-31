
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load trained model, scaler, encoder
model, scaler, encoder = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = None

    if request.method == 'POST':
        values = [
            float(request.form['Area']),
            float(request.form['Perimeter']),
            float(request.form['Major_Axis_Length']),
            float(request.form['Minor_Axis_Length']),
            float(request.form['Eccentricity']),
            float(request.form['Convex_Area']),
            float(request.form['Equiv_Diameter']),
            float(request.form['Solidity']),
            float(request.form['Extent']),
            float(request.form['Roundness']),
            float(request.form['Aspect_Ratio']),
            float(request.form['Compactness'])
        ]

        values_scaled = scaler.transform([values])
        pred = model.predict(values_scaled)[0]
        result = encoder.inverse_transform([pred])[0]

        prediction_text = f"Hence, based on calculation: Your seed lies in {result} class"

    return render_template('predict.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=False, port=5000)
