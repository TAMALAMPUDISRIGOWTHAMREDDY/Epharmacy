import joblib
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user inputs
        age = request.form['age']
        gender = request.form['Gender']
        symptoms = request.form['symptoms']
        
        # Predict the medicine based on the entered symptoms
        prediction = model.predict([symptoms])
        medicine = prediction[0]
        
        return render_template('results.html', gender=gender, age=age, symptoms=symptoms, medicine=medicine)
    
    return render_template('index.html')



# Emergency First Aid Pages (Ensure correct paths)
@app.route("/cpr")
def cpr():
    return render_template("first_aid/cpr.html")

@app.route("/electric_shock")
def electric_shock():
    return render_template("first_aid/electric_shocks.html")

@app.route("/fire_accidents")
def fire_accidents():
    return render_template("first_aid/fire_accidents.html")

@app.route("/snake_bites")
def snake_bites():
    return render_template("first_aid/snake_bites.html")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
