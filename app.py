import json
import os
from flask import Flask, render_template, session, request, flash, redirect, url_for
from utils.ml_model import StudentModel

# Initialize Flask App
app = Flask(__name__, template_folder='template', static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-123')

# Load Configuration
try:
    with open('config.json', 'r') as c:
        config = json.load(c)
        wparams = config.get("params", {})
except FileNotFoundError:
    wparams = {"admin_user": "admin", "admin_password": "password"}

# Initialize Model
try:
    model_handler = StudentModel('model.sav')
except Exception as e:
    print(f"CRITICAL: Could not load model: {e}")
    model_handler = None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        username = request.form.get("uname")
        userpass = request.form.get("pass")
        
        if username == wparams.get('admin_user') and userpass == wparams.get('admin_password'):
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials. Please try again.", "danger")
            
    return render_template('login.html', params=wparams)

@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template("index.html", params=wparams)

@app.route("/prediction", methods=['POST'])
def prediction():
    if 'user' not in session:
        return redirect(url_for('index'))
    
    if model_handler is None:
        return "Model not loaded. Please contact administrator.", 500

    # Collect form data
    form_data = {
        'sex': request.form.get('sex'),
        'age': request.form.get('age'),
        'address': request.form.get('address'),
        'Medu': request.form.get('Medu'),
        'Fedu': request.form.get('Fedu'),
        'traveltime': request.form.get('traveltime'),
        'failures': request.form.get('failures'),
        'paid': request.form.get('paid'),
        'higher': request.form.get('higher'),
        'internet': request.form.get('internet'),
        'goout': request.form.get('goout'),
        'G1': request.form.get('G1'),
        'G2': request.form.get('G2')
    }

    # Perform prediction
    result = model_handler.predict(form_data)
    importance = model_handler.get_feature_importance()
    
    if result is None:
        flash("Error during prediction. Please check your inputs.", "danger")
        return redirect(url_for('dashboard'))

    # Format result for display
    prediction_text = f"Score Category: {result}"
    
    return render_template('prediction.html', 
                           prediction_text=prediction_text, 
                           importance=importance)

@app.route("/logout")
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
