from flask import Flask, request, render_template_string
import joblib
import numpy as np
import pandas as pd

# Load the trained model
try:
    model = joblib.load('titanic_survival_model.pkl')
    print(f"Model loaded successfully: {type(model)}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Titanic Survival Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f0f8ff; }
        .container { max-width: 600px; margin: 50px auto; background: #e6e6ff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #4b0082; font-size: 2.5em; }
        .result { text-align: center; margin-top: 20px; padding: 20px; border-radius: 5px; }
        .btn { background-color: #4b0082; color: white; border: none; padding: 15px; width: 100%; border-radius: 5px; cursor: pointer; }
        input, select { width: 100%; padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid #ddd; }
        .survived { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .not-survived { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .btn:hover { background-color: #6a0dad; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Titanic Survival Prediction</h1>
        <form method="post">
            <label>Age:</label>
            <input type="text" name="age" required>

            <label>Fare:</label>
            <input type="text" name="fare" required>

            <label>Number of Siblings/Spouses Aboard:</label>
            <input type="text" name="sibsp" required>

            <label>Number of Parents/Children Aboard:</label>
            <input type="text" name="parch" required>

            <label>Class (1/2/3):</label>
            <select name="pclass" required>
                <option value="1">1st Class</option>
                <option value="2">2nd Class</option>
                <option value="3">3rd Class</option>
            </select>

            <label>Gender:</label>
            <select name="sex" required>
                <option value="0">Female</option>
                <option value="1">Male</option>
            </select>

            <button type="submit" class="btn">Submit</button>
        </form>
        
        {% if prediction_text %}
        <div class="result {{ result_class }}">
            <h2>{{ prediction_text }} {{ emoji }}</h2>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ''
    result_class = ''
    emoji = ''
    
    if request.method == 'POST':
        try:
            # Collect form data
            age = float(request.form['age'])
            fare = float(request.form['fare'])
            sibsp = int(request.form['sibsp'])
            parch = int(request.form['parch'])
            pclass = int(request.form['pclass'])
            sex = int(request.form['sex'])
            
            # Prepare the input data for prediction
            input_data = {
                'Pclass': [pclass],
                'Sex': [sex],
                'Age': [age],
                'Fare': [fare],
                'Embarked': ['S' if sex == 0 else 'C'],  # Encoding 'S' for Female, 'C' for Male
                'Familyno': [sibsp + parch]
            }
            
            input_df = pd.DataFrame(input_data)

            # Check if the model is loaded and perform prediction
            if model:
                prediction = model.predict(input_df)
                if prediction[0] == 1:
                    prediction_text = "Survived"
                    result_class = 'survived'
                    emoji = "✅"
                else:
                    prediction_text = "Not Survived"
                    result_class = 'not-survived'
                    emoji = "❌"
            else:
                prediction_text = "Model not loaded. Please check the model file."
                result_class = 'not-survived'
                emoji = "⚠️"
        
        except Exception as e:
            prediction_text = f"Error processing the input: {e}"
            result_class = 'not-survived'
            emoji = "❌"

    return render_template_string(html_template, prediction_text=prediction_text, result_class=result_class, emoji=emoji)

if __name__ == '__main__':
    app.run(debug=True)
