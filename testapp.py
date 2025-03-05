from flask import Flask, render_template, request
import pickle

import numpy as np

app = Flask(__name__)

@app.route('/')
def career():
    return render_template("hometest.html")

@app.route('/blog')
def blog():
    return render_template('blog.html')    

@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        print(result)
        
        # Convert form data to a list of values
        res = result.to_dict(flat=True)
        arr = list(res.values())

        try:
            # Convert the list of values to a numpy array of type float
            data = np.array(arr, dtype=float).reshape(1, -1)
        except ValueError:
            return "Error: Please ensure all inputs are numeric."

        print(data)

        # Load the model
        loaded_model = pickle.load(open("careerlast.pkl", 'rb'))
        
        # Make predictions
        predictions = loaded_model.predict(data)
        print(predictions)

        # Get prediction probabilities
        pred_proba = loaded_model.predict_proba(data)
        print(pred_proba)
        
        pred_proba = pred_proba > 0.05

        i = 0
        j = 0
        index = 0
        res = {}
        final_res = {}

        while j < 17:
            if pred_proba[i, j]:
                res[index] = j
                index += 1
            j += 1

        index = 0
        for key, values in res.items():
            if values != predictions[0]:
                final_res[index] = values
                index += 1

        jobs_dict = {
            0: 'AI ML Specialist',
            1: 'API Integration Specialist',
            2: 'Application Support Engineer',
            3: 'Business Analyst',
            4: 'Customer Service Executive',
            5: 'Cyber Security Specialist',
            6: 'Data Scientist',
            7: 'Database Administrator',
            8: 'Graphics Designer',
            9: 'Hardware Engineer',
            10: 'Helpdesk Engineer',
            11: 'Information Security Specialist',
            12: 'Networking Engineer',
            13: 'Project Manager',
            14: 'Software Developer',
            15: 'Software Tester',
            16: 'Technical Writer'
        }

        job0 = predictions[0]
        return render_template("testafter.html", final_res=final_res, job_dict=jobs_dict, job0=job0)

    # Return a default response in case the request method is not POST
    return "Invalid request method."

if __name__ == '__main__':
    app.run(debug=True)