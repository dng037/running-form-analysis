import os
import json
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, session
from helper_functions import get_model_output, draw_selected_keypoints_per_person, EXPLANATIONS

app = Flask(__name__)
app.secret_key = os.urandom(12)

@app.route('/inference', methods=["POST"])
def inference():
    if request.method == 'POST':
        file = request.files['image']
        img_bytes = file.read()

        # Get the output from the model
        output = get_model_output(img_bytes)

        return_dict = draw_selected_keypoints_per_person(img_bytes, output["keypoints"], output["keypoints_scores"])
        plt.imsave("static/images/result.jpg", return_dict['image'])
        
        messages = json.dumps({
            "lean": return_dict['lean'],
            "stride": return_dict['stride'],
        })
        session['messages'] = messages
        return redirect(url_for('results'))

@app.route('/results', methods=['GET'])
def results():
    if request.method=='GET':
        messages = json.loads(session['messages'])

        # Overstriding
        if float(messages['stride']) < 83.67:
            stride_explanation = EXPLANATIONS[0]
        # Good stride
        else:
            stride_explanation = EXPLANATIONS[1]
        # Leaning too far forward
        if float(messages['lean']) < 62.11:
            lean_explanation = EXPLANATIONS[4]
        # Too upright
        elif float(messages['lean']) > 73.09:
            lean_explanation = EXPLANATIONS[3]
        else:
            lean_explanation = EXPLANATIONS[2]

        return render_template("results.html", stride=stride_explanation, lean=lean_explanation)

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("homepage.html")

if __name__ == '__main__':
    app.run(port=5000)
