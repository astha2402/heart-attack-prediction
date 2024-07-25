import pickle
from flask import Flask, render_template, request
import numpy as np

#create an object of the class flask

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#functions
#route is actually used to specify the end points.

@app.route('/')

def home():
    return render_template('index.html') #returns an html file when the url is clicked on

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    output = model.predict(final_features)
    if(output[0] == 1):
        answer = "high"
    else:
        answer = "low"
    return render_template('index.html', prediction_text=f'Chance of heart attack is {answer}.')

if __name__ =='__main__':
    app.run(debug=True)