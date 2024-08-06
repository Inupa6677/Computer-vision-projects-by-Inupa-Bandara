# import libraries
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('Dog_model.h5')

# Define the home route
@app.route('/')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)



