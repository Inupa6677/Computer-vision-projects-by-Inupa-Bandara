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

# Define a route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Load and preprocess the image
    img = image.load_img(file, target_size=(128, 128))  # Adjust target_size as needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    
    # Map the predicted class to class names (adjust according to your classes)
    class_names = [
        'Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German Shepherd',
        'Golden Retriever', 'Labrador Retriever', 'Poodle', 'Rottweiler', 'Yorkshire Terrier'
    ]
    
    result = class_names[predicted_class]
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)



