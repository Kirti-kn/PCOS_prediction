from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
model = load_model('pcos_model.h5')  # Load the trained model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.json['image_data']
    image = load_img(image_data, target_size=(100, 100))
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # Normalize the image array
    image_array = np.expand_dims(image_array, axis=0)  # Add an extra dimension
    prediction = model.predict(image_array)[0]
    class_labels = ['Non-PCOS', 'Acne', 'Hairy', 'Bache', 'Skin Tags']
    predicted_class = class_labels[np.argmax(prediction)]
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run()
