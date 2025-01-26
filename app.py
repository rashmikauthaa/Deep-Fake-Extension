from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model("deepfake_image.h5")  # Load your model

def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size based on your model
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']  # Get uploaded file
    image = Image.open(file.stream)  # Open as PIL image
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]  # Adjust based on your output layer
    result = "Fake" if prediction > 0.5 else "Real"
    return jsonify({"prediction": result, "confidence": float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)