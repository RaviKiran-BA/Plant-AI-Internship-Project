from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__)

# Load models
models = {
    "plant_disease_detection": load_model("models/plant_disease_detection.h5"),
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read and preprocess the image
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100, 100))  # Resize to the expected input shape
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            img = img.astype('float32') / 255.0  # Normalize the image

            results = {}
            for model_name, model in models.items():
                prediction = model.predict(img)
                results[model_name] = np.argmax(prediction)  # Get the class index

            return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

