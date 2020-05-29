from flask import Flask, flash, request, redirect, jsonify
import cv2
import json
import numpy as np
from preprocess import preprocess_data
from gen_model import create_model


app = Flask(__name__)
app.secret_key = "1234"
IMG_SIZE = (32, 32)
PARAMS_FILE = "mymodel.json"

model = create_model()
with open(PARAMS_FILE) as json_file:
    params = json.load(json_file)
for i, param in params.items():
    model.layers[int(i)].weights = np.array(param[0])
    model.layers[int(i)].bias = np.array(param[1])


@app.route('/predict', methods=['POST'])
def predict():
    files = [f for _, f in request.files.items()]
    if not files:
        flash('No files')
        return "No Files"
    images = np.array([cv2.resize(cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_UNCHANGED), IMG_SIZE)
                       for f in files])
    images = preprocess_data(images)
    images = images.transpose((0, 3, 1, 2))
    predicted = model.predict(images, False)
    predicted = np.argmax(predicted, axis=1)
    results = {i + 1: 'hotdog' if p == 0 else 'not hotdog' for i, p in enumerate(predicted)}
    return jsonify(results)
