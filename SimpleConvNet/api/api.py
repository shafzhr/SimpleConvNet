import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import requests


UPLOAD_FOLDER = './images'
PREDICTION_SERVICE = "http://127.0.0.1:5001/predict"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "1234"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    files = [f for _, f in request.files.items()]
    if not files:
        flash('No files')
        return redirect(request.url)
    
    for f in files:
        f.seek(0)
    req = {i: (f.read())
            for i, f in request.files.items() if not f.filename == ''}
    res = requests.post(PREDICTION_SERVICE, files=req)
    return res.content
