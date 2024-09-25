import os

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from predict import main_flask
import logging
cors_resource = "*" # Change this to the URL of your frontend app

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": cors_resource}})


@app.route("/")
def home():
    return "Hello Flowers App! please use the predict endpoint to classify an image"


@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    print(file.filename)
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    try:

        filename = secure_filename(file.filename)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tmp_dir = os.path.join(script_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        filepath = os.path.join(script_dir, '/tmp', filename)
        file.save(filepath)

        # Call the main_flask function with the appropriate parameters
        prediction_result = main_flask(filepath, './checkpoint_squeezenet.pth')
        print(prediction_result)
        logging.info(prediction_result)
        return jsonify(prediction_result), 200

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))