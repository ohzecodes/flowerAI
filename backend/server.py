import os

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from predict import main_flask

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})


@app.route("/")
def home():
    return "Hello Flowers App!"


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
        prediction_result = main_flask(filepath, './checkpoint_vgg11.pth')
        print(prediction_result)
        return jsonify(prediction_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


app.run(port=5000)
