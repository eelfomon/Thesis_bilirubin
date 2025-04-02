import os
import uuid
from io import BytesIO

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from process_head import Bili_Level

app = Flask(__name__)

@app.route("/")
def root():
    return jsonify({"message": "Hello World"})

@app.route("/hello/<name>")
def say_hello(name):
    return jsonify({"message": f"Hello {name}"})

def generate_random_filename(filename):
    ext = filename.split(".")[-1]
    random_name = str(uuid.uuid4())
    return f"{random_name}.{ext}"

@app.route("/files", methods=["POST"])
def file_contents():
    files = request.files.getlist("files")
    filenames = [secure_filename(file.filename) for file in files]
    return jsonify({"filenames": filenames})

@app.route("/process", methods=["POST"])
def process_head():
    try:
        head = request.files["head"]
        chest = request.files["chest"]
        # Read image data from buffers
        head_data = head.read()
        chest_data = chest.read()

        # Process the image using head_processor
        predicted_head_value = Bili_Level(BytesIO(head_data).getvalue())
        predicted_chest_value = Bili_Level(BytesIO(chest_data).getvalue(), 'chest')

        # Calculate the average predicted value
        average_predicted_value = (predicted_head_value + predicted_chest_value) / 2

        chart_data = {
            "green_bar": [
                {"x": 10.90533563, "y": 4.039812646},
                {"x": 16.35800344, "y": 4.918032787},
                {"x": 21.81067126, "y": 5.035128806},
                {"x": 25.77624785, "y": 5.971896956},
                {"x": 30.98106713, "y": 7.025761124},
                {"x": 37.17728055, "y": 7.903981265},
                {"x": 50.06540448, "y": 9.25058548},
                {"x": 64.68846816, "y": 11.18266979},
                {"x": 79.55938038, "y": 11.94379391},
                {"x": 101.6179002, "y": 12.93911007},
                {"x": 112.2753873, "y": 13.64168618},
                {"x": 136.0688468, "y": 13.34894614},
                {"x": 142.0172117, "y": 13.34894614},
            ],
            "red_bar": [
                {"x": 10.90533563, "y": 7.142857143},
                {"x": 21.81067126, "y": 8.079625293},
                {"x": 36.43373494, "y": 12.11943794},
                {"x": 55.51807229, "y": 15.39812646},
                {"x": 88.48192771, "y": 17.85714286},
                {"x": 113.0189329, "y": 17.62295082},
                {"x": 136.0688468, "y": 17.44730679},
                {"x": 142.5129088, "y": 17.91569087},
            ],
            "blue_bar": [
                {"x": 10.16179002, "y": 5.386416862},
                {"x": 15.36660929, "y": 6.206088993},
                {"x": 22.55421687, "y": 6.206088993},
                {"x": 37.17728055, "y": 10.24590164},
                {"x": 57.25301205, "y": 12.88056206},
                {"x": 85.25989673, "y": 15.22248244},
                {"x": 111.5318417, "y": 16.15925059},
                {"x": 132.5989673, "y": 15.45667447},
                {"x": 142.0172117, "y": 15.6323185},
            ]
        }
        return jsonify({"success": "success", "data": {"predicted_value": average_predicted_value, "chart_data": chart_data}})

    except Exception as e:
        return jsonify({"success": "error", "error_message": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000,)
