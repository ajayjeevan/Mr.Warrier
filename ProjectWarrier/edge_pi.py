import logging
import os
import requests

from flask import Flask, jsonify

app = Flask(__name__)

DEVICE_ID = os.environ.get("EDGE_DEVICE_ID", "edge-pi-0")
SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:5000/api/v1/voice")
TIMEOUT = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@app.route("/api/v1/sendvoice", methods=["GET"])
def send_voice():
    wav_bytes = get_voice()
    headers = {
        "X-Device-Id": DEVICE_ID,
    }
    files = {
        "file": ("speech.wav", wav_bytes, "audio/wav"),
    }

    data = {"device_id": DEVICE_ID}

    resp = requests.post(SERVER_URL, headers=headers, files=files, data=data, timeout=TIMEOUT)
    return jsonify({"status": "ok"}), 200


def get_voice():
    file_path = "/home/sedaiadmin/Desktop/kzp/ProjectWarrier/harvard.wav"
    return audio_to_bytes_basic(file_path)

def audio_to_bytes_basic(file_path):
    with open(file_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9090"))
    logging.info("Starting edge pi on 0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)

