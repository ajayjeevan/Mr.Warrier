import logging
import os
import tempfile

import whisper

from flask import Flask, jsonify, request
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)

USE_LOCAL_ASR = os.environ.get("USE_LOCAL_ASR", "1") == "1"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1")


@app.route("/api/v1/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/api/v1/voice", methods=["POST"])
def audio_bytes_handler():
    device_id = request.headers.get("X-Device-Id") or request.form.get("device_id") or "unknown-device"
    logging.info("Received audio file from device : %s", device_id)

    if "file" not in request.files:
        logging.error("No audio file found on request")
        return jsonify({"ERROR": "No audio file provided"}), 400

    audio_byte_file = request.files["file"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        audio_byte_file.save(tmp_path)

    try:
        text = None
        if USE_LOCAL_ASR:
            text = run_local_asr(tmp_path)
        if not text:
            logging.warning("Cannot execute local ASR")
            return jsonify({"ERROR" : "Transcription unavailable"}), 422

        try:
            assistant_reply_text = call_openrouter(text)
        except Exception as e:
            logging.exception("Openrouter call failed: %s", e)
            return jsonify({"ERROR": "LLM call failed", "details": str(e)}), 502

        print(assistant_reply_text)

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

def call_openrouter(system_prompt):
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not configured on central server.")

    logging.info("Calling OpenRouter (model=%s) for prompt length %d", OPENROUTER_MODEL, len(system_prompt))
    messages = [{"role": "user", "content": system_prompt}]

    client = OpenAI(
        base_url=OPENROUTER_URL,
        api_key=OPENROUTER_KEY,
    )

    try:
        completion = client.chat.completions.create(
            model="openrouter/sherlock-think-alpha",
            messages=messages,
        )
    except Exception as e:
        logging.error(f"ERROR: {e}" )

    return completion.choices[0].message.content


def run_local_asr(audio_path):

    if not USE_LOCAL_ASR:
        logging.debug("Local ASR is disabled.")
    try:
        logging.info("Running local ASR on file : %s", audio_path)
        model = whisper.load_model("small")
        result = model.transcribe(audio_path, language="en", verbose=False, fp16=False)
        text = result.get("text", "").strip()
        return text
    except Exception as e:
        logging.error("Local ASR failed: %s", e)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    logging.info("Starting central voice proxy on 0.0.0.0:%d (USE_LOCAL_ASR=%s)", port, USE_LOCAL_ASR)
    app.run(host="0.0.0.0", port=port, debug=False)

