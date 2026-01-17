
import os
import struct
import wave
import pvporcupine
import pyaudio
from openai import OpenAI
import requests
import json
from text_to_speech import speak
import webrtcvad
from collections import deque
from dotenv import load_dotenv
import time

import threading
import whisper

is_processing = False

# --- Configuration ---
# IMPORTANT: Replace with your Picovoice Access Key
load_dotenv()
PICOVOICE_ACCESS_KEY = os.environ.get("PICOVOICE_ACCESS_KEY")
# You can get your free access key from https://console.picovoice.ai/

# Wake word model - for custom wake words, you need to create a model in the Picovoice Console
# For this example, we'll use a built-in keyword.
# Note: "Homie" is a custom wake word. To use it, you would need to create a .ppn file
# on the Picovoice Console and place it in the project directory.
# For now, we'll use "porcupine" as a stand-in.
WAKE_WORD_KEYWORD_PATHS = [pvporcupine.KEYWORD_PATHS["porcupine"]]

# --- Audio Settings ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_LENGTH = 512  # Corresponds to Porcupine's frame length

# --- Speech to Text Provider ---
# Set this environment variable to "local" to use a local Whisper model.
# Otherwise, it will default to using the OpenAI API.
SPEECH_TO_TEXT_PROVIDER = os.environ.get("SPEECH_TO_TEXT_PROVIDER", "openai").lower()

# --- OpenAI Whisper Settings ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if SPEECH_TO_TEXT_PROVIDER == "openai" and not OPENAI_API_KEY:
    print("Warning: SPEECH_TO_TEXT_PROVIDER is 'openai' but OPENAI_API_KEY is not set. Transcription will fail.")

def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

def transcribe_audio(file_path, local_model=None):
    """
    Transcribes the given audio file.
    Uses a local Whisper model if provided, otherwise uses the OpenAI API.
    """
    if local_model:
        print("Using local Whisper model for transcription.")
        try:
            result = local_model.transcribe(file_path)
            return result["text"]
        except Exception as e:
            print(f"Error during local transcription: {e}")
            return None
    else:
        print("Using OpenAI Whisper API for transcription.")
        client = get_openai_client()
        try:
            with open(file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    prompt="This is a conversation with a home assistant."
                )
            return transcript.text
        except Exception as e:
            print(f"Error during OpenAI API transcription: {e}")
            return None

# --- VAD (Voice Activity Detection) Settings ---
VAD_AGGRESSIVENESS = 3
VAD_FRAME_MS = 30
VAD_PADDING_MS = 300
VAD_RATIO = 0.75

def record_until_silence(p, stream, file_path="prompt.wav"):
    """
    Records audio from the stream until a period of silence is detected using webrtcvad.
    """
    print("Listening for speech... Speak your prompt.")
    
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    
    frame_size = int(RATE * VAD_FRAME_MS / 1000)
    ring_buffer_size = VAD_PADDING_MS // VAD_FRAME_MS
    ring_buffer = deque(maxlen=ring_buffer_size)
    
    triggered = False
    frames = []
    
    start_time = time.time()
    TIMEOUT_SECONDS = 10 

    while not triggered and (time.time() - start_time) < TIMEOUT_SECONDS:
        frame = stream.read(frame_size)
        is_speech = vad.is_speech(frame, RATE)
        if is_speech:
            print("Speech detected, starting to record.")
            triggered = True
            frames.extend(list(ring_buffer))
            frames.append(frame)
            ring_buffer.clear()
        else:
            ring_buffer.append(frame)

    if not triggered:
        print("No speech detected within the timeout period.")
        return False

    SILENCE_FRAMES = int((VAD_PADDING_MS * 2) / VAD_FRAME_MS) 
    silent_frames_count = 0

    while silent_frames_count < SILENCE_FRAMES:
        frame = stream.read(frame_size)
        frames.append(frame)
        is_speech = vad.is_speech(frame, RATE)
        if not is_speech:
            silent_frames_count += 1
        else:
            silent_frames_count = 0

    print("Recording finished.")

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return True

def chat_with_assistant(prompt):
    """
    Sends the prompt to the chat API and gets a response.
    """
    url = "http://127.0.0.1:8000/api/chat"
    headers = {"Content-Type": "application/json"}
    try:
        data = {"prompt": prompt}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        chat_response = response.json()
        return chat_response.get("response")
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the server: {e}")
        return "I am having trouble connecting to my brain."

def process_wake_word_interaction(pa, audio_stream, local_model):
    """
    Handles the entire interaction after a wake word is detected.
    This runs in a separate thread to avoid blocking the main audio loop.
    """
    global is_processing
    try:
        is_processing = True
        print("Wake word detected!")
        
        # The main audio stream is already paused by the main loop.
        # We can now safely record without overflowing the buffer.
        recording_made = record_until_silence(pa, audio_stream, "prompt.wav")
        
        if recording_made:
            prompt_text = transcribe_audio("prompt.wav", local_model=local_model)
            if prompt_text:
                print(f"You said: {prompt_text}")
                
                assistant_response = chat_with_assistant(prompt_text)
                print(f"Assistant: {assistant_response}")
                
                speak(assistant_response)
            else:
                speak("I'm sorry, I didn't catch that.")
        
        print("Listening for wake word ('porcupine')...")

    finally:
        is_processing = False

def main():
    """
    Main function to run the voice client.
    """
    global is_processing

    if PICOVOICE_ACCESS_KEY == "YOUR_PICOVOICE_ACCESS_KEY_HERE":
        print("="*80)
        print("ERROR: Picovoice Access Key not set.")
        print("Please get your free access key from https://console.picovoice.ai/ and")
        print("set it as the PICOVOICE_ACCESS_KEY environment variable or in the script.")
        print("="*80)
        return

    porcupine = None
    pa = None
    audio_stream = None
    local_whisper_model = None

    if SPEECH_TO_TEXT_PROVIDER == "local":
        print("Loading local Whisper model. This may take a moment...")
        try:
            local_whisper_model = whisper.load_model("base")
            print("Local Whisper model loaded successfully.")
        except Exception as e:
            print(f"Error loading local Whisper model: {e}")
            pass

    try:
        porcupine = pvporcupine.create(
            access_key=PICOVOICE_ACCESS_KEY,
            keyword_paths=WAKE_WORD_KEYWORD_PATHS
        )

        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=CHANNELS,
            format=FORMAT,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )

        print(f"Speech-to-text provider: {SPEECH_TO_TEXT_PROVIDER}")
        print("Listening for wake word ('porcupine')...")

        while True:
            if not is_processing:
                try:
                    pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow = False)
                    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                    keyword_index = porcupine.process(pcm)

                    if keyword_index >= 0:
                        thread = threading.Thread(target=process_wake_word_interaction, args=(pa, audio_stream, local_whisper_model))
                        thread.start()
                except (IOError, OSError) as e:
                    # This is to catch the input overflow error and continue
                    print(f"Error reading from audio stream: {e}")
                    # We can try to recover by stopping and starting the stream
                    audio_stream.stop_stream()
                    audio_stream.start_stream()

            else:
                # If processing, sleep briefly to avoid busy-waiting
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()
        if pa is not None:
            pa.terminate()

if __name__ == '__main__':
    main()
