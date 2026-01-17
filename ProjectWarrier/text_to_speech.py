
import os
from gtts import gTTS
from pygame import mixer

def speak(text, file_path="response.mp3"):
    """
    Converts text to speech and plays it.
    """
    if not text:
        print("No text to speak.")
        return

    try:
        # Create the gTTS object
        tts = gTTS(text=text, lang='en')
        
        # Save the audio file
        tts.save(file_path)
        
        # Initialize the pygame mixer
        mixer.init()
        
        # Load the audio file
        mixer.music.load(file_path)
        
        # Play the audio file
        mixer.music.play()
        
        # Wait for the audio to finish playing
        while mixer.music.get_busy():
            pass
        
        # Clean up
        mixer.quit()
        
        # Optionally remove the file
        # os.remove(file_path)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Example usage when running the script directly
    speak("Hello, this is a test of the text-to-speech functionality.")
    speak("This is a second test.")
