
import requests
import json
from text_to_speech import speak

def chat_with_assistant():
    """
    Main loop to chat with the AI assistant.
    """
    url = "http://127.0.0.1:8000/api/chat"
    headers = {"Content-Type": "application/json"}

    print("Chat with the AI assistant. Type 'exit' to quit.")

    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() == 'exit':
                break

            data = {"prompt": prompt}
            
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # Raise an exception for bad status codes

            chat_response = response.json()
            assistant_response = chat_response.get("response")

            print(f"Assistant: {assistant_response}")
            
            # Convert the assistant's response to speech
            speak(assistant_response)

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with the server: {e}")
            speak("I am having trouble connecting to my brain.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    chat_with_assistant()
