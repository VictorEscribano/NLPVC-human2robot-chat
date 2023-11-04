import pyaudio
import whisper
import numpy as np
import audioop
import time
import requests
import json

class LiveTranscriber:
    def __init__(self):
        # Carga el modelo de Whisper
        self.model = whisper.load_model("medium")

        # Inicializa PyAudio
        self.p = pyaudio.PyAudio()

        # Define los parámetros de audio
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.silence_threshold = 700
        self.speech_timeout = 1.0

        # Inicializa el stream de audio
        self.stream = self.p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)

        # Inicializa el buffer de audio y las variables de control
        self.audio_buffer = np.array([], dtype=np.int16)
        self.last_speech_time = None
        self.is_speaking = False

    def send_prompt_to_api(self, prompt):
        # The URL of the API endpoint
        url = "http://localhost:11434/api/generate"

        # The data payload as a dictionary, with the prompt variable
        payload = {
            "model": "orca-mini",
            "prompt": prompt
        }

        # Convert the payload to JSON format
        payload_json = json.dumps(payload)

        # Set the appropriate headers for JSON
        headers = {
            'Content-Type': 'application/json'
        }

        # Send the POST request to the API
        response = requests.post(url, data=payload_json, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Initialize a variable to hold the full response text
            full_response_text = ""
            # Print each part of the response and concatenate the text
            for line in response.iter_lines():
                if line:  # filter out keep-alive new lines
                    response_part = json.loads(line.decode('utf-8'))
                    full_response_text += response_part["response"]
            return full_response_text
        else:
            print(f"Failed to get response: {response.status_code}")
            return None
    
    def listen_and_transcribe(self):
        print("Listening...")
        try:
            while True:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)

                volume = audioop.rms(data, 2)
                if volume > self.silence_threshold:
                    if not self.is_speaking:
                        print("Speech detected, recording...")
                    self.is_speaking = True
                    self.last_speech_time = time.time()
                    self.audio_buffer = np.append(self.audio_buffer, audio_data)
                else:
                    if self.is_speaking and (time.time() - self.last_speech_time > self.speech_timeout):
                        audio_float = self.audio_buffer.astype(np.float32)
                        audio_float /= np.iinfo(np.int16).max

                        print("Silence detected, transcribing...")
                        result = self.model.transcribe(audio_float)
                        print("Transcribed text:", result["text"])

                        # Enviar el texto transcribido como prompt al modelo de lenguaje natural
                        full_response = self.send_prompt_to_api(result["text"])
                        if full_response is not None:
                            print("Response from language model:", full_response)
                        
                        self.audio_buffer = np.array([], dtype=np.int16)
                        self.is_speaking = False

        except KeyboardInterrupt:
            self.cleanup()

    def cleanup(self):
        print("Stopping...")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# To use the class
transcriber = LiveTranscriber()
transcriber.listen_and_transcribe()
