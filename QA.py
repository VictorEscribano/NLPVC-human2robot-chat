import pyaudio
import whisper
import numpy as np
import audioop
import time
import requests
import json
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
import simpleaudio as sa
from datasets import load_dataset

class LiveTranscriber:
    def __init__(self):
        # Load Whisper model
        self.model = whisper.load_model("medium")
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        # Audio parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.silence_threshold = 700
        self.speech_timeout = 1.0
        # Initialize audio stream
        self.stream = self.p.open(format=self.format, channels=self.channels,
                                  rate=self.rate, input=True, frames_per_buffer=self.chunk)
        # Initialize audio buffer and control variables
        self.audio_buffer = np.array([], dtype=np.int16)
        self.last_speech_time = None
        self.is_speaking = False

        # Load TTS models
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        # Load speaker embeddings
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        # Initialize the play object as None
        self.play_obj = None

    def send_prompt_to_api(self, prompt):
        # API endpoint URL
        url = "http://localhost:11434/api/generate"
        # Data payload with prompt
        payload = {
            "model": "orca-mini",
            "prompt": prompt
        }
        # Convert payload to JSON
        payload_json = json.dumps(payload)
        # Set headers for JSON
        headers = {
            'Content-Type': 'application/json'
        }
        # Send POST request to API
        response = requests.post(url, data=payload_json, headers=headers)
        # Check if request was successful
        if response.status_code == 200:
            full_response_text = ""
            for line in response.iter_lines():
                if line:
                    response_part = json.loads(line.decode('utf-8'))
                    full_response_text += response_part["response"]
            return full_response_text
        else:
            print(f"Failed to get response: {response.status_code}")
            return None

    def text_to_speech(self, text):
        # Convert text to speech
        inputs = self.processor(text=text, return_tensors="pt")
        generated_speech = self.tts_model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        return generated_speech.numpy()

    def play_audio(self, audio_data):
        # Save as WAV file
        sf.write("response.wav", audio_data, samplerate=16000)
        # Play the WAV file
        wave_obj = sa.WaveObject.from_wave_file("response.wav")
        self.play_obj = wave_obj.play()

    def is_audio_playing(self):
        # Check if the audio is still playing
        if self.play_obj is not None:
            return self.play_obj.is_playing()
        return False

    def listen_and_transcribe(self):
        print("Listening...")
        try:
            while True:
                # Check if audio is playing
                if not self.is_audio_playing():
                    # Read audio stream
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)

                    # Check volume to detect speech
                    volume = audioop.rms(data, 2)
                    if volume > self.silence_threshold:
                        if not self.is_speaking:
                            print("Speech detected, recording...")
                        self.is_speaking = True
                        self.last_speech_time = time.time()
                        self.audio_buffer = np.append(self.audio_buffer, audio_data)
                    else:
                        if self.is_speaking and (time.time() - self.last_speech_time > self.speech_timeout):
                            print("Silence detected, transcribing...")
                            # Transcribe speech
                            audio_float = self.audio_buffer.astype(np.float32)
                            audio_float /= np.iinfo(np.int16).max
                            result = self.model.transcribe(audio_float)
                            print("Transcribed text:", result["text"])
                            # Send transcribed text to API
                            full_response = self.send_prompt_to_api(result["text"])
                            if full_response is not None:
                                print("Received response:", full_response)
                                # Convert text response to speech and play it
                                speech_audio = self.text_to_speech(full_response)
                                self.play_audio(speech_audio)
                            # Reset buffer and control variables
                            self.audio_buffer = np.array([], dtype=np.int16)
                            self.is_speaking = False
                # Sleep a bit to prevent high CPU usage
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            # Stop and close the stream and terminate PyAudio
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

if __name__ == "__main__":
    transcriber = LiveTranscriber()
    transcriber.listen_and_transcribe()
