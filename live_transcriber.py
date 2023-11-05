import pyaudio
import numpy as np
import audioop
import time
import os
from whisper_transcription import transcribe_speech
from text_to_speech import text_to_speech, play_audio, is_playing
from response_generation import generate_response

class LiveTranscriber:
    def __init__(self):
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        # Audio parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.silence_threshold = 1000
        self.speech_timeout = 1.0
        # Initialize audio stream
        self.stream = self.p.open(format=self.format, channels=self.channels,
                                  rate=self.rate, input=True, frames_per_buffer=self.chunk)
        # Initialize audio buffer and control variables
        self.audio_buffer = np.array([], dtype=np.int16)
        self.last_speech_time = None
        self.is_speaking = False
        self.play_obj = None
        #enable microphone
        self.unmute_microphone()

    def mute_microphone(self):
        # Mute the microphone using amixer (Linux)
        os.system("amixer set Capture nocap")

    def unmute_microphone(self):
        # Unmute the microphone using amixer (Linux)
        os.system("amixer set Capture cap")

    def listen_and_transcribe(self):
        print("Listening...")
        try:
            while True:
                if self.play_obj and self.play_obj.is_playing():
                    print("Playing response...")
                    continue

                if not is_playing(self.play_obj):
                    if not self.is_speaking: print("Listening...") 
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume = audioop.rms(data, 2)

                    if volume > self.silence_threshold:
                        if not self.is_speaking: print("Speech detected, recording...")
                        self.is_speaking = True
                        self.last_speech_time = time.time()
                        self.audio_buffer = np.append(self.audio_buffer, audio_data)
                        
                    elif self.is_speaking and (time.time() - self.last_speech_time > self.speech_timeout):
                        print("Silence detected, transcribing...")
                        self.is_speaking = False
                        result = transcribe_speech(self.audio_buffer)
                        self.audio_buffer = np.array([], dtype=np.int16)
                        if result:
                            response_text = generate_response(result)
                            if response_text:
                                print("Received response:", response_text)
                                self.mute_microphone()  # Mute the microphone before playing the response
                                speech_audio = text_to_speech(response_text)
                                self.play_obj = play_audio(speech_audio)
                                duration = (len(speech_audio) / 16000) + 2
                                print("Duration:", duration)
                                time.sleep(duration)
                                self.unmute_microphone()  # Unmute the microphone after playing the response
                                #clean audio buffer
                                self.audio_buffer = np.array([], dtype=np.int16)

                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
