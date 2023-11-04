import pyaudio
import numpy as np
import audioop
import time
from whisper_transcription import transcribe_speech
from text_to_speech import text_to_speech, play_audio

class LiveTranscriber:
    def __init__(self):
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
        self.play_obj = None

    def is_audio_playing(self):
        return play_audio.is_playing(self.play_obj)

    def listen_and_transcribe(self):
        print("Listening...")
        try:
            while True:
                if not self.is_audio_playing():
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume = audioop.rms(data, 2)
                    if volume > self.silence_threshold:
                        self.is_speaking = True
                        self.last_speech_time = time.time()
                        self.audio_buffer = np.append(self.audio_buffer, audio_data)
                    elif self.is_speaking and (time.time() - self.last_speech_time > self.speech_timeout):
                        self.is_speaking = False
                        result = transcribe_speech(self.audio_buffer)
                        self.audio_buffer = np.array([], dtype=np.int16)
                        if result:
                            speech_audio = text_to_speech(result)
                            self.play_obj = play_audio(speech_audio)
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
