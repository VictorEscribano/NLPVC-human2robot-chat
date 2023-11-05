import os
import time
#os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

from bark import SAMPLE_RATE, generate_audio, preload_models
from IPython.display import Audio
import soundfile as sf
import simpleaudio as sa

# download and load all models
preload_models()

SPEAKER = "v2/es_speaker_0"

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
speech_array = generate_audio(text_prompt, history_prompt=SPEAKER)

#save audio to file
sf.write("response.wav", speech_array, samplerate=SAMPLE_RATE)
#play audio from file
time.sleep(2)
wave_obj = sa.WaveObject.from_wave_file("response.wav")
play_obj = wave_obj.play()


