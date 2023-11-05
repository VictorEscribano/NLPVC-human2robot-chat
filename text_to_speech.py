import os
#os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

import time
import soundfile as sf
import simpleaudio as sa
from bark import generate_audio, preload_models, SAMPLE_RATE

# Set environment variable for using small models if desired
# os.environ["SUNO_USE_SMALL_MODELS"] = "True"

# Preload models for faster audio generation
preload_models()

# Define the speaker model to use
SPEAKER = "v2/es_speaker_0"

def text_to_speech(text):
    # Generate audio from text using the specified speaker model
    speech_array = generate_audio(text, history_prompt=SPEAKER)
    return speech_array

def play_audio(audio_data):
    # Save the audio data to a file
    sf.write("response.wav", audio_data, samplerate=SAMPLE_RATE)
    # Play the audio file after a short delay
    time.sleep(1)
    wave_obj = sa.WaveObject.from_wave_file("response.wav")
    play_obj = wave_obj.play()
    return play_obj

def is_playing(play_obj):
    if play_obj is not None:
        return play_obj.is_playing()
    return False
