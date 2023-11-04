import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
import simpleaudio as sa
from datasets import load_dataset

# Load TTS models and speaker embeddings
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def text_to_speech(text):
    inputs = processor(text=text, return_tensors="pt")
    generated_speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    return generated_speech.numpy()

def play_audio(audio_data):
    sf.write("response.wav", audio_data, samplerate=16000)
    wave_obj = sa.WaveObject.from_wave_file("response.wav")
    play_obj = wave_obj.play()
    return play_obj

def is_playing(play_obj):
    if play_obj is not None:
        return play_obj.is_playing()
    return False
