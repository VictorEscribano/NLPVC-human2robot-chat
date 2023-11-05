import whisper
import numpy as np

def transcribe_speech(audio_buffer):
    model = whisper.load_model("base")
    audio_float = audio_buffer.astype(np.float32)
    audio_float /= np.iinfo(np.int16).max
    result = model.transcribe(audio_float)
    print("Transcribed text:", result["text"])
    return result["text"]
