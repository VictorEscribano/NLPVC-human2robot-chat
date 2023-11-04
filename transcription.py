import pyaudio
import whisper
import numpy as np
import audioop
import time


# Carga el modelo de Whisper
model = whisper.load_model("base")

# Inicializa PyAudio
p = pyaudio.PyAudio()

# Define los parámetros de audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500  # Nivel de volumen para considerar como silencio
SPEECH_TIMEOUT = 1.0  # Tiempo en segundos para esperar después de la última palabra antes de transcribir

# Inicializa el stream de audio
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Inicializa el buffer de audio y las variables de control
audio_buffer = np.array([], dtype=np.int16)
last_speech_time = None
is_speaking = False

print("Listening...")

try:
    while True:
        # Lee datos del micrófono
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Convierte los datos de audio a flotantes
        audio_float = audio_data.astype(np.float32)
        # Normaliza los datos de audio para que estén en el rango de -1.0 a 1.0
        audio_float /= np.iinfo(np.int16).max

        # Detecta si hay voz en el audio actual
        volume = audioop.rms(data, 2)  # Obtén el volumen del audio
        if volume > SILENCE_THRESHOLD:
            if not is_speaking:
                print("Speech detected, recording...")
            is_speaking = True
            last_speech_time = time.time()
            audio_buffer = np.append(audio_buffer, audio_float)
        else:
            if is_speaking and (time.time() - last_speech_time > SPEECH_TIMEOUT):
                # Transcribe el audio si se detecta silencio después de hablar
                print("Silence detected, transcribing...")
                result = model.transcribe(audio_buffer)
                print(result["text"])
                # Limpia el buffer y actualiza las variables de control
                audio_buffer = np.array([], dtype=np.float32)
                is_speaking = False

except KeyboardInterrupt:
    # Detiene y cierra el stream
    print("Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()
