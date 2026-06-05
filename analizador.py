import librosa
import numpy as np

archivo = "cancion.mp3"
y, sr = librosa.load(archivo)
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
tiempos = librosa.frames_to_time(onsets, sr=sr)

print(f"Detectadas {len(tiempos)} notas.")
print(tiempos[:10])
