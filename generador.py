import librosa
import numpy as np

# El nombre del archivo que quieres convertir
archivo = "cancion.mp3"

print("Analizando audio, espera un momento...")
y, sr = librosa.load(archivo)
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
times = librosa.frames_to_time(onsets, sr=sr)

print("Creando archivo de notas...")
bpm = 120
ticks_per_second = (bpm / 60) * 192

with open("cancion.chart", "w") as f:
    f.write("[Song]\n{\n  Name = \"Mi Cancion\"\n}\n")
    f.write("[SyncTrack]\n{\n  0 = B 120000\n}\n")
    f.write("[ExpertSingle]\n{\n")
    for t in times:
        tick = int(t * ticks_per_second)
        f.write(f"  {tick} = N 0 0\n")
    f.write("}")

print("¡LISTO! Ya tienes un archivo llamado 'cancion.chart' en la carpeta.")
