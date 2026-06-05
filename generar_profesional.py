# -*- coding: utf-8 -*-
"""
Generador PROFESIONAL de charts para Clone Hero.
- Detecta onsets y tempo real
- Asigna trastes (0-4) segun la altura tonal (centro espectral) -> melodia, NO todo verde
- Cuantiza a grilla de semicorcheas (alineado y limpio)
- Genera 4 dificultades: Easy / Medium / Hard / Expert
- Escribe notes.chart valido, song.ini, convierte a song.ogg y crea album.png
"""
import os, struct, zlib
import numpy as np
import librosa
import soundfile as sf

AUDIO = "cancion.mp3"
RES = 192                      # resolution estandar Clone Hero
OUT_DIR = "Salida CloneHero/Mi Cancion"
SONG_NAME = "Mi Cancion"
ARTIST = "Desconocido"
CHARTER = "GuitarAI (auto)"

print("==> Cargando audio...")
y, sr = librosa.load(AUDIO)
dur = librosa.get_duration(y=y, sr=sr)

print("==> Detectando tempo y beats...")
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
bpm = float(np.atleast_1d(tempo)[0])
beat_times = librosa.frames_to_time(beats, sr=sr)
phase = float(beat_times[0]) if len(beat_times) else 0.0   # fase del primer beat
beat_dur = 60.0 / bpm
sixteenth = beat_dur / 4.0                                  # semicorchea

print(f"    BPM={bpm:.2f}  duracion={dur:.1f}s  beats={len(beat_times)}")

print("==> Detectando onsets (golpes)...")
oenv = librosa.onset.onset_strength(y=y, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, backtrack=False)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
onset_strength = oenv[onset_frames]

# --- Altura tonal por onset: centro espectral (Hz) -> mas agudo = traste mas alto
print("==> Analizando altura tonal para asignar trastes...")
cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
cent_at = cent[np.clip(onset_frames, 0, len(cent) - 1)]
logc = np.log2(cent_at + 1e-9)

# Cuantiza a grilla de semicorcheas (fase del primer beat)
def snap(t):
    k = round((t - phase) / sixteenth)
    return phase + k * sixteenth

snapped = np.array([snap(t) for t in onset_times])

# Asignacion de trastes por cuantiles -> distribucion balanceada verde..naranja
def frets_from_pitch(values, n_frets):
    qs = np.quantile(values, np.linspace(0, 1, n_frets + 1))
    qs[0] -= 1e-6; qs[-1] += 1e-6
    f = np.digitize(values, qs) - 1
    return np.clip(f, 0, n_frets - 1)

def build_difficulty(min_gap_beats, n_frets, anti_repeat=True):
    """Devuelve lista de (tick, fret, sustain_ticks) respetando separacion minima."""
    min_gap = min_gap_beats * beat_dur
    frets = frets_from_pitch(logc, n_frets)
    kept = []           # (time, fret)
    last_t = -1e9
    order_strength = onset_strength
    for i in range(len(snapped)):
        t = snapped[i]
        if t < 0:
            continue
        if t - last_t < min_gap - 1e-4:
            # demasiado cerca: nos quedamos con el golpe mas fuerte
            if kept and order_strength[i] > order_strength[kept[-1][2]]:
                kept[-1] = (t, int(frets[i]), i)
                last_t = t
            continue
        kept.append((t, int(frets[i]), i))
        last_t = t

    # anti machine-gun: evita >3 mismos trastes seguidos
    if anti_repeat:
        run = 0
        for j in range(1, len(kept)):
            if kept[j][1] == kept[j - 1][1]:
                run += 1
                if run >= 3:
                    nf = (kept[j][1] + 1) % n_frets
                    kept[j] = (kept[j][0], nf, kept[j][2])
                    run = 0
            else:
                run = 0

    # ticks + sustains (si el hueco al siguiente es grande)
    notes = []
    for j, (t, fr, _) in enumerate(kept):
        tick = int(round(t * (bpm / 60.0) * RES))
        sus = 0
        if j + 1 < len(kept):
            gap = kept[j + 1][0] - t
            if gap > 1.5 * beat_dur:
                sus = int(round((gap * 0.6) * (bpm / 60.0) * RES))
        notes.append((tick, fr, sus))
    # quita duplicados de tick exacto
    seen = set(); clean = []
    for tk, fr, sus in notes:
        if tk in seen:
            continue
        seen.add(tk); clean.append((tk, fr, sus))
    return clean

# Dificultades: separacion minima (en beats), nro de trastes
diffs = {
    "ExpertSingle": build_difficulty(0.25, 5),   # semicorchea, 5 trastes
    "HardSingle":   build_difficulty(0.5, 5),    # corchea, 5 trastes  <-- MEDIA/DIFICIL
    "MediumSingle": build_difficulty(1.0, 4),    # negra, 4 trastes
    "EasySingle":   build_difficulty(2.0, 3),    # blanca, 3 trastes
}
for k, v in diffs.items():
    print(f"    {k}: {len(v)} notas")

# Secciones cada 8 compases (4/4) para modo practica
print("==> Generando secciones...")
events = []
ticks_per_bar = RES * 4
total_ticks = int(round(dur * (bpm / 60.0) * RES))
n = 1
for tk in range(0, total_ticks, ticks_per_bar * 8):
    events.append((tk, f'section Parte {n}'))
    n += 1

# --- Escribir notes.chart
os.makedirs(OUT_DIR, exist_ok=True)
chart_path = os.path.join(OUT_DIR, "notes.chart")
print(f"==> Escribiendo {chart_path}")
diff_tier = 3  # 0..6 -> tier medio/dificil
with open(chart_path, "w", encoding="utf-8") as f:
    f.write("[Song]\n{\n")
    f.write(f'  Name = "{SONG_NAME}"\n')
    f.write(f'  Artist = "{ARTIST}"\n')
    f.write(f'  Charter = "{CHARTER}"\n')
    f.write(f'  Offset = 0\n')
    f.write(f'  Resolution = {RES}\n')
    f.write(f'  Player2 = bass\n')
    f.write(f'  Difficulty = {diff_tier}\n')
    f.write(f'  PreviewStart = 0\n')
    f.write(f'  PreviewEnd = 0\n')
    f.write(f'  Genre = "rock"\n')
    f.write(f'  MediaType = "cd"\n')
    f.write(f'  MusicStream = "song.ogg"\n')
    f.write("}\n")

    f.write("[SyncTrack]\n{\n")
    f.write("  0 = TS 4\n")
    f.write(f"  0 = B {int(round(bpm * 1000))}\n")
    f.write("}\n")

    f.write("[Events]\n{\n")
    for tk, ev in events:
        f.write(f'  {tk} = E "{ev}"\n')
    f.write("}\n")

    for track in ["ExpertSingle", "HardSingle", "MediumSingle", "EasySingle"]:
        f.write(f"[{track}]\n" + "{\n")
        for tk, fr, sus in diffs[track]:
            f.write(f"  {tk} = N {fr} {sus}\n")
        f.write("}\n")

# --- song.ini
ini_path = os.path.join(OUT_DIR, "song.ini")
print(f"==> Escribiendo {ini_path}")
with open(ini_path, "w", encoding="utf-8") as f:
    f.write("[song]\n")
    f.write(f"name = {SONG_NAME}\n")
    f.write(f"artist = {ARTIST}\n")
    f.write(f"charter = {CHARTER}\n")
    f.write("album = \n")
    f.write("genre = Rock\n")
    f.write("year = 2025\n")
    f.write(f"diff_guitar = {diff_tier}\n")
    f.write("diff_bass = -1\n")
    f.write("diff_drums = -1\n")
    f.write(f"song_length = {int(dur*1000)}\n")
    f.write("preview_start_time = 30000\n")
    f.write("icon = \n")
    f.write("delay = 0\n")
    f.write("loading_phrase = Generado con GuitarAI\n")

# --- song.ogg (convertir mp3 -> ogg con soundfile)
ogg_path = os.path.join(OUT_DIR, "song.ogg")
print(f"==> Convirtiendo audio a {ogg_path}")
y2, sr2 = librosa.load(AUDIO, sr=None, mono=False)
if y2.ndim == 1:
    data = y2
else:
    data = y2.T  # soundfile espera (frames, canales)
sf.write(ogg_path, data, sr2, format="OGG", subtype="VORBIS")

# --- album.png (PNG puro, sin PIL): degradado 512x512
def write_png(path, w, h, rgb_func):
    raw = bytearray()
    for yy in range(h):
        raw.append(0)  # filtro None por fila
        for xx in range(w):
            r, g, b = rgb_func(xx, yy, w, h)
            raw += bytes((r, g, b))
    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 9)
    with open(path, "wb") as f:
        f.write(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))

def grad(x, yy, w, h):
    t = yy / h
    r = int(20 + 60 * t); g = int(20 + 20 * t); b = int(60 + 120 * (1 - t))
    # diagonal acento
    if abs((x / w) - (yy / h)) < 0.06:
        return (240, 200, 60)
    return (r, g, b)

album_path = os.path.join(OUT_DIR, "album.png")
print(f"==> Generando {album_path}")
write_png(album_path, 512, 512, grad)

print("\n==> LISTO. Archivos en:", os.path.abspath(OUT_DIR))
for fn in sorted(os.listdir(OUT_DIR)):
    p = os.path.join(OUT_DIR, fn)
    print(f"    {fn:14s} {os.path.getsize(p):>10,} bytes")
