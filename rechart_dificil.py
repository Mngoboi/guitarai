# -*- coding: utf-8 -*-
"""Regenera SOLO notes.chart al DOBLE de dificultad:
- mas densidad de notas (grilla mas fina)
- acordes (2 notas simultaneas) en los golpes fuertes
- mas saltos de traste
No toca song.mp3 / album.png / song.ogg."""
import os, numpy as np, librosa

AUDIO = "cancion.mp3"
RES = 192
OUT_DIR = "Salida CloneHero/Mi Cancion"
SONG_NAME = "Mi Cancion"; ARTIST = "Desconocido"; CHARTER = "GuitarAI (auto)"

print("==> Analizando audio...")
y, sr = librosa.load(AUDIO)
dur = librosa.get_duration(y=y, sr=sr)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
bpm = float(np.atleast_1d(tempo)[0])
beat_times = librosa.frames_to_time(beats, sr=sr)
phase = float(beat_times[0]) if len(beat_times) else 0.0
beat_dur = 60.0 / bpm
sixteenth = beat_dur / 4.0
thirtytwo = beat_dur / 8.0

oenv = librosa.onset.onset_strength(y=y, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, backtrack=False)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
onset_strength = oenv[onset_frames]
cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
logc = np.log2(cent[np.clip(onset_frames, 0, len(cent)-1)] + 1e-9)
print(f"    BPM={bpm:.2f}  onsets={len(onset_times)}")

def snap(t, grid):
    k = round((t - phase) / grid); return phase + k * grid

def frets_from_pitch(values, n):
    qs = np.quantile(values, np.linspace(0,1,n+1)); qs[0]-=1e-6; qs[-1]+=1e-6
    return np.clip(np.digitize(values, qs)-1, 0, n-1)

def build(min_gap_beats, n_frets, grid, chord_pct, anti_repeat=True):
    """chord_pct: % de notas (las mas fuertes) que llevan acorde de 2 notas."""
    min_gap = min_gap_beats * beat_dur
    snapped = np.array([snap(t, grid) for t in onset_times])
    frets = frets_from_pitch(logc, n_frets)
    kept = []; last_t = -1e9
    for i in range(len(snapped)):
        t = snapped[i]
        if t - last_t < min_gap - 1e-4:
            if kept and onset_strength[i] > onset_strength[kept[-1][2]]:
                kept[-1] = (t, int(frets[i]), i); last_t = t
            continue
        kept.append((t, int(frets[i]), i)); last_t = t
    # anti machine-gun
    if anti_repeat:
        run = 0
        for j in range(1, len(kept)):
            if kept[j][1] == kept[j-1][1]:
                run += 1
                if run >= 3:
                    kept[j] = (kept[j][0], (kept[j][1]+1) % n_frets, kept[j][2]); run = 0
            else: run = 0
    # umbral de fuerza para acordes
    if kept and chord_pct > 0:
        strs = np.array([onset_strength[k[2]] for k in kept])
        thr = np.quantile(strs, 1 - chord_pct)
    else:
        thr = 1e18
    # ticks + sustains + acordes
    notes = []  # (tick, [frets], sustain)
    for j,(t,fr,idx) in enumerate(kept):
        tick = int(round(t*(bpm/60.0)*RES))
        sus = 0
        if j+1 < len(kept):
            gap = kept[j+1][0]-t
            if gap > 1.5*beat_dur: sus = int(round((gap*0.6)*(bpm/60.0)*RES))
        flist = [fr]
        if onset_strength[idx] >= thr:           # acorde
            alt = fr+2 if fr+2 < n_frets else fr-2
            if 0 <= alt < n_frets and alt != fr: flist.append(alt)
        notes.append((tick, flist, sus))
    # dedup por tick
    seen=set(); clean=[]
    for tk,fl,sus in notes:
        if tk in seen: continue
        seen.add(tk); clean.append((tk,fl,sus))
    return clean

# DOBLE de dificultad: grilla mas fina, menos separacion, acordes
diffs = {
    "ExpertSingle": build(0.125, 5, thirtytwo, 0.45),  # muy denso + 45% acordes
    "HardSingle":   build(0.25, 5, sixteenth, 0.30),   # antes 0.5 sin acordes -> ahora 0.25 + 30% acordes
    "MediumSingle": build(0.5, 5, sixteenth, 0.15),    # antes 1.0/4 frets -> ahora 0.5/5 frets + acordes
    "EasySingle":   build(1.0, 4, sixteenth, 0.0),     # antes 2.0/3 -> ahora 1.0/4
}
for k,v in diffs.items():
    nn = sum(len(f) for _,f,_ in v)
    print(f"    {k}: {len(v)} posiciones, {nn} gemas (notas)")

# secciones
events=[]; tpb=RES*4; total=int(round(dur*(bpm/60.0)*RES)); n=1
for tk in range(0,total,tpb*8): events.append((tk,f'section Parte {n}')); n+=1

path=os.path.join(OUT_DIR,"notes.chart")
diff_tier=5
print(f"==> Escribiendo {path}")
with open(path,"w",encoding="utf-8") as f:
    f.write("[Song]\n{\n")
    f.write(f'  Name = "{SONG_NAME}"\n  Artist = "{ARTIST}"\n  Charter = "{CHARTER}"\n')
    f.write(f'  Offset = 0\n  Resolution = {RES}\n  Player2 = bass\n  Difficulty = {diff_tier}\n')
    f.write(f'  PreviewStart = 0\n  PreviewEnd = 0\n  Genre = "rock"\n  MediaType = "cd"\n  MusicStream = "song.mp3"\n}}\n')
    f.write("[SyncTrack]\n{\n  0 = TS 4\n  0 = B %d\n}\n" % int(round(bpm*1000)))
    f.write("[Events]\n{\n")
    for tk,ev in events: f.write(f'  {tk} = E "{ev}"\n')
    f.write("}\n")
    for track in ["ExpertSingle","HardSingle","MediumSingle","EasySingle"]:
        f.write(f"[{track}]\n"+"{\n")
        for tk,fl,sus in diffs[track]:
            for fr in fl:
                f.write(f"  {tk} = N {fr} {sus}\n")
        f.write("}\n")

# subir tier en song.ini
ini=os.path.join(OUT_DIR,"song.ini")
lines=open(ini,encoding="utf-8").read().splitlines()
out=[]
for ln in lines:
    if ln.startswith("diff_guitar"): out.append("diff_guitar = 5")
    else: out.append(ln)
open(ini,"w",encoding="utf-8").write("\n".join(out)+"\n")
print("==> song.ini diff_guitar = 5 (mas dificil)")
print("LISTO")
