# -*- coding: utf-8 -*-
"""
motor.py — Pipeline completo: MP3 + nombre + artista + foto -> carpeta Clone Hero.
Genera: notes.chart (dinamico voz/instrumental, acordes/sustains/StarPower),
        song.ini, song.mp3, album.png, video.mp4 (late con el ritmo), letra karaoke.
Uso:  from motor import generar_cancion
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # evita "OMP Error #15" (choque OpenMP torch/librosa)
import re, json, shutil, subprocess, tempfile, bisect
import numpy as np
import librosa

RES = 192

# ---------------- utilidades ----------------
def _noop(msg, pct):  # callback de progreso por defecto
    print(f"[{pct:3d}%] {msg}")

def _slug(s):
    s = re.sub(r'[^\w\- ]', '', s).strip().replace(' ', '_')
    return s or "cancion"

# ---------------- separacion de instrumentos (demucs) ----------------
def separar_voz(mp3_path, tmp, progress):
    """Separa en stems. Intenta 6 fuentes (voz/batería/bajo/GUITARRA/piano/otros)
    para tener la guitarra aislada de verdad; si falla, cae a 4 stems (other=melodía)."""
    progress("Separando instrumentos (voz/guitarra/bajo/batería)...", 10)
    py = os.path.join(os.path.dirname(__file__), "venv", "bin", "python")
    if not os.path.exists(py):
        py = "python"
    base = os.path.splitext(os.path.basename(mp3_path))[0]
    # --- intento 6 stems (htdemucs_6s) ---
    try:
        cmd = [py, "-m", "demucs", "-n", "htdemucs_6s", "-o", tmp, mp3_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        d = os.path.join(tmp, "htdemucs_6s", base)
        stems = {k: os.path.join(d, k + ".wav") for k in ("vocals", "drums", "bass", "guitar", "piano", "other")}
        if all(os.path.exists(stems[k]) for k in ("vocals", "drums", "bass", "guitar")):
            return stems
    except Exception as e:
        print("htdemucs_6s falló, uso 4 stems:", e)
    # --- respaldo 4 stems (vocals/drums/bass/other) ---
    cmd = [py, "-m", "demucs", "-o", tmp, mp3_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = os.path.join(tmp, "htdemucs", base)
    return {k: os.path.join(d, k + ".wav") for k in ("vocals", "drums", "bass", "other")}

# ---------------- construccion del chart ----------------
def _frets_from(values, n):
    qs = np.quantile(values, np.linspace(0, 1, n + 1)); qs[0]-=1e-6; qs[-1]+=1e-6
    return np.clip(np.digitize(values, qs) - 1, 0, n - 1)

def _chord_frets(base, size, n=5):
    if size <= 1: return [base]
    lo = max(0, min(base - (size // 2), n - size)); return list(range(lo, lo + size))

def _repetir_coros(master, yi, sr, bt, beat_dur):
    """Donde se repiten los ACORDES (coros/secciones), copia el MISMO patrón de notas.
    Detecta por similitud de chroma (acordes) en ventanas de ~2 compases."""
    try:
        if len(bt) < 24 or len(master) < 20: return master
        chroma = librosa.feature.chroma_cqt(y=yi, sr=sr)
        bf = np.clip(librosa.time_to_frames(bt, sr=sr), 0, chroma.shape[1]-1)
        cb = librosa.util.sync(chroma, bf, aggregate=np.median)        # 12 x nbeats
        cb = cb/(np.linalg.norm(cb, axis=0, keepdims=True)+1e-9)
        n = cb.shape[1]; W = 8
        def sim(i,j):
            A=cb[:,i:i+W]; B=cb[:,j:j+W]
            if A.shape[1]!=W or B.shape[1]!=W: return 0.0
            return float(np.mean(np.sum(A*B,axis=0)))
        out=list(master); i=W
        while i+W<=n:
            best=-1; bestsim=0.0
            for j in range(0,i-W+1):
                sm=sim(i,j)
                if sm>bestsim: bestsim=sm; best=j
            if best>=0 and bestsim>0.92:                               # acordes casi iguales -> copiar patrón
                src_t0=float(bt[best]); dst_t0=float(bt[i]); dur=W*beat_dur; shift=dst_t0-src_t0
                srcs=[m for m in master if src_t0-1e-3 <= m[0] < src_t0+dur]
                if srcs:
                    out=[m for m in out if not (dst_t0-1e-3 <= m[0] < dst_t0+dur)]
                    for (t,p,s,k) in srcs: out.append((t+shift,p,s,k))
            i+=W
        out.sort(key=lambda m:m[0])
        return out
    except Exception as e:
        print("repetir coros: sin cambios (", e, ")"); return master

def _snap(t, grid, phase):
    k = round((t - phase) / grid); return phase + k * grid

def _starpower(notes, every_beats=16, len_beats=4):
    if not notes: return []
    sp=[]; step=int(every_beats*RES); length=int(len_beats*RES)
    ticks=[n[0] for n in notes]; cur=ticks[0]; end=ticks[-1]
    while cur<end:
        i=bisect.bisect_left(ticks,cur)
        if i>=len(ticks): break
        if not sp or sp[-1][0]!=ticks[i]: sp.append((ticks[i],length))
        cur=max(cur+step, ticks[i]+step)   # en huecos largos salta a la nota siguiente (sin frases duplicadas)
    return sp

def _build_notes(times, base_frets, strg, src, phase, bpm, beat_dur,
                 min_gap_beats, n_frets, grid, sus_gap_beats,
                 pct_triple, pct_double, max_chord, has_guitar=True, allow_sustain=True):
    """Constructor genérico de notas melódicas (guitarra/bajo/voz). En grilla,
    1 nota por slot (la más fuerte), acordes solo en los golpes más fuertes."""
    if not has_guitar: pct_triple*=0.4
    min_gap=min_gap_beats*beat_dur
    snp=np.array([_snap(t,grid,phase) for t in times]); kept=[]; last=-1e9
    for i in range(len(snp)):
        t=snp[i]
        if t-last<min_gap-1e-4:
            if kept and strg[i]>strg[kept[-1][2]]: kept[-1]=(t,int(round(base_frets[i]*((n_frets-1)/4))),i); last=t
            continue
        kept.append((t,int(round(base_frets[i]*((n_frets-1)/4))),i)); last=t
    run=0
    for j in range(1,len(kept)):
        if kept[j][1]==kept[j-1][1]:
            run+=1
            if run>=3: kept[j]=(kept[j][0],(kept[j][1]+1)%n_frets,kept[j][2]); run=0
        else: run=0
    thr3=thr2=1e18
    if kept:
        s=np.array([strg[k[2]] for k in kept])
        thr3=np.quantile(s,1-pct_triple) if pct_triple>0 else 1e18
        thr2=np.quantile(s,1-pct_triple-pct_double) if (pct_triple+pct_double)>0 else 1e18
    notes=[]
    for j,(t,f,idx) in enumerate(kept):
        tick=int(round(t*(bpm/60.0)*RES)); sus=0
        if allow_sustain and j+1<len(kept):
            gap=kept[j+1][0]-t
            if gap>sus_gap_beats*beat_dur: sus=int(round((gap*0.55)*(bpm/60.0)*RES))
        if src[idx]=='V':
            size = 2 if (max_chord>=2 and strg[idx]>=thr3) else 1   # VOZ: 1 o 2, nunca 3
        else:
            if max_chord>=3 and strg[idx]>=thr3: size=3            # rasgueos/quintas
            elif max_chord>=2 and strg[idx]>=thr2: size=2
            else: size=1
        notes.append((tick,_chord_frets(f,min(size,max_chord),n_frets),sus))
    seen=set(); clean=[]
    for tk,fl,sus in notes:
        if tk in seen: continue
        seen.add(tk); clean.append((tk,fl,sus))
    return clean

def construir_bateria(yd, sr, bpm, phase, beat_dur, grids):
    """Pista de BATERÍA real: detecta cada golpe del stem de batería y lo manda a su carril
    (0=bombo, 1=rojo/redoblante, 2=amarillo/hi-hat, 3=azul/tom, 4=verde/crash)."""
    g16,g8,g4 = grids
    oe = librosa.onset.onset_strength(y=yd, sr=sr)
    f = librosa.onset.onset_detect(onset_envelope=oe, sr=sr, backtrack=False)
    if not len(f): return {}
    t = librosa.frames_to_time(f, sr=sr); strg = oe[f]
    S = np.abs(librosa.stft(yd, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low=freqs<150; lomid=(freqs>=150)&(freqs<400); mid=(freqs>=400)&(freqs<2000); high=freqs>=5000
    cols=np.clip(librosa.time_to_frames(t, sr=sr, hop_length=512),0,S.shape[1]-1)
    feats=[]
    for tt,c,st in zip(t,cols,strg):
        col=S[:,c]
        el=float(col[low].sum()); elm=float(col[lomid].sum()); em=float(col[mid].sum()); eh=float(col[high].sum())
        feats.append((tt,float(st),el,elm,em,eh))
    ehs=[f[5] for f in feats]
    crash_thr=float(np.quantile(ehs,0.92)) if ehs else 1e18   # platillazos top 8% -> crash
    hits=[]
    for tt,st,el,elm,em,eh in feats:
        tot=el+elm+em+eh+1e-9
        lanes=[]
        if el/tot>0.40: lanes.append(0)                  # bombo
        if eh/tot>0.30:
            lanes.append(4 if eh>=crash_thr else 2)      # crash fuerte -> verde; hi-hat -> amarillo
        elif elm/tot>0.28: lanes.append(1)               # redoblante -> rojo
        elif em/tot>0.30: lanes.append(3)                # tom medio -> azul
        if not lanes: lanes.append(1)
        hits.append((tt,st,lanes))
    def build_d(grid, min_gap_beats, keep_frac):
        if not hits: return []
        thr=np.quantile([h[1] for h in hits], 1-keep_frac) if keep_frac<1 else -1e18
        out=[]; last=-1e9; mg=min_gap_beats*beat_dur
        for tt,st,lanes in hits:
            if st<thr: continue
            ts=_snap(tt,grid,phase)
            if ts-last<mg-1e-4: continue
            last=ts
            out.append((int(round(ts*(bpm/60.0)*RES)), list(lanes), 0))
        seen=set(); cl=[]
        for tk,fl,su in out:
            if tk in seen: continue
            seen.add(tk); cl.append((tk,fl,su))
        return cl
    return {
        "ExpertDrums": build_d(g16,0.25,1.0),
        "HardDrums":   build_d(g8,0.5,0.80),
        "MediumDrums": build_d(g8,1.0,0.55),
        "EasyDrums":   build_d(g4,2.0,0.40),
    }

def construir_bajo(yb, sr, bpm, phase, beat_dur, grids):
    """Pista de BAJO: sigue el tono real del bajo (pyin grave), notas simples (sin acordes)."""
    g16,g8,g4 = grids
    oe = librosa.onset.onset_strength(y=yb, sr=sr)
    f = librosa.onset.onset_detect(onset_envelope=oe, sr=sr, backtrack=False)
    if not len(f): return {}
    t = librosa.frames_to_time(f, sr=sr); s = oe[f]
    f0,_,_ = librosa.pyin(yb, fmin=35, fmax=400, sr=sr, frame_length=2048, hop_length=512)
    times=[]; pit=[]; strg=[]
    for tt,st in zip(t,s):
        k=int(np.clip(tt/(512/sr),0,len(f0)-1)); lo=max(0,k-2); hi=min(len(f0),k+3)
        seg=f0[lo:hi]; seg=seg[~np.isnan(seg)]
        if not len(seg): continue
        times.append(tt); pit.append(np.log2(np.median(seg))); strg.append(st)
    if len(times)<4: return {}
    times=np.array(times); strg=np.array(strg); base=_frets_from(np.array(pit),5); src=['I']*len(times)
    def mk(min_gap,grid,sus_gap):
        return _build_notes(times,base,strg,src,phase,bpm,beat_dur,min_gap,5,grid,sus_gap,0.0,0.0,1,
                            has_guitar=True, allow_sustain=True)
    return {
        "ExpertDoubleBass": mk(0.5,g8,1.0),
        "HardDoubleBass":   mk(0.75,g8,1.2),
        "MediumDoubleBass": mk(1.0,g4,1.4),
        "EasyDoubleBass":   mk(2.0,g4,1.6),
    }

def construir_chart(stems, progress, pistas=None):
    """pistas = qué instrumentos chartear (casillas del dashboard).
    Por defecto SOLO 'guitarra' (o melodía/voz si la canción no tiene guitarra).
    Opciones: 'guitarra', 'voz', 'bajo', 'bateria'."""
    pistas = set(pistas) if pistas else {"guitarra"}
    if not pistas: pistas = {"guitarra"}
    progress("Analizando ritmo y melodía...", 35)
    yv, sr = librosa.load(stems["vocals"])          # VOZ
    yd, _  = librosa.load(stems["drums"], sr=sr)    # BATERÍA
    yb, _  = librosa.load(stems["bass"], sr=sr)     # BAJO
    # fuente de la GUITARRA: stem 'guitar' (6s) si existe; si no, 'other' (4s = melodía)
    gpath = stems.get("guitar")
    if not (gpath and os.path.exists(gpath)): gpath = stems.get("other")
    yo, _  = librosa.load(gpath, sr=sr) if (gpath and os.path.exists(gpath)) else (np.zeros_like(yv), sr)
    progress("Limpiando melodía (sin batería)...", 38)
    yi = librosa.effects.harmonic(yo, margin=4.5)   # guitarra armónica limpia (sin bleed percusivo)
    dur = librosa.get_duration(y=yv, sr=sr)
    # tempo desde la BATERÍA (lo más fiable para el beat); respaldo melodía+bajo
    try:
        tempo, beats = librosa.beat.beat_track(y=yd, sr=sr)
        if not len(np.atleast_1d(beats)): raise ValueError
    except Exception:
        tempo, beats = librosa.beat.beat_track(y=(yo+yb), sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(bpm) or bpm<=0: bpm=120.0   # canción sin pulso detectable: tempo neutro
    bt = librosa.frames_to_time(beats, sr=sr); phase = float(bt[0]) if len(bt) else 0.0
    beat_dur = 60.0 / bpm; g16 = beat_dur/4; g8 = beat_dur/2; g4 = beat_dur
    grids=(g16,g8,g4)

    # actividad vocal (para secciones y para el respaldo cuando NO hay guitarra)
    hop = 512
    rms = librosa.feature.rms(y=yv, hop_length=hop)[0]
    rt = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    thr = np.percentile(rms, 60)*0.8 + rms.max()*0.04
    active = rms > thr
    def smooth(mask, times, min_on=0.4, min_off=0.3):
        segs=[]; i=0; n=len(mask)
        while i<n:
            if mask[i]:
                j=i
                while j<n and mask[j]: j+=1
                segs.append([times[i], times[min(j,n-1)]]); i=j
            else: i+=1
        merged=[]
        for s in segs:
            if merged and s[0]-merged[-1][1]<min_off: merged[-1][1]=s[1]
            else: merged.append(s)
        return [s for s in merged if s[1]-s[0]>=min_on]
    voice_segs = smooth(active, rt)
    def in_voice(t):
        for a,b in voice_segs:
            if a<=t<=b: return True
        return False

    # ¿hay guitarra? (energía de la guitarra limpia)
    other_rms=float(np.sqrt(np.mean(yi**2)))
    has_guitar = other_rms > 0.009

    def _onsets_guitarra():
        # GUITARRA: onsets de la guitarra; sesgo leve a lo más BRILLANTE = línea de solo (2ª guitarra/lead)
        oe_i = librosa.onset.onset_strength(y=yi, sr=sr)
        fi = librosa.onset.onset_detect(onset_envelope=oe_i, sr=sr, backtrack=False)
        ti = librosa.frames_to_time(fi, sr=sr); si = oe_i[fi]
        cent = librosa.feature.spectral_centroid(y=yi, sr=sr)[0]
        cmed = float(np.median(cent[cent>0])) if np.any(cent>0) else 1.0
        out=[]
        for t,s in zip(ti,si):
            k=int(np.clip(t/(512/sr),0,len(cent)-1)); c=cent[k]+1e-9
            bias=1.0+0.5*np.clip((c-cmed)/(cmed+1e-9),0,1)   # más brillo -> más peso (lead/solo)
            out.append((t, np.log2(c), float(s)*bias, 'I'))
        return out
    def _onsets_voz():
        # VOZ: melodía del cantante (solo donde canta)
        oe_v = librosa.onset.onset_strength(y=yv, sr=sr)
        fv = librosa.onset.onset_detect(onset_envelope=oe_v, sr=sr, backtrack=False)
        tv = librosa.frames_to_time(fv, sr=sr); sv = oe_v[fv]
        f0v,_,_ = librosa.pyin(yv, fmin=80, fmax=1000, sr=sr, frame_length=2048, hop_length=hop)
        out=[]
        for t,s in zip(tv,sv):
            if not in_voice(t): continue
            k=int(np.clip(t/(hop/sr),0,len(f0v)-1)); lo=max(0,k-3); hi=min(len(f0v),k+4)
            seg=f0v[lo:hi]; seg=seg[~np.isnan(seg)]
            if not len(seg): continue
            out.append((t, np.log2(np.median(seg)), float(s), 'V'))
        return out

    # === qué entra a la pista de guitarra, según las CASILLAS ===
    quiere_guit = "guitarra" in pistas
    quiere_voz  = "voz" in pistas
    master=[]; repsrc=yi
    if quiere_guit and has_guitar:
        progress("Extrayendo melodía de la guitarra...", 45)
        master += _onsets_guitarra(); repsrc = yi
        if quiere_voz:                      # voz AGREGADA a la guitarra (casilla extra)
            master += _onsets_voz()
    elif quiere_guit and not has_guitar:
        # en su defecto MELODÍA: sin guitarra, guiarse por la voz del cantante
        progress("Sin guitarra: siguiendo la voz...", 45)
        master += _onsets_voz(); repsrc = yv
    elif quiere_voz:                        # solo voz pedida explícitamente
        progress("Extrayendo melodía de la voz...", 45)
        master += _onsets_voz(); repsrc = yv
    master.sort(key=lambda m:m[0])

    progress("Detectando coros / patrones repetidos...", 48)
    master=_repetir_coros(master, repsrc, sr, bt, beat_dur)   # repetir patrón en coros/estrofas repetidas
    if master:
        times=np.array([m[0] for m in master]); pitch=np.array([m[1] for m in master])
        strg=np.array([m[2] for m in master]); src=[m[3] for m in master]
        # trastes POR FUENTE: la voz y la guitarra viven en escalas distintas;
        # cada una se cuantila aparte para que ambas usen los 5 trastes completos
        base_frets=np.zeros(len(master),dtype=int)
        for tag in set(src):
            idxs=[j for j,s_ in enumerate(src) if s_==tag]
            base_frets[idxs]=_frets_from(pitch[idxs],5)
        def bg(min_gap,n,grid,sg,pt,pd,mc):
            return _build_notes(times,base_frets,strg,src,phase,bpm,beat_dur,min_gap,n,grid,sg,pt,pd,mc,
                                has_guitar=has_guitar, allow_sustain=True)
        guit={
         "ExpertSingle": bg(0.25,5,g16,0.9,0.17,0.26,3),
         "HardSingle":   bg(0.5,5,g8,1.0,0.14,0.24,3),
         "MediumSingle": bg(1.0,5,g8,1.25,0.0,0.18,2),
         "EasySingle":   bg(2.0,4,g4,1.5,0.0,0.0,1),
        }
    else:
        guit={k:[] for k in ("ExpertSingle","HardSingle","MediumSingle","EasySingle")}

    # === pistas de BATERÍA y BAJO solo si sus casillas están marcadas ===
    drums={}; bass={}
    if "bateria" in pistas:
        progress("Construyendo batería...", 52)
        drums = construir_bateria(yd, sr, bpm, phase, beat_dur, grids)
    if "bajo" in pistas:
        progress("Construyendo bajo...", 55)
        bass  = construir_bajo(yb, sr, bpm, phase, beat_dur, grids)

    tracks=dict(guit)
    tracks.update(drums)
    tracks.update(bass)
    sp={"ExpertSingle":_starpower(guit["ExpertSingle"]),"HardSingle":_starpower(guit["HardSingle"])}
    if bass: sp["ExpertDoubleBass"]=_starpower(bass.get("ExpertDoubleBass",[]))
    if drums:
        sp["ExpertDrums"]=_starpower(drums.get("ExpertDrums",[]))
        sp["HardDrums"]=_starpower(drums.get("HardDrums",[]))

    events=[]
    for a,b in voice_segs:
        events.append((int(round(a*(bpm/60.0)*RES)),'section Voz'))
    for m in [0.0]+[b for a,b in voice_segs]:
        events.append((int(round(m*(bpm/60.0)*RES)),'section Instrumental'))
    events=sorted(set(events))
    return dict(bpm=bpm,dur=dur,tracks=tracks,sp=sp,events=events,voice_segs=voice_segs,
                has_guitar=has_guitar, tiene_guitarra=any(len(v)>0 for v in guit.values()),
                tiene_bateria=bool(drums), tiene_bajo=bool(bass))

# orden canónico de pistas: Guitarra, Bajo, Batería (todas sus dificultades)
_ORDEN_PISTAS=["ExpertSingle","HardSingle","MediumSingle","EasySingle",
               "ExpertDoubleBass","HardDoubleBass","MediumDoubleBass","EasyDoubleBass",
               "ExpertDrums","HardDrums","MediumDrums","EasyDrums"]

def escribir_chart(data, name, artist, out_path):
    bpm=data["bpm"]; tracks=data["tracks"]; sp=data["sp"]; events=data["events"]
    with open(out_path,"w",encoding="utf-8") as f:
        f.write("[Song]\n{\n")
        f.write(f'  Name = "{name}"\n  Artist = "{artist}"\n  Charter = "GuitarAI"\n')
        f.write(f'  Offset = 0\n  Resolution = {RES}\n  Player2 = bass\n  Difficulty = 6\n')
        f.write(f'  PreviewStart = 0\n  PreviewEnd = 0\n  Genre = "rock"\n  MediaType = "cd"\n  MusicStream = "song.mp3"\n}}\n')
        f.write("[SyncTrack]\n{\n  0 = TS 4\n  0 = B %d\n}\n"%int(round(bpm*1000)))
        f.write("[Events]\n{\n")
        for tk,ev in events: f.write(f'  {tk} = E "{ev}"\n')
        f.write("}\n")
        for track in _ORDEN_PISTAS:
            notes=tracks.get(track)
            if not notes: continue                      # solo escribe pistas con contenido
            f.write(f"[{track}]\n"+"{\n")
            rows=[]
            for tk,fl,sus in notes:
                for fr in fl: rows.append((tk,f"N {fr} {sus}"))
            for tk,length in sp.get(track,[]): rows.append((tk,f"S 2 {length}"))
            rows.sort(key=lambda r:(r[0],0 if r[1][0]=='S' else 1))
            for tk,txt in rows: f.write(f"  {tk} = {txt}\n")
            f.write("}\n")

# ---------------- letra (whisper) ----------------
_WHISPER=None
def _get_whisper(size="small"):
    global _WHISPER
    if _WHISPER is None:
        import whisper; _WHISPER=whisper.load_model(size)
    return _WHISPER

def _alinear_aeneas(voc, utok):
    """Alineación FORZADA (aeneas): calza TUS palabras al audio de la voz con precisión.
    Devuelve [(tiempo, palabra, linea)] o None si falla (para usar el respaldo)."""
    try:
        import tempfile, shutil
        os.environ.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        if "/usr/local/bin" not in os.environ.get("PATH",""):
            os.environ["PATH"]="/usr/local/bin:"+os.environ.get("PATH","")
        from aeneas.executetask import ExecuteTask
        from aeneas.task import Task
        tmp=tempfile.mkdtemp()
        txtf=os.path.join(tmp,"words.txt")
        with open(txtf,"w",encoding="utf-8") as f:
            f.write("\n".join((w or "_") for w,_ in utok))
        task=Task(config_string="task_language=spa|os_task_file_format=json|is_text_type=plain")
        task.audio_file_path_absolute=os.path.abspath(voc)
        task.text_file_path_absolute=txtf
        ExecuteTask(task).execute()
        leaves=[l for l in task.sync_map_leaves() if l.text is not None and l.text.strip()]
        shutil.rmtree(tmp, ignore_errors=True)
        if len(leaves)!=len(utok):
            return None
        return [(float(leaves[j].begin), utok[j][0], utok[j][1]) for j in range(len(utok))]
    except Exception as e:
        print("aeneas no disponible/fallo, uso respaldo:", e)
        return None

def agregar_letra(voc, chart_path, bpm, progress, size="small", letra=None):
    def tick(t): return int(round(t*(bpm/60.0)*RES))
    def clean(s): return s.replace('"','').strip()

    if letra and letra.strip():
        lines=[ln.strip() for ln in letra.splitlines() if ln.strip()]
        utok=[(clean(wd),li) for li,ln in enumerate(lines) for wd in ln.split() if clean(wd)]
        progress("Alineando tu letra con la voz (alineador forzado)...", 82)
        timed=_alinear_aeneas(voc, utok)            # ALINEADOR FORZADO (preciso)
        if timed is None:
            # RESPALDO: onsets de la voz por sílabas
            progress("Calzando tu letra con la voz (respaldo)...", 82)
            yv, sr = librosa.load(voc)
            oenv = librosa.onset.onset_strength(y=yv, sr=sr)
            of = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, backtrack=False)
            ot = list(librosa.frames_to_time(of, sr=sr))
            hop=512; rms=librosa.feature.rms(y=yv, hop_length=hop)[0]
            thr=float(np.percentile(rms,55))
            ot=[t for t in ot if rms[int(np.clip(t/(hop/sr),0,len(rms)-1))]>thr]
            if not ot:
                dur=librosa.get_duration(y=yv,sr=sr); ot=[i*dur/40.0 for i in range(40)]
            N=len(ot)
            def syll(w): return max(1, len(re.findall(r'[aeiouáéíóúü]+', w.lower())))
            sy=[syll(w) for w,_ in utok]; Sn=sum(sy) or 1
            cum=[]; c=0
            for s in sy: cum.append(c); c+=s
            timed=[(ot[min(N-1,int(round((cum[j]/Sn)*(N-1))))], utok[j][0], utok[j][1]) for j in range(len(utok))]
        for j in range(1,len(timed)):
            if timed[j][0]<timed[j-1][0]: timed[j]=(timed[j-1][0],timed[j][1],timed[j][2])
        phrases=[]; cur=[]; curline=timed[0][2] if timed else 0
        for s,wd,li in timed:
            if li!=curline and cur: phrases.append(cur); cur=[]; curline=li
            cur.append((s,s,wd))
        if cur: phrases.append(cur)
        nwords=len(utok)
    else:
        # AUTO (sin pegar letra): transcripción con whisper, frases por silencio
        progress("Transcribiendo letra (karaoke)...", 80)
        m=_get_whisper(size)
        r=m.transcribe(voc, language="es", word_timestamps=True, verbose=False)
        words=[]
        for seg in r["segments"]:
            for w in seg.get("words",[]):
                t=clean(w.get("word",""))
                if t: words.append((round(w["start"],3),round(w["end"],3),t))
        if not words: return 0
        phrases=[]; cur=[]
        for w in words:
            if cur and w[0]-cur[-1][1]>1.2: phrases.append(cur); cur=[]
            cur.append(w)
        if cur: phrases.append(cur)
        nwords=len(words)
    txt=open(chart_path,encoding="utf-8").read()
    mm=re.search(r"\[Events\]\s*\{(.*?)\}", txt, re.S)
    existing=[]
    for line in mm.group(1).splitlines():
        g=re.match(r"(\d+)\s*=\s*E\s*\"(.+)\"", line.strip())
        if g and "section" in g.group(2): existing.append((int(g.group(1)),'E "%s"'%g.group(2)))
    lyr=[]
    for ph in phrases:
        lyr.append((tick(ph[0][0]),'E "phrase_start"'))
        for s,e,t in ph: lyr.append((tick(s),'E "lyric %s"'%t))
        lyr.append((tick(ph[-1][1]),'E "phrase_end"'))
    def keyf(e):
        pr=2
        if "phrase_start" in e[1]: pr=0
        elif "section" in e[1]: pr=1
        elif "lyric" in e[1]: pr=3
        elif "phrase_end" in e[1]: pr=4
        return (e[0],pr)
    allev=sorted(existing+lyr,key=keyf)
    block="[Events]\n{\n"+"".join("  %d = %s\n"%(t,s) for t,s in allev)+"}"
    open(chart_path,"w",encoding="utf-8").write(txt[:mm.start()]+block+txt[mm.end():])
    return nwords

# ---------------- caratula + video que late ----------------
def hacer_album(photo_path, out_png):
    from PIL import Image
    if photo_path and os.path.exists(photo_path):
        im=Image.open(photo_path).convert("RGB")
        # recorte cuadrado centrado 512x512
        s=min(im.size); l=(im.width-s)//2; t=(im.height-s)//2
        im=im.crop((l,t,l+s,t+s)).resize((512,512),Image.LANCZOS); im.save(out_png)
    else:
        import struct, zlib
        def fn(x,y,w,h):
            tt=y/h; return (int(20+60*tt),int(20+20*tt),int(60+120*(1-tt)))
        w=h=512; raw=bytearray()
        for y in range(h):
            raw.append(0)
            for x in range(w):
                r,g,b=fn(x,y,w,h); raw+=bytes((r,g,b))
        def ch(ty,d): return struct.pack(">I",len(d))+ty+d+struct.pack(">I",zlib.crc32(ty+d)&0xffffffff)
        open(out_png,"wb").write(b"\x89PNG\r\n\x1a\n"+ch(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+ch(b"IDAT",zlib.compress(bytes(raw),9))+ch(b"IEND",b""))

def _ffmpeg_bin():
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

def _reencode_ch(src, out_mp4):
    """Reencoda cualquier video a H.264 yuv420p (lo que Clone Hero sí carga). Sin audio."""
    cmd=[_ffmpeg_bin(),"-y","-i",src,"-an","-c:v","libx264","-pix_fmt","yuv420p",
         "-movflags","+faststart","-vf","scale='min(1280,iw)':-2",out_mp4]
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def video_desde_mp4(src_mp4, out_mp4, progress):
    progress("Preparando tu video para Clone Hero...", 88)
    _reencode_ch(src_mp4, out_mp4)

def video_desde_youtube(url, out_mp4, tmp, progress):
    progress("Descargando video de YouTube...", 80)
    raw=os.path.join(tmp,"yt_video.mp4")
    cmd=["/usr/local/bin/yt-dlp","-f","mp4/bestvideo+bestaudio","--merge-output-format","mp4",
         "-o",raw,url]
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    real=raw if os.path.exists(raw) else next((os.path.join(tmp,f) for f in os.listdir(tmp) if f.startswith("yt_video")),raw)
    progress("Convirtiendo video para Clone Hero...", 90)
    _reencode_ch(real, out_mp4)

def audio_desde_youtube(url, out_mp3, tmp, progress):
    progress("Descargando audio de YouTube...", 8)
    cmd=["/usr/local/bin/yt-dlp","-x","--audio-format","mp3","-o",os.path.join(tmp,"yt_audio.%(ext)s"),url]
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    mp3=next((os.path.join(tmp,f) for f in os.listdir(tmp) if f.startswith("yt_audio") and f.endswith(".mp3")),None)
    if mp3: shutil.copy(mp3, out_mp3)
    else: raise RuntimeError("no se pudo descargar el audio de YouTube")

# ---------------- pipeline principal (MODULAR) ----------------
def generar_cancion(name, artist, out_root="Salida CloneHero", photo_path=None,
                    mp3_path=None, youtube_url=None, video_mp4=None, video_source="mp4",
                    make_chart=True, make_lyrics=True, make_video=False,
                    whisper_size="small", letra=None, pistas=None, progress=_noop):
    folder=os.path.join(out_root, f"{_slug(artist)} - {_slug(name)}")
    os.makedirs(folder, exist_ok=True)
    progress("Iniciando...", 2)
    res=dict(folder=os.path.abspath(folder), hizo=[])
    with tempfile.TemporaryDirectory() as tmp:
        # ---------- CHART (+ letra) ----------
        if make_chart:
            audio=mp3_path
            if not audio and youtube_url:
                audio=os.path.join(tmp,"audio.mp3"); audio_desde_youtube(youtube_url,audio,tmp,progress)
            if not audio: raise RuntimeError("Falta el MP3 (o link de YouTube) para el chart")
            stems=separar_voz(audio,tmp,progress)
            data=construir_chart(stems,progress,pistas=pistas)
            chart_path=os.path.join(folder,"notes.chart")
            escribir_chart(data,name,artist,chart_path)
            progress("Copiando audio...", 60)
            shutil.copy(audio, os.path.join(folder,"song.mp3"))
            with open(os.path.join(folder,"song.ini"),"w",encoding="utf-8") as f:
                f.write("[song]\n")
                f.write(f"name = {name}\nartist = {artist}\ncharter = GuitarAI\n")
                f.write("album = \ngenre = Rock\nyear = 2025\n")
                f.write("diff_guitar = %d\n" % (5 if data.get("tiene_guitarra") else -1))
                f.write("diff_bass = %d\n" % (5 if data.get("tiene_bajo") else -1))
                f.write("diff_drums = %d\n" % (5 if data.get("tiene_bateria") else -1))
                f.write(f"song_length = {int(data['dur']*1000)}\npreview_start_time = 30000\n")
                f.write("video_start_time = 0\nicon = \ndelay = 0\nloading_phrase = Generado con GuitarAI\n")
            progress("Generando carátula...", 65)
            hacer_album(photo_path, os.path.join(folder,"album.png"))
            res["hizo"].append("chart"); res["bpm"]=round(data["bpm"],1); res["dur"]=round(data["dur"],1)
            res["guitarra"]=bool(data.get("has_guitar"))
            res["bajo"]=bool(data.get("tiene_bajo")); res["bateria"]=bool(data.get("tiene_bateria"))
            res["notas"]={k:sum(len(f) for _,f,_ in v) for k,v in data["tracks"].items() if v}
            if make_lyrics:
                res["palabras"]=agregar_letra(stems["vocals"],chart_path,data["bpm"],progress,whisper_size,letra=letra)
                res["hizo"].append("letra")
        # ---------- VIDEO (mp4 propio o YouTube) ----------
        if make_video:
            out=os.path.join(folder,"video.mp4")
            if video_source=="youtube" and youtube_url:
                video_desde_youtube(youtube_url,out,tmp,progress)
            elif video_mp4 and os.path.exists(video_mp4):
                video_desde_mp4(video_mp4,out,progress)
            else:
                raise RuntimeError("Para el video sube un MP4 o pon un link de YouTube")
            res["hizo"].append("video")
        progress("¡Listo!", 100)
    return res

if __name__=="__main__":
    import sys
    mp3=sys.argv[1] if len(sys.argv)>1 else "cancion.mp3"
    name=sys.argv[2] if len(sys.argv)>2 else "Mi Cancion"
    artist=sys.argv[3] if len(sys.argv)>3 else "Desconocido"
    photo=sys.argv[4] if len(sys.argv)>4 else None
    print(generar_cancion(mp3,name,artist,photo))
