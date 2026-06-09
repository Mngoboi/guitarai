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

# ---------------- separacion de voz (demucs) ----------------
def separar_voz(mp3_path, tmp, progress):
    progress("Separando voz y melodía (guitarra)...", 10)
    py = os.path.join(os.path.dirname(__file__), "venv", "bin", "python")
    if not os.path.exists(py):
        py = "python"
    cmd = [py, "-m", "demucs", "-o", tmp, mp3_path]   # 4 stems: vocals/drums/bass/other
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = os.path.splitext(os.path.basename(mp3_path))[0]
    d = os.path.join(tmp, "htdemucs", base)
    return {k: os.path.join(d, k + ".wav") for k in ("vocals", "drums", "bass", "other")}

# ---------------- construccion del chart ----------------
def _frets_from(values, n):
    qs = np.quantile(values, np.linspace(0, 1, n + 1)); qs[0]-=1e-6; qs[-1]+=1e-6
    return np.clip(np.digitize(values, qs) - 1, 0, n - 1)

def _chord_frets(base, size, n=5):
    if size <= 1: return [base]
    lo = max(0, min(base - (size // 2), n - size)); return list(range(lo, lo + size))

def construir_chart(stems, progress):
    progress("Analizando ritmo y melodía...", 35)
    yv, sr = librosa.load(stems["vocals"])          # VOZ
    yo, _  = librosa.load(stems["other"], sr=sr)    # MELODÍA principal (guitarra/sintes/teclas)
    yb, _  = librosa.load(stems["bass"], sr=sr)     # bajo (solo para el tempo; NO se chartea)
    # quitar BLEED de batería de la melodía: HPSS -> nos quedamos solo con lo armónico (tono real),
    # se eliminan los transitorios percusivos (las "notas fantasma rápidas" de la batería)
    progress("Limpiando melodía (sin batería)...", 38)
    yi = librosa.effects.harmonic(yo, margin=3.0)   # instrumental = melodía armónica limpia
    dur = librosa.get_duration(y=yv, sr=sr)
    # tempo SIN batería: lo saco de la melodía + bajo (contenido tonal)
    try:
        tempo, beats = librosa.beat.beat_track(y=(yo+yb), sr=sr)
        if not len(np.atleast_1d(beats)): raise ValueError
    except Exception:
        tempo, beats = librosa.beat.beat_track(y=yo, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    bt = librosa.frames_to_time(beats, sr=sr); phase = float(bt[0]) if len(bt) else 0.0
    beat_dur = 60.0 / bpm; g16 = beat_dur/4; g32 = beat_dur/8

    # actividad vocal
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

    progress("Extrayendo melodía vocal...", 45)
    oe_v = librosa.onset.onset_strength(y=yv, sr=sr)
    fv = librosa.onset.onset_detect(onset_envelope=oe_v, sr=sr, backtrack=False)
    tv = librosa.frames_to_time(fv, sr=sr); sv = oe_v[fv]
    oe_i = librosa.onset.onset_strength(y=yi, sr=sr)
    fi = librosa.onset.onset_detect(onset_envelope=oe_i, sr=sr, backtrack=False)
    ti = librosa.frames_to_time(fi, sr=sr); si = oe_i[fi]
    f0, _, _ = librosa.pyin(yv, fmin=80, fmax=1000, sr=sr, frame_length=2048, hop_length=hop)
    cent = librosa.feature.spectral_centroid(y=yi, sr=sr)[0]
    def vocal_pitch(t):
        k=int(np.clip(t/(hop/sr),0,len(f0)-1)); lo=max(0,k-3); hi=min(len(f0),k+4)
        seg=f0[lo:hi]; seg=seg[~np.isnan(seg)]
        return np.median(seg) if len(seg) else np.nan

    # === ¿la canción TIENE guitarra/melodía? (energía del stem 'other') ===
    other_rms=float(np.sqrt(np.mean(yi**2)))   # energía de la melodía LIMPIA (sin bleed de batería)
    has_guitar = other_rms > 0.009      # hay instrumento melódico audible (umbral sensible → 90/10)
    voc_frac = 0.10 if has_guitar else 0.62   # con guitarra 90/10 ; sin guitarra la voz manda

    inst_on=[]   # melodía 'other'
    for t,s in zip(ti,si):
        k=int(np.clip(t/(512/sr),0,len(cent)-1)); inst_on.append((t,np.log2(cent[k]+1e-9),s))
    voc_on=[]    # voz donde canta
    for t,s in zip(tv,sv):
        if in_voice(t):
            p=vocal_pitch(t)
            if not np.isnan(p): voc_on.append((t,np.log2(p),s))
    if has_guitar:
        target_voc=int(len(inst_on)*(voc_frac/(1-voc_frac)))     # 90/10 -> recortar voz
        if target_voc>0 and len(voc_on)>target_voc:
            voc_on=sorted(voc_on,key=lambda x:-x[2])[:target_voc]
    else:
        target_inst=int(len(voc_on)*((1-voc_frac)/voc_frac))     # sin guitarra -> recortar inst
        if target_inst>0 and len(inst_on)>target_inst:
            inst_on=sorted(inst_on,key=lambda x:-x[2])[:target_inst]
    # master con FUENTE: 'I'=instrumental/guitarra, 'V'=voz
    master=[(t,p,s,'I') for (t,p,s) in inst_on]+[(t,p,s,'V') for (t,p,s) in voc_on]
    master.sort(key=lambda m:m[0])
    times=np.array([m[0] for m in master]); pitch=np.array([m[1] for m in master])
    strg=np.array([m[2] for m in master]); src=[m[3] for m in master]
    base_frets=_frets_from(pitch,5)

    def snap(t,grid): k=round((t-phase)/grid); return phase+k*grid
    def build(min_gap_beats,n_frets,grid,sus_gap_beats,pct_triple,pct_double,max_chord):
        if not has_guitar: pct_triple*=0.4   # sin guitarra: menos notas de 3
        min_gap=min_gap_beats*beat_dur
        snp=np.array([snap(t,grid) for t in times]); kept=[]; last=-1e9
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
        if kept:
            s=np.array([strg[k[2]] for k in kept])
            thr3=np.quantile(s,1-pct_triple) if pct_triple>0 else 1e18
            thr2=np.quantile(s,1-pct_triple-pct_double) if (pct_triple+pct_double)>0 else 1e18
        notes=[]
        for j,(t,f,idx) in enumerate(kept):
            tick=int(round(t*(bpm/60.0)*RES)); sus=0
            if j+1<len(kept):
                gap=kept[j+1][0]-t
                if gap>sus_gap_beats*beat_dur: sus=int(round((gap*0.55)*(bpm/60.0)*RES))
            if src[idx]=='V':
                # VOZ: 1 o 2 notas (como solo de guitarra/piano), nunca 3
                size = 2 if (max_chord>=2 and strg[idx]>=thr3) else 1
            else:
                # INSTRUMENTAL/guitarra: triple SOLO en los golpes más fuertes (rasgueos/quintas)
                if max_chord>=3 and strg[idx]>=thr3: size=3
                elif max_chord>=2 and strg[idx]>=thr2: size=2
                else: size=1
            notes.append((tick,_chord_frets(f,min(size,max_chord),n_frets),sus))
        seen=set(); clean=[]
        for tk,fl,sus in notes:
            if tk in seen: continue
            seen.add(tk); clean.append((tk,fl,sus))
        return clean
    def starpower(notes,every_beats=16,len_beats=4):
        if not notes: return []
        sp=[]; step=int(every_beats*RES); length=int(len_beats*RES)
        ticks=[n[0] for n in notes]; cur=ticks[0]; end=ticks[-1]
        while cur<end:
            i=bisect.bisect_left(ticks,cur)
            if i<len(ticks): sp.append((ticks[i],length))
            cur+=step
        return sp

    diffs={
     "ExpertSingle": build(0.156,5,g32,0.9,0.17,0.26,3),
     "HardSingle":   build(0.30,5,g16,1.0,0.14,0.24,3),
     "MediumSingle": build(0.5,5,g16,1.25,0.0,0.18,2),
     "EasySingle":   build(1.0,4,g16,1.5,0.0,0.0,1),
    }
    sp={"ExpertSingle":starpower(diffs["ExpertSingle"]),"HardSingle":starpower(diffs["HardSingle"])}
    events=[]
    for a,b in voice_segs:
        events.append((int(round(a*(bpm/60.0)*RES)),'section Voz'))
    for m in [0.0]+[b for a,b in voice_segs]:
        events.append((int(round(m*(bpm/60.0)*RES)),'section Instrumental'))
    events=sorted(set(events))
    return dict(bpm=bpm,dur=dur,diffs=diffs,sp=sp,events=events,voice_segs=voice_segs,has_guitar=has_guitar)

def escribir_chart(data, name, artist, out_path):
    bpm=data["bpm"]; diffs=data["diffs"]; sp=data["sp"]; events=data["events"]
    with open(out_path,"w",encoding="utf-8") as f:
        f.write("[Song]\n{\n")
        f.write(f'  Name = "{name}"\n  Artist = "{artist}"\n  Charter = "GuitarAI"\n')
        f.write(f'  Offset = 0\n  Resolution = {RES}\n  Player2 = bass\n  Difficulty = 6\n')
        f.write(f'  PreviewStart = 0\n  PreviewEnd = 0\n  Genre = "rock"\n  MediaType = "cd"\n  MusicStream = "song.mp3"\n}}\n')
        f.write("[SyncTrack]\n{\n  0 = TS 4\n  0 = B %d\n}\n"%int(round(bpm*1000)))
        f.write("[Events]\n{\n")
        for tk,ev in events: f.write(f'  {tk} = E "{ev}"\n')
        f.write("}\n")
        for track in ["ExpertSingle","HardSingle","MediumSingle","EasySingle"]:
            f.write(f"[{track}]\n"+"{\n")
            rows=[]
            for tk,fl,sus in diffs[track]:
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

def agregar_letra(voc, chart_path, bpm, progress, size="small", letra=None):
    progress("Sincronizando letra (karaoke)...", 80)
    m=_get_whisper(size)
    r=m.transcribe(voc, language="es", word_timestamps=True, verbose=False)
    words=[]
    for seg in r["segments"]:
        for w in seg.get("words",[]):
            t=w.get("word","").replace('"','').strip()
            if t: words.append((round(w["start"],3),round(w["end"],3),t))
    if not words: return 0
    def tick(t): return int(round(t*(bpm/60.0)*RES))
    def clean(s): return s.replace('"','').strip()

    if letra and letra.strip():
        # LETRA DEL USUARIO: alineación por COINCIDENCIA de palabras con la voz (difflib) + interpolación
        import difflib
        def norm(s): return re.sub(r'[^0-9a-záéíóúüñ]', '', s.lower())
        lines=[ln.strip() for ln in letra.splitlines() if ln.strip()]
        utok=[(clean(wd),li) for li,ln in enumerate(lines) for wd in ln.split() if clean(wd)]
        M=len(utok)
        wnorm=[norm(w[2]) for w in words]; unorm=[norm(t[0]) for t in utok]
        tu=[None]*M
        sm=difflib.SequenceMatcher(None, wnorm, unorm, autojunk=False)
        for a,b,size in sm.get_matching_blocks():           # anclar palabras que coinciden
            for k in range(size): tu[b+k]=words[a+k][0]
        known=[j for j in range(M) if tu[j] is not None]
        if not known:                                       # sin coincidencias: repartir por la voz
            t0=words[0][0]; t1=words[-1][0]
            for j in range(M): tu[j]=t0+(t1-t0)*j/max(M-1,1)
        else:                                               # interpolar huecos
            tstart=words[0][0]; tend=words[-1][0]
            a1=known[0]                                      # antes de la 1ª ancla: desde el inicio de la voz
            for j in range(0,a1): tu[j]=tstart+(tu[a1]-tstart)*j/max(a1,1)
            for ii in range(len(known)-1):                  # entre anclas
                a0,a1=known[ii],known[ii+1]; t0,t1=tu[a0],tu[a1]
                for j in range(a0+1,a1): tu[j]=t0+(t1-t0)*(j-a0)/(a1-a0)
            a0=known[-1]                                     # después de la última: hasta el fin de la voz
            for j in range(a0+1,M): tu[j]=tu[a0]+(tend-tu[a0])*(j-a0)/max(M-1-a0,1)
        timed=[(tu[j], utok[j][0], utok[j][1]) for j in range(M)]
        for j in range(1,len(timed)):   # tiempos no decrecientes
            if timed[j][0]<timed[j-1][0]: timed[j]=(timed[j-1][0],timed[j][1],timed[j][2])
        phrases=[]; cur=[]; curline=timed[0][2] if timed else 0
        for s,wd,li in timed:
            if li!=curline and cur: phrases.append(cur); cur=[]; curline=li
            cur.append((s,s,wd))
        if cur: phrases.append(cur)
        nwords=M
    else:
        # AUTO: transcripción de whisper, frases por silencio
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
                    whisper_size="small", letra=None, progress=_noop):
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
            data=construir_chart(stems,progress)
            chart_path=os.path.join(folder,"notes.chart")
            escribir_chart(data,name,artist,chart_path)
            progress("Copiando audio...", 60)
            shutil.copy(audio, os.path.join(folder,"song.mp3"))
            with open(os.path.join(folder,"song.ini"),"w",encoding="utf-8") as f:
                f.write("[song]\n")
                f.write(f"name = {name}\nartist = {artist}\ncharter = GuitarAI\n")
                f.write("album = \ngenre = Rock\nyear = 2025\n")
                f.write("diff_guitar = 6\ndiff_bass = -1\ndiff_drums = -1\n")
                f.write(f"song_length = {int(data['dur']*1000)}\npreview_start_time = 30000\n")
                f.write("video_start_time = 0\nicon = \ndelay = 0\nloading_phrase = Generado con GuitarAI\n")
            progress("Generando carátula...", 65)
            hacer_album(photo_path, os.path.join(folder,"album.png"))
            res["hizo"].append("chart"); res["bpm"]=round(data["bpm"],1); res["dur"]=round(data["dur"],1)
            res["guitarra"]=bool(data.get("has_guitar"))
            res["notas"]={k:sum(len(f) for _,f,_ in v) for k,v in data["diffs"].items()}
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
