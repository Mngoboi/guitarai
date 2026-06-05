# -*- coding: utf-8 -*-
"""
motor.py — Pipeline completo: MP3 + nombre + artista + foto -> carpeta Clone Hero.
Genera: notes.chart (dinamico voz/instrumental, acordes/sustains/StarPower),
        song.ini, song.mp3, album.png, video.mp4 (late con el ritmo), letra karaoke.
Uso:  from motor import generar_cancion
"""
import os, re, json, shutil, subprocess, tempfile, bisect
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
    progress("Separando voz de la instrumental (demucs)...", 10)
    py = os.path.join(os.path.dirname(__file__), "venv", "bin", "python")
    if not os.path.exists(py):
        py = "python"
    cmd = [py, "-m", "demucs", "--two-stems=vocals", "-o", tmp, mp3_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = os.path.splitext(os.path.basename(mp3_path))[0]
    d = os.path.join(tmp, "htdemucs", base)
    return os.path.join(d, "vocals.wav"), os.path.join(d, "no_vocals.wav")

# ---------------- construccion del chart ----------------
def _frets_from(values, n):
    qs = np.quantile(values, np.linspace(0, 1, n + 1)); qs[0]-=1e-6; qs[-1]+=1e-6
    return np.clip(np.digitize(values, qs) - 1, 0, n - 1)

def _chord_frets(base, size, n=5):
    if size <= 1: return [base]
    lo = max(0, min(base - (size // 2), n - size)); return list(range(lo, lo + size))

def construir_chart(voc, inst, progress):
    progress("Analizando ritmo y melodía...", 35)
    yv, sr = librosa.load(voc); yi, _ = librosa.load(inst, sr=sr)
    dur = librosa.get_duration(y=yi, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=yi, sr=sr); bpm = float(np.atleast_1d(tempo)[0])
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

    master=[]
    for t,s in zip(tv,sv):
        if in_voice(t):
            p=vocal_pitch(t)
            if not np.isnan(p): master.append((t,np.log2(p),s))
    for t,s in zip(ti,si):
        if not in_voice(t):
            k=int(np.clip(t/(512/sr),0,len(cent)-1)); master.append((t,np.log2(cent[k]+1e-9),s))
    master.sort()
    times=np.array([m[0] for m in master]); pitch=np.array([m[1] for m in master]); strg=np.array([m[2] for m in master])
    base_frets=_frets_from(pitch,5)

    def snap(t,grid): k=round((t-phase)/grid); return phase+k*grid
    def build(min_gap_beats,n_frets,grid,sus_gap_beats,pct_triple,pct_double,max_chord):
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
            size=1
            if max_chord>=3 and strg[idx]>=thr3: size=3
            elif max_chord>=2 and strg[idx]>=thr2: size=2
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
     "ExpertSingle": build(0.125,5,g32,0.9,0.22,0.33,3),
     "HardSingle":   build(0.25,5,g16,1.0,0.18,0.30,3),
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
    return dict(bpm=bpm,dur=dur,diffs=diffs,sp=sp,events=events,voice_segs=voice_segs)

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

def agregar_letra(voc, chart_path, bpm, progress, size="small"):
    progress("Transcribiendo letra (karaoke)...", 80)
    m=_get_whisper(size)
    r=m.transcribe(voc, language="es", word_timestamps=True, verbose=False)
    words=[]
    for seg in r["segments"]:
        for w in seg.get("words",[]):
            t=w.get("word","").replace('"','').strip()
            if t: words.append((round(w["start"],3),round(w["end"],3),t))
    if not words: return 0
    def tick(t): return int(round(t*(bpm/60.0)*RES))
    phrases=[]; cur=[]
    for w in words:
        if cur and w[0]-cur[-1][1]>1.2: phrases.append(cur); cur=[]
        cur.append(w)
    if cur: phrases.append(cur)
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
    return len(words)

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

def hacer_video(img_png, mp3_path, out_mp4, progress, fps=24, amp=0.12):
    progress("Creando video que late con el ritmo...", 90)
    import numpy as np
    from PIL import Image
    from moviepy import VideoClip, AudioFileClip
    y,sr=librosa.load(mp3_path)
    dur=librosa.get_duration(y=y,sr=sr)
    env=librosa.onset.onset_strength(y=y,sr=sr); et=librosa.times_like(env,sr=sr)
    env=(env-env.min())/(np.ptp(env)+1e-9); env=env**1.5
    W,H=1280,720
    base=Image.open(img_png).convert("RGB")
    scale=max(W/base.width,H/base.height)*1.25
    base=base.resize((int(base.width*scale),int(base.height*scale)),Image.LANCZOS)
    bw,bh=base.size
    def pulse(t): return float(np.interp(t,et,env))
    def make_frame(t):
        z=1.0+amp*pulse(t); cw,ch=int(W/z),int(H/z)
        left=(bw-cw)//2; top=(bh-ch)//2
        return np.asarray(base.crop((left,top,left+cw,top+ch)).resize((W,H),Image.LANCZOS))
    clip=VideoClip(make_frame,duration=dur).with_audio(AudioFileClip(mp3_path))
    clip.write_videofile(out_mp4,fps=fps,codec="libx264",audio_codec="aac",logger=None)

# ---------------- pipeline principal ----------------
def generar_cancion(mp3_path, name, artist, photo_path=None,
                    out_root="Salida CloneHero", make_video=True, make_lyrics=True,
                    whisper_size="small", progress=_noop):
    folder=os.path.join(out_root, f"{_slug(artist)} - {_slug(name)}")
    os.makedirs(folder, exist_ok=True)
    progress("Iniciando...", 2)
    with tempfile.TemporaryDirectory() as tmp:
        voc,inst=separar_voz(mp3_path,tmp,progress)
        data=construir_chart(voc,inst,progress)
        chart_path=os.path.join(folder,"notes.chart")
        escribir_chart(data,name,artist,chart_path)
        # song.mp3
        progress("Copiando audio...", 60)
        shutil.copy(mp3_path, os.path.join(folder,"song.mp3"))
        # song.ini
        with open(os.path.join(folder,"song.ini"),"w",encoding="utf-8") as f:
            f.write("[song]\n")
            f.write(f"name = {name}\nartist = {artist}\ncharter = GuitarAI\n")
            f.write("album = \ngenre = Rock\nyear = 2025\n")
            f.write("diff_guitar = 6\ndiff_bass = -1\ndiff_drums = -1\n")
            f.write(f"song_length = {int(data['dur']*1000)}\npreview_start_time = 30000\n")
            f.write("video_start_time = 0\nicon = \ndelay = 0\nloading_phrase = Generado con GuitarAI\n")
        # album
        progress("Generando carátula...", 65)
        album=os.path.join(folder,"album.png"); hacer_album(photo_path,album)
        # letra
        words=0
        if make_lyrics:
            words=agregar_letra(voc,chart_path,data["bpm"],progress,whisper_size)
        # video
        if make_video:
            hacer_video(album, mp3_path, os.path.join(folder,"video.mp4"), progress)
        progress("¡Listo!", 100)
    return dict(folder=os.path.abspath(folder), bpm=round(data["bpm"],1),
                dur=round(data["dur"],1), palabras=words,
                notas={k:sum(len(f) for _,f,_ in v) for k,v in data["diffs"].items()})

if __name__=="__main__":
    import sys
    mp3=sys.argv[1] if len(sys.argv)>1 else "cancion.mp3"
    name=sys.argv[2] if len(sys.argv)>2 else "Mi Cancion"
    artist=sys.argv[3] if len(sys.argv)>3 else "Desconocido"
    photo=sys.argv[4] if len(sys.argv)>4 else None
    print(generar_cancion(mp3,name,artist,photo))
