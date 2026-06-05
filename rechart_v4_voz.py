# -*- coding: utf-8 -*-
"""Chart v4 DINAMICO: sigue la VOZ cuando hay canto, y la INSTRUMENTAL cuando no.
- Separa por demucs (ya hecho): vocals.wav / no_vocals.wav
- Detecta actividad vocal (RMS)
- En tramos con voz: onsets+tono de la VOZ (melodia cantada -> trastes)
- En tramos sin voz: onsets de la instrumental
- Mantiene triples/dobles/simples, sustains y Star Power (Hard/Expert)
- Marca secciones 'Voz' / 'Instrumental'
No toca song.mp3 / album.png."""
import os, bisect, numpy as np, librosa

VOC="separated/htdemucs/cancion/vocals.wav"
INST="separated/htdemucs/cancion/no_vocals.wav"
MIX="cancion.mp3"
RES=192; OUT_DIR="Salida CloneHero/Mi Cancion"
SONG_NAME="boca arriba"; ARTIST="silvestre33"; CHARTER="GuitarAI (auto)"

print("==> Cargando stems...")
yv,sr=librosa.load(VOC); yi,_=librosa.load(INST,sr=sr)
dur=librosa.get_duration(y=yi,sr=sr)
tempo,beats=librosa.beat.beat_track(y=yi,sr=sr); bpm=float(np.atleast_1d(tempo)[0])
bt=librosa.frames_to_time(beats,sr=sr); phase=float(bt[0]) if len(bt) else 0.0
beat_dur=60.0/bpm; g16=beat_dur/4; g32=beat_dur/8
print(f"    BPM={bpm:.2f}  dur={dur:.1f}s")

# ---- Actividad vocal por RMS
print("==> Detectando dónde canta la voz...")
hop=512
rms=librosa.feature.rms(y=yv,hop_length=hop)[0]
rt=librosa.frames_to_time(np.arange(len(rms)),sr=sr,hop_length=hop)
thr=np.percentile(rms,60)*0.8 + rms.max()*0.04
active=rms>thr
# suavizar: rellenar huecos cortos y quitar islas cortas
def smooth(mask,times,min_on=0.4,min_off=0.3):
    segs=[]; i=0; n=len(mask)
    while i<n:
        if mask[i]:
            j=i
            while j<n and mask[j]: j+=1
            segs.append([times[i],times[min(j,n-1)]]); i=j
        else: i+=1
    # unir si el hueco entre segmentos < min_off
    merged=[]
    for s in segs:
        if merged and s[0]-merged[-1][1]<min_off: merged[-1][1]=s[1]
        else: merged.append(s)
    # descartar segmentos muy cortos
    return [s for s in merged if s[1]-s[0]>=min_on]
voice_segs=smooth(active,rt)
voiced_total=sum(b-a for a,b in voice_segs)
print(f"    Tramos con voz: {len(voice_segs)}  ({voiced_total:.0f}s de {dur:.0f}s)")

def in_voice(t):
    for a,b in voice_segs:
        if a<=t<=b: return True
    return False

# ---- Onsets de cada stem
print("==> Onsets de voz e instrumental...")
oe_v=librosa.onset.onset_strength(y=yv,sr=sr)
fv=librosa.onset.onset_detect(onset_envelope=oe_v,sr=sr,backtrack=False)
tv=librosa.frames_to_time(fv,sr=sr); sv=oe_v[fv]
oe_i=librosa.onset.onset_strength(y=yi,sr=sr)
fi=librosa.onset.onset_detect(onset_envelope=oe_i,sr=sr,backtrack=False)
ti=librosa.frames_to_time(fi,sr=sr); si=oe_i[fi]

# ---- Tono de la VOZ (pyin) para que las notas sigan la melodia cantada
print("==> Extrayendo melodía vocal (pyin)... (tarda un poco)")
f0,vflag,vprob=librosa.pyin(yv,fmin=80,fmax=1000,sr=sr,frame_length=2048,hop_length=hop)
f0t=librosa.frames_to_time(np.arange(len(f0)),sr=sr,hop_length=hop)
def vocal_pitch(t):
    k=int(np.clip(t/(hop/sr),0,len(f0)-1))
    lo=max(0,k-3); hi=min(len(f0),k+4)
    seg=f0[lo:hi]; seg=seg[~np.isnan(seg)]
    return np.median(seg) if len(seg) else np.nan

# centro espectral instrumental (para trastes sin voz)
cent=librosa.feature.spectral_centroid(y=yi,sr=sr)[0]
ct=librosa.frames_to_time(np.arange(len(cent)),sr=sr)

# ---- Construir lista maestra de onsets (tiempo, valor_tono, fuerza, fuente)
master=[]   # (t, pitchval, strength, source)
for t,s in zip(tv,sv):
    if in_voice(t):
        p=vocal_pitch(t)
        if not np.isnan(p): master.append((t,np.log2(p),s,'V'))
for t,s in zip(ti,si):
    if not in_voice(t):
        k=int(np.clip(t/(512/sr),0,len(cent)-1))
        master.append((t,np.log2(cent[k]+1e-9),s,'I'))
master.sort()
print(f"    Onsets totales: {len(master)}  (voz {sum(1 for m in master if m[3]=='V')}, inst {sum(1 for m in master if m[3]=='I')})")

times=np.array([m[0] for m in master]); pitch=np.array([m[1] for m in master])
strg=np.array([m[2] for m in master])

def snap(t,grid): k=round((t-phase)/grid); return phase+k*grid
def frets_from(values,n):
    qs=np.quantile(values,np.linspace(0,1,n+1)); qs[0]-=1e-6; qs[-1]+=1e-6
    return np.clip(np.digitize(values,qs)-1,0,n-1)
def chord_frets(base,size,n=5):
    if size<=1: return [base]
    lo=max(0,min(base-(size//2),n-size)); return list(range(lo,lo+size))

base_frets=frets_from(pitch,5)  # trastes siguiendo el tono (voz o inst)

def build(min_gap_beats,n_frets,grid,sus_gap_beats,pct_triple,pct_double,max_chord):
    min_gap=min_gap_beats*beat_dur
    snp=np.array([snap(t,grid) for t in times])
    kept=[]; last=-1e9
    for i in range(len(snp)):
        t=snp[i]
        if t-last<min_gap-1e-4:
            if kept and strg[i]>strg[kept[-1][2]]: kept[-1]=(t,int(base_frets[i]*( (n_frets-1)/4 )+0.5),i); last=t
            continue
        fr=int(round(base_frets[i]*((n_frets-1)/4)))
        kept.append((t,fr,i)); last=t
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
        notes.append((tick,chord_frets(f,min(size,max_chord),n_frets),sus))
    seen=set(); clean=[]
    for tk,fl,sus in notes:
        if tk in seen: continue
        seen.add(tk); clean.append((tk,fl,sus))
    return clean

def starpower(notes,every_beats=16,len_beats=4):
    if not notes: return []
    sp=[]; step=int(every_beats*RES); length=int(len_beats*RES)
    ticks=[n[0] for n in notes]; cursor=ticks[0]; end=ticks[-1]
    while cursor<end:
        i=bisect.bisect_left(ticks,cursor)
        if i<len(ticks): sp.append((ticks[i],length))
        cursor+=step
    return sp

diffs={
 "ExpertSingle": build(0.125,5,g32,0.9,0.22,0.33,3),
 "HardSingle":   build(0.25,5,g16,1.0,0.18,0.30,3),
 "MediumSingle": build(0.5,5,g16,1.25,0.0,0.18,2),
 "EasySingle":   build(1.0,4,g16,1.5,0.0,0.0,1),
}
sp={"ExpertSingle":starpower(diffs["ExpertSingle"]),"HardSingle":starpower(diffs["HardSingle"])}
for k,v in diffs.items():
    g=sum(len(f) for _,f,_ in v); su=sum(1 for _,_,s in v if s>0)
    t1=sum(1 for _,f,_ in v if len(f)==1); t2=sum(1 for _,f,_ in v if len(f)==2); t3=sum(1 for _,f,_ in v if len(f)==3)
    print(f"    {k}: {len(v)} pos | simples {t1} dobles {t2} triples {t3} | sustains {su} | SP {len(sp.get(k,[]))}")

# eventos: secciones Voz / Instrumental
events=[]
for a,b in voice_segs:
    tk=int(round(a*(bpm/60.0)*RES)); events.append((tk,'section Voz'))
# instrumental: inicio de la cancion y despues de cada tramo de voz
prev_end=0.0
inst_marks=[0.0]+[b for a,b in voice_segs]
for m in inst_marks:
    tk=int(round(m*(bpm/60.0)*RES)); events.append((tk,'section Instrumental'))
events=sorted(set(events))

path=os.path.join(OUT_DIR,"notes.chart")
with open(path,"w",encoding="utf-8") as f:
    f.write("[Song]\n{\n")
    f.write(f'  Name = "{SONG_NAME}"\n  Artist = "{ARTIST}"\n  Charter = "{CHARTER}"\n')
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
print("==> notes.chart escrito (dinamico voz/instrumental)")
print("LISTO")
