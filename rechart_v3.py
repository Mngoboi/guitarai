# -*- coding: utf-8 -*-
"""Chart v3: triples+dobles+simples y sustains en Hard/Expert + Star Power.
Conserva nombre 'boca arriba' / 'silvestre33'. No toca audio ni album."""
import os, numpy as np, librosa

AUDIO="cancion.mp3"; RES=192; OUT_DIR="Salida CloneHero/Mi Cancion"
SONG_NAME="boca arriba"; ARTIST="silvestre33"; CHARTER="GuitarAI (auto)"

print("==> Analizando audio...")
y,sr=librosa.load(AUDIO); dur=librosa.get_duration(y=y,sr=sr)
tempo,beats=librosa.beat.beat_track(y=y,sr=sr); bpm=float(np.atleast_1d(tempo)[0])
bt=librosa.frames_to_time(beats,sr=sr); phase=float(bt[0]) if len(bt) else 0.0
beat_dur=60.0/bpm; g16=beat_dur/4; g32=beat_dur/8
oenv=librosa.onset.onset_strength(y=y,sr=sr)
of=librosa.onset.onset_detect(onset_envelope=oenv,sr=sr,backtrack=False)
ot=librosa.frames_to_time(of,sr=sr); ostr=oenv[of]
cent=librosa.feature.spectral_centroid(y=y,sr=sr)[0]
logc=np.log2(cent[np.clip(of,0,len(cent)-1)]+1e-9)
print(f"    BPM={bpm:.2f}  onsets={len(ot)}")

def snap(t,grid): k=round((t-phase)/grid); return phase+k*grid
def frets(values,n):
    qs=np.quantile(values,np.linspace(0,1,n+1)); qs[0]-=1e-6; qs[-1]+=1e-6
    return np.clip(np.digitize(values,qs)-1,0,n-1)

def chord_frets(base,size,n=5):
    """Devuelve 'size' trastes distintos centrados en base, dentro de 0..n-1."""
    if size<=1: return [base]
    lo=max(0,min(base-(size//2),n-size))
    return list(range(lo,lo+size))

def build(min_gap_beats,n_frets,grid,sus_gap_beats,
          pct_triple=0.0,pct_double=0.0,max_chord=1):
    min_gap=min_gap_beats*beat_dur
    snp=np.array([snap(t,grid) for t in ot]); fr=frets(logc,n_frets)
    kept=[]; last=-1e9
    for i in range(len(snp)):
        t=snp[i]
        if t-last<min_gap-1e-4:
            if kept and ostr[i]>ostr[kept[-1][2]]: kept[-1]=(t,int(fr[i]),i); last=t
            continue
        kept.append((t,int(fr[i]),i)); last=t
    # anti machine-gun
    run=0
    for j in range(1,len(kept)):
        if kept[j][1]==kept[j-1][1]:
            run+=1
            if run>=3: kept[j]=(kept[j][0],(kept[j][1]+1)%n_frets,kept[j][2]); run=0
        else: run=0
    # umbrales de acorde por fuerza
    if kept:
        s=np.array([ostr[k[2]] for k in kept])
        thr3=np.quantile(s,1-pct_triple) if pct_triple>0 else 1e18
        thr2=np.quantile(s,1-pct_triple-pct_double) if (pct_triple+pct_double)>0 else 1e18
    notes=[]
    for j,(t,f,idx) in enumerate(kept):
        tick=int(round(t*(bpm/60.0)*RES))
        # sustain razonable
        sus=0
        if j+1<len(kept):
            gap=kept[j+1][0]-t
            if gap>sus_gap_beats*beat_dur:
                sus=int(round((gap*0.55)*(bpm/60.0)*RES))
        # tamano de acorde
        size=1
        if max_chord>=3 and ostr[idx]>=thr3: size=3
        elif max_chord>=2 and ostr[idx]>=thr2: size=2
        fl=chord_frets(f,min(size,max_chord),n_frets)
        notes.append((tick,fl,sus))
    seen=set(); clean=[]
    for tk,fl,sus in notes:
        if tk in seen: continue
        seen.add(tk); clean.append((tk,fl,sus))
    return clean

def starpower(notes,every_beats=16,len_beats=4):
    """Genera frases de Star Power (S 2 length) sobre grupos de notas."""
    if not notes: return []
    sp=[]; step=int(every_beats*RES); length=int(len_beats*RES)
    cursor=notes[0][0]; end=notes[-1][0]
    ticks=[n[0] for n in notes]
    import bisect
    while cursor<end:
        i=bisect.bisect_left(ticks,cursor)
        if i<len(ticks):
            sp.append((ticks[i],length))
        cursor+=step
    return sp

# Hard y Expert: triples+dobles+simples + sustains + star power
diffs={
 "ExpertSingle": build(0.125,5,g32,0.9, pct_triple=0.22,pct_double=0.33,max_chord=3),
 "HardSingle":   build(0.25,5,g16,1.0, pct_triple=0.18,pct_double=0.30,max_chord=3),
 "MediumSingle": build(0.5,5,g16,1.25, pct_triple=0.0,pct_double=0.18,max_chord=2),
 "EasySingle":   build(1.0,4,g16,1.5, pct_triple=0.0,pct_double=0.0,max_chord=1),
}
sp={"ExpertSingle":starpower(diffs["ExpertSingle"]),
    "HardSingle":starpower(diffs["HardSingle"])}

for k,v in diffs.items():
    g=sum(len(f) for _,f,_ in v); s=sum(1 for _,_,su in v if su>0)
    t1=sum(1 for _,f,_ in v if len(f)==1); t2=sum(1 for _,f,_ in v if len(f)==2); t3=sum(1 for _,f,_ in v if len(f)==3)
    spn=len(sp.get(k,[]))
    print(f"    {k}: {len(v)} pos, {g} gemas | simples {t1} dobles {t2} triples {t3} | sustains {s} | StarPower {spn}")

events=[]; tpb=RES*4; total=int(round(dur*(bpm/60.0)*RES)); n=1
for tk in range(0,total,tpb*8): events.append((tk,f'section Parte {n}')); n+=1

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
        rows.sort(key=lambda r:(r[0], 0 if r[1][0]=='S' else 1))
        for tk,txt in rows: f.write(f"  {tk} = {txt}\n")
        f.write("}\n")
print("==> notes.chart escrito (Difficulty=6)")

ini=os.path.join(OUT_DIR,"song.ini")
lines=open(ini,encoding="utf-8").read().splitlines(); out=[]
for ln in lines:
    if ln.startswith("diff_guitar"): out.append("diff_guitar = 6")
    else: out.append(ln)
open(ini,"w",encoding="utf-8").write("\n".join(out)+"\n")
print("LISTO")
