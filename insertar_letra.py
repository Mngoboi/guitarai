# -*- coding: utf-8 -*-
"""Inserta la letra de letra.json como eventos de karaoke en notes.chart."""
import json, re, numpy as np, librosa
RES=192; OUT="Salida CloneHero/Mi Cancion/notes.chart"
# BPM (mismo que el chart)
y,sr=librosa.load("separated/htdemucs/cancion/no_vocals.wav")
tempo,_=librosa.beat.beat_track(y=y,sr=sr); bpm=float(np.atleast_1d(tempo)[0])
def tick(t): return int(round(t*(bpm/60.0)*RES))

words=json.load(open("letra.json"))["words"]
# limpiar y agrupar en frases por silencios > 1.2s
def clean(w):
    w=w.replace('"',"").replace("\n"," ").strip()
    return w
phrases=[]; cur=[]
for i,w in enumerate(words):
    txt=clean(w["w"])
    if not txt: continue
    if cur and w["t"]-cur[-1]["e"]>1.2:
        phrases.append(cur); cur=[]
    cur.append({"t":w["t"],"e":w["e"],"w":txt})
if cur: phrases.append(cur)

# leer chart y extraer eventos de seccion existentes
txt=open(OUT,encoding="utf-8").read()
m=re.search(r"\[Events\]\s*\{(.*?)\}", txt, re.S)
existing=[]
for line in m.group(1).splitlines():
    line=line.strip()
    mm=re.match(r"(\d+)\s*=\s*E\s*\"(.+)\"", line)
    if mm and "section" in mm.group(2):
        existing.append((int(mm.group(1)), 'E "%s"'%mm.group(2)))

# construir eventos de letra
lyr=[]
for ph in phrases:
    lyr.append((tick(ph[0]["t"]), 'E "phrase_start"'))
    for wd in ph:
        lyr.append((tick(wd["t"]), 'E "lyric %s"'%wd["w"]))
    lyr.append((tick(ph[-1]["e"]), 'E "phrase_end"'))

allev=existing+lyr
# orden estable: por tick, y dentro del tick phrase_start antes que lyric antes que phrase_end
def keyf(e):
    pr=2
    if "phrase_start" in e[1]: pr=0
    elif "section" in e[1]: pr=1
    elif "lyric" in e[1]: pr=3
    elif "phrase_end" in e[1]: pr=4
    return (e[0],pr)
allev.sort(key=keyf)

block="[Events]\n{\n"+"".join("  %d = %s\n"%(t,s) for t,s in allev)+"}"
newtxt=txt[:m.start()]+block+txt[m.end():]
open(OUT,"w",encoding="utf-8").write(newtxt)
print("Frases:",len(phrases)," palabras:",sum(len(p) for p in phrases)," eventos letra:",len(lyr))
