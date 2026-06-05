# -*- coding: utf-8 -*-
"""Dashboard web GuitarAI — sube MP3 + foto, escribe nombre/artista, genera la carpeta."""
import os, threading, uuid, zipfile, io, traceback
from flask import Flask, request, jsonify, send_file, Response
from motor import generar_cancion

BASE=os.path.dirname(os.path.abspath(__file__))
UPLOAD=os.path.join(BASE,"_uploads"); OUT=os.path.join(BASE,"Salida CloneHero")
os.makedirs(UPLOAD,exist_ok=True); os.makedirs(OUT,exist_ok=True)
app=Flask(__name__)
JOBS={}   # id -> {pct,msg,done,error,result}

def run_job(jid, mp3, name, artist, photo, make_video, make_lyrics):
    def prog(msg,pct): JOBS[jid].update(msg=msg,pct=pct)
    try:
        r=generar_cancion(mp3,name,artist,photo_path=photo,out_root=OUT,
                          make_video=make_video,make_lyrics=make_lyrics,progress=prog)
        JOBS[jid].update(done=True,result=r,pct=100,msg="¡Listo!")
    except Exception as e:
        JOBS[jid].update(done=True,error=str(e),trace=traceback.format_exc())

@app.route("/")
def index(): return Response(HTML,mimetype="text/html")

@app.route("/generar",methods=["POST"])
def generar():
    name=request.form.get("name","Mi Cancion").strip() or "Mi Cancion"
    artist=request.form.get("artist","Desconocido").strip() or "Desconocido"
    make_video=request.form.get("video","1")=="1"
    make_lyrics=request.form.get("lyrics","1")=="1"
    mp3f=request.files.get("mp3")
    if not mp3f: return jsonify(error="Falta el MP3"),400
    jid=uuid.uuid4().hex[:8]; d=os.path.join(UPLOAD,jid); os.makedirs(d,exist_ok=True)
    mp3=os.path.join(d,"audio.mp3"); mp3f.save(mp3)
    photo=None
    pf=request.files.get("photo")
    if pf and pf.filename:
        photo=os.path.join(d,"foto"+os.path.splitext(pf.filename)[1]); pf.save(photo)
    JOBS[jid]=dict(pct=0,msg="En cola...",done=False,error=None,result=None)
    threading.Thread(target=run_job,args=(jid,mp3,name,artist,photo,make_video,make_lyrics),daemon=True).start()
    return jsonify(job=jid)

@app.route("/estado/<jid>")
def estado(jid):
    j=JOBS.get(jid)
    if not j: return jsonify(error="job no existe"),404
    return jsonify(j)

@app.route("/descargar/<jid>")
def descargar(jid):
    j=JOBS.get(jid)
    if not j or not j.get("result"): return "No listo",404
    folder=j["result"]["folder"]
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        for root,_,files in os.walk(folder):
            for fn in files:
                fp=os.path.join(root,fn); z.write(fp,os.path.relpath(fp,os.path.dirname(folder)))
    buf.seek(0)
    return send_file(buf,mimetype="application/zip",as_attachment=True,
                     download_name=os.path.basename(folder)+".zip")

HTML=r"""<!doctype html><html lang=es><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>GuitarAI — Generador Clone Hero</title>
<style>
*{box-sizing:border-box;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
body{margin:0;color:#fff;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px;overflow:hidden}

/* ===== ESCENA ATARDECER ANIME ===== */
.scene{position:fixed;inset:0;z-index:0;overflow:hidden;
  background:linear-gradient(180deg,#1a1a4e 0%,#4b2a73 28%,#9b3b78 48%,#e0608e 63%,#ff9a5a 78%,#ffd06b 92%,#ffe9a8 100%)}
.sun{position:absolute;left:50%;top:60%;width:200px;height:200px;margin:-100px 0 0 -100px;border-radius:50%;
  background:radial-gradient(circle,#fff6d8 0%,#ffe07a 35%,#ffb24a 60%,rgba(255,150,70,0) 72%);
  filter:blur(2px);animation:sun 8s ease-in-out infinite}
@keyframes sun{0%,100%{transform:scale(1);opacity:.95}50%{transform:scale(1.06);opacity:1}}
.cloud{position:absolute;border-radius:50%;background:rgba(255,225,210,.55);filter:blur(14px)}
.c1{width:240px;height:60px;top:18%;animation:drift 70s linear infinite}
.c2{width:340px;height:80px;top:30%;opacity:.7;animation:drift 100s linear infinite;animation-delay:-30s}
.c3{width:180px;height:48px;top:12%;opacity:.5;animation:drift 85s linear infinite;animation-delay:-55s}
@keyframes drift{from{left:115%}to{left:-40%}}

.hills{position:absolute;left:0;bottom:0;width:100%;display:block}
.far{height:58vh}.near{height:34vh}

.trees{position:absolute;left:0;bottom:7vh;width:100%;height:30vh;z-index:3;pointer-events:none}
.tree{position:absolute;bottom:0;transform-origin:bottom center}
.tree .trunk{position:absolute;bottom:0;left:50%;width:8px;height:46px;margin-left:-4px;background:#140d22;border-radius:3px}
.tree .leaf-ball{position:absolute;bottom:34px;left:50%;width:78px;height:78px;margin-left:-39px;border-radius:50%;
  background:radial-gradient(circle at 40% 35%,#1c1330,#0d0820);transform-origin:bottom center;animation:sway 4.5s ease-in-out infinite}
.t1{left:14%;transform:scale(.8)}.t1 .leaf-ball{animation-delay:-1s}
.t2{left:78%;transform:scale(1.05)}.t2 .leaf-ball{animation-delay:-2.4s;animation-duration:5.5s}
.t3{left:90%;transform:scale(.7)}.t3 .leaf-ball{animation-delay:-3.2s}
@keyframes sway{0%,100%{transform:rotate(-3.5deg)}50%{transform:rotate(3.5deg)}}

.grass{position:absolute;left:0;bottom:0;width:100%;height:8vh;z-index:4;display:flex;align-items:flex-end;gap:5px;padding:0 4px;overflow:hidden}
.blade{flex:1;height:100%;background:linear-gradient(180deg,#0e0a1c,#1a1230);border-radius:50% 50% 0 0/22px;transform-origin:bottom center;animation:gsway 3.6s ease-in-out infinite}
@keyframes gsway{0%,100%{transform:skewX(-7deg)}50%{transform:skewX(7deg)}}

.leaves{position:absolute;inset:0;z-index:5;pointer-events:none;overflow:hidden}
.lf{position:absolute;width:13px;height:9px;border-radius:0 100% 0 100%;
  background:linear-gradient(135deg,#ffcf6e,#e07a3c);opacity:.92;animation:fly linear infinite}
@keyframes fly{
  0%{transform:translate(110vw,0) rotate(0deg)}
  25%{transform:translate(80vw,5vh) rotate(120deg)}
  50%{transform:translate(52vw,-3vh) rotate(220deg)}
  75%{transform:translate(24vw,7vh) rotate(320deg)}
  100%{transform:translate(-12vw,2vh) rotate(420deg)}}

@media (prefers-reduced-motion:reduce){.sun,.cloud,.leaf-ball,.blade,.lf{animation:none}}

/* ===== TARJETA / FORM ===== */
.card{position:relative;z-index:10;background:rgba(16,9,28,.6);backdrop-filter:blur(14px);
  border:1px solid rgba(255,210,150,.25);border-radius:18px;padding:30px;max-width:540px;width:100%;
  box-shadow:0 24px 70px #0009,0 0 0 1px #0003;max-height:92vh;overflow:auto}
h1{margin:0 0 4px;font-size:26px;background:linear-gradient(90deg,#ffd200,#ff8f5e);-webkit-background-clip:text;background-clip:text;color:transparent}
p.sub{margin:0 0 22px;color:#e9d6c8;font-size:13px}
label{display:block;font-size:13px;margin:14px 0 6px;color:#f0e2d6}
input[type=text],input[type=file]{width:100%;padding:11px;border-radius:10px;border:1px solid #ffffff22;background:rgba(8,5,16,.72);color:#fff;font-size:14px}
.row{display:flex;gap:12px}.row>div{flex:1}
.chk{display:flex;align-items:center;gap:8px;margin-top:14px;font-size:13px;color:#f0e2d6}
button{margin-top:22px;width:100%;padding:14px;border:0;border-radius:12px;font-size:16px;font-weight:700;cursor:pointer;background:linear-gradient(90deg,#f7971e,#ffd200);color:#201500}
button:disabled{opacity:.5;cursor:wait}
.bar{height:10px;background:#0e0c1f;border-radius:6px;overflow:hidden;margin-top:18px;display:none}
.bar>i{display:block;height:100%;width:0;background:linear-gradient(90deg,#f7971e,#ffd200);transition:width .4s}
#msg{font-size:12px;color:#ffd9a8;margin-top:8px;min-height:16px}
#res{margin-top:18px;padding:14px;background:rgba(8,5,16,.72);border-radius:10px;font-size:13px;display:none}
a.dl{display:inline-block;margin-top:10px;color:#ffd200;font-weight:700;text-decoration:none}
</style></head><body>
<div class="scene">
  <div class="sun"></div>
  <span class="cloud c1"></span><span class="cloud c2"></span><span class="cloud c3"></span>
  <!-- cerros lejanos + castillo -->
  <svg class="hills far" viewBox="0 0 1440 600" preserveAspectRatio="xMidYMax slice">
    <path d="M0 360 Q 240 250 480 320 T 960 300 T 1440 330 V600 H0 Z" fill="#5a2f73"/>
    <g fill="#3c2153">
      <rect x="980" y="232" width="120" height="92"/>
      <rect x="966" y="220" width="14" height="20"/><rect x="994" y="214" width="14" height="26"/>
      <rect x="1022" y="210" width="14" height="30"/><rect x="1050" y="214" width="14" height="26"/>
      <rect x="1078" y="220" width="14" height="20"/><rect x="1100" y="232" width="14" height="20"/>
      <rect x="1024" y="168" width="34" height="160"/>
      <polygon points="1024,168 1041,138 1058,168"/>
      <rect x="1036" y="150" width="10" height="14"/>
    </g>
    <path d="M0 420 Q 360 330 720 400 T 1440 410 V600 H0 Z" fill="#34204f"/>
  </svg>
  <!-- cerro cercano -->
  <svg class="hills near" viewBox="0 0 1440 400" preserveAspectRatio="xMidYMax slice">
    <path d="M0 200 Q 400 90 820 180 T 1440 170 V400 H0 Z" fill="#1d1330"/>
  </svg>
  <!-- arboles -->
  <div class="trees">
    <div class="tree t1"><div class="leaf-ball"></div><div class="trunk"></div></div>
    <div class="tree t2"><div class="leaf-ball"></div><div class="trunk"></div></div>
    <div class="tree t3"><div class="leaf-ball"></div><div class="trunk"></div></div>
  </div>
  <!-- pasto -->
  <div class="grass" id="grass"></div>
  <!-- hojas al viento -->
  <div class="leaves" id="leaves"></div>
</div>
<div class=card>
<h1>🎸 GuitarAI</h1><p class=sub>Sube un MP3, una foto y genera tu canción para Clone Hero (chart difícil, voz dinámica, karaoke y video que late).</p>
<form id=f>
<label>Archivo MP3 *</label><input type=file name=mp3 accept=".mp3,audio/*" required>
<label>Foto (carátula + video) — opcional</label><input type=file name=photo accept="image/*">
<div class=row><div><label>Nombre de la canción</label><input type=text name=name placeholder="nombre cancion"></div>
<div><label>Artista</label><input type=text name=artist placeholder="nombre artista"></div></div>
<label class=chk><input type=checkbox name=lyrics value=1 checked> Letra / karaoke automático</label>
<label class=chk><input type=checkbox name=video value=1 checked> Video que late con el ritmo</label>
<button type=submit id=btn>Generar canción</button>
</form>
<div class=bar id=bar><i id=fill></i></div><div id=msg></div>
<div id=res></div>
</div>
<script>
// ===== poblar escena: pasto + hojas al viento =====
(function(){
  const g=document.getElementById('grass');
  if(g){const N=64;for(let i=0;i<N;i++){const b=document.createElement('span');b.className='blade';
    b.style.height=(60+Math.random()*40)+'%';b.style.animationDuration=(2.8+Math.random()*2.2)+'s';
    b.style.animationDelay=(-Math.random()*4)+'s';g.appendChild(b);}}
  const L=document.getElementById('leaves');
  if(L){const N=11;for(let i=0;i<N;i++){const lf=document.createElement('span');lf.className='lf';
    lf.style.top=(Math.random()*70)+'%';const s=.7+Math.random()*1.1;
    lf.style.transform='scale('+s+')';lf.style.animationDuration=(11+Math.random()*12)+'s';
    lf.style.animationDelay=(-Math.random()*20)+'s';
    lf.style.background=Math.random()<.5?'linear-gradient(135deg,#ffcf6e,#e07a3c)':'linear-gradient(135deg,#ffe08a,#d65f48)';
    L.appendChild(lf);}}
})();
const f=document.getElementById('f'),btn=document.getElementById('btn'),bar=document.getElementById('bar'),
fill=document.getElementById('fill'),msg=document.getElementById('msg'),res=document.getElementById('res');
f.onsubmit=async e=>{e.preventDefault();btn.disabled=true;res.style.display='none';bar.style.display='block';
msg.textContent='Subiendo...';fill.style.width='3%';
const r=await fetch('/generar',{method:'POST',body:new FormData(f)});
const j=await r.json();if(j.error){msg.textContent='Error: '+j.error;btn.disabled=false;return;}
const id=j.job;const poll=setInterval(async()=>{
 const s=await(await fetch('/estado/'+id)).json();
 fill.style.width=(s.pct||0)+'%';msg.textContent=s.msg||'';
 if(s.done){clearInterval(poll);btn.disabled=false;
  if(s.error){msg.textContent='Error: '+s.error;return;}
  const n=s.result.notas;
  res.style.display='block';res.innerHTML=
   '✅ <b>Listo.</b><br>BPM: '+s.result.bpm+' · Duración: '+s.result.dur+'s · Palabras letra: '+s.result.palabras+
   '<br>Notas — Experto: '+n.ExpertSingle+' · Hard: '+n.HardSingle+' · Medium: '+n.MediumSingle+' · Easy: '+n.EasySingle+
   '<br><span style=color:#9ad;font-size:11px>'+s.result.folder+'</span>'+
   '<br><a class=dl href="/descargar/'+id+'">⬇ Descargar carpeta (.zip)</a>';
 }},1500);
};
</script></body></html>"""

if __name__=="__main__":
    import os
    port=int(os.environ.get("PORT","5050"))
    print(f"GuitarAI dashboard -> http://127.0.0.1:{port}")
    app.run(host="0.0.0.0",port=port,threaded=True)
