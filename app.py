"""Péndulo de Coriolis: física en Python + reproducción interactiva en navegador.
Ejecutar: streamlit run app.py
"""
from dataclasses import replace
import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from fisica import Parametros, OMEGA_TIERRA, calcular, magnitudes, precesion_grados_hora
from ecuaciones import GRUPOS

st.set_page_config(page_title="Péndulo de Coriolis | Python", page_icon="🌀", layout="wide", initial_sidebar_state="expanded")
st.markdown('''<style>
.stApp {background:#0B0F19;color:#EAF2FF}
.block-container {max-width:1280px;padding-top:1rem}
h1 {color:#00F0FF!important}
[data-testid="stSidebar"] {background:#151C2C}
</style>''',unsafe_allow_html=True)

if 'config' not in st.session_state:
    st.session_state.config=vars(Parametros()).copy()
cfg=st.session_state.config

with st.sidebar:
    st.header('⚙️ Parámetros del péndulo')
    st.caption('Cambia los valores y pulsa «Aplicar cambios». Puedes ocultar este panel con la flecha superior.')
    with st.form('params'):
        L=st.slider('Longitud de cuerda (m)',0.5,8.0,float(cfg['longitud']),0.1)
        mass=st.slider('Masa (kg)',0.1,10.0,float(cfg['masa']),0.1)
        ang=st.slider('Ángulo inicial (°)',1,60,int(cfg['angulo']))
        az=st.slider('Azimut (°)',0,360,int(cfg['azimut']),5)
        lat=st.slider('Latitud (°)',-90,90,int(cfg['latitud']))
        vel=st.slider('Velocidad inicial (m/s)',0.0,3.0,float(cfg['velocidad']),0.1)
        mode_values={'Real ×1':1.0,'Demo ×100':100.0,'Demo ×1000':1000.0}
        mode=st.radio('Rotación terrestre',list(mode_values),index=[1.0,100.0,1000.0].index(float(cfg['multiplicador'])))
        coriolis=st.toggle('Activar efecto Coriolis',value=bool(cfg['coriolis']))
        submit=st.form_submit_button('Aplicar cambios',type='primary',use_container_width=True)
    if submit:
        st.session_state.config=dict(longitud=L,masa=mass,angulo=float(ang),azimut=float(az),latitud=float(lat),velocidad=vel,multiplicador=mode_values[mode],coriolis=coriolis,gravedad=9.81)
        st.rerun()

p=Parametros(**st.session_state.config)
st.title('🌀 Péndulo de Coriolis')
st.caption('Cálculos en **Python + NumPy + SciPy (DOP853)**. Usa el panel lateral izquierdo para cambiar los parámetros. Animación y valores físicos sincronizados en tu navegador.')

@st.cache_data(show_spinner=False,max_entries=6)
def resolver(pares):
    par=Parametros(**dict(pares))
    # Tiempo de cálculo interno, NO un límite del botón de reproducción.
    ts, ys=calcular(par,duracion=600.0,dt=0.16)
    _, sin=calcular(replace(par,coriolis=False),duracion=600.0,dt=0.16)
    return ts,ys,sin

with st.spinner('Python está calculando la trayectoria y las fuerzas…'):
    ts,ys,sin=resolver(tuple(sorted(vars(p).items())))

# Reducir transferencia de datos: geometría y magnitudes se calculan en Python.
# Todos los números del panel móvil salen de magnitudes(), no de JS.
idx=np.unique(np.r_[np.arange(0,len(ts),2),len(ts)-1]).astype(int)
records=[]
for i in idx:
    q=magnitudes(ys[i],p)
    records.append([
        round(float(ts[i]),3),
        *[round(float(z),7) for z in ys[i,:3]],
        *[round(float(z),7) for z in sin[i,:2]],
        *[round(float(np.linalg.norm(q[k])),9) for k in ('v','a','Fg','Fc')],
        round(float(q['tension']),8),round(float(q['E']),8),
        *[round(float(z),7) for k in ('Fg','T','Fc','v','a') for z in q[k]],
    ])

payload=json.dumps({'frames':records,'length':p.longitud,'grav':p.gravedad,
                   'precession':precesion_grados_hora(p),'coriolis':p.coriolis,
                   'rotation':p.multiplicador},separators=(',',':'))
# Index mapping: [t,x,y,z,x_sin,y_sin, |v|,|a|,|Fg|,|Fc|,T,E, (Fg xyz),(T xyz),(Fc xyz),(v xyz),(a xyz)]
html=r'''<!doctype html><html lang="es"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{box-sizing:border-box}body{font-family:system-ui,Arial,sans-serif;background:#0B0F19;color:#EAF2FF;margin:0}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0 10px}
button,select{background:#18283B;color:#EAF2FF;border:1px solid #496080;border-radius:8px;padding:9px 12px;font-size:14px;cursor:pointer}
button:hover{background:#22415D}.time{font-variant-numeric:tabular-nums;color:#ffcc55;font-weight:bold}
.grid{display:grid;grid-template-columns:1.25fr 1fr;gap:9px}.panel{background:#151C2C;border:1px solid #283D59;border-radius:12px;overflow:hidden;padding:4px}
.plot{width:100%;height:280px}.plot3{height:570px}.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin-top:10px}
.metric{background:#151C2C;border:1px solid #344054;border-radius:10px;padding:12px;min-width:0}.metric label{display:block;color:#AAB8CC;font-size:12px}.metric strong{display:block;font-size:clamp(14px,2vw,21px);margin-top:5px;font-variant-numeric:tabular-nums}
.note{color:#AAB8CC;font-size:12px;margin:8px 1px 14px}.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:12px}.legend label{white-space:nowrap}
@media(max-width:730px){.grid{grid-template-columns:1fr}.plot3{height:400px}.plot{height:250px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body>
<div class="controls">
<button id="play">▶ Reproducir</button><button id="restart">↺ Reiniciar</button>
<label>Velocidad <select id="speed"><option value="1">×1</option><option value="5">×5</option><option value="10" selected>×10</option><option value="25">×25</option><option value="50">×50</option><option value="100">×100</option></select></label>
<span class="time" id="clock">t = 0,0 s</span>
</div>
<div class="legend" id="vectors"><label><input type="checkbox" data-vector="0" checked> 🔴 Gravedad</label><label><input type="checkbox" data-vector="1" checked> 🔵 Tensión</label><label><input type="checkbox" data-vector="2" checked> 🟣 Coriolis</label><label><input type="checkbox" data-vector="3" checked> 🟢 Velocidad</label><label><input type="checkbox" data-vector="4" checked> 🟡 Aceleración</label></div>
<div class="grid"><div class="panel"><div id="plot3d" class="plot3"></div></div><div class="panel"><div id="top" class="plot"></div><div id="horizontal" class="plot"></div></div></div>
<div class="metrics">
<div class="metric"><label>Rapidez</label><strong id="velocity">—</strong></div>
<div class="metric"><label>Aceleración</label><strong id="accel">—</strong></div>
<div class="metric"><label>Fuerza de Coriolis</label><strong id="cforce">—</strong></div>
<div class="metric"><label>Fuerza de gravedad</label><strong id="weight">—</strong></div>
<div class="metric"><label>Tensión de la cuerda</label><strong id="tension">—</strong></div>
<div class="metric"><label>Energía total</label><strong id="energy">—</strong></div>
</div>
<div class="note" id="extra"></div>
<div class="note">El botón Reproducir no se detiene solo: cuando termina el tramo precalculado, comienza otro ciclo desde el inicio. La trayectoria del siguiente ciclo no es una continuación física de la anterior. Las flechas se amplifican por separado para visualizarlas, especialmente Coriolis. Puedes girar y acercar el gráfico 3D con el dedo.</div>
<script>
const D=__PAYLOAD__;
const rows=D.frames, L=D.length, maxIdx=rows.length-1;
const color={bg:'#0B0F19',panel:'#151C2C',grid:'#344054',fg:'#FF6175',t:'#559CFF',fc:'#EC70FF',v:'#44E08A',a:'#FFCC55',pos:'#00F0FF',txt:'#EAF2FF',muted:'#AAB8CC'};
let current=0, running=false, prev=0, simT=0, lastDraw=0;
const g=(id)=>document.getElementById(id);
const fmt=(v,n=2)=>Number(v).toLocaleString('es-CO',{minimumFractionDigits:n,maximumFractionDigits:n});
function fN(v){if(v!==0 && Math.abs(v)<1e-6)return fmt(v*1e6,3)+' µN';if(v!==0 && Math.abs(v)<1e-3)return fmt(v*1e3,3)+' mN';return fmt(v,3)+' N'}
const common={paper_bgcolor:color.panel,plot_bgcolor:color.panel,font:{color:color.txt,size:11},margin:{l:43,r:12,t:36,b:40},showlegend:false};
function tr3(x=[],y=[],z=[],c=color.pos,w=3){return {type:'scatter3d',mode:'lines',x,y,z,line:{color:c,width:w},hoverinfo:'skip',showlegend:false}}
let data3=[tr3(),tr3([],[],[],color.txt,5),{type:'scatter3d',mode:'markers',x:[],y:[],z:[],marker:{size:8,color:color.pos},hoverinfo:'skip'}];
const vs=[color.fg,color.t,color.fc,color.v,color.a];for(const c of vs)data3.push(tr3([],[],[],c,5));
const scene={xaxis:{title:'Este x (m)',range:[-1.15*L,1.15*L],backgroundcolor:color.panel},yaxis:{title:'Norte y (m)',range:[-1.15*L,1.15*L],backgroundcolor:color.panel},zaxis:{title:'Arriba z (m)',range:[-1.23*L,.2*L],backgroundcolor:color.panel},aspectmode:'manual',aspectratio:{x:1,y:1,z:.75},camera:{eye:{x:1.4,y:1.5,z:1.0}}};
const topData=[{type:'scatter',mode:'lines',x:[],y:[],line:{color:color.muted,dash:'dash',width:1.5},name:'Sin Coriolis'}, {type:'scatter',mode:'lines',x:[],y:[],line:{color:color.pos,width:2},name:'Actual'}, {type:'scatter',mode:'markers',x:[],y:[],marker:{color:color.a,size:8},showlegend:false}];
const horData=[{type:'scatter',mode:'lines',x:[],y:[],line:{color:color.pos,width:2},name:'Este (x)'},{type:'scatter',mode:'lines',x:[],y:[],line:{color:color.fc,width:2},name:'Norte (y)'}];
let ready=false;
async function init(){if(!window.Plotly){g('extra').textContent='No se pudo cargar Plotly. Comprueba tu conexión a internet y recarga la página.';return;}
await Plotly.newPlot('plot3d',data3,{...common,margin:{l:0,r:0,t:28,b:0},title:{text:'Péndulo 3D',font:{size:15}},scene}, {responsive:true,displaylogo:false,scrollZoom:false});
await Plotly.newPlot('top',topData,{...common,title:{text:'Vista superior',font:{size:14}},xaxis:{title:'Este (m)',range:[-1.12*L,1.12*L],gridcolor:color.grid},yaxis:{title:'Norte (m)',range:[-1.12*L,1.12*L],scaleanchor:'x',gridcolor:color.grid}}, {responsive:true,displaylogo:false});
await Plotly.newPlot('horizontal',horData,{...common,title:{text:'Movimiento horizontal',font:{size:14}},xaxis:{title:'Tiempo (s)',range:[0,60],gridcolor:color.grid},yaxis:{title:'Posición (m)',range:[-.85*L,.85*L],gridcolor:color.grid}}, {responsive:true,displaylogo:false});
ready=true;render(0);requestAnimationFrame(tick);
}
function arrow(r,v,scale){let q=v.map(z=>z*scale),norm=Math.hypot(...q);if(norm<1e-11)return {x:[null],y:[null],z:[null]};let e=r.map((z,i)=>z+q[i]);let u=q.map(z=>z/norm),ref=Math.abs(u[2])<.93?[0,0,1]:[0,1,0];let cross=[u[1]*ref[2]-u[2]*ref[1],u[2]*ref[0]-u[0]*ref[2],u[0]*ref[1]-u[1]*ref[0]];let cn=Math.hypot(...cross);let p1=e.map((z,i)=>z-.27*q[i]+.22*norm*cross[i]/cn),p2=e.map((z,i)=>z-.27*q[i]-.22*norm*cross[i]/cn);let pts=[r,e,null,e,p1,null,e,p2];return {x:pts.map(z=>z?z[0]:null),y:pts.map(z=>z?z[1]:null),z:pts.map(z=>z?z[2]:null)}}
function render(i){if(!ready)return;let d=rows[i],r=d.slice(1,4),stride=Math.max(1,Math.floor((i+1)/160)),hist=[];for(let j=Math.max(0,i-5500);j<=i;j+=stride)hist.push(rows[j]);if(hist.at(-1)!==d)hist.push(d);
const x3=[hist.map(q=>q[1]),[0,r[0]],[r[0]]],y3=[hist.map(q=>q[2]),[0,r[1]],[r[1]]],z3=[hist.map(q=>q[3]),[0,r[2]],[r[2]]];
const denom=[D.grav?d[8]:1,D.grav?d[8]:1,Math.max(2*d[8]/D.grav*7.2921159e-5*D.rotation*Math.sqrt(D.grav*L),1e-12),Math.sqrt(D.grav*L),D.grav];
for(let k=0;k<5;k++){const checked=document.querySelector('[data-vector="'+k+'"]').checked;const q=checked?arrow(r,d.slice(12+k*3,15+k*3),.43*L/denom[k]):{x:[null],y:[null],z:[null]};x3.push(q.x);y3.push(q.y);z3.push(q.z)}
Plotly.restyle('plot3d',{x:x3,y:y3,z:z3},[0,1,2,3,4,5,6,7]);
const hidx=[];for(let j=Math.max(0,i-1800);j<=i;j+=Math.max(1,Math.floor((i+1)/220)))hidx.push(j);if(hidx.at(-1)!==i)hidx.push(i);
const h=hidx.map(j=>rows[j]);
Plotly.restyle('top',{x:[h.map(z=>z[4]),h.map(z=>z[1]),[d[1]]],y:[h.map(z=>z[5]),h.map(z=>z[2]),[d[2]]]},[0,1,2]);
Plotly.restyle('horizontal',{x:[h.map(z=>z[0]),h.map(z=>z[0])],y:[h.map(z=>z[1]),h.map(z=>z[2])]},[0,1]);
Plotly.relayout('horizontal',{'xaxis.range':[Math.max(0,d[0]-60),Math.max(60,d[0])]});
g('velocity').textContent=fmt(d[6],3)+' m/s';g('accel').textContent=fmt(d[7],3)+' m/s²';g('weight').textContent=fN(d[8]);g('cforce').textContent=fN(d[9]);g('tension').textContent=fN(d[10]);g('energy').textContent=fmt(d[11],3)+' J';g('clock').textContent='t = '+fmt(d[0],1)+' s';
}
function tick(now){if(running){if(prev){simT+=(now-prev)/1000*Number(g('speed').value);if(simT>rows[maxIdx][0])simT=simT%rows[maxIdx][0];}if(now-lastDraw>=85){let n=Math.round(simT/(rows[maxIdx][0]/maxIdx));n=Math.max(0,Math.min(maxIdx,n));if(n!==current){current=n;render(n)}lastDraw=now;}}prev=now;requestAnimationFrame(tick)}
g('play').onclick=()=>{running=!running;prev=0;g('play').textContent=running?'⏸ Pausar':'▶ Reproducir'};
g('restart').onclick=()=>{running=false;simT=0;prev=0;current=0;g('play').textContent='▶ Reproducir';render(0)};
g('extra').textContent='Gravedad: '+fmt(D.grav,2)+' m/s² · Precesión aproximada: '+(D.coriolis?fmt(D.precession,2)+' °/h':'no aplicable (Coriolis desactivado)')+' · Rotación terrestre ×'+fmt(D.rotation,0);
init();
</script></body></html>'''
html=html.replace('__PAYLOAD__',payload)
components.html(html,height=1300,scrolling=True)
with st.expander('∑ Ecuaciones utilizadas y significado'):
    for name,items in GRUPOS:
        st.subheader(name)
        for title,formula,description in items:
            st.markdown(f'**{title}**')
            st.latex(formula)
            st.caption(description)
st.caption('Modelo ideal sin rozamiento · Python + SciPy para los cálculos físicos; navegador para la visualización.')
