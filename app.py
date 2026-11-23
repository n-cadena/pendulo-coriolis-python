"""Péndulo de Coriolis. Cálculos: Python/NumPy/SciPy DOP853.
La reproducción de los fotogramas YA calculados se efectúa en Plotly en el navegador.
Ejecutar: streamlit run app.py
"""
from dataclasses import replace
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from fisica import Parametros, OMEGA_TIERRA, calcular, magnitudes, precesion_grados_hora
from ecuaciones import GRUPOS

st.set_page_config(page_title="Péndulo de Coriolis | Python", page_icon="🌀", layout="wide", initial_sidebar_state="collapsed")
C = {"fondo":"#0B0F19","panel":"#151C2C","cian":"#00F0FF","rosa":"#EC70FF", "verde":"#44E08A","amarillo":"#FFCC55","rojo":"#FF6175","azul":"#559CFF","texto":"#EAF2FF","gris":"#AAB8CC"}
st.markdown("""<style>
.stApp {background:#0B0F19;color:#EAF2FF}
.block-container {max-width:1250px; padding-top:1rem; padding-bottom:2rem}
h1 {color:#00F0FF!important;font-size:clamp(1.45rem,4vw,2.25rem)!important}
[data-testid="stMetric"] {background:#151C2C;border:1px solid #344054;border-radius:12px;padding:0.65rem}
[data-testid="stMetricValue"] {font-size:clamp(1.05rem,2.5vw,1.5rem)}
[data-testid="stMetricLabel"] {color:#AAB8CC}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] {background:#00cbdc;color:#081521;border:0}
.small {color:#AAB8CC;font-size:.85rem}
</style>""", unsafe_allow_html=True)

def fmt(n, dec=2):
    """Español: 1.234,56; números pequeños permiten decimales suficientes."""
    value = f"{float(n):,.{dec}f}"
    return value.replace(",", "§").replace(".", ",").replace("§", ".")

def numero_fuerza(n):
    n=float(n)
    if n != 0 and abs(n)<0.000001:
        return f"{fmt(n*1e6, 3)} µN"
    if n != 0 and abs(n)<0.001:
        return f"{fmt(n*1e3, 3)} mN"
    return f"{fmt(n, 3)} N"

if "config" not in st.session_state:
    st.session_state.config=vars(Parametros()).copy()
if "revision" not in st.session_state:
    st.session_state.revision=0

st.title("🌀 Péndulo de Coriolis")
st.caption("Física original en **Python + NumPy + SciPy (DOP853)**. El navegador reproduce los fotogramas precalculados, sin recalcular la física durante la animación.")
st.subheader("⚙️ Parámetros y modos de demostración")
st.caption("Los controles están aquí, visibles tanto en celular como en computador. Modifica los valores y pulsa **Aplicar cambios**.")
cfg=st.session_state.config
with st.form("parametros"):
    k1,k2=st.columns(2)
    with k1:
        L=st.slider("Longitud de cuerda (m)",0.5,8.0,float(cfg["longitud"]),0.1)
        ang=st.slider("Ángulo inicial (°)",1,60,int(round(cfg["angulo"])))
        lat=st.slider("Latitud (°)",-90,90,int(round(cfg["latitud"])))
    with k2:
        mass=st.slider("Masa (kg)",0.1,10.0,float(cfg["masa"]),0.1)
        az=st.slider("Azimut (°)",0,360,int(round(cfg["azimut"])),5)
        vel=st.slider("Velocidad inicial (m/s)",0.0,3.0,float(cfg["velocidad"]),0.1)
    a,b=st.columns(2)
    options=["Real ×1","Demo ×100","Demo ×1000"]
    mult={"Real ×1":1.0,"Demo ×100":100.0,"Demo ×1000":1000.0}
    with a:
        modo=st.radio("Rotación terrestre",options,index=[1.0,100.0,1000.0].index(float(cfg["multiplicador"])),horizontal=True)
    with b:
        cor=st.toggle("Activar efecto Coriolis",value=bool(cfg["coriolis"]))
    aplicar=st.form_submit_button("Aplicar cambios y recalcular",type="primary",use_container_width=True)
if aplicar:
    st.session_state.config=dict(longitud=L,masa=mass,angulo=float(ang),azimut=float(az),latitud=float(lat),velocidad=vel,multiplicador=mult[modo],coriolis=cor,gravedad=9.81)
    st.session_state.revision+=1
    st.rerun()

p=Parametros(**st.session_state.config)

@st.cache_data(show_spinner=False,max_entries=8)
def resolver(key):
    parametros=Parametros(**dict(key))
    ts,ys=calcular(parametros, duracion=240.0,dt=0.08)
    _,base=calcular(replace(parametros,coriolis=False),duracion=240.0,dt=0.08)
    return ts,ys,base

with st.spinner("Calculando trayectorias en Python…"):
    ts,ys,base=resolver(tuple(sorted(vars(p).items())))

st.markdown("### 🎞️ Animación 3D y gráficas sincronizadas")
st.caption("Pulsa **▶ Reproducir**: el péndulo 3D, la vista superior y los desplazamientos se dibujan juntos, desde el inicio. Arrastra el péndulo para cambiar la cámara.")

v1,v2,v3,v4,v5=st.columns(5)
with v1: vg=st.checkbox("Gravedad",True)
with v2: vt=st.checkbox("Tensión",True)
with v3: vc=st.checkbox("Coriolis",True)
with v4: vv=st.checkbox("Velocidad",True)
with v5: va=st.checkbox("Aceleración",True)
visible={"Fg":vg,"T":vt,"Fc":vc,"v":vv,"a":va}

def escala_vec(vector,clave):
    denomin={"Fg":p.masa*p.gravedad,"T":p.masa*p.gravedad,
             "Fc":max(2*p.masa*OMEGA_TIERRA*p.multiplicador*np.sqrt(p.gravedad*p.longitud),1e-12),
             "v":np.sqrt(p.gravedad*p.longitud),"a":p.gravedad}[clave]
    return np.asarray(vector)/denomin*(.43*p.longitud)

def coord_flecha(r,vec):
    vec=np.asarray(vec,dtype=float)
    if np.linalg.norm(vec)<1e-12:
        return [None],[None],[None]
    end=r+vec
    unit=vec/np.linalg.norm(vec)
    ref=np.array([0.,0.,1.]) if abs(unit[2])<.93 else np.array([0.,1.,0.])
    perpendicular=np.cross(unit,ref)
    perpendicular/=np.linalg.norm(perpendicular)
    p1=end-.27*vec+.22*np.linalg.norm(vec)*perpendicular
    p2=end-.27*vec-.22*np.linalg.norm(vec)*perpendicular
    points=[r,end,None,end,p1,None,end,p2]
    return ([None if q is None else q[0] for q in points],
            [None if q is None else q[1] for q in points],
            [None if q is None else q[2] for q in points])

names={"Fg":"Gravedad","T":"Tensión","Fc":"Coriolis","v":"Velocidad","a":"Aceleración"}
colors={"Fg":C["rojo"],"T":C["azul"],"Fc":C["rosa"],"v":C["verde"],"a":C["amarillo"]}
keys=[key for key in names if visible[key]]

# Precalculamos unos 121 fotogramas, separados por 2 s simulados.
# Los datos que se visualizan se obtuvieron mediante integración en Python.
frame_indices=np.unique(np.r_[np.arange(0,len(ts),25),len(ts)-1]).astype(int)

def datos_trazas(idx):
    r=ys[idx,:3]
    # Rastro completo hasta el fotograma actual, pero muestreado para aligerar Plotly.
    hist=ys[:idx+1:max(1,(idx+1)//180),:3]
    result=[go.Scatter3d(x=hist[:,0],y=hist[:,1],z=hist[:,2],mode="lines",line=dict(color=C["cian"],width=3),name="Trayectoria",hoverinfo="skip"),
            go.Scatter3d(x=[0,r[0]],y=[0,r[1]],z=[0,r[2]],mode="lines",line=dict(color=C["texto"],width=6),name="Cuerda",hoverinfo="skip"),
            go.Scatter3d(x=[r[0]],y=[r[1]],z=[r[2]],mode="markers",marker=dict(color=C["cian"],size=9),name="Bola",hoverinfo="skip")]
    q=magnitudes(ys[idx],p)
    for key in keys:
        xx,yy,zz=coord_flecha(r,escala_vec(q[key],key))
        result.append(go.Scatter3d(x=xx,y=yy,z=zz,mode="lines",name=names[key],line=dict(color=colors[key],width=5),hoverinfo="skip"))
    return result

# Un solo gráfico de Plotly para sincronizar las tres vistas con el MISMO botón Play.
# Todos los puntos vienen de las trayectorias calculadas previamente en Python.
fig=make_subplots(
    rows=3, cols=1,
    specs=[[{"type":"scene"}],[{"type":"xy"}],[{"type":"xy"}]],
    row_heights=[0.50,0.25,0.25], vertical_spacing=0.065,
    subplot_titles=("Péndulo tridimensional", "Vista superior (trayectoria)", "Movimiento horizontal")
)
for traza in datos_trazas(0):
    fig.add_trace(traza,row=1,col=1)

# Cinco trazas adicionales en los dos gráficos 2D. En t=0 solo muestran un punto.
fig.add_trace(go.Scatter(x=[base[0,0]],y=[base[0,1]],mode="lines",name="Sin Coriolis",line=dict(color=C["gris"],dash="dash",width=2),legendgroup="sin"),row=2,col=1)
fig.add_trace(go.Scatter(x=[ys[0,0]],y=[ys[0,1]],mode="lines",name="Trayectoria actual",line=dict(color=C["cian"],width=2),legendgroup="actual"),row=2,col=1)
fig.add_trace(go.Scatter(x=[ys[0,0]],y=[ys[0,1]],mode="markers",name="Posición actual",marker=dict(color=C["amarillo"],size=9),showlegend=False),row=2,col=1)
fig.add_trace(go.Scatter(x=[ts[0]],y=[ys[0,0]],mode="lines",name="Este x",line=dict(color=C["cian"],width=2),legendgroup="este"),row=3,col=1)
fig.add_trace(go.Scatter(x=[ts[0]],y=[ys[0,1]],mode="lines",name="Norte y",line=dict(color=C["rosa"],width=2),legendgroup="norte"),row=3,col=1)

def graficas_hasta(idx):
    """Devuelve los puntos recorridos HASTA ese instante, nunca la curva futura."""
    muestras=np.unique(np.linspace(0,int(idx),min(int(idx)+1,230),dtype=int))
    return [
        go.Scatter(x=base[muestras,0],y=base[muestras,1]),
        go.Scatter(x=ys[muestras,0],y=ys[muestras,1]),
        go.Scatter(x=[ys[idx,0]],y=[ys[idx,1]]),
        go.Scatter(x=ts[muestras],y=ys[muestras,0]),
        go.Scatter(x=ts[muestras],y=ys[muestras,1]),
    ]

n_trazas=3+len(keys)+5
fig.frames=[
    go.Frame(name=str(int(idx)),
             data=datos_trazas(int(idx))+graficas_hasta(int(idx)),
             traces=list(range(n_trazas)))
    for idx in frame_indices
]
steps=[dict(method="animate",label=fmt(ts[idx],0),args=[[str(int(idx))],
    {"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}])
    for idx in frame_indices]

fig.update_layout(
    height=1090,margin=dict(l=4,r=4,t=42,b=25),
    paper_bgcolor=C["fondo"],plot_bgcolor=C["panel"],font=dict(color=C["texto"],size=11),
    showlegend=True,legend=dict(orientation="h",y=1.025,x=0,font=dict(size=10)),
    scene=dict(xaxis=dict(title="Este x (m)",range=[-1.15*p.longitud,1.15*p.longitud],backgroundcolor=C["panel"]),
               yaxis=dict(title="Norte y (m)",range=[-1.15*p.longitud,1.15*p.longitud],backgroundcolor=C["panel"]),
               zaxis=dict(title="Arriba z (m)",range=[-1.23*p.longitud,.2*p.longitud],backgroundcolor=C["panel"]),
               aspectmode="manual",aspectratio=dict(x=1,y=1,z=.75),camera=dict(eye=dict(x=1.4,y=1.5,z=1.0))),
    updatemenus=[dict(type="buttons",direction="left",showactive=False,x=0,y=-.055,xanchor="left",yanchor="top",buttons=[
        dict(label="▶ Reproducir",method="animate",args=[None,{"fromcurrent":True,"mode":"immediate","frame":{"duration":115,"redraw":True},"transition":{"duration":0}}]),
        dict(label="⏸ Pausar",method="animate",args=[[None],{"mode":"immediate","frame":{"duration":0,"redraw":False},"transition":{"duration":0}}]),
        dict(label="↺ Reiniciar",method="animate",args=[[str(int(frame_indices[0]))],{"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}])])],
    sliders=[dict(active=0,x=.08,y=-.075,len=.86,steps=steps,
             currentvalue={"prefix":"Tiempo simulado ≈ ","suffix":" s","font":{"color":C["texto"]}},
             tickcolor=C["gris"],font={"color":C["gris"],"size":9},minorticklen=0)],
    uirevision=f"config-{st.session_state.revision}"
)
fig.update_xaxes(title_text="Este, x (m)",row=2,col=1,gridcolor="#344054",range=[-1.12*p.longitud,1.12*p.longitud])
fig.update_yaxes(title_text="Norte, y (m)",row=2,col=1,gridcolor="#344054",range=[-1.12*p.longitud,1.12*p.longitud],scaleanchor="x",scaleratio=1)
fig.update_xaxes(title_text="Tiempo simulado (s)",row=3,col=1,gridcolor="#344054",range=[0,240])
fig.update_yaxes(title_text="Desplazamiento (m)",row=3,col=1,gridcolor="#344054",range=[-.85*p.longitud,.85*p.longitud])
st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False,"scrollZoom":False},key="grafica_principal")
st.caption("Las flechas utilizan escalas gráficas independientes; la de Coriolis está ampliada para poder verse. La animación utiliza fotogramas muestreados de la trayectoria calculada en Python.")

# Medidas correspondientes a un instante elegido. Esto NO reintegra: usa caché.
st.markdown("### 📏 Valores físicos en un instante")
instante=st.slider("Selecciona el tiempo para consultar las magnitudes (s)",0.0,240.0,0.0,1.0)
i=int(round(instante/.08))
i=min(i,len(ts)-1)
magn=magnitudes(ys[i],p)
a1,a2,a3=st.columns(3)
a1.metric("Rapidez",f"{fmt(np.linalg.norm(magn['v']),3)} m/s")
a2.metric("Aceleración",f"{fmt(np.linalg.norm(magn['a']),3)} m/s²")
a3.metric("Fuerza de Coriolis",numero_fuerza(np.linalg.norm(magn['Fc'])))
b1,b2,b3=st.columns(3)
b1.metric("Fuerza de gravedad",f"{fmt(np.linalg.norm(magn['Fg']),2)} N")
b2.metric("Tensión de la cuerda",f"{fmt(magn['tension'],3)} N")
b3.metric("Energía total",f"{fmt(magn['E'],3)} J")
st.caption(f"Aceleración de gravedad: **{fmt(p.gravedad,2)} m/s²** · Precesión aproximada: **{fmt(precesion_grados_hora(p),2)} °/h**" if p.coriolis else f"Aceleración de gravedad: **{fmt(p.gravedad,2)} m/s²** · Precesión: no aplicable (Coriolis desactivado)")
st.info("Las tres vistas gráficas avanzan juntas. El control de esta sección permite consultar los valores físicos de cualquier instante, independientemente de la animación.",icon="ℹ️")

with st.expander("∑ Ver las ecuaciones y su significado"):
    for name,items in GRUPOS:
        st.subheader(name)
        for title,formula,description in items:
            st.markdown(f"**{title}**")
            st.latex(formula)
            st.caption(description)
    st.caption("Sin rozamiento; gravedad efectiva constante; precesión aproximada para oscilaciones pequeñas.")
st.caption("Modelo ideal · Sistema de ejes Este–Norte–Arriba · Python + SciPy + Streamlit + Plotly")

