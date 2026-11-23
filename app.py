"""Péndulo de Coriolis — aplicación web en PYTHON con Streamlit.

Física original: fisica.py (NumPy + SciPy, método DOP853).
Interfaz web: Streamlit. Gráficos: Plotly (presentación, no física).

Ejecutar: streamlit run app.py
"""
from dataclasses import replace
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fisica import Parametros, OMEGA_TIERRA, calcular, magnitudes, precesion_grados_hora
from ecuaciones import GRUPOS

# Configuración visual: el núcleo físico NO depende de esto.
st.set_page_config(page_title="Péndulo de Coriolis | Python", page_icon="🌀", layout="wide", initial_sidebar_state="collapsed")
C = {
    "fondo": "#0B0F19", "panel": "#151C2C", "cian": "#00F0FF",
    "rosa": "#EC70FF", "verde": "#44E08A", "amarillo": "#FFCC55",
    "rojo": "#FF6175", "azul": "#559CFF", "blanco": "#EAF2FF", "gris": "#AAB8CC",
}
DT = 0.08
DURACION = 240.0

st.markdown("""
<style>
.stApp {background: #0B0F19; color: #EAF2FF;}
.block-container {padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1120px;}
h1,h2,h3 {letter-spacing: -0.025em;}
h1 {color: #00F0FF !important; font-size: clamp(1.6rem, 4vw, 2.4rem) !important;}
[data-testid="stMetric"] {background: #151C2C; border: 1px solid #29374E;
    padding: .7rem .8rem; border-radius: 12px;}
[data-testid="stMetricLabel"] {color: #AAB8CC;}
[data-testid="stMetricValue"] {font-size: clamp(1rem, 3vw, 1.5rem);}
[data-testid="stSidebar"] {background: #101726;}
.stButton > button[kind="primary"] {background: #00CFE0; color: #061321; border: 0;}
.intro {color:#AAB8CC; font-size:0.94rem; margin-top:-0.5rem; margin-bottom:1.0rem;}
.small-note {color:#AAB8CC; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)


def p_desde_configuracion(datos):
    """Construye los parámetros del modelo físico, sin lógica visual."""
    return Parametros(**datos)


if "configuracion" not in st.session_state:
    st.session_state.configuracion = vars(Parametros()).copy()
if "seek" not in st.session_state:
    st.session_state.seek = 0.0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "vel_reproduccion" not in st.session_state:
    st.session_state.vel_reproduccion = 10

st.title("🌀 Péndulo de Coriolis")
st.markdown('<div class="intro">Laboratorio interactivo · Física calculada con <b>Python, NumPy y SciPy</b> · Visualización táctil en Plotly</div>', unsafe_allow_html=True)

# Las modificaciones se envían al soltar/aplicar el formulario: el modelo
# se recalcula una sola vez, y el resultado se guarda en caché.
with st.sidebar:
    st.header("⚙️ Parámetros del péndulo")
    st.caption("Ajusta los valores y pulsa **Aplicar parámetros**.")
    cfg = st.session_state.configuracion
    with st.form("ajustes", clear_on_submit=False):
        longitud = st.slider("Longitud de cuerda (m)", 0.5, 8.0, float(cfg["longitud"]), 0.1)
        masa = st.slider("Masa (kg)", 0.1, 10.0, float(cfg["masa"]), 0.1)
        angulo = st.slider("Inclinación inicial (°)", 1.0, 60.0, float(cfg["angulo"]), 1.0)
        azimut = st.slider("Azimut inicial (°)", 0.0, 360.0, float(cfg["azimut"]), 5.0)
        latitud = st.slider("Latitud geográfica (°)", -90.0, 90.0, float(cfg["latitud"]), 1.0)
        velocidad = st.slider("Velocidad inicial (m/s)", 0.0, 3.0, float(cfg["velocidad"]), 0.1)
        coriolis = st.checkbox("Activar fuerza de Coriolis", bool(cfg["coriolis"]))
        modos = {"Real ×1": 1.0, "Demo ×100": 100.0, "Demo ×1000": 1000.0}
        seleccionado = list(modos.keys())[list(modos.values()).index(float(cfg["multiplicador"]))]
        modo = st.radio("Rotación terrestre", options=list(modos), index=list(modos).index(seleccionado))
        st.caption("Los modos Demo modifican artificialmente la rotación, no representan la Tierra real.")
        enviar = st.form_submit_button("Aplicar parámetros", type="primary", use_container_width=True)
        if enviar:
            st.session_state.configuracion = dict(longitud=longitud, masa=masa, angulo=angulo,
                                                  azimut=azimut, latitud=latitud, velocidad=velocidad,
                                                  multiplicador=modos[modo], coriolis=coriolis, gravedad=9.81)
            st.session_state.seek = 0.0
            st.session_state.playing = False
    st.divider()
    st.header("🎯 Vectores visibles")
    ver = {
        "Fg": st.checkbox("Gravedad (rojo)", value=True),
        "T": st.checkbox("Tensión (azul)", value=True),
        "Fc": st.checkbox("Coriolis (rosa)", value=True),
        "v": st.checkbox("Velocidad (verde)", value=True),
        "a": st.checkbox("Aceleración (amarillo)", value=True),
    }
    st.caption("Cada flecha usa su propia escala visual. La de Coriolis se amplifica para poder verla.")
    st.divider()
    st.caption("Modelo ideal sin rozamiento. Gravedad efectiva constante. Ejes Este–Norte–Arriba.")


@st.cache_data(show_spinner=False, max_entries=12)
def resolver_dos_trayectorias(tupla_parametros):
    """Los dos cálculos de trayectorias se ejecutan REALMENTE en Python.

    st.cache_data evita repetir la integración de SciPy en cada fotograma.
    """
    p = p_desde_configuracion(dict(tupla_parametros))
    tiempos, estados = calcular(p, duracion=DURACION, dt=DT)
    _, estados_sin = calcular(replace(p, coriolis=False), duracion=DURACION, dt=DT)
    return tiempos, estados, estados_sin


p = p_desde_configuracion(st.session_state.configuracion)
with st.spinner("Calculando trayectoria física en Python (SciPy DOP853)…"):
    ts, ys, ys_sin = resolver_dos_trayectorias(tuple(sorted(vars(p).items())))


def formato_3d():
    """Estilo uniforme de las escenas 3D."""
    return dict(
        xaxis=dict(title="Este, x (m)", range=[-1.15*p.longitud, 1.15*p.longitud],
                   gridcolor="#334258", zerolinecolor="#62738A", backgroundcolor=C["panel"]),
        yaxis=dict(title="Norte, y (m)", range=[-1.15*p.longitud, 1.15*p.longitud],
                   gridcolor="#334258", zerolinecolor="#62738A", backgroundcolor=C["panel"]),
        zaxis=dict(title="Arriba, z (m)", range=[-1.23*p.longitud, .20*p.longitud],
                   gridcolor="#334258", zerolinecolor="#62738A", backgroundcolor=C["panel"]),
        aspectmode="manual", aspectratio=dict(x=1, y=1, z=.75),
        camera=dict(eye=dict(x=1.45, y=1.55, z=.95)),
    )


def flecha_3d(fig, r, vector, denom, nombre, color):
    """Dibuja flechas. La escala es solo VISUAL y no cambia las fuerzas."""
    norma = np.linalg.norm(vector)
    if norma < 1e-12 or not np.isfinite(norma):
        return
    final = r + (vector / denom) * (.43 * p.longitud)
    eje = final - r
    n = np.linalg.norm(eje)
    if n < 1e-10:
        return
    u = eje / n
    ref = np.array([0., 0., 1.]) if abs(np.dot(u, [0., 0., 1.])) < .93 else np.array([0., 1., 0.])
    v = np.cross(u, ref)
    v /= np.linalg.norm(v)
    p1 = final - .26 * eje + .19 * n * v
    p2 = final - .26 * eje - .19 * n * v
    fig.add_trace(go.Scatter3d(
        x=[r[0],final[0],None,final[0],p1[0],None,final[0],p2[0]],
        y=[r[1],final[1],None,final[1],p1[1],None,final[1],p2[1]],
        z=[r[2],final[2],None,final[2],p1[2],None,final[2],p2[2]],
        mode="lines", line=dict(color=color,width=6), name=nombre,
        hovertemplate=f"{nombre}: {norma:.5g}<extra></extra>",
    ))


def grafica_3d(i, valores, ver):
    r = ys[i, :3]
    traza = ys[:i+1:max(1, (i+1)//360), :3]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=traza[:,0], y=traza[:,1], z=traza[:,2],
                               mode="lines",line=dict(color=C["cian"], width=3),
                               name="Rastro 3D",opacity=.6))
    fig.add_trace(go.Scatter3d(x=[0,r[0]],y=[0,r[1]],z=[0,r[2]],
                               mode="lines",line=dict(color="#EAF2FF",width=7),name="Cuerda"))
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode="markers",name="Soporte",
                               marker=dict(color=C["gris"],size=5)))
    fig.add_trace(go.Scatter3d(x=[r[0]],y=[r[1]],z=[r[2]],mode="markers",name="Bola",
                               marker=dict(color=C["cian"],size=11)))
    d = {"Fg":p.masa*p.gravedad,"T":p.masa*p.gravedad,
         "Fc":max(2*p.masa*OMEGA_TIERRA*p.multiplicador*np.sqrt(p.gravedad*p.longitud),1e-12),
         "v":np.sqrt(p.gravedad*p.longitud),"a":p.gravedad}
    etiquetas = {"Fg":"Gravedad (N)", "T":"Tensión (N)", "Fc":"Coriolis (N)",
                 "v":"Velocidad (m/s)","a":"Aceleración (m/s²)"}
    colores = {"Fg":C["rojo"],"T":C["azul"],"Fc":C["rosa"],
               "v":C["verde"],"a":C["amarillo"]}
    for clave in ("Fg", "T", "Fc", "v", "a"):
        if ver[clave]:
            flecha_3d(fig,r,valores[clave],d[clave],etiquetas[clave],colores[clave])
    fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=5),height=430,paper_bgcolor=C["fondo"],
        font=dict(color=C["blanco"],size=11),
        scene=formato_3d(),
        legend=dict(orientation="h",y=-.05,x=.0,font=dict(size=10)),
        uirevision="camara-pendulo",
        showlegend=True,
    )
    return fig


def grafica_plano(i):
    fig = go.Figure()
    salto = max(1, len(ts)//1800)
    fig.add_trace(go.Scatter(x=ys_sin[::salto,0],y=ys_sin[::salto,1], mode="lines",
                             name="Sin Coriolis", line=dict(color=C["gris"],dash="dash",width=1.7)))
    fig.add_trace(go.Scatter(x=ys[:i+1:salto,0],y=ys[:i+1:salto,1], mode="lines",
                             name="Con configuración", line=dict(color=C["cian"],width=2.3)))
    fig.add_trace(go.Scatter(x=[ys[i,0]],y=[ys[i,1]],mode="markers",name="Bola",
                             marker=dict(color=C["amarillo"],size=10)))
    fig.update_layout(xaxis_title="Este x (m)", yaxis_title="Norte y (m)")
    fig.update_yaxes(scaleanchor="x",scaleratio=1)
    return terminar_grafica(fig)


def grafica_desplazamiento(i):
    fig = go.Figure()
    start = max(0, int(ts[i]/DT)-int(60/DT))
    salto = max(1,(i-start)//500)
    idx = slice(start,i+1,salto)
    fig.add_trace(go.Scatter(x=ts[idx],y=ys[idx,0],name="Este (x)",line=dict(color=C["cian"])))
    fig.add_trace(go.Scatter(x=ts[idx],y=ys[idx,1],name="Norte (y)",line=dict(color=C["rosa"])))
    fig.update_layout(xaxis_title="Tiempo simulado (s)", yaxis_title="Posición (m)")
    return terminar_grafica(fig)


def terminar_grafica(fig):
    fig.update_layout(paper_bgcolor=C["fondo"],plot_bgcolor=C["panel"],
                      font=dict(color=C["blanco"],size=11),height=290,
                      margin=dict(l=7,r=7,t=10,b=7),
                      legend=dict(orientation="h",y=1.22,x=0),
                      hovermode="closest",uirevision="graficas")
    fig.update_xaxes(gridcolor="#344054",zerolinecolor="#56667F")
    fig.update_yaxes(gridcolor="#344054",zerolinecolor="#56667F")
    return fig


def detener_al_mover_slider():
    st.session_state.playing = False

# Fragmento: actualiza solo el reproductor; NO vuelve a integrar la física.
# La integración está almacenada por st.cache_data con los parámetros elegidos.
@st.fragment(run_every=0.55)
def reproductor():
    b1,b2,b3 = st.columns([1,1,2])
    with b1:
        if st.button("⏸ Pausar" if st.session_state.playing else "▶ Iniciar",use_container_width=True):
            if st.session_state.seek >= DURACION:
                st.session_state.seek=0.0
            st.session_state.playing = not st.session_state.playing
            st.rerun(scope="fragment")
    with b2:
        if st.button("↺ Reiniciar",use_container_width=True):
            st.session_state.playing=False
            st.session_state.seek=0.0
            st.rerun(scope="fragment")
    with b3:
        st.select_slider("Reproducción (×)", options=[1,2,5,10,20,50,100,200],
                         key="vel_reproduccion")
    if st.session_state.playing:
        # Avanzar en pasos compatibles con la salida cada DT=0.08 s.
        indice_actual = int(round(st.session_state.seek / DT))
        avance = max(1, int(round(0.55 * st.session_state.vel_reproduccion / DT)))
        siguiente = min(len(ts) - 1, indice_actual + avance)
        st.session_state.seek = float(ts[siguiente])
        if siguiente == len(ts) - 1:
            st.session_state.playing = False
    tiempo = st.slider("Tiempo simulado", 0.0,DURACION,step=DT,key="seek",
                       on_change=detener_al_mover_slider, format="%.2f s")
    i = min(int(round(tiempo/DT)),len(ts)-1)
    vals = magnitudes(ys[i], p)
    st.caption(f"t = {ts[i]:.2f} s  ·  modo {'Real' if p.multiplicador==1 else f'Demo ×{p.multiplicador:g}'}  ·  Coriolis {'activo' if p.coriolis else 'inactivo'}")
    a1,a2,a3 = st.columns(3)
    a1.metric("Rapidez",f"{np.linalg.norm(vals['v']):.3f} m/s")
    a2.metric("Aceleración",f"{np.linalg.norm(vals['a']):.3f} m/s²")
    a3.metric("Fuerza Coriolis",f"{np.linalg.norm(vals['Fc']):.5f} N")
    b1,b2,b3 = st.columns(3)
    b1.metric("Gravedad",f"{np.linalg.norm(vals['Fg']):.3f} N")
    b2.metric("Tensión",f"{vals['tension']:.3f} N")
    b3.metric("Energía",f"{vals['E']:.3f} J")
    st.plotly_chart(grafica_3d(i,vals,ver),use_container_width=True,config={"displaylogo":False})
    st.markdown('<div class="small-note">Arrastra con el dedo para girar la vista 3D y junta o separa los dedos para acercar. Flechas en escalas independientes, especialmente Coriolis.</div>',unsafe_allow_html=True)
    g1,g2 = st.columns(2)
    with g1:
        st.markdown("**Trayectoria vista desde arriba**")
        st.plotly_chart(grafica_plano(i),use_container_width=True,config={"displaylogo":False})
    with g2:
        st.markdown("**Desplazamiento horizontal (últimos 60 s)**")
        st.plotly_chart(grafica_desplazamiento(i),use_container_width=True,config={"displaylogo":False})
    st.caption("Precesión aproximada: " +
               (f"{precesion_grados_hora(p):+.2f} °/h" if p.coriolis else "no aplicable (Coriolis desactivado)"))
    with st.expander("Ver componentes numéricas de posición, velocidad y aceleración"):
        r,v,a=ys[i,:3],ys[i,3:],vals["a"]
        st.dataframe({"Eje":["Este x","Norte y","Arriba z"],
                      "Posición (m)":[float(x) for x in r],
                      "Velocidad (m/s)":[float(x) for x in v],
                      "Aceleración (m/s²)":[float(x) for x in a]},hide_index=True,use_container_width=True)
        st.write(f"Energía cinética: **{vals['Ec']:.4f} J** · Potencial: **{vals['Ep']:.4f} J**")


reproductor()
st.divider()
with st.expander("∑ Ver todas las ecuaciones y su significado",expanded=False):
    st.caption("Las expresiones provienen del panel de ecuaciones del programa original. La del período se emplea para comprobar el modelo.")
    for nombre,items in GRUPOS:
        st.subheader(nombre)
        for titulo,formula,descripcion in items:
            st.markdown(f"**{titulo}**")
            st.latex(formula)
            st.caption(descripcion)
    st.info("Supuestos: sin rozamiento, gravedad efectiva constante (incluye el efecto centrífugo local), soporte fijo en el marco terrestre. La precesión es una aproximación válida para oscilaciones pequeñas.")

with st.expander("🐍 Qué partes están realmente programadas en Python",expanded=False):
    st.markdown("""
**Todo el cálculo físico se ejecuta en Python:**

- `fisica.py` contiene `rotacion_local()`, `condiciones_iniciales()`, `magnitudes()`, `ecuaciones()` y `calcular()`.
- NumPy calcula los vectores de posición, velocidad y fuerza.
- SciPy `solve_ivp(method="DOP853")` resuelve el sistema diferencial y obtiene la trayectoria.
- `app.py` usa Streamlit para controles y Plotly para dibujar los datos **ya calculados en Python**.
- `ecuaciones.py` guarda las expresiones para estudiar y defender el modelo.

La aplicación web necesita conexión al servidor para recalcular parámetros. El teléfono no ejecuta el núcleo numérico: lo ejecuta Python en el servidor.
""")

st.caption("Proyecto universitario de física · Sistema local Este–Norte–Arriba · Cálculos físicos originales de Python · Interfaz Streamlit")
