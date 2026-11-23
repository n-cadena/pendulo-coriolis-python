"""Péndulo de Coriolis · Streamlit + SciPy + visor táctil.

Estructura:
- fisica.py implementa las leyes de Newton y la fuerza de Coriolis.
- app.py integra TRAMOS SUCESIVOS con solve_ivp/DOP853 en Python.
- pendulo_frontend/index.html dibuja esos resultados y pide el siguiente tramo.

No se reciclan trayectorias y no existe una duración total fija.
Para ejecutar: streamlit run app.py
"""
from dataclasses import replace
from pathlib import Path
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from scipy.integrate import solve_ivp

from fisica import (
    Parametros, condiciones_iniciales, ecuaciones, magnitudes,
    precesion_grados_hora,
)
from ecuaciones import GRUPOS

st.set_page_config(
    page_title="Péndulo de Coriolis | Laboratorio Python",
    page_icon="🌀", layout="wide", initial_sidebar_state="expanded",
)
st.markdown("""<style>
.stApp {background:#0B0F19; color:#EAF2FF}
.block-container {padding-top:0.6rem; padding-bottom:1.2rem; max-width:1450px}
[data-testid="stSidebar"] {background:#151C2C}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label {color:#EAF2FF}
h1,h2,h3 {color:#EAF2FF!important}
header[data-testid="stHeader"] {background:transparent}
[data-testid="stExpander"] {border:1px solid #2E425C; border-radius:13px}
</style>""", unsafe_allow_html=True)

FRONTEND = components.declare_component(
    "pendulo_coriolis_en_vivo",
    path=str(Path(__file__).parent / "pendulo_frontend"),
)

# Se puede cambiar el tamaño de los tramos, pero no existe un máximo de tiempo.
# Los cálculos siguen en Python incluso cuando se observan desde el celular.
DURACION_PRIMER_TRAMO = 240.0
DURACION_TRAMOS_SIGUIENTES = 180.0
PASO_SALIDA = 0.12


def integrar_tramo(p: Parametros, t0: float, inicio: np.ndarray, duracion: float):
    """Continuar la solución desde t0, sin reemplazarla por la condición inicial.

    DOP853 resuelve dr/dt=v y dv/dt=a sobre un intervalo nuevo. Se conserva
    la misma proyección geométrica del programa original: |r|=L y r·v=0.
    """
    n = round(duracion / PASO_SALIDA)
    tiempos = t0 + np.arange(n + 1, dtype=float) * (duracion / n)
    salida = solve_ivp(
        lambda t, y: ecuaciones(t, y, p),
        (float(tiempos[0]), float(tiempos[-1])),
        np.array(inicio, dtype=float, copy=True),
        t_eval=tiempos,
        method="DOP853", rtol=2e-9, atol=2e-11, max_step=0.15,
    )
    if not salida.success:
        raise RuntimeError(f"No se pudo continuar la integración: {salida.message}")
    estados = salida.y.T.copy()
    for y in estados:
        r = y[:3]
        r *= p.longitud / np.linalg.norm(r)
        y[3:] -= r * (np.dot(r, y[3:]) / p.longitud**2)
    return tiempos, estados


def generar_bloque(p: Parametros, t0: float, con_inicio, sin_inicio, duracion):
    """Genera otro tramo y devuelve las condiciones para el tramo siguiente.

    La rama 'sin Coriolis' es independiente y también continúa desde su
    propio último estado, para una comparación físicamente consistente.
    """
    ts, con = integrar_tramo(p, t0, con_inicio, duracion)
    _, sin = integrar_tramo(replace(p, coriolis=False), t0, sin_inicio, duracion)
    registros = []
    # La primera fila de cada nuevo bloque ya existía en el bloque anterior.
    # Omitirla evita duplicados en el navegador.
    primero = 0 if t0 == 0.0 else 1
    for i in range(primero, len(ts)):
        estado = con[i]
        q = magnitudes(estado, p)
        r = [
            float(ts[i]),
            *map(float, estado[:3]),
            float(sin[i, 0]), float(sin[i, 1]),
            float(np.linalg.norm(q["v"])),
            float(np.linalg.norm(q["a"])),
            float(np.linalg.norm(q["Fg"])),
            float(q["tension"]),
            float(np.linalg.norm(q["Fc"])),
            float(q["Ec"]), float(q["Ep"]), float(q["E"]),
        ]
        for fuerza in ("Fg", "T", "Fc", "v", "a"):
            r.extend(map(float, q[fuerza]))
        # Se evitan cadenas con miles de dígitos en la transmisión de datos.
        registros.append([round(x, 9) for x in r])
    return registros, float(ts[-1]), con[-1].copy(), sin[-1].copy()


def nueva_sesion(p: Parametros, generacion: int):
    registros, t, yc, ys = generar_bloque(
        p, 0.0, condiciones_iniciales(p), condiciones_iniciales(p),
        DURACION_PRIMER_TRAMO,
    )
    return {
        "generacion": generacion,
        "seq": 0,
        "last_request": -1,
        "t": t,
        "yc": yc,
        "ys": ys,
        "bloque": registros,
    }


if "config" not in st.session_state:
    st.session_state.config = vars(Parametros()).copy()
if "generacion" not in st.session_state:
    st.session_state.generacion = 1
cfg = st.session_state.config

with st.sidebar:
    st.markdown("### ⚙️ Parámetros del péndulo")
    st.caption("Puedes ocultar esta barra con la flecha de Streamlit. Los cambios se aplican al pulsar el botón de abajo.")
    with st.form(f"formulario_parametros_{st.session_state.generacion}", clear_on_submit=False):
        L = st.slider("Longitud (m)", 0.5, 8.0, float(cfg["longitud"]), 0.1)
        masa = st.slider("Masa (kg)", 0.1, 10.0, float(cfg["masa"]), 0.1)
        theta = st.slider("Ángulo inicial (°)", 1, 60, int(cfg["angulo"]))
        azimut = st.slider("Azimut (°)", 0, 360, int(cfg["azimut"]), 5)
        lat = st.slider("Latitud (°)", -90, 90, int(cfg["latitud"]))
        velocidad = st.slider("Velocidad inicial (m/s)", 0.0, 3.0, float(cfg["velocidad"]), 0.1)
        opciones = {"Real ×1": 1., "Demo ×100": 100., "Demo ×1000": 1000.}
        idx = (1., 100., 1000.).index(float(cfg["multiplicador"]))
        modo = st.radio("Rotación terrestre", list(opciones), index=idx, horizontal=False)
        coriolis = st.toggle("Activar efecto Coriolis", value=bool(cfg["coriolis"]))
        aplicar = st.form_submit_button("Aplicar cambios", type="primary", use_container_width=True)
    if aplicar:
        st.session_state.config = dict(
            longitud=L, masa=masa, angulo=float(theta), azimut=float(azimut),
            latitud=float(lat), velocidad=velocidad,
            multiplicador=opciones[modo], coriolis=bool(coriolis), gravedad=9.81,
        )
        st.session_state.generacion += 1
        st.session_state.pop("trayectoria", None)
        st.rerun()
    st.caption("Física: NumPy + SciPy · método DOP853. No se repite ningún tramo.")

p = Parametros(**st.session_state.config)
st.markdown('<p style="font-size:0.70rem;color:#00F0FF;letter-spacing:.16em;font-weight:800;margin-bottom:0">LABORATORIO INTERACTIVO · FÍSICA</p>', unsafe_allow_html=True)
st.markdown('## PÉNDULO DE <span style="color:#00F0FF">CORIOLIS</span>', unsafe_allow_html=True)
st.caption("Modelo físico en Python · Vista interactiva desde el celular · Coordenadas Este / Norte / Arriba")

if ("trayectoria" not in st.session_state or
    st.session_state.trayectoria["generacion"] != st.session_state.generacion):
    with st.spinner("Calculando el primer tramo físico en Python…"):
        st.session_state.trayectoria = nueva_sesion(p, st.session_state.generacion)

s = st.session_state.trayectoria
componente = FRONTEND(
    generation=s["generacion"], seq=s["seq"], frames=s["bloque"],
    settings={
        "length": p.longitud, "mass": p.masa, "gravity": p.gravedad,
        "rotation": p.multiplicador, "coriolis": p.coriolis,
        "precession": precesion_grados_hora(p),
    },
    key="pendulo_reproductor_continuo",
    default=None,
)

# El visor NO calcula fuerzas ni trayectoria. Solo pide datos antes de que se agoten.
# Cuando llega la petición, se conserva el estado físico exacto del último
# fotograma y se empieza otro intervalo temporal con DOP853.
# Controles táctiles de reproducción y rotación: estos cambios vuelven a
# calcular físicamente desde t=0 con la configuración elegida.
if isinstance(componente, dict) and componente.get("generation") == s["generacion"]:
    tipo = componente.get("type")
    if tipo in ("restart", "set_mode", "set_coriolis"):
        if tipo == "set_mode" and componente.get("rotation") in (1, 100, 1000):
            st.session_state.config["multiplicador"] = float(componente["rotation"])
        elif tipo == "set_coriolis":
            st.session_state.config["coriolis"] = not bool(p.coriolis)
        st.session_state.generacion += 1
        st.session_state.pop("trayectoria", None)
        st.rerun()

if (isinstance(componente, dict)
        and componente.get("type") == "need_more"
        and componente.get("generation") == s["generacion"]):
    try:
        peticion = int(componente["request"])
    except (ValueError, TypeError, KeyError):
        peticion = -1
    if peticion > s["last_request"]:
        s["last_request"] = peticion
        with st.spinner("Python está calculando el siguiente tramo…"):
            reg, s["t"], s["yc"], s["ys"] = generar_bloque(
                p, s["t"], s["yc"], s["ys"], DURACION_TRAMOS_SIGUIENTES,
            )
        s["seq"] += 1
        s["bloque"] = reg
        st.rerun()

with st.expander("∑ Las 25 ecuaciones del modelo (desarrollo completo)", expanded=False):
    st.caption("Estas ecuaciones son las implementadas en fisica.py y utilizadas por SciPy.")
    for nombre, grupo in GRUPOS:
        st.subheader(nombre)
        for titulo, formula, descripcion in grupo:
            st.markdown(f"**{titulo}**")
            st.latex(formula)
            st.caption(descripcion)

st.caption("Cálculos físicos realizados en Python; la interfaz táctil usa HTML/JavaScript únicamente para dibujar y animar los resultados.")
