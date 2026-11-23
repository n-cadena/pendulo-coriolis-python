"""Nucleo fisico del pendulo de Coriolis.

Las ecuaciones y el metodo de integracion se conservan del simulador original.
Este modulo no depende de Streamlit: puede probarse de forma independiente.

Ejes locales: x=Este, y=Norte, z=Arriba; origen en el soporte.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

OMEGA_TIERRA = 7.2921159e-5  # rad/s: rotacion terrestre real


@dataclass(frozen=True)
class Parametros:
    longitud: float = 2.0         # m
    masa: float = 1.0             # kg
    angulo: float = 25.0          # grados desde la vertical
    azimut: float = 0.0           # grados Este hacia Norte
    latitud: float = 45.0         # grados
    velocidad: float = 0.0        # m/s, velocidad horizontal inicial
    multiplicador: float = 1.0   # x1, x100, x1000 (demostracion)
    coriolis: bool = True
    gravedad: float = 9.81        # m/s2, gravedad EFECTIVA


def rotacion_local(latitud, multiplicador=1.0):
    """Omega en coordenadas ENU (Este, Norte, Arriba)."""
    phi = np.deg2rad(latitud)
    return OMEGA_TIERRA * multiplicador * np.array([0., np.cos(phi), np.sin(phi)])


def condiciones_iniciales(p):
    """Devuelve [x,y,z,vx,vy,vz] con |r|=L y r.v=0."""
    theta, az = np.deg2rad([p.angulo, p.azimut])
    r = p.longitud * np.array([
        np.sin(theta) * np.cos(az),
        np.sin(theta) * np.sin(az),
        -np.cos(theta),
    ])
    direccion = np.array([-np.sin(az), np.cos(az), 0.])
    v = p.velocidad * direccion
    return np.r_[r, v]


def magnitudes(y, p):
    """Calcula fuerzas y energias del estado instantaneo y=[r,v].

    La gravedad efectiva ya representa el efecto centrifugo terrestre local.
    Se desprecia rozamiento y variacion espacial de la gravedad.
    """
    r = y[:3]
    v = y[3:]
    omega = rotacion_local(p.latitud, p.multiplicador) if p.coriolis else np.zeros(3)
    fg = np.array([0., 0., -p.masa * p.gravedad])
    fc = -2.0 * p.masa * np.cross(omega, v)
    # Fuerza de tension -T*r/L. La cuerda impone r.a = -v.v.
    otras_ac = (fg + fc) / p.masa
    tension = p.masa * (np.dot(v, v) + np.dot(r, otras_ac)) / p.longitud
    ft = -tension * (r / p.longitud)
    a = (fg + fc + ft) / p.masa
    ec = 0.5 * p.masa * np.dot(v, v)
    ep = p.masa * p.gravedad * (r[2] + p.longitud)
    return {"Fg": fg, "Fc": fc, "T": ft, "v": v, "a": a,
            "tension": tension, "Ec": ec, "Ep": ep, "E": ec + ep}


def ecuaciones(t, y, p):
    """Sistema de seis EDO: derivadas de posicion y de velocidad."""
    vals = magnitudes(y, p)
    return np.r_[y[3:], vals["a"]]


def calcular(p, duracion=240., dt=0.08):
    """Integra con SciPy DOP853, exactamente como el programa de escritorio."""
    tiempos = np.arange(0., duracion + dt * 0.5, dt)
    resultado = solve_ivp(
        lambda t, y: ecuaciones(t, y, p), (0., tiempos[-1]),
        condiciones_iniciales(p), t_eval=tiempos,
        method="DOP853", rtol=2e-9, atol=2e-11,
        max_step=0.15,
    )
    if not resultado.success:
        raise RuntimeError(resultado.message)
    yy = resultado.y.T.copy()
    # El solver casi cumple |r|=L y r.v=0: se corrige deriva numerica minima.
    for y in yy:
        r = y[:3]
        r *= p.longitud / np.linalg.norm(r)
        y[3:] -= r * (np.dot(r, y[3:]) / p.longitud**2)
    return tiempos, yy


def precesion_grados_hora(p):
    """Aproximacion de Foucault. No se muestra como activa si Coriolis=NO."""
    if not p.coriolis:
        return 0.0
    rad_s = OMEGA_TIERRA * np.sin(np.deg2rad(p.latitud)) * p.multiplicador
    return float(np.rad2deg(rad_s) * 3600.0)
