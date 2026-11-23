"""Simulador universitario del pendulo de Coriolis para Linux Mint.

Dependencias: numpy, scipy, matplotlib. Ejecutar: python pendulo_coriolis.py
Pruebas: python pendulo_coriolis.py --test
"""

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401: activa la proyeccion 3D

OMEGA_TIERRA = 7.2921159e-5  # rad/s
COLORES = {
    "fondo": "#0B0F19", "panel": "#151C2C", "rejilla": "#344054",
    "texto": "#EAF2FF", "cian": "#00F0FF", "rosa": "#EC70FF",
    "verde": "#44E08A", "amarillo": "#FFCC55", "rojo": "#FF6175",
    "azul": "#559CFF", "naranja": "#FF9F43", "gris": "#AAB8CC",
}


@dataclass
class Parametros:
    longitud: float = 2.0
    masa: float = 1.0
    angulo: float = 25.0      # grados desde la vertical
    azimut: float = 0.0      # grados desde el Este hacia el Norte
    latitud: float = 45.0    # grados geograficos
    velocidad: float = 0.0   # m/s, tangente horizontal inicial
    multiplicador: float = 1.0
    coriolis: bool = True
    gravedad: float = 9.81   # gravedad efectiva local, m/s^2


def rotacion_local(latitud, multiplicador=1.0):
    """Vector terrestre en coordenadas Este-Norte-Arriba (ENU)."""
    phi = np.deg2rad(latitud)
    return OMEGA_TIERRA * multiplicador * np.array([0., np.cos(phi), np.sin(phi)])


def condiciones_iniciales(p):
    """Posicion relativa al soporte y velocidad tangente a la esfera de radio L."""
    theta, az = np.deg2rad([p.angulo, p.azimut])
    r = p.longitud * np.array([
        np.sin(theta) * np.cos(az), np.sin(theta) * np.sin(az), -np.cos(theta)
    ])
    # Velocidad tangencial perpendicular al vector de la cuerda.
    direccion = np.array([-np.sin(az), np.cos(az), 0.])
    v = p.velocidad * direccion
    return np.r_[r, v]


def magnitudes(y, p):
    """Devuelve fuerzas reales/aparentes, aceleracion y energia.

    Se usa gravedad EFECTIVA (incluye la parte centrifuga terrestre local).
    Se desprecia la variacion espacial de esa gravedad: pendulo pequeno frente
    al radio de la Tierra y soporte fijo en el marco rotante.
    """
    r = y[:3]
    v = y[3:]
    omega = rotacion_local(p.latitud, p.multiplicador) if p.coriolis else np.zeros(3)
    fg = np.array([0., 0., -p.masa * p.gravedad])
    fc = -2.0 * p.masa * np.cross(omega, v)
    # La fuerza hacia el soporte es -T*r/L. De r.r=L^2 se obtiene
    # r.a=-v.v y, despejando, T=m(|v|^2 + r.(g + a_cor))/L.
    otras_ac = (fg + fc) / p.masa
    tension = p.masa * (np.dot(v, v) + np.dot(r, otras_ac)) / p.longitud
    ft = -tension * (r / p.longitud)
    a = (fg + fc + ft) / p.masa
    ec = 0.5 * p.masa * np.dot(v, v)
    ep = p.masa * p.gravedad * (r[2] + p.longitud)
    return {"Fg": fg, "Fc": fc, "T": ft, "v": v, "a": a,
            "tension": tension, "Ec": ec, "Ep": ep, "E": ec + ep}


def ecuaciones(t, y, p):
    vals = magnitudes(y, p)
    return np.r_[y[3:], vals["a"]]


def calcular(p, duracion=240., dt=0.08):
    """Integracion numerica; proyeccion pequena para reducir deriva de cuerda."""
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
    # Proyectar salida numerica sobre |r|=L y r.v=0 (el solver ya respeta
    # la restriccion con error pequeno; esto evita deriva en la visualizacion).
    for y in yy:
        r = y[:3]
        r *= p.longitud / np.linalg.norm(r)
        y[3:] -= r * (np.dot(r, y[3:]) / p.longitud**2)
    return tiempos, yy


class Flecha3D:
    """Flecha persistente: no se elimina ni se vuelve a crear por fotograma."""
    def __init__(self, ax, color, ancho=2.1):
        self.linea, = ax.plot([], [], [], color=color, lw=ancho, zorder=10)
        self.ala1, = ax.plot([], [], [], color=color, lw=ancho, zorder=10)
        self.ala2, = ax.plot([], [], [], color=color, lw=ancho, zorder=10)

    def actualizar(self, origen, vector, visible=True):
        norm = np.linalg.norm(vector)
        if not visible or norm < 1e-12:
            for line in (self.linea, self.ala1, self.ala2):
                line.set_visible(False)
            return
        for line in (self.linea, self.ala1, self.ala2):
            line.set_visible(True)
        extremo = origen + vector
        eje = vector / norm
        ref = np.array([0., 0., 1.])
        if abs(np.dot(eje, ref)) > .93:
            ref = np.array([0., 1., 0.])
        perpendicular = np.cross(eje, ref)
        perpendicular /= np.linalg.norm(perpendicular)
        longitud_ala = .22 * norm
        p1 = extremo - .27 * vector + longitud_ala * perpendicular
        p2 = extremo - .27 * vector - longitud_ala * perpendicular
        for linea, a, b in ((self.linea, origen, extremo),
                            (self.ala1, extremo, p1), (self.ala2, extremo, p2)):
            linea.set_data([a[0], b[0]], [a[1], b[1]])
            linea.set_3d_properties([a[2], b[2]])


def estilo_ejes(ax, es3d=False):
    ax.set_facecolor(COLORES["panel"])
    ax.tick_params(colors=COLORES["gris"], labelsize=8)
    if es3d:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.line.set_color(COLORES["gris"])
    else:
        for spine in ax.spines.values():
            spine.set_color(COLORES["rejilla"])
        ax.grid(True, color=COLORES["rejilla"], alpha=.40)


class Simulador:
    def __init__(self):
        self.p = Parametros()
        self.tiempos = None
        self.estados = None
        self.comparacion = None
        self.segundos = 0.
        self.play = False
        self.desactivar_eventos = False
        self.pendiente = False
        self.visibilidad = [True, True, True, True, True]
        self.fig_ecuaciones = None  # Ventana secundaria de formulas y explicaciones
        self.construir_interfaz()
        self.recalcular()
        self.timer = self.fig.canvas.new_timer(interval=45)  # ~22 fotogramas/s
        self.timer.add_callback(self.paso)
        self.timer.start()
        self.fig.canvas.mpl_connect("close_event", lambda evento: self.timer.stop())
        self.fig.canvas.mpl_connect("button_release_event", self.aplicar_pendientes)

    def construir_interfaz(self):
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                             "text.color": COLORES["texto"],
                             "axes.labelcolor": COLORES["texto"]})
        self.fig = plt.figure(figsize=(15.2, 9.1), facecolor=COLORES["fondo"])
        self.fig.canvas.manager.set_window_title("Péndulo de Coriolis | Laboratorio de Física")
        self.fig.text(.045, .958, "PÉNDULO DE CORIOLIS", color=COLORES["cian"],
                      fontsize=18, weight="bold")
        self.fig.text(.046, .929, "Simulación tridimensional · coordenadas Este–Norte–Arriba",
                      color=COLORES["gris"], fontsize=9)

        self.ax_pos = self.fig.add_axes([.055, .665, .38, .215])
        self.ax_plano = self.fig.add_axes([.055, .355, .38, .225])
        self.ax_3d = self.fig.add_axes([.485, .346, .37, .525], projection="3d")
        for ax in (self.ax_pos, self.ax_plano):
            estilo_ejes(ax)
        estilo_ejes(self.ax_3d, True)
        self.ax_pos.set_title("Desplazamiento horizontal", loc="left", color=COLORES["texto"], fontsize=11)
        self.ax_pos.set_xlabel("Tiempo simulado (s)")
        self.ax_pos.set_ylabel("Posición (m)")
        self.x_line, = self.ax_pos.plot([], [], color=COLORES["cian"], lw=1.9, label="x / Este")
        self.y_line, = self.ax_pos.plot([], [], color=COLORES["rosa"], lw=1.7, label="y / Norte")
        self.ax_pos.legend(loc="upper right", facecolor=COLORES["panel"],
                           labelcolor=COLORES["texto"], framealpha=.8, fontsize=8)
        self.ax_plano.set_title("Trayectoria vista desde arriba", loc="left", color=COLORES["texto"], fontsize=11)
        self.ax_plano.set_xlabel("Este, x (m)")
        self.ax_plano.set_ylabel("Norte, y (m)")
        self.ax_plano.set_aspect("equal", adjustable="box")
        self.plano_traza, = self.ax_plano.plot([], [], color=COLORES["cian"], lw=1.25)
        self.plano_sin, = self.ax_plano.plot([], [], color=COLORES["gris"],
                                            lw=1., ls="--", alpha=.6, label="Sin Coriolis")
        self.plano_punto, = self.ax_plano.plot([], [], marker="o", ms=6,
                                              color=COLORES["amarillo"])
        self.ax_plano.legend(loc="upper right", fontsize=7, facecolor=COLORES["panel"],
                             labelcolor=COLORES["texto"])

        self.ax_3d.set_title("Péndulo 3D · gira y acerca con el ratón", color=COLORES["texto"], fontsize=11)
        self.ax_3d.set_xlabel("Este x (m)", labelpad=3)
        self.ax_3d.set_ylabel("Norte y (m)", labelpad=3)
        self.ax_3d.set_zlabel("Arriba z (m)", labelpad=3)
        self.ax_3d.view_init(elev=23, azim=-58)
        self.cuerda, = self.ax_3d.plot([], [], [], color="#DAE6FF", lw=2.5)
        self.bola, = self.ax_3d.plot([], [], [], marker="o", ms=11,
                                     color=COLORES["cian"], ls="")
        self.soporte, = self.ax_3d.plot([0], [0], [0], marker="o", ms=6,
                                        color=COLORES["gris"], ls="")
        self.rastro3d, = self.ax_3d.plot([], [], [], color=COLORES["cian"],
                                         alpha=.60, lw=1.)
        self.vectores = {
            "Fg": Flecha3D(self.ax_3d, COLORES["rojo"]),
            "T": Flecha3D(self.ax_3d, COLORES["azul"]),
            "Fc": Flecha3D(self.ax_3d, COLORES["rosa"]),
            "v": Flecha3D(self.ax_3d, COLORES["verde"]),
            "a": Flecha3D(self.ax_3d, COLORES["amarillo"]),
        }

        self.fig.text(.055, .296, "PARÁMETROS", color=COLORES["cian"], fontsize=10, weight="bold")
        # Sliders divididos en dos columnas; el evento se dispara al soltar el ratón
        specs = [
            ("Longitud (m)", .5, 8., 2., .055, .254, "longitud"),
            ("Masa (kg)", .1, 10., 1., .45, .254, "masa"),
            ("Ángulo (°)", 1., 60., 25., .055, .212, "angulo"),
            ("Azimut (°)", 0., 360., 0., .45, .212, "azimut"),
            ("Latitud (°)", -90., 90., 45., .055, .170, "latitud"),
            ("Vel. inicial (m/s)", 0., 3., 0., .45, .170, "velocidad"),
        ]
        self.sliders = {}
        for titulo, minimo, maximo, valor, x, y, nombre in specs:
            ax = self.fig.add_axes([x + .13, y, .205, .017], facecolor=COLORES["panel"])
            slider = Slider(ax, titulo, minimo, maximo, valinit=valor,
                            color=COLORES["cian"],
                            initcolor="none", valfmt="%1.1f")
            slider.label.set_color(COLORES["texto"])
            slider.valtext.set_color(COLORES["amarillo"])
            slider.label.set_fontsize(8)
            slider.valtext.set_fontsize(8)
            slider.on_changed(self.cambiar_parametros)
            self.sliders[nombre] = slider

        ax_speed = self.fig.add_axes([.19, .112, .25, .017], facecolor=COLORES["panel"])
        self.slider_speed = Slider(ax_speed, "Reproducción (×)", 1, 200,
                                   valinit=10, valstep=1, color=COLORES["verde"],
                                   initcolor="none", valfmt="%d")
        self.slider_speed.label.set_color(COLORES["texto"])
        self.slider_speed.valtext.set_color(COLORES["amarillo"])

        self.btn_play = self._boton([.055, .057, .095, .042], "▶ INICIAR", self.alternar)
        self.btn_reset = self._boton([.166, .057, .095, .042], "↺ REINICIAR", self.reiniciar)
        self.btn_camera = self._boton([.278, .057, .12, .042], "VISTA INICIAL", self.camara)
        self.btn_cor = self._boton([.417, .057, .14, .042], "CORIOLIS: SÍ", self.alternar_coriolis)
        self.fig.text(.585, .135, "Rotación terrestre", color=COLORES["texto"], fontsize=9)
        rax = self.fig.add_axes([.594, .045, .11, .085], facecolor=COLORES["fondo"])
        self.radios = RadioButtons(rax, ["Real ×1", "Demo ×100", "Demo ×1000"], active=0,
                                   activecolor=COLORES["cian"])
        for texto in self.radios.labels:
            texto.set_color(COLORES["texto"])
            texto.set_fontsize(8)
        self.radios.on_clicked(self.rotacion_cambiada)

        self.fig.text(.826, .283, "VECTORES", color=COLORES["cian"], fontsize=10, weight="bold")
        cax = self.fig.add_axes([.83, .068, .14, .20], facecolor=COLORES["fondo"])
        self.checks = CheckButtons(cax, ["Gravedad", "Tensión", "Coriolis", "Velocidad", "Aceleración"],
                                   actives=self.visibilidad)
        for texto, color in zip(self.checks.labels,
                                [COLORES["rojo"], COLORES["azul"], COLORES["rosa"],
                                 COLORES["verde"], COLORES["amarillo"]]):
            texto.set_color(color)
            texto.set_fontsize(9)
        self.checks.on_clicked(self.cambiar_vectores)

        self.estado_texto = self.fig.text(.50, .973, "", color=COLORES["texto"], fontsize=9, va="top")
        self.datos_texto = self.fig.text(.50, .945, "", color=COLORES["gris"], fontsize=8, va="top")
        self.ecuaciones_texto = self.fig.text(
            .055, .019, r"$m\vec a=\vec F_g+\vec T+\vec F_C$    "
            r"$\vec F_C=-2m(\vec\Omega\times\vec v)$    "
            r"$\omega_p\approx\Omega_T\sin\varphi$",
            color=COLORES["gris"], fontsize=10)
        self.fig.text(.82, .041, "Flechas en escalas propias; Fc ampliada visualmente.",
                      fontsize=7, color=COLORES["gris"], va="center")
        # Abre un formulario independiente para no tapar los graficos ni controles.
        self.btn_ecuaciones = self._boton(
            [.665, .007, .142, .032], "∑ VER ECUACIONES", self.mostrar_ecuaciones
        )

    def _boton(self, rect, titulo, callback):
        ax = self.fig.add_axes(rect)
        boton = Button(ax, titulo, color=COLORES["panel"], hovercolor="#264258")
        boton.label.set_color(COLORES["texto"])
        boton.label.set_fontsize(8)
        boton.on_clicked(callback)
        return boton

    def mostrar_ecuaciones(self, _):
        """Muestra TODAS las ecuaciones fisicas usadas por el simulador.

        Se usa otra ventana para conservar intacta la animacion 3D. Las
        expresiones se corresponden con las funciones condiciones_iniciales,
        rotacion_local, magnitudes, ecuaciones, calcular y pruebas.
        """
        # Si la ventana ya existe, se recupera en lugar de duplicarla.
        if self.fig_ecuaciones is not None and plt.fignum_exists(self.fig_ecuaciones.number):
            self.fig_ecuaciones.show()
            return

        figura = plt.figure(figsize=(15.5, 9.1), facecolor=COLORES["fondo"])
        figura.canvas.manager.set_window_title("Ecuaciones | Pendulo de Coriolis")
        self.fig_ecuaciones = figura
        figura.text(.035, .953, "ECUACIONES DEL PÉNDULO DE CORIOLIS",
                    fontsize=20, weight="bold", color=COLORES["cian"])
        figura.text(.035, .914,
                    "Las fórmulas corresponden al modelo implementado en el código original.",
                    fontsize=10, color=COLORES["gris"])

        # Nota: r y v son vectores Este-Norte-Arriba y sus tres componentes.
        columnas = [
            ("01   POSICIÓN Y ROTACIÓN", COLORES["cian"], [
                ("Posición inicial: Este y Norte",
                 r"$x_0=L\sin\theta\cos\alpha,\quad y_0=L\sin\theta\sin\alpha$"),
                ("Posición inicial: altura",
                 r"$z_0=-L\cos\theta$"),
                ("Velocidad inicial, tangente a la cuerda",
                 r"$\vec v_0=v_0(-\sin\alpha,\cos\alpha,0)$"),
                ("Giro terrestre en ejes locales",
                 r"$\vec\Omega=\Omega_T M(0,\cos\varphi,\sin\varphi)$"),
                ("Rotación real terrestre",
                 r"$\Omega_T=7.2921159\times10^{-5}\ \mathrm{rad/s}$"),
                ("Cuerda de longitud constante",
                 r"$\vec r\cdot\vec r=L^2$"),
                ("Velocidad tangente a la trayectoria",
                 r"$\vec r\cdot\vec v=0$"),
                ("Conversión de grados a radianes",
                 r"$\theta_{rad}=\theta_{deg}\,\pi/180$"),
            ]),
            ("02   FUERZAS Y ACELERACIÓN", COLORES["rosa"], [
                ("Fuerza de gravedad",
                 r"$\vec F_g=(0,0,-mg)$"),
                ("Fuerza aparente de Coriolis",
                 r"$\vec F_C=-2m(\vec\Omega\times\vec v)$"),
                ("Otras aceleraciones (sin tensión)",
                 r"$\vec a_{ext}=(\vec F_g+\vec F_C)/m$"),
                ("Restricción de aceleración de la cuerda",
                 r"$\vec r\cdot\vec a=-\vec v\cdot\vec v$"),
                ("Magnitud de la tensión",
                 r"$T=\frac{m}{L}(\vec v\cdot\vec v+\vec r\cdot\vec a_{ext})$"),
                ("Fuerza de tensión, hacia el soporte",
                 r"$\vec F_T=-T\,\vec r/L$"),
                ("Segunda ley de Newton",
                 r"$m\vec a=\vec F_g+\vec F_C+\vec F_T$"),
                ("Aceleración total",
                 r"$\vec a=(\vec F_g+\vec F_C+\vec F_T)/m$"),
            ]),
            ("03   MOVIMIENTO Y ENERGÍA", COLORES["verde"], [
                ("La velocidad cambia la posición",
                 r"$\frac{d\vec r}{dt}=\vec v$"),
                ("La aceleración cambia la velocidad",
                 r"$\frac{d\vec v}{dt}=\vec a$"),
                ("Energía cinética",
                 r"$E_c=\frac{1}{2}m(\vec v\cdot\vec v)$"),
                ("Altura desde el punto más bajo",
                 r"$h=z+L$"),
                ("Energía potencial gravitatoria",
                 r"$E_p=mg(z+L)$"),
                ("Energía mecánica total",
                 r"$E=E_c+E_p$"),
                ("Precesión aproximada (giro del plano)",
                 r"$\omega_p\approx\Omega_T M\sin\varphi$"),
                ("Paso de rad/s a grados/hora",
                 r"$\omega_{\mathrm{deg/h}}=\omega_p\,\frac{180}{\pi}\,3600$"),
            ]),
        ]
        izquierda = [.035, .360, .685]
        for x, (titulo, color, items) in zip(izquierda, columnas):
            ax = figura.add_axes([x, .128, .295, .755], facecolor=COLORES["panel"])
            ax.set_xticks([])
            ax.set_yticks([])
            for borde in ax.spines.values():
                borde.set_visible(False)
            ax.text(.045, .955, titulo, color=color, fontsize=11,
                    weight="bold", va="top", transform=ax.transAxes)
            ax.plot([.045, .955], [.906, .906], transform=ax.transAxes,
                    color=COLORES["rejilla"], linewidth=1)
            for i, (etiqueta, expresion) in enumerate(items):
                y = .869 - i * .103
                ax.text(.045, y, etiqueta, transform=ax.transAxes,
                        color=COLORES["gris"], fontsize=9.4, va="top")
                ax.text(.045, y - .039, expresion, transform=ax.transAxes,
                        color=COLORES["texto"], fontsize=12.8, va="top")

        figura.text(.037, .088,
                    "Comprobación adicional en el código: período para ángulos pequeños  "
                    r"$P\approx2\pi\sqrt{L/g}$" 
                    "   |   Integración: DOP853, salida cada 0,08 s.",
                    color=COLORES["texto"], fontsize=10)
        figura.text(.037, .053,
                    "m: masa   L: longitud   θ: inclinación   α: azimut   φ: latitud   "
                    "M: multiplicador de rotación   g: gravedad efectiva",
                    color=COLORES["gris"], fontsize=9)
        figura.text(.037, .024,
                    "Modelo ideal sin rozamiento. La gravedad efectiva ya incluye "
                    "la contribución centrífuga local; las flechas del 3D tienen escalas visuales distintas.",
                    color=COLORES["gris"], fontsize=8.5)
        figura.canvas.mpl_connect("close_event", self._cerrar_ecuaciones)
        figura.canvas.draw_idle()
        figura.show()

    def _cerrar_ecuaciones(self, _):
        """Permite volver a abrir el panel despues de cerrarlo."""
        self.fig_ecuaciones = None

    def cambiar_parametros(self, _):
        if self.desactivar_eventos:
            return
        self.pendiente = True

    def aplicar_pendientes(self, _):
        """Calcular al soltar el control, no en cada pixel del arrastre."""
        if not self.pendiente:
            return
        self.pendiente = False
        estaba = self.play
        self.play = False
        for nombre, control in self.sliders.items():
            setattr(self.p, nombre, float(control.val))
        self.recalcular()
        self.play = estaba
        self.btn_play.label.set_text("Ⅱ PAUSAR" if estaba else "▶ INICIAR")

    def recalcular(self):
        """Precalcula solo unos minutos: rapido, simple y sin bloquear cada frame."""
        self.tiempos, self.estados = calcular(self.p)
        # Comparacion con exactamente las mismas condiciones iniciales.
        p_sin = Parametros(**vars(self.p))
        p_sin.coriolis = False
        _, self.comparacion = calcular(p_sin)
        self.segundos = 0.
        L = self.p.longitud
        self.ax_3d.set_xlim(-1.15*L, 1.15*L)
        self.ax_3d.set_ylim(-1.15*L, 1.15*L)
        self.ax_3d.set_zlim(-1.23*L, .20*L)
        self.ax_3d.set_box_aspect((1, 1, .75))
        self.ax_plano.set_xlim(-1.12*L, 1.12*L)
        self.ax_plano.set_ylim(-1.12*L, 1.12*L)
        self.ax_pos.set_xlim(0, 60.)
        self.ax_pos.set_ylim(-.85*L, .85*L)
        self.actualizar(0)

    def alternar(self, _):
        if self.segundos >= self.tiempos[-1]:
            self.segundos = 0.
        self.play = not self.play
        self.btn_play.label.set_text("Ⅱ PAUSAR" if self.play else "▶ INICIAR")
        self.fig.canvas.draw_idle()

    def reiniciar(self, _):
        self.play = False
        self.segundos = 0.
        self.btn_play.label.set_text("▶ INICIAR")
        self.actualizar(0)

    def camara(self, _):
        self.ax_3d.view_init(elev=23, azim=-58)
        self.fig.canvas.draw_idle()

    def alternar_coriolis(self, _):
        self.p.coriolis = not self.p.coriolis
        self.btn_cor.label.set_text("CORIOLIS: SÍ" if self.p.coriolis else "CORIOLIS: NO")
        self.recalcular()

    def rotacion_cambiada(self, etiqueta):
        self.p.multiplicador = {"Real ×1": 1., "Demo ×100": 100., "Demo ×1000": 1000.}[etiqueta]
        self.recalcular()

    def cambiar_vectores(self, _):
        self.visibilidad = list(self.checks.get_status())
        self.actualizar(min(int(self.segundos / .08), len(self.tiempos)-1))

    def paso(self):
        if not self.play or self.tiempos is None:
            return
        self.segundos += (.045 * float(self.slider_speed.val))
        if self.segundos >= self.tiempos[-1]:
            self.segundos = float(self.tiempos[-1])
            self.play = False
            self.btn_play.label.set_text("▶ INICIAR")
        indice = int(np.searchsorted(self.tiempos, self.segundos, side="right")-1)
        self.actualizar(np.clip(indice, 0, len(self.tiempos)-1))

    def actualizar(self, i):
        p, ts, ys = self.p, self.tiempos, self.estados
        if ts is None:
            return
        r, v = ys[i, :3], ys[i, 3:]
        m = magnitudes(ys[i], p)
        t = ts[i]
        # Desplazamiento: dibujar solamente los ultimos 60 s.
        inicio = max(0, int(np.searchsorted(ts, max(0., t-60.))))
        self.x_line.set_data(ts[inicio:i+1], ys[inicio:i+1, 0])
        self.y_line.set_data(ts[inicio:i+1], ys[inicio:i+1, 1])
        self.ax_pos.set_xlim(max(0, t-60.), max(60., t))
        # Rastro superior y comparacion siempre con las mismas condiciones.
        self.plano_traza.set_data(ys[:i+1, 0], ys[:i+1, 1])
        self.plano_sin.set_data(self.comparacion[:i+1, 0], self.comparacion[:i+1, 1])
        self.plano_punto.set_data([r[0]], [r[1]])
        # Solo modificar datos de artistas existentes; nunca ax.clear().
        self.cuerda.set_data([0, r[0]], [0, r[1]])
        self.cuerda.set_3d_properties([0, r[2]])
        self.bola.set_data([r[0]], [r[1]])
        self.bola.set_3d_properties([r[2]])
        # Limitar puntos del rastro 3D para rendimiento razonable.
        paso = max(1, i // 1100)
        rastro = ys[:i+1:paso, :3]
        self.rastro3d.set_data(rastro[:, 0], rastro[:, 1])
        self.rastro3d.set_3d_properties(rastro[:, 2])
        # Cada magnitud usa escala propia para ser visible. Estas flechas no
        # comparten escala; en especial, Fc es graficamente amplificada.
        L, g, masa = p.longitud, p.gravedad, p.masa
        denominadores = {"Fg": masa*g, "T": masa*g,
                          "Fc": max(2*masa*OMEGA_TIERRA*p.multiplicador*np.sqrt(g*L), 1e-12),
                          "v": np.sqrt(g*L), "a": g}
        for j, clave in enumerate(("Fg", "T", "Fc", "v", "a")):
            vector = m[clave]
            escalado = (vector / denominadores[clave]) * (.43 * L)
            self.vectores[clave].actualizar(r, escalado, self.visibilidad[j])
        precesion = OMEGA_TIERRA * np.sin(np.deg2rad(p.latitud)) * p.multiplicador
        modo = "REAL" if p.multiplicador == 1 else f"DEMO ×{p.multiplicador:g}"
        self.estado_texto.set_text(f"t={t:6.1f} s   |   {modo}   |   "
                                   f"Coriolis {'activo' if p.coriolis else 'inactivo'}")
        self.datos_texto.set_text(
            f"|v|={np.linalg.norm(v):.3f} m/s    |a|={np.linalg.norm(m['a']):.3f} m/s²\n"
            f"|Fg|={np.linalg.norm(m['Fg']):.3f} N    "
            f"T={m['tension']:.3f} N    |Fc|={np.linalg.norm(m['Fc']):.5f} N\n"
            f"E={m['E']:.3f} J    "
            f"Precesión aprox.: {np.rad2deg(precesion)*3600:.2f} °/h"
        )
        self.fig.canvas.draw_idle()


def pruebas():
    """Pruebas numericas para verificar el comportamiento basico."""
    from dataclasses import replace
    p = Parametros(angulo=5., velocidad=0., coriolis=False)
    t, ys = calcular(p, duracion=15., dt=.04)
    longitud = np.linalg.norm(ys[:, :3], axis=1)
    assert np.max(np.abs(longitud-p.longitud)) < 1e-9
    assert np.max(np.abs(np.einsum('ij,ij->i', ys[:, :3], ys[:, 3:]))) < 1e-9
    energias = np.array([magnitudes(y, p)["E"] for y in ys])
    assert np.max(np.abs(energias - energias[0])) < 1e-6
    # El periodo de angulo pequeno se aproxima al periodo teorico.
    x = ys[:, 0]
    cruces = np.where((x[:-1] > 0.) & (x[1:] <= 0.))[0]
    periodo = np.mean(np.diff(t[cruces]))
    teorico = 2*np.pi*np.sqrt(p.longitud/p.gravedad)
    assert abs(periodo-teorico) / teorico < .02, (periodo, teorico)
    # La masa no altera la posicion del sistema ideal.
    _, ys_m = calcular(replace(p, masa=5.), duracion=15., dt=.04)
    assert np.max(np.abs(ys_m[:, :3]-ys[:, :3])) < 1e-6
    # Fuerza Coriolis nula cuando esta desactivada.
    assert np.linalg.norm(magnitudes(ys[20], p)["Fc"]) == 0.
    # En el ecuador la precesion ideal es cero; signo cambia de hemisferio.
    assert abs(rotacion_local(0.)[2]) < 1e-15
    assert rotacion_local(45.)[2] > 0 and rotacion_local(-45.)[2] < 0
    print("OK: restriccion, energia, periodo, independencia de masa y latitudes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laboratorio universitario de Coriolis")
    parser.add_argument("--test", action="store_true", help="Ejecutar pruebas fisicas sin abrir ventana")
    args = parser.parse_args()
    if args.test:
        pruebas()
    else:
        Simulador()
        plt.show()
