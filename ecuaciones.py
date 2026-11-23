"""Las 25 ecuaciones del panel original, con una nota didactica cada una."""

GRUPOS = [
    ("01 — Posición y rotación", [
        ("Posición horizontal inicial", r"x_0=L\sin\theta\cos\alpha,\quad y_0=L\sin\theta\sin\alpha", "Se obtiene proyectando la cuerda con trigonometría sobre Este y Norte."),
        ("Posición vertical inicial", r"z_0=-L\cos\theta", "La coordenada z es negativa porque la bola está debajo del soporte."),
        ("Velocidad inicial tangencial", r"\vec v_0=v_0(-\sin\alpha,\cos\alpha,0)", "La velocidad inicial se elige perpendicular a la cuerda."),
        ("Rotación terrestre local", r"\vec\Omega=\Omega_T M(0,\cos\varphi,\sin\varphi)", "La latitud permite separar la rotación en componentes Norte y Arriba."),
        ("Velocidad angular terrestre", r"\Omega_T=7.2921159\times10^{-5}\ \mathrm{rad/s}", "Velocidad angular física real de rotación de la Tierra."),
        ("Longitud fija de cuerda", r"\vec r\cdot\vec r=L^2", "La distancia entre soporte y bola siempre es L."),
        ("Velocidad tangente", r"\vec r\cdot\vec v=0", "Si la cuerda no cambia de longitud, la velocidad es perpendicular a ella."),
        ("Grados a radianes", r"\theta_{rad}=\theta_{deg}\,\frac{\pi}{180}", "NumPy usa radianes para seno y coseno."),
    ]),
    ("02 — Fuerzas y aceleración", [
        ("Gravedad", r"\vec F_g=(0,0,-mg)", "Apunta verticalmente hacia abajo."),
        ("Coriolis", r"\vec F_C=-2m(\vec\Omega\times\vec v)", "Fuerza aparente del sistema de referencia terrestre que gira."),
        ("Aceleración sin tensión", r"\vec a_{ext}=\frac{\vec F_g+\vec F_C}{m}", "Suma provisional de la gravedad y Coriolis."),
        ("Restricción de aceleración", r"\vec r\cdot\vec a=-\vec v\cdot\vec v", "Resulta de derivar la condición de longitud fija dos veces."),
        ("Magnitud de la tensión", r"T=\frac{m}{L}(\vec v\cdot\vec v+\vec r\cdot\vec a_{ext})", "La cuerda ejerce la fuerza necesaria para mantener distancia L."),
        ("Fuerza de tensión", r"\vec F_T=-T\frac{\vec r}{L}", "Apunta desde la bola hacia el soporte."),
        ("Segunda ley de Newton", r"m\vec a=\vec F_g+\vec F_C+\vec F_T", "La suma de fuerzas determina la aceleración."),
        ("Aceleración total", r"\vec a=\frac{\vec F_g+\vec F_C+\vec F_T}{m}", "Es la aceleración usada para actualizar el movimiento."),
    ]),
    ("03 — Movimiento y energía", [
        ("Derivada de la posición", r"\frac{d\vec r}{dt}=\vec v", "La velocidad indica qué tan rápido cambia la posición."),
        ("Derivada de la velocidad", r"\frac{d\vec v}{dt}=\vec a", "La aceleración indica qué tan rápido cambia la velocidad."),
        ("Energía cinética", r"E_c=\frac12 m(\vec v\cdot\vec v)", "La energía asociada al movimiento."),
        ("Altura sobre el punto inferior", r"h=z+L", "El nivel de referencia es el punto más bajo de la trayectoria."),
        ("Energía potencial", r"E_p=mg(z+L)", "Energía gravitatoria asociada a la altura."),
        ("Energía mecánica", r"E=E_c+E_p", "Suma de energías; prácticamente constante en este modelo ideal."),
        ("Precesión aproximada", r"\omega_p\approx\Omega_T M\sin\varphi", "Giro ideal aproximado del plano para oscilaciones pequeñas."),
        ("Precesión en grados por hora", r"\omega_{deg/h}=\omega_p\,\frac{180}{\pi}\,3600", "Cambio de unidades desde radianes por segundo."),
        ("Periodo pequeño ángulo (pruebas)", r"P\approx2\pi\sqrt{\frac{L}{g}}", "Aproximación que se usa para verificar el modelo con oscilaciones pequeñas."),
    ]),
]
