# Péndulo de Coriolis — aplicación web con Python

Esta versión permite presentar desde un celular **el mismo núcleo de física de Python** del simulador de escritorio. El navegador muestra la interfaz, pero los cálculos se hacen con `numpy` + `scipy` en el servidor de Streamlit.

## Archivos

- `app.py`: interfaz Streamlit, gráficos Plotly y reproducción.
- `fisica.py`: núcleo físico de Python; conserva las ecuaciones y el integrador SciPy `DOP853` del original.
- `ecuaciones.py`: las 25 expresiones físicas para consultar durante la sustentación.
- `requirements.txt`: bibliotecas a instalar.
- `simulador_original_escritorio.py`: copia del código anterior para comparar y sustentar su evolución (no se abre desde el navegador).
- `test_fisica.py`: comprobaciones y comparación contra el original.

## Ejecutar en un computador

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abre la dirección que Streamlit muestra en la terminal (usualmente `http://localhost:8501`). En un teléfono con conexión, usa el enlace público de la aplicación una vez publicada.

## Publicar gratis en Streamlit Community Cloud

1. Crea/inicia sesión en https://github.com y crea un repositorio público nuevo, por ejemplo `pendulo-coriolis-python`.
2. Sube **el contenido** de esta carpeta, con `app.py` y `requirements.txt` en la raíz. Si subes el ZIP intacto, primero tienes que descomprimirlo.
3. Abre https://share.streamlit.io/ e inicia sesión con GitHub.
4. Pulsa *Create app* o *New app*, elige tu repositorio, rama principal y archivo principal `app.py`.
5. Pulsa *Deploy*. Streamlit instalará los paquetes desde `requirements.txt` y ejecutará Python en su servidor.
6. Guarda la dirección `https://...streamlit.app` que recibas. Ábrela en Chrome en tu celular o en el computador del aula.

**Importante:** un repositorio público permite que cualquiera vea el código. Si no deseas hacerlo público, consulta las opciones vigentes de privacidad/hosting antes de compartirlo.

## Qué explicar al profesor

- `rotacion_local()`: cómo depende de la latitud el vector de rotación terrestre.
- `condiciones_iniciales()`: posición/velocidad de partida en coordenadas Este-Norte-Arriba.
- `magnitudes()`: gravedad, Coriolis, tensión, aceleración y energías.
- `ecuaciones()`: derivadas de posición y velocidad.
- `calcular()`: `solve_ivp` / `DOP853`; dos simulaciones con/sin Coriolis.
- `app.py`: interfaz web; la visualización con Plotly **no** reemplaza los cálculos de Python.

## Límites del modelo

- Péndulo ideal sin rozamiento, gravedad efectiva uniforme y soporte fijo al suelo.
- El efecto centrífugo local está incluido en la gravedad efectiva; no se suma otra vez.
- Demo ×100/×1000 exageran la rotación terrestre para mostrar desviaciones.
- La fórmula de precesión es aproximada para oscilaciones pequeñas.
- Las flechas 3D tienen escalas independientes y su longitud NO permite comparar fuerzas visualmente.
- La reproducción del servidor puede ser menos fluida que una animación web puramente local; si se bloquea o va lento, pausa y usa el deslizador de tiempo.

## Pruebas

```bash
python test_fisica.py
```

El código del modelo y de la interfaz siguen siendo editables en Python.
