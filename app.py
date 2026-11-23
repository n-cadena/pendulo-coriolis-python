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

# Precalculamos fotogramas a intervalos de ~1.5 s simulados (161 fotogramas).
# Los datos que se visualizan se obtuvieron mediante integración real en Python.
frame_indices=np.unique(np.r_[np.arange(0,len(ts),19),len(ts)-1]).astype(int)

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

fig=go.Figure(data=datos_trazas(0))
frames=[go.Frame(name=str(int(idx)),data=datos_trazas(int(idx)),traces=list(range(3+len(keys)))) for idx in frame_indices]
fig.frames=frames
steps=[]
for idx in frame_indices:
    steps.append(dict(method="animate",label=fmt(ts[idx],0),args=[[str(int(idx))],{"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}]))
fig.update_layout(
    height=520,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor=C["fondo"],font=dict(color=C["texto"],size=11),
    showlegend=True,legend=dict(orientation="h",y=1,x=0,font=dict(size=10)),
    scene=dict(xaxis=dict(title="Este x (m)",range=[-1.15*p.longitud,1.15*p.longitud],backgroundcolor=C["panel"]),
               yaxis=dict(title="Norte y (m)",range=[-1.15*p.longitud,1.15*p.longitud],backgroundcolor=C["panel"]),
               zaxis=dict(title="Arriba z (m)",range=[-1.23*p.longitud,.2*p.longitud],backgroundcolor=C["panel"]),
               aspectmode="manual",aspectratio=dict(x=1,y=1,z=.75),camera=dict(eye=dict(x=1.4,y=1.5,z=1.0))),
    updatemenus=[dict(type="buttons",direction="left",showactive=False,x=0,y=-.11,xanchor="left",yanchor="top",buttons=[
        dict(label="▶ Reproducir",method="animate",args=[None,{"fromcurrent":True,"mode":"immediate","frame":{"duration":100,"redraw":True},"transition":{"duration":0}}]),
        dict(label="⏸ Pausar",method="animate",args=[[None],{"mode":"immediate","frame":{"duration":0,"redraw":False},"transition":{"duration":0}}])])],
    sliders=[dict(active=0,x=.06,y=-.22,len=.91,steps=steps,currentvalue={"prefix":"Tiempo simulado ≈ ","suffix":" s","font":{"color":C["texto"]}},tickcolor=C["gris"],font={"color":C["gris"],"size":9},minorticklen=0)],
    uirevision=f"config-{st.session_state.revision}")
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
st.info("El control de tiempo de esta sección consulta valores calculados en Python; el control dentro del gráfico reproduce la animación visual. Son controles independientes.",icon="ℹ️")

def estilo_2d(fig,height=320):
    fig.update_layout(height=height,paper_bgcolor=C["fondo"],plot_bgcolor=C["panel"],font=dict(color=C["texto"]),margin=dict(l=5,r=5,t=10,b=15),legend=dict(orientation="h",y=1.18,x=0),hovermode="closest")
    fig.update_xaxes(gridcolor="#344054")
    fig.update_yaxes(gridcolor="#344054")
    return fig

st.markdown("### 📈 Trayectorias calculadas en Python")
left,right=st.columns(2)
with left:
    plane=go.Figure()
    skip=3
    plane.add_trace(go.Scattergl(x=base[::skip,0],y=base[::skip,1],mode="lines",name="Sin Coriolis",line=dict(color=C["gris"],dash="dash")))
    plane.add_trace(go.Scattergl(x=ys[::skip,0],y=ys[::skip,1],mode="lines",name="Con configuración actual",line=dict(color=C["cian"])))
    plane.add_trace(go.Scatter(x=[ys[i,0]],y=[ys[i,1]],mode="markers",name="Punto seleccionado",marker=dict(color=C["amarillo"],size=9)))
    plane.update_xaxes(title="Este, x (m)")
    plane.update_yaxes(title="Norte, y (m)",scaleanchor="x",scaleratio=1)
    st.plotly_chart(estilo_2d(plane),use_container_width=True,config={"displaylogo":False})
with right:
    dis=go.Figure()
    dis.add_trace(go.Scattergl(x=ts[::3],y=ys[::3,0],mode="lines",name="Este x",line=dict(color=C["cian"])))
    dis.add_trace(go.Scattergl(x=ts[::3],y=ys[::3,1],mode="lines",name="Norte y",line=dict(color=C["rosa"])))
    dis.update_xaxes(title="Tiempo simulado (s)")
    dis.update_yaxes(title="Desplazamiento (m)")
    st.plotly_chart(estilo_2d(dis),use_container_width=True,config={"displaylogo":False})

with st.expander("∑ Ver las ecuaciones y su significado"):
    for name,items in GRUPOS:
        st.subheader(name)
        for title,formula,description in items:
            st.markdown(f"**{title}**")
            st.latex(formula)
            st.caption(description)
    st.caption("Sin rozamiento; gravedad efectiva constante; precesión aproximada para oscilaciones pequeñas.")
with st.expander("🐍 Cómo sustentar el código Python"):
    st.markdown("""El archivo **`fisica.py`** contiene las funciones `rotacion_local`, `condiciones_iniciales`, `magnitudes`, `ecuaciones` y `calcular`. **NumPy** opera con vectores y **SciPy DOP853** integra el movimiento. `app.py` controla la interfaz y entrega los resultados a Plotly. **Plotly únicamente anima en el navegador los resultados precalculados por Python**; no sustituye el cálculo físico.""")
st.caption("Modelo ideal · Sistema de ejes Este–Norte–Arriba · Python + SciPy + Streamlit + Plotly")

