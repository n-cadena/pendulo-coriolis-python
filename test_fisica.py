"""Pruebas ejecutables de la física y comparación con el archivo original."""
from dataclasses import replace
import importlib.util
import sys
from pathlib import Path
import numpy as np
from fisica import Parametros, calcular, magnitudes, rotacion_local, precesion_grados_hora


def ejecutar():
    p = Parametros(angulo=5., coriolis=False)
    t, y = calcular(p, duracion=15., dt=.04)
    assert np.max(np.abs(np.linalg.norm(y[:,:3],axis=1)-p.longitud)) < 1e-9
    assert np.max(np.abs(np.einsum("ij,ij->i",y[:,:3],y[:,3:]))) < 1e-9
    en = np.array([magnitudes(yi,p)["E"] for yi in y])
    assert np.max(np.abs(en-en[0])) < 1e-6
    _,y_m = calcular(replace(p,masa=5.),duracion=15.,dt=.04)
    assert np.max(np.abs(y_m[:,:3]-y[:,:3])) < 1e-6
    assert np.linalg.norm(magnitudes(y[20],p)["Fc"]) == 0
    assert precesion_grados_hora(p) == 0
    assert abs(rotacion_local(0.)[2]) < 1e-15
    assert rotacion_local(45.)[2] > 0 and rotacion_local(-45.)[2] < 0
    original = Path(__file__).with_name("simulador_original_escritorio.py")
    spec = importlib.util.spec_from_file_location("pendulo_original",original)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    for lat, mult, ang, v0 in [(45,1,25,0),(60,100,20,.5),(-35,1000,15,1.)]:
        pp = Parametros(latitud=lat,multiplicador=mult,angulo=ang,velocidad=v0)
        op = modulo.Parametros(**vars(pp))
        t1,y1=calcular(pp,duracion=12,dt=.08)
        t2,y2=modulo.calcular(op,duracion=12,dt=.08)
        assert np.array_equal(t1,t2)
        assert np.allclose(y1,y2,rtol=0,atol=1e-12)
        for j in [0,30,100]:
            a,b=magnitudes(y1[j],pp),modulo.magnitudes(y2[j],op)
            for k in ['Fg','Fc','T','a','v','tension','Ec','Ep','E']:
                assert np.allclose(a[k],b[k],rtol=0,atol=1e-11),k
    print("OK: física y método numérico coinciden con el programa original.")


if __name__=="__main__":
    ejecutar()
