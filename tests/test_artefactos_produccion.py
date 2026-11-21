# -*- coding: utf-8 -*-
"""PUNTO 21 — Invariantes sobre los artefactos reales de produccion (data.json).

Estos tests validan que los datos que la web y el informe consumen cumplen las
reglas del sistema. Se saltan limpiamente si data.json no existe (repo recien
clonado sin datos), de modo que el CI funcione tambien en ese caso.
"""
import json
from pathlib import Path
import pytest

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / 'data.json'

pytestmark = pytest.mark.skipif(not DATA.exists(), reason='data.json no presente')

def _data():
    return json.loads(DATA.read_text())

def test_desglose_ranking_suma_exacta():
    """Punto 19: los cuatro componentes suman el ranking (tolerancia de redondeo)."""
    for v in _data().get('values', []):
        if v.get('ranking') is not None and v.get('ranking_desglose'):
            suma = round(sum(v['ranking_desglose'].values()), 1)
            assert abs(suma - v['ranking']) < 0.25, \
                f"{v['ticker']}: ranking {v['ranking']} != suma desglose {suma}"

def test_techo_30_dias_en_evaluaciones():
    """Techo de evaluacion: nada ABIERTO mas alla de 30 dias.

    Ajustado el 21/11/2026. Desde el P102 (06/11) un setup que supera el dia 30 se
    resuelve UNA vez como 'caducado' en la primera ejecucion en que dias > 30, y esa
    evaluacion lleva su dias real: 31-33 con fines de semana y el lunes sin ejecucion.
    La version antigua exigia dias <= 30 y estaba en rojo desde el 07/11 sin que el
    codigo fallase. Lo que sigue prohibido es lo que el test protegia: que pasado el
    horizonte se vigile o se alerte una posicion abierta.
    """
    TOPE_HOLGURA = 35   # dia 30 + huecos entre ejecuciones (fin de semana, lunes, festivo)
    for e in _data().get('evaluaciones', []):
        if not isinstance(e, dict) or e.get('dias') is None:
            continue
        tk, d = e.get('ticker'), e['dias']
        assert d >= 1, f'evaluacion con dias < 1: {tk} dias={d}'
        if d > 30:
            assert e.get('resultado') in ('caducado', 'stop', 'target'), \
                f'posicion ABIERTA evaluada pasado el horizonte: {tk} dias={d} resultado={e.get("resultado")}'
            assert d <= TOPE_HOLGURA, f'evaluacion demasiado lejos del horizonte: {tk} dias={d}'


def test_resoluciones_dentro_del_horizonte():
    """Un stop u objetivo solo cuenta si se toco dentro de los 30 dias del setup."""
    import datetime as dt
    ruta = RAIZ / 'resoluciones_history.json'
    if not ruta.exists():
        pytest.skip('resoluciones_history.json no presente')
    for r in json.loads(ruta.read_text()):
        if r.get('resultado') in ('stop', 'target') and r.get('fecha_resolucion'):
            d = (dt.date.fromisoformat(r['fecha_resolucion'])
                 - dt.date.fromisoformat(r['fecha_setup'])).days
            assert 0 <= d <= 30, f"{r.get('ticker')} {r['fecha_setup']}: {r['resultado']} a {d} dias"

def test_sin_restos_de_campos_renombrados():
    """El campo fmp_eps_8q fue renombrado a fmp_eps_q el 13/07/2026."""
    for tk, f in _data().get('fundamentales', {}).items():
        assert 'fmp_eps_8q' not in f, f'{tk} conserva el campo antiguo fmp_eps_8q'

def test_marcas_sobrecompra_coherentes_con_rsi():
    """Punto 18: la marca solo puede existir en pullbacks con RSI >= 70."""
    for v in _data().get('values', []):
        er = v.get('entry_range') or {}
        if er.get('rsi_sobrecompra'):
            assert str(er.get('tipo', '')).startswith('Pullback')
            assert v.get('rsi') is not None and v['rsi'] >= 70, \
                f"{v['ticker']}: marca de sobrecompra con RSI {v.get('rsi')}"
