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
    """Techo de evaluacion: ninguna evaluacion mas alla de 30 dias."""
    for e in _data().get('evaluaciones', []):
        if isinstance(e, dict) and e.get('dias') is not None:
            assert 1 <= e['dias'] <= 30, f"evaluacion fuera de rango: {e.get('ticker')} dias={e['dias']}"

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
