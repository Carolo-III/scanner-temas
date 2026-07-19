# -*- coding: utf-8 -*-
"""PUNTO 21 — serie de amplitud (Punto 24): fecha de sesion, merge y tope."""
import pandas as pd

BREADTH = {'dispersion': {'ratio': 0.322}, 'mcclellan': -0.7, 'pct_sobre_mm20': 57.7,
           'pct_sobre_mm50': 63.1, 'pct_sobre_mm200': 69.0, 'avance': 155, 'descenso': 368,
           'nuevos_max_52s': 34, 'nuevos_min_52s': 2}
MACRO = {'^VIX': {'current': 18.77}}

def test_usa_fecha_de_sesion_del_cache(sc):
    sc._BREADTH_CACHE['ultima_sesion'] = '2026-07-17'
    e = sc.construir_entrada_breadth(BREADTH, MACRO, '18/07/2026 00:04')
    assert e['fecha'] == '2026-07-17'  # sesion del panel, no el reloj de la nocturna
    assert e['dispersion_ratio'] == 0.322 and e['vix'] == 18.77

def test_fallback_nocturna_tras_medianoche(sc):
    sc._BREADTH_CACHE.pop('ultima_sesion', None)
    e = sc.construir_entrada_breadth(BREADTH, MACRO, '18/07/2026 00:04')  # sabado 00:04
    assert e['fecha'] == '2026-07-17'  # retrocede al ultimo dia habil (viernes)

def test_fallback_ejecucion_vespertina_normal(sc):
    sc._BREADTH_CACHE.pop('ultima_sesion', None)
    e = sc.construir_entrada_breadth(BREADTH, MACRO, '16/07/2026 20:23')  # jueves tarde
    assert e['fecha'] == '2026-07-16'

def test_fallback_fin_de_semana(sc):
    sc._BREADTH_CACHE.pop('ultima_sesion', None)
    e = sc.construir_entrada_breadth(BREADTH, MACRO, '12/07/2026 10:45')  # domingo
    assert e['fecha'] == '2026-07-10'  # viernes anterior

def test_merge_sustituye_misma_fecha_y_ordena(sc):
    bh = [{'fecha': '2026-07-16', 'dispersion_ratio': 0.326},
          {'fecha': '2026-07-15', 'dispersion_ratio': 0.332}]
    nuevo = sc.merge_breadth_entry(bh, {'fecha': '2026-07-16', 'dispersion_ratio': 0.325})
    assert [e['fecha'] for e in nuevo] == ['2026-07-15', '2026-07-16']
    assert nuevo[-1]['dispersion_ratio'] == 0.325  # la reejecucion del dia sustituye

def test_merge_respeta_tope(sc):
    bh = [{'fecha': f'2026-01-{d:02d}'} for d in range(1, 29)]
    nuevo = sc.merge_breadth_entry(bh, {'fecha': '2026-02-01'}, max_entradas=10)
    assert len(nuevo) == 10 and nuevo[-1]['fecha'] == '2026-02-01'
