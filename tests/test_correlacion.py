# -*- coding: utf-8 -*-
"""PUNTO 21 — calc_correlacion_candidatos (Punto 17): estructura, betas y bordes."""
import numpy as np
import pandas as pd

def _cargar_cache(sc):
    np.random.seed(42)
    n = 90; idx = pd.bdate_range('2026-03-01', periods=n)
    factor = np.random.normal(0, 0.015, n)
    mkt = np.random.normal(0.0005, 0.010, n)
    def serie(base, beta, carga, ruido):
        r = beta*mkt + carga*factor + np.random.normal(0, ruido, n)
        return pd.Series(base*np.cumprod(1+r), index=idx)
    sc._CLOSES_CACHE.clear()
    sc._CLOSES_CACHE['__BENCH__'] = pd.Series(700*np.cumprod(1+mkt), index=idx)
    sc._CLOSES_CACHE['AAA'] = serie(90, 1.4, 1.0, 0.008)
    sc._CLOSES_CACHE['BBB'] = serie(480, 1.1, 0.8, 0.006)
    sc._CLOSES_CACHE['DDD'] = serie(95, 0.5, 0.0, 0.010)

def test_recupera_estructura_plantada(sc):
    _cargar_cache(sc)
    cc = sc.calc_correlacion_candidatos(['AAA', 'BBB', 'DDD'])
    assert cc is not None and cc['n_sesiones'] >= 30
    # AAA-BBB comparten factor -> par alto; DDD independiente -> fuera
    pares = {(a, b) for a, b, _ in cc['pares_altos']}
    assert ('AAA', 'BBB') in pares
    assert all('DDD' not in p for p in pares)
    # betas ordenadas segun lo plantado (1.4 > 1.1 > 0.5)
    assert cc['betas']['AAA'] > cc['betas']['BBB'] > cc['betas']['DDD']

def test_bordes_degradan_a_none_o_sin_betas(sc):
    _cargar_cache(sc)
    assert sc.calc_correlacion_candidatos(['AAA', 'NOEXISTE']) is None  # <2 series
    sc._CLOSES_CACHE['CORTO'] = sc._CLOSES_CACHE['AAA'].tail(10)
    assert sc.calc_correlacion_candidatos(['AAA', 'CORTO']) is None    # muestra corta
    del sc._CLOSES_CACHE['__BENCH__']
    cc = sc.calc_correlacion_candidatos(['AAA', 'BBB'])
    assert cc is not None and cc['betas'] == {}                        # sin bench: corr sin betas

def test_formato_summary_contiene_avisos(sc):
    _cargar_cache(sc)
    cc = sc.calc_correlacion_candidatos(['AAA', 'BBB', 'DDD'])
    valid = [{'ticker': t, 'group': g} for t, g in
             [('AAA', 'TemaX'), ('BBB', 'TemaX'), ('DDD', 'TemaY')]]
    txt = sc.formato_correlacion_summary(cc, valid)
    assert 'CORRELACION ENTRE CANDIDATOS' in txt
    assert 'AVISO CONCENTRACION' in txt
    assert 'CONCENTRACION SECTORIAL' in txt and 'TemaX' in txt
