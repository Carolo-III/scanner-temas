# -*- coding: utf-8 -*-
"""PUNTO 21 — check_frescura_panel (Punto 22): retraso y cobertura."""
import numpy as np
import pandas as pd

def _panel_hasta(fecha_fin, n=60, n_tickers=10, nan_ultima_fila=0):
    idx = pd.bdate_range(end=fecha_fin, periods=n)
    df = pd.DataFrame({f'T{k}': np.linspace(50, 60, n) for k in range(n_tickers)}, index=idx)
    for k in range(nan_ultima_fila):
        df.iloc[-1, k] = np.nan
    return df

def test_panel_fresco_sin_retraso(sc, capsys):
    hoy_ny = pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize()
    ult_habil = hoy_ny if hoy_ny.dayofweek < 5 else hoy_ny - pd.offsets.BDay(1)
    r = sc.check_frescura_panel(_panel_hasta(ult_habil))
    assert r['retraso_habiles'] == 0
    assert r['pct_tickers_frescos'] == 100.0
    assert 'AVISO' not in capsys.readouterr().out

def test_panel_rancio_avisa_fuerte(sc, capsys):
    hoy_ny = pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize()
    vieja = hoy_ny - pd.offsets.BDay(5)
    r = sc.check_frescura_panel(_panel_hasta(vieja))
    assert r['retraso_habiles'] >= 2
    assert 'AVISO FUERTE' in capsys.readouterr().out

def test_descarga_parcial_avisa(sc, capsys):
    hoy_ny = pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize()
    ult_habil = hoy_ny if hoy_ny.dayofweek < 5 else hoy_ny - pd.offsets.BDay(1)
    r = sc.check_frescura_panel(_panel_hasta(ult_habil, n_tickers=10, nan_ultima_fila=2))
    assert r['pct_tickers_frescos'] == 80.0
    assert 'descarga parcial' in capsys.readouterr().out

def test_panel_vacio_devuelve_none(sc):
    assert sc.check_frescura_panel(pd.DataFrame()) is None
    assert sc.check_frescura_panel(None) is None
