# -*- coding: utf-8 -*-
"""PUNTO 21 — calc_market_breadth: MM20/MM50/MM200, McClellan y avance/descenso."""
import numpy as np
import pandas as pd

def test_mcclellan_coincide_con_calculo_manual(sc, panel_sintetico):
    uni, bench = panel_sintetico
    b = sc.calc_market_breadth(uni, bench)
    pff = uni.ffill()
    rh = pff.pct_change().iloc[-90:]
    adv = (rh > 0).sum(axis=1); dec = (rh < 0).sum(axis=1)
    rana = (1000.0 * (adv - dec) / (adv + dec).replace(0, np.nan)).dropna()
    esperado = round(float(rana.ewm(span=19, adjust=False).mean().iloc[-1]
                           - rana.ewm(span=39, adjust=False).mean().iloc[-1]), 1)
    assert b['mcclellan'] == esperado

def test_mcclellan_signo_segun_sesgo(sc, panel_sintetico):
    uni, bench = panel_sintetico
    assert sc.calc_market_breadth(uni, bench)['mcclellan'] > 0  # sesgo comprador plantado
    np.random.seed(12)
    n = 260; idx = pd.bdate_range('2025-07-01', periods=n)
    mkt = np.concatenate([np.random.normal(0.0002, 0.008, n-30),
                          np.random.normal(-0.004, 0.006, 30)])
    uni_bajista = pd.DataFrame({f'B{k:03d}': pd.Series(50*np.cumprod(1+mkt+np.random.normal(0,0.01,n)), index=idx)
                                for k in range(100)})
    assert sc.calc_market_breadth(uni_bajista, bench)['mcclellan'] < 0

def test_pct_sobre_medias_panel_construido(sc, panel_sintetico):
    """Panel estrictamente creciente: el 100% debe estar sobre todas sus medias."""
    _, bench = panel_sintetico
    n = 260; idx = pd.bdate_range('2025-07-01', periods=n)
    creciente = pd.DataFrame({f'C{k}': pd.Series(np.linspace(50, 100, n), index=idx)
                              for k in range(20)})
    b = sc.calc_market_breadth(creciente, bench)
    assert b['pct_sobre_mm20'] == 100.0
    assert b['pct_sobre_mm50'] == 100.0
    assert b['pct_sobre_mm200'] == 100.0

def test_avance_descenso_suman_universo_activo(sc, panel_sintetico):
    uni, bench = panel_sintetico
    b = sc.calc_market_breadth(uni, bench)
    assert b['avance'] + b['descenso'] <= len(uni.columns)
    assert b['avance'] >= 0 and b['descenso'] >= 0
