# -*- coding: utf-8 -*-
"""Fixtures compartidas: carga del modulo scanner y paneles sinteticos."""
import sys, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent

@pytest.fixture(scope='session')
def sc():
    """Modulo scanner importado desde la raiz del repo (sin ejecutar main)."""
    spec = importlib.util.spec_from_file_location('scanner', RAIZ / 'scanner.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['scanner'] = mod
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture()
def panel_sintetico():
    """Universo sintetico de 100 tickers con sesgo comprador en las ultimas 30 sesiones."""
    np.random.seed(11)
    n = 260
    idx = pd.bdate_range('2025-07-01', periods=n)
    mkt = np.concatenate([np.random.normal(0.0002, 0.008, n-30),
                          np.random.normal(0.004, 0.006, 30)])
    cols = {f'T{k:03d}': pd.Series(50*np.cumprod(1 + mkt + np.random.normal(0, 0.01, n)), index=idx)
            for k in range(100)}
    bench = pd.Series(700*np.cumprod(1+mkt), index=idx)
    return pd.DataFrame(cols), bench
