# -*- coding: utf-8 -*-
"""PUNTO 21 — smoke test de tools/analisis_historial.py sobre los artefactos reales."""
import subprocess, sys
from pathlib import Path
import pytest

RAIZ = Path(__file__).resolve().parent.parent

@pytest.mark.skipif(not (RAIZ / 'data.json').exists() or not (RAIZ / 'setups_history.json').exists(),
                    reason='artefactos de produccion no presentes')
def test_analisis_historial_corre_sin_errores():
    r = subprocess.run([sys.executable, str(RAIZ / 'tools' / 'analisis_historial.py')],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert 'POR TIPO DE SETUP' in r.stdout
