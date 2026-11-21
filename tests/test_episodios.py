# -*- coding: utf-8 -*-
"""Episodios en tools/analisis_rendimiento.py (21/11/2026): las altas repetidas de un
mismo valor en sesiones seguidas cuentan una vez."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('analisis_rendimiento',
                                               RAIZ / 'tools' / 'analisis_rendimiento.py')
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)


def test_altas_seguidas_son_un_episodio():
    # UPS: 02, 03, 04 y 09/11 (09 esta a 3 habiles del 04: mismo episodio);
    # 20/11 esta a 9 habiles del 09: episodio nuevo.
    apar = [('2026-11-02', 'UPS'), ('2026-11-03', 'UPS'), ('2026-11-04', 'UPS'),
            ('2026-11-09', 'UPS'), ('2026-11-20', 'UPS'), ('2026-11-03', 'MA')]
    ep = ar.inicios_de_episodio(apar)
    assert ep[('2026-11-09', 'UPS')] == ('2026-11-02', 'UPS')
    assert ep[('2026-11-20', 'UPS')] == ('2026-11-20', 'UPS')
    assert len(set(ep.values())) == 3


def test_filtrar_conserva_la_alta_mas_temprana_presente():
    apar = [('2026-11-02', 'UPS'), ('2026-11-03', 'UPS'), ('2026-11-04', 'UPS')]
    ep = ar.inicios_de_episodio(apar)
    # La primera alta (02/11) no tiene registro aqui: se conserva la del 03, no se pierde el episodio.
    regs = [{'fecha_setup': '2026-11-03', 'ticker': 'UPS', 'dias': 5},
            {'fecha_setup': '2026-11-03', 'ticker': 'UPS', 'dias': 10},
            {'fecha_setup': '2026-11-04', 'ticker': 'UPS', 'dias': 5}]
    out = ar.filtrar_episodios(regs, ep)
    assert [(r['fecha_setup'], r['dias']) for r in out] == [('2026-11-03', 5), ('2026-11-03', 10)]


@pytest.mark.skipif(not (RAIZ / 'checkpoints_history.json').exists(), reason='sin artefactos')
def test_analisis_rendimiento_corre_en_los_dos_modos():
    for extra in ([], ['--crudo']):
        r = subprocess.run([sys.executable, str(RAIZ / 'tools' / 'analisis_rendimiento.py'),
                            '--dir', str(RAIZ), '--sin-spy'] + extra,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-500:]
        assert ('MODO CRUDO' if extra else 'MODO EPISODIOS') in r.stdout
