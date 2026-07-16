# -*- coding: utf-8 -*-
"""PUNTO 21 — calendario macro (Punto 23): integridad de fechas y proximidad.

Las fechas del calendario fueron verificadas el 15/07/2026 contra las fuentes
primarias (federalreserve.gov y bls.gov); estos tests protegen su integridad
estructural y la logica de proximidad, no re-verifican las fuentes.
"""
import pandas as pd

def test_calendario_bien_formado(sc):
    cal = sc.get_calendario_macro()
    assert set(k.split(' (')[0] for k in cal) == {'FOMC', 'IPC', 'NFP'}
    for evento, fechas in cal.items():
        assert len(fechas) >= 8, f'{evento}: solo {len(fechas)} fechas'
        parseadas = [pd.Timestamp(f) for f in fechas]
        assert parseadas == sorted(parseadas), f'{evento}: fechas desordenadas'
        assert all(f.year == 2026 for f in parseadas)
        assert all(f.dayofweek < 5 for f in parseadas), f'{evento}: fecha en fin de semana'

def test_fechas_ancla_conocidas(sc):
    """Fechas contrastadas a mano con las fuentes el dia de la implementacion."""
    cal = sc.get_calendario_macro()
    fomc = [f for k, v in cal.items() if k.startswith('FOMC') for f in v]
    assert '2026-07-29' in fomc and '2026-09-16' in fomc and '2026-12-09' in fomc
    ipc = [f for k, v in cal.items() if k.startswith('IPC') for f in v]
    assert '2026-07-14' in ipc and '2026-08-12' in ipc
    nfp = [f for k, v in cal.items() if k.startswith('NFP') for f in v]
    assert '2026-08-07' in nfp

def test_proximidad_fomc_a_dos_dias(sc):
    """Lunes 27/07: la decision FOMC del miercoles 29/07 esta a 2 dias habiles."""
    evs = sc.check_eventos_macro(umbral_dias=2, hoy='2026-07-27')
    assert any(e['evento'].startswith('FOMC') and e['dias_habiles'] == 2 for e in evs)

def test_evento_hoy_dia_cero(sc):
    evs = sc.check_eventos_macro(umbral_dias=2, hoy='2026-08-12')
    assert any(e['evento'].startswith('IPC') and e['dias_habiles'] == 0 for e in evs)

def test_semana_tranquila_sin_eventos(sc):
    """20/07/2026 (lunes): sin FOMC/IPC/NFP en 2 dias habiles."""
    assert sc.check_eventos_macro(umbral_dias=2, hoy='2026-07-20') == []

def test_fin_de_semana_no_infla_distancia(sc):
    """Viernes 07/08 hay NFP; el jueves 06/08 esta a 1 dia habil, no mas."""
    evs = sc.check_eventos_macro(umbral_dias=2, hoy='2026-08-06')
    assert any(e['evento'].startswith('NFP') and e['dias_habiles'] == 1 for e in evs)

def test_calendario_agotado_avisa(sc, capsys):
    evs = sc.check_eventos_macro(umbral_dias=2, hoy='2027-03-01')
    assert evs == []
    assert 'AVISO CALENDARIO MACRO' in capsys.readouterr().out

def test_formato_summary(sc):
    evs = sc.check_eventos_macro(umbral_dias=2, hoy='2026-07-28')
    txt = sc.formato_eventos_macro_summary(evs)
    assert 'EVENTOS MACRO PROGRAMADOS PROXIMOS' in txt and '2026-07-29' in txt
    assert sc.formato_eventos_macro_summary([]) == ''
