"""P106 (21/11/2026) — redaccion: sectores pasados por el codigo, percentiles sin
"historico", Supertrend persistente etiquetado, VIX casi plano, vencimiento mensual
y retirada de «Mercado bajista»."""
import inspect

import scanner as s


def test_vencimiento_mensual_y_trimestral_excluyentes():
    assert s.es_vencimiento_mensual('2026-11-20')        # tercer viernes de noviembre
    assert not s.es_vencimiento_mensual('2026-12-18')    # diciembre es trimestral
    assert s.es_vencimiento_trimestral('2026-12-18')
    assert not s.es_vencimiento_mensual('2026-11-13')    # segundo viernes
    assert not s.es_vencimiento_mensual(None)


def test_nivel_extremo_sin_historico():
    txt = s._texto_nivel_extremo({'nivel_extremo': {'alerta_nivel': True, 'percentil': 99.5,
                                                    'n': 200, 'referencia': 3.48}})
    assert 'extremo historico' not in txt
    assert 'ultimas 200 sesiones' in txt


def test_seguimiento_etiqueta_precio_nivel_y_sector():
    data = {'evaluaciones': [{
        'ticker': 'MLM', 'group': 'Materiales', 'fecha_setup': '2026-10-27', 'dias': 25,
        'precio_actual': 506.75, 'ret_pct': 1.0, 'resultado': 'abierto',
        'alerta_cmf': False, 'alerta_supertrend': True, 'supertrend_dias': 3,
        'supertrend_nivel': 556.11, 'supertrend_nivel_cruzado': None}]}
    txt = s.bloque_seguimiento(data)
    assert '[Materiales]' in txt
    assert 'precio actual $506.75' in txt
    assert 'retorno DESDE LA ENTRADA 1.0%' in txt
    assert 'nivel de Supertrend vigente: $556.11, que NO es el precio' in txt


def test_extendidos_llevan_sector():
    values = [{'ticker': 'CBOE', 'group': 'Finanzas', 'price': 250.0,
               'entry_range': {'tipo': 'Extendido', 'dist_ma50_pct': 23.1,
                               'trigger': 'AVISO: distancia extrema'}}]
    assert 'CBOE [Finanzas]' in s.bloque_extendidos(values)


def test_sin_mercado_bajista_ni_30_fijo():
    src = inspect.getsource(s)
    assert 'Mercado bajista. ' not in src
    assert 'Scores penalizados 30%' not in src
    assert 'sensor adelantado de aversion al riesgo mas fiable' not in src
