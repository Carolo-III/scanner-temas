# -*- coding: utf-8 -*-
"""
PUNTO 25 (18/07/2026) — Analisis del historial de setups del Scanner de Temas Emergentes.

Herramienta A DEMANDA, fuera del camino nocturno (riesgo cero para produccion): interroga
los datos que el scanner acumula para alimentar las dos decisiones agendadas:
  - Umbral RSI del Punto 18 (¿rinden peor los pullbacks nacidos con RSI 65-70?)
  - Punto 16 / septiembre (¿rinden peor los setups nacidos en descorrelacion extrema,
    o con Supertrend bajista al crearse?)

Uso (desde la raiz del repo, con data.json y setups_history.json presentes):
    python tools/analisis_historial.py

Limitaciones honestas:
  - Las evaluaciones viven en data.json y cubren solo la ventana de 30 dias vigente;
    los resultados de setups mas antiguos ya no son reconstruibles desde aqui.
  - rsi_al_crear y dispersion_al_crear existen solo en setups creados desde el
    18/07/2026; los anteriores aparecen como "sin dato".
  - Con muestras pequeñas (n<10) los porcentajes son orientativos, no evidencia.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def cargar(nombre):
    ruta = RAIZ / nombre
    if not ruta.exists():
        print(f'AVISO: {nombre} no encontrado en {RAIZ} — ejecutar desde la raiz del repo')
        return None
    return json.loads(ruta.read_text())


def banda_rsi(rsi):
    if rsi is None: return 'sin dato'
    if rsi < 50: return '<50'
    if rsi < 65: return '50-65'
    if rsi < 70: return '65-70'
    return '>=70'


def banda_dispersion(ratio):
    if ratio is None: return 'sin dato'
    if ratio < 0.35: return '<0.35 (extrema)'
    if ratio < 0.45: return '0.35-0.45'
    return '>=0.45'


def tabla(titulo, grupos):
    """grupos: dict clave -> lista de evaluaciones (dicts con resultado/ret/dias)."""
    print(f'\n== {titulo} ==')
    cab = f'{"segmento":<22} {"setups":>6} {"checkp.":>7} {"stops":>5} {"targets":>7} {"%stop":>6} {"ret medio":>9}'
    print(cab); print('-' * len(cab))
    for clave in sorted(grupos):
        evs = grupos[clave]
        setups = {(e.get("ticker"), e.get("fecha_setup")) for e in evs}
        cps = [e for e in evs if e.get('checkpoint')]
        stops = sum(1 for e in cps if e.get('resultado') == 'stop')
        targets = sum(1 for e in cps if e.get('resultado') == 'target')
        rets = [e.get('ret_pct') for e in evs if isinstance(e.get('ret_pct'), (int, float))]
        pct_stop = f'{100*stops/len(cps):.0f}%' if cps else '-'
        ret_medio = f'{sum(rets)/len(rets):+.1f}%' if rets else '-'
        print(f'{str(clave):<22} {len(setups):>6} {len(cps):>7} {stops:>5} {targets:>7} {pct_stop:>6} {ret_medio:>9}')
    print('(muestras n<10: orientativo, no evidencia)')


def main():
    data = cargar('data.json')
    sh = cargar('setups_history.json')
    if data is None or sh is None:
        sys.exit(1)

    # Metadatos de cada setup por (ticker, fecha) — el contexto al crear
    meta = {}
    for dia in sh:
        for s in dia.get('setups', []):
            meta[(s.get('ticker'), dia.get('date'))] = s

    evs = [e for e in data.get('evaluaciones', []) if isinstance(e, dict)]
    print(f'Evaluaciones en ventana: {len(evs)} | setups en historico: {sum(len(d.get("setups",[])) for d in sh)} dias={len(sh)}')

    por_tipo, por_mes, por_st, por_rsi, por_disp = (defaultdict(list) for _ in range(5))
    for e in evs:
        clave = (e.get('ticker'), e.get('fecha_setup'))
        m = meta.get(clave, {})
        por_tipo[e.get('tipo') or m.get('tipo') or '?'].append(e)
        por_mes[(e.get('fecha_setup') or '?')[:7]].append(e)
        por_st[m.get('supertrend_al_crear') or 'sin dato'].append(e)
        por_rsi[banda_rsi(m.get('rsi_al_crear'))].append(e)
        por_disp[banda_dispersion(m.get('dispersion_al_crear'))].append(e)

    tabla('POR TIPO DE SETUP', por_tipo)
    tabla('POR MES DE COHORTE', por_mes)
    tabla('POR SUPERTREND AL CREAR (decision sept., junto a P16)', por_st)
    tabla('POR RSI AL CREAR (decision umbral P18; solo setups desde 18/07)', por_rsi)
    tabla('POR DISPERSION AL CREAR (decision P16/sept.; solo desde 18/07)', por_disp)

    bh = cargar('breadth_history.json')
    if bh:
        print(f'\n== SERIE DE AMPLITUD ({len(bh)} sesiones) ==')
        # PUNTO 30 (21/07/2026) — resumen estadístico de la serie de dispersión para la
        # calibracion de septiembre (P16): tendencia, rango y si la serie ha rebotado alguna
        # vez desde que se instrumentó. Con n pequeño es orientativo, no concluyente.
        ratios = [e['dispersion_ratio'] for e in bh if e.get('dispersion_ratio') is not None]
        if ratios:
            tendencia = 'bajista' if ratios[-1] < ratios[0] else 'alcista'
            n_min = sum(1 for i in range(1, len(ratios)) if ratios[i] < ratios[i-1])
            n_max = len(ratios) - 1 - n_min
            print(f'  Dispersion: min={min(ratios):.3f} | max={max(ratios):.3f} | '
                  f'actual={ratios[-1]:.3f} | tendencia={tendencia} | '
                  f'sesiones bajando={n_min}/{len(ratios)-1} ({"sin ningun rebote" if n_max==0 else f"{n_max} rebote(s)"})')
        mccs = [e['mcclellan'] for e in bh if e.get('mcclellan') is not None]
        if mccs:
            print(f'  McClellan: min={min(mccs):.1f} | max={max(mccs):.1f} | actual={mccs[-1]:.1f} | '
                  f'sesiones negativas={sum(1 for m in mccs if m<0)}/{len(mccs)}')
        vix_vals = [e['vix'] for e in bh if e.get('vix') is not None]
        if vix_vals:
            print(f'  VIX: min={min(vix_vals):.2f} | max={max(vix_vals):.2f} | actual={vix_vals[-1]:.2f}')
        print()
        print(f'  {"Fecha":<12} {"Disp":>6} {"McC":>7} {"MM20":>6} {"A/D":>10} {"VIX":>6} {"Tipo":<12}')
        print('  ' + '-'*58)
        for e in bh[-15:]:
            tipo = ('cierre' if e.get('es_cierre') else 'intradía') if 'es_cierre' in e else e.get('fuente','cierre')
            ad = f"{e.get('avance','?')}/{e.get('descenso','?')}"
            print(f"  {str(e.get('fecha','')):<12} {str(e.get('dispersion_ratio','?')):>6} "
                  f"{str(e.get('mcclellan','?')):>7} {str(e.get('pct_sobre_mm20','?')):>6} "
                  f"{ad:>10} {str(e.get('vix','?')):>6} {tipo:<12}")


if __name__ == '__main__':
    main()
