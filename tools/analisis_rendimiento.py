# -*- coding: utf-8 -*-
"""
tools/analisis_rendimiento.py — ¿APORTA VALOR EL SCANNER? (14/08/2026)

El sistema lleva meses acumulando datos que nadie ha mirado: 514 checkpoints y 52
resoluciones al 14/08/2026. Esta herramienta no añade ninguna medida nueva: solo lee
lo que ya esta persistido y responde a cuatro preguntas que hasta ahora no tenian
respuesta numerica.

  1. ¿Bate al SPY? Retorno de los setups a 5/10/20 sesiones contra el indice en la
     MISMA ventana natural. Es la pregunta que importa.
  2. ¿Los stops saltan demasiado pronto? De los setups que acabaron en stop, cuantos
     estaban en verde en algun checkpoint previo.
  3. ¿Discrimina el sistema? Rendimiento por tipo de setup, por grupo y —donde haya
     datos— por nivel de riesgo y RSI de entrada.
  4. ¿Cual es la foto real de las resoluciones? Sin el sesgo de mirar solo lo cerrado.

POR QUE LOS CHECKPOINTS Y NO LAS RESOLUCIONES
Las resoluciones tienen sesgo de seleccion: solo cuentan los setups que se resolvieron,
y los stops se resuelven rapido mientras los objetivos tardan, asi que sobrerrepresentan
las perdidas. Los checkpoints miden TODOS los setups a horizonte fijo (5/10/20 sesiones),
resueltos o no, que es la comparacion honesta. La estadistica de rendimiento debe
filtrar checkpoint==true, y eso es exactamente lo que hay en checkpoints_history.json.

LIMITACIONES QUE NO SE PUEDEN ARREGLAR CON MAS CODIGO
- La muestra esta dominada por julio de 2026, un crash de momentum (SOX -25% desde
  maximos de junio, peor mes desde 2008). Juzgar un sistema de momentum con esa ventana
  es juzgar un paraguas en un huracan. Los numeros son un PRIMER VISTAZO, no un veredicto.
- Con ~500 checkpoints y ~50 resoluciones no hay significacion estadistica para casi
  ninguna comparacion por segmentos. Los tamaños de muestra se imprimen SIEMPRE al lado
  de cada media precisamente para que no se lean como conclusiones.
- setups_history.json es una ventana movil de 30 dias: riesgo, rsi_al_crear y
  dispersion_al_crear solo existen para los setups recientes. La cobertura real se
  imprime en la seccion correspondiente en vez de asumirse.
- El SPY se compara sobre la ventana NATURAL (fecha de setup + N dias naturales), no
  sobre sesiones exactas de mercado. Es una aproximacion, suficiente para el orden de
  magnitud pero no para decimas.

USO
    python tools/analisis_rendimiento.py
    python tools/analisis_rendimiento.py --dir /ruta/a/los/json
    python tools/analisis_rendimiento.py --sin-spy     (omite la descarga de yfinance)
"""

import argparse
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

HORIZONTES = (5, 10, 20)
SEP = '=' * 78


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------

def cargar(directorio, nombre, por_defecto):
    """Lee un JSON del repo. Devuelve por_defecto si no existe o no es legible.

    Se avisa explicitamente de lo que falta: un analisis con menos ficheros de los
    esperados debe decirlo, no producir tablas mas cortas en silencio.
    """
    ruta = os.path.join(directorio, nombre)
    if not os.path.exists(ruta):
        print(f'  AVISO: no se encontro {nombre} — las secciones que dependan de el se omiten.')
        return por_defecto
    try:
        with open(ruta, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception as e:
        print(f'  AVISO: {nombre} no se pudo leer ({type(e).__name__}: {e}).')
        return por_defecto


def descargar_spy(fecha_min, fecha_max):
    """Serie de cierres del SPY para comparar. Devuelve {} si no hay yfinance o red.

    Sin benchmark el resto del analisis sigue siendo util, asi que un fallo aqui
    degrada la seccion 2 pero no aborta nada.
    """
    try:
        import yfinance as yf
        ini = (datetime.strptime(fecha_min, '%Y-%m-%d') - timedelta(days=10)).strftime('%Y-%m-%d')
        fin = (datetime.strptime(fecha_max, '%Y-%m-%d') + timedelta(days=40)).strftime('%Y-%m-%d')
        df = yf.download('SPY', start=ini, end=fin, progress=False, auto_adjust=True)
        if df is None or df.empty:
            print('  AVISO: la descarga del SPY vino vacia — se omite la comparacion con el indice.')
            return {}
        cierres = df['Close']
        if hasattr(cierres, 'columns'):        # yfinance puede devolver MultiIndex
            cierres = cierres.iloc[:, 0]
        return {d.strftime('%Y-%m-%d'): float(v) for d, v in cierres.items()}
    except Exception as e:
        print(f'  AVISO: no se pudo descargar el SPY ({type(e).__name__}) — se omite la comparacion.')
        return {}


def spy_retorno(spy, fecha_ini, dias):
    """Retorno % del SPY entre fecha_ini y fecha_ini+dias naturales.

    Si alguno de los dos extremos cae en fin de semana o festivo, se busca hacia
    adelante hasta 5 dias. Devuelve None si no hay datos para ese tramo.
    """
    if not spy:
        return None

    def cierre_desde(f):
        d = datetime.strptime(f, '%Y-%m-%d')
        for _ in range(6):
            clave = d.strftime('%Y-%m-%d')
            if clave in spy:
                return spy[clave]
            d += timedelta(days=1)
        return None

    p0 = cierre_desde(fecha_ini)
    p1 = cierre_desde((datetime.strptime(fecha_ini, '%Y-%m-%d') + timedelta(days=dias)).strftime('%Y-%m-%d'))
    if p0 is None or p1 is None or p0 == 0:
        return None
    return (p1 / p0 - 1) * 100


# --------------------------------------------------------------------------
# Utilidades de presentacion
# --------------------------------------------------------------------------

def resumen(valores):
    """(n, media, mediana, % positivos) tolerando listas vacias o de un elemento."""
    vals = [v for v in valores if isinstance(v, (int, float))]
    if not vals:
        return 0, None, None, None
    n = len(vals)
    media = statistics.mean(vals)
    mediana = statistics.median(vals)
    pct_pos = 100.0 * sum(1 for v in vals if v > 0) / n
    return n, media, mediana, pct_pos


def fmt(x, dec=1, sufijo=''):
    return '   n/d' if x is None else f'{x:+.{dec}f}{sufijo}'


def linea_segmento(etiqueta, valores, ancho=26, minimo=3):
    """Una fila de tabla por segmento. Marca los segmentos con muestra insuficiente.

    El aviso va PEGADO al dato, no en una nota al pie: una media de 3 observaciones
    no debe poder leerse como un hallazgo.
    """
    n, media, mediana, pct = resumen(valores)
    if n == 0:
        return f'  {etiqueta:<{ancho}} sin datos'
    marca = '  ⚠ muestra minima' if n < minimo else ''
    return (f'  {etiqueta:<{ancho}} n={n:<4} media {fmt(media)}%   '
            f'mediana {fmt(mediana)}%   positivos {pct:5.1f}%{marca}')


# --------------------------------------------------------------------------
# Secciones
# --------------------------------------------------------------------------

def seccion_cobertura(checkpoints, resoluciones, setups_hist):
    print(SEP)
    print('1. COBERTURA DE LOS DATOS')
    print(SEP)
    fechas = sorted({c.get('fecha_setup') for c in checkpoints if c.get('fecha_setup')})
    setups_unicos = {(c.get('fecha_setup'), c.get('ticker')) for c in checkpoints}
    print(f'  Checkpoints:          {len(checkpoints)} registros sobre {len(setups_unicos)} setups unicos')
    if fechas:
        print(f'  Rango de setups:      {fechas[0]} .. {fechas[-1]}  ({len(fechas)} fechas distintas)')
    por_dias = defaultdict(int)
    for c in checkpoints:
        por_dias[c.get('dias')] += 1
    print('  Por horizonte:        ' + ' | '.join(f'{d} dias: {por_dias.get(d, 0)}' for d in HORIZONTES))
    print(f'  Resoluciones:         {len(resoluciones)}')
    dias_setups = {h.get('date') for h in setups_hist if isinstance(h, dict)}
    print(f'  setups_history:       {len(dias_setups)} fechas (ventana movil de 30 dias — '
          'el contexto al crear solo existe para las recientes)')
    return fechas


def seccion_vs_spy(checkpoints, spy):
    print()
    print(SEP)
    print('2. ¿BATE AL SPY? — checkpoints a horizonte fijo contra el indice')
    print(SEP)
    print('  El SPY se mide en la MISMA ventana natural desde la fecha del setup.')
    print('  "Exceso" = retorno del setup menos retorno del indice en ese tramo.')
    print()
    hay_spy = bool(spy)
    for dias in HORIZONTES:
        grupo = [c for c in checkpoints if c.get('dias') == dias]
        rets = [c.get('ret_pct') for c in grupo]
        n, media, mediana, pct = resumen(rets)
        if n == 0:
            print(f'  {dias:>2} sesiones: sin datos')
            continue
        print(f'  {dias:>2} sesiones  n={n}')
        print(f'     Setups     media {fmt(media)}%   mediana {fmt(mediana)}%   positivos {pct:5.1f}%')
        if not hay_spy:
            print('     SPY        no disponible')
            continue
        excesos, spy_rets = [], []
        for c in grupo:
            r_spy = spy_retorno(spy, c.get('fecha_setup'), dias)
            if r_spy is None or not isinstance(c.get('ret_pct'), (int, float)):
                continue
            spy_rets.append(r_spy)
            excesos.append(c['ret_pct'] - r_spy)
        n_s, media_s, mediana_s, _ = resumen(spy_rets)
        n_e, media_e, mediana_e, pct_e = resumen(excesos)
        if n_s:
            print(f'     SPY        media {fmt(media_s)}%   mediana {fmt(mediana_s)}%   (n={n_s} emparejados)')
            print(f'     EXCESO     media {fmt(media_e)}%   mediana {fmt(mediana_e)}%   '
                  f'baten al indice {pct_e:5.1f}%')
        else:
            print('     SPY        no se pudo emparejar ninguna fecha')
        print()


def seccion_stops(checkpoints, resoluciones):
    print(SEP)
    print('3. ¿SALTAN LOS STOPS DEMASIADO PRONTO?')
    print(SEP)
    print('  De los setups que acabaron en STOP, se mira si en algun checkpoint previo')
    print('  estaban en positivo. Un stop que corta una posicion que iba ganando puede')
    print('  ser un stop demasiado ajustado, no una mala seleccion.')
    print()
    stops = {(r.get('fecha_setup'), r.get('ticker')): r
             for r in resoluciones if r.get('resultado') == 'stop'}
    if not stops:
        print('  Sin resoluciones en stop registradas.')
        return
    por_setup = defaultdict(list)
    for c in checkpoints:
        por_setup[(c.get('fecha_setup'), c.get('ticker'))].append(c)
    verdes, rojos, sin_datos, mejores = 0, 0, 0, []
    for clave in stops:
        chks = por_setup.get(clave, [])
        rets = [c.get('ret_pct') for c in chks if isinstance(c.get('ret_pct'), (int, float))]
        if not rets:
            sin_datos += 1
            continue
        mejor = max(rets)
        mejores.append(mejor)
        if mejor > 0:
            verdes += 1
        else:
            rojos += 1
    total = verdes + rojos
    print(f'  Setups cerrados en stop:            {len(stops)}')
    print(f'  Con checkpoints disponibles:        {total} (sin datos: {sin_datos})')
    if total:
        print(f'  Estuvieron en VERDE antes del stop: {verdes} ({100.0*verdes/total:.1f}%)')
        print(f'  Nunca en verde:                     {rojos} ({100.0*rojos/total:.1f}%)')
        n_m, media_m, mediana_m, _ = resumen(mejores)
        print(f'  Mejor retorno alcanzado (media):    {fmt(media_m)}%   mediana {fmt(mediana_m)}%')
        print()
        print('  LECTURA: un porcentaje alto de "estuvieron en verde" con mejores retornos')
        print('  apreciables apunta a stop ajustado o a falta de gestion parcial, no a mala')
        print('  seleccion. Un porcentaje bajo apunta a que los setups eran malos de origen.')


def seccion_segmentos(checkpoints, setups_hist):
    print()
    print(SEP)
    print('4. ¿DISCRIMINA EL SISTEMA? — rendimiento por segmento (checkpoint de 20 sesiones)')
    print(SEP)
    grupo20 = [c for c in checkpoints if c.get('dias') == 20]
    if not grupo20:
        print('  Sin checkpoints de 20 sesiones todavia.')
        return

    print()
    print('  POR TIPO DE SETUP')
    por_tipo = defaultdict(list)
    for c in grupo20:
        por_tipo[(c.get('tipo') or 'sin tipo')].append(c.get('ret_pct'))
    for tipo in sorted(por_tipo, key=lambda t: -len(por_tipo[t])):
        print(linea_segmento(tipo[:26], por_tipo[tipo]))

    print()
    print('  POR GRUPO (solo los de 5 o mas observaciones)')
    por_grupo = defaultdict(list)
    for c in grupo20:
        por_grupo[(c.get('group') or 'sin grupo')].append(c.get('ret_pct'))
    mostrados = [g for g in por_grupo if len(por_grupo[g]) >= 5]
    for g in sorted(mostrados, key=lambda g: -statistics.mean(
            [v for v in por_grupo[g] if isinstance(v, (int, float))] or [0])):
        print(linea_segmento(g[:26], por_grupo[g]))
    if not mostrados:
        print('  Ningun grupo alcanza 5 observaciones todavia.')

    # Contexto al crear: solo disponible mientras el setup siga en la ventana de 30 dias.
    contexto = {}
    for dia in setups_hist:
        if not isinstance(dia, dict):
            continue
        for s in dia.get('setups', []):
            contexto[(dia.get('date'), s.get('ticker'))] = s
    emparejados = [c for c in grupo20 if (c.get('fecha_setup'), c.get('ticker')) in contexto]
    print()
    print(f'  CONTEXTO AL CREAR — cobertura: {len(emparejados)} de {len(grupo20)} checkpoints '
          f'({100.0*len(emparejados)/len(grupo20):.0f}%)')
    if not emparejados:
        print('  Los setups con checkpoint de 20 sesiones ya salieron de la ventana de 30 dias')
        print('  de setups_history: sin contexto al crear no se puede segmentar por riesgo/RSI.')
        return

    por_riesgo = defaultdict(list)
    por_rsi = defaultdict(list)
    por_st = defaultdict(list)
    for c in emparejados:
        s = contexto[(c.get('fecha_setup'), c.get('ticker'))]
        por_riesgo[s.get('riesgo') or 'sin dato'].append(c.get('ret_pct'))
        rsi = s.get('rsi_al_crear')
        if isinstance(rsi, (int, float)):
            banda = '<55' if rsi < 55 else '55-65' if rsi < 65 else '65-70' if rsi < 70 else '>=70'
            por_rsi[banda].append(c.get('ret_pct'))
        por_st[f"supertrend {s.get('supertrend_al_crear') or 'sin dato'}"].append(c.get('ret_pct'))

    print()
    print('  POR NIVEL DE RIESGO')
    for k in sorted(por_riesgo):
        print(linea_segmento(k, por_riesgo[k]))
    print()
    print('  POR BANDA DE RSI AL CREAR (decision pendiente del P18)')
    for k in ('<55', '55-65', '65-70', '>=70'):
        if k in por_rsi:
            print(linea_segmento(k, por_rsi[k]))
    print()
    print('  POR SUPERTREND AL CREAR (decision pendiente: ¿debe ser filtro de entrada?)')
    for k in sorted(por_st):
        print(linea_segmento(k, por_st[k]))


def seccion_resoluciones(resoluciones):
    print()
    print(SEP)
    print('5. RESOLUCIONES — la foto sesgada, para contrastar')
    print(SEP)
    print('  OJO: esta seccion tiene sesgo de seleccion. Los stops se resuelven rapido y')
    print('  los objetivos tardan, asi que las perdidas estan sobrerrepresentadas frente')
    print('  a la seccion 2. Se incluye para comparar, no para concluir.')
    print()
    if not resoluciones:
        print('  Sin resoluciones registradas.')
        return
    stops = [r.get('ret_pct') for r in resoluciones if r.get('resultado') == 'stop']
    objetivos = [r.get('ret_pct') for r in resoluciones if r.get('resultado') == 'objetivo']
    todos = [r.get('ret_pct') for r in resoluciones
             if isinstance(r.get('ret_pct'), (int, float))]
    print(linea_segmento('Stops', stops))
    print(linea_segmento('Objetivos', objetivos))
    print(linea_segmento('TODAS', todos))
    n, media, _, _ = resumen(todos)
    n_s, media_s, _, _ = resumen(stops)
    n_o, media_o, _, _ = resumen(objetivos)
    if n_s and n_o:
        print()
        print(f'  Ratio objetivos/stops: {n_o}/{n_s} = {n_o/n_s:.2f}')
        print(f'  Esperanza por operacion (media de todas): {fmt(media)}%')
        print('  Con esta proporcion, el sistema depende de que los objetivos sean MUCHO')
        print('  mayores que los stops en magnitud. Comprobado arriba: '
              f'{fmt(media_o)}% frente a {fmt(media_s)}%.')


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Analisis de rendimiento del scanner')
    ap.add_argument('--dir', default='.', help='directorio con los .json (por defecto: actual)')
    ap.add_argument('--sin-spy', action='store_true', help='omite la descarga del benchmark')
    args = ap.parse_args()

    print()
    print(SEP)
    print('ANALISIS DE RENDIMIENTO — Scanner de Temas Emergentes')
    print(f'Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(SEP)
    print()

    checkpoints = cargar(args.dir, 'checkpoints_history.json', [])
    resoluciones = cargar(args.dir, 'resoluciones_history.json', [])
    setups_hist = cargar(args.dir, 'setups_history.json', [])
    checkpoints = [c for c in checkpoints if isinstance(c, dict)]
    resoluciones = [r for r in resoluciones if isinstance(r, dict)]
    setups_hist = setups_hist if isinstance(setups_hist, list) else []

    if not checkpoints:
        print('\n  Sin checkpoints: no hay nada que analizar. Comprueba la ruta con --dir.')
        return

    fechas = seccion_cobertura(checkpoints, resoluciones, setups_hist)

    spy = {}
    if not args.sin_spy and fechas:
        print()
        print('  Descargando SPY para la comparacion...')
        spy = descargar_spy(fechas[0], fechas[-1])
        if spy:
            print(f'  SPY: {len(spy)} sesiones descargadas.')

    seccion_vs_spy(checkpoints, spy)
    seccion_stops(checkpoints, resoluciones)
    seccion_segmentos(checkpoints, setups_hist)
    seccion_resoluciones(resoluciones)

    print()
    print(SEP)
    print('ADVERTENCIA DE INTERPRETACION')
    print(SEP)
    print('  La muestra esta dominada por julio de 2026, un crash de momentum. Es el peor')
    print('  mes posible para juzgar un sistema de momentum, y ninguna cifra de aqui tiene')
    print('  significacion estadistica: son ordenes de magnitud, no conclusiones.')
    print('  Lo unico que se puede afirmar con solidez es lo que sea MUY grande y MUY')
    print('  consistente entre horizontes. Todo lo demas necesita mas meses.')
    print()


if __name__ == '__main__':
    main()
