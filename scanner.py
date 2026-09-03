"""
scanner.py — Script autonomo del Scanner de Temas Emergentes
Se ejecuta automaticamente via GitHub Actions cada noche de lunes a viernes.
Tambien puede ejecutarse manualmente: python scanner.py
"""

import math, warnings, json, base64, os, time
from datetime import datetime
from pathlib import Path
import numpy as np
import io
import pandas as pd
import requests
import yfinance as yf
import pytz
warnings.filterwarnings('ignore')

GITHUB_USER      = 'Carolo-III'
GITHUB_REPO      = 'scanner-temas'
GITHUB_TOKEN     = os.environ.get('SCANNER_TOKEN', '')
ANTHROPIC_KEY    = os.environ.get('ANTHROPIC_KEY', '')
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')  # FIX (11/07): sin default hardcodeado — el repo es publico; el chat ID vive SOLO en secrets
FMP_KEY          = os.environ.get('FMP_KEY', '')  # NUEVO (05/07) — Financial Modeling Prep, free tier 250 req/dia

PERSONAL_WATCHLIST = {
    'Semiconductores': ['ARM','AMD','MU','MTSI','POET','SMCI'],
    'Infraestructura AI': ['ANET','VRT','APLD','CORZ','IREN','CIFR','CRWV'],
    'Espacio y Defensa': ['RKLB','LUNR','ASTS','KTOS','BWXT'],
    'Cuantica': ['IONQ','RGTI','QBTS','QUBT','INFQ','ARQQ'],
    'Robotica AI': ['BBAI','TSLA'],
    'Minerales': ['MP','UAMY'],
    'Biotech': ['VKTX','IBRX'],
    'Memoria y Almacenamiento': ['SNDK','WDC','STX'],
    'Momentum': ['KOPN','ONDS','BE','LITE','GILT'],
}

SECTOR_LABELS = {
    'Technology':'Tecnologia','Health Care':'Salud','Financials':'Finanzas',
    'Consumer Discretionary':'Consumo Discrecional','Industrials':'Industriales',
    'Communication Services':'Comunicacion','Consumer Staples':'Consumo Basico',
    'Energy':'Energia','Utilities':'Utilities','Real Estate':'Real Estate','Materials':'Materiales',
}

madrid = pytz.timezone('Europe/Madrid')

def download_prices(tickers, period='1y'):
    """
    Descarga precios por lotes de 50 con reintentos.

    NOTA (fix 27/06, revertido): se probaron dos versiones de un "fix" para el resultado
    degenerado que se vio en calc_market_breadth (100% sobre MM50 y MM200 simultaneamente).
    Ambas intentaban arreglarlo aqui, a nivel de fila completa del DataFrame:
      v1: reintentar la descarga si la ultima fila venia mayoritariamente vacia (demasiado
          ruidoso: en fin de semana/festivo TODOS los lotes comparten el mismo hueco, reintentar
          no soluciona nada y genera avisos de alarma para una condicion normal).
      v2: descartar en silencio la ultima fila si venia vacia para mas del 50% del universo.
          Mas silencioso, pero con un fallo real: es una decision GLOBAL (toda la fila se
          descarta para TODOS los tickers a la vez). Si en una descarga concreta solo una parte
          del universo tuvo un fallo puntual (cualquier proveedor de datos puede fallar para un
          subconjunto de tickers sin motivo aparente), esta logica tiraba tambien el dato BUENO
          de los tickers que si lo tenian — caso real detectado: $TMO con cierre valido de
          $513.03 descartado y sustituido por el cierre del dia anterior, solo porque viajaba en
          la misma fila que datos vacios de otros tickers.

    Ambas versiones quedan revertidas. El criterio correcto, que el resto del sistema ya
    aplicaba desde el principio, es que CADA ticker caiga a su PROPIO ultimo valor valido
    (via dropna() individual en price_stats/ma_health/etc.), nunca una decision de fila
    completa que dependa de como les fue a los demas tickers en la misma descarga. La
    proteccion para calc_market_breadth especificamente ya esta resuelta de la forma correcta
    (per-columna, no por fila completa) en calc_market_breadth() — ver su propia nota de fix.
    """
    if not tickers:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    LOTE = 50
    all_c, all_v, all_h, all_l = [], [], [], []
    lotes = [tickers[i:i+LOTE] for i in range(0, len(tickers), LOTE)]
    for i, lote in enumerate(lotes):
        if len(lotes) > 1:
            print(f'    Lote {i+1}/{len(lotes)}...', end=' ')
        for intento in range(3):
            try:
                raw = yf.download(lote, period=period, interval='1d',
                                  auto_adjust=True, progress=False, threads=True)
                if raw.empty:
                    if intento < 2:
                        import time; time.sleep(5); continue
                    if len(lotes) > 1: print('vacio')
                    break
                if isinstance(raw.columns, pd.MultiIndex):
                    c, v, h, l = raw['Close'], raw['Volume'], raw['High'], raw['Low']
                else:
                    c = raw[['Close']]; c.columns = lote[:1]
                    v = raw[['Volume']]; v.columns = lote[:1]
                    h = raw[['High']];  h.columns = lote[:1]
                    l = raw[['Low']];   l.columns = lote[:1]
                all_c.append(c); all_v.append(v)
                all_h.append(h); all_l.append(l)
                if len(lotes) > 1: print('OK')
                break
            except Exception as e:
                if intento < 2:
                    import time; time.sleep(5)
                else:
                    if len(lotes) > 1: print(f'error tras 3 intentos: {e}')
    if not all_c:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return (pd.concat(all_c, axis=1), pd.concat(all_v, axis=1),
            pd.concat(all_h, axis=1), pd.concat(all_l, axis=1))

def spy_health(bench_close, confirm_days=3):
    """
    NUEVO (05/07) — Anti-whiplash: filtro de confirmacion ASIMETRICO.

    Antes: healthy = ultimo cierre > MM200 (un solo punto de datos, binario). Si el SPY
    oscila alrededor de su MM200 varios dias (transiciones, lateral volatil), el sistema
    penalizaba el score un dia y no el siguiente, generando inconsistencia en los setups
    sin que hubiera cambiado nada fundamental en el mercado.

    Ahora (maquina de estados reconstruida sobre el propio historico descargado, sin
    persistencia externa — deterministica: mismos datos, mismo estado):
      - Entrar en BAJISTA (activar penalizacion): INMEDIATO. Un solo cierre bajo MM200
        basta — el coste de una falsa alarma (perder un setup) es menor que el de
        ignorar un deterioro real.
      - Volver a ALCISTA (desactivar penalizacion): requiere confirm_days cierres
        CONSECUTIVOS por encima de la MM200. Un rebote de un solo dia no reactiva el
        sistema (podria ser un pullback dentro de una tendencia bajista).

    P57 (09/08/2026) — Score continuo 0.0-1.0:
      Ademas del booleano (compatible con todo el codigo existente), la funcion devuelve
      un float spy_score = clip(0.5 + distancia * 10, 0.0, 1.0), donde distancia es
      (spy_close - mm200) / mm200. Con k=10:
        - +5% sobre MM200 -> score 1.0 (techo)
        - Exactamente en MM200 -> score 0.5
        - -5% bajo MM200 -> score 0.0 (suelo)
        - ±1% -> score 0.60 / 0.40 (zona de transicion suave)
      El score se persiste en spy_health_history.json (fichero separado, opcion A) para
      no alterar la comparabilidad del historico de scores ya acumulado.

    Nota: independiente del Punto 16 (acoplamiento regimen macro -> composite_score,
    bloqueado hasta finales de septiembre 2026). El position sizing segun regimen es
    decision de Carlos caso a caso, no del codigo.

    Retorna: tupla (healthy: bool, spy_score: float)
    """
    try:
        p = bench_close.dropna()
        if len(p) < 200: return True, 1.0
        ma200_series = p.rolling(200).mean()
        current = float(p.iloc[-1]); ma200 = float(ma200_series.iloc[-1])
        # Serie booleana de cierres sobre MM200 (solo donde la MM200 ya es valida)
        validos = ~ma200_series.isna()
        p_v = p[validos].values; m_v = ma200_series[validos].values
        above = p_v > m_v
        # Maquina de estados asimetrica, reconstruida desde el primer punto valido
        healthy = bool(above[0])
        racha = 0  # cierres consecutivos sobre MM200 (solo relevante en estado bajista)
        for a in above[1:]:
            if a:
                racha += 1
                if not healthy and racha >= confirm_days:
                    healthy = True
            else:
                racha = 0
                healthy = False
        raw = current > ma200  # estado bruto de un solo cierre (informativo)
        if healthy:
            estado = 'ALCISTA'
        elif raw:
            estado = f'BAJISTA (pendiente de confirmacion: {min(racha, confirm_days)}/{confirm_days} cierres sobre MM200)'
        else:
            estado = 'BAJISTA'
        # Score continuo 0.0-1.0 (P57)
        distancia = (current - ma200) / ma200
        spy_score = round(min(1.0, max(0.0, 0.5 + distancia * 10)), 4)
        print(f'  SPY: ${round(current,2)} | MM200: ${round(ma200,2)} | {estado} | spy_score: {spy_score}')
        return healthy, spy_score
    except Exception as _e:
        _traza('spy_health', _e)
        return True, 1.0

def calc_market_breadth(universe_close, bench_close):
    """
    PUNTO 8 — Amplitud de mercado como variable adicional del regimen (informativa,
    NO modula composite_score por ahora, solo se incluye en el informe).

    universe_close: DataFrame de precios de cierre del universo amplio (S&P500+Nasdaq100+SOX,
                     variable 'cs' en main()), columnas = tickers, filas = fechas.
    bench_close: serie de precios de cierre del SPY (variable 'bs' en main()).

    Devuelve dict con:
      pct_sobre_mm50, pct_sobre_mm200   -> % de valores del universo por encima de su propia MM50/MM200
      nuevos_max_52s, nuevos_min_52s    -> cuantos valores tocan maximo/minimo de las ultimas 252 sesiones
      avance, descenso                  -> cuantos valores subieron/bajaron en la ultima sesion
      pendiente_mm200_spy               -> 'ascendente'/'descendente'/'plana', MM200 hoy vs hace 20 sesiones
      n_valores                         -> tamaño del universo evaluado (para contexto)
    Si no hay datos suficientes, los campos faltantes devuelven None sin interrumpir el resto.

    NOTA (fix 27/06): version anterior usaba precios.iloc[-1] (la ultima FILA del DataFrame
    completo) como "valor actual" de todos los tickers a la vez. Si esa fila viene incompleta
    para casi todo el universo (por ejemplo, ejecucion muy temprana en la sesion antes de que
    yfinance tenga vela diaria fiable para la mayoria, o un desajuste de fechas al concatenar
    los lotes de download_prices), el resultado se calcula sobre un puñado de tickers en vez del
    universo real — sintoma detectado en produccion: avance+descenso=1, 100% sobre MM50 Y MM200
    simultaneamente (caso degenerado con casi todos los datos ausentes). El resto del sistema
    (analyze_universe) no sufre esto porque cada ticker cae a su propio ultimo valor valido via
    dropna() individual; aqui se aplica el mismo criterio por columna en vez de una fila fija
    sincronizada para todo el universo.
    """
    result = {'pct_sobre_mm20': None, 'pct_sobre_mm50': None, 'pct_sobre_mm200': None,
              'nuevos_max_52s': None, 'nuevos_min_52s': None, 'avance': None, 'descenso': None,
              'mcclellan': None, 'pendiente_mm200_spy': None, 'n_valores': None}
    try:
        precios = universe_close.dropna(axis=1, how='all')
        if precios.empty:
            return result
        result['n_valores'] = precios.shape[1]

        # Ultimo valor VALIDO de cada ticker (no la ultima fila sincronizada del DataFrame
        # completo) — si la fila mas reciente esta incompleta para algunos tickers, cada uno
        # cae a su propio ultimo dato real disponible, igual que el resto del sistema.
        ultimo = precios.ffill().iloc[-1]

        # PUNTO 20 — % sobre MM20: amplitud tactica de corto plazo (mas rapida y ruidosa
        # que MM50/MM200; util para detectar giros de participacion antes que las lentas)
        if len(precios) >= 20:
            mm20 = precios.rolling(20).mean().ffill()
            ultimo_mm20 = mm20.iloc[-1]
            valid20 = ultimo.notna() & ultimo_mm20.notna()
            if valid20.sum() > 0:
                result['pct_sobre_mm20'] = round((ultimo[valid20] > ultimo_mm20[valid20]).mean() * 100, 1)

        # % sobre MM50 / MM200 (requiere al menos 50/200 sesiones de historico)
        if len(precios) >= 50:
            mm50 = precios.rolling(50).mean().ffill()
            ultimo_mm50 = mm50.iloc[-1]
            valid50 = ultimo.notna() & ultimo_mm50.notna()
            if valid50.sum() > 0:
                result['pct_sobre_mm50'] = round((ultimo[valid50] > ultimo_mm50[valid50]).mean() * 100, 1)
        if len(precios) >= 200:
            mm200 = precios.rolling(200).mean().ffill()
            ultimo_mm200 = mm200.iloc[-1]
            valid200 = ultimo.notna() & ultimo_mm200.notna()
            if valid200.sum() > 0:
                result['pct_sobre_mm200'] = round((ultimo[valid200] > ultimo_mm200[valid200]).mean() * 100, 1)

        # Nuevos maximos / minimos de 52 semanas (ventana de hasta 252 sesiones disponibles)
        ventana = min(252, len(precios))
        if ventana >= 20:  # umbral minimo para que el dato tenga algun sentido
            max_52s = precios.iloc[-ventana:].max()
            min_52s = precios.iloc[-ventana:].min()
            validez = ultimo.notna() & max_52s.notna() & min_52s.notna()
            result['nuevos_max_52s'] = int((ultimo[validez] >= max_52s[validez]).sum())
            result['nuevos_min_52s'] = int((ultimo[validez] <= min_52s[validez]).sum())

        # Avance / descenso: variacion del ultimo valor valido de cada ticker vs su valor
        # valido anterior (ffill antes de pct_change evita que una fila reciente incompleta
        # se cuente como caida/ausencia para tickers que en realidad no se movieron)
        precios_ff = precios.ffill()
        if len(precios_ff) >= 2:
            ret_diario = precios_ff.pct_change().iloc[-1]
            ret_diario = ret_diario.dropna()
            result['avance'] = int((ret_diario > 0).sum())
            result['descenso'] = int((ret_diario < 0).sum())

        # PUNTO 20 — McClellan Oscillator (version ratio-adjusted, robusta a cambios del
        # tamaño del universo): RANA = 1000*(avances-descensos)/(avances+descensos) por
        # sesion; oscilador = EMA19(RANA) - EMA39(RANA). Positivo = el momento de la
        # amplitud es comprador; negativo = vendedor; el cruce de 0 marca el giro.
        # Se calcula sobre las ultimas ~90 sesiones del universo (suficiente para que las
        # EMAs converjan). ffill previo: un ticker sin dato nuevo cuenta como sin cambio
        # (ni avance ni descenso), igual que en el bloque avance/descenso.
        if len(precios_ff) >= 60:
            rets_hist = precios_ff.pct_change().iloc[-90:]
            adv_d = (rets_hist > 0).sum(axis=1)
            dec_d = (rets_hist < 0).sum(axis=1)
            tot_d = (adv_d + dec_d).replace(0, np.nan)
            rana = (1000.0 * (adv_d - dec_d) / tot_d).dropna()
            if len(rana) >= 45:
                ema19 = rana.ewm(span=19, adjust=False).mean()
                ema39 = rana.ewm(span=39, adjust=False).mean()
                result['mcclellan'] = round(float(ema19.iloc[-1] - ema39.iloc[-1]), 1)

        _BREADTH_CACHE['ultimo'] = result  # PUNTO 25 — breadth del dia para etiquetar setups nuevos

        # Pendiente de MM200 del SPY (hoy vs hace 20 sesiones), NO del universo
        bp = bench_close.dropna()
        if len(bp) >= 220:
            mm200_spy = bp.rolling(200).mean()
            hoy = float(mm200_spy.iloc[-1])
            hace_20 = float(mm200_spy.iloc[-21])
            if hoy > hace_20 * 1.001:
                result['pendiente_mm200_spy'] = 'ascendente'
            elif hoy < hace_20 * 0.999:
                result['pendiente_mm200_spy'] = 'descendente'
            else:
                result['pendiente_mm200_spy'] = 'plana'

        # NUEVO (10/07) — DISPERSION INDICE/VALOR, FASE 1: INSTRUMENTACION SIN ALERTA.
        # Ratio = vol realizada 20s del SPY / media de vol realizada 20s de los valores.
        # Ratio BAJO = los valores se mueven mucho pero DESCORRELACIONADOS y el indice apenas
        # se mueve -> el VIX cuenta media verdad; si las correlaciones suben de golpe, el VIX
        # salta sin aviso. Dos analistas independientes señalaron brechas record indice/valor
        # (07 y 09/07/2026). NOTA de honestidad: esto es volatilidad REALIZADA (va con retraso);
        # los analistas hablan de IMPLICITA (opciones, no disponible gratis) — mismo fenomeno
        # estructural, distinta metrica: no confundirlas al interpretar.
        # SIN UMBRAL y SIN mencion en el informe hasta tener 2-3 meses de historial propio
        # (calibracion en la revision de septiembre, junto al Punto 16). Solo se persiste
        # en data.json (via breadth) y se imprime una linea de seguimiento en consola.
        try:
            rets_20 = precios.pct_change().iloc[-20:]
            suficientes = rets_20.count() >= 15  # exigir >=15 retornos validos por ticker
            vols = (rets_20.loc[:, suficientes].std() * (252 ** 0.5)).dropna()
            bp_d = bench_close.dropna()
            rets_spy = bp_d.pct_change().iloc[-20:].dropna()
            if len(vols) >= 50 and len(rets_spy) >= 15:
                vol_spy = float(rets_spy.std() * (252 ** 0.5))
                vol_media = float(vols.mean())
                if vol_media > 0:
                    result['dispersion'] = {
                        'ratio': round(vol_spy / vol_media, 3),
                        'vol_spy_20s_pct': round(vol_spy * 100, 1),
                        'vol_media_valores_20s_pct': round(vol_media * 100, 1),
                        'n_valores_vol': int(len(vols)),
                    }
                    d = result['dispersion']
                    print(f"  Dispersion indice/valor: ratio {d['ratio']} "
                          f"(vol SPY {d['vol_spy_20s_pct']}% vs media valores {d['vol_media_valores_20s_pct']}%) "
                          f"— instrumentacion, sin umbral hasta septiembre")
        except Exception as _e:
            _traza('amplitud/dispersion', _e)
    except Exception as e:
        print(f'  Aviso: calc_market_breadth fallo parcialmente: {e}')
    return result

# NUEVO (07/07) — VECTOR MACRO DE REGIMEN: umbrales de velocidad que definen cuando un
# movimiento macro es informacion de regimen y no ruido diario. Solo si un umbral dispara,
# la serie llega al prompt del informe (silencio impuesto POR CODIGO, no confiado al modelo:
# sin disparo, las lineas ALERTA DE REGIMEN ni siquiera existen en los datos que ve Claude).
# Umbrales relativos (velocidad), no niveles psicologicos, para que envejezcan bien.
UMBRAL_30Y_PB      = 25    # |cambio en 20 sesiones| del bono a 30 años, en puntos basicos
UMBRAL_WTI_PCT     = 10    # |cambio en 20 sesiones| del crudo WTI, en %
UMBRAL_CREDITO_PCT = -2.0  # caida del ratio HYG/IEF en 20 sesiones (credito tensionandose)

def calc_pendientes_curva(niveles):
    """Pendientes de la curva en puntos basicos a partir de {clave: yield en %}.

    Omite la pendiente cuyo tramo falte en vez de inventarlo. p10y_3m es la
    pendiente de referencia (inversion = recesion descontada); p30y_10y separa
    prima por plazo del tramo ultralargo; p10y_5y aisla el vientre de la curva.
    """
    pendientes = {}
    for nombre, largo, corto in (('p10y_3m', 'us10y', 'us3m'),
                                 ('p10y_5y', 'us10y', 'us5y'),
                                 ('p30y_10y', 'us30y', 'us10y')):
        a, b = niveles.get(largo), niveles.get(corto)
        if a is None or b is None:
            continue
        pendientes[nombre] = round((a - b) * 100.0, 1)
    return pendientes

UMBRAL_BTC_FINDE_PCT = 5.0   # P34: caida de fin de semana que dispara aviso de riesgo-off

# P77 (01/09/2026) — DISPARADOR DE NIVEL, ademas del de velocidad.
#
# Las alertas de regimen (P7) miden VELOCIDAD: |cambio| a 20 sesiones contra un umbral. Eso
# detecta movimientos bruscos pero es CIEGO a los niveles extremos alcanzados despacio, y el
# efecto es perverso: cuanto mas tiempo lleva una serie arriba, MENOS probable es la alerta,
# porque la referencia de hace 20 sesiones tambien esta alta.
#
# Casos medidos: el 18/08 el US30Y marco maximo de 19 anos y el bloque decia "sin alertas de
# regimen" (la variacion era +15,5 pb con umbral de 25, y habia BAJADO desde +20,1 mientras
# el nivel subia). El unico disparador de nivel que existia, cruce_5pct, solo actua el dia
# exacto del cruce: una vez por encima del 5% no vuelve a saltar jamas. El 01/09, con el
# 30 anos al 5,21% y el gasto en intereses en record historico, el informe seguia sin
# mencionarlo.
#
# Criterio: el nivel esta en el percentil superior de su propia historia reciente. Se usa
# percentil y no umbral absoluto porque un numero fijo (5%, 90$) envejece mal y obliga a
# recalibrar; el percentil se adapta solo y significa lo mismo en cualquier regimen.
#
# El credito es ASIMETRICO por diseno (igual que su alerta de velocidad): un ratio HYG/IEF
# en minimos es tension crediticia, que es lo que se quiere ver; en maximos es apetito de
# riesgo y no se avisa.

UMBRAL_NIVEL_PCT   = 95    # percentil a partir del cual el nivel se considera extremo
MIN_SESIONES_NIVEL = 60    # por debajo, un percentil no significa nada y no se evalua

def _nivel_extremo(valores, pct=UMBRAL_NIVEL_PCT, alto=True):
    """¿Esta el ULTIMO valor en el extremo de su propia historia? Devuelve dict o None.

    `alto=True` mira el extremo superior (tipos, crudo); `alto=False` el inferior, para el
    credito, donde la tension es el ratio cayendo.

    Devuelve None —y no False— cuando no hay muestra suficiente: "no se puede afirmar" y
    "no es extremo" son cosas distintas, y confundirlas es lo que hizo que el P69 fuera
    necesario.
    """
    try:
        vals = [float(v) for v in valores if v is not None and float(v) == float(v)]
    except (TypeError, ValueError) as _e:
        _traza('regimen/nivel-extremo', _e); return None
    if len(vals) < MIN_SESIONES_NIVEL:
        return None
    actual = vals[-1]
    if alto:
        peores = sum(1 for v in vals if v <= actual)
        extremo, ref = peores / len(vals) * 100.0, max(vals)
    else:
        peores = sum(1 for v in vals if v >= actual)
        extremo, ref = peores / len(vals) * 100.0, min(vals)
    return {'percentil': round(extremo, 1), 'n': len(vals),
            'referencia': round(ref, 4), 'alerta_nivel': bool(extremo >= pct)}

def _serie_tramo_historico(clave):
    """Serie completa de niveles de un tramo de curva desde rates_history.json.

    Reutiliza el cache que ya carga _nivel_hace_n_sesiones (P73), asi que no anade ninguna
    lectura extra. Necesario porque Yahoo lleva desde el 12/08 sirviendo ~16 sesiones de los
    simbolos de deuda: sin el historico propio no habria muestra para calcular un percentil.
    """
    _nivel_hace_n_sesiones(clave, 1)          # fuerza la carga del cache si aun no esta
    entradas = _RATES_HIST_CACHE.get('entradas') or []
    salida = []
    for e in entradas:
        v = ((e or {}).get('niveles') or {}).get(clave)
        if isinstance(v, (int, float)):
            salida.append(float(v))
    return salida

def _calc_macro_regimen():
    """
    NUEVO (07/07) — series de regimen: US30Y (^TYX), WTI (CL=F) y credito high-yield via
    ratio HYG/IEF (bonos high-yield contra Tesoro 7-10a, con auto_adjust para que los
    dividendos mensuales de HYG no distorsionen el ratio). Para cada serie: nivel, cambio
    de 1 dia y cambio de 20 SESIONES — la velocidad define el regimen, el dia a dia es ruido.
    El credito es el sensor mas valioso del bloque: los diferenciales tensionandose suelen
    anticipar el estres en renta variable antes que el VIX o la amplitud.
    TODO se persiste siempre en data.json (historial para la futura calibracion del
    acoplamiento macro->score, que NO se toca hasta tener evidencia — misma disciplina que
    el Punto 16; probablemente mas alla de septiembre: los regimenes de tipos cambian en
    años, no en trimestres). Al informe solo llegan las series cuya alerta dispara.
    """
    tickers = ['^TYX', '^TNX', '^FVX', '^IRX', 'CL=F', 'HYG', 'IEF', 'BTC-USD']
    raw = yf.download(tickers, period='3mo', interval='1d',
                      auto_adjust=True, progress=False, threads=True)
    if raw.empty: return {}
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    # P69 (14/08/2026) — se registra QUE series faltan o llegan cortas. Antes _serie
    # devolvia una Serie vacia y cada bloque hacia `if len(s) >= 21: ...` sin else: la
    # serie desaparecia del informe Y de data.json como si nunca se hubiera pedido. Del
    # 12 al 14/08 el bloque macro perdio primero la curva de tipos y luego tambien el
    # US30Y, en dos escalones, sin una sola linea de log — con ellos se fue la
    # instrumentacion de curva sembrada con 120 sesiones el 01/08. El P41/P42 acabo con
    # los fallos mudos por excepcion capturada; este es el otro tipo de silencio, el del
    # dato que simplemente no viene, y no lo cubria ninguna traza.
    _faltan, _cortas = [], []
    def _serie(tk, minimo=21):
        if tk not in close.columns:
            _faltan.append(tk)
            return pd.Series(dtype=float)
        s = close[tk].dropna()
        if len(s) < minimo:
            _cortas.append(f'{tk}({len(s)})')
        return s
    reg = {}
    # 1) US30Y — bono a 30 años
    # P73 — bastan DOS sesiones: el nivel y la variacion diaria solo necesitan eso, y la
    # variacion a 20 sesiones se toma del historico propio cuando Yahoo no da para tanto.
    s30 = _serie('^TYX', minimo=2)
    if len(s30) >= 2:
        vals = s30.values.astype(float)
        if vals[-1] > 20: vals = vals / 10  # Yahoo reporta ^TYX en decimas (50.2 = 5.02%)
        nivel = round(float(vals[-1]), 2)
        chg_1d_pb  = round(float(vals[-1] - vals[-2]) * 100, 1)
        if len(vals) >= 21:
            chg_20s_pb = round(float(vals[-1] - vals[-21]) * 100, 1)
        else:
            chg_20s_pb = _variacion_20s_pb('us30y', nivel)   # P73: serie propia
        cruce_5 = bool((vals[-2] < 5.0 <= vals[-1]) or (vals[-2] >= 5.0 > vals[-1]))
        # P77 — el percentil se calcula sobre el historico PROPIO: la serie descargada
        # trae ~16 sesiones y no da para nada. Si tampoco hay historico, se degrada a la
        # serie descargada y, si esa no llega, _nivel_extremo devuelve None y no se afirma.
        _niv = _nivel_extremo(_serie_tramo_historico('us30y')) or _nivel_extremo(list(vals))
        reg['us30y'] = {'nivel': nivel, 'chg_1d_pb': chg_1d_pb, 'chg_20s_pb': chg_20s_pb,
                        'cruce_5pct': cruce_5, 'nivel_extremo': _niv,
                        'alerta': bool((chg_20s_pb is not None and abs(chg_20s_pb) >= UMBRAL_30Y_PB)
                                       or cruce_5
                                       or (_niv or {}).get('alerta_nivel'))}
    # 2) WTI — crudo
    swti = _serie('CL=F')
    if len(swti) >= 21:
        nivel = round(float(swti.iloc[-1]), 2)
        chg_1d  = round(float(swti.iloc[-1] / swti.iloc[-2] - 1) * 100, 1)
        chg_20s = round(float(swti.iloc[-1] / swti.iloc[-21] - 1) * 100, 1)
        _niv = _nivel_extremo(list(swti.values.astype(float)))   # P77
        reg['wti'] = {'nivel': nivel, 'chg_1d_pct': chg_1d, 'chg_20s_pct': chg_20s,
                      'nivel_extremo': _niv,
                      'alerta': bool(abs(chg_20s) >= UMBRAL_WTI_PCT
                                     or (_niv or {}).get('alerta_nivel'))}
    # 3) Credito — ratio HYG/IEF (asimetrico: solo alerta el tensionamiento, no la mejora)
    shyg, sief = _serie('HYG'), _serie('IEF')
    if len(shyg) >= 21 and len(sief) >= 21:
        ratio = (shyg / sief).dropna()
        if len(ratio) >= 21:
            r = ratio.values.astype(float)
            chg_1d  = round(float(r[-1] / r[-2] - 1) * 100, 2)
            chg_20s = round(float(r[-1] / r[-21] - 1) * 100, 2)
            # P77 — asimetrico como su alerta de velocidad: el extremo que importa es el
            # ratio en MINIMOS (tension crediticia), no en maximos.
            _niv = _nivel_extremo(list(r), alto=False)
            reg['credito_hyg_ief'] = {'ratio': round(float(r[-1]), 4),
                                      'chg_1d_pct': chg_1d, 'chg_20s_pct': chg_20s,
                                      'nivel_extremo': _niv,
                                      'alerta': bool(chg_20s <= UMBRAL_CREDITO_PCT
                                                     or (_niv or {}).get('alerta_nivel'))}
    # 3b) BITCOIN — apetito de riesgo (PUNTO 34, 02/08/2026)
    # Valor diferencial: BTC cotiza 24/7, incluida la sesion del sabado y del domingo que
    # NINGUNA otra serie del scanner cubre. La ejecucion de la madrugada del lunes puede por
    # tanto llevar informacion que el panel de acciones aun no tiene. Por eso la variacion de
    # fin de semana se mide contra el CIERRE DEL VIERNES, no contra las ultimas 24h.
    # NO entra en el ranking ni en el score: es demasiado ruidoso como para ponderar setups
    # (misma disciplina que la curva de tipos). Solo describe el regimen y, si cae con fuerza
    # en fin de semana, avisa de posible riesgo-off en la apertura del lunes.
    sbtc = _serie('BTC-USD')
    if len(sbtc) >= 8:
        nivel = round(float(sbtc.iloc[-1]), 0)
        chg_1d = round(float(sbtc.iloc[-1] / sbtc.iloc[-2] - 1) * 100, 1)
        chg_7d = round(float(sbtc.iloc[-1] / sbtc.iloc[-8] - 1) * 100, 1)
        btc = {'nivel': nivel, 'chg_1d_pct': chg_1d, 'chg_7d_pct': chg_7d, 'alerta': False}
        # Variacion desde el ultimo cierre de VIERNES disponible (dayofweek==4). Solo tiene
        # sentido si el ultimo dato es de sabado, domingo o lunes: el resto de la semana
        # este calculo no aporta nada y se omite en vez de emitir un numero engañoso.
        try:
            idx = sbtc.index
            if idx[-1].dayofweek in (5, 6, 0):
                viernes = [i for i, f in enumerate(idx) if f.dayofweek == 4]
                if viernes and viernes[-1] < len(sbtc) - 1:
                    cierre_v = float(sbtc.iloc[viernes[-1]])
                    chg_finde = round(float(sbtc.iloc[-1] / cierre_v - 1) * 100, 1)
                    btc['chg_finde_pct'] = chg_finde
                    btc['alerta'] = bool(chg_finde <= -UMBRAL_BTC_FINDE_PCT)
        except Exception as _ebtc:
            print(f'  AVISO P34: no se pudo calcular la variacion de fin de semana de BTC: {_ebtc}')
        reg['btc'] = btc
    # 4) CURVA DE TIPOS (01/08/2026) — tramos 3M/5Y/10Y/30Y con nivel, variacion 20s y
    # pendientes. Sale de la MISMA descarga que ya se hacia para ^TYX: sin peticiones
    # extra y heredando la normalizacion de decimas de Yahoo (^TNX/^TYX/^FVX/^IRX
    # llegan a veces como 47.45 en vez de 4.745). Es INSTRUMENTACION: no genera alerta
    # ni toca el score. La serie se persiste en rates_history.json para que el P16
    # pueda calibrarse algun dia con evidencia propia, no con intuicion.
    # P70 (14/08/2026) — el NIVEL y la VARIACION tienen requisitos distintos y ya no
    # comparten umbral. Antes el bucle exigia 21 sesiones para todo y descartaba el tramo
    # entero, pero un nivel solo necesita el ULTIMO dato: las 21 sesiones hacen falta para
    # la variacion a 20 sesiones, no para el nivel. El 14/08 ^TNX (bono 10 años) llego con
    # 15 sesiones y se descarto completo, y como las DOS pendientes (10y-3m y 30y-10y) se
    # apoyan en us10y, la curva entera desaparecio del bloque macro teniendo el dato de hoy
    # disponible. Ahora el nivel entra con una sola sesion y la variacion se omite si no
    # hay historico suficiente: se degrada el indicador que falta, no el tramo completo.
    curva_niveles, curva_var = {}, {}
    _sin_variacion, _var_historico = [], []
    for clave, tk in (('us3m', '^IRX'), ('us5y', '^FVX'),
                      ('us10y', '^TNX'), ('us30y', '^TYX')):
        serie = _serie(tk, minimo=1)
        if len(serie) < 1:
            continue
        vals = serie.values.astype(float)
        if vals[-1] > 20: vals = vals / 10  # misma guarda de decimas que us30y
        curva_niveles[clave] = round(float(vals[-1]), 3)
        if len(vals) >= 21:
            curva_var[clave] = round(float(vals[-1] - vals[-21]) * 100, 1)
        else:
            # P73 — la serie descargada no da para 20 sesiones: se usa la propia.
            _desde_hist = _variacion_20s_pb(clave, curva_niveles[clave])
            if _desde_hist is not None:
                curva_var[clave] = _desde_hist
                _var_historico.append(clave)
            else:
                _sin_variacion.append(f'{clave}({len(vals)})')
    if curva_niveles:
        _pend = calc_pendientes_curva(curva_niveles)
        reg['curva'] = {'niveles': curva_niveles, 'variacion_20s_pb': curva_var,
                        'pendientes_pb': _pend}
        # P69 — caso sutil: hay niveles (y la entrada SI se persiste en rates_history)
        # pero faltan tramos para calcular pendientes, asi que la linea de curva
        # desaparece del log sin que se pierda la serie. Es la degradacion parcial que
        # nadie detecta: el fichero sigue en el commit y el ojo no echa en falta nada.
        if not _pend:
            print(f'  🔴 AVISO macro: curva SIN pendientes calculables — solo llegaron los '
                  f'tramos {", ".join(sorted(curva_niveles))}. La serie se persiste igual, '
                  'pero la curva no aparecera en el bloque macro.')
        # P70 — el nivel esta pero la variacion a 20 sesiones no: la pendiente SI se
        # calcula (solo usa niveles), lo que se pierde es la lectura de velocidad.
        if _var_historico:
            print('  Curva: variacion a 20 sesiones tomada de rates_history.json (la serie '
                  'descargada venia corta) en: ' + ', '.join(sorted(_var_historico)))
        if _sin_variacion:
            print('  AVISO macro: tramos de curva sin variacion a 20 sesiones (ni descargada '
                  'ni en el historico propio): ' + ', '.join(sorted(_sin_variacion)))
    # P69 — el aviso sale SIEMPRE que falte algo, aunque el resto del bloque este bien:
    # una degradacion parcial es justo la que pasa desapercibida.
    if _faltan or _cortas:
        _det = []
        if _faltan:
            _det.append('sin datos: ' + ', '.join(sorted(set(_faltan))))
        if _cortas:
            _det.append('serie corta (<21 sesiones): ' + ', '.join(sorted(set(_cortas))))
        print('  🔴 AVISO macro: no se pudieron calcular todas las series de regimen — '
              + ' | '.join(_det) + '. Los indicadores afectados NO aparecen en el informe '
              'ni se persisten; si se repite varios dias, revisar los simbolos en Yahoo.')
        _traza('macro/series-ausentes', RuntimeError('; '.join(_det)))
    if reg:
        alertas = [k for k, v in reg.items() if v.get('alerta')]
        partes = []
        if 'us30y' in reg:
            # ARREGLO (16/08/2026): el P73 permitio que chg_20s_pb sea None (nivel disponible
            # pero sin 20 sesiones ni descargadas ni en el historico) y este formato lo
            # reventaba con TypeError, tumbando el bloque macro ENTERO — el 16/08 desaparecio
            # el regimen del log y rates_history se quedo sin subir. Un dato que falta debe
            # degradar su propia linea, nunca llevarse el bloque por delante.
            _v30 = reg['us30y'].get('chg_20s_pb')
            _t30 = f'{_v30:+} pb/20s' if isinstance(_v30, (int, float)) else 'sin variacion 20s'
            partes.append(f"US30Y {reg['us30y']['nivel']}% ({_t30})")
        if 'wti' in reg: partes.append(f"WTI ${reg['wti']['nivel']} ({reg['wti']['chg_20s_pct']:+}%/20s)")
        if 'credito_hyg_ief' in reg: partes.append(f"HYG/IEF {reg['credito_hyg_ief']['chg_20s_pct']:+}%/20s")
        if 'btc' in reg:
            _b = reg['btc']
            _txt = f"BTC ${_b['nivel']:,.0f} ({_b['chg_7d_pct']:+}%/7d"
            if 'chg_finde_pct' in _b: _txt += f", finde {_b['chg_finde_pct']:+}%"
            partes.append(_txt + ')')
        if 'curva' in reg and reg['curva'].get('pendientes_pb'):
            _p = reg['curva']['pendientes_pb']
            partes.append('curva 10y-3m ' + (f"{_p['p10y_3m']:+.0f}pb" if 'p10y_3m' in _p else 'n/d') +
                          (f" | 30y-10y {_p['p30y_10y']:+.0f}pb" if 'p30y_10y' in _p else ''))
        # P77 — la alerta dice AHORA por que salta: velocidad, nivel o ambas. Sin esto,
        # "ALERTAS: us30y" no distingue un movimiento brusco de un maximo historico.
        _detalle = []
        for k in alertas:
            _n = (reg[k] or {}).get('nivel_extremo') or {}
            _detalle.append(f"{k} (nivel en percentil {_n['percentil']} de {_n['n']} sesiones)"
                            if _n.get('alerta_nivel') else k)
        print('  Regimen macro: ' + ' | '.join(partes) +
              (' | ⚠️ ALERTAS: ' + ', '.join(_detalle) if alertas else ' | sin alertas de regimen'))
    return reg

def _calc_estructura_indices():
    """
    NUEVO (11/07) — ESTRUCTURA DE INDICES EN LA MM50: pulso TACTICO intermedio entre la
    MM200 del SPY (regimen, spy_health) y las MM50 de los valores individuales (setups).
    Origen: analisis externo del 08/07 ("el S&P reboto en su MM50; el Nasdaq sigue por
    debajo de la suya") — ese dia los cinco setups eran pullbacks a MM50 individuales
    mientras el propio SPX hacia pullback a la suya, y el informe no podia nombrarlo.

    Para SPY y QQQ: precio vs MM50, distancia en %, estado (sobre/bajo) y sesiones en el
    estado actual (mismo patron de conteo que dias_en_direccion del Supertrend).
    DISPARADORES (solo entonces llega al prompt; silencio por codigo el resto de dias):
      - CRUCE: el estado ha cambiado en las ultimas 2 sesiones (sesiones_en_estado <= 2).
      - TEST:  el cierre esta a menos de UMBRAL_TEST_MM50 (±1%) de la MM50.
    Se persiste SIEMPRE en data.json (macro['indices_mm50']) aunque no dispare.
    Es contexto tactico, NO regimen: el regimen sigue siendo la MM200.
    """
    UMBRAL_TEST_MM50 = 1.0  # % de distancia a la MM50 que cuenta como "testando el nivel"
    raw = yf.download(['SPY', 'QQQ'], period='6mo', interval='1d',
                      auto_adjust=True, progress=False, threads=True)
    if raw.empty: return {}
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    out = {}
    for tk in ('SPY', 'QQQ'):
        if tk not in close.columns: continue
        p = close[tk].dropna()
        if len(p) < 55: continue
        ma50 = p.rolling(50).mean()
        validos = ~ma50.isna()
        pv, mv = p[validos].values, ma50[validos].values
        if len(pv) < 2: continue
        encima = pv > mv
        estado = 'sobre' if encima[-1] else 'bajo'
        sesiones = 1
        for i in range(len(encima) - 2, -1, -1):
            if encima[i] == encima[-1]: sesiones += 1
            else: break
        dist_pct = round(float(pv[-1] / mv[-1] - 1) * 100, 2)
        cruce = sesiones <= 2
        test = abs(dist_pct) <= UMBRAL_TEST_MM50
        out[tk] = {'precio': round(float(pv[-1]), 2), 'mm50': round(float(mv[-1]), 2),
                   'dist_pct': dist_pct, 'estado': estado, 'sesiones_en_estado': int(sesiones),
                   'cruce_reciente': bool(cruce), 'test_nivel': bool(test),
                   'alerta': bool(cruce or test)}
    if out:
        partes = [f"{tk} {v['estado']} MM50 ({v['dist_pct']:+}%, {v['sesiones_en_estado']}s)" for tk, v in out.items()]
        alertas = [tk for tk, v in out.items() if v['alerta']]
        print('  Indices vs MM50: ' + ' | '.join(partes) +
              (' | ⚠️ ' + ', '.join(alertas) if alertas else ' | sin eventos'))
    return out

def get_macro_data():
    try:
        tickers = ['^VIX', 'DX-Y.NYB', '^TNX']
        raw = yf.download(tickers, period='5d', interval='1d',
                          auto_adjust=True, progress=False, threads=True)
        if raw.empty: return {}
        close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
        result = {}
        for tk in tickers:
            if tk in close.columns:
                serie = close[tk].dropna()
                if len(serie) >= 2:
                    current = round(float(serie.iloc[-1]), 2)
                    prev    = round(float(serie.iloc[-2]), 2)
                    # ^TNX en Yahoo reporta el yield en décimas (43.7 en vez de 4.37)
                    # — normalizar si el valor parece estar en esa escala
                    if tk == '^TNX' and current > 20:
                        current = round(current / 10, 2)
                        prev    = round(prev / 10, 2)
                    chg     = round(current - prev, 2)
                    chg_pct = round((current/prev - 1)*100, 1)
                    result[tk] = {'current': current, 'prev': prev, 'chg': chg, 'chg_pct': chg_pct}
        # Si ^TNX no devolvió datos, intentar ^TYX (otro ticker del bono 10a en Yahoo)
        if '^TNX' not in result:
            try:
                raw2 = yf.download('^TYX', period='5d', interval='1d',
                                   auto_adjust=True, progress=False, threads=True)
                if not raw2.empty:
                    close2 = raw2['Close'] if isinstance(raw2.columns, pd.MultiIndex) else raw2[['Close']]
                    if isinstance(close2, pd.DataFrame):
                        serie2 = close2.iloc[:,0].dropna()
                    else:
                        serie2 = close2.dropna()
                    if len(serie2) >= 2:
                        current2 = round(float(serie2.iloc[-1]), 2)
                        prev2    = round(float(serie2.iloc[-2]), 2)
                        if current2 > 20: current2 = round(current2/10, 2); prev2 = round(prev2/10, 2)
                        result['^TNX'] = {'current': current2, 'prev': prev2,
                                          'chg': round(current2-prev2, 2),
                                          'chg_pct': round((current2/prev2-1)*100, 1)}
            except Exception as _e:
                _traza('macro/us10y-fallback-tyx', _e)
        # FALLBACK FINAL — API publica del Tesoro americano (fiscaldata.treasury.gov)
        # completamente independiente de Yahoo Finance, sin limite de peticiones, gratuita.
        # Se usa solo si los dos intentos anteriores via yfinance fallaron.
        if '^TNX' not in result:
            try:
                url = ('https://api.fiscaldata.treasury.gov/services/api/v1/accounting/od/'
                       'avg_interest_rates?fields=record_date,avg_interest_rate_amt'
                       '&filter=security_desc:eq:Treasury%20Notes&sort=-record_date&page[size]=2')
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data_tr = resp.json().get('data', [])
                    if len(data_tr) >= 2:
                        current3 = round(float(data_tr[0]['avg_interest_rate_amt']), 2)
                        prev3    = round(float(data_tr[1]['avg_interest_rate_amt']), 2)
                        result['^TNX'] = {'current': current3, 'prev': prev3,
                                          'chg': round(current3-prev3, 2),
                                          'chg_pct': round((current3/prev3-1)*100, 1),
                                          'fuente': 'treasury.gov'}
            except Exception as _e:
                _traza('macro/us10y-fallback-treasury', _e)
        # NUEVO (07/07) — vector macro de regimen (US30Y, WTI, credito HYG/IEF). Va dentro
        # de get_macro_data para que main() y las Celdas 3/4 lo hereden sin tocar sus call
        # sites, y para que persista en data.json automaticamente (data['macro']['regimen']).
        # Cualquier fallo aqui deja intacto el comportamiento clasico (VIX/DXY/US10Y).
        try:
            regimen = _calc_macro_regimen()
            if regimen: result['regimen'] = regimen
        except Exception as _e:
            _traza('macro/regimen', _e)
        # NUEVO (11/07) — estructura tactica de indices vs MM50, mismo patron de herencia
        try:
            indices = _calc_estructura_indices()
            if indices: result['indices_mm50'] = indices
        except Exception as _e:
            _traza('macro/indices-mm50', _e)
        return result
    except Exception as e:
        print(f'  Error macro: {e}')
        return {}

def get_fundamentals(tickers):
    """
    NOTA (28/06): se añaden campos de calificacion de analistas y precio objetivo de consenso.
    No requieren FMP ni ninguna API nueva — vienen incluidos en el mismo info de yfinance que
    ya se descarga aqui (recommendationKey/Mean, numberOfAnalystOpinions, targetMeanPrice, etc.).
    Distinto de las "revisiones de analistas" que si requeriran FMP (esto es una foto fija del
    consenso actual, no la tendencia de si las estimaciones han subido o bajado recientemente).
    """
    result = {}
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info
            precio_actual = info.get('currentPrice') or info.get('regularMarketPrice') or 0
            target_medio = info.get('targetMeanPrice', 0) or 0
            result[tk] = {
                'per_trailing':  round(info.get('trailingPE',  0) or 0, 1),
                'per_forward':   round(info.get('forwardPE',   0) or 0, 1),
                'peg':           round(info.get('pegRatio',    0) or 0, 2),
                'ev_ebitda':     round(info.get('enterpriseToEbitda', 0) or 0, 1),
                'margen_neto':   round((info.get('profitMargins', 0) or 0) * 100, 1),
                # NUEVO (05/07, opcion B) — margenes bruto y operativo: estaban en la lista FMP
                # pero vienen en el MISMO info de yfinance que ya se descarga aqui, sin coste
                'margen_bruto':     round((info.get('grossMargins', 0) or 0) * 100, 1) or None,
                'margen_operativo': round((info.get('operatingMargins', 0) or 0) * 100, 1) or None,
                'roe':           round((info.get('returnOnEquity', 0) or 0) * 100, 1),
                'deuda_equity':  round(info.get('debtToEquity', 0) or 0, 2),
                'rev_growth':    round((info.get('revenueGrowth', 0) or 0) * 100, 1),
                'eps_trailing':  round(info.get('trailingEps', 0) or 0, 2),
                'eps_fwd':       round(info.get('forwardEps', 0) or 0, 2),
                'eps_growth':    round((info.get('earningsGrowth', 0) or 0) * 100, 1),
                'sector':        info.get('sector', ''),
                'mkt_cap_b':     round((info.get('marketCap', 0) or 0) / 1e9, 1),
                # NUEVO (28/06) — consenso de analistas, sin coste adicional (mismo info de yfinance)
                'analista_consenso':    info.get('recommendationKey', ''),
                'analista_n_opiniones': info.get('numberOfAnalystOpinions', 0) or 0,
                'analista_target_medio': round(target_medio, 2) if target_medio else None,
                'analista_target_alto':  round(info.get('targetHighPrice', 0) or 0, 2) or None,
                'analista_target_bajo':  round(info.get('targetLowPrice', 0) or 0, 2) or None,
                'analista_upside_pct':  (round((target_medio/precio_actual - 1) * 100, 1)
                                         if target_medio and precio_actual else None),
            }
        except Exception as _e:
            _traza('fundamentales/ticker', _e)
            result[tk] = {}
    return result

def get_fundamentales_extra(tickers):
    """
    NUEVO (05/07, opcion B) — campos de catalizador y calidad via yfinance, sin API key y
    sin restriccion de simbolos (el free tier de FMP resulto restringir por SIMBOLO: solo
    mega-caps muy seguidos, inutil para este universo — verificado en produccion 05/07).

    Sustituye a los endpoints FMP equivalentes:
      - Revisiones de analistas 90d  <- Ticker.upgrades_downgrades (antes /stable/grades)
      - Sorpresa de earnings         <- Ticker.earnings_history    (antes /stable/earnings)
      - Crecimiento de FCF (anual)   <- Ticker.cashflow            (antes /stable/cash-flow-statement-growth)
    (Margen bruto/operativo se extraen en get_fundamentals(), del mismo info ya descargado.)

    Se llama SOLO sobre los setups validos finales (<=5), igual que get_news_earnings.
    yfinance es scraping de Yahoo: cada bloque va protegido y degrada campo a campo.
    PRIMERA EJECUCION: validar contra datos reales, como todo lo demas.
    """
    from datetime import datetime as _dt
    hoy = _dt.now(madrid).date()
    result = {tk: {} for tk in tickers}
    for tk in tickers:
        f = {}
        try:
            t = yf.Ticker(tk)
        except Exception as _e:
            _traza('fundamentales-extra/ticker', _e)
            result[tk] = f; continue
        # 1) Revisiones de analistas: acciones de los ultimos 90 dias
        try:
            ud = t.upgrades_downgrades
            if ud is not None and len(ud) > 0:
                subidas = bajadas = 0
                ultima = None
                # index = GradeDate (Timestamp, posiblemente tz-aware); columnas Firm/ToGrade/FromGrade/Action
                filas = sorted(ud.iterrows(), key=lambda x: str(x[0]), reverse=True)
                for fecha_idx, fila in filas:
                    try:
                        fecha = pd.Timestamp(fecha_idx).date()
                    except Exception as _e:
                        _traza('fundamentales-extra/fila-revision', _e)
                        continue
                    if (hoy - fecha).days > 90: continue
                    accion = str(fila.get('Action', '')).lower()
                    if accion == 'up': subidas += 1
                    elif accion == 'down': bajadas += 1
                    if ultima is None:
                        etiqueta = {'up': 'upgrade', 'down': 'downgrade', 'init': 'inicio de cobertura',
                                    'main': 'mantiene', 'reit': 'reitera'}.get(accion, accion or '?')
                        de = str(fila.get('FromGrade', '') or '').strip()
                        a = str(fila.get('ToGrade', '') or '').strip()
                        cambio = f'{de} -> {a}' if de else a
                        ultima = f"{etiqueta} de {fila.get('Firm','?')}: {cambio} ({fecha.isoformat()})"
                if subidas or bajadas or ultima:
                    f['upgrades_90d'] = subidas
                    f['downgrades_90d'] = bajadas
                    if ultima: f['ultima_revision'] = ultima
        except Exception as _e:
            _traza('fundamentales/revisiones-analistas', _e)
        # 2) Sorpresa de earnings: ultimo trimestre reportado (EPS real vs estimado)
        try:
            eh = t.earnings_history
            if eh is not None and len(eh) > 0:
                # index = fecha del trimestre; columnas epsEstimate/epsActual (+surprisePercent).
                # Se calcula la sorpresa manualmente, no se confia en la escala de surprisePercent.
                reportados = [(str(ix), fila) for ix, fila in eh.iterrows()
                              if pd.notna(fila.get('epsActual')) and pd.notna(fila.get('epsEstimate'))]
                if reportados:
                    reportados.sort(key=lambda x: x[0], reverse=True)
                    fecha_q, fila = reportados[0]
                    est = float(fila['epsEstimate']); real = float(fila['epsActual'])
                    if est != 0:
                        f['sorpresa_eps_pct'] = round((real - est) / abs(est) * 100, 1)
                        f['sorpresa_fecha'] = fecha_q[:10]
        except Exception as _e:
            _traza('fundamentales/sorpresa-eps', _e)
        # 3) Crecimiento de FCF: ultimo ejercicio anual vs anterior (solo si la base es positiva,
        # un crecimiento sobre FCF negativo no es interpretable como porcentaje)
        try:
            cf = t.cashflow
            if cf is not None and 'Free Cash Flow' in cf.index and cf.shape[1] >= 2:
                serie_fcf = cf.loc['Free Cash Flow'].dropna()
                if len(serie_fcf) >= 2:
                    actual, previo = float(serie_fcf.iloc[0]), float(serie_fcf.iloc[1])
                    if previo > 0:
                        f['fcf_growth'] = round((actual / previo - 1) * 100, 1)
        except Exception as _e:
            _traza('fundamentales/fcf-growth', _e)
        result[tk] = f
    n_con_datos = sum(1 for v in result.values() if v)
    print(f'  Extra (yfinance): datos de catalizador para {n_con_datos}/{len(tickers)} tickers')
    return result

def get_fundamentales_fmp(tickers, api_key=None):
    """
    REESCRITA (05/07, opcion B) — FMP queda como complemento OPCIONAL, reducido a los 2
    unicos campos sin equivalente en yfinance:
      - ROIC TTM                       <- /stable/key-metrics-ttm
      - Serie de EPS trimestrales      <- /stable/earnings (max 5T: el free tier limita limit<=5 desde 13/07/2026)
    El resto de campos (revisiones, sorpresa, margenes, FCF growth) los cubre yfinance
    via get_fundamentales_extra()/get_fundamentals() sin restriccion de simbolos.

    Semantica de rechazo verificada en produccion (05/07): el free tier de FMP restringe
    por SIMBOLO ("This value set for 'symbol' is not available under your current
    subscription"), no por endpoint. Un 402 que mencione el simbolo veta SOLO ese ticker
    (y corta sus endpoints restantes para no quemar cuota); un 402 sin mencion al simbolo
    (endpoint premium real) veta el endpoint globalmente. Presupuesto: 2 req/ticker, <=10/dia.
    Si no hay FMP_KEY, se omite en silencio: el scanner es plenamente funcional sin FMP.
    """
    if api_key is None: api_key = FMP_KEY
    api_key = (api_key or '').strip()  # keys copiadas a Colab Secrets pueden traer \n o espacios
    result = {tk: {} for tk in tickers}
    if not api_key or not tickers:
        return result
    base = 'https://financialmodelingprep.com/stable'
    no_disponibles = set()         # endpoints premium reales (veto global)
    simbolos_no_cubiertos = set()  # tickers fuera del universo del free tier (veto solo del ticker)

    def _fmp_get(endpoint, params):
        """Devuelve (datos, motivo_402) con motivo_402 in (None, 'simbolo', 'endpoint')."""
        if endpoint in no_disponibles: return None, None
        try:
            r = requests.get(f'{base}/{endpoint}', params={**params, 'apikey': api_key}, timeout=15)
            if r.status_code in (402, 403):
                cuerpo = r.text[:200].strip()
                if 'symbol' in cuerpo.lower():
                    return None, 'simbolo'
                no_disponibles.add(endpoint)
                print(f'  FMP: endpoint "{endpoint}" rechazado para todos los simbolos ({r.status_code}): {cuerpo}')
                return None, 'endpoint'
            if r.status_code != 200: return None, None
            data = r.json()
            return (data if data else None), None
        except Exception as _e:
            _traza('fmp/peticion', _e)
            return None, None

    for tk in tickers:
        f = {}
        # 1) ROIC TTM
        km, motivo = _fmp_get('key-metrics-ttm', {'symbol': tk})
        if motivo == 'simbolo':
            simbolos_no_cubiertos.add(tk)
            continue  # la restriccion es del simbolo: no quemar cuota en el otro endpoint
        if isinstance(km, list) and km:
            roic = km[0].get('returnOnInvestedCapitalTTM', km[0].get('roicTTM'))
            if roic is not None: f['fmp_roic'] = round(float(roic) * 100, 1)
        # 2) Serie de EPS de 8 trimestres (la sorpresa del ultimo trimestre ya la da yfinance)
        # FIX (13/07/2026): el free tier ahora limita 'limit' a un maximo de 5 (402:
        # "Premium Query Parameter... values for 'limit' must be between 0 and 5").
        # Con limit=12 el endpoint quedaba vetado globalmente. Se pide el maximo permitido:
        # 5 trimestres siguen cubriendo la comparacion interanual + 1 trimestre extra.
        earnings, motivo = _fmp_get('earnings', {'symbol': tk, 'limit': 5})
        if motivo == 'simbolo':
            simbolos_no_cubiertos.add(tk); result[tk] = f; continue
        if isinstance(earnings, list):
            reportados = [e for e in earnings if e.get('epsActual') is not None]
            reportados.sort(key=lambda e: str(e.get('date', '')), reverse=True)
            serie = [e.get('epsActual') for e in reportados[:5]][::-1]  # antiguo -> reciente (max 5, limite free tier)
            serie = [round(float(x), 2) for x in serie if x is not None]
            if len(serie) >= 4:
                f['fmp_eps_q'] = serie  # renombrado desde fmp_eps_8q al reducirse a 5T (13/07)
        result[tk] = f
    n_con_datos = sum(1 for v in result.values() if v)
    print(f'  FMP (complemento ROIC/EPStrim): datos para {n_con_datos}/{len(tickers)} tickers'
          + (f' | simbolos fuera del plan free: {", ".join(sorted(simbolos_no_cubiertos))}' if simbolos_no_cubiertos else '')
          + (f' | endpoints no disponibles: {", ".join(sorted(no_disponibles))}' if no_disponibles else ''))
    return result

def get_news_earnings(tickers, hoy=None):
    """
    PUNTO 7 — Earnings proximos + noticias recientes, via yfinance.

    Umbrales de earnings (consenso de gestion de riesgo en momentum swing trading:
    el stop tecnico no protege de un gap overnight tras un informe de resultados):
      < 10 dias  -> 'descartar'  (el setup no debe llegar a ENTRADAS)
      10-21 dias -> 'avisar'     (incluir con aviso explicito de Claude)
      > 21 dias  -> 'ignorar'    (no se menciona, no aporta valor)

    Devuelve dict por ticker:
      {'earnings_date': date|None, 'dias_earnings': int|None, 'earnings_accion': str,
       'noticias': [{'titulo':..., 'fuente':..., 'fecha':...}, ...]}
    """
    from datetime import datetime as _dt, date as _date
    if hoy is None:
        hoy = _dt.now(madrid).date()
    result = {}
    for tk in tickers:
        entry = {'earnings_date': None, 'dias_earnings': None, 'earnings_accion': 'ignorar', 'noticias': []}
        try:
            t = yf.Ticker(tk)
            # --- Earnings ---
            try:
                cal = t.calendar
                ed_list = cal.get('Earnings Date') if cal else None
                if ed_list:
                    ed = ed_list[0] if isinstance(ed_list, list) else ed_list
                    if isinstance(ed, _date):
                        dias = (ed - hoy).days
                        entry['earnings_date'] = ed.isoformat()
                        entry['dias_earnings'] = dias
                        if dias < 0:
                            # Earnings ya pasado (calendar de yfinance puede devolver la ultima
                            # fecha reportada si todavia no hay estimacion confirmada de la proxima).
                            # No es un riesgo de evento: no descarta ni avisa.
                            entry['earnings_accion'] = 'ignorar'
                        elif dias < 10:
                            entry['earnings_accion'] = 'descartar'
                        elif dias <= 21:
                            entry['earnings_accion'] = 'avisar'
                        else:
                            entry['earnings_accion'] = 'ignorar'
            except Exception as _e:
                _traza('earnings/calendario', _e)
            # --- Noticias recientes (max 3, solo titulo+fuente+fecha, sin imagenes/urls largas) ---
            try:
                news = t.news or []
                for n in news[:3]:
                    c = n.get('content', {})
                    titulo = c.get('title')
                    if not titulo:
                        continue
                    entry['noticias'].append({
                        'titulo': titulo,
                        'fuente': (c.get('provider') or {}).get('displayName', ''),
                        'fecha': (c.get('pubDate') or '')[:10],
                    })
            except Exception as _e:
                _traza('noticias/titulares', _e)
        except Exception as _e:
            _traza('noticias/bloque-ticker', _e)
        result[tk] = entry
    return result

def rs_score(t, b, w):
    try:
        t, b = t.dropna(), b.dropna()
        if len(t) < w or len(b) < w: return None
        return round((t.iloc[-1]/t.iloc[-w]-1)*100 - (b.iloc[-1]/b.iloc[-w]-1)*100, 2)
    except Exception as _e:
        _traza('rs_score', _e)
        return None

def volume_zscore(v, w=20):
    try:
        v = v.dropna()
        if len(v) < w+1: return None
        m, s = v.iloc[-(w+1):-1].mean(), v.iloc[-(w+1):-1].std()
        return round((v.iloc[-1]-m)/s, 2) if s else 0.0
    except Exception as _e:
        _traza('volume_zscore', _e)
        return None

def calc_rvol(v, w=30):
    """
    PUNTO 9 — RVOL (volumen relativo): volumen de la sesion actual / promedio de las
    ultimas w sesiones (30 por defecto, sin incluir la sesion actual en el promedio).
    Umbral sugerido por el documento de mejoras: RVOL > 1.2x confirma que una ruptura
    tiene respaldo de volumen real, no es un movimiento de baja participacion.
    Devuelve None si no hay suficiente historico.
    """
    try:
        v = v.dropna()
        if len(v) < w+1: return None
        promedio = v.iloc[-(w+1):-1].mean()
        if not promedio or promedio == 0: return None
        return round(v.iloc[-1] / promedio, 2)
    except Exception as _e:
        _traza('calc_rvol', _e)
        return None

def calc_volume_profile(h, l, v, window=60):
    """
    PUNTO 11 — Volume Profile aproximado (sustituto gratuito de datos de dark pool).
    Reparte el volumen de cada sesion uniformemente entre su minimo y su maximo (aproximacion
    TPO sobre datos diarios, no tick data real — los rangos high-low diarios aproximan pero no
    son equivalentes a un Volume Profile real con datos intraday).

    Ventana: 60 dias. Buckets de precio: ancho ~0.5% del precio medio del rango de la ventana.
    POC (Point of Control): nivel de precio con mas volumen acumulado.
    VAH/VAL (Value Area High/Low): rango de precio que concentra el 70% del volumen total,
    expandido desde el POC hacia el lado (superior o inferior) con mas volumen en cada paso.

    Devuelve {'poc':, 'vah':, 'val':} o None si no hay suficiente historico (<60 sesiones)
    o el rango de precio de la ventana es degenerado.
    """
    try:
        idx = h.dropna().index.intersection(l.dropna().index).intersection(v.dropna().index)
        if len(idx) < window: return None
        h_w = h.loc[idx].iloc[-window:]; l_w = l.loc[idx].iloc[-window:]; v_w = v.loc[idx].iloc[-window:]
        precio_min = float(l_w.min()); precio_max = float(h_w.max())
        if precio_max <= precio_min or precio_min <= 0: return None
        precio_medio = (precio_min + precio_max) / 2
        ancho_bucket = precio_medio * 0.005
        n_buckets = int((precio_max - precio_min) / ancho_bucket)
        n_buckets = max(min(n_buckets, 150), 15)  # margenes razonables independientemente de la volatilidad del rango
        bordes = np.linspace(precio_min, precio_max, n_buckets + 1)
        vol_bucket = np.zeros(n_buckets)
        for dia_lo, dia_hi, dia_vol in zip(l_w.values, h_w.values, v_w.values):
            dia_lo, dia_hi, dia_vol = float(dia_lo), float(dia_hi), float(dia_vol)
            if dia_hi <= dia_lo or dia_vol <= 0: continue
            b_lo = max(0, min(int(np.searchsorted(bordes, dia_lo, side='right')) - 1, n_buckets - 1))
            b_hi = max(0, min(int(np.searchsorted(bordes, dia_hi, side='right')) - 1, n_buckets - 1))
            if b_hi < b_lo: b_lo, b_hi = b_hi, b_lo
            vol_bucket[b_lo:b_hi + 1] += dia_vol / (b_hi - b_lo + 1)
        vol_total = vol_bucket.sum()
        if vol_total <= 0: return None
        centros = (bordes[:-1] + bordes[1:]) / 2
        idx_poc = int(np.argmax(vol_bucket))
        # Value area: expandir desde el POC hacia el lado adyacente con mas volumen, hasta cubrir el 70%
        objetivo = vol_total * 0.70
        acumulado = vol_bucket[idx_poc]
        izq, der = idx_poc - 1, idx_poc + 1
        incl_lo, incl_hi = idx_poc, idx_poc
        while acumulado < objetivo and (izq >= 0 or der < n_buckets):
            v_izq = vol_bucket[izq] if izq >= 0 else -1.0
            v_der = vol_bucket[der] if der < n_buckets else -1.0
            if v_izq >= v_der:
                acumulado += v_izq; incl_lo = izq; izq -= 1
            else:
                acumulado += v_der; incl_hi = der; der += 1
        return {'poc': round(float(centros[idx_poc]), 2),
                'val': round(float(centros[incl_lo]), 2),
                'vah': round(float(centros[incl_hi]), 2)}
    except Exception as _e:
        _traza('calc_volume_profile', _e)
        return None

def detect_breakout(p, v, lb=50, vm=1.4):
    try:
        p, v = p.dropna(), v.dropna()
        if len(p) < lb: return {'breakout': False, 'days_ago': None, 'breakout_level': None}
        for d in range(1, 11):
            bh = p.iloc[-(lb+d):-(d+5)].max(); bvm = v.iloc[-(lb+d):-(d+5)].mean()
            if p.iloc[-d] > bh and v.iloc[-d] > vm*bvm:
                return {'breakout': True, 'days_ago': d, 'breakout_level': round(float(bh), 2)}
        return {'breakout': False, 'days_ago': None, 'breakout_level': round(float(p.iloc[-lb:-5].max()), 2)}
    except Exception as _e:
        _traza('detect_breakout', _e)
        return {'breakout': False, 'days_ago': None, 'breakout_level': None}

def ma_health(p):
    try:
        p = p.dropna(); result = {'ma20':False,'ma50':False,'ma200':False,'ma20_val':None,'ma50_val':None,'ma200_val':None}
        c = float(p.iloc[-1])
        if len(p)>=20:
            ma20=float(p.rolling(20).mean().iloc[-1]); result['ma20']=bool(c>ma20); result['ma20_val']=round(ma20,2); result['pct_ma20']=round((c/ma20-1)*100,1)
        if len(p)>=50:
            ma50=float(p.rolling(50).mean().iloc[-1]); result['ma50']=bool(c>ma50); result['ma50_val']=round(ma50,2); result['pct_ma50']=round((c/ma50-1)*100,1)
        if len(p)>=200:
            ma200=float(p.rolling(200).mean().iloc[-1]); result['ma200']=bool(c>ma200); result['ma200_val']=round(ma200,2); result['pct_ma200']=round((c/ma200-1)*100,1)
        return result
    except Exception as _e:
        _traza('ma_health', _e)
        return {'ma20':False,'ma50':False,'ma200':False}

def calc_atr_series(high, low, close, period=14):
    """
    OPTIMIZACION — True Range y ATR como serie completa, factorizado fuera de calc_atr() y
    calc_adx(). Antes ambas funciones calculaban el mismo True Range de forma independiente
    con la formula identica (h-l, |h-c_prev|, |l-c_prev|) y el mismo rolling(14).mean() — trabajo
    duplicado en cada uno de los ~524 tickers del universo, todos los dias. calc_atr() y calc_adx()
    ahora comparten esta serie en vez de recalcularla cada una por su lado.

    Devuelve la serie completa de ATR14 (no solo el ultimo valor) o None si no hay datos suficientes.
    """
    try:
        h = high.squeeze().dropna(); l = low.squeeze().dropna(); c = close.squeeze().dropna()
        if len(c) < period + 1: return None
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    except Exception as _e:
        _traza('calc_atr_series', _e)
        return None

def calc_atr(high, low, close, period=14):
    try:
        atr_series = calc_atr_series(high, low, close, period)
        if atr_series is None: return None
        return round(float(atr_series.iloc[-1]), 4)
    except Exception as _e:
        _traza('calc_atr', _e)
        return None

def calc_rsi(close, period=14):
    try:
        c = close.squeeze().dropna()
        if len(c) < period + 1: return None
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return round(float(rsi.iloc[-1]), 1)
    except Exception as _e:
        _traza('calc_rsi', _e)
        return None

def calc_macd(close):
    try:
        c = close.squeeze().dropna()
        if len(c) < 35: return None, None, None
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - signal
        return round(float(macd.iloc[-1]), 4), round(float(signal.iloc[-1]), 4), round(float(hist.iloc[-1]), 4)
    except Exception as _e:
        _traza('calc_macd', _e)
        return None, None, None

def calc_adx(high, low, close, period=14, atr_series=None):
    try:
        h = high.squeeze().dropna(); l = low.squeeze().dropna(); c = close.squeeze().dropna()
        if len(c) < period * 2: return None
        # OPTIMIZACION — reutiliza la serie de ATR ya calculada (ver calc_atr_series) en vez de
        # recalcular el mismo True Range; si se llama de forma independiente sin pasarla, la
        # calcula igual que antes (comportamiento sin cambios para cualquier otro caller).
        atr14 = atr_series if atr_series is not None else calc_atr_series(high, low, close, period)
        if atr14 is None: return None
        dm_plus  = (h - h.shift(1)).clip(lower=0)
        dm_minus = (l.shift(1) - l).clip(lower=0)
        dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
        dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
        di_plus  = 100 * (dm_plus.rolling(period).mean()  / atr14)
        di_minus = 100 * (dm_minus.rolling(period).mean() / atr14)
        dx = 100 * ((di_plus - di_minus).abs() / (di_plus + di_minus))
        adx = dx.rolling(period).mean()
        return round(float(adx.iloc[-1]), 1)
    except Exception as _e:
        _traza('calc_adx', _e)
        return None

def calc_supertrend(high, low, close, period=10, multiplier=3):
    """
    NUEVO (27/06) — Supertrend como stop dinamico complementario al PUNTO 10 (seguimiento de
    posiciones abiertas). El sistema actual solo da una regla generica ("reducir al 50% si
    CMF<0.05"); Supertrend aporta un nivel de precio concreto y una señal binaria de cambio
    de tendencia, en vez de solo "presion compradora debilitandose".

    Formula estandar (ATR(10), multiplicador 3x — periodo distinto del ATR(14) que usa el resto
    del sistema para distancias de stop; se respeta la convencion habitual de Supertrend en vez
    de forzar el mismo periodo por consistencia interna):
      banda_basica_sup = (high+low)/2 + multiplier*ATR
      banda_basica_inf = (high+low)/2 - multiplier*ATR
    Las bandas finales son "pegajosas": solo se mueven a favor de la tendencia vigente, nunca
    en contra, hasta que el cierre cruza al otro lado y la direccion cambia. Mientras la
    direccion es alcista, el nivel relevante es la banda inferior (sube con el precio, nunca
    baja); si el cierre cruza por debajo, la direccion cambia a bajista (señal de salida).

    Devuelve {'nivel': float, 'direccion': 'alcista'|'bajista'} con el ultimo valor, o None si
    no hay datos suficientes (~3x el periodo, por la naturaleza iterativa del calculo).

    NOTA (fix 27/06): version anterior usaba idx.get_loc(fecha) para localizar la primera fila
    valida del ATR. Si el indice de fechas tuviera alguna fecha duplicada (yfinance puede
    producir esto en casos puntuales, p.ej. artefactos de zona horaria entre lotes descargados
    en momentos distintos), get_loc no devuelve un entero simple y el bucle posterior rompe.
    Reescrito para trabajar exclusivamente con arrays de numpy y posiciones enteras (.values,
    range()), sin ninguna busqueda por etiqueta de fecha — inmune a duplicados en el indice.
    """
    try:
        h = high.squeeze().dropna(); l = low.squeeze().dropna(); c = close.squeeze().dropna()
        idx = h.index.intersection(l.index).intersection(c.index)
        if len(idx) < period * 3: return None
        h, l, c = h.loc[idx], l.loc[idx], c.loc[idx]
        atr = calc_atr_series(h, l, c, period)
        if atr is None: return None

        # A partir de aqui, todo opera sobre arrays de numpy puros con posiciones enteras —
        # sin pd.Series, sin .loc/.iloc por etiqueta, inmune a indices con fechas duplicadas.
        h_a = h.values; l_a = l.values; c_a = c.values; atr_a = atr.values
        n = len(c_a)
        hl2 = (h_a + l_a) / 2
        banda_sup_basica = hl2 + multiplier * atr_a
        banda_inf_basica = hl2 - multiplier * atr_a

        validos = ~np.isnan(atr_a)
        if not validos.any(): return None
        inicio = int(np.argmax(validos))  # posicion del primer True (primer ATR valido)

        banda_sup_final = banda_sup_basica.copy()
        banda_inf_final = banda_inf_basica.copy()
        direccion = np.empty(n, dtype=object)
        direccion[:inicio + 1] = 'alcista'

        for i in range(inicio + 1, n):
            if banda_sup_basica[i] < banda_sup_final[i-1] or c_a[i-1] > banda_sup_final[i-1]:
                banda_sup_final[i] = banda_sup_basica[i]
            else:
                banda_sup_final[i] = banda_sup_final[i-1]
            if banda_inf_basica[i] > banda_inf_final[i-1] or c_a[i-1] < banda_inf_final[i-1]:
                banda_inf_final[i] = banda_inf_basica[i]
            else:
                banda_inf_final[i] = banda_inf_final[i-1]

            if c_a[i] > banda_sup_final[i-1]:
                direccion[i] = 'alcista'
            elif c_a[i] < banda_inf_final[i-1]:
                direccion[i] = 'bajista'
            else:
                direccion[i] = direccion[i-1]

        dir_final = direccion[-1]
        nivel = float(banda_inf_final[-1]) if dir_final == 'alcista' else float(banda_sup_final[-1])
        # NUEVO (05/07) — sesiones consecutivas en la direccion actual, contando hacia atras
        # desde el final hasta el ultimo cambio. Permite distinguir en el informe una ruptura
        # NUEVA (cambio en la sesion mas reciente, dias=1, urgente de actuar) de una ruptura
        # PERSISTENTE (lleva N sesiones rota y el precio no ha recuperado el nivel — la
        # decision de salida ya deberia haberse tomado; el informe la mantiene como
        # recordatorio pero el framing de urgencia no es el mismo). Sin coste de calculo:
        # la serie completa de direcciones ya existia internamente.
        # Nota: si la direccion no ha cambiado nunca en el historico descargado, el conteo
        # incluye el tramo inicial sembrado como 'alcista' — irrelevante en la practica
        # porque el dato se usa para alertas bajistas, que nunca son el tramo sembrado.
        dias_dir = 1
        for i in range(n - 2, -1, -1):
            if direccion[i] == dir_final: dias_dir += 1
            else: break
        return {'nivel': round(nivel, 2), 'direccion': dir_final,
                'dias_en_direccion_actual': int(dias_dir)}
    except Exception as _e:
        _traza('calc_supertrend', _e)
        return None

def calc_bollinger(close, period=20):
    try:
        c = close.squeeze().dropna()
        if len(c) < period: return None, None, None
        ma  = c.rolling(period).mean()
        std = c.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        bw    = (upper - lower) / ma  # bandwidth
        bw_min = bw.rolling(125).min()  # minimo 6 meses
        squeeze = bool(bw.iloc[-1] <= bw_min.iloc[-1] * 1.1)  # cerca del minimo
        pct_b = float((c.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]))
        return round(pct_b, 3), squeeze, round(float(bw.iloc[-1]), 4)
    except Exception as _e:
        _traza('calc_bollinger', _e)
        return None, None, None

def calc_cmf(high, low, close, volume, period=20):
    """
    Devuelve (cmf_actual, dias_negativo): el valor CMF de la ultima sesion y el numero de
    sesiones CONSECUTIVAS (contando desde la mas reciente) con CMF < 0.
    NUEVO (06/07) — dias_negativo sustenta la alerta blanda recalibrada: un CMF negativo
    un solo dia es ruido; negativo 3+ sesiones consecutivas es distribucion confirmada.
    Misma filosofia de persistencia que spy_health (anti-whiplash) y supertrend_dias.
    La serie ya se calculaba internamente: el conteo es gratis.
    """
    try:
        h = high.squeeze().dropna(); l = low.squeeze().dropna()
        c = close.squeeze().dropna(); v = volume.squeeze().dropna()
        if len(c) < period: return None, None
        hl_range = (h - l).replace(0, float('nan'))
        mfm = ((c - l) - (h - c)) / hl_range
        mfm = mfm.fillna(0)  # rango cero = sin presion compradora ni vendedora
        mfv = mfm * v
        vol_sum = v.rolling(period).sum()
        cmf = mfv.rolling(period).sum() / vol_sum.replace(0, float('nan'))
        val = float(cmf.iloc[-1])
        import math
        if math.isnan(val) or math.isinf(val): return None, None
        # Sesiones consecutivas con CMF<0 desde el final (los NaN cortan el conteo)
        dias_neg = 0
        for x in cmf.values[::-1]:
            if not math.isnan(x) and x < 0: dias_neg += 1
            else: break
        return round(val, 3), int(dias_neg)
    except Exception as _e:
        _traza('calc_cmf', _e)
        return None, None

def calc_obv(close, volume):
    try:
        c = close.squeeze().dropna(); v = volume.squeeze().dropna()
        if len(c) < 20: return None
        direction = c.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * v).cumsum()
        # OBV en maximo de 20 dias junto con precio
        obv_max20 = obv.rolling(20).max().iloc[-1]
        obv_curr  = obv.iloc[-1]
        obv_trend = round(float((obv_curr / obv_max20) if obv_max20 != 0 else 0), 3)
        return obv_trend  # 1.0 = en maximo, <1 = por debajo
    except Exception as _e:
        _traza('calc_obv', _e)
        return None

def calc_sct(adx, rsi, macd_hist, pct_b, squeeze, cmf, obv_trend):
    score = 0
    # ADX (20%) — tendencia fuerte si > 25
    if adx is not None:
        if adx >= 40:   score += 20
        elif adx >= 25: score += 14
        elif adx >= 15: score += 7
    # RSI (20%) — optimo entre 50-70
    if rsi is not None:
        if 55 <= rsi <= 70:   score += 20
        elif 50 <= rsi < 55:  score += 14
        elif 70 < rsi <= 80:  score += 10  # sobrecomprado pero en tendencia
        elif 45 <= rsi < 50:  score += 5
    # MACD histograma (20%) — positivo y creciendo
    if macd_hist is not None:
        if macd_hist > 0:   score += 20
        elif macd_hist > -0.001: score += 8
    # Bollinger (15%) — saliendo de squeeze al alza
    if pct_b is not None and squeeze is not None:
        if squeeze and pct_b > 0.5:   score += 15  # ruptura de compresion
        elif not squeeze and pct_b > 0.8: score += 10  # en banda superior
        elif pct_b > 0.5:              score += 7
    # CMF (15%) — flujo positivo; negativo en tendencia fuerte = divergencia bajista
    if cmf is not None:
        if cmf >= 0.2:    score += 15
        elif cmf >= 0.05: score += 10
        elif cmf >= 0:    score += 5
        elif cmf < 0 and adx is not None and adx >= 25:
            score -= 8  # divergencia bajista: distribucion institucional oculta
    # OBV (10%) — en maximos junto con precio
    if obv_trend is not None:
        if obv_trend >= 0.98:  score += 10
        elif obv_trend >= 0.90: score += 6
        elif obv_trend >= 0.80: score += 3
    return round(score, 1)

def calc_riesgo(sct, rsi, cmf, adx):
    """
    PUNTO 1 — Logica de RIESGO calculada en Python, no en el prompt de Claude.
    Elimina la correccion manual que antes hacia Claude sobre una formula mecanica inconsistente.

    BAJO:  SCT>=80 Y RSI entre 55-68 Y CMF>=0.10 Y ADX>=20   (las 4 condiciones, todas obligatorias)
    MEDIO: SCT entre 70-79 O RSI entre 68-70 O CMF entre 0.05-0.09
    ALTO:  cualquier otro caso — se descarta antes de llegar a Claude (no entra en valid[])

    NOTA (limpieza 27/06): existia ademas una banda "ADX en 17-19" en MEDIO. Era codigo muerto
    desde que se introdujo el filtro obligatorio ADX>=20 en valid[] (Punto 2): ningun ticker con
    ADX<20 llega nunca a esta funcion con opcion de generar informe, asi que esa rama jamas se
    ejecutaba en produccion. Documentado en su momento en la TABLA DE REGLAS (Punto 15); eliminada
    ahora que toca limpieza de codigo. Sin efecto en el comportamiento real del sistema.

    Si falta algun dato (None), se trata como el caso mas desfavorable para esa condicion.

    Devuelve (etiqueta, motivo). El motivo es la condicion EXACTA que disparo la clasificacion,
    para que Claude la use literalmente en vez de inferir o inventar una causa plausible pero
    no verificada — evita que el informe explique un MEDIO o BAJO con un razonamiento que suena
    coherente pero no es la condicion real que la formula evaluo.
    """
    if sct is None: sct = 0
    if rsi is None: rsi = 0
    if cmf is None: cmf = -1
    if adx is None: adx = 0

    bajo = (sct >= 80) and (55 <= rsi <= 68) and (cmf >= 0.10) and (adx >= 20)
    if bajo:
        return 'BAJO', f'SCT>=80 ({sct}), RSI 55-68 ({rsi}), CMF>=0.10 ({cmf}), ADX>=20 ({adx}) — las 4 condiciones cumplidas'

    motivos_medio = []
    if 70 <= sct <= 79: motivos_medio.append(f'SCT en 70-79 ({sct})')
    if 68 < rsi <= 70: motivos_medio.append(f'RSI en 68-70 ({rsi})')
    if 0.05 <= cmf <= 0.09: motivos_medio.append(f'CMF en 0.05-0.09 ({cmf})')
    if motivos_medio:
        return 'MEDIO', ' + '.join(motivos_medio)

    return 'ALTO', 'ninguna condicion de BAJO o MEDIO se cumple (no debe llegar a Claude)'

def price_stats(p, h, v_series):
    try:
        p=p.dropna()
        if len(p)<5: return {}
        current=float(p.iloc[-1]); w52=min(252,len(p)); ventana52=p.iloc[-w52:]
        high52=float(ventana52.max()); low52=float(ventana52.min())
        # PUNTO 12 — deteccion de correccion fuerte tras el maximo de 52 semanas: localizar el
        # minimo alcanzado DESPUES de la fecha del maximo (no el minimo absoluto de la ventana,
        # que podria ser anterior al maximo y no reflejar una correccion real post-maximo)
        idx_max = int(ventana52.values.argmax())
        posteriores = ventana52.iloc[idx_max:]
        min_post_max52 = float(posteriores.min()) if len(posteriores) >= 2 else None
        caida_post_max52_pct = (round((high52 - min_post_max52) / high52 * 100, 1)
                                 if min_post_max52 and high52 > 0 else None)
        return {'price':round(current,2),'high52':round(high52,2),'low52':round(low52,2),
                'pct_from_high':round((current/high52-1)*100,1),
                'ret_1w':round((current/p.iloc[-5]-1)*100,1) if len(p)>=5 else None,
                'ret_1m':round((current/p.iloc[-20]-1)*100,1) if len(p)>=20 else None,
                'ret_3m':round((current/p.iloc[-65]-1)*100,1) if len(p)>=65 else None,
                'low5':round(float(p.iloc[-5:].min()),2) if len(p)>=5 else round(current*0.95,2),
                'min_post_max52':round(min_post_max52,2) if min_post_max52 else None,
                'caida_post_max52_pct':caida_post_max52_pct}
    except Exception as _e:
        _traza('price_stats', _e)
        return {}

def calc_techo_realista(ps, entry_hi, stop, atr_val=None, rb_minimo=2.0):
    """
    PUNTO 12 — Techo realista para el objetivo cuando el maximo de 52 semanas quedo
    invalidado por una correccion fuerte posterior (>=20% desde ese maximo hasta su minimo
    posterior). Reconstruir todo el camino de vuelta al maximo bruto no es realista a corto
    plazo tras una caida de esa magnitud; se sustituye por un retroceso de Fibonacci de la
    propia caida, no por una extension del impulso de recuperacion (criterio mas conservador:
    asume recuperacion parcial de lo perdido, no superacion del maximo previo).

    Criterio (confirmado explicitamente): evaluar primero el retroceso 0.764 (mas generoso);
    si el R/B resultante respecto a entry_hi/stop no alcanza rb_minimo (2.0), evaluar 0.618;
    si tampoco alcanza, techo por ATR proyectado = entry_hi + ATR14*10 como ultimo recurso.
    Si no hubo correccion >=20% (o no hay datos suficientes), comportamiento sin cambios:
    techo = maximo de 52 semanas (high52).

    Devuelve (techo, metodo). 'metodo' queda registrado para trazabilidad/justificacion del
    objetivo en el informe (Punto 13), no se usa para ningun otro calculo.
    """
    high52 = ps.get('high52')
    if not high52: return None, 'sin_datos'
    min_post = ps.get('min_post_max52'); caida_pct = ps.get('caida_post_max52_pct')
    if not min_post or caida_pct is None or caida_pct < 20:
        return high52, 'max52'
    rango_caida = high52 - min_post
    techo_764 = round(min_post + rango_caida * 0.764, 2)
    techo_618 = round(min_post + rango_caida * 0.618, 2)
    riesgo = (entry_hi - stop) if (entry_hi and stop) else None
    if riesgo and riesgo > 0:
        if (techo_764 - entry_hi) / riesgo >= rb_minimo:
            return techo_764, 'fib0.764_post_correccion'
        if (techo_618 - entry_hi) / riesgo >= rb_minimo:
            return techo_618, 'fib0.618_post_correccion'
        if atr_val:
            return round(entry_hi + atr_val * 10, 2), 'atr_proyectado_post_correccion'
        return techo_764, 'fib0.764_post_correccion_sin_atr'
    return techo_764, 'fib0.764_post_correccion'

def _objetivo_escalonado(elo, ehi, stop, techo, ma200=None, high_recent=None, rb_minimo=2.0, rb_tope=5.0):
    """
    PUNTO 0 — Objetivo derivado del stop y de la distancia tecnica real, acotado a un R/B maximo,
    en lugar del maximo 52s bruto usado directamente como objetivo.

    riesgo        = entry_hi - stop   (medido conservador desde el limite alto del rango de entrada)
    rb_disponible = (techo - entry_hi) / riesgo   [techo = maximo 52s u otro nivel tecnico real]
    rb_objetivo   = min(rb_disponible, rb_tope)
    target_final  = entry_hi + riesgo * rb_objetivo
    rr_final      = (target_final - entry_hi) / riesgo   -> coincide con rb_objetivo por construccion

    El maximo 52s sigue siendo informacion (techo realista del valor) pero deja de ser el
    objetivo directo: si la distancia hasta el techo implica mas de rb_tope (5) veces el riesgo
    asumido, el objetivo se acota a 5R y el resto de la distancia queda fuera del informe.
    El R/B se mide siempre desde entry_hi (caso conservador), no desde entry_lo, para que el
    R/B mostrado sea siempre alcanzable en la practica.

    Ejemplo VRSK: entry_hi $178.5, stop $166.3 (riesgo=$12.2), maximo 52s=$310.13
    -> rb_disponible=10.79 (>5) -> rb_objetivo=5.0 -> target=178.5+12.2*5=$239.5, R/B=1:5.0
    en vez de $310.13 con R/B=1:15.5 inconsistente con el resto del ranking.

    Si rb_disponible < rb_minimo (2.0), el techo real no da margen suficiente: se devuelve igual
    el target acotado a esa distancia menor para que el filtro de validez lo descarte con un
    numero correcto, en lugar de forzar un objetivo irreal.

    PUNTO 4 — Objetivo escalonado: parcial (resistencia inmediata) y final (formula anterior).
    Devuelve (target_final, target_parcial, rr_final).
    """
    riesgo = ehi - stop
    if riesgo <= 0 or not techo:
        return None, None, None
    rb_disponible = (techo - ehi) / riesgo
    rb_objetivo = min(rb_disponible, rb_tope)
    target_final = round(ehi + riesgo * rb_objetivo, 2)
    rr_final = round((target_final - ehi) / riesgo, 1)
    # Objetivo parcial = resistencia inmediata (MM200 si aplica, o maximo reciente), siempre <= final
    candidatos_parciales = [x for x in [ma200, high_recent] if x and x > ehi]
    if candidatos_parciales:
        target_parcial = round(min(min(candidatos_parciales), target_final), 2)
    else:
        # Si no hay resistencia intermedia clara, el parcial es el punto medio del recorrido
        target_parcial = round(elo + riesgo * max(rb_minimo * 0.5, 1.0), 2)
        target_parcial = round(min(target_parcial, target_final), 2)
    # FIX (11/07, caso real $ZS 08/07) — colision del parcial con el techo acotado: si el
    # techo (fib 0.764 / ATR / tope 5R) queda por debajo de la resistencia intermedia, el
    # min() de arriba iguala parcial y final ($ZS: ambos en $191.68) y el parcial pierde su
    # funcion (no hay donde asegurar parte). Recalculo: parcial = 50% del recorrido
    # entrada->objetivo final. Si el recorrido es tan minusculo que ni eso da un nivel por
    # encima de la entrada, mejor sin parcial que con un parcial sin sentido.
    if target_parcial is not None and target_final is not None and target_parcial >= target_final:
        recalculado = round(ehi + (target_final - ehi) * 0.5, 2)
        target_parcial = recalculado if recalculado > ehi else None
    return target_final, target_parcial, rr_final

def _aviso_overextension(price, ma50, umbral=15.0, umbral_severo=20.0):
    """
    PUNTO 14 — un setup que ya cotiza a mas del umbral% (15% por defecto) sobre su MM50 esta
    sobre-extendido respecto a su propia estructura de base, incluso si tecnicamente sigue siendo
    valido segun el criterio especifico de su rama (p.ej. cerca del nivel de ruptura, o dentro
    del rango de pullback de otra media como la MM200, cuando MM50 y MM200 se han separado mucho).
    Se excluye de ENTRADAS y se redirige a EXTENDIDOS con aviso.

    Reutiliza el mismo criterio de severidad del Punto 6: si la distancia supera ademas el
    umbral_severo (20%), el trigger escala al mismo lenguaje "AVISO:" usado en la rama de ruptura
    extendida — independientemente de por que rama se detecto la sobre-extension, una misma
    distancia a MM50 debe redactarse con la misma severidad.

    Devuelve None si no aplica (sin MM50 disponible, o la distancia no supera el umbral).
    Si aplica, devuelve directamente el dict de retorno listo para usar como 'Extendido'.
    """
    if not ma50 or ma50 <= 0: return None
    dist_ma50 = round((price / ma50 - 1) * 100, 1)
    if dist_ma50 <= umbral: return None
    if dist_ma50 > umbral_severo:
        caida_necesaria = round((1 - (ma50*1.04)/price)*100, 1)
        trigger = (f'AVISO: el precio esta un {dist_ma50}% sobre su MM50 (${round(ma50,2)}), una distancia extrema. '
                   f'Alcanzar la zona de reentrada (±4% de MM50) exigiria una caida de aproximadamente {caida_necesaria}% '
                   f'desde el precio actual — un movimiento de esa magnitud probablemente invalidaria las condiciones '
                   f'de RSI>55 y CMF>0.05 exigidas para confirmar que la correccion no daño el momentum. '
                   f'Este trigger no debe interpretarse como un plan de reentrada realista a corto plazo; '
                   f'tratar como descartado salvo cambio sustancial de contexto.')
    else:
        trigger = (f'Vigilar reentrada cuando precio quede dentro de ±4% de MM50 (${round(ma50,2)}) '
                   f'y simultaneamente RSI>55 y CMF>0.05, confirmando que la correccion no ha danado el momentum.')
    return {'tipo':'Extendido — esperar pullback','entry_lo':None,'entry_hi':None,'stop':None,'target':None,
            'target_parcial':None,'rr':None,
            'nota':f'Precio {dist_ma50}% sobre su MM50 — sobre-extendido (umbral {umbral}%).','trigger':trigger,
            'dist_ma50_pct':dist_ma50}

def calc_entry_range(ps, bi, mah, atr_val=None):
    try:
        price=ps.get('price')
        if not price: return {}
        low5=ps.get('low5',price*0.95); high52=ps.get('high52',price)
        ma50=mah.get('ma50_val'); ma200=mah.get('ma200_val'); bl=bi.get('breakout_level')
        if bi.get('breakout') and bl and price<=bl*1.15:
            elo=round(price*0.995,2); ehi=round(price*1.020,2)
            stop=round(max(low5,price*0.930),2)
            if stop>=elo: stop=round(price*0.930,2)
            # Distancia minima stop: 1 ATR o 2% (lo que sea mayor)
            min_dist = max(atr_val if atr_val else 0, elo*0.02)
            if (elo - stop) < min_dist: stop=round(elo - min_dist, 2)
            # PUNTO 14 — esta rama no comprobaba distancia a MM50: una ruptura activa cerca del
            # nivel de ruptura podia estar, aun asi, muy sobre-extendida respecto a su MM50.
            overext = _aviso_overextension(price, ma50)
            if overext: return overext
            # PUNTO 0+4+12: techo = maximo 52s, salvo correccion fuerte posterior (ver calc_techo_realista)
            techo, techo_metodo = calc_techo_realista(ps, ehi, stop, atr_val)
            target, target_parcial, rr = _objetivo_escalonado(elo, ehi, stop, techo, ma200=ma200, high_recent=bl)
            tipo = 'Ruptura activa'
            # PUNTO 5: si la entrada exige un precio por encima del precio actual, es ruptura pendiente
            if elo > price:
                tipo = 'Ruptura pendiente'
            return {'tipo':tipo,'entry_lo':elo,'entry_hi':ehi,'stop':stop,'target':target,
                    'target_parcial':target_parcial,'rr':rr,'techo_metodo':techo_metodo}
        if bi.get('breakout') and bl and price>bl*1.15:
            # PUNTO 6: trigger concreto de entrada futura, no solo el nivel de MM50
            dist_ma50 = round((price/ma50-1)*100,1) if ma50 else None
            # Salvaguarda: si la distancia real sobre la MM50 es extrema (>20%), el trigger de
            # reentrada exigiria una caida brusca que probablemente invalidaria las propias
            # condiciones de RSI>55/CMF>0.05 exigidas para confirmar la reentrada. Detectado en
            # produccion: WDC a 51% sobre su MM50 con "solo" 19.9% sobre el nivel de ruptura —
            # las dos magnitudes son distintas y un trigger redactado sin esta distincion suena
            # a plan de reentrada realista cuando en la practica exige un desplome del ~31%.
            if ma50 and dist_ma50 is not None and dist_ma50 > 20:
                caida_necesaria = round((1 - (ma50*1.04)/price)*100, 1)
                trigger = (f'AVISO: el precio esta un {dist_ma50}% sobre su MM50 (${round(ma50,2)}), una distancia extrema. '
                           f'Alcanzar la zona de reentrada (±4% de MM50) exigiria una caida de aproximadamente {caida_necesaria}% '
                           f'desde el precio actual — un movimiento de esa magnitud probablemente invalidaria las condiciones '
                           f'de RSI>55 y CMF>0.05 exigidas para confirmar que la correccion no daño el momentum. '
                           f'Este trigger no debe interpretarse como un plan de reentrada realista a corto plazo; '
                           f'tratar como descartado salvo cambio sustancial de contexto.')
            elif ma50:
                trigger = (f'Vigilar reentrada cuando precio quede dentro de ±4% de MM50 (${round(ma50,2)}) '
                           f'y simultaneamente RSI>55 y CMF>0.05, confirmando que la correccion no ha danado el momentum.')
            else:
                trigger = 'Esperar pullback a una referencia tecnica clara antes de reevaluar.'
            return {'tipo':'Extendido — esperar pullback','entry_lo':None,'entry_hi':None,'stop':None,'target':None,
                    'target_parcial':None,'rr':None,
                    'nota':f'Precio {round((price/bl-1)*100,1)}% sobre ruptura.','trigger':trigger,
                    'dist_ma50_pct':dist_ma50}
        if ma50 and abs(price/ma50-1)<0.04:
            elo=round(ma50*0.990,2); ehi=round(ma50*1.010,2); stop=round(ma50*0.950,2)
            # Distancia minima stop: 1 ATR o 2%
            min_dist = max(atr_val if atr_val else 0, elo*0.02)
            if (elo - stop) < min_dist: stop=round(elo - min_dist, 2)
            # PUNTO 5 (3 casos) — comparar price contra el RANGO DE ENTRADA real, no contra el
            # filtro laxo de +-4% usado solo para decidir si activar esta rama del pullback.
            if price > ehi:
                # Caso 3: el precio ya supero la zona de entrada sin completar el retroceso.
                # No es un setup operable: se trata como EXTENDIDO con trigger de reentrada a MM50.
                dist_ma50 = round((price/ma50-1)*100,1)
                trigger = (f'Vigilar reentrada cuando precio quede dentro de ±4% de MM50 (${round(ma50,2)}) '
                           f'y simultaneamente RSI>55 y CMF>0.05, confirmando que la correccion no ha danado el momentum.')
                return {'tipo':'Extendido — esperar pullback','entry_lo':None,'entry_hi':None,'stop':None,'target':None,
                        'target_parcial':None,'rr':None,
                        'nota':f'Precio {dist_ma50}% sobre zona de entrada del pullback MM50.','trigger':trigger,
                        'dist_ma50_pct':dist_ma50}
            # PUNTO 0+4+12: techo = maximo 52s, salvo correccion fuerte posterior
            techo, techo_metodo = calc_techo_realista(ps, ehi, stop, atr_val)
            target, target_parcial, rr = _objetivo_escalonado(elo, ehi, stop, techo, ma200=ma200, high_recent=None)
            # Caso 1: entry_lo <= price <= entry_hi -> operativo ahora ('Pullback MM50')
            # Caso 2: price < entry_lo -> esperar a que baje hasta la zona ('Ruptura pendiente')
            tipo = 'Pullback MM50' if price >= elo else 'Ruptura pendiente'
            return {'tipo':tipo,'entry_lo':elo,'entry_hi':ehi,'stop':stop,'target':target,
                    'target_parcial':target_parcial,'rr':rr,'techo_metodo':techo_metodo}
        if ma200 and abs(price/ma200-1)<0.04:
            elo=round(ma200*0.990,2); ehi=round(ma200*1.010,2); stop=round(ma200*0.940,2)
            # Distancia minima stop: 1 ATR o 2%
            min_dist = max(atr_val if atr_val else 0, elo*0.02)
            if (elo - stop) < min_dist: stop=round(elo - min_dist, 2)
            # PUNTO 5 (3 casos) — mismo criterio que en MM50
            if price > ehi:
                dist_ma200 = round((price/ma200-1)*100,1)
                trigger = (f'Vigilar reentrada cuando precio quede dentro de ±4% de MM200 (${round(ma200,2)}) '
                           f'y simultaneamente RSI>55 y CMF>0.05, confirmando que la correccion no ha danado el momentum.')
                return {'tipo':'Extendido — esperar pullback','entry_lo':None,'entry_hi':None,'stop':None,'target':None,
                        'target_parcial':None,'rr':None,
                        'nota':f'Precio {dist_ma200}% sobre zona de entrada del pullback MM200.','trigger':trigger,
                        'dist_ma200_pct':dist_ma200}
            # PUNTO 14 — aunque el precio este cerca de su MM200, podria estar muy lejos de su
            # MM50 si ambas medias se han separado bastante; comprobacion por consistencia.
            overext = _aviso_overextension(price, ma50)
            if overext: return overext
            # PUNTO 0+4+12: techo = maximo 52s (sustituye al antiguo ma200*1.20 fijo), salvo correccion fuerte posterior
            techo, techo_metodo = calc_techo_realista(ps, ehi, stop, atr_val)
            target, target_parcial, rr = _objetivo_escalonado(elo, ehi, stop, techo, ma200=None, high_recent=None)
            tipo = 'Pullback MM200' if price >= elo else 'Ruptura pendiente'
            return {'tipo':tipo,'entry_lo':elo,'entry_hi':ehi,'stop':stop,'target':target,
                    'target_parcial':target_parcial,'rr':rr,'techo_metodo':techo_metodo}
        if bl and price<=bl*1.03:
            return {'tipo':'En vigilancia','entry_lo':None,'entry_hi':None,'stop':None,'target':None,
                    'target_parcial':None,'rr':None,'nivel_ruptura':round(bl,2)}
        return {'tipo':'Sin setup','entry_lo':None,'entry_hi':None,'stop':None,'target':None,
                'target_parcial':None,'rr':None}
    except Exception as _e:
        _traza('calc_entry_range', _e)
        return {}

def composite_score(r4, r13, vz, bo, ma, spy_healthy=True, spy_score=None):
    """
    P57 (09/08/2026): penalizacion proporcional via spy_score (float 0.0-1.0).
      factor = 0.70 + 0.30 * spy_score
        spy_score=1.0 -> factor 1.00 (sin penalizacion)
        spy_score=0.5 -> factor 0.85 (penalizacion del 15%, zona de transicion)
        spy_score=0.0 -> factor 0.70 (penalizacion maxima del 30%)
    Si spy_score es None, se mantiene el comportamiento binario original.
    """
    s=0
    if r4 is not None: s+=30*min(1,max(0,(r4+30)/60))
    if r13 is not None: s+=20*min(1,max(0,(r13+50)/100))
    if vz is not None: s+=25*min(1,max(0,(vz+1)/4))
    if bo: s+=15
    if ma: s+=10*(sum([ma.get('ma20',False),ma.get('ma50',False),ma.get('ma200',False)])/3)
    if spy_score is not None:
        factor = 0.70 + 0.30 * spy_score
        s = round(s * factor, 1)
    elif not spy_healthy:
        s = round(s * 0.70, 1)
    return round(s,1)



# ============================================================
# PUNTO 23 — CALENDARIO MACRO ESTATICO (15/07/2026)
# Equivalente a escala de mercado del detector de earnings por ticker: los eventos
# macro programados (FOMC, IPC, NFP) son riesgo binario de gap que afecta a TODAS
# las posiciones a la vez. Sin feed externo: las fechas son oficiales, se publican
# con un año de antelacion y se refrescan UNA VEZ AL AÑO (diciembre).
# Esta capa cubre lo PROGRAMADO; los eventos sorpresa (geopolitica, crisis) los
# cubre la capa de regimen por precios (VIX/WTI/tipos/credito). Se complementan.
# ============================================================

def get_calendario_macro():
    """Fechas oficiales 2026 de eventos macro USA de alto impacto. VERIFICADAS el
    15/07/2026 contra las fuentes primarias:
      - FOMC (dia de la decision, 14:00 ET):
        https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
      - IPC / CPI (08:30 ET): https://www.bls.gov/schedule/news_release/cpi.htm
      - NFP / Employment Situation (08:30 ET):
        https://www.bls.gov/schedule/news_release/empsit.htm
    PCE (BEA) excluido deliberadamente en v1: menor impacto de mercado que el trio
    incluido; candidato para el refresco anual si se echa en falta.
    REFRESCO ANUAL: en diciembre, sustituir por las fechas del año siguiente desde
    las mismas fuentes (el calendario FOMC 2027 ya esta publicado en la misma URL).
    """
    return {
        'FOMC (decision de tipos)': [
            '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17',
            '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09'],
        'IPC (CPI, 08:30 ET)': [
            '2026-01-13', '2026-02-13', '2026-03-11', '2026-04-10',
            '2026-05-12', '2026-06-10', '2026-07-14', '2026-08-12',
            '2026-09-11', '2026-10-14', '2026-11-10', '2026-12-10'],
        'NFP (Employment Situation, 08:30 ET)': [
            '2026-01-09', '2026-02-11', '2026-03-06', '2026-04-03',
            '2026-05-08', '2026-06-05', '2026-07-02', '2026-08-07',
            '2026-09-04', '2026-10-02', '2026-11-06', '2026-12-04'],
    }

HORA_PUBLICACION_ET = {'FOMC': 14, 'IPC': 9, 'NFP': 9}   # hora ET a partir de la cual el dato YA se conocio

def _evento_ya_publicado(nombre, dias_habiles, ahora_ny):
    """PUNTO 51 (08/08/2026) — ¿el evento de HOY ya se ha publicado?

    El P23 solo manejaba fechas, asi que el dia del evento marcaba 0 dias habiles y se
    anunciaba como pendiente hasta la medianoche. Fallo real: la nocturna de las 23:39
    del 07/08 decia "HOY se publica el NFP" y recomendaba esperar al dato, cuando
    llevaba nueve horas publicado, ya estaba en los precios del panel y habia provocado
    un +35% en TEAM. Recomendar prudencia ante un evento consumado es peor que callar.

    Se distingue por tipo porque no todos salen a la misma hora: IPC y NFP a las 08:30
    ET (se da por publicado a las 09:00, con margen) y la decision del FOMC a las 14:00.
    Solo aplica al dia del evento; para los futuros la pregunta no tiene sentido.
    """
    if dias_habiles != 0:
        return False
    clave = next((k for k in HORA_PUBLICACION_ET if nombre.upper().startswith(k)), None)
    if clave is None:
        return False
    return ahora_ny.hour >= HORA_PUBLICACION_ET[clave]

def check_eventos_macro(umbral_dias=2, hoy=None, ahora=None):
    """Eventos macro programados en los proximos `umbral_dias` dias habiles
    (incluido hoy: un evento esta mañana antes de la apertura sigue siendo riesgo
    para posiciones abiertas esta noche). `hoy` y `ahora` inyectables para tests.
    Devuelve lista de dicts {evento, fecha, dias_habiles, ya_publicado} ordenada por
    proximidad. Si el calendario esta agotado (ultima fecha en el pasado), avisa en el
    log del refresco anual pendiente — el fallo silencioso seria quedarse ciego sin saberlo.
    """
    if ahora is None:
        ahora = pd.Timestamp.now(tz='America/New_York').tz_localize(None)
    else:
        ahora = pd.Timestamp(ahora)
    if hoy is None:
        hoy = ahora.normalize()
    else:
        hoy = pd.Timestamp(hoy).normalize()
    proximos = []
    ultima_fecha = pd.Timestamp('1900-01-01')
    for evento, fechas in get_calendario_macro().items():
        for f in fechas:
            fecha = pd.Timestamp(f)
            ultima_fecha = max(ultima_fecha, fecha)
            if fecha < hoy:
                continue
            dias_habiles = max(0, len(pd.bdate_range(hoy, fecha)) - 1)
            if dias_habiles <= umbral_dias:
                proximos.append({'evento': evento, 'fecha': str(fecha.date()),
                                 'dias_habiles': int(dias_habiles),
                                 'ya_publicado': _evento_ya_publicado(evento, dias_habiles, ahora)})
    if ultima_fecha < hoy:
        print('  AVISO CALENDARIO MACRO: agotado (ultima fecha ' +
              str(ultima_fecha.date()) + ') — pendiente el refresco anual de fechas')
    return sorted(proximos, key=lambda x: x['dias_habiles'])

def formato_eventos_macro_summary(eventos):
    """PUNTO 23 — bloque de texto para el summary del prompt."""
    if not eventos:
        return ''
    pendientes = [e for e in eventos if not e.get('ya_publicado')]
    publicados = [e for e in eventos if e.get('ya_publicado')]
    s = ''
    if pendientes:
        s += '\nEVENTOS MACRO PROGRAMADOS PROXIMOS (riesgo de gap a escala de mercado):\n'
        for e in pendientes:
            cuando = 'HOY, AUN POR PUBLICAR' if e['dias_habiles'] == 0 else f"en {e['dias_habiles']} dia(s) habil(es)"
            s += f"- {e['evento']}: {e['fecha']} ({cuando})\n"
    # PUNTO 51 — un evento ya publicado NO es riesgo de gap: es contexto de lo que ya
    # ha movido los precios de esta sesion. Se separa para que el informe deje de
    # recomendar esperar a un dato consumado.
    if publicados:
        s += '\nEVENTOS MACRO YA PUBLICADOS HOY (el dato YA esta en los precios de esta sesion):\n'
        for e in publicados:
            s += f"- {e['evento']}: {e['fecha']} — YA PUBLICADO\n"
        s += ('NO recomiendes esperar a estos datos ni los presentes como riesgo de gap pendiente: '
              'ya han ocurrido. Si el movimiento de la sesion es coherente con ellos, dilo; si no '
              'dispones de la cifra concreta, limitate a señalar que la sesion ya los incorpora.\n')
    return s


# ============================================================
# PUNTO 24 — SERIE HISTORICA DE AMPLITUD (18/07/2026)
# El bloque de amplitud (dispersion, McClellan, MM20/50/200, A/D) se calculaba cada
# noche y se TIRABA: history.json solo persiste temas. La calibracion del umbral de
# dispersion (septiembre, junto al Punto 16) presupone una serie diaria que no
# existia como dato. Este punto la persiste en breadth_history.json, incluida en el
# commit unico de cada ejecucion. El fichero se sembro el 18/07 con las lecturas
# retrospectivas reconstruidas del historial de git (anotadas como tales).
# ============================================================

def construir_entrada_breadth(breadth, macro, ts, ahora_ny=None):
    """Una linea de la serie: fecha ISO + metricas de amplitud del dia + VIX.

    FECHA = la de la ULTIMA SESION DEL PANEL (dejada en _BREADTH_CACHE por
    check_frescura_panel), NO la del reloj: las nocturnas corren pasada la medianoche
    (00:0x) y fecharlas por reloj asignaria el cierre del jueves al viernes, colisionando
    ademas con el cierre real del viernes (defecto detectado al construir la semilla el
    18/07). Fallback si el cache no esta: fecha del reloj corregida — hora <06:00 o fin
    de semana retroceden al ultimo dia habil anterior."""
    fecha_iso = _BREADTH_CACHE.get('ultima_sesion')
    if not fecha_iso:
        partes = str(ts).split(' ')[0].split('/')
        if len(partes) == 3:
            t = pd.Timestamp(f'{partes[2]}-{partes[1]}-{partes[0]}')
            try:
                hora = int(str(ts).split(' ')[1].split(':')[0])
            except Exception as _e:
                _traza('amplitud/hora-fallback', _e)
                hora = 12
            if hora < 6 or t.dayofweek >= 5:
                t = t - pd.offsets.BDay(1)
            fecha_iso = str(t.date())
        else:
            fecha_iso = str(ts)
    b = breadth or {}
    disp = b.get('dispersion') or {}
    # PUNTO 79 (28/08/2026) — es_cierre por hora REAL de Nueva York.
    #
    # El P29 leia la hora de `ts`, que es hora de MADRID pese a que la variable se llamaba
    # `hora_ny`, y marcaba cierre solo si esa hora era <06:00. El comentario asumia que la
    # nocturna corre "pasada la medianoche (00:0x)", pero el cron disparaba entre las 23:24
    # y las 23:49 de Madrid: SIEMPRE antes de medianoche. Consecuencia medida el 28/08:
    # NINGUNA ejecucion programada habia sido nunca un cierre, y los CINCO consumidores que
    # leen este campo (checkpoints P39, resoluciones, put/call P56, alertas P68 y registro
    # P75) omitian la escritura toda noche sin ejecucion manual de madrugada —
    # setups_registro.json tenia 67 filas en solo cuatro fechas.
    #
    # Criterio nuevo, independiente del reloj de Madrid y del cron: la sesion del panel esta
    # CERRADA si es anterior a hoy en Nueva York, o si es la de hoy y ya pasaron las 16:00
    # ET. Asi el campo describe el mercado, no la hora a la que alguien lanza el proceso, y
    # no se rompe con el cambio de horario de octubre.
    #
    # `ahora_ny` es inyectable para los tests, mismo patron que check_eventos_macro (P51).
    if ahora_ny is None:
        try:
            ahora_ny = pd.Timestamp.now(tz='America/New_York')
        except Exception as _e:
            _traza('amplitud/hora-ny', _e)
            ahora_ny = None
    # Se evalua la fecha que ESTA ENTRADA va a llevar (fecha_iso), no la del cache: cuando el
    # cache no esta, el fallback de arriba ya ha calculado la sesion correcta y esa es la que
    # debe decidir. Solo se acepta si tiene forma de fecha ISO, porque el ultimo fallback
    # (`str(ts)`) puede devolver cualquier cosa y compararla seria peor que no comparar.
    _sesion_panel = fecha_iso if (isinstance(fecha_iso, str) and len(fecha_iso) == 10
                                  and fecha_iso[4] == '-' and fecha_iso[7] == '-') else None
    if ahora_ny is None or not _sesion_panel:
        # Sin poder comprobarlo se asume PROVISIONAL: escribir de menos se corrige en la
        # ejecucion siguiente, pero congelar por error es irreversible (leccion del P39).
        es_cierre = False
    else:
        _hoy_ny = str(pd.Timestamp(ahora_ny).normalize().date())
        if _sesion_panel < _hoy_ny:
            es_cierre = True                      # sesion pasada: cerrada con seguridad
        elif _sesion_panel == _hoy_ny:
            es_cierre = int(pd.Timestamp(ahora_ny).hour) >= 16   # cierra a las 16:00 ET
        else:
            es_cierre = False                     # fecha futura: dato raro, no congelar
    return {
        'fecha': fecha_iso,
        'es_cierre': es_cierre,
        'dispersion_ratio': disp.get('ratio'),
        'mcclellan': b.get('mcclellan'),
        'pct_sobre_mm20': b.get('pct_sobre_mm20'),
        'pct_sobre_mm50': b.get('pct_sobre_mm50'),
        'pct_sobre_mm200': b.get('pct_sobre_mm200'),
        'avance': b.get('avance'),
        'descenso': b.get('descenso'),
        'nuevos_max_52s': b.get('nuevos_max_52s'),
        'nuevos_min_52s': b.get('nuevos_min_52s'),
        'vix': ((macro or {}).get('^VIX') or {}).get('current'),
    }

def calcular_tendencia(valores, tolerancia=0.15):
    """PUNTO 37 (02/08/2026) — sesiones consecutivas en la misma tendencia.

    El informe describia cada noche la amplitud como si fuera informacion nueva:
    "dispersion 0.329" no dice si lleva 3 sesiones cayendo o 18. Esta funcion aporta
    ese contexto temporal.

    CRITERIO. Conteo estricto de sesiones consecutivas en la misma direccion no sirve:
    la serie de julio tuvo varios rebotes de un dia que lo habrian reiniciado, cuando
    la lectura util era "lleva 18 sesiones bajando". Aqui se cuenta desde el ultimo
    EXTREMO LOCAL, tolerando retrocesos menores: la racha sigue viva mientras el valor
    no se aleje del extremo mas de `tolerancia` veces el recorrido acumulado de la
    propia racha. Un rebote pequeño no la rompe; uno que devuelve buena parte del
    movimiento, si.

    La alternativa era definir el regimen por umbral, pero eso exige fijar un umbral
    de dispersion, que es justo lo que el P32 dejo sin base empirica para fijar.

    valores: lista de floats en orden cronologico (el ultimo es el mas reciente).
    Devuelve (direccion, sesiones) con direccion en {'bajista','alcista','plana'}.
    Con menos de 3 valores devuelve ('plana', 0): no hay serie que interpretar.
    """
    limpios = [float(v) for v in (valores or []) if v is not None]
    if len(limpios) < 3:
        return 'plana', 0

    actual = limpios[-1]
    mejor_baja = mejor_alta = None   # sesiones sostenidas en cada direccion

    for direccion in ('bajista', 'alcista'):
        extremo = limpios[-1]
        sesiones = 0
        recorrido = 0.0
        # Se recorre hacia atras: cada paso comprueba si la racha sigue viva.
        for i in range(len(limpios) - 2, -1, -1):
            v = limpios[i]
            if direccion == 'bajista':
                # Bajista = el valor ha venido cayendo: hacia atras deberia SUBIR.
                if v >= extremo:
                    extremo = v
                    recorrido = abs(extremo - actual)
                    sesiones += 1
                elif recorrido > 0 and abs(v - extremo) <= tolerancia * recorrido:
                    sesiones += 1      # retroceso menor: no rompe la racha
                else:
                    break
            else:
                if v <= extremo:
                    extremo = v
                    recorrido = abs(actual - extremo)
                    sesiones += 1
                elif recorrido > 0 and abs(extremo - v) <= tolerancia * recorrido:
                    sesiones += 1
                else:
                    break
        if direccion == 'bajista':
            mejor_baja = sesiones
        else:
            mejor_alta = sesiones

    if mejor_baja >= 2 and mejor_baja > mejor_alta:
        return 'bajista', mejor_baja
    if mejor_alta >= 2 and mejor_alta > mejor_baja:
        return 'alcista', mejor_alta
    return 'plana', 0

def merge_breadth_entry(bh, entrada, max_entradas=500):
    """PUNTO 29 — Fusiona una entrada en la serie con regla de precedencia:
    - Un CIERRE (es_cierre=True) sobreescribe siempre cualquier entrada del mismo dia.
    - Una PROVISIONAL (es_cierre=False) solo se escribe si NO hay ya un cierre del mismo
      dia — evita que una ejecucion manual vespertina borre el cierre de la madrugada
      cuando el usuario lanza una prueba el mismo dia que ya corrio la nocturna.
    Antes del P29, cualquier ejecucion del dia borraba la anterior; ahora el cierre
    es inamovible salvo por otro cierre (p. ej. la nocturna que corrige datos rancios).
    """
    fecha = entrada.get('fecha')
    existente = next((e for e in (bh or []) if e.get('fecha') == fecha), None)
    if existente and not entrada.get('es_cierre') and existente.get('es_cierre'):
        return sorted(bh, key=lambda e: e.get('fecha') or '')[-max_entradas:]  # provisional no pisa cierre
    bh = [e for e in (bh or []) if e.get('fecha') != fecha]
    bh.append(entrada)
    bh = sorted(bh, key=lambda e: e.get('fecha') or '')[-max_entradas:]
    # PUNTO 37 — el contador se calcula DESPUES de ordenar, sobre la serie ya fusionada:
    # necesita el historico, no solo la entrada del dia. Se anota unicamente en la ultima
    # entrada (las anteriores conservan el valor que tuvieran cuando fueron la ultima),
    # de modo que la serie guarda como se veia la tendencia en cada momento.
    if bh:
        for campo, clave in (('dispersion_ratio', 'tendencia_dispersion'),
                             ('mcclellan', 'tendencia_mcclellan')):
            direccion, sesiones = calcular_tendencia([e.get(campo) for e in bh])
            bh[-1][clave] = direccion
            bh[-1][clave.replace('tendencia_', 'sesiones_')] = sesiones
    return bh

def anotar_tendencia_en_breadth(breadth, macro, ts):
    """PUNTO 37b (02/08/2026) — lleva el contador de tendencia AL INFORME.

    El P37 persistia el contador en breadth_history.json, pero generate_analysis
    trabaja con data['breadth'] (la foto del dia) y nunca veia la serie: el informe
    seguia describiendo "dispersion 0.329" como si fuera informacion nueva, sin decir
    si lleva 3 sesiones subiendo o 18 bajando.

    Lee la serie ANTES del analisis y anota breadth['tendencia'] con direccion y
    sesiones de dispersion y McClellan, incluyendo ya el valor de hoy. La serie leida
    queda en _BREADTH_CACHE para que actualizar_breadth_history no tenga que releerla:
    una sola peticion a GitHub por ejecucion.

    Degrada en silencio: si la lectura falla o la serie es corta no anota nada y el
    prompt omite la linea. Nunca aborta ni inventa un contador.
    """
    if not isinstance(breadth, dict):
        return breadth
    bh, _ = get_github_file('breadth_history.json')
    if bh == '__ERROR__' or not isinstance(bh, list):
        return breadth
    _BREADTH_CACHE['historial'] = bh
    # ARREGLO (04/08/2026) — DOBLE CONTEO. Abajo se anade el valor de hoy a la serie leida,
    # correcto en la nocturna porque esa sesion aun no esta en el fichero. Pero al reejecutar
    # el mismo panel la fecha YA esta guardada y el valor se contaba dos veces, inflando el
    # contador del prompt en uno (los informes del 02/08 decian "4 sesiones" donde el fichero
    # guardaba 3, en dispersion y en McClellan). El dato persistido nunca se corrompio: solo
    # lo que veia el modelo. Se filtra la entrada de la sesion del panel antes de anadir.
    fecha_panel = construir_entrada_breadth({}, macro, ts).get('fecha')
    previas = [e for e in bh if e.get('fecha') != fecha_panel]
    disp_hoy = (breadth.get('dispersion') or {}).get('ratio')
    serie_disp = [e.get('dispersion_ratio') for e in previas] + [disp_hoy]
    serie_mcc = [e.get('mcclellan') for e in previas] + [breadth.get('mcclellan')]
    dir_d, ses_d = calcular_tendencia(serie_disp)
    dir_m, ses_m = calcular_tendencia(serie_mcc)
    if ses_d or ses_m:
        breadth['tendencia'] = {'dispersion': dir_d, 'sesiones_dispersion': ses_d,
                                'mcclellan': dir_m, 'sesiones_mcclellan': ses_m}
    return breadth

def actualizar_breadth_history(breadth, macro, ts):
    """Lee la serie del repo, fusiona la entrada de hoy y la devuelve para subirla.
    Mismo cinturon de seguridad que update_history (incidente 08/07): si la lectura
    de GitHub falla de forma persistente ('__ERROR__'), se OMITE la actualizacion de
    esta ejecucion y el fichero del repo queda intacto — devuelve None y el llamador
    no lo incluye en el commit. Un 404 (fichero aun inexistente) arranca de lista vacia."""
    # P37b — si anotar_tendencia_en_breadth ya leyo la serie en esta ejecucion se
    # reutiliza (una sola peticion a GitHub); si no la leyo o fallo, se lee aqui.
    bh = _BREADTH_CACHE.get('historial')
    if bh is None:
        bh, _ = get_github_file('breadth_history.json')
    if bh == '__ERROR__':
        print('  🔴 breadth_history OMITIDO en esta ejecucion (fallo de lectura de GitHub) — el fichero del repo queda intacto')
        return None
    if not isinstance(bh, list):
        bh = []
    return merge_breadth_entry(bh, construir_entrada_breadth(breadth, macro, ts))

# ============================================================
# PUT/CALL DE CBOE — INSTRUMENTACION (PUNTO 56, 08/08/2026)
# Unico componente del sentimiento que el scanner no podia ver: mide posicionamiento en
# OPCIONES, no precio ni volumen de acciones. El resto del indice Fear&Greed de CNN ya
# esta cubierto (momentum, amplitud, maximos/minimos, VIX, HYG/IEF), asi que se instrumenta
# solo esta pieza en vez del compuesto, que seria empaquetar en un digito lo que ya se mide
# con mas resolucion.
#
# Fuente: pagina de estadisticas diarias de Cboe, publica y sin clave. NO admite fecha: sirve
# siempre la ultima sesion disponible. En la nocturna (00:00 Madrid = 18:00 ET) el dato ya es
# de cierre; una ejecucion intradia cogeria datos parciales, asi que solo persiste el cierre,
# mismo criterio que el P39.
#
# Se parsea con pandas.read_html en vez de expresiones regulares: la pagina puede recolocar
# las tablas sin avisar, y buscar por CONTENIDO de la celda aguanta eso mucho mejor que
# depender del orden o del maquetado.
#
# Es INSTRUMENTACION: no alerta, no toca el score y de momento no va al prompt. Igual que la
# curva de tipos, primero se acumula serie propia y despues se decide si acoplarla.
# ============================================================

URL_PUTCALL_CBOE = 'https://www.cboe.com/markets/us/options/market-statistics/daily/'

RATIOS_PUTCALL = {
    'equity': 'EQUITY PUT/CALL RATIO',        # el termometro clasico de sentimiento minorista
    'total': 'TOTAL PUT/CALL RATIO',
    'index': 'INDEX PUT/CALL RATIO',          # contaminado por coberturas institucionales
    'etp': 'EXCHANGE TRADED PRODUCTS PUT/CALL RATIO',
    'vix': 'CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO',
    'spx': 'SPX + SPXW PUT/CALL RATIO',
}

def get_putcall_cboe():
    """Descarga los ratios put/call de la pagina diaria de Cboe. Devuelve dict o {}.

    Degrada en silencio-con-traza: si la pagina cambia, no responde o no trae la tabla,
    se devuelve {} y el scanner sigue. Nunca aborta por esto.
    """
    # Se parsea con BeautifulSoup y NO con pandas.read_html: read_html exige lxml (o
    # bs4 + html5lib), y el workflow solo instala beautifulsoup4. Usar lo que ya hay
    # evita añadir una dependencia para una sola funcion — y una dependencia que falte
    # aqui no romperia nada visible, solo dejaria de recogerse el dato en silencio.
    try:
        from bs4 import BeautifulSoup
        r = requests.get(URL_PUTCALL_CBOE, timeout=30,
                         headers={'User-Agent': 'Mozilla/5.0 (scanner-temas)'})
        if r.status_code != 200:
            _traza('putcall/http', RuntimeError(f'status {r.status_code}'))
            return {}
        sopa = BeautifulSoup(r.text, 'html.parser')
    except Exception as _e:
        _traza('putcall/descarga', _e)
        return {}
    # Se busca por CONTENIDO de celda, recorriendo todas las filas de la pagina: la
    # posicion de la tabla puede cambiar, el texto "PUT/CALL RATIO" no.
    etiquetas = {}
    for fila in sopa.find_all('tr'):
        celdas = [c.get_text(strip=True) for c in fila.find_all(['td', 'th'])]
        if len(celdas) < 2:
            continue
        clave = celdas[0].upper()
        if 'PUT/CALL RATIO' in clave:
            try:
                etiquetas[clave] = float(celdas[1])
            except (TypeError, ValueError):
                continue
    if not etiquetas:
        _traza('putcall/tabla-no-encontrada', RuntimeError('sin filas PUT/CALL RATIO'))
        return {}
    salida = {}
    for nombre, etiqueta in RATIOS_PUTCALL.items():
        if etiqueta in etiquetas:
            salida[nombre] = etiquetas[etiqueta]
    return salida

def construir_entrada_putcall(putcall, macro, ts):
    """Una linea de la serie: fecha de SESION del panel + los ratios del dia."""
    if not putcall:
        return None
    base = construir_entrada_breadth({}, macro, ts)
    return {'fecha': base.get('fecha'), 'es_cierre': base.get('es_cierre'), **putcall}

def merge_putcall(ph, entrada, max_entradas=2000):
    """Misma regla de precedencia que el P29: un cierre no lo pisa una provisional."""
    fecha = entrada.get('fecha')
    existente = next((e for e in (ph or []) if e.get('fecha') == fecha), None)
    if existente and not entrada.get('es_cierre') and existente.get('es_cierre'):
        return sorted(ph, key=lambda e: e.get('fecha') or '')[-max_entradas:]
    ph = [e for e in (ph or []) if e.get('fecha') != fecha]
    ph.append(entrada)
    return sorted(ph, key=lambda e: e.get('fecha') or '')[-max_entradas:]

# ============================================================
# P57 (09/08/2026) — SPY HEALTH SCORE CONTINUO
# Persiste la serie diaria del score continuo 0.0-1.0 en spy_health_history.json
# (fichero separado — opcion A — para no alterar la comparabilidad del historico
# de scores ya acumulado). El booleano spy_ok sigue siendo la variable de control
# del sistema; el score es instrumentacion, igual que el put/call.
# No alerta, no toca el prompt, no va al ranking hasta tener serie acumulada.
# ============================================================

MAX_SPY_HEALTH_ENTRIES = 500  # ~2 años de sesiones

def actualizar_spy_health_history(spy_score, spy_ok, spy_close, ma200, ts):
    """Persiste una entrada diaria en spy_health_history.json."""
    if spy_score is None:
        return None
    try:
        from datetime import datetime as _dt
        fecha = _dt.strptime(ts, '%d/%m/%Y %H:%M').strftime('%Y-%m-%d')
    except Exception:
        fecha = ts[:10]
    entrada = {
        'fecha': fecha,
        'spy_score': spy_score,
        'spy_ok': bool(spy_ok),
        'spy_close': round(spy_close, 2) if spy_close is not None else None,
        'ma200': round(ma200, 2) if ma200 is not None else None,
    }
    sh, _ = get_github_file('spy_health_history.json')
    if sh == '__ERROR__':
        print('  🔴 spy_health_history OMITIDO (fallo de lectura de GitHub) — fichero intacto')
        return None
    if not isinstance(sh, list):
        sh = []
    fechas_existentes = {e.get('fecha') for e in sh}
    if fecha in fechas_existentes:
        print(f'  spy_health_history: fecha {fecha} ya registrada — omitido duplicado')
        return sh
    sh.append(entrada)
    sh = sorted(sh, key=lambda e: e.get('fecha') or '')[-MAX_SPY_HEALTH_ENTRIES:]
    print(f'  spy_health_history persistido: {len(sh)} entradas acumuladas | score hoy: {spy_score}')
    return sh

def actualizar_putcall_history(putcall, macro, ts):
    """Persiste la serie. Solo en ejecucion de cierre: intradia el dato es parcial."""
    if not putcall:
        return None
    entrada = construir_entrada_putcall(putcall, macro, ts)
    if entrada is None:
        return None
    if not entrada.get('es_cierre'):
        print('  Put/call OMITIDO: sesion en curso (el dato de Cboe seria parcial).')
        return None
    ph, _ = get_github_file('putcall_history.json')
    if ph == '__ERROR__':
        print('  🔴 putcall_history OMITIDO (fallo de lectura de GitHub) — fichero intacto')
        return None
    if not isinstance(ph, list):
        ph = []
    fusionado = merge_putcall(ph, entrada)
    print(f'  Put/call persistido: {len(fusionado)} entradas acumuladas')
    return fusionado

# ============================================================
# RESOLUCIONES — HISTORICO PERMANENTE Y FECHA DE DETECCION (PUNTO 43, 05/08/2026)
# El P40 saco a la luz los setups resueltos y de golpe apareceieron 22, todo julio
# junto: el informe se corto por limite de max_tokens a mitad de RIESGOS. Ademas las
# resoluciones desaparecen al envejecer el setup fuera de la ventana de 30 dias, el
# mismo problema que motivo el P39 con los checkpoints.
#
# Se resuelven las dos cosas registrando CUANDO se detecto cada resolucion. El sistema
# solo sabe que hoy esta resuelta, no el dia exacto en que el precio toco el nivel, asi
# que la fecha de deteccion es lo mas honesto que se puede afirmar — y es suficiente
# para lo que importa: separar lo que acaba de pasar de lo que ya se conto hace semanas.
#
# Igual que en el P39, solo persiste la ejecucion de CIERRE: una resolucion vista con la
# sesion a medio hacer podria revertirse antes del cierre, y la primera deteccion es
# inmutable por diseño (en caso de colision se conserva la existente).
# ============================================================

DIAS_RESOLUCION_RECIENTE = 5   # ventana que el informe detalla; el resto se resume

def construir_entradas_resoluciones(resoluciones, fecha_panel):
    """Convierte las resoluciones del dia en entradas con fecha de deteccion."""
    salida = []
    for r in (resoluciones or []):
        if not r.get('ticker') or not r.get('fecha_setup'):
            continue
        salida.append({
            'fecha_setup': r.get('fecha_setup'), 'ticker': r.get('ticker'),
            'dias': r.get('dias'), 'resultado': r.get('resultado'),
            'ret_pct': r.get('ret_pct'), 'precio_entrada': r.get('precio_entrada'),
            'precio_actual': r.get('precio_actual'), 'fecha_deteccion': fecha_panel,
        })
    return salida

def merge_resoluciones(rh, entradas, max_entradas=5000):
    """Dedupe por (fecha_setup, ticker) CONSERVANDO la existente: la primera deteccion
    es la buena y no se reescribe en ejecuciones posteriores."""
    rh = [e for e in (rh or []) if isinstance(e, dict)]
    vistos = {(e.get('fecha_setup'), e.get('ticker')) for e in rh}
    for entrada in (entradas or []):
        clave = (entrada.get('fecha_setup'), entrada.get('ticker'))
        if clave in vistos:
            continue
        vistos.add(clave)
        rh.append(entrada)
    rh.sort(key=lambda e: (e.get('fecha_deteccion') or '', e.get('ticker') or ''))
    return rh[-max_entradas:]

def _dias_habiles_entre(desde, hasta):
    """Dias habiles entre dos fechas ISO. Devuelve None si no se pueden interpretar."""
    try:
        a = pd.Timestamp(desde); b = pd.Timestamp(hasta)
        return int(len(pd.bdate_range(a, b)) - 1)
    except Exception as _e:
        _traza('resoluciones/dias-habiles', _e)
        return None

def anotar_resoluciones(evaluaciones, macro, ts):
    """Devuelve las resoluciones de hoy anotadas con su fecha de PRIMERA deteccion.

    Se llama ANTES del analisis para que el prompt pueda distinguir lo reciente de lo
    antiguo. La serie leida queda en cache y actualizar_resoluciones_history la reutiliza:
    una sola peticion a GitHub por ejecucion.
    """
    resoluciones = resoluciones_por_ticker(evaluaciones)
    if not resoluciones:
        return []
    base = construir_entrada_breadth({}, macro, ts)
    fecha_panel, es_cierre = base.get('fecha'), base.get('es_cierre')
    rh, _ = get_github_file('resoluciones_history.json')
    if rh == '__ERROR__' or not isinstance(rh, list):
        rh = []
    else:
        _BREADTH_CACHE['resoluciones'] = rh
    previas = {(e.get('fecha_setup'), e.get('ticker')): e.get('fecha_deteccion') for e in rh}
    for r in resoluciones:
        clave = (r.get('fecha_setup'), r.get('ticker'))
        # Si no estaba registrada, se detecta HOY (aunque solo se persista si es cierre).
        r['fecha_deteccion'] = previas.get(clave) or fecha_panel
        dias = _dias_habiles_entre(r['fecha_deteccion'], fecha_panel)
        r['dias_desde_deteccion'] = dias
        r['reciente'] = (dias is None) or (dias <= DIAS_RESOLUCION_RECIENTE)
    _BREADTH_CACHE['resoluciones_hoy'] = (resoluciones, fecha_panel, es_cierre)
    return resoluciones

def actualizar_resoluciones_history(evaluaciones, macro, ts):
    """Persiste las resoluciones nuevas. Solo en ejecucion de cierre (ver cabecera)."""
    datos = _BREADTH_CACHE.get('resoluciones_hoy')
    if not datos:
        return None
    resoluciones, fecha_panel, es_cierre = datos
    if not es_cierre:
        print('  Resoluciones OMITIDAS: sesion en curso (dato provisional). '
              'Solo la ejecucion de cierre las persiste.')
        return None
    rh = _BREADTH_CACHE.get('resoluciones')
    if rh is None:
        rh, _ = get_github_file('resoluciones_history.json')
        if rh == '__ERROR__':
            print('  🔴 resoluciones_history OMITIDO (fallo de lectura de GitHub) — fichero intacto')
            return None
    if not isinstance(rh, list):
        rh = []
    fusionado = merge_resoluciones(rh, construir_entradas_resoluciones(resoluciones, fecha_panel))
    nuevas = len(fusionado) - len(rh)
    print(f'  Resoluciones persistidas: {nuevas} nuevas | total acumulado: {len(fusionado)}')
    return fusionado

# ============================================================
# ALERTAS — HISTORICO PERMANENTE Y DESENLACE (PUNTO 68, 13/08/2026)
# Los setups tienen checkpoints (P39) y resoluciones (P43): se sabe que paso con ellos.
# Las ALERTAS no tenian nada. Se emiten cada noche —"Supertrend bajista, salida total",
# "CMF negativo 3+ sesiones, considerar reducir al 50%"— y desaparecen del informe
# siguiente sin dejar rastro de si acertaron.
#
# El 13/08 XYL toco stop a -6,0% tras ocho sesiones en la lista de alertas blandas de CMF.
# Es el primer caso conocido de alerta blanda que se materializa, y se supo por casualidad,
# comparando dos informes a mano. Sin serie no se puede responder a la pregunta que
# importa: ¿reducir al 50% cuando el CMF lleva 3 sesiones negativo mejora el resultado, o
# solo corta ganadoras antes de tiempo? AMZN lleva siete sesiones alertada y sigue en
# +13,0% — el mismo aviso puede ser prudencia o coste de oportunidad, y hoy no hay forma
# de distinguirlo.
#
# Se registra la PRIMERA emision de cada alerta (clave (fecha_setup, ticker, tipo)) con el
# retorno de ese momento, y en cada ejecucion posterior se actualiza el retorno mas
# reciente y el desenlace si el setup se resolvio. La primera emision es inmutable: es el
# momento en que se habria actuado. Lo que evoluciona es el seguimiento.
#
# Mismo criterio que el P39: solo en cierre. Una alerta vista a media sesion con precios
# intradia no es el dato sobre el que se decide.
# ============================================================

MAX_ALERTAS_ENTRADAS = 5000

def construir_entradas_alertas(evaluaciones, fecha_panel):
    """Una entrada por alerta ACTIVA hoy. Sin dedupe por ticker: aqui interesan todas."""
    entradas = []
    for e in (evaluaciones or []):
        base = {'fecha_setup': e.get('fecha_setup'), 'ticker': e.get('ticker'),
                'fecha_alerta': fecha_panel, 'ret_pct_alerta': e.get('ret_pct')}
        if e.get('alerta_supertrend'):
            entradas.append({**base, 'tipo': 'supertrend',
                             'dias_senal': e.get('supertrend_dias'),
                             'nivel': e.get('supertrend_nivel')})
        if e.get('alerta_cmf'):
            entradas.append({**base, 'tipo': 'cmf',
                             'dias_senal': e.get('cmf_dias_negativo'),
                             'nivel': e.get('cmf_actual')})
    return entradas

def merge_alertas(ah, entradas, evaluaciones, fecha_panel, max_entradas=MAX_ALERTAS_ENTRADAS):
    """Conserva la PRIMERA emision y actualiza el seguimiento de las ya registradas.

    Campos de seguimiento sobre cada entrada existente:
      ret_pct_ultimo / fecha_ultimo -> como va el setup ahora
      desenlace -> 'stop' | 'target' | 'remitida' | None (sigue alertando)

    El dedupe es por (fecha_setup, ticker, tipo): una segunda alerta de CMF sobre el mismo
    setup no crea fila nueva, actualiza la que hay. Si el mismo ticker vuelve a dar setup
    otro dia, fecha_setup difiere y si es una alerta distinta.

    P72 (14/08/2026) — DESENLACE 'remitida'. El P68 solo cerraba una alerta con stop u
    objetivo, asi que una alerta que deja de emitirse porque la señal se recupero quedaba
    "sin desenlace" para siempre. Caso real: SPGI, seis sesiones de CMF negativo con
    +2,9%, el flujo vuelve a positivo y desaparece de la lista sin dejar rastro.

    Ese es JUSTO el contrafactual que hace falta. Sin el, la serie solo contendria alertas
    que acabaron mal y responderia que reducir al 50% siempre acierta — por construccion,
    no por evidencia. Con 'remitida' y su ret_pct_remitida se puede comparar lo que paso
    tras las que se materializaron y tras las que no.

    Solo se marca si el setup SIGUE evaluandose (ev presente): si no, no se distingue
    "la señal se recupero" de "el setup salio de la ventana de 30 dias".

    Una alerta remitida que vuelve a emitirse NO crea fila nueva (misma clave): se limpia
    el desenlace y se cuenta en reemisiones. La primera emision y su ret siguen intactos,
    porque es el momento en que se habria actuado.
    """
    ah = [e for e in (ah or []) if isinstance(e, dict)]
    estado = {(e.get('fecha_setup'), e.get('ticker')): e for e in (evaluaciones or [])}
    indice = {(e.get('fecha_setup'), e.get('ticker'), e.get('tipo')): e for e in ah}
    activas_hoy = {(e.get('fecha_setup'), e.get('ticker'), e.get('tipo')) for e in (entradas or [])}
    for entrada in (entradas or []):
        clave = (entrada.get('fecha_setup'), entrada.get('ticker'), entrada.get('tipo'))
        if clave not in indice:
            indice[clave] = entrada
            ah.append(entrada)
    # Seguimiento: se recorre TODO el historico, no solo lo alertado hoy — el desenlace
    # de una alerta de hace dos semanas es justo lo que se quiere capturar.
    nuevas_resueltas = nuevas_remitidas = 0
    for e in ah:
        ev = estado.get((e.get('fecha_setup'), e.get('ticker')))
        if not ev:
            continue  # fuera de la ventana de 30 dias: se queda como estaba
        e['ret_pct_ultimo'] = ev.get('ret_pct')
        e['fecha_ultimo'] = fecha_panel
        clave = (e.get('fecha_setup'), e.get('ticker'), e.get('tipo'))
        res = ev.get('resultado')
        # ARREGLO (16/08/2026): la condicion decia ('stop', 'objetivo') y el valor real que
        # escribe el scanner es 'target' (ver la asignacion en update_setups_history), asi
        # que un setup que alcanzaba objetivo NUNCA se marcaba resuelto: al dejar de alertar
        # acababa clasificado como 'remitida', justo el contrario de lo que paso. Lo detecto
        # tools/analisis_rendimiento.py, que arrastraba el mismo error de nombre.
        if res in ('stop', 'target') and e.get('desenlace') != res:
            # stop/target mandan sobre 'remitida': la señal se recupero pero el setup
            # acabo resolviendose igual, y eso es lo que interesa saber.
            e['desenlace'] = res
            e['fecha_desenlace'] = fecha_panel
            nuevas_resueltas += 1
        elif clave in activas_hoy:
            if e.get('desenlace') == 'remitida':
                e['desenlace'] = None          # volvio a alertar: se reabre
                e['fecha_desenlace'] = None
                e['reemisiones'] = int(e.get('reemisiones') or 0) + 1
        elif not e.get('desenlace') and e.get('fecha_alerta') != fecha_panel:
            e['desenlace'] = 'remitida'
            e['fecha_desenlace'] = fecha_panel
            e['ret_pct_remitida'] = ev.get('ret_pct')
            nuevas_remitidas += 1
    ah.sort(key=lambda e: (e.get('fecha_alerta') or '', e.get('ticker') or '', e.get('tipo') or ''))
    return ah[-max_entradas:], nuevas_resueltas, nuevas_remitidas

def actualizar_alertas_history(evaluaciones, macro, ts):
    """Persiste el historico de alertas. Solo en cierre (ver cabecera)."""
    if not evaluaciones:
        return None
    base = construir_entrada_breadth({}, macro, ts)
    fecha_panel, es_cierre = base.get('fecha'), base.get('es_cierre')
    if not es_cierre:
        print('  Alertas OMITIDAS: sesion en curso (dato provisional).')
        return None
    ah, _ = get_github_file('alertas_history.json')
    if ah == '__ERROR__':
        print('  🔴 alertas_history OMITIDO (fallo de lectura de GitHub) — fichero intacto')
        return None
    if not isinstance(ah, list):
        ah = []
    previas = len(ah)
    fusionado, resueltas, remitidas = merge_alertas(
        ah, construir_entradas_alertas(evaluaciones, fecha_panel), evaluaciones, fecha_panel)
    _abiertas = sum(1 for e in fusionado if not e.get('desenlace'))
    _rem = sum(1 for e in fusionado if e.get('desenlace') == 'remitida')
    print(f'  Alertas persistidas: {len(fusionado) - previas} nuevas | {resueltas} resueltas hoy | '
          f'{remitidas} remitidas hoy | total acumulado: {len(fusionado)} '
          f'({_abiertas} alertando, {_rem} remitidas)')
    return fusionado

# ============================================================
# CHECKPOINTS DE SETUPS — HISTORICO PERMANENTE (PUNTO 39, 04/08/2026)
# Las evaluaciones de update_setups_history se RECALCULAN cada noche desde
# setups_history + precios actuales, y solo para setups dentro de la ventana de 30
# dias. Es decir: el resultado de un setup existe mientras es reciente y despues
# DESAPARECE. data.json guarda la foto del dia, no un historico. Sin esto no hay
# forma de responder "¿los setups con SCT alto tienen menor stop rate?" — cada mes
# se vuelve a empezar con ~50 checkpoints en vez de acumular.
#
# Aqui se persisten SOLO los checkpoints (dias 5/10/20), que son los horizontes
# fijos comparables sobre los que ya se calcula la estadistica de rendimiento. Las
# evaluaciones intermedias no se guardan: son ruido diario del mismo setup.
#
# Clave de dedupe: (fecha_setup, ticker, dias). Un checkpoint concreto de un setup
# concreto es inmutable una vez alcanzado, asi que reejecutar el mismo dia no lo
# duplica ni lo altera.
# ============================================================

def construir_entradas_checkpoints(evaluaciones):
    """Extrae de las evaluaciones del dia las que son checkpoint (5/10/20 dias).

    Se guarda lo minimo para poder calibrar despues: identidad del setup, horizonte,
    resultado y las condiciones al crearse que ya viajan en setups_history. Los campos
    de alerta (CMF/Supertrend) NO se persisten: son estado del dia, no del checkpoint.
    """
    salida = []
    for e in (evaluaciones or []):
        if not e.get('checkpoint'):
            continue
        salida.append({
            'fecha_setup': e.get('fecha_setup'),
            'ticker':      e.get('ticker'),
            'dias':        e.get('dias'),
            'group':       e.get('group'),
            'tipo':        e.get('tipo'),
            'precio_entrada': e.get('precio_entrada'),
            'precio_actual':  e.get('precio_actual'),
            'stop':        e.get('stop'),
            'target':      e.get('target'),
            'ret_pct':     e.get('ret_pct'),
            'resultado':   e.get('resultado'),
        })
    return salida

def merge_checkpoints(ch, entradas, max_entradas=20000):
    """Fusiona por (fecha_setup, ticker, dias). Un checkpoint ya alcanzado no cambia,
    asi que en caso de colision se CONSERVA el existente: reejecutar no reescribe
    historia. Ordena por fecha de setup y horizonte para que el fichero sea legible."""
    ch = [e for e in (ch or []) if isinstance(e, dict)]
    vistos = {(e.get('fecha_setup'), e.get('ticker'), e.get('dias')) for e in ch}
    for entrada in (entradas or []):
        clave = (entrada.get('fecha_setup'), entrada.get('ticker'), entrada.get('dias'))
        if clave in vistos:
            continue
        vistos.add(clave)
        ch.append(entrada)
    ch.sort(key=lambda e: (e.get('fecha_setup') or '', e.get('ticker') or '', e.get('dias') or 0))
    return ch[-max_entradas:]

def actualizar_checkpoints_history(evaluaciones, macro, ts):
    """Lee el historico del repo, anade los checkpoints de hoy y lo devuelve para subir.

    ARREGLO (04/08/2026). La primera version no comprobaba es_cierre porque se diseño
    asumiendo que solo escribiria la nocturna. Fallo real ese mismo dia: una ejecucion
    manual a las 10:01 de NY, con la sesion a medio hacer, congelo 32 checkpoints con
    precios intradia. Y como el criterio de dedupe es "lo existente no se toca" (un
    checkpoint alcanzado es inmutable), esos valores NO se corregian nunca — al contrario
    que breadth_history y rates_history, que la nocturna sobreescribe por precedencia.
    La inmutabilidad solo es segura si lo que se escribe es un cierre, asi que aqui la
    ejecucion intradia no escribe NADA en vez de escribir algo provisional.
    """
    es_cierre = construir_entrada_breadth({}, macro, ts).get('es_cierre')
    if not es_cierre:
        print('  Checkpoints OMITIDOS: sesion en curso (dato provisional). '
              'Solo la ejecucion de cierre persiste checkpoints.')
        return None
    ch, _ = get_github_file('checkpoints_history.json')
    if ch == '__ERROR__':
        print('  🔴 checkpoints_history OMITIDO en esta ejecucion (fallo de lectura de GitHub) — el fichero del repo queda intacto')
        return None
    if not isinstance(ch, list):
        ch = []
    entradas = construir_entradas_checkpoints(evaluaciones)
    if not entradas:
        return None
    fusionado = merge_checkpoints(ch, entradas)
    nuevos = len(fusionado) - len(ch)
    print(f'  Checkpoints persistidos: {nuevos} nuevos | total acumulado: {len(fusionado)}')
    return fusionado

# ============================================================
# CURVA DE TIPOS — SERIE DIARIA (01/08/2026)
# Mismo patron que el P24/P29 para la amplitud: construir -> merge con precedencia
# de cierre -> actualizar leyendo del repo. Fichero PROPIO (rates_history.json) y no
# campos dentro de breadth_history.json: las dos series tienen calendarios distintos
# (tipos desde febrero, amplitud desde julio) y fusionarlas dejaria el 85% de los
# campos vacios o tiraria 100 sesiones de tipos. Separado es ademas reversible.
# Los datos salen de _calc_macro_regimen(), que ya los descarga: sin peticiones extra.
# ============================================================

def construir_entrada_rates(macro, ts):
    """Una linea de la serie de tipos: fecha de SESION + niveles, variaciones y pendientes.

    Reutiliza la misma fecha y el mismo criterio es_cierre que la entrada de amplitud
    (construir_entrada_breadth), para que ambas series sean cruzables por fecha sin
    conciliaciones raras. Devuelve None si el vector de curva no vino en esta ejecucion
    — mejor no escribir entrada que escribir una vacia.
    """
    curva = ((macro or {}).get('regimen') or {}).get('curva') or {}
    if not curva.get('niveles'):
        return None
    base = construir_entrada_breadth({}, macro, ts)
    return {
        'fecha': base.get('fecha'),
        'es_cierre': base.get('es_cierre'),
        'niveles': curva.get('niveles') or {},
        'variacion_20s_pb': curva.get('variacion_20s_pb') or {},
        'pendientes_pb': curva.get('pendientes_pb') or {},
    }

def merge_rates_entry(rh, entrada, max_entradas=1000):
    """Fusiona con la misma regla de precedencia del P29: un cierre no lo pisa una
    provisional del mismo dia; un cierre si sobreescribe a otro cierre (la nocturna
    que corrige datos rancios). Tope mas alto que el de amplitud porque la serie de
    tipos se siembra retroactivamente con meses de historico."""
    fecha = entrada.get('fecha')
    existente = next((e for e in (rh or []) if e.get('fecha') == fecha), None)
    if existente and not entrada.get('es_cierre') and existente.get('es_cierre'):
        return sorted(rh, key=lambda e: e.get('fecha') or '')[-max_entradas:]
    rh = [e for e in (rh or []) if e.get('fecha') != fecha]
    rh.append(entrada)
    return sorted(rh, key=lambda e: e.get('fecha') or '')[-max_entradas:]

def actualizar_rates_history(macro, ts):
    """Lee la serie del repo, fusiona la entrada de hoy y la devuelve para subirla.
    Mismo cinturon que actualizar_breadth_history: si la lectura de GitHub falla de
    forma persistente se OMITE la actualizacion y el fichero del repo queda intacto."""
    rh, _ = get_github_file('rates_history.json')
    if rh == '__ERROR__':
        print('  🔴 rates_history OMITIDO en esta ejecucion (fallo de lectura de GitHub) — el fichero del repo queda intacto')
        return None
    if not isinstance(rh, list):
        rh = []
    entrada = construir_entrada_rates(macro, ts)
    if entrada is None:
        return None
    return merge_rates_entry(rh, entrada)

# ============================================================
# PUNTO 17 — CORRELACION ENTRE CANDIDATOS (12/07/2026)
# Cache en memoria de las series de cierre del universo, rellenada por analyze_universe
# y consumida por generate_analysis para medir la dependencia entre los <=5 setups
# finales. NO se serializa a data.json (solo memoria de proceso): en GitHub Actions
# y en Colab el pipeline completo corre en el mismo proceso/kernel, asi que el cache
# siempre esta poblado cuando generate_analysis lo necesita. Si por cualquier motivo
# no lo estuviera (ejecucion parcial), la funcion devuelve None y el informe sale
# sin el bloque — degradacion silenciosa, nunca un fallo.
# ============================================================
_CLOSES_CACHE = {}

# P59 (11/08/2026) — interruptor del diagnostico de betas. RETIRADO el 11/08 tras
# confirmar que el calculo es correcto (beta = corr * sd_tk/sd_bench cuadra al decimal).
# Se deja el interruptor por si hay que volver a mirar; poner a True en AMBOS ficheros.
DIAG_BETAS = False

# P60 (11/08/2026) — |correlacion| minima con el SPY para considerar la beta una medida
# real de exposicion al mercado. Con 60 sesiones, por debajo de 0.30 el coeficiente no
# alcanza significacion y la beta resultante es ruido escalado por el cociente de
# volatilidades (los candidatos se mueven 2.5-3.5x mas que el indice).
MIN_CORR_BETA_SIGNIFICATIVA = 0.30

# P62 (12/08/2026) — tope de tokens de SALIDA del informe. El truncado recurrente NO venia
# del tamaño del prompt (eso lo atacaron el P43 y el P54, que recortan la ENTRADA): venia de
# este techo. El 11/08 a las 14:31 el informe se corto tras 17.118 caracteres (~8.000 tokens
# en español, ~2,1 car/token) y se llevo entera la seccion RIESGOS DEL ESCENARIO, que el P36
# cazo. Con 12.000 queda ~50% de margen sobre lo que se genera hoy. Si el informe crece hasta
# rozar tambien este tope, la solucion siguiente es acotar longitud POR SECCION en el prompt,
# no seguir subiendo el techo.
MAX_TOKENS_INFORME = 12000

# P63 (12/08/2026) — miembros minimos para que un tema entre en el ranking y su fortaleza
# sectorial se considere medida. Por debajo, el score es el de sus pocos miembros (con n=1,
# literalmente el de ese ticker) y el % sobre MM200 solo puede ser 0 o 100. Afecta sobre todo
# a los grupos de PERSONAL_WATCHLIST, que son tematicos y deliberadamente pequeños.
# BAJADO A 2 el 13/08/2026: con 3 quedaban fuera del ranking 4 grupos de watchlist (Minerales,
# Biotech, Robotica AI con n=2, y Hardware con n=1) justo mientras la rotacion tematica se
# aceleraba, y perder de vista tres temas activos costaba mas que el sesgo residual de un n=2.
# Con 2 solo queda fuera el caso indefendible: el grupo de UN solo valor, donde el score del
# tema ES el del ticker y el % sobre MM200 es un 0/100 sin termino medio.
MIN_MIEMBROS_TEMA = 2

# P65 (13/08/2026) — % minimo de tickers con dato en la ultima sesion para que data.json
# pueda sustituir al que ya esta publicado. El P22 avisa desde 90; aqui el umbral es mas
# bajo a proposito (bloquear es mas caro que avisar) y solo frena vuelcos claramente
# incompletos como el del 12/08, que se quedo en 85,5%.
MIN_FRESCOS_PARA_PUBLICAR = 88.0
# PUNTO 25 — ultimo breadth calculado en esta ejecucion (calc_market_breadth corre antes
# que update_setups_history en ambos flujos), para etiquetar cada setup con la dispersion
# del dia en que nacio sin cambiar firmas de funciones.
_BREADTH_CACHE = {}

def calc_correlacion_candidatos(tickers, ventana=60, umbral_aviso=0.70, min_sesiones=30, min_sesiones_beta=60):
    """Correlacion de retornos diarios entre los candidatos finales + beta vs benchmark.

    - ventana: sesiones usadas (60 ~ 3 meses; mismo horizonte que el Volume Profile).
    - umbral_aviso: pares con correlacion >= umbral se marcan como concentracion.
    - min_sesiones: minimo de retornos comunes para que la CORRELACION sea fiable;
      por debajo se devuelve None (mejor sin bloque que con una correlacion de juguete).
    - min_sesiones_beta: minimo de retornos comunes para que la BETA sea fiable.
      P58 (09/08/2026): separado de min_sesiones porque la beta es implausible con series
      cortas (con 30 sesiones la covarianza es inestable; umbral minimo recomendado: 60
      sesiones ~ 3 meses). Con menos de min_sesiones_beta la beta se omite (None).
    Devuelve dict con matriz, betas, pares altos y n de sesiones, o None si no aplica.
    """
    series = {tk: _CLOSES_CACHE[tk] for tk in tickers if tk in _CLOSES_CACHE}
    if len(series) < 2:
        return None
    completo = pd.DataFrame(series).dropna()
    df = completo.tail(ventana)
    rets = df.pct_change().dropna()
    if len(rets) < min_sesiones:
        return None
    corr = rets.corr()
    betas = {}
    corrs_bench = {}   # P60: correlacion de cada candidato con el benchmark
    bench = _CLOSES_CACHE.get('__BENCH__')
    if bench is not None:
        # P58 (ARREGLO 11/08/2026): la beta usa VENTANA PROPIA. Antes reutilizaba `rets`,
        # que sale de tail(ventana=60) y por tanto tiene 59 retornos: la condicion
        # len(par) >= min_sesiones_beta (60) no se cumplia NUNCA y todas las betas
        # salian None en produccion. Se toma una sesion mas de la necesaria para que
        # min_sesiones_beta retornos sean alcanzables cuando el historico da para ello.
        rets_beta = completo.tail(max(ventana, min_sesiones_beta + 1)).pct_change().dropna()
        bench_ret = bench.pct_change().reindex(rets_beta.index).dropna()
        _diag = []
        for tk in rets_beta.columns:
            par = pd.concat([rets_beta[tk], bench_ret], axis=1).dropna()
            if len(par) >= min_sesiones_beta:  # P58: umbral propio para beta
                cov = np.cov(par.iloc[:, 0], par.iloc[:, 1])
                betas[tk] = round(float(cov[0, 1] / cov[1, 1]), 2) if cov[1, 1] > 0 else None
                # P60 (11/08/2026): se guarda tambien la CORRELACION con el bench. Sin ella la
                # beta es indistinguible entre "poca exposicion al mercado" y "ninguna relacion
                # con el mercado": beta = corr * (sd_tk / sd_bench), y con sd_tk 2.5-3x la del
                # SPY una correlacion de ruido (0.05-0.18) produce betas de 0.5 de magnitud.
                try:
                    corrs_bench[tk] = round(float(par.iloc[:, 0].corr(par.iloc[:, 1])), 3)
                except Exception as _e:
                    _traza('beta/corr-bench', _e); corrs_bench[tk] = None
            else:
                betas[tk] = None  # serie insuficiente — mejor None que valor espurio
                corrs_bench[tk] = None
            if DIAG_BETAS:
                # P59 (11/08/2026) — DIAGNOSTICO TEMPORAL, no toca ningun calculo.
                # Las betas del 11/08 salieron implausibles (CPRT -0.48, UBER +0.51). Para
                # separar "dato real" de "series mal emparejadas" hace falta ver, por ticker,
                # cuantas sesiones entran y que correlacion tienen: si la CORRELACION tambien
                # es negativa, la beta negativa es lo que dicen los datos; si la correlacion
                # es ~0 con beta grande, el emparejamiento esta corrido.
                try:
                    _corr_b = round(float(par.iloc[:, 0].corr(par.iloc[:, 1])), 3) if len(par) > 2 else None
                    _sd_tk = round(float(par.iloc[:, 0].std()) * 100, 2) if len(par) > 2 else None
                    _sd_bn = round(float(par.iloc[:, 1].std()) * 100, 2) if len(par) > 2 else None
                except Exception as _e:
                    _traza('beta/diagnostico', _e); _corr_b = _sd_tk = _sd_bn = None
                _diag.append(f'{tk}: n={len(par)} beta={betas.get(tk)} corr={_corr_b} '
                             f'sd_tk={_sd_tk}% sd_bench={_sd_bn}%')
        if DIAG_BETAS:
            def _rango(s):
                try:
                    return f'{s.index[0].date()}..{s.index[-1].date()}'
                except Exception:
                    return 'n/d'
            print(f'  [DIAG betas] rets_beta {len(rets_beta)} filas {_rango(rets_beta)} | '
                  f'bench_ret {len(bench_ret)} filas {_rango(bench_ret)} | '
                  f'indices identicos: {rets_beta.index.equals(bench_ret.index)}')
            for _d in _diag:
                print(f'  [DIAG betas]   {_d}')
    pares_altos = []
    tks = list(corr.columns)
    for i in range(len(tks)):
        for j in range(i + 1, len(tks)):
            c = float(corr.iloc[i, j])
            if c >= umbral_aviso:
                pares_altos.append((tks[i], tks[j], round(c, 2)))
    return {'corr': corr.round(2), 'betas': betas, 'corrs_bench': corrs_bench,
            'pares_altos': pares_altos, 'n_sesiones': int(len(rets))}

def formato_correlacion_summary(cc, valid):
    """PUNTO 17 — bloque de texto para el summary del prompt (estilo AMPLITUD DE MERCADO)."""
    if not cc:
        return ''
    tks = list(cc['corr'].columns)
    grupo_por_ticker = {v['ticker']: v.get('group', '?') for v in valid}
    s = '\nCORRELACION ENTRE CANDIDATOS (retornos diarios, ultimas %d sesiones):\n' % cc['n_sesiones']
    for i in range(len(tks)):
        for j in range(i + 1, len(tks)):
            s += f'- {tks[i]} vs {tks[j]}: {cc["corr"].iloc[i, j]}\n'
    if cc['betas']:
        betas_validas = {tk: b for tk, b in cc['betas'].items() if b is not None}
        corrs_b = cc.get('corrs_bench') or {}
        if betas_validas:
            # P60 (11/08/2026): la beta va SIEMPRE acompañada de su correlacion con el SPY y,
            # por debajo de MIN_CORR_BETA_SIGNIFICATIVA, marcada como no significativa. El
            # 11/08 el informe leyo betas de -0.48/-0.43 con correlaciones de -0.18/-0.12 como
            # "cartera defensiva que amortigua la exposicion al SPY de cara al dato de manana":
            # una beta baja por DESCORRELACION no protege de un gap: el valor no sigue al
            # indice ni a la baja ni al alza. Se resuelve la semantica en Python en vez de
            # esperar que el modelo la derive (misma leccion que el P33 y el ratio de dispersion).
            partes = []
            no_sig = []
            for tk, b in betas_validas.items():
                c = corrs_b.get(tk)
                if c is None:
                    partes.append(f'{tk}:{b}')
                elif abs(c) < MIN_CORR_BETA_SIGNIFICATIVA:
                    partes.append(f'{tk}:{b} (corr {c} — NO SIGNIFICATIVA)')
                    no_sig.append(tk)
                else:
                    partes.append(f'{tk}:{b} (corr {c})')
            s += 'Beta vs SPY: ' + ' | '.join(partes) + '\n'
            # P81 (02/09/2026): la REGLA DE REDACCION de las betas es PERMANENTE, no solo
            # cuando alguna es no significativa. El 01/09 las dos betas superaban
            # MIN_CORR_BETA_SIGNIFICATIVA, no se emitio el AVISO BETAS y el informe publico
            # "$CF -1.08, comportamiento defensivo/contraciclico" sin citar ninguna correlacion:
            # el P60/P66 solo cubria el caso de betas de ruido y dejaba sin instruccion el caso
            # contrario. Se saca la regla de la condicional para que rija siempre.
            s += ('REGLA BETAS: al mencionar una beta en el informe, cita SIEMPRE su correlacion '
                  'con el SPY entre parentesis, tal como figura arriba. NO agrupes varias betas '
                  'bajo un adjetivo comun de signo ("las betas negativas de X e Y"): cada una '
                  'tiene el signo que figura arriba. NO califiques un valor ni la cartera de '
                  'defensivo, contraciclico, protegido ni de baja beta apoyandote solo en la '
                  'beta: la beta mide sensibilidad, no proteccion.\n')
            if no_sig:
                # P66 (13/08/2026): el aviso NOMBRA los tickers afectados en vez de dar solo un
                # recuento. El 12/08 el informe escribio "las betas negativas de $FOXA (-0.41) y
                # $CBRE (0.28)" — 0.28 es POSITIVA: al reconstruir la lista por su cuenta agrupo
                # las no significativas bajo un adjetivo comun. Dandole los nombres hechos y
                # prohibiendo el agrupamiento por signo, no hay nada que reconstruir.
                s += (f'AVISO BETAS: las betas de {", ".join(no_sig)} ({len(no_sig)} de '
                      f'{len(betas_validas)}) tienen |correlacion| con el SPY por debajo de '
                      f'{MIN_CORR_BETA_SIGNIFICATIVA} sobre {cc["n_sesiones"]} sesiones: NO son '
                      'medidas fiables de exposicion al mercado. Una beta baja por DESCORRELACION '
                      'no amortigua el riesgo de un evento macro ni de un gap del indice — el valor '
                      'simplemente no sigue al SPY, y puede caer mas que el. NO describir el '
                      'conjunto como defensivo, de baja beta ni protegido apoyandose en estas '
                      'cifras, y NO agruparlas bajo un adjetivo comun de signo ("las betas '
                      'negativas de X e Y"): cada una tiene el signo que figura arriba y hay '
                      'positivas y negativas entre ellas.\n')
        elif cc['betas']:
            s += 'Beta vs SPY: no disponible (serie < 60 sesiones)\n'
    if cc['pares_altos']:
        pares_txt = ', '.join(f'{a}-{b} ({c})' for a, b, c in cc['pares_altos'])
        s += f'AVISO CONCENTRACION: pares con correlacion >=0.70: {pares_txt}\n'
    grupos = {}
    for tk in tks:
        grupos.setdefault(grupo_por_ticker.get(tk, '?'), []).append(tk)
    repetidos = {g: t for g, t in grupos.items() if len(t) > 1}
    if repetidos:
        s += 'CONCENTRACION SECTORIAL: ' + ' | '.join(f'{g}: {", ".join(t)}' for g, t in repetidos.items()) + '\n'
    return s


def check_frescura_panel(close_df):
    """PUNTO 22 — chequeo de frescura del panel de datos (15/07/2026).

    Si yfinance devolviera datos rancios, el scanner calcularia sobre el pasado en
    silencio y el informe seria pura arqueologia. Este chequeo compara la ultima fecha
    del panel con el ultimo dia habil estadounidense esperado y mide que fraccion de
    tickers tiene dato en esa ultima sesion (rezagados de descarga parcial).
    SOLO AVISA en el log — nunca aborta: un retraso de 1 dia habil puede ser un festivo
    de mercado en EEUU (el calendario habil de pandas no conoce festivos NYSE) o una
    ejecucion previa a la apertura, asi que el nivel de alarma es gradual.
    Devuelve dict con los datos del chequeo (o None si el panel esta vacio).
    """
    if close_df is None or len(close_df) == 0:
        print('  AVISO FRESCURA: panel de cierres vacio')
        return None
    ultima = pd.Timestamp(close_df.index[-1]).normalize()
    hoy_ny = pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize()
    ult_habil = hoy_ny if hoy_ny.dayofweek < 5 else hoy_ny - pd.offsets.BDay(1)
    # dias habiles de retraso entre la ultima sesion del panel y el ultimo dia habil
    retraso = max(0, len(pd.bdate_range(ultima, ult_habil)) - 1)
    pct_frescos = round(float(close_df.iloc[-1].notna().mean() * 100), 1)
    msg = f'  Frescura panel: ultima sesion {ultima.date()} (retraso {retraso} dia(s) habil(es)) | {pct_frescos}% tickers con dato en la ultima sesion'
    if retraso >= 2:
        msg += ' | AVISO FUERTE: posible dato rancio de la fuente (o festivo prolongado)'
    elif retraso == 1:
        msg += ' | aviso: puede ser festivo de mercado en EEUU o ejecucion previa a la apertura'
    if pct_frescos < 90.0:
        msg += ' | AVISO: mas del 10% de tickers sin dato en la ultima sesion (descarga parcial)'
    print(msg)
    _BREADTH_CACHE['ultima_sesion'] = str(ultima.date())  # PUNTO 24 — fecha de sesion para la serie de amplitud
    resultado = {'ultima_sesion': str(ultima.date()), 'retraso_habiles': int(retraso),
                 'pct_tickers_frescos': pct_frescos}
    # P65 — se guarda el chequeo del PANEL AMPLIO (el ultimo que se ejecuta, sobre 500+
    # valores) para que la subida pueda decidir si data.json es publicable. La watchlist
    # son 36 tickers y su porcentaje es mucho mas volatil, asi que no debe mandar.
    if close_df.shape[1] >= 100:
        _BREADTH_CACHE['frescura_panel'] = resultado
    return resultado


def check_data_health(close_df, high_df, low_df, vol_df, max_var_diaria=0.50, min_vol_dias=3):
    """PUNTO 35 (01/08/2026) — Validación de coherencia OHLCV en el panel descargado.

    Detecta silenciosamente datos corruptos que pasan el filtro de frescura pero
    contaminan todos los indicadores calculados sobre esos precios:
      - Variación diaria > max_var_diaria (split no ajustado, precio aberrante)
      - Precio de cierre fuera del rango High-Low del mismo día
      - Volumen = 0 en más de min_vol_dias de los últimos 5 días
    Solo avisa en el log — nunca aborta. Devuelve el conjunto de tickers sospechosos.
    """
    sospechosos = set()
    if close_df is None or close_df.empty:
        return sospechosos
    # 1. Variación diaria extrema
    rets = close_df.pct_change().iloc[-10:]
    for tk in close_df.columns:
        if (rets[tk].abs() > max_var_diaria).any():
            sospechosos.add(tk)
    # 2. Cierre fuera del rango High-Low
    if high_df is not None and low_df is not None:
        for tk in close_df.columns:
            if tk not in high_df.columns or tk not in low_df.columns:
                continue
            c = close_df[tk].iloc[-5:]; h = high_df[tk].iloc[-5:]; l = low_df[tk].iloc[-5:]
            mask = c.notna() & h.notna() & l.notna()
            if mask.any():
                fuera = ((c[mask] > h[mask] * 1.01) | (c[mask] < l[mask] * 0.99))
                if fuera.any():
                    sospechosos.add(tk)
    # 3. Volumen cero persistente
    if vol_df is not None and not vol_df.empty:
        vol_rec = vol_df.iloc[-5:]
        for tk in vol_df.columns:
            if tk in vol_rec and (vol_rec[tk].fillna(0) == 0).sum() >= min_vol_dias:
                sospechosos.add(tk)
    if sospechosos:
        print(f'  AVISO data_health: {len(sospechosos)} ticker(s) con datos sospechosos: '
              f'{", ".join(sorted(sospechosos)[:10])}{"..." if len(sospechosos)>10 else ""}')
    return sospechosos

def analyze_universe(grps, bench, close_df, vol_df, high_df=None, low_df=None, spy_healthy=True, spy_score=None):
    check_frescura_panel(close_df)  # PUNTO 22 — aviso en log si el panel llega rancio o incompleto
    _sospechosos = check_data_health(close_df, high_df, low_df, vol_df)  # PUNTO 35 — coherencia OHLCV
    _CLOSES_CACHE['__BENCH__'] = bench  # PUNTO 17 — benchmark para betas
    res=[]
    for gn, tickers in grps.items():
        for tk in tickers:
            if tk not in close_df.columns: continue
            p=close_df[tk]; v=vol_df[tk] if tk in vol_df.columns else pd.Series(dtype=float)
            try:
                p_clean = p.dropna()
                v_clean = v.dropna()
                # Validar datos: precio debe ser positivo y no nulo
                if len(p_clean) < 20: continue
                precio_actual = float(p_clean.iloc[-1])
                if precio_actual <= 0 or pd.isna(precio_actual): continue
                # Filtro de liquidez: precio x volumen medio 20d > $2M
                vol_medio = float(v_clean.iloc[-20:].mean()) if len(v_clean) >= 20 else 0
                if pd.isna(vol_medio) or vol_medio <= 0: continue
                if precio_actual * vol_medio < 2_000_000: continue
            except Exception as _e:
                _traza('universo/ticker', _e)
                continue
            h=high_df[tk] if high_df is not None and tk in high_df.columns else p
            l=low_df[tk] if low_df is not None and tk in low_df.columns else p
            r4=rs_score(p,bench,20); r13=rs_score(p,bench,65)
            vz=volume_zscore(v) if not v.empty else None
            rvol=calc_rvol(v) if not v.empty else None  # PUNTO 9
            # NUEVO (05/07) — volumen medio absoluto de 30 sesiones (acciones/dia), base del
            # filtro de volumen minimo 500K en valid[]. RVOL es relativo (hoy / media 30d):
            # un valor de base baja puede dar RVOL alto en una sesion excepcional para sus
            # estandares siendo de los menos activos del universo (caso real $ROP, ~1.2M
            # acc/dia, RVOL 2.63x). Se calcula aqui sobre v_clean ya disponible, sin coste
            # de datos adicional.
            vol_media_30d = round(float(v_clean.iloc[-30:].mean())) if len(v_clean) >= 30 else None
            bi=detect_breakout(p,v) if not v.empty else {'breakout':False,'days_ago':None,'breakout_level':None}
            mah=ma_health(p); ps=price_stats(p,h,v)
            _CLOSES_CACHE[tk] = p_clean  # PUNTO 17 — serie de cierres para correlacion de candidatos
            # OPTIMIZACION — serie ATR calculada una sola vez, reutilizada por calc_atr (valor
            # actual) y calc_adx (serie completa para DI+/DI-) en vez de duplicar el calculo
            atr_series_pre = calc_atr_series(h,l,p) if not h.empty else None
            atr_pre = round(float(atr_series_pre.iloc[-1]),4) if atr_series_pre is not None else None
            er=calc_entry_range(ps,bi,mah,atr_val=atr_pre)
            # PUNTO 11 — Volume Profile aproximado (POC/VAH/VAL), 60 sesiones
            volp = calc_volume_profile(h, l, v, window=60) if not h.empty and not v.empty else None
            # NUEVO (05/07) — VAL como soporte: ademas de coincidir con la entrada (±2%),
            # el VAL debe estar suficientemente POR ENCIMA del stop para citarse como
            # argumento de calidad. Caso real $FIS: VAL a $1.01 del stop (60% del camino
            # entrada->stop ya recorrido) citado como "soporte con respaldo de volumen"
            # cuando en realidad estaba dentro de la zona de riesgo. Regla: el VAL debe
            # quedar al menos a VAL_MARGEN_MINIMO del rango entry_lo/stop por encima del stop:
            #   VAL >= stop + (entry_lo - stop) * VAL_MARGEN_MINIMO
            # Si no se cumple, confirma_pullback queda en False (no se cita como soporte)
            # y val_valido_como_soporte=False queda registrado para trazabilidad.
            # OJO — inconsistencia detectada en el documento PENDIENTE (verificada 05/07):
            # con la fraccion 0.50 propuesta, el propio caso $FIS que motivo la regla PASA
            # el filtro (su VAL esta al 60.5% del rango sobre el stop; el "60% del camino
            # recorrido" del documento era en realidad el margen restante, no el recorrido).
            # Se implementa 0.50 tal y como esta documentado, parametrizado para poder
            # subirlo (0.75 invalidaria el caso $FIS) cuando Carlos confirme el criterio.
            VAL_MARGEN_MINIMO = 0.50
            if volp and er.get('entry_lo'):
                val = volp.get('val'); entry_lo = er['entry_lo']; stop_er = er.get('stop')
                entry_hi = er.get('entry_hi')
                coincide = abs(entry_lo / val - 1) <= 0.02 if val else False
                if val and stop_er is not None and entry_lo > stop_er:
                    val_sobre_stop = val >= stop_er + (entry_lo - stop_er) * VAL_MARGEN_MINIMO
                else:
                    val_sobre_stop = False  # sin stop valido no se puede validar el margen
                volp['coincide_val'] = coincide
                volp['val_valido_como_soporte'] = val_sobre_stop
                volp['confirma_pullback'] = coincide and val_sobre_stop
                # P67 (13/08/2026) — POSICION REAL del VAL respecto al rango de entrada.
                # `coincide` solo compara VAL con entry_lo con una tolerancia del 2%, asi que
                # un VAL POR DEBAJO de todo el rango la pasa igual: el 13/08 $COST salio con
                # VAL 923.65 y entrada 939.87-958.85 (VAL un 1.76% por debajo de entry_lo) y
                # el prompt afirmaba "la entrada coincide con VAL". El modelo lo detecto y se
                # corrigio a media frase, dejando el razonamiento visible en el informe. Se
                # calcula aqui la relacion exacta para que el texto la diga en vez de asumirla.
                if val and entry_lo:
                    if entry_hi and entry_lo <= val <= entry_hi:
                        volp['val_posicion'] = 'dentro'
                    elif val < entry_lo:
                        volp['val_posicion'] = 'debajo'
                    else:
                        volp['val_posicion'] = 'encima'
                    volp['val_dist_entry_lo_pct'] = round((val / entry_lo - 1) * 100, 2)
            sc=composite_score(r4,r13,vz,bi['breakout'],mah,spy_healthy,spy_score=spy_score)
            # Score de Confirmacion Tecnica (SCT)
            atr_val = atr_pre  # ya calculado antes
            rsi_val = calc_rsi(p)
            macd_val, macd_sig, macd_hist = calc_macd(p)
            adx_val = calc_adx(h, l, p, atr_series=atr_series_pre) if not h.empty else None
            pct_b, squeeze, bw = calc_bollinger(p)
            cmf_val, cmf_dias_neg = calc_cmf(h, l, p, v) if not h.empty and not v.empty else (None, None)
            obv_val = calc_obv(p, v) if not v.empty else None
            sct_val = calc_sct(adx_val, rsi_val, macd_hist, pct_b, squeeze, cmf_val, obv_val)
            # PUNTO 1 — RIESGO calculado en Python con la formula exacta (BAJO/MEDIO/ALTO)
            # Devuelve tambien el motivo exacto que dispara la clasificacion, para que Claude
            # lo use literalmente en el informe en vez de inferir una causa no verificada.
            riesgo_val, riesgo_motivo = calc_riesgo(sct_val, rsi_val, cmf_val, adx_val)
            # PUNTO 18 — pullback con RSI en sobrecompra (>=70): la clasificacion de pullback
            # solo mira la distancia a la media (±4% de MM50/MM200), no exige que el retroceso
            # haya aliviado la sobrecompra. Un "pullback" con RSI 70+ es un retroceso minimo
            # dentro de una subida casi vertical — setup de menor calidad (caso real detectado
            # por revision externa: UHS con RSI ~74 presentado como pullback, 11/07/2026).
            # De momento SOLO se marca (sin degradar la clasificacion ni tocar el historico);
            # la decision de endurecer (degradar a 'En vigilancia') queda pendiente de ver
            # cuantos casos reales aparecen marcados en produccion.
            if er.get('tipo','').startswith('Pullback') and rsi_val is not None and rsi_val >= 70:
                er['rsi_sobrecompra'] = True
            # NUEVO (27/06) — Supertrend como stop dinamico complementario al Punto 10
            supertrend_val = calc_supertrend(h, l, p) if not h.empty else None
            res.append({'ticker':tk,'group':gn,'rs_4w':r4,'rs_13w':r13,'vol_z':vz,
                'breakout':bi['breakout'],'days_ago':bi['days_ago'],'breakout_level':bi.get('breakout_level'),
                'ma20':mah.get('ma20',False),'ma50':mah.get('ma50',False),'ma200':mah.get('ma200',False),
                'pct_ma50':mah.get('pct_ma50'),'pct_ma200':mah.get('pct_ma200'),
                'ma50_val':mah.get('ma50_val'),'ma200_val':mah.get('ma200_val'),
                'price':ps.get('price'),'high52':ps.get('high52'),'low52':ps.get('low52'),
                'pct_from_high':ps.get('pct_from_high'),'ret_1w':ps.get('ret_1w'),
                'ret_1m':ps.get('ret_1m'),'ret_3m':ps.get('ret_3m'),'entry_range':er,'score':sc,
                'sct':sct_val,'rsi':rsi_val,'adx':adx_val,'macd_hist':macd_hist,
                'cmf':cmf_val,'cmf_dias_negativo':cmf_dias_neg,'atr':atr_val,'squeeze':squeeze,'pct_b':pct_b,
                'riesgo':riesgo_val,'riesgo_motivo':riesgo_motivo,'rvol':rvol,'vol_media_30d':vol_media_30d,'vol_profile':volp,
                'techo_metodo':er.get('techo_metodo'),'supertrend':supertrend_val})
    return res

def calc_groups(res, is_sp=False):
    gs={}
    for r in res: gs.setdefault(r['group'],[]).append(r)
    out=[]
    for gn,mb in gs.items():
        scores=[m['score'] for m in mb if m['score'] is not None]
        r4s=[m['rs_4w'] for m in mb if m['rs_4w'] is not None]
        mb_s=sorted(mb,key=lambda x:x['score'] or 0,reverse=True)
        # PUNTO 31 (01/08/2026) — fortaleza sectorial real: % de miembros sobre su MM200.
        # Sustituye la media de scores de los miembros (que creaba un bucle de retroalimentacion:
        # un lider excepcional en un sector mediocre quedaba penalizado por sus companeros).
        # ma200 ya es un booleano (True=cotiza sobre la MM200) calculado en analyze_universe.
        n_sobre_mm200 = sum(1 for m in mb if m.get('ma200'))
        pct_sobre_mm200 = round(100.0 * n_sobre_mm200 / len(mb), 1) if mb else 0.0
        # P63 (12/08/2026) — un grupo con MUY POCOS miembros no produce magnitudes comparables
        # con las de un sector poblado, ni en score ni en fortaleza estructural:
        #   - score = media de los scores de sus miembros: con n=1 es el score de ESE ticker,
        #     sin promediar nada. El 11-12/08 'Cuantica Seguridad' (solo ARQQ) lidero el top de
        #     temas con 52,1 y luego 60,0, por encima de sectores con decenas de valores.
        #   - pct_sobre_mm200 con n=1 solo puede valer 0 o 100, nunca un valor intermedio.
        # Se marca aqui y cada consumidor decide: el ranking individual sustituye el componente
        # sectorial por un valor neutro (ver construir_prompt) y el top de temas los separa.
        comparable = len(mb) >= MIN_MIEMBROS_TEMA
        out.append({'group':gn,'score':round(np.mean(scores),1) if scores else 0,
                    'pct_sobre_mm200': pct_sobre_mm200,
                    'tema_comparable': comparable,
                    'rs_mean':round(np.mean(r4s),1) if r4s else 0,
                    'breakouts':sum(1 for m in mb if m['breakout']),
                    'n':len(mb),'is_sp':is_sp,'top3':mb_s[:3],'members':mb_s})
    return sorted(out,key=lambda x:x['score'],reverse=True)


def rescatar_analisis_anterior(analisis, avisos):
    """PUNTO 55 (08/08/2026) — conserva el ultimo informe bueno si el de hoy viene vacio.

    Caso real del 08/08: se agoto el saldo de la API de Anthropic, generate_analysis
    devolvio cadena vacia y el scanner subio data.json igualmente, SOBRESCRIBIENDO el
    informe correcto de una hora antes. La web se quedo sin analisis, y el resto de los
    datos —amplitud, setups, checkpoints— si eran validos y convenia actualizarlos.

    Un informe de hace unas horas es infinitamente mejor que ninguno: se rescata el
    anterior de data.json y se marca claramente como no actual, para que nadie lo lea
    creyendo que corresponde a la sesion de hoy.

    Solo actua cuando el informe viene VACIO o es demasiado corto. Un informe generado
    pero incompleto (truncado, o al que le falta una seccion) NO se sustituye: es peor
    perder el analisis de hoy que tenerlo a medias.

    Devuelve (analisis, rescatado, no_publicar_data). El tercero es True SOLO cuando el
    informe viene vacio Y ademas no se ha podido leer el data.json anterior: en ese caso
    el llamador debe sacar data.json de la subida.
    """
    if analisis and len(analisis.strip()) >= 500:
        return analisis, False, False
    previo, _ = get_github_file('data.json')
    if previo == '__ERROR__' or not isinstance(previo, dict):
        # P78 (20/08/2026): TERCER valor de retorno. Antes se devolvia (analisis, False) y
        # main subia data.json igualmente CON EL INFORME VACIO, encima del bueno — pese a
        # que get_github_file ya imprimia "el llamador debe abortar para no machacar datos"
        # y nadie abortaba. Paso el 20/08: fallo la lectura, se subio vacio y la web perdio
        # el informe. Si no se pudo LEER el anterior no se sabe que hay al otro lado, asi
        # que no se pisa: mismo criterio del P65 con la frescura.
        print('  🔴 informe vacio y NO se pudo leer el anterior — data.json NO se subira '
              '(no se pisa lo que no se ha podido comprobar)')
        return analisis, False, True
    anterior = (previo.get('analisis') or '').strip()
    if len(anterior) < 500:
        # Aqui SI se sube: se ha comprobado que el anterior tampoco valia, asi que no hay
        # nada que destruir y el resto de data.json (amplitud, setups, seguimiento) si es
        # bueno y conviene actualizarlo.
        print('  🔴 informe vacio y el anterior tampoco servia — se sube vacio')
        return analisis, False, False
    sello = previo.get('timestamp', 'ejecucion anterior')
    print(f'  ⚠️  Informe vacio: se CONSERVA el analisis anterior ({sello}) en vez de borrarlo')
    aviso = (f'> **AVISO DEL SISTEMA — informe no actualizado.** La generacion del analisis '
             f'fallo en esta ejecucion, asi que lo que sigue es el informe de {sello}. '
             f'Los datos, setups y seguimiento de la web SI estan actualizados; solo este '
             f'texto es anterior.\n\n')
    return aviso + anterior, True, False

def validar_informe(analisis, universo_tickers):
    """PUNTO 36 (01/08/2026) — Validación post-generación del informe.

    El informe puede omitir secciones obligatorias o mencionar tickers que no están
    inventar tickers sin que nadie lo detecte. Esta función comprueba:
      - Presencia de las secciones obligatorias del prompt
      - Que los tickers citados existan en el universo analizado (no inventados)
    Devuelve (valido: bool, avisos: list[str]) — nunca aborta.
    """
    avisos = []
    if not analisis or len(analisis) < 200:
        return False, ['Informe vacío o demasiado corto']
    secciones = ['RIESGOS DEL ESCENARIO', 'CONCLUS', 'MERCADO']
    for s in secciones:
        if s not in analisis.upper():
            avisos.append(f'Sección obligatoria ausente: {s}')
    # Tickers mencionados como candidatos (con $TICKER o sin $)
    import re
    # P82 (03/09/2026) — el patron cortaba en el guion: de "$BF-B" extraia "BF", que no
    # existe en el universo, y el aviso del P36 lo denunciaba como ticker inventado cada
    # vez que el informe citaba una clase B (BF-B, BRK-B). Falso positivo puro. Se admite
    # el sufijo de clase para que el ticker se compare entero.
    tickers_informe = set(re.findall(r'\$([A-Z]{1,5}(?:-[A-Z]{1,2})?)\b', analisis))
    # REDEFINIDO (02/08/2026): el criterio original comparaba contra los 4-5 FINALISTAS con
    # tolerancia de 2, y habria disparado cada noche: las secciones EXTENDIDOS y SEGUIMIENTO
    # DE POSICIONES ABIERTAS nombran por diseño decenas de tickers que no son finalistas
    # (vigilancias, posiciones de sesiones anteriores). Eso no es un error del informe.
    # Lo que SI es un error es que el modelo invente un ticker que no existe en el universo
    # analizado. Ese es el criterio ahora, y por eso la tolerancia es CERO.
    universo = set()
    for x in (universo_tickers or []):
        universo.add(x['ticker'] if isinstance(x, dict) else x)
    if universo:
        inventados = tickers_informe - universo
        if inventados:
            avisos.append(f'Tickers citados que NO existen en el universo analizado: {sorted(inventados)[:5]}')
    else:
        avisos.append('Universo vacio: comprobacion de tickers OMITIDA (no concluyente)')
    valido = len(avisos) == 0
    return valido, avisos

UMBRAL_VOL_MINIMO = 500_000   # acciones/dia de media 30d (P38, extraido de generate_analysis)

def cumple_calidad(v):
    """PUNTO 38 (02/08/2026) — filtros de CALIDAD compartidos por informe y alertas.

    Hasta hoy generate_analysis y generate_alerts seleccionaban por su cuenta y el filtro
    de las alertas era estrictamente mas laxo (solo ruptura + score>=65 + R/B>=2.0). Efecto:
    Telegram podia empujar al movil un valor que el informe habia descartado a proposito por
    calidad, y la señal mas visible era la MENOS filtrada. Caso real 02/08/2026: FIS enviado
    por Telegram sin estar entre los finalistas.

    Aqui viven los criterios de calidad, y los dos llamadores usan este mismo predicado. NO
    incluye el ranking ni el top-5: una ruptura puede ser legitima sin entrar en el informe
    por posicion, y esa alerta temprana se conserva a proposito. Lo que se elimina es la
    contradiccion, no el aviso.

    La watchlist personal queda FUERA de esto: son valores que el usuario ha elegido seguir
    y ahi el filtro laxo es deliberado.
    """
    if (v.get('sct') or 0) < 40:
        return False
    if (v.get('adx') or 0) < 20:
        return False
    if v.get('riesgo') == 'ALTO':
        return False
    tipo = (v.get('entry_range') or {}).get('tipo', '')
    if tipo in ('Ruptura activa', 'Ruptura pendiente'):
        rvol = v.get('rvol')
        if rvol is not None and rvol < 1.2:   # sin dato no se penaliza
            return False
    vm = v.get('vol_media_30d')
    if vm is not None and vm < UMBRAL_VOL_MINIMO:
        return False
    return True

def bloque_macro(data):
    """PUNTO 48 (06/08/2026) — texto del bloque MACRO del prompt.

    Extraido de construir_prompt sin cambiar una letra: el cuerpo resuelve por su
    cuenta data.get('macro'), asi que recibe data y no el sub-dict.
    """
    macro_txt = ''
    macro = data.get('macro', {})
    if macro:
        vix = macro.get('^VIX', {})
        dxy = macro.get('DX-Y.NYB', {})
        usy = macro.get('^TNX', {})
        macro_txt = (
            f'DATOS MACRO:\n'
            f'- VIX (miedo): {vix.get("current","?")} (cambio: {vix.get("chg_pct","?")}%)\n'
            f'- DXY (dolar): {dxy.get("current","?")} (cambio: {dxy.get("chg_pct","?")}%)\n'
            f'- US10Y (bono 10a): {usy.get("current","?")}% (cambio: {usy.get("chg","?")} pb)\n'
        )
        # NUEVO (07/07) — alertas de regimen: SOLO se añaden al prompt las series cuyo umbral
        # de velocidad ha disparado (silencio impuesto por codigo: sin disparo, Claude ni ve
        # estas series y por tanto no puede rellenar con comentario macro diario).
        reg = macro.get('regimen', {})
        u30 = reg.get('us30y', {})
        # FIX (08/07, primer caso real): incluir tambien el cambio de 1 dia en la linea.
        # El 08/07 el WTI marcaba -20.7% en 20 sesiones (regimen bajista) PERO habia
        # repuntado la sesion anterior por geopolitica — sin el dato diario, el informe
        # solo podia decir "crudo en fuerte descenso" sin matizar el rebote del dia.
        # Regimen y taktica del dia pueden divergir: Claude debe ver ambos.
        if u30.get('alerta'):
            extra = ' [ha cruzado el nivel del 5.00%]' if u30.get('cruce_5pct') else ''
            # ARREGLO (16/08/2026): la alerta puede dispararse solo por el cruce del 5%,
            # con chg_20s_pb a None desde el P73. Sin esta guarda el formato reventaria
            # justo en la ejecucion mas informativa: aquella en la que hay alerta.
            _v = u30.get('chg_20s_pb')
            _t = f'{_v:+} pb en 20 sesiones' if isinstance(_v, (int, float)) else 'variacion 20s no disponible'
            macro_txt += (f'- ALERTA DE REGIMEN — US30Y (bono 30 años): {u30.get("nivel")}% | '
                          f'{_t} | '
                          f'{u30.get("chg_1d_pb"):+} pb hoy{extra}{_texto_nivel_extremo(u30)}\n')
        wti_r = reg.get('wti', {})
        if wti_r.get('alerta'):
            macro_txt += (f'- ALERTA DE REGIMEN — WTI (petroleo): ${wti_r.get("nivel")} | '
                          f'{wti_r.get("chg_20s_pct"):+}% en 20 sesiones | '
                          f'{wti_r.get("chg_1d_pct"):+}% hoy{_texto_nivel_extremo(wti_r)}\n')
        cred = reg.get('credito_hyg_ief', {})
        if cred.get('alerta'):
            macro_txt += (f'- ALERTA DE REGIMEN — CREDITO (ratio HYG/IEF): '
                          f'{cred.get("chg_20s_pct"):+}% en 20 sesiones | '
                          f'{cred.get("chg_1d_pct"):+}% hoy '
                          f'(diferenciales high-yield tensionandose)\n')
        # NUEVO (11/07) — estructura de indices vs MM50: solo llega al prompt con disparo
        # (cruce reciente o test del nivel); sin disparo, silencio por codigo.
        idx50 = macro.get('indices_mm50', {})
        partes_idx = []
        for tk in ('SPY', 'QQQ'):
            v = idx50.get(tk, {})
            if not v.get('alerta'): continue
            if v.get('cruce_reciente'):
                detalle = f"ha cruzado {'por encima de' if v['estado']=='sobre' else 'por debajo de'} su MM50 hace {v['sesiones_en_estado']} sesion(es)"
            else:
                detalle = f"testando su MM50 como {'soporte' if v['estado']=='sobre' else 'resistencia'}"
            partes_idx.append(f"{tk}: {detalle} (MM50 ${v['mm50']}, precio a {v['dist_pct']:+}%, {v['estado']} desde hace {v['sesiones_en_estado']} sesiones)")
        if partes_idx:
            macro_txt += '- ESTRUCTURA DE INDICES (MM50, tactico): ' + ' | '.join(partes_idx) + '\n'
        macro_txt += '\n'
    return macro_txt


def bloque_amplitud(data):
    """PUNTO 48 (06/08/2026) — texto del bloque AMPLITUD DE MERCADO del prompt.

    Extraido de construir_prompt sin cambiar una letra. Es el bloque que mas ha
    crecido por acumulacion (dispersion, contador de tendencia, direccion de la
    dispersion), con varios añadidos condicionales pegados detras de una
    concatenacion de f-strings; aislarlo es el paso previo a poder reordenarlo con
    el prompt golden como red.
    """
    breadth_txt = ''
    breadth = data.get('breadth', {})
    if breadth and breadth.get('pct_sobre_mm50') is not None:
        n_val = breadth.get('n_valores', '?')
        breadth_txt = (
            f'AMPLITUD DE MERCADO (sobre {n_val} valores del universo S&P500+Nasdaq100+SOX):\n'
            f'- % de valores por encima de su MM20: {breadth.get("pct_sobre_mm20","?")}% (amplitud tactica de corto plazo)\n'
            f'- % de valores por encima de su MM50: {breadth["pct_sobre_mm50"]}%\n'
            f'- % de valores por encima de su MM200: {breadth["pct_sobre_mm200"]}%\n'
            f'- Nuevos maximos de 52 semanas: {breadth.get("nuevos_max_52s","?")} | '
            f'Nuevos minimos de 52 semanas: {breadth.get("nuevos_min_52s","?")}\n'
            f'- Avance/Descenso (sesion actual): {breadth.get("avance","?")} valores suben, '
            f'{breadth.get("descenso","?")} bajan\n'
            f'- McClellan Oscillator (ratio-adjusted): {breadth.get("mcclellan","?")}\n'
            f'- Pendiente MM200 del SPY: {breadth.get("pendiente_mm200_spy","?")}\n'
        )
        # P37c (02/08/2026) — el ratio de dispersion se instrumenta desde julio y sale en
        # el log, pero NUNCA se pasaba al prompt: el informe hablaba de amplitud sin verlo.
        # Se detecto al añadir el contador de tendencia (P37b): el modelo ignoraba la
        # tendencia de la dispersion porque no conocia el valor. Se incluye con su desglose
        # de volatilidades, que es lo que lo hace interpretable, y SIN umbral: no hay base
        # empirica para fijarlo (ver P32).
        _disp = breadth.get('dispersion') or {}
        if _disp.get('ratio') is not None:
            breadth_txt += (
                # P82 (03/09/2026) — la etiqueta decia "Dispersion indice/valor: 0.216" y el
                # informe del 03/09 escribio "La dispersion indice/valor de 0,216 es baja: los
                # valores se mueven mucho mas que el indice", contradictorio en la superficie.
                # El nombre invertido del ratio (documentado desde el 04/08) reaparece en cuanto
                # la palabra "dispersion" viaja pegada al numero. Se separan: el numero se llama
                # RATIO, y la dispersion solo se nombra ya interpretada.
                f'- Ratio de convergencia indice/valor: {_disp["ratio"]} '
                f'(volatilidad anualizada del SPY {_disp.get("vol_spy_20s_pct","?")}% frente a '
                f'{_disp.get("vol_media_valores_20s_pct","?")}% de media de los valores). '
                f'ATENCION: este numero NO es el nivel de dispersion, es su INVERSO. Ratio BAJO = '
                f'ALTA dispersion (los valores se mueven mucho mas que el indice, mercado disperso '
                f'donde la seleccion individual pesa mas); ratio ALTO = BAJA dispersion (todo se '
                f'mueve junto). NUNCA escribas "la dispersion es baja/alta" citando este numero al '
                f'lado, ni llames "dispersion" a la cifra: si mencionas el numero, llamalo RATIO. '
                f'NO hay umbral calibrado: describe el nivel y su evolucion, no lo clasifiques '
                f'en categorias inventadas.\n')
        # P37b — contexto temporal: sin esto el informe describe la cifra del dia como
        # si fuera nueva cada noche. Se anade fuera de la concatenacion de arriba porque
        # es condicional (puede no haber serie suficiente).
        _tend = breadth.get('tendencia')
        if _tend:
            # ARREGLO (04/08/2026) — la DIRECCION se resuelve aqui, no en el prompt. Se pedia al
            # modelo que invirtiera la relacion (ratio subiendo = mercado MENOS disperso) y lo
            # hacia bien de forma intermitente: 2 aciertos de 3 con la misma instruccion. El
            # nombre de la metrica esta invertido respecto a su escala y ninguna redaccion
            # arregla eso de forma fiable. Ahora recibe la frase ya resuelta: sin inversion que
            # hacer, no hay inversion que fallar. Misma leccion que el P33.
            _dir = _tend['dispersion']
            if _dir == 'alcista':
                _lectura = ('la DISPERSION DEL MERCADO ESTA DISMINUYENDO (el ratio sube): los valores '
                            'convergen con el indice y la ventaja de seleccionar nombres se estrecha')
            elif _dir == 'bajista':
                _lectura = ('la DISPERSION DEL MERCADO ESTA AUMENTANDO (el ratio baja): los valores se '
                            'separan del indice y la ventaja de seleccionar nombres crece')
            else:
                _lectura = 'la dispersion del mercado no muestra tendencia definida'
            _racha = (f', y lleva asi {_tend["sesiones_dispersion"]} sesiones'
                      if _tend['sesiones_dispersion'] else '')
            # ARREGLO (05/08/2026) — mismo caso que la dispersion, que se corrigio y este no:
            # con tendencia plana el contador es 0 y el prompt decia "lleva 0 sesiones en
            # tendencia plana". El modelo improviso una explicacion sin sentido ("describe un
            # momento de amplitud que acaba de formarse"). Sin racha, no se menciona racha.
            _mcc = (f'el McClellan lleva {_tend["sesiones_mcclellan"]} sesiones en tendencia '
                    f'{_tend["mcclellan"]}'
                    if _tend.get('sesiones_mcclellan')
                    # PUNTO 53 (08/08/2026) — sin la coletilla final, el modelo convertia
                    # "no muestra tendencia definida" en "amplitud sostenida y estable"
                    # (visto el 05, el 06 y el 07/08): afirmaba estabilidad donde el dato
                    # solo dice indeterminacion. Son cosas distintas y la diferencia importa:
                    # una racha rota no es lo mismo que una situacion asentada.
                    else ('el McClellan no muestra una tendencia definida en las ultimas '
                          'sesiones. OJO: esto significa INDETERMINACION —la racha se ha roto '
                          'o alterna—, NO que la amplitud sea estable o sostenida. No lo '
                          'describas como equilibrio ni como normalidad asentada'))
            breadth_txt += (
                f'- DIRECCION DE LA DISPERSION: {_lectura}{_racha}. Usa esta lectura TAL CUAL; no la '
                f'reinterpretes ni deduzcas tu la relacion entre el ratio y la dispersion.\n'
                f'- CONTEXTO TEMPORAL: {_mcc}. USA ESTE DATO: di si la amplitud es una '
                f'situacion NUEVA o SOSTENIDA, en vez de describir la cifra del dia como si fuera '
                f'informacion nueva. Es descripcion del pasado, NO prediccion.\n')
        breadth_txt += '\n'
    return breadth_txt


def bloque_seguimiento(data):
    """PUNTO 49 (06/08/2026) — texto de SEGUIMIENTO: alertas abiertas y resoluciones.

    Extraido de construir_prompt sin cambiar una letra. A diferencia del resto de la
    funcion, este bloque NO sale a la red: solo lee data y llama a dos funciones puras
    (dedup_alertas_por_ticker, resoluciones_por_ticker). Por eso es reproducible al 100%
    y se puede probar con datos fijos, sin depender de capturar el prompt en Colab —
    que es el criterio que guia esta fase del refactor.

    Devuelve el fragmento de texto; cadena vacia si no hay alertas ni resoluciones.
    """
    summary = ''
    evaluaciones = data.get('evaluaciones', [])
    alertas_seguimiento = dedup_alertas_por_ticker([e for e in evaluaciones if e.get('alerta_cmf') or e.get('alerta_supertrend')])
    if alertas_seguimiento:
        summary += '\nSEGUIMIENTO — ALERTAS (setups de sesiones anteriores con señal de salida o flujo deteriorado):\n'
        summary += 'CMF<0.05 = presion compradora debilitandose (señal blanda: valorar reducir 50%).\n'
        summary += 'SUPERTREND BAJISTA = el precio ya cruzo el nivel dinamico de stop (señal dura: salida total, no solo reduccion).\n'
        for a in alertas_seguimiento:
            flags = []
            if a.get('alerta_cmf'): flags.append(f'CMF NEGATIVO {a.get("cmf_dias_negativo")} SESIONES CONSECUTIVAS(actual:{a["cmf_actual"]})')
            # NUEVO (05/07) — distinguir ruptura nueva (cambio a bajista en la sesion mas
            # reciente) de ruptura persistente (lleva N sesiones rota y el precio no ha
            # recuperado el nivel). Caso real $TSLA: +8.46% en la sesion pero Supertrend
            # seguia bajista de dias antes; el informe lo redactaba como ruptura del dia.
            if a.get('alerta_supertrend'):
                dias_st = a.get('supertrend_dias')
                if dias_st is not None and dias_st > 1:
                    flags.append(f'SUPERTREND BAJISTA PERSISTENTE desde hace {dias_st} sesiones(nivel:${a.get("supertrend_nivel")})')
                else:
                    flags.append(f'SUPERTREND BAJISTA NUEVO — ruptura de esta sesion(nivel:${a.get("supertrend_nivel")})')
            summary += (f'- {a["ticker"]}: setup del {a["fecha_setup"]} ({a["dias"]}d) | '
                        f'Ret: {a["ret_pct"]}% | {" + ".join(flags)} | Estado: {a["resultado"]}\n')
        summary += '\n'
    # PUNTO 40 — resoluciones: stops y targets alcanzados. Van ANTES del aviso porque son
    # hechos consumados sobre posiciones del usuario, no vigilancia.
    # P43 — solo se detallan las resoluciones RECIENTES; las antiguas se resumen en una
    # linea. Con el P40 aparecieron 22 de golpe (todo julio) y el informe se corto por
    # max_tokens. Lo accionable es lo que acaba de pasar: un stop de hace tres semanas ya
    # no es una decision pendiente.
    resoluciones = data.get('resoluciones') or resoluciones_por_ticker(evaluaciones)
    if resoluciones:
        recientes = [r for r in resoluciones if r.get('reciente', True)]
        antiguas = [r for r in resoluciones if not r.get('reciente', True)]
        # PUNTO 54 (08/08/2026) — TOPE por numero, ademas del de recencia del P43. La
        # ventana de 5 dias no bastaba: el 08/08 seguian entrando ~21 resoluciones y el
        # informe se corto por max_tokens perdiendo la seccion RIESGOS DEL ESCENARIO
        # entera (lo cazo el P36). Se detallan las de MAYOR MAGNITUD —los peores stops y
        # los mejores objetivos, que es lo accionable— y el resto pasa al resumen. Un
        # stop del -4% no cambia ninguna decision; uno del -32% si.
        if len(recientes) > MAX_RESOLUCIONES_DETALLADAS:
            recientes = sorted(recientes, key=lambda r: abs(r.get('ret_pct') or 0), reverse=True)
            recientes, resto = recientes[:MAX_RESOLUCIONES_DETALLADAS], recientes[MAX_RESOLUCIONES_DETALLADAS:]
            antiguas = antiguas + resto
        summary += ('\nSEGUIMIENTO — SETUPS RESUELTOS (el precio ya ha tocado stop u objetivo; '
                    'son hechos consumados, NO vigilancia ni recomendacion de entrada):\n')
        for r in recientes:
            que = 'STOP ALCANZADO' if r['resultado'] == 'stop' else 'OBJETIVO ALCANZADO'
            summary += (f'- {r["ticker"]}: setup del {r["fecha_setup"]} ({r["dias"]}d) | {que} | '
                        f'Entrada ${r["precio_entrada"]} -> ${r["precio_actual"]} | Ret: {r["ret_pct"]}%\n')
        if not recientes:
            summary += '- (ninguna resolucion nueva en las ultimas sesiones)\n'
        if antiguas:
            _st = [r for r in antiguas if r['resultado'] == 'stop']
            _tg = [r for r in antiguas if r['resultado'] == 'target']
            summary += (f'RESTO (ya reportadas antes o de menor magnitud, NO las detalles una por una): '
                        f'{len(_st)} stops y {len(_tg)} objetivos siguen dentro de la ventana de '
                        f'seguimiento. Resumelas en UNA sola frase.\n')
        summary += ('Menciona SIEMPRE estas resoluciones en la seccion de seguimiento, con una linea por '
                    'valor, ANTES de las alertas de posiciones abiertas. Un stop alcanzado es la '
                    'informacion mas accionable del informe. Si un mismo ticker aparece aqui y tambien '
                    'entre las alertas abiertas, son OCURRENCIAS DISTINTAS del mismo valor (setups '
                    'creados en fechas distintas, con stops distintos): dilo explicitamente en vez de '
                    'presentarlo como una contradiccion.\n\n')
    return summary


def bloque_extendidos(values):
    """PUNTO 50 (06/08/2026) — texto de EXTENDIDOS: vigilancia activa y descartados.

    Extraido de construir_prompt sin cambiar una letra. Como bloque_seguimiento, es
    PURO: solo depende de la lista de valores, no sale a la red y no lee estado global.
    Por eso se puede probar con datos fijos en vez de capturando el prompt en Colab.

    Devuelve el fragmento de texto; cadena vacia si no hay extendidos ni descartados.
    """
    summary = ''
    extendidos=[v for v in values if v.get('entry_range',{}).get('tipo','').startswith('Extendido')]
    if extendidos:
        # PUNTO 6/14 — separar candidatos de vigilancia activa real (trigger normal) de los
        # casos severos (trigger con AVISO: distancia extrema, tratados como descartados).
        # Mezclarlos en el mismo formato de parrafo por ticker generaba confusion: un caso
        # explicitamente descartado no debe presentarse con el mismo detalle que un candidato
        # de vigilancia real. Los severos se comprimen en una sola linea de contexto.
        vigilancia = [v for v in extendidos if not (v['entry_range'].get('trigger') or '').startswith('AVISO:')]
        # El universo completo (524 tickers) puede acumular un pool grande de candidatos
        # "Extendido" (todo lo que supera price>bl*1.15 en algun momento de su historia).
        # Sin ordenar, el tope de 5 mostraba los primeros por orden de aparicion en la lista
        # (arbitrario), pudiendo dejar fuera a candidatos con el trigger de reentrada mas cercano
        # a activarse en favor de otros mas lejanos. Se ordena por distancia a MM50 ascendente:
        # los mas proximos a reentrar (mas accionables a corto plazo) tienen prioridad.
        vigilancia = sorted(vigilancia, key=lambda v: v['entry_range'].get('dist_ma50_pct') if v['entry_range'].get('dist_ma50_pct') is not None else 999)
        severos = [v for v in extendidos if (v['entry_range'].get('trigger') or '').startswith('AVISO:')]
        if vigilancia:
            summary+='\nEXTENDIDOS — VIGILANCIA ACTIVA:\n'
            for v in vigilancia[:5]:
                er = v['entry_range']
                summary+=f'- {v["ticker"]}: ${v.get("price","?")} | {er.get("nota","Extendido")}'
                if er.get('trigger'): summary+=f' | TRIGGER: {er["trigger"]}'
                summary+='\n'
        if severos:
            partes = [f'{v["ticker"]} ({v["entry_range"].get("dist_ma50_pct","?")}% sobre MM50)' for v in severos[:8]]
            summary+=f'\nEXTENDIDOS — DESCARTADOS POR DISTANCIA EXTREMA (no son candidatos de vigilancia, solo contexto de sector): {", ".join(partes)}.\n'
    return summary


def bloque_setups(valid, fundamentales):
    """PUNTO 52 (08/08/2026) — texto de SETUPS VALIDOS: un parrafo por candidato.

    Es el bloque mas grande del prompt (149 lineas) y el ultimo que quedaba dentro de
    construir_prompt. Extraido sin cambiar una letra.

    A pesar de su tamaño es PURO: cuando se ejecuta, las llamadas a red ya se han hecho
    (get_news_earnings y los fundamentales vienen resueltos dentro de `valid` y del dict
    `fundamentales`). Por eso se puede probar con datos fijos, que era el criterio de
    esta fase: separar lo que sale a la red de lo que solo ensambla texto.
    """
    summary = ''
    for v in valid[:5]:
        er=v['entry_range']
        rsi_s = v.get("rsi"); adx_s = v.get("adx"); cmf_s = v.get("cmf"); sq_s = v.get("squeeze"); sct_s = v.get("sct")
        indicadores = []
        if rsi_s: indicadores.append(f"RSI:{rsi_s}")
        if adx_s: indicadores.append(f"ADX:{adx_s}")
        if cmf_s: indicadores.append(f"CMF:{cmf_s}")
        if sct_s: indicadores.append(f"SCT:{sct_s}")
        if sq_s: indicadores.append("SQUEEZE")
        ind_str = " | ".join(indicadores) if indicadores else ""
        # NUEVO (27/06) — el objetivo es una proyeccion, no un nivel de orden real como entrada/
        # stop: mostrarlo con 2 decimales ($671.36) da una falsa sensacion de precision que el
        # mercado no tiene. Se redondea solo para el TEXTO del informe (no afecta al calculo
        # interno de R/B, que sigue usando el valor exacto). Entrada y stop SI son niveles de
        # orden reales que Carlos necesita exactos para operar, no se tocan.
        def _redondeo_display(precio):
            if precio is None: return precio
            return round(precio) if precio >= 20 else round(precio, 1)
        target_disp = _redondeo_display(er.get("target"))
        target_parcial_disp = _redondeo_display(er.get("target_parcial"))
        summary+=f'- {v["ticker"]} ({v["group"]}): ${v.get("price","?")} | {er["tipo"]} | Entrada:${er["entry_lo"]}-${er["entry_hi"]} | Stop:${er["stop"]}'
        # PUNTO 4 — objetivo escalonado: parcial y final
        if target_parcial_disp is not None: summary+=f' | Obj.parcial:${target_parcial_disp}'
        summary+=f' | Obj.final:${target_disp}'
        if ind_str: summary+=f' | {ind_str}'
        if er.get('rr'): summary+=f' | R/B:1:{er["rr"]}'
        # PUNTO 33 (01/08/2026) — estimación de tiempo al objetivo en días hábiles
        # Fórmula: distancia al objetivo / ATR. Se emite como rango (ATR mínimo y máximo
        # del período disponible) para no dar falsa precisión. Si el ATR es muy bajo
        # (universo en calma extrema) o la distancia es cero, se omite silenciosamente.
        # ARREGLO (02/08/2026): la version original pedia er['objetivo'] y er['entrada'],
        # claves que NO existen en entry_range (son 'target', 'entry_lo', 'entry_hi'). El
        # bloque devolvia None siempre y la linea no se emitia nunca, en silencio porque
        # el except de abajo se lo tragaba. Verificado contra data.json de produccion.
        try:
            precio_obj = er.get('target')
            _elo, _ehi = er.get('entry_lo'), er.get('entry_hi')
            precio_ent = ((_elo + _ehi) / 2) if (_elo and _ehi) else (_elo or _ehi)
            atr_val = v.get('atr')
            if precio_obj and precio_ent and atr_val and atr_val > 0:
                distancia = abs(precio_obj - precio_ent)
                # CORRECCION DE FORMULA (02/08/2026). La version anterior emitia
                # distancia/ATR con un rango de +-40% y producia cifras irreales: DOCU
                # daba "4-9 dias habiles" para un +37%. El error es conceptual, no de
                # calibracion: el ATR es el rango medio diario ABSOLUTO (incluye el
                # movimiento en contra), no la deriva neta. distancia/ATR solo seria
                # correcto si el precio avanzara un ATR entero en la misma direccion
                # cada sesion, sin un solo dia en contra.
                #
                # Los dos extremos teoricos acotan la respuesta:
                #   n     = distancia/ATR      -> SUELO: avance en linea recta
                #   n**2  = (distancia/ATR)**2 -> TECHO: paseo aleatorio puro, donde el
                #                                 desplazamiento neto escala con ATR*sqrt(n)
                # La realidad (tendencia con retrocesos) cae entre ambos, asi que se emite
                # el intervalo completo y se etiqueta como ORDEN DE MAGNITUD. Es un rango
                # ancho a proposito: fingir precision aqui es lo que hacia dano.
                ratio = distancia / atr_val
                dias_min = max(1, round(ratio))
                dias_max = min(250, round(ratio ** 2))   # tope: ~1 año bursatil
                if dias_max > dias_min:
                    summary += (f' | Tiempo estimado al objetivo: {dias_min}-{dias_max} sesiones'
                                f' (orden de magnitud, NO prediccion: {dias_min}=avance en linea'
                                f' recta, {dias_max}=con retrocesos tipicos; ATR ${round(atr_val,2)}/dia)')
        except Exception as _e33:
            # Antes era un pass mudo: un bug de claves sobrevivio asi una jornada entera.
            print(f'  AVISO P33: no se pudo estimar tiempo al objetivo de {v.get("ticker","?")}: {_e33}')
        # PUNTO 18 — aviso explicito para que Claude no venda el pullback como retroceso sano
        if er.get('rsi_sobrecompra'):
            summary+=' | AVISO: PULLBACK CON RSI EN SOBRECOMPRA (>=70) — el retroceso apenas ha aliviado la sobrecompra'
        # PUNTO 9 — RVOL informativo (el filtro ya elimino rupturas con RVOL<1.2, pero el valor
        # exacto sigue siendo util para que Claude calibre la solidez del respaldo de volumen)
        if v.get('rvol') is not None: summary+=f' | RVOL:{v["rvol"]}x'
        # PUNTO 12+13 — si el techo del objetivo no es el maximo de 52s por defecto, indicar
        # el motivo explicito (correccion fuerte posterior invalido el maximo bruto como techo realista)
        tm = v.get('techo_metodo')
        if tm and tm not in (None, 'max52', 'sin_datos'):
            etiqueta_tm = {'fib0.764_post_correccion': 'objetivo acotado a un retroceso del 76.4% de la caida previa (no al maximo de 52s, invalidado por una correccion >20% posterior)',
                           'fib0.618_post_correccion': 'objetivo acotado a un retroceso del 61.8% de la caida previa (no al maximo de 52s, invalidado por una correccion >20% posterior)',
                           'fib0.764_post_correccion_sin_atr': 'objetivo acotado a un retroceso del 76.4% de la caida previa (maximo de 52s invalidado por correccion >20%; sin ATR disponible para alternativa)',
                           'atr_proyectado_post_correccion': 'objetivo proyectado por rango medio (ATR), ya que ni el maximo de 52s ni los retrocesos de Fibonacci de la caida previa dejaban margen suficiente'}.get(tm)
            if etiqueta_tm: summary+=f' | OBJETIVO: {etiqueta_tm}'
        # PUNTO 11 — Volume Profile aproximado: POC como referencia de soporte/resistencia
        # institucional, y aviso explicito cuando entry_lo coincide con VAL (confirma el pullback)
        vprof = v.get('vol_profile')
        if vprof:
            summary+=f' | POC:${vprof["poc"]} VAH:${vprof["vah"]} VAL:${vprof["val"]}'
            if vprof.get('confirma_pullback'):
                # P67 — se describe la posicion REAL del VAL, no una coincidencia asumida.
                _pos = vprof.get('val_posicion')
                _d = vprof.get('val_dist_entry_lo_pct')
                if _pos == 'dentro':
                    summary+=' (VAL DENTRO de la zona de entrada — refuerza calidad del pullback:'
                    summary+=' el soporte tiene respaldo de volumen, no solo tecnico)'
                elif _pos == 'debajo' and _d is not None:
                    summary+=(f' (VAL {abs(_d)}% POR DEBAJO del limite inferior de la entrada — cerca,'
                              ' pero NO coincide con la zona: es un soporte de volumen inmediatamente'
                              ' por debajo, no un soporte sobre el que se entra. NO afirmar que la'
                              ' entrada coincide con el VAL)')
                else:
                    summary+=' (VAL proximo a la zona de entrada — respaldo de volumen cercano)'
            # NUEVO (05/07) — VAL coincide con la entrada pero esta demasiado cerca del stop
            # (dentro de la mitad inferior del rango entrada->stop): NO es soporte util. Se
            # marca explicitamente para que Claude no lo cite como argumento de calidad.
            elif vprof.get('coincide_val') and not vprof.get('val_valido_como_soporte'):
                summary+=' (VAL dentro del rango de riesgo, demasiado cerca del stop — NO citarlo como soporte)'
        summary+=f' | RIESGO:{v.get("riesgo","?")} (motivo exacto: {v.get("riesgo_motivo","?")}) | RANKING:{v.get("ranking","?")}'
        # PUNTO 19 — desglose auditable del ranking
        dsg = v.get('ranking_desglose')
        if dsg:
            summary+=f' (desglose: SCT {dsg["sct"]}/40 + R/B {dsg["rb"]}/20 + sector {dsg["sector"]}/20 + CMF {dsg["cmf"]}/20)'
        # PUNTO 7 — earnings dentro de la ventana de aviso (10-21 dias)
        if v.get('earnings_accion') == 'avisar':
            summary+=f' | EARNINGS EN {v["dias_earnings"]} DIAS ({v["earnings_date"]}) — AVISO OBLIGATORIO'
        # Datos fundamentales si disponibles
        fund = fundamentales.get(v['ticker'], {})
        if fund:
            f_parts = []
            if fund.get('per_forward'): f_parts.append(f'PERfwd:{fund["per_forward"]}x')
            if fund.get('ev_ebitda'):   f_parts.append(f'EV/EBITDA:{fund["ev_ebitda"]}x')
            if fund.get('peg'):         f_parts.append(f'PEG:{fund["peg"]}')
            if fund.get('eps_trailing') and fund.get('eps_fwd'):
                f_parts.append(f'EPS:{fund["eps_trailing"]}->{fund["eps_fwd"]}')
            if fund.get('eps_growth'):  f_parts.append(f'EPSgrowth:{fund["eps_growth"]}%')
            if fund.get('margen_neto'): f_parts.append(f'Margen:{fund["margen_neto"]}%')
            if fund.get('roe'):         f_parts.append(f'ROE:{fund["roe"]}%')
            if fund.get('deuda_equity'):f_parts.append(f'D/E:{fund["deuda_equity"]}')
            if fund.get('rev_growth'):  f_parts.append(f'RevGrowth:{fund["rev_growth"]}%')
            # NUEVO (28/06) — consenso de analistas y precio objetivo, sin coste adicional
            if fund.get('analista_consenso'):
                consenso_txt = f'Consenso:{fund["analista_consenso"]}'
                if fund.get('analista_n_opiniones'): consenso_txt += f'({fund["analista_n_opiniones"]} analistas)'
                f_parts.append(consenso_txt)
            if fund.get('analista_target_medio'):
                target_txt = f'TargetMedio:${fund["analista_target_medio"]}'
                if fund.get('analista_upside_pct') is not None:
                    target_txt += f'({fund["analista_upside_pct"]:+}% vs precio actual)'
                f_parts.append(target_txt)
            # NUEVO (05/07, opcion B) — catalizador (revisiones, sorpresa: via yfinance) y
            # calidad fundamental adicional (margenes via yfinance; ROIC/EPStrim via FMP opcional)
            if fund.get('upgrades_90d') is not None:
                f_parts.append(f'RevisionesAnalistas90d:{fund["upgrades_90d"]}subidas/{fund.get("downgrades_90d",0)}bajadas')
            if fund.get('ultima_revision'):
                f_parts.append(f'UltimaRevision:{fund["ultima_revision"]}')
            if fund.get('sorpresa_eps_pct') is not None:
                f_parts.append(f'SorpresaEPS:{fund["sorpresa_eps_pct"]:+}%({fund.get("sorpresa_fecha","?")})')
            if fund.get('fmp_eps_q'):
                f_parts.append(f'EPStrim:{"->".join(str(x) for x in fund["fmp_eps_q"])}')
            if fund.get('margen_bruto') is not None:
                f_parts.append(f'MargenBruto:{fund["margen_bruto"]}%')
            if fund.get('margen_operativo') is not None:
                f_parts.append(f'MargenOperativo:{fund["margen_operativo"]}%')
            if fund.get('fmp_roic') is not None:
                f_parts.append(f'ROIC:{fund["fmp_roic"]}%')
            if fund.get('fcf_growth') is not None:
                f_parts.append(f'FCFgrowth:{fund["fcf_growth"]}%')
            if f_parts: summary+=f'        FUND: {" | ".join(f_parts)}\n'
        # PUNTO 7 — noticias recientes (max 3, solo si existen)
        if v.get('noticias'):
            noticias_str = ' / '.join(f'"{n["titulo"]}" ({n["fuente"]}, {n["fecha"]})' for n in v['noticias'])
            summary+=f'        NOTICIAS: {noticias_str}\n'
        summary+='\n'
    return summary


MAX_RESOLUCIONES_DETALLADAS = 10   # P54: tope de resoluciones que se detallan en el prompt

def recolectar_datos_candidatos(valid, fundamentales):
    """P64 (12/08/2026) — FASE DE RED del prompt: las cuatro llamadas externas, juntas.

    Ultimo tramo del refactor P47-P52. construir_prompt mezclaba en su fase final cuatro
    llamadas a red (earnings/noticias, fundamentales extra, FMP, calendario macro y
    correlacion) con el ensamblado del texto, de modo que NADA del ensamblado final se
    podia ejecutar ni verificar sin salir a internet — la unica forma de comprobar un
    cambio ahi era subirlo y lanzar Colab.

    Separadas: esta funcion devuelve un dict con todo lo externo, y
    ensamblar_bloques_externos() lo convierte en texto sin tocar la red. Los tests pueden
    inyectar un dict fijo (construir_prompt(data, externos=...)) y verificar el texto.

    OJO — dos efectos deliberados que NO son puros y por eso viven aqui y no en el
    ensamblado: se mutan los dicts de `valid` (campos de earnings/noticias) y el dict
    `fundamentales` compartido con main() y la celda del pipeline, que es como esos campos
    acaban persistiendo en data.json.
    """
    # PUNTO 7 — earnings proximos + noticias, SOLO para los candidatos ya ordenados por
    # ranking, consultando un margen extra (hasta 8) para poder reemplazar los descartados
    # por earnings y seguir entregando hasta 5 setups cuando haya candidatos suficientes.
    candidatos_para_news = valid[:8]
    news_data = get_news_earnings([v['ticker'] for v in candidatos_para_news])
    valid_final = []
    descartados_por_earnings = []
    for v in candidatos_para_news:
        ne = news_data.get(v['ticker'], {})
        v['earnings_date'] = ne.get('earnings_date')
        v['dias_earnings'] = ne.get('dias_earnings')
        v['earnings_accion'] = ne.get('earnings_accion', 'ignorar')
        v['noticias'] = ne.get('noticias', [])
        if v['earnings_accion'] == 'descartar':
            descartados_por_earnings.append(v)
            continue
        valid_final.append(v)
        if len(valid_final) == 5:
            break

    # NUEVO (05/07, opcion B) — campos de catalizador/calidad SOLO para los setups validos
    # finales (<=5): yfinance como fuente principal (sin key, sin restriccion de simbolos)
    # y FMP como complemento opcional (ROIC + serie EPS 8T, solo si hay key y el simbolo
    # esta cubierto por el free tier). Los campos se fusionan en el dict `fundamentales`
    # compartido con main() y la celda del pipeline (mutacion deliberada del mismo objeto),
    # de modo que persisten en data.json — mismo patron por el que earnings/noticias
    # persisten via values.
    if valid_final:
        tickers_setups = [v['ticker'] for v in valid_final]
        extra_data = get_fundamentales_extra(tickers_setups)
        for tk, campos in extra_data.items():
            if campos:
                fundamentales.setdefault(tk, {}).update(campos)
        if FMP_KEY:
            fmp_data = get_fundamentales_fmp(tickers_setups)
            for tk, campos in fmp_data.items():
                if campos:
                    fundamentales.setdefault(tk, {}).update(campos)

    # PUNTO 23 — eventos macro programados proximos (riesgo de gap de mercado)
    eventos_macro = check_eventos_macro()

    # PUNTO 17 — correlacion entre los candidatos finales (concentracion de cartera).
    # Se calcula sobre los mismos <=5 setups que van al informe; si el cache no esta
    # poblado o la muestra comun es corta, cc=None y el informe sale sin el bloque.
    cc = calc_correlacion_candidatos([v['ticker'] for v in valid_final[:5]])

    return {'valid': valid_final, 'descartados_por_earnings': descartados_por_earnings,
            'eventos_macro': eventos_macro, 'correlacion': cc}


def ensamblar_bloques_externos(externos, valid):
    """P64 — texto de los bloques macro y correlacion. PURO: no toca la red.

    Los print() de traza se quedan aqui a proposito: describen lo que entra en el prompt,
    no la descarga, y asi el log sigue apareciendo en el mismo punto de siempre.
    """
    s = ''
    eventos_macro = externos.get('eventos_macro')
    if eventos_macro:
        s += formato_eventos_macro_summary(eventos_macro)
        print('  Eventos macro proximos: ' +
              ', '.join(f"{e['evento'].split(' (')[0]} {e['fecha']} ({'HOY' if e['dias_habiles']==0 else str(e['dias_habiles'])+'dh'})" for e in eventos_macro))
    cc = externos.get('correlacion')
    if cc:
        s += formato_correlacion_summary(cc, valid[:5])
        if cc['pares_altos']:
            print('  Correlacion candidatos: pares >=0.70: ' +
                  ', '.join(f'{a}-{b} ({c})' for a, b, c in cc['pares_altos']))
        else:
            print(f'  Correlacion candidatos: sin pares >=0.70 ({cc["n_sesiones"]} sesiones)')
    return s


def data_json_publicable(min_frescos=MIN_FRESCOS_PARA_PUBLICAR):
    """P65 (13/08/2026) — ¿es data.json lo bastante bueno para pisar el que ya hay?

    El 12/08 yfinance sirvio el panel incompleto (85,5% de tickers con dato en la ultima
    sesion, 29,7% en la watchlist). El P22 lo detecto y lo imprimio DOS veces... y el
    scanner subio igualmente, sustituyendo un data.json bueno por otro peor: la amplitud
    perdio 74 valores del recuento (A/D 246/200 en vez de 288/232) y varias resoluciones
    RETROCEDIERON a precios antiguos. Se repitio en tres ejecuciones seguidas.

    Avisar no basta cuando el efecto es pisar el dato bueno — misma leccion que el P55 con
    el informe vacio. Aqui la asimetria es clara: conservar el data.json de ayer degrada la
    web unas horas; publicar uno incompleto corrompe lo que se lee y se decide HOY.

    Solo afecta a data.json. Los historicos (breadth, rates, checkpoints, resoluciones,
    putcall, spy_health) siguen su curso: son series con dedupe por fecha y la ejecucion
    siguiente los corrige, mientras que data.json es un vuelco completo sin historial.
    """
    fr = _BREADTH_CACHE.get('frescura_panel')
    if not fr:
        return True, None   # sin chequeo (panel pequeño o vacio): no bloquear
    pct = fr.get('pct_tickers_frescos')
    if pct is None or pct >= min_frescos:
        return True, None
    return False, (f'data.json NO SE SUBE: solo {pct}% de tickers con dato en la ultima '
                   f'sesion ({fr.get("ultima_sesion")}), por debajo del minimo de {min_frescos}%. '
                   'Se conserva el data.json anterior para no pisarlo con datos incompletos. '
                   'El resto de historicos si se actualizan.')


def _texto_nivel_extremo(serie):
    """P77 — frase lista para el prompt cuando el NIVEL es lo extremo, no la velocidad.

    Se le da resuelta al modelo en vez de dejarle deducirlo de un percentil suelto: misma
    leccion que el P33, el P53 y el P60. Y se dice explicitamente que puede haber alerta sin
    movimiento brusco, porque si no el modelo tiende a buscar una variacion grande que no
    existe y a describir el dato como tranquilo.
    """
    n = (serie or {}).get('nivel_extremo') or {}
    if not n.get('alerta_nivel'):
        return ''
    return (f' | NIVEL EXTREMO: percentil {n["percentil"]} de sus ultimas {n["n"]} sesiones '
            f'(referencia del periodo: {n["referencia"]}). Esta alerta la dispara el NIVEL, '
            'no la velocidad: puede no haber habido ningun movimiento brusco reciente y aun '
            'asi estar en un extremo historico. NO describirlo como un entorno tranquilo')

def construir_prompt(data, externos=None):
    """PUNTO 47 (06/08/2026) — construye el prompt del informe. Extraida de generate_analysis.

    generate_analysis tenia 552 lineas y hacia cuatro cosas: filtrar candidatos, calcular
    el tiempo al objetivo, ordenar el ranking y ensamblar el texto del prompt; ademas de
    llamar a la API y validar la respuesta. Separar la construccion del texto de su envio
    permite capturar y comparar el prompt sin tocar la red, que es lo unico que detecta un
    cambio accidental de redaccion: el scanner no falla cuando el prompt empeora, ni los
    tests se ponen rojos — simplemente el informe sale peor y nadie se entera.

    Esta extraccion NO cambia una sola letra del texto: el cuerpo se movio tal cual. La
    verificacion es un diff byte a byte del prompt generado antes y despues sobre los
    mismos datos reales.
    """
    groups=data['groups']; values=data['values']; ts=data['timestamp']; spy_ok=data.get('spy_healthy',True)
    strong=[g for g in groups if g['score']>=70]; emerging=[g for g in groups if 50<=g['score']<70]; weak=[g for g in groups if g['score']<30]
    # Datos macro para el prompt
    macro_txt = bloque_macro(data)
    breadth_txt = bloque_amplitud(data)

    summary=f'DATOS — {ts}\nSPY sobre MM200: {"SI" if spy_ok else "NO"}\n\n' + macro_txt + breadth_txt + 'FUERTES:\n'
    for g in strong[:5]:
        leaders=', '.join(m['ticker']+'('+str(m['score'])+')' for m in g['top3'])
        summary+=f'- {g["group"]}: {g["score"]} | RS:{g["rs_mean"]}% | {leaders}\n'
    summary+='\nEMERGENTES:\n'
    for g in emerging[:5]: summary+=f'- {g["group"]}: {g["score"]} | RS:{g["rs_mean"]}%\n'
    fundamentales = data.get('fundamentales', {})
    # PUNTO 1 — descartar RIESGO=ALTO antes de pasar a Claude (no llegan setups de riesgo alto)
    # PUNTO 2 — filtro ADX>=20 obligatorio en setups validos
    # PUNTO 9 — filtro RVOL>1.2 para rupturas (activas y pendientes): una ruptura sin volumen
    # es sospechosa. Pullbacks no se filtran por RVOL (no requieren volumen extraordinario).
    # P38 (02/08/2026) — los criterios de calidad (SCT, ADX, riesgo, RVOL de ruptura y
    # volumen minimo de 500K/dia) viven ahora en cumple_calidad() a nivel de modulo, para
    # que generate_alerts aplique EXACTAMENTE los mismos y Telegram no pueda enviar un
    # valor que este informe descarto. Aqui solo quedan los criterios propios del informe.

    valid=[v for v in values
           if v.get('entry_range',{}).get('entry_lo')
           and (v.get('entry_range',{}).get('rr') or 0)>=2.0
           and cumple_calidad(v)]

    # Diagnostico de calibracion del umbral (el 500K es propuesta inicial, no calibrada
    # contra historico): cuantos valores del universo quedan bajo el umbral, y cuales de
    # ellos eran candidatos que habrian llegado a valid[] de no ser por este filtro.
    _bajo_umbral = [v for v in values if v.get('vol_media_30d') is not None
                    and v['vol_media_30d'] < UMBRAL_VOL_MINIMO]
    if _bajo_umbral:
        print(f'  Filtro volumen minimo {UMBRAL_VOL_MINIMO//1000}K: {len(_bajo_umbral)} de '
              f'{len(values)} valores del universo por debajo del umbral')
    # P38 — el diagnostico usaba _rvol_ok/_vol_minimo_ok, que ya no existen (absorbidas por
    # cumple_calidad). Se reconstruye: candidatos que pasarian TODO salvo el volumen minimo.
    def _solo_falla_volumen(v):
        vm = v.get('vol_media_30d')
        if vm is None or vm >= UMBRAL_VOL_MINIMO:
            return False          # no lo excluye el volumen
        copia = dict(v); copia['vol_media_30d'] = None   # neutraliza solo ese criterio
        return cumple_calidad(copia)
    _excluidos_vol = [v for v in values
                      if v.get('entry_range',{}).get('entry_lo')
                      and (v.get('entry_range',{}).get('rr') or 0)>=2.0
                      and _solo_falla_volumen(v)]
    if _excluidos_vol:
        print('  Candidatos excluidos por volumen minimo: ' +
              ', '.join(f'{v["ticker"]}({round(v["vol_media_30d"]/1000)}K)' for v in _excluidos_vol))
    # PUNTO 3 — ranking calculado en Python con tope R/B=5, no delegado a Claude.
    # Formula: SCT 40% + R/B normalizado (tope 5) 20% + fuerza sectorial del grupo 20% + CMF 20%
    # PUNTO 31 (01/08/2026) — score sectorial por % sobre MM200 (fortaleza estructural real)
    # en lugar de la media de scores de los miembros del grupo.
    # ANTES: sector_n = media de scores de los miembros / 100
    #   → bucle de retroalimentacion: un lider en grupo mediocre queda penalizado por sus
    #     companeros (detectado por revision externa de Deepseek, 28/07/2026).
    # AHORA: sector_n = % de valores del sector sobre su MM200 / 100
    #   → mide la fortaleza estructural del sector de forma independiente del scoring
    #     individual; un sector donde el 80% de sus valores cotizan sobre la MM200 tiene
    #     viento de cola real, independientemente de cuantos de ellos sean candidatos hoy.
    group_mm200_map = {g['group']: g.get('pct_sobre_mm200', 0) for g in groups}
    # P63 (12/08/2026): en grupos por debajo de MIN_MIEMBROS_TEMA el % sobre MM200 solo puede
    # ser 0 o 100 (con n=1, el propio ticker), asi que el componente sectorial —20 de los 100
    # puntos del ranking— se convertia en un todo-o-nada regalado por el tamaño del grupo y no
    # por fortaleza sectorial real. Se sustituye por la MEDIANA de los grupos que si son
    # comparables: neutro, y calculado con los datos del propio dia.
    _mm200_comparables = [g.get('pct_sobre_mm200', 0) for g in groups
                          if g.get('tema_comparable') and g.get('pct_sobre_mm200') is not None]
    _mm200_neutro = round(float(np.median(_mm200_comparables)), 1) if _mm200_comparables else 50.0
    _grupos_no_comparables = {g['group'] for g in groups if not g.get('tema_comparable')}
    for v in valid:
        sct_n = (v.get('sct') or 0) / 100.0
        rb_n = min((v.get('entry_range',{}).get('rr') or 0), 5.0) / 5.0
        if v['group'] in _grupos_no_comparables:
            sector_n = _mm200_neutro / 100.0   # P63: neutro, no premio por grupo pequeño
            v['sector_neutralizado'] = True
        else:
            sector_n = group_mm200_map.get(v['group'], 0) / 100.0
        cmf_raw = v.get('cmf')
        cmf_n = min(max((cmf_raw if cmf_raw is not None else 0) / 0.3, 0), 1)  # 0.3 CMF ~ tope practico
        v['ranking'] = round((sct_n*0.40 + rb_n*0.20 + sector_n*0.20 + cmf_n*0.20) * 100, 1)
        # PUNTO 19 — desglose del ranking (13/07/2026): los cuatro componentes ya ponderados
        # (suman exactamente el ranking) se persisten y se emiten en el summary para que el
        # informe sea auditable sin abrir el codigo — respuesta a la critica externa de
        # "ranking caja negra". Pesos fijos: SCT 40 + R/B 20 + sector 20 + CMF 20 (sobre 100).
        v['ranking_desglose'] = {'sct': round(sct_n*40, 1), 'rb': round(rb_n*20, 1),
                                 'sector': round(sector_n*20, 1), 'cmf': round(cmf_n*20, 1)}
    valid = sorted(valid, key=lambda x: x.get('ranking', 0), reverse=True)

    # P64 (12/08/2026) — FASE DE RED, extraida a recolectar_datos_candidatos(). A partir de
    # aqui construir_prompt no toca la red: recibe (o pide una vez) un dict con todo lo
    # externo y solo ENSAMBLA. Ver la docstring de esa funcion para el porque.
    if externos is None:
        externos = recolectar_datos_candidatos(valid, fundamentales)
    valid = externos['valid']
    descartados_por_earnings = externos['descartados_por_earnings']
    summary += ensamblar_bloques_externos(externos, valid)

    summary+='\nSETUPS VALIDOS CON ANALISIS FUNDAMENTAL (usa EXACTAMENTE estos niveles, RIESGO y RANKING ya calculados — no los recalcules):\n'
    summary += bloque_setups(valid, fundamentales)
    if descartados_por_earnings:
        summary+='\nDESCARTADOS POR EARNINGS INMINENTE (no incluir en ENTRADAS, se puede mencionar brevemente si aporta contexto):\n'
        for v in descartados_por_earnings:
            summary+=f'- {v["ticker"]}: earnings en {v["dias_earnings"]} dias ({v["earnings_date"]})\n'
    summary += bloque_extendidos(values)
    summary+=f'\nFuertes:{len(strong)} Emergentes:{len(emerging)} Sin momentum:{len(weak)} Setups:{len(valid)}\n'
    # PUNTO 10 + NUEVO (27/06) — alertas de seguimiento: CMF deteriorado (señal blanda) y/o
    # Supertrend ya bajista (señal dura: el precio ya cruzo el nivel dinamico de stop)
    summary += bloque_seguimiento(data)

    aviso='ATENCION: SPY bajo MM200. Mercado bajista. ' if not spy_ok else ''
    prompt = (
        'Analista tecnico momentum. Informe ejecutivo espanol directo. ' + aviso +
        'Estructura: 1.MERCADO 2.TEMAS PRIORITARIOS 3.ENTRADAS(niveles exactos) 4.EXTENDIDOS 5.CONCLUSION. La CONCLUSION es OBLIGATORIO que termine con un parrafo separado titulado exactamente **RIESGOS DEL ESCENARIO** que liste en 3-4 puntos concisos que condiciones invalidarian la tesis alcista actual: perdida de MM200 por SPY, perdida de liderazgo sectorial, VIX elevado, deterioro de amplitud de mercado, resultados empresariales negativos proximos. Este parrafo NO es opcional. '
        'PUNTO 8 — AMPLITUD DE MERCADO: si el bloque AMPLITUD DE MERCADO aparece en los datos, usalo en la seccion MERCADO para matizar el regimen general (SPY>MM200 es una sola variable; la amplitud dice si ese movimiento esta respaldado por la mayoria de valores o solo por unos pocos). Interpreta los datos asi: % sobre MM50/MM200 por debajo del 50% mientras el SPY esta alcista indica un "rally estrecho" (pocos valores liderando, mercado fragil bajo la superficie); % por encima del 60-70% indica amplitud sana. Si nuevos minimos de 52 semanas superan a los nuevos maximos, es una señal de deterioro interno aunque el indice general suba. Esta informacion es CONTEXTO, no debe usarse para invalidar ni recalcular el RIESGO de los setups individuales (que depende solo de SCT/RSI/CMF/ADX del valor) — su unico rol es matizar la lectura del regimen de mercado en la seccion 1 y, si aplica, dar contenido especifico y verificable al punto de "deterioro de amplitud de mercado" en RIESGOS DEL ESCENARIO en lugar de mencionarlo de forma generica. '
        'ALERTAS DE REGIMEN MACRO: las lineas "ALERTA DE REGIMEN" solo aparecen en los datos cuando un umbral de VELOCIDAD se ha disparado (US30Y: ±25 pb en 20 sesiones o cruce del 5.00%; WTI: ±10% en 20 sesiones; CREDITO: ratio HYG/IEF cayendo mas de un 2% en 20 sesiones, señal de diferenciales high-yield tensionandose — el sensor adelantado de aversion al riesgo mas fiable de este bloque, porque el credito suele estresarse antes que la renta variable). Si aparecen, integralas en la seccion 1 (MERCADO) y en RIESGOS DEL ESCENARIO aplicando el mapa de transmision sectorial: tipos largos subiendo con fuerza presionan los sectores de duracion larga (biotech sin beneficios, tecnologia de multiplo alto) y favorecen relativamente a las financieras; petroleo subiendo con fuerza favorece a energia y presiona a transporte, aerolineas y consumo discrecional; credito tensionandose da mas valor relativo a los setups de calidad y defensivos (utilities, telecos, salud, basicos) frente a los de beta alta. Usa el mapa para contextualizar los setups del dia cuando la alerta afecte a sus sectores, sin forzarlo cuando no aplique. Cada linea trae DOS ventanas: el cambio en 20 sesiones (el regimen) y el cambio de hoy (la tactica del dia) — pueden divergir (ej.: crudo cayendo un 20% en el mes pero repuntando hoy por geopolitica) y en ese caso debes reflejar ambas, con el regimen como lectura principal y el movimiento del dia como matiz. Estas alertas describen el REGIMEN actual, no predicen la proxima sesion: nunca las conviertas en pronosticos. Si NO aparece ninguna linea de ALERTA DE REGIMEN, tienes PROHIBIDO mencionar el bono a 30 años, el petroleo o el credito high-yield en el informe — ni siquiera para decir que estan estables o sin cambios: el silencio es la señal de normalidad y el comentario macro de relleno esta prohibido. '
        'ESTRUCTURA DE INDICES (MM50): la linea "ESTRUCTURA DE INDICES" solo aparece cuando el SPY o el QQQ han cruzado su media movil de 50 sesiones recientemente o estan testandola (±1%). Es contexto TACTICO de corto plazo, distinto del regimen (que sigue siendo la MM200): un indice rebotando en su MM50 con la MM200 alcista es una correccion normal dentro de tendencia; un indice bajo su MM50 durante muchas sesiones indica que el tramo tactico es debil aunque el regimen aguante. Integralo en la seccion 1 (MERCADO) en una o dos frases, y conectalo con los setups del dia cuando aplique (ej.: si los setups son pullbacks a MM50 individuales el mismo dia que el indice testa la suya, es una correccion sincronizada de mercado — nombrable como contexto, ni mejor ni peor per se). Si la linea NO aparece, tienes PROHIBIDO mencionar las medias de 50 sesiones de los indices — el silencio es normalidad. Nunca lo conviertas en pronostico. '
        'PUNTO 10 — SEGUIMIENTO CMF + SUPERTREND: si aparece el bloque "SEGUIMIENTO — ALERTAS" en los datos, anyadelo como una seccion separada ENTRE la seccion 4 (EXTENDIDOS) y la CONCLUSION, titulada exactamente "SEGUIMIENTO DE POSICIONES ABIERTAS". Distingue claramente DOS niveles de urgencia, nunca los trates igual: (A) si un ticker lleva la marca "CMF NEGATIVO N SESIONES CONSECUTIVAS", es una señal blanda pero ya CONFIRMADA: no es un mal dia aislado, el flujo lleva N sesiones seguidas en terreno vendedor (distribucion sostenida) — la regla es valorar reducir el tamaño de la posicion a la mitad hasta que el CMF recupere terreno positivo. Cuanto mayor sea N, mas asentada esta la distribucion, pero la REGLA OPERATIVA ES LA MISMA para todas las alertas blandas independientemente de N: valorar reducir al 50%. NUNCA degrades las rachas cortas (3-5 sesiones) a "vigilar sin reducir" ni inventes niveles intermedios de urgencia dentro de las blandas — si una alerta aparece en los datos es porque ya cumple el minimo de confirmacion del sistema, y las rachas cortas en posiciones muy ganadoras son precisamente los giros de flujo mas frescos y accionables, no los menos urgentes. (B) si un ticker lleva la marca SUPERTREND BAJISTA, es una señal dura y mas urgente: el precio ya ha cruzado por debajo del nivel dinamico de Supertrend (indicado en la marca), lo que significa que la estructura de tendencia que sostenia el setup ya se ha roto — no es "vigilar", es una señal de salida total de la posicion, mas decisiva que la alerta CMF. Las señales son EXCLUYENTES por diseño: la dura absorbe a la blanda (nunca veras ambas en el mismo ticker, y nunca debes sugerir a la vez "reducir 50%" y "salida total" para un mismo valor). Ademas, AMBOS tipos de alerta corresponden UNICAMENTE a condiciones surgidas DESPUES de la entrada del setup: las rupturas Supertrend son posteriores a la entrada, y las rachas de CMF negativo empezaron tras la entrada — los setups que ya nacieron con el Supertrend bajista o con el CMF en negativo no generan alerta (no tiene sentido "salir" o "reducir" por una condicion que ya existia al entrar), asi que toda alerta que veas es un deterioro genuinamente nuevo sobre la posicion. Dentro del caso (B) los datos distinguen DOS variantes que NUNCA debes redactar igual: "SUPERTREND BAJISTA NUEVO — ruptura de esta sesion" significa que la direccion ha cambiado a bajista precisamente en la sesion mas reciente — es el caso mas urgente de actuar y puedes redactarlo como ruptura del dia ("ha cruzado por debajo..."). "SUPERTREND BAJISTA PERSISTENTE desde hace N sesiones" significa que la ruptura NO es de hoy: lleva N sesiones activa y el precio simplemente no ha recuperado el nivel necesario para revertirla — NUNCA lo redactes como si la ruptura acabara de producirse; indica explicitamente desde cuando esta rota la tendencia ("Supertrend bajista desde hace N sesiones") y ajusta el framing: la decision de salida ya deberia haberse tomado en su momento, la mencion actual es un recordatorio de que la estructura sigue rota, no una alerta urgente de esta sesion. Un dia con subida fuerte puede coexistir con Supertrend bajista persistente si el cierre no ha cruzado el nivel indicado — no lo presentes como contradiccion, es el diseño del indicador. Si hay mas de 10 alertas en total, agrupa: menciona primero los casos con SUPERTREND BAJISTA (los mas urgentes, hasta 5), luego los 3-5 casos de CMF mas relevantes por retorno, y resume el resto en una frase tipo "otros N valores con distribucion confirmada — misma regla de reduccion al 50%". Cita siempre el retorno actual (ret_pct). Si no hay alertas, no incluyas esta seccion. Seccion operativa, no analitica. '
        'Escribe cada seccion de forma que sea comprensible tanto para un analista tecnico como para un inversor adulto con conocimientos generales de bolsa pero sin experiencia en analisis tecnico. '
        'No uses parrafos separados ni marcadores especiales para las explicaciones: integra el contexto y el significado directamente en el texto de cada seccion. '
        'Cuando menciones un indicador tecnico (RSI, ADX, SCT, CMF, MM50, etc.) explica brevemente en la misma frase que implica ese valor concreto para la decision. '
        'Para cada setup de entrada, integra el contexto fundamental (PER, margen, ROE, crecimiento) con el tecnico: explica si la valoracion apoya o cuestiona la entrada tecnica. '
        'Si el PER forward es elevado (>40x) pero el crecimiento de ingresos es alto (>30%), mencionalo como valoracion de crecimiento que requiere gestion de riesgo estricta. '
        'Si el R/B de un setup es exactamente 1:5.0, indica brevemente que el objetivo esta acotado al tope maximo del sistema (5R) y que la distancia tecnica real hasta el maximo de 52 semanas es mayor — el techo real sigue siendo informacion util aunque no sea el objetivo del trade. '
        'EV/EBITDA<10x=valoracion razonable, >20x=cara. EPSgrowth>20% justifica multiples elevados. Si EPS forward>EPS trailing la empresa acelera beneficios. '
        'Si CMF<0 y ADX>25 advierte de divergencia bajista (presion vendedora mientras el precio sube) — el scanner ya ha penalizado el SCT pero Claude debe mencionarlo explicitamente sin atribuirlo a instituciones especificas. '
        'Si el margen neto es negativo, mencionalo brevemente pero sin descartar el setup si el momentum es solido. '
        'IMPORTANTE sobre los datos EPS: el campo "EPS:X->Y" compara el beneficio por accion de los ultimos 12 meses reportados (trailing) con la estimacion de consenso para los proximos 12 meses (forward) — es una ventana anual completa. El campo "EPSgrowth:Z%" es un dato DISTINTO: el crecimiento interanual del ultimo trimestre reportado frente al mismo trimestre del año anterior. Pueden mostrar magnitudes muy diferentes sin ser contradictorios entre si (ej. un trimestre reciente con crecimiento modesto puede coexistir con una proyeccion de aceleracion mayor para el año completo si el consenso espera mejora en los proximos trimestres). Cuando menciones ambos datos, acláralo brevemente para que no parezcan inconsistentes — nunca presentes el EPSgrowth trimestral como si fuera el mismo concepto que el salto trailing->forward, ni viceversa. '
        'Distingue siempre el tipo de setup tal y como aparece en el campo correspondiente: "Ruptura activa" es una ruptura ya confirmada con precio actual dentro del rango de entrada; "Ruptura pendiente" significa que el precio actual esta por DEBAJO del rango de entrada y hay que esperar a que el valor suba hasta esa zona antes de poder ejecutar — explicalo asi explicitamente, nunca lo trates como una entrada inmediata; "Pullback MM50"/"Pullback MM200" es una correccion hacia una media movil que actua como soporte. No mezcles estos conceptos. '
        'PUNTO 6/14 — EXTENDIDOS, DOS BLOQUES DISTINTOS: los datos separan "EXTENDIDOS — VIGILANCIA ACTIVA" (candidatos reales, con trigger normal "vigilar cuando...") de "EXTENDIDOS — DESCARTADOS POR DISTANCIA EXTREMA" (lista comprimida de tickers con su % sobre MM50, ya evaluados como no viables). Trata cada bloque de forma distinta en la redaccion: para VIGILANCIA ACTIVA, escribe un parrafo breve por candidato explicando el trigger concreto, igual que con un setup normal. Para DESCARTADOS, NO le dediques un parrafo por ticker ni redactes su trigger — menciona el bloque en una sola frase de contexto (ej. "ademas, X, Y y Z cotizan muy por encima de su MM50, demasiado extendidos para una reentrada realista") y, si aporta valor narrativo, usalo para matizar el liderazgo sectorial en la seccion de TEMAS PRIORITARIOS (ej. si varios pertenecen al mismo sector, señala que ese sector esta sobre-extendido a corto plazo). Nunca presentes un ticker de este bloque como oportunidad pendiente de vigilancia. '
        'PUNTO 7 — EARNINGS: hay DOS fuentes distintas de informacion sobre earnings, y NO deben mezclarse en tono ni redaccion. '
        '(A) AVISO ESTRUCTURADO: si un setup incluye "EARNINGS EN X DIAS — AVISO OBLIGATORIO" (viene de earnings_accion="avisar" en los datos, ventana 10-21 dias = evento genuinamente inminente), debes mencionarlo explicitamente con una advertencia clara y directa: el informe de resultados puede provocar un gap de precio que el stop tecnico no protege (el stop se ejecuta al precio de apertura siguiente, no al nivel fijado). Sugiere reducir el tamaño de posicion o esperar a que pase el informe si el horizonte del trade puede solaparse con esa fecha. Indica siempre los dias exactos y la fecha. '
        '(B) MENCION POR NOTICIAS: si un setup NO incluye el aviso estructurado (earnings_accion distinto de "avisar") pero el campo NOTICIAS trae un titular sobre resultados/earnings preview, SI puedes mencionarlo, pero unicamente como contexto informativo suave, nunca con el lenguaje de riesgo de gap ni "el stop no protege" — esa redaccion esta reservada en exclusiva al caso (A). Para el caso (B) usa un tono tipo "hay cobertura de prensa sobre el proximo informe de resultados, aun a varias semanas vista" o equivalente, sin instar a reducir tamaño ni a esperar al informe, ya que el propio sistema ya evaluo que el evento no esta dentro de la ventana de riesgo inminente. Si tienes el dato dias_earnings disponible, puedes citarlo para contextualizar cuanto falta ("la fecha de resultados aun queda a unas N semanas"). '
        'En cualquier caso que no sea (A) ni (B) — es decir, si no hay aviso estructurado NI titular de noticias sobre earnings — NO menciones earnings, resultados, ni ninguna variante tipo "vigilar fechas de resultados" en ese setup bajo ningun concepto, ni siquiera como cautela generica de buena practica: citar earnings sin que ningun dato lo respalde es tan grave como inventar un motivo de RIESGO falso, y esta PROHIBIDO. '
        'Si aparecen tickers en la lista "DESCARTADOS POR EARNINGS INMINENTE", debes mencionarlos brevemente en EXTENDIDOS o en una nota aparte como candidatos tecnicamente validos pero descartados por proximidad de resultados, indicando SIEMPRE los dias y la fecha exacta de earnings tal y como aparecen en los datos (ej. "earnings en 3 dias, el 2026-06-20") — nunca los menciones sin esa fecha concreta, y nunca les asignes niveles de entrada. Si un setup tiene noticias recientes sin relacion con earnings (campo NOTICIAS), incorpora el titular mas relevante de forma natural en el texto si aporta contexto util (catalizador, riesgo, confirmacion), sin listarlas todas mecanicamente. '
        'Cada setup valido incluye un objetivo PARCIAL (resistencia inmediata, primera toma de beneficios parcial) y un objetivo FINAL (techo tecnico acotado a R/B 1:5 maximo). Menciona ambos: el parcial como punto donde valorar asegurar parte de la posicion, el final como objetivo de cierre completo. '
        'Ejemplos de tono correcto: '
        '"IONQ cotiza con RSI en 65, zona de momentum optimo sin sobrecompra, y ADX en 42, lo que confirma que la tendencia alcista tiene solidez suficiente para nuevas entradas." '
        '"El SCT de 78 sobre 100 indica que varios indicadores tecnicos apuntan en la misma direccion, lo que reduce el riesgo de una senal falsa." '
        'Para interpretar indicadores: RSI 55-70=momentum optimo, RSI>70=sobrecomprado con cautela; '
        'ADX>40=tendencia muy fuerte, ADX>25=tendencia confirmada, ADX<20=mercado lateral evitar (ya excluido del scanner); '
        'CMF>0.1=presion compradora sostenida (no afirmes que son instituciones, di acumulacion o presion compradora), CMF<0=presion vendedora; SCT>70=confirmacion tecnica alta; '
        'SQUEEZE=compresion de volatilidad previa a ruptura potente. '
        'RVOL (volumen relativo) = volumen actual / promedio de los ultimos 30 dias: RVOL>2.0=volumen excepcional que confirma participacion institucional real; RVOL 1.2-2.0=respaldo de volumen solido (umbral minimo exigido en rupturas); RVOL<1.2=volumen insuficiente (este caso ya no llega a ENTRADAS en rupturas, pero puede aparecer en pullbacks donde el filtro no aplica). Menciona el RVOL en el texto de cada setup de forma natural cuando aporte informacion util (ej. "el volumen multiplica por 1.8 su media de 30 dias, confirmando la participacion compradora") pero no lo cites mecanicamente en todos los casos. '
        'PUNTO 11 — VOLUME PROFILE: si un setup incluye POC/VAH/VAL, son niveles de precio donde se ha concentrado el volumen de las ultimas 60 sesiones (aproximacion sobre datos diarios, no tick data real) y funcionan como soporte/resistencia institucional — zonas donde mas dinero ha cambiado de manos, no un indicador de momentum. POC es el nivel con mayor volumen acumulado; VAH/VAL acotan el rango donde se concentra el 70% del volumen. Sobre la relacion entre el VAL y la zona de entrada, USA LITERALMENTE lo que diga la nota y no la reformules: si dice "VAL DENTRO de la zona de entrada", puedes afirmar que se entra sobre el limite inferior del area de valor, lo que refuerza la calidad del pullback como soporte real con respaldo de volumen y no solo tecnico (medias moviles); si dice "VAL X% POR DEBAJO del limite inferior de la entrada", esta PROHIBIDO escribir que la entrada coincide con el VAL — el VAL queda por debajo de toda la zona y solo es un soporte de volumen cercano bajo ella. Comprueba siempre el dato: si el VAL es menor que el extremo inferior del rango de entrada, no hay coincidencia. Si no hay coincidencia, puedes citar el POC como referencia de resistencia o soporte cercano si esta a una distancia relevante del precio actual, pero sin forzar la mencion en cada setup. Si aparece la nota "VAL dentro del rango de riesgo, demasiado cerca del stop — NO citarlo como soporte", significa que el VAL coincide con la zona de entrada pero queda a menos de la mitad del recorrido entrada->stop por encima del stop: si el precio baja hasta el VAL ya ha consumido la mayor parte del margen hacia el stop, asi que ese nivel NO protege la posicion. En ese caso esta PROHIBIDO citar el VAL como soporte o como argumento de calidad del pullback; como mucho puedes mencionarlo como referencia de volumen neutral, o directamente omitirlo. '
        'CONSENSO DE ANALISTAS Y PRECIO OBJETIVO: si el campo FUND incluye "Consenso:" y/o "TargetMedio:", son datos de Wall Street (calificacion media y precio objetivo a 12 meses de los analistas que cubren el valor), no un dato tecnico ni del sistema. Menciona el consenso como contexto adicional cuando refuerce o contradiga la tesis tecnica (ej. consenso "buy" con upside relevante respalda la entrada; un consenso tibio o un upside ya agotado por el rally reciente merece mencionarse como matiz de cautela), pero NUNCA confundas el TargetMedio de los analistas con el objetivo tecnico del sistema (target_parcial/target_final) — son cosas distintas, calculadas de forma distinta, y deben presentarse siempre como dos referencias separadas, no intercambiables. Si el numero de analistas (n_opiniones) es bajo (1-3), matiza que el consenso tiene poco respaldo estadistico. No fuerces la mencion si no aporta nada relevante al caso concreto. '
        'DATOS DE CATALIZADOR Y CALIDAD (si aparecen en FUND): RevisionesAnalistas90d, UltimaRevision y SorpresaEPS son contexto de CATALIZADOR — explican POR QUE puede estar entrando dinero ahora (analistas mejorando estimaciones recientemente, ultimo trimestre batiendo expectativas), no son datos de valoracion estatica: usalos igual que las noticias, integrados en la narrativa del setup cuando refuercen o cuestionen el momentum (varias subidas de calificacion recientes respaldan la entrada; varias bajadas recientes con momentum tecnico alcista merecen mencionarse como divergencia de opinion). Una sorpresa de EPS claramente positiva (>+5%) es catalizador favorable; una negativa reciente es matiz de cautela aunque el precio haya aguantado. EPStrim es la serie de EPS reales de los ultimos trimestres reportados (hasta 5, del mas antiguo al mas reciente): usala solo para describir la tendencia (aceleracion, estancamiento, deterioro), sin recitar la serie completa en el texto. MargenBruto, MargenOperativo, ROIC y FCFgrowth son calidad fundamental adicional: ROIC>15% indica uso eficiente del capital, FCFgrowth positivo indica generacion de caja creciente — trata estos campos igual que el resto de fundamentales (PER, ROE...), como contexto que apoya o cuestiona la entrada tecnica, sin darles mas peso que al momentum. Estos campos pueden faltar en algunos setups (cobertura parcial del proveedor): si no estan, no los menciones ni especules sobre ellos. '
        'PUNTO 12-13 — OBJETIVO CON TECHO NO ESTANDAR: en la inmensa mayoria de setups el objetivo final se deriva del maximo de 52 semanas (techo estandar, no necesita explicacion en el texto). Si el setup incluye una nota "OBJETIVO:", significa que el valor sufrio una correccion superior al 20% desde su maximo de 52 semanas, lo que invalida ese maximo como techo realista a corto plazo — el sistema ha sustituido el techo por un retroceso de Fibonacci de esa caida (61.8% o 76.4%, el que deje margen suficiente) o, si ninguno de los dos da margen, por una proyeccion de rango medio (ATR). Cuando esto ocurra, menciona brevemente en el parrafo del setup que el objetivo es mas conservador de lo habitual precisamente porque el valor no ha recuperado el terreno perdido tras esa correccion, en vez de asumir que el objetivo viene del maximo historico sin matizar. No necesitas explicar la formula exacta (61.8 vs 76.4 vs ATR), basta con transmitir que el objetivo esta acotado por una correccion previa relevante. '
        'FORMATO: en la seccion ENTRADAS, separa cada recomendacion con un salto de linea — una recomendacion por parrafo independiente. Escribe siempre el ticker con el simbolo dolar delante: $AZTA, $TECH, $CDW, etc. '
        'Al final de la seccion ENTRADAS incluye una tabla comparativa con columnas: TICKER | SETUP | R/B | SCT | RIESGO | RANKING. RIESGO y RANKING vienen YA CALCULADOS en los datos (campos RIESGO: y RANKING: de cada setup) — usalos exactamente como aparecen, NO los recalcules ni apliques ninguna formula propia sobre ellos. Cada setup incluye ademas "motivo exacto" entre parentesis junto al RIESGO: esa es la UNICA causa real que la formula evaluo para clasificarlo como BAJO o MEDIO. Cuando expliques por que un setup tiene ese RIESGO, usa literalmente ese motivo (reformulado en prosa natural, no copiado palabra por palabra) — NUNCA inventes ni infieras una causa distinta aunque suene plausible (por ejemplo, no atribuyas el RIESGO al tipo de setup —ruptura pendiente, pullback, etc.— salvo que el motivo exacto mencione ADX o RSI relacionados con eso; el tipo de setup y el RIESGO son cosas independientes). '
        'PUNTO 17 — CORRELACION ENTRE CANDIDATOS: si aparece el bloque CORRELACION ENTRE CANDIDATOS, usalo para evaluar la CARTERA como conjunto, no cada idea aislada. Correlacion >=0.70 entre dos candidatos significa que en la practica son una sola apuesta con dos nombres: si ambos se toman, la diversificacion real es menor de lo que aparenta y una misma sorpresa macro (tipos, growth, sector) golpea a la vez a ambos. Las betas indican la sensibilidad de cada candidato al mercado: beta agregada alta = cartera que amplifica al SPY. Integra esta lectura en un parrafo breve dentro de la CONCLUSION (o en RIESGOS DEL ESCENARIO si hay AVISO CONCENTRACION), nombrando los pares concretos y su correlacion. Si hay CONCENTRACION SECTORIAL (dos o mas candidatos del mismo tema), menciona que comparten motor sectorial. Esta informacion es CONTEXTO DE CARTERA: NO invalida ni recalcula el RIESGO ni el RANKING individual de ningun setup, y NO debe usarse para descartar candidatos — solo para advertir sobre tomarlos simultaneamente y sugerir, si aplica, dimensionar posiciones considerando la correlacion. Si el bloque no aparece, no menciones correlaciones. '
        'PUNTO 18 — PULLBACK EN SOBRECOMPRA: si un setup lleva la marca AVISO: PULLBACK CON RSI EN SOBRECOMPRA, significa que el valor cotiza cerca de su media movil de referencia PERO su RSI sigue >=70: el retroceso ha sido tan superficial que no ha aliviado la sobrecompra, por lo que hablar de "pullback de calidad" no es convincente. Redactalo con ese matiz explicito (retroceso minimo, momentum aun sobrecalentado, mayor probabilidad de que la correccion no haya terminado) y evita presentarlo como una entrada de libro en soporte. La marca NO cambia el RIESGO ni el RANKING calculados — es un matiz de calidad del setup que el lector debe conocer. Si ningun setup lleva la marca, no menciones este concepto. '
        'PUNTO 19/31 — DESGLOSE DEL RANKING: cada setup incluye junto al RANKING su desglose entre parentesis (SCT x/40 + R/B x/20 + sector x/20 + CMF x/20; los cuatro componentes suman exactamente el ranking). El componente SECTOR (hasta 20 puntos) refleja desde el 01/08/2026 el % de valores del sector que cotizan sobre su MM200: un sector donde el 80% de sus miembros estan sobre la MM200 puntua 16/20 (0.80 × 20); el componente es una medida de fortaleza estructural del sector, no de calidad tecnica individual. Usalo cuando compares candidatos entre si: un setup con sector x/20 alto tiene el viento del sector de cola; uno con sector bajo compite contra corriente. No recites el desglose completo de cada setup — usalo para fundamentar comparaciones concretas. NUNCA inventes pesos ni componentes distintos de los cuatro dados. '
        'PUNTO 20 — AMPLITUD TACTICA (MM20 y McClellan): el % sobre MM20 es la version rapida de la amplitud — reacciona en dias, no en semanas: si es claramente inferior al % sobre MM50, la participacion de corto plazo se esta enfriando aunque la estructura de fondo aguante; si lo supera con claridad, hay reaceleracion tactica. El McClellan Oscillator (ratio-adjusted) mide el MOMENTO de la amplitud: positivo = el empuje neto comprador domina, negativo = domina el vendedor, y el cruce de 0 marca el giro; lecturas mas alla de aproximadamente +/-50 indican empuje amplio posiblemente sobreextendido a corto plazo (no es señal operativa, es contexto). Integra ambos en la lectura de AMPLITUD DE MERCADO de la seccion 1 (una o dos frases), coherente con el resto de metricas de amplitud: divergencias entre la amplitud rapida (MM20/McClellan) y la lenta (MM50/MM200) son el matiz mas valioso que puedes señalar. NO recalcules ni modules el RIESGO de setups individuales con estos datos. '
        'PUNTO 23 — EVENTOS MACRO PROGRAMADOS: si aparece el bloque EVENTOS MACRO PROGRAMADOS PROXIMOS, tratalo como la version a escala de mercado del aviso de earnings: son citas binarias (FOMC, IPC, NFP) capaces de provocar gaps de apertura que NINGUN stop tecnico protege, y afectan a todas las posiciones a la vez, no a un ticker. Menciona el evento y su fecha en la seccion de contexto de mercado y añade una frase de prudencia operativa sobre las ENTRADAS NUEVAS (considerar esperar al evento o reducir tamaño si el horizonte del trade lo cruza), con mas enfasis cuanto mas cerca este (HOY o 1 dia habil = maxima cautela). NO recalcules el RIESGO ni el RANKING de setups individuales, NO especules sobre el resultado del evento ni sobre la direccion del mercado tras el dato, y si el bloque no aparece, no menciones eventos macro programados. '
        'PUNTO 33 — TIEMPO ESTIMADO AL OBJETIVO: cada setup puede incluir la línea "Tiempo estimado al objetivo: X-Y sesiones". Es un ORDEN DE MAGNITUD, no una predicción: X supone avance en línea recta (suelo teórico, casi nunca se da) e Y supone un paseo aleatorio con retrocesos. El rango es ancho a propósito. Cítala UNA sola vez por setup, en el párrafo de niveles, SIEMPRE como rango completo y SIEMPRE con la advertencia de que es orientativa. NUNCA cites solo el extremo bajo ni presentes la cifra como plazo esperado. Úsala para contextualizar el R/B a escala (semanas vs meses). Si el dato no aparece, no lo menciones. '
        'Aviso final obligatorio: "Este analisis no constituye asesoramiento financiero."\n\n' + summary
    )
    return prompt

def generate_analysis(data, anthropic_key):
    import anthropic as ant
    client=ant.Anthropic(api_key=anthropic_key.strip())
    prompt = construir_prompt(data)
    # ARREGLO (20/08/2026): httpx se importaba aqui SOLO para construir el timeout, y el
    # 20/08 la Action fallo con "No module named 'httpx'" en los tres intentos, dejando el
    # informe sin generar. La conexion con Anthropic funcionaba ("modelos disponibles: 10"),
    # asi que el SDK estaba bien: lo que desaparecio fue httpx como modulo importable, muy
    # probablemente por un cambio de version del SDK que ya no lo arrastra como dependencia
    # visible. El workflow instala `yfinance pandas requests beautifulsoup4 anthropic pytz`,
    # sin httpx explicito, asi que dependia de que anthropic lo trajera de rebote.
    #
    # Un DETALLE DE CONFIGURACION no puede tumbar la generacion entera: el SDK acepta un
    # float como timeout, que cubre lo esencial. Se pierde solo el timeout de conexion
    # diferenciado (30s), y eso se dice en la traza en vez de fallar en silencio.
    try:
        import httpx
        _timeout = httpx.Timeout(120.0, connect=30.0)
    except Exception as _e:
        _traza('anthropic/httpx-ausente', _e)
        _timeout = 120.0
    msg=client.messages.create(model='claude-opus-4-8',max_tokens=MAX_TOKENS_INFORME,messages=[{'role':'user','content':prompt}], timeout=_timeout)
    texto = next(b.text for b in msg.content if hasattr(b, 'text'))
    if msg.stop_reason == 'max_tokens':
        print(f'  AVISO: el analisis se corto por limite de max_tokens (texto truncado, {len(texto)} caracteres generados). '
              f'Considera revisar si el prompt o el numero de setups ha crecido demasiado.')
        texto += '\n\n*(Aviso del sistema: este informe quedo incompleto por limite de longitud de la respuesta.)*'
    return texto

def get_github_file(filename, reintentos=3):
    """
    FIX CRITICO (08/07) — devuelve (contenido, sha) si el fichero existe; (None, None)
    SOLO si el fichero NO EXISTE (404); y ('__ERROR__', None) si la lectura fallo de
    forma persistente (rate limit, 5xx, red) tras los reintentos.
    Antes, cualquier fallo devolvia (None, None), indistinguible de "fichero inexistente":
    el llamador reconstruia el historico desde cero y lo SUBIA, machacando los datos.
    Caso real 08/07 07:34: un fallo transitorio de lectura borro los 30 dias de
    setups_history.json (1.699 setups, recuperados del historial de git). Los llamadores
    DEBEN comprobar el centinela '__ERROR__' y abortar su actualizacion si aparece.
    """
    url=f'https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{filename}'
    for intento in range(reintentos):
        try:
            r=requests.get(url,headers={'Authorization':f'token {GITHUB_TOKEN}'},timeout=20)
            if r.status_code==200:
                _j = r.json()
                _txt = base64.b64decode(_j.get('content') or '').decode('utf-8')
                if not _txt.strip():
                    # P80 (01/09/2026) — LIMITE DE 1 MB DE LA CONTENTS API.
                    # La Contents API solo devuelve `content` en base64 hasta 1 MB; por
                    # encima responde 200 con content vacio y encoding 'none'. Decodificar
                    # eso da cadena vacia y json.loads revienta con JSONDecodeError, que es
                    # justo lo que se vio el 28/08 y el 01/09 — y SOLO con data.json, porque
                    # es el unico fichero que ha cruzado el megabyte (1.697.502 bytes).
                    #
                    # No era un fallo transitorio de red: era permanente y creciente, y dejaba
                    # INUTILIZADO el rescate del P55, que necesita leer data.json para saber
                    # si el informe anterior servia. La guarda del P78 evitaba el destrozo,
                    # pero el informe se perdia igual cada vez que fallaba la generacion.
                    #
                    # La Blobs API no tiene ese limite (hasta 100 MB) y acepta el mismo token.
                    _sha = _j.get('sha')
                    if _sha:
                        rb = requests.get(
                            f'https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/git/blobs/{_sha}',
                            headers={'Authorization': f'token {GITHUB_TOKEN}',
                                     'Accept': 'application/vnd.github.raw+json'}, timeout=40)
                        if rb.status_code == 200:
                            print(f'  {filename} leido por la Blobs API (supera 1 MB, la Contents API no lo sirve)')
                            return json.loads(rb.text), _sha
                        print(f'  ⚠️ Blobs API fallo para {filename} (HTTP {rb.status_code})')
                    raise ValueError(f'{filename}: contenido vacio en la Contents API y sin blob utilizable')
                return json.loads(_txt), _j['sha']
            if r.status_code==404:
                return None,None  # el fichero genuinamente no existe: empezar de cero es correcto
            print(f'  ⚠️ Lectura de {filename} fallo (HTTP {r.status_code}) — intento {intento+1}/{reintentos}')
        except Exception as e:
            print(f'  ⚠️ Lectura de {filename} fallo ({type(e).__name__}) — intento {intento+1}/{reintentos}')
        time.sleep(2*(intento+1))
    print(f'  🔴 Lectura de {filename} fallida tras {reintentos} intentos — el llamador debe abortar para no machacar datos')
    return '__ERROR__',None

def resoluciones_por_ticker(evaluaciones):
    """PUNTO 40 (05/08/2026) — setups RESUELTOS: stop o target alcanzados.

    Caso real que lo motiva: la noche del 04/08 el setup de CIFR del 25/07 toco stop
    (precio 20.37 contra stop 20.44, por siete centimos) y el informe NO lo dijo. Solo
    mostro la ocurrencia del 12/07, aun abierta, como "posicion abierta, -10.6%". El
    seguimiento solo listaba ALERTAS, y una alerta exige resultado=='abierto': un stop
    alcanzado no generaba ninguna linea. El dato estaba en data.json y en los
    checkpoints, pero no llegaba al usuario — y un stop es justo el evento que mas
    quiere ver.

    Nota sobre por que la fecha de setup "retrocedia" en el log: al resolverse la
    ocurrencia mas reciente, deja de alertar y dedup_alertas_por_ticker se queda con
    otra anterior que sigue viva. No era un fallo del dedupe; era la consecuencia
    visible de una resolucion invisible.

    Igual que las alertas persistentes, una resolucion se repite mientras el setup siga
    dentro del horizonte: no hay registro de "ya avisado", y prefiero recordarlo de mas
    a perderlo. Se conserva una por ticker (la del setup mas reciente resuelto) y se
    ordena por retorno ascendente, para que los stops encabecen la lista.
    """
    por_ticker = {}
    for e in (evaluaciones or []):
        if e.get('resultado') not in ('stop', 'target'):
            continue
        tk = e.get('ticker')
        if tk is None:
            continue
        previa = por_ticker.get(tk)
        if previa is None or str(e.get('fecha_setup', '')) > str(previa.get('fecha_setup', '')):
            por_ticker[tk] = e
    return sorted(por_ticker.values(), key=lambda x: (x.get('ret_pct') is None, x.get('ret_pct')))

def dedup_alertas_por_ticker(alertas):
    """
    NUEVO (10/07) — con la evaluacion diaria continua, un mismo ticker con setups en varias
    fechas (dentro del horizonte de 20 dias) genera varias evaluaciones el mismo dia. Para
    el informe y la consola se conserva UNA por ticker: la del setup MAS RECIENTE (la
    posicion mas viva). data.json conserva TODAS las evaluaciones para trazabilidad y
    estadistica — este deduplicado es solo de presentacion.
    """
    por_ticker = {}
    for a in alertas:
        tk = a.get('ticker')
        if tk is None: continue
        if tk not in por_ticker or str(a.get('fecha_setup', '')) > str(por_ticker[tk].get('fecha_setup', '')):
            por_ticker[tk] = a
    return sorted(por_ticker.values(), key=lambda x: str(x.get('fecha_setup', '')))

# ============================================================
# REGISTRO PERMANENTE DE SETUPS (PUNTO 75, 16/08/2026)
# El analisis de rendimiento del 16/08 choco con un muro: no existe ningun registro
# completo de los setups creados. setups_history.json es una VENTANA MOVIL de 30 dias y
# checkpoints_history.json solo guarda tres fotos (dias 5/10/20) y solo desde el 04/08.
# Consecuencia medida: 16 de los 46 stops no tienen NINGUN checkpoint, y las dos vistas
# del rendimiento —+6,7% a 20 sesiones frente a una esperanza de -3,8% por operacion—
# describen conjuntos que apenas se solapan, asi que no son reconciliables.
#
# Este fichero cierra ese hueco: UNA fila por setup creado, para siempre. Con el se puede
# calcular la cohorte COMPLETA (todos los setups de una fecha, resueltos o no) en vez de
# la muestra que sobrevivio a dos ventanas moviles.
#
# INMUTABLE: el dedupe conserva la entrada existente. Lo que se guarda son las condiciones
# EN EL MOMENTO DE CREAR el setup, y esas no cambian nunca. Reejecutar no reescribe.
#
# Solo en CIERRE, por la leccion del P39: una ejecucion intradia congelaria niveles de
# entrada calculados con precios parciales, y al ser inmutable no habria forma de corregirlo.
#
# Guarda ademas sct y ranking, que los checkpoints NO llevan y por eso hoy no se puede
# segmentar el rendimiento por calidad tecnica ni por posicion en el ranking.
# ============================================================

MAX_REGISTRO_SETUPS = 20000

def construir_entradas_registro(setups_hoy, valores, fecha):
    """Una fila por setup creado hoy, con su contexto completo al crearse."""
    por_ticker = {v.get('ticker'): v for v in (valores or []) if v.get('ticker')}
    salida = []
    for s in (setups_hoy or []):
        v = por_ticker.get(s.get('ticker')) or {}
        salida.append({
            'fecha_setup': fecha, 'ticker': s.get('ticker'), 'group': s.get('group'),
            'tipo': s.get('tipo'), 'entry_lo': s.get('entry_lo'), 'entry_hi': s.get('entry_hi'),
            'stop': s.get('stop'), 'target': s.get('target'), 'rr': s.get('rr'),
            'riesgo': s.get('riesgo'), 'score': s.get('score'), 'rs_4w': s.get('rs_4w'),
            'rsi_al_crear': s.get('rsi_al_crear'),
            'supertrend_al_crear': s.get('supertrend_al_crear'),
            'dispersion_al_crear': s.get('dispersion_al_crear'),
            # NO estan en setups_history y son justo las dos dimensiones que faltaban:
            'sct': v.get('sct'), 'ranking': v.get('ranking'),
            'cmf_al_crear': v.get('cmf'), 'adx_al_crear': v.get('adx'),
        })
    return salida

def merge_registro_setups(reg, entradas, max_entradas=MAX_REGISTRO_SETUPS):
    """Dedupe por (fecha_setup, ticker) CONSERVANDO lo existente: es un registro de
    creacion, y las condiciones de creacion no se reescriben."""
    reg = [e for e in (reg or []) if isinstance(e, dict)]
    vistos = {(e.get('fecha_setup'), e.get('ticker')) for e in reg}
    nuevas = 0
    for entrada in (entradas or []):
        clave = (entrada.get('fecha_setup'), entrada.get('ticker'))
        if clave in vistos:
            continue
        vistos.add(clave)
        reg.append(entrada)
        nuevas += 1
    reg.sort(key=lambda e: (e.get('fecha_setup') or '', e.get('ticker') or ''))
    return reg[-max_entradas:], nuevas

def actualizar_setups_registro(setups_hoy, valores, macro, ts):
    """Persiste el registro permanente. Solo en cierre (ver cabecera)."""
    if not setups_hoy:
        return None
    base = construir_entrada_breadth({}, macro, ts)
    fecha, es_cierre = base.get('fecha'), base.get('es_cierre')
    if not es_cierre:
        print('  Registro de setups OMITIDO: sesion en curso (los niveles serian parciales).')
        return None
    reg, _ = get_github_file('setups_registro.json')
    if reg == '__ERROR__':
        print('  🔴 setups_registro OMITIDO (fallo de lectura de GitHub) — fichero intacto')
        return None
    if not isinstance(reg, list):
        reg = []
    fusionado, nuevas = merge_registro_setups(
        reg, construir_entradas_registro(setups_hoy, valores, fecha))
    print(f'  Registro de setups: {nuevas} nuevos | total acumulado: {len(fusionado)}')
    return fusionado

def update_setups_history(values, all_groups):
    today = datetime.now(madrid).strftime('%Y-%m-%d')
    history, _ = get_github_file('setups_history.json')
    if history == '__ERROR__':
        # FIX CRITICO (08/07): sin lectura fiable NO se actualiza nada — mejor perder el
        # registro de setups de UNA ejecucion que machacar 30 dias de historico (paso de
        # verdad el 08/07 a las 07:34). upload_to_github rechazara el None resultante.
        print('  🔴 update_setups_history OMITIDO en esta ejecucion (fallo de lectura de GitHub) — el historico en el repo queda intacto')
        return None, []
    if history is None: history = []

    # Guardar setups validos de hoy
    setups_hoy = []
    for v in values:
        er = v.get('entry_range', {})
        if (er.get('entry_lo') and (er.get('rr') or 0) >= 2.0
                and (v.get('adx') or 0) >= 20 and v.get('riesgo') != 'ALTO'):
            setups_hoy.append({
                'ticker':    v['ticker'],
                'group':     v['group'],
                'tipo':      er['tipo'],
                'price':     v.get('price'),
                'entry_lo':  er['entry_lo'],
                'entry_hi':  er['entry_hi'],
                'stop':      er['stop'],
                'target':    er['target'],
                'target_parcial': er.get('target_parcial'),
                'rr':        er.get('rr'),
                'score':     v.get('score'),
                'rs_4w':     v.get('rs_4w'),
                'riesgo':    v.get('riesgo'),
                # NUEVO (06/07) — instrumentacion: direccion del Supertrend al CREAR el setup.
                # No filtra nada (decision deliberada: los pullbacks compran cerca de minimos
                # de correccion, donde un Supertrend(10,3) suele estar aun bajista; exigirlo
                # alcista sesgaria hacia entradas tardias). Con 2-3 meses de historico se
                # comparara la tasa de exito de setups nacidos alcistas vs bajistas y se
                # decidira CON EVIDENCIA si merece ser filtro de entrada (revision conjunta
                # con el Punto 16, finales de septiembre 2026).
                'supertrend_al_crear': (v.get('supertrend') or {}).get('direccion'),
                # PUNTO 25 (18/07/2026) — contexto AL CREAR para segmentar resultados a
                # posteriori con tools/analisis_historial.py: rsi de entrada (decision del
                # umbral del Punto 18: ¿rinden peor los pullbacks nacidos con RSI 65-70?) y
                # dispersion del dia (Punto 16/septiembre: ¿rinden peor los setups nacidos
                # en descorrelacion extrema?). Solo instrumentacion, no filtra nada. Los
                # setups anteriores al 18/07 no llevan estos campos (None en el analisis).
                'rsi_al_crear': v.get('rsi'),
                'dispersion_al_crear': ((_BREADTH_CACHE.get('ultimo') or {}).get('dispersion') or {}).get('ratio'),
            })

    # Eliminar entrada de hoy si existe y añadir nueva
    history = [h for h in history if h['date'] != today]
    history.append({'date': today, 'setups': setups_hoy})
    history = sorted(history, key=lambda x: x['date'])[-30:]  # 30 dias

    # Evaluar setups anteriores con precios actuales
    current_prices = {v['ticker']: v.get('price') for v in values if v.get('price')}
    # PUNTO 10 — CMF actual de cada ticker para detectar deterioro de flujo post-entrada
    current_cmf = {v['ticker']: v.get('cmf') for v in values if v.get('cmf') is not None}
    # NUEVO (06/07) — sesiones consecutivas con CMF<0, base de la alerta blanda recalibrada
    current_cmf_dias = {v['ticker']: v.get('cmf_dias_negativo') for v in values
                        if v.get('cmf_dias_negativo') is not None}
    # NUEVO (27/06) — Supertrend actual de cada ticker, como stop dinamico complementario
    current_supertrend = {v['ticker']: v.get('supertrend') for v in values if v.get('supertrend')}
    evaluaciones = []
    for entrada in history[:-1]:  # todos menos hoy
        dias = (datetime.now(madrid).date() - datetime.strptime(entrada['date'], '%Y-%m-%d').date()).days
        # RECALIBRADO (10/07) — EVALUACION DIARIA CONTINUA. Antes solo se evaluaba en los dias
        # exactos [5,10,20]: las alertas caducaban sin resolverse (casos reales: $VRSK con
        # alerta blanda el dia 20 y $UCTT con alerta DURA de salida total el dia 20
        # desaparecieron el dia 21 con las posiciones aun abiertas), y entre checkpoints los
        # setups no se vigilaban (una ruptura en el dia 7 no se veia hasta el dia 10).
        # Ahora: dos funciones separadas sobre el mismo bucle —
        #   ALERTAS: toda posicion de hasta HORIZONTE_SEGUIMIENTO dias se evalua a diario;
        #            una alerta permanece hasta resolverse (flujo recuperado/stop/target) o
        #            hasta que el setup envejece mas alla del horizonte.
        #   ESTADISTICA: el flag 'checkpoint' marca las evaluaciones de los dias exactos
        #            5/10/20 — los horizontes fijos comparables NO cambian; cualquier
        #            estadistica de rendimiento debe filtrar checkpoint==true.
        # Horizonte=20 mantenido a proposito (decision anexa 30/hasta-resolucion pendiente:
        # no cambiar dos cosas a la vez).
        # RECALIBRADO (11/07 tarde) — HORIZONTES SEPARADOS POR TIPO DE SEÑAL:
        #   DURAS (Supertrend post-entrada): vigilancia HASTA RESOLUCION — una señal de
        #     "salida total" no caduca por administracion, caduca cuando deja de ser cierta
        #     (la estructura se repara -> la condicion 'bajista hoy' desaparece sola; toca
        #     stop/target -> resultado deja de ser 'abierto'). Caso real $BF-B: alerta dura
        #     activa el 10/07 (dia 20) evaporada el 11/07 (dia 21) con la posicion abierta
        #     y la estructura rota. TECHO FISICO documentado: el historico retiene 30
        #     entradas, asi que "hasta resolucion" significa en la practica hasta ~30 dias
        #     de vida del setup — asumido: una posicion de 6 semanas sin stop ni target ya
        #     no es un trade de este sistema.
        #   BLANDAS (CMF) y marcadores de condicion al crear: expiran a los 20 dias — son
        #     matiz de gestion, no orden de salida, y 20 es el horizonte estadistico del
        #     sistema (checkpoint final). Las posiciones de 21-30 dias siguen apareciendo
        #     como vigiladas (por si rompen estructura), sin señales blandas.
        HORIZONTE_BLANDAS = 20
        # FIX (11/07 noche, primera ejecucion real) — el techo de ~30 dias estaba documentado
        # pero NO impuesto: la retencion son 30 ENTRADAS del historico, que con entradas de
        # fin de semana equivalen a ~42 dias naturales (se observaron evaluaciones en el dia
        # 32). Techo explicito: mas alla de 30 dias no hay señal posible (blandas <=20,
        # checkpoints <=20, duras <=30) y una posicion de 6 semanas ya no es un trade de
        # este sistema.
        HORIZONTE_DURAS = 30
        if dias < 1 or dias > HORIZONTE_DURAS: continue
        dentro_horizonte_blandas = dias <= HORIZONTE_BLANDAS
        es_checkpoint = dias in [5, 10, 20]
        # NUEVO (06/07) — sesiones de mercado aproximadas desde el setup (floor de dias*5/7).
        # Se usa para comparar con supertrend_dias (que cuenta sesiones, no dias naturales).
        # El floor infraestima ligeramente, endureciendo el filtro: ante la duda en la
        # frontera, se suprime la alerta (el objetivo es eliminar ruido, no añadirlo).
        sesiones_desde_setup = dias * 5 // 7
        for s in entrada['setups']:
            tk = s['ticker']
            precio_actual = current_prices.get(tk)
            if not precio_actual: continue
            precio_entrada = (s['entry_lo'] + s['entry_hi']) / 2
            ret_pct = round((precio_actual / precio_entrada - 1) * 100, 1)
            stop_tocado = precio_actual <= s['stop']
            target_tocado = precio_actual >= s['target']
            resultado = 'stop' if stop_tocado else ('target' if target_tocado else 'abierto')
            # NUEVO (27/06, RECALIBRADO 06/07) — Supertrend dinamico: señal dura de salida total.
            # Recalibracion: alertar SOLO si la ruptura es POSTERIOR a la creacion del setup
            # (supertrend_dias < sesiones_desde_setup). Caso real 05/07: 20+ de 27 alertas eran
            # setups NACIDOS con el Supertrend ya bajista ($IDXX 105 sesiones, $GEHC 82, $MCD 74
            # con setups de 20 dias) — alertar "salida total" por una condicion que ya existia
            # al crear el setup es incoherente. Los pre-rotos quedan marcados (supertrend_pre_roto)
            # para trazabilidad en data.json, pero sin alerta.
            st = current_supertrend.get(tk)
            st_nivel = st.get('nivel') if st else None
            st_direccion = st.get('direccion') if st else None
            st_dias = st.get('dias_en_direccion_actual') if st else None
            es_bajista_abierto = (resultado == 'abierto' and st_direccion == 'bajista')
            if es_bajista_abierto and st_dias is not None:
                ruptura_posterior = st_dias < sesiones_desde_setup
            else:
                # Sin dato de dias (historico previo al 05/07): comportamiento anterior (alertar)
                ruptura_posterior = es_bajista_abierto
            alerta_supertrend = es_bajista_abierto and ruptura_posterior
            supertrend_pre_roto = es_bajista_abierto and not ruptura_posterior and dentro_horizonte_blandas
            # PUNTO 10 (RECALIBRADO 06/07, CORRECCION SIMETRICA misma fecha) — alerta blanda:
            # DISTRIBUCION CONFIRMADA que empezo DESPUES de la entrada, no cruce de un dia ni
            # condicion preexistente. Dos filtros:
            # (1) Persistencia: CMF<0 durante CMF_DIAS_ALERTA sesiones consecutivas (antes,
            #     CMF<0.05 en la ultima sesion disparaba en el 67% de las posiciones — 100/150
            #     el 05/07, incluidas ganadoras de +16% con CMF -0.003).
            # (2) Posterioridad: la racha negativa debe ser MAS CORTA que la vida del setup
            #     (cmf_dias_negativo < sesiones_desde_setup). Mismo principio que la alerta
            #     Supertrend: un setup que NACIO con CMF negativo ($OTIS 84 sesiones, $AVY 78,
            #     $MCD 48 para setups de 14 sesiones el 05/07) no esta "deteriorandose" —
            #     ya estaba asi al entrar. Los preexistentes quedan marcados (cmf_pre_negativo)
            #     para trazabilidad, sin alerta.
            # Ademas, jerarquia de señales: si hay señal dura activa (salida total), la blanda
            # (reducir 50%) se suprime — nunca instrucciones contradictorias sobre el mismo ticker.
            CMF_DIAS_ALERTA = 3
            # RECALIBRADO (11/07 tarde) — UMBRAL DE MAGNITUD ademas de persistencia: un CMF
            # de -0.001 es flujo neutro con signo de redondeo, no distribucion, y ocupaba el
            # mismo renglon de "reducir 50%" que un -0.193. Expediente: 6 casos de milesimas
            # en 4 sesiones ($EXPE -0.002, $HOOD -0.009 y -0.001, $IP -0.007/-0.017,
            # $IR -0.008, $MDT -0.001). Umbral sobre el VALOR ACTUAL (no la media de la
            # racha): simple, coherente con lo mostrado, y con la propiedad de que un CMF
            # recuperandose hacia cero deja de alertar ANTES de que la racha se rompa
            # formalmente. Un goteo tibio en posicion perdedora lo gestiona el stop; si el
            # goteo se convierte en salida real de dinero, cruzara el umbral y la alerta
            # volvera con motivo. Aplicado dentro de cmf_confirmado: el marcador de
            # preexistentes hereda el mismo criterio.
            UMBRAL_CMF_MAGNITUD = -0.02
            cmf_actual = current_cmf.get(tk)
            cmf_dias_neg = current_cmf_dias.get(tk)
            cmf_confirmado = (resultado == 'abierto' and
                              dentro_horizonte_blandas and
                              cmf_dias_neg is not None and
                              cmf_dias_neg >= CMF_DIAS_ALERTA and
                              cmf_actual is not None and
                              cmf_actual < UMBRAL_CMF_MAGNITUD)
            cmf_posterior = cmf_confirmado and cmf_dias_neg < sesiones_desde_setup
            alerta_cmf = cmf_posterior and not alerta_supertrend
            cmf_pre_negativo = cmf_confirmado and not cmf_posterior
            evaluaciones.append({
                'fecha_setup': entrada['date'],
                'checkpoint': es_checkpoint,
                'dias':        dias,
                'ticker':      tk,
                'group':       s['group'],
                'tipo':        s['tipo'],
                'precio_entrada': round(precio_entrada, 2),
                'precio_actual':  precio_actual,
                'stop':        s['stop'],
                'target':      s['target'],
                'ret_pct':     ret_pct,
                'resultado':   resultado,
                'cmf_actual':  round(cmf_actual, 3) if cmf_actual is not None else None,
                'cmf_dias_negativo': cmf_dias_neg,
                'alerta_cmf':  alerta_cmf,
                'cmf_pre_negativo': cmf_pre_negativo,
                'supertrend_nivel':     st_nivel,
                'supertrend_direccion': st_direccion,
                'supertrend_dias':      st_dias,
                'alerta_supertrend':    alerta_supertrend,
                'supertrend_pre_roto':  supertrend_pre_roto,
            })

    return history, evaluaciones

_TRAZAS_VISTAS = set()

def _traza(clave, exc, extra=''):
    """PUNTO 41 (05/08/2026) — deja constancia de un fallo que se degrada en silencio.

    El scanner tiene 38 manejadores que capturan sin decir nada. En los indicadores
    (calc_rsi, calc_adx...) ese silencio es DELIBERADO y correcto: se ejecutan 523 veces
    y un ticker con datos raros no debe tumbar el panel entero. El problema son los de
    orquestacion: el bug del P33 (claves inexistentes en entry_range) vivio una jornada
    completa porque un `except: pass` se lo tragaba, y solo se descubrio comparando el
    informe con lo que deberia haber salido.

    Esta funcion NO cambia el comportamiento — se sigue degradando igual — solo lo hace
    visible. Imprime UNA sola vez por clave y ejecucion: si algo falla en los 523 tickers
    veras una linea, no 523. La clave identifica el sitio, no el ticker.
    """
    if clave in _TRAZAS_VISTAS:
        return
    _TRAZAS_VISTAS.add(clave)
    print(f'  ⚠️  degradado en {clave}: {type(exc).__name__}: {exc}{extra}')

def clean_nan(obj):
    if isinstance(obj, float):
        import math
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj

def upload_to_github(filename, content):
    url=f'https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{filename}'
    headers={'Authorization':f'token {GITHUB_TOKEN}','Content-Type':'application/json'}
    r=requests.get(url,headers=headers); sha=r.json().get('sha') if r.status_code==200 else None
    # FIX CRITICO (08/07) — cinturon de seguridad: si el contenido es nulo (json.dumps(None)
    # cuando una funcion de historico aborto por fallo de lectura), NO subir nada: la version
    # que ya esta en GitHub es la buena. Protege main() y las Celdas 3/4 sin tocar su codigo.
    if content is None or content in ('null', b'null'):
        print(f'  ⚠️ Subida de {filename} OMITIDA: contenido nulo (fallo previo de lectura) — se conserva la version del repo')
        return
    content_b64=base64.b64encode(content.encode('utf-8') if isinstance(content,str) else content).decode('utf-8')
    payload={'message':f'Actualizar {filename} - {datetime.now(madrid).strftime("%d/%m/%Y %H:%M")}','content':content_b64}
    if sha: payload['sha']=sha
    r=requests.put(url,headers=headers,json=payload)
    if r.status_code in [200,201]:
        print(f'  OK {filename}')
    else:
        # P71 (14/08/2026) — el manejador de errores REVENTABA al manejar el error:
        # r.json() asume cuerpo JSON y un 401 de GitHub puede venir vacio o en HTML, asi
        # que lanzaba JSONDecodeError, abortaba la celda entera y dejaba el pipeline a
        # medias. El 14/08 subio data.json, history.json y setups_history.json y los otros
        # siete se quedaron sin subir: exactamente la inconsistencia que el commit unico
        # existe para evitar. Un fallo de subida debe INFORMAR, nunca tumbar la ejecucion.
        print(f'  Error {filename}: HTTP {r.status_code} — {_mensaje_error_github(r)}')
        raise RuntimeError(f'subida de {filename} fallida: HTTP {r.status_code}')

_RATES_HIST_CACHE = {}

def _nivel_hace_n_sesiones(clave, n=20):
    """P73 (16/08/2026) — nivel de un tramo de curva hace n sesiones, leido de la serie PROPIA.

    Yahoo lleva desde el 12/08 recortando el historico que sirve para los simbolos de deuda:
    empezo con ^TNX en 15 sesiones y el 14/08 ya eran cuatro tramos con ~16. Con menos de 21
    no se puede calcular la variacion a 20 sesiones, asi que la lectura de VELOCIDAD de la
    curva desaparecio del informe entera — justo lo que hacia util la serie para el P16.

    Pero ese dato no hace falta pedirselo a Yahoo: rates_history.json guarda el nivel diario
    de cada tramo desde el 09/02/2026 (sembrado con 120 sesiones el 01/08). La variacion se
    calcula contra el historico propio y el scanner deja de depender de la ventana que la
    fuente decida servir cada dia. Mismo principio que el resto de series acumuladas.

    Solo se usan entradas de CIERRE: una provisional intradia falsearia la referencia.
    El fichero se lee UNA vez por ejecucion y se cachea; si falla, se devuelve None y el
    llamador degrada como venia haciendo.
    """
    if 'entradas' not in _RATES_HIST_CACHE:
        try:
            rh, _ = get_github_file('rates_history.json')
            if rh == '__ERROR__' or not isinstance(rh, list):
                rh = []
        except Exception as _e:
            _traza('rates/historico-variacion', _e); rh = []
            # ARREGLO 2 (16/08/2026, sobre datos comprobados): el primer intento asumio que las
        # 120 sesiones sembradas con instrumentar_tipos.py NO llevaban el campo es_cierre.
        # FALSO — lo llevan y vale False, puesto por prudencia del script, aunque son cierres
        # EOD de Yahoo. Con el filtro anterior quedaban solo las ~11 entradas escritas por el
        # scanner desde el 01/08, por debajo de las 20 necesarias, y el rescate fallaba casi
        # siempre.
        #
        # Criterio correcto: una entrada provisional solo puede existir para la sesion MAS
        # RECIENTE (la del panel en curso), porque la nocturna la sustituye por el cierre.
        # Asi que se ordena todo y se descartan unicamente las provisionales del FINAL; el
        # resto del historico entra tal cual, venga sembrado o escrito por el scanner.
        cierres = sorted([e for e in rh if isinstance(e, dict) and e.get('fecha')],
                         key=lambda e: e.get('fecha') or '')
        # Se descartan como mucho las 2 ultimas si son provisionales: mas atras no puede
        # haberlas, y un tope fijo evita que un historico entero marcado False (el caso de
        # las sembradas) se vacie por completo.
        for _ in range(2):
            if cierres and cierres[-1].get('es_cierre') is False:
                cierres.pop()
            else:
                break
        _RATES_HIST_CACHE['entradas'] = cierres
    entradas = _RATES_HIST_CACHE['entradas']
    # La entrada de HOY aun no esta escrita en este punto del pipeline, asi que la ultima
    # del fichero es la sesion anterior: n sesiones atras desde hoy es el indice -n.
    if len(entradas) < n:
        return None
    niveles = (entradas[-n] or {}).get('niveles') or {}
    valor = niveles.get(clave)
    return float(valor) if isinstance(valor, (int, float)) else None

def _variacion_20s_pb(clave, nivel_hoy):
    """P73 — variacion a 20 sesiones en puntos basicos usando la serie propia. None si no hay."""
    if not isinstance(nivel_hoy, (int, float)):
        return None
    previo = _nivel_hace_n_sesiones(clave, 20)
    return None if previo is None else round((float(nivel_hoy) - previo) * 100, 1)

def _mensaje_error_github(r):
    """P71 — mensaje de error de la API sin asumir que el cuerpo es JSON.

    GitHub devuelve JSON en casi todos los errores, pero no en todos: un 401 puede llegar
    con cuerpo vacio. Llamar a r.json() ahi lanza JSONDecodeError DENTRO del manejador de
    errores, que es la peor forma posible de fallar.
    """
    try:
        return r.json().get('message', 'sin detalle')
    except Exception:
        texto = (r.text or '').strip().replace('\n', ' ')
        return (texto[:120] + '...') if len(texto) > 120 else (texto or 'sin cuerpo de respuesta')

def upload_files_to_github(files):
    """
    NUEVO (11/07) — sube VARIOS ficheros en UN SOLO commit usando la API Git Data/Trees
    (blobs -> tree -> commit -> ref). Motivo: cada PUT individual a la Contents API crea un
    commit, cada commit dispara un despliegue de GitHub Pages, y los despliegues casi
    simultaneos compiten entre si — los que pierden fallan con "Deployment failed" y generan
    correo de alerta (diagnosticado el 05/07). Un commit por tanda = un despliegue, cero
    carreras, historial 3x mas limpio y menos peticiones a la API (menos rate-limits, la
    causa raiz del incidente de perdida de datos del 08/07).

    files: dict {filename: contenido_json_str}. Los contenidos nulos se filtran con el mismo
    cinturon de seguridad que upload_to_github (fallo previo de lectura -> conservar repo).
    FALLBACK: si CUALQUIER paso de la cadena Trees falla, se degrada a los PUT individuales
    de upload_to_github (comportamiento probado en produccion desde el inicio del proyecto).
    """
    limpios = {}
    for fn, contenido in (files or {}).items():
        if contenido is None or contenido in ('null', b'null'):
            print(f'  ⚠️ Subida de {fn} OMITIDA: contenido nulo (fallo previo de lectura) — se conserva la version del repo')
            continue
        limpios[fn] = contenido
    if not limpios:
        return
    base = f'https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}'
    headers = {'Authorization': f'token {GITHUB_TOKEN}', 'Content-Type': 'application/json'}
    try:
        # 1) sha del ultimo commit de main
        r = requests.get(f'{base}/git/refs/heads/main', headers=headers, timeout=20)
        r.raise_for_status()
        commit_sha = r.json()['object']['sha']
        # 2) sha del tree base
        r = requests.get(f'{base}/git/commits/{commit_sha}', headers=headers, timeout=20)
        r.raise_for_status()
        base_tree = r.json()['tree']['sha']
        # 3) un blob por fichero
        tree_items = []
        for fn, contenido in limpios.items():
            r = requests.post(f'{base}/git/blobs', headers=headers, timeout=30,
                              json={'content': contenido, 'encoding': 'utf-8'})
            r.raise_for_status()
            tree_items.append({'path': fn, 'mode': '100644', 'type': 'blob', 'sha': r.json()['sha']})
        # 4) tree nuevo sobre el base
        r = requests.post(f'{base}/git/trees', headers=headers, timeout=30,
                          json={'base_tree': base_tree, 'tree': tree_items})
        r.raise_for_status()
        tree_sha = r.json()['sha']
        # 5) commit unico
        mensaje = f'Actualizar datos scanner {datetime.now(madrid).strftime("%d/%m/%Y %H:%M")}'
        r = requests.post(f'{base}/git/commits', headers=headers, timeout=30,
                          json={'message': mensaje, 'tree': tree_sha, 'parents': [commit_sha]})
        r.raise_for_status()
        nuevo_commit = r.json()['sha']
        # 6) mover la rama
        r = requests.patch(f'{base}/git/refs/heads/main', headers=headers, timeout=20,
                           json={'sha': nuevo_commit})
        r.raise_for_status()
        print(f'  OK commit unico ({", ".join(limpios.keys())})')
    except Exception as e:
        print(f'  ⚠️ Commit unico fallido ({type(e).__name__}) — degradando a subidas individuales')
        # P71 — cada fichero va en su propio try: que uno falle no puede impedir que los
        # demas se suban. Antes la primera excepcion abortaba el bucle y el resto de la
        # ejecucion (top de temas incluido). Al final se resume que quedo sin subir, para
        # que la degradacion parcial no pase desapercibida.
        _fallidos = []
        for fn, contenido in limpios.items():
            try:
                upload_to_github(fn, contenido)
            except Exception as _eu:
                _traza(f'subida/{fn}', _eu)
                _fallidos.append(fn)
        if _fallidos:
            print(f'  🔴 SUBIDA INCOMPLETA: {len(_fallidos)} de {len(limpios)} ficheros NO se '
                  f'subieron ({", ".join(sorted(_fallidos))}). El repo queda en estado MIXTO: '
                  'unos ficheros son de esta ejecucion y otros de la anterior. Si el error es '
                  '401, el token de GitHub ha caducado o ha sido revocado.')

def update_history(all_groups, all_values=None):
    today=datetime.now(madrid).strftime('%Y-%m-%d')
    history,_=get_github_file('history.json')
    if history == '__ERROR__':
        # FIX CRITICO (08/07): misma proteccion que en update_setups_history — un fallo de
        # lectura no debe reconstruir el historico de temas desde cero y machacarlo al subir.
        print('  🔴 update_history OMITIDO en esta ejecucion (fallo de lectura de GitHub) — history.json en el repo queda intacto')
        return None
    if history is None: history=[]
    history=[h for h in history if h['date']!=today]
    entry={'date':today,'scores':{g['group']:g['score'] for g in all_groups},'ranks':{g['group']:i+1 for i,g in enumerate(all_groups)}}
    if all_values:
        entry['ma50_state']={v['ticker']:v.get('ma50',False) for v in all_values}
    # Guardar lider de cada grupo
    entry['leaders']={g['group']:g['top3'][0]['ticker'] if g['top3'] else None for g in all_groups}
    history.append(entry)
    return sorted(history,key=lambda x:x['date'])[-7:]

_IDENTIDAD_TELEGRAM_MOSTRADA = []

def _identidad_telegram():
    """PUNTO 44 (06/08/2026) — imprime QUE bot y QUE chat se estan usando, una vez por ejecucion.

    El caso que lo motiva: los mensajes salian desde Colab y NUNCA desde GitHub Actions, con
    error 403. Diagnosticarlo requeria comparar los secrets de los dos entornos a mano, y no
    habia forma de saber desde el log cual era cual. El identificador de bot son los digitos
    anteriores a los dos puntos del token — no es secreto, el propio token si — y del chat solo
    se muestran los ultimos digitos. Con eso basta para ver de un vistazo si los dos entornos
    apuntan al mismo sitio, sin exponer credenciales en un log publico.
    """
    if _IDENTIDAD_TELEGRAM_MOSTRADA:
        return
    _IDENTIDAD_TELEGRAM_MOSTRADA.append(True)
    bot = (TELEGRAM_TOKEN or '').split(':')[0] or '?'
    chat = str(TELEGRAM_CHAT_ID or '?')
    chat_vis = ('...' + chat[-4:]) if len(chat) > 4 else chat
    print(f'     [identidad] bot id={bot} | chat={chat_vis} — comparalo con el otro entorno')

def send_telegram(message):
    if not TELEGRAM_TOKEN:
        print('  TELEGRAM_TOKEN no configurado — alerta omitida')
        return
    if not TELEGRAM_CHAT_ID:
        # FIX (11/07) — guarda explicita: antes el chat ID tenia default hardcodeado en un
        # repo publico; ahora vive solo en secrets (GitHub Actions y Colab). Si falta, avisar
        # en voz alta en lugar de fallar en silencio contra la API de Telegram.
        print('  ⚠️ TELEGRAM_CHAT_ID no configurado (revisar secrets) — alerta omitida')
        return
    for intento in range(3):
        try:
            r=requests.post(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',json={'chat_id':TELEGRAM_CHAT_ID,'text':message},timeout=30)
            if r.status_code==200:
                print('  OK Telegram'); return
            else:
                print(f'  Error Telegram: {r.status_code}')
                # PUNTO 44 (06/08/2026) — la descripcion que devuelve la API es lo que de verdad
                # dice que pasa ("Forbidden: bot was blocked by the user", "chat not found"...).
                # Estaba ahi todo el tiempo y no se imprimia: el diagnostico se hacia a ciegas.
                try:
                    _desc = (r.json() or {}).get('description')
                    if _desc:
                        print(f'     Telegram dice: {_desc}')
                except Exception as _ej:
                    _traza('telegram/respuesta-no-json', _ej)
                _identidad_telegram()
                if r.status_code == 401:
                    # PUNTO 27 (19/07/2026) — el 401 estuvo semanas mudo en logs de Actions que
                    # nadie mira; ahora se explica a si mismo para acortar el proximo diagnostico.
                    print('     401 = token invalido o revocado: regenerar en @BotFather y actualizar el secret TELEGRAM_TOKEN (en GitHub Actions y, si se usa Colab, en sus secretos)')
                elif r.status_code == 403:
                    # PUNTO 44 — problema DISTINTO del 401 y con solucion distinta. 403 = el token
                    # es valido (Telegram reconoce al bot) pero ese bot no puede escribir a ese
                    # chat. Caso real: los envios funcionaban desde Colab y nunca desde Actions,
                    # lo que apunta a que el par token/chat_id difiere entre los dos entornos.
                    print('     403 = el token es VALIDO pero el bot no puede escribir a ese chat.')
                    print('     Causas habituales: nunca se pulso /start en ese chat con ESE bot,')
                    print('     el bot fue bloqueado o el chat borrado, o el chat_id pertenece a')
                    print('     otro bot. Compara el ID de bot y el chat de ambos entornos (se')
                    print('     imprimen arriba): si difieren, iguala los secrets al par que funciona.')
                return
        except Exception as e:
            print(f'  Intento Telegram {intento+1}/3: {e}')
            if intento < 2:
                import time; time.sleep(5)

def generate_alerts(data, history, spy_healthy=True, bench_series=None):
    if not TELEGRAM_TOKEN:
        print('  TELEGRAM_TOKEN no configurado — alertas omitidas')
        return
    values=data['values']; groups=data['groups']; ts=data['timestamp']
    bs = bench_series
    prev_scores=history[-2].get('scores',{}) if len(history)>=2 else {}
    wl=set(t for grp in PERSONAL_WATCHLIST.values() for t in grp)
    msgs=[]
    wl_bo=[v for v in values if v['breakout'] and v['ticker'] in wl and v.get('days_ago')==1]
    if wl_bo:
        msg=f'RUPTURA EN WATCHLIST — {ts}\n\n'
        for v in wl_bo[:5]:
            er=v.get('entry_range',{}); msg+=f'{v["ticker"]} ({v["group"]})\n  ${v.get("price","?")} | RS4s:{v.get("rs_4w","?")}%\n'
            if er.get('entry_lo'): msg+=f'  Entrada:${er["entry_lo"]}-${er["entry_hi"]} | Stop:${er["stop"]} | Obj:${er["target"]}\n'
            msg+='\n'
        msgs.append(msg)
    # P38 — mismos filtros de calidad que el informe (cumple_calidad): sin esto Telegram
    # enviaba rupturas que generate_analysis habia descartado por SCT/ADX/riesgo/RVOL/volumen.
    # NO se limita al top-5 del informe: una ruptura fuera del ranking sigue siendo un aviso
    # util, solo deja de contradecir al informe.
    sp_bo=[v for v in values if v['breakout'] and v['ticker'] not in wl and v.get('days_ago')==1 and (v.get('score') or 0)>=65 and (v.get('entry_range',{}).get('rr') or 0)>=2.0 and cumple_calidad(v)]
    if sp_bo:
        msg=f'RUPTURAS S&P500 — {ts}\n\n'
        for v in sp_bo[:5]:
            er=v.get('entry_range',{}); msg+=f'{v["ticker"]} ({v["group"]})\n  ${v.get("price","?")} | Score:{v.get("score","?")} | R/B:1:{er.get("rr","?")}\n'
            if er.get('entry_lo'): msg+=f'  Entrada:${er["entry_lo"]}-${er["entry_hi"]} | Stop:${er["stop"]}\n'
            msg+='\n'
        msgs.append(msg)
    vol_exc=[v for v in values if v['ticker'] in wl and (v.get('vol_z') or 0)>=3.0 and not v['breakout']]
    if vol_exc:
        msg=f'VOLUMEN EXCEPCIONAL — {ts}\n\n'
        for v in vol_exc[:5]: msg+=f'{v["ticker"]}: ${v.get("price","?")} | Vol:{v.get("vol_z","?")}s | RS:{v.get("rs_4w","?")}%\n'
        msgs.append(msg)
    if prev_scores:
        emerg=[{'group':g['group'],'prev':prev_scores[g['group']],'curr':g['score'],'sub':round(g['score']-prev_scores[g['group']],1)} for g in groups if g['group'] in prev_scores and (g['score']-prev_scores[g['group']])>=10]
        if emerg:
            msg=f'TEMA EMERGENTE — {ts}\n\n'
            for e in sorted(emerg,key=lambda x:x['sub'],reverse=True): msg+=f'{e["group"]}: {e["prev"]} -> {e["curr"]} (+{e["sub"]} pts)\n'
            msgs.append(msg)
        det=[{'group':g['group'],'prev':prev_scores[g['group']],'curr':g['score'],'caida':round(prev_scores[g['group']]-g['score'],1)} for g in groups if g['group'] in prev_scores and (prev_scores[g['group']]-g['score'])>=15]
        if det:
            msg=f'DETERIORO — {ts}\n\n'
            for d in sorted(det,key=lambda x:x['caida'],reverse=True): msg+=f'{d["group"]}: {d["prev"]} -> {d["curr"]} (-{d["caida"]} pts)\n'
            msgs.append(msg)
        bajadas=sum(1 for g in groups if prev_scores.get(g['group'],g['score'])>g['score'])
        if len(groups)>0 and bajadas/len(groups)>=0.70:
            msgs.append(f'ALERTA RISK-OFF — {ts}\n{bajadas} de {len(groups)} temas bajan.\nReducir exposicion.')
    if not spy_healthy:
        msgs.append(f'MERCADO BAJISTA — {ts}\nSPY bajo MM200. Scores penalizados 30%.\nReducir tamano de posicion.')

    # Divergencia SPY/amplitud
    # SPY en maximo de 52 semanas pero pocos temas en verde
    try:
        spy_series = bs.dropna()
        spy_high52 = float(spy_series.iloc[-252:].max()) if len(spy_series) >= 252 else float(spy_series.max())
        spy_current = float(spy_series.iloc[-1])
        spy_en_maximo = spy_current >= spy_high52 * 0.995  # dentro del 0.5% del maximo
        temas_verdes = sum(1 for g in groups if g['score'] >= 70)
        pct_verdes = temas_verdes / len(groups) if groups else 0
        if spy_en_maximo and pct_verdes < 0.15:  # SPY en maximo pero menos del 15% de temas en verde
            # P61 (12/08/2026): el texto afirmaba "el mercado sube concentrado en pocas
            # megacaps", una CAUSA que esta alerta no comprueba. El 11/08 salto con el SPX
            # cerrando -0.32% y el Russell +0.32%, con ~300 valores del SPX en positivo: es
            # decir, justo lo contrario de concentracion en megacaps. Pocos temas fuertes
            # tiene DOS lecturas opuestas y las distingue la amplitud, que ya esta medida:
            #   - amplitud DEBIL  -> concentracion real, pocos valores sostienen el indice
            #   - amplitud SANA   -> mercado DISPERSO, sube mucha cosa pero nada lidera en
            #                        bloque; no es fragilidad, es ausencia de tema dominante
            # Se emite el dato y la lectura que corresponda, sin inventar la causa.
            _br = (data.get('breadth') or {})
            _mm200 = _br.get('pct_sobre_mm200')
            _mm50 = _br.get('pct_sobre_mm50')
            msg = f'DIVERGENCIA SPY/TEMAS — {ts}\n\n'
            msg += f'SPY cerca de maximos de 52 semanas (${round(spy_current,2)})\n'
            msg += f'Solo {temas_verdes} de {len(groups)} temas en zona fuerte ({round(pct_verdes*100,0):.0f}%)\n'
            if _mm200 is not None:
                msg += f'Amplitud: {_mm200}% sobre MM200'
                msg += f' | {_mm50}% sobre MM50\n' if _mm50 is not None else '\n'
                if _mm200 < 50:
                    msg += ('Amplitud DEBIL: pocos valores sostienen al indice. '
                            'Concentracion real, extremar cautela.')
                else:
                    msg += ('Amplitud SANA: mercado DISPERSO, no concentrado — sube mucho valor '
                            'pero ningun tema lidera en bloque. Favorece la seleccion individual; '
                            'no es señal de fragilidad por si sola.')
            else:
                msg += ('Sin dato de amplitud: no se puede distinguir concentracion real de '
                        'mercado disperso. Contrastar antes de concluir.')
            msgs.append(msg)
    except Exception as _e:
        _traza('alertas/divergencia-spy-amplitud', _e)

    # Lider cambiante en temas fuertes
    if len(history) >= 2:
        prev_leaders = history[-2].get('leaders', {})
        curr_leaders = {g['group']: g['top3'][0]['ticker'] if g['top3'] else None for g in groups}
        cambios_lider = []
        for g in groups:
            if g['score'] >= 50:  # solo temas relevantes
                prev_l = prev_leaders.get(g['group'])
                curr_l = curr_leaders.get(g['group'])
                if prev_l and curr_l and prev_l != curr_l:
                    cambios_lider.append({'group': g['group'], 'prev': prev_l, 'curr': curr_l, 'score': g['score']})
        if cambios_lider:
            msg = f'CAMBIO DE LIDER — {ts}\n\n'
            for c in cambios_lider:
                msg += f'{c["group"]} (score {c["score"]})\n'
                msg += f'  {c["prev"]} -> {c["curr"]}\n\n'
            msgs.append(msg)

    # Recuperacion de MM50 (valor cruzó por encima hoy)
    if len(history) >= 2:
        prev_ma50 = history[-2].get('ma50_state', {})
        curr_ma50 = {v['ticker']: v.get('ma50', False) for v in values}
        recuperaciones = [
            v for v in values
            if v['ticker'] in wl
            and curr_ma50.get(v['ticker']) is True
            and prev_ma50.get(v['ticker']) is False
        ]
        if recuperaciones:
            msg = f'RECUPERACION MM50 EN WATCHLIST — {ts}\n\n'
            for v in recuperaciones[:5]:
                msg += f'{v["ticker"]} ({v["group"]})\n'
                msg += f'  Precio: ${v.get("price","?")} | MM50: ${v.get("ma50_val","?")} | RS4s: {v.get("rs_4w","?")}%\n\n'
            msgs.append(msg)

    if msgs:
        print(f'  {len(msgs)} alerta(s) generada(s) — enviando...')  # PUNTO 27 — recuento siempre visible
        for m in msgs: send_telegram(m)
    else:
        print('  Sin alertas hoy (ninguna condicion de disparo cumplida)')

def main():
    print('\n═══════════════════════════════════════')
    print('  SCANNER DE TEMAS EMERGENTES')
    ts=datetime.now(madrid).strftime('%d/%m/%Y %H:%M')
    print(f'  {ts}\n═══════════════════════════════════════\n')
    import pickle, os
    CACHE_FILE = '/tmp/scanner_cache.pkl'
    cache = {}

    print('▸ SPY...')
    bc,bv,bh,bl=download_prices(['SPY'],period='1y'); bs=bc['SPY']
    spy_ok, spy_score_continuo = spy_health(bs)  # P57: tupla (bool, float)

    # Comparativa SPY semanal
    try:
        spy_ret_1w = round((float(bs.iloc[-1]) / float(bs.iloc[-5])  - 1) * 100, 1)
        spy_ret_1m = round((float(bs.iloc[-1]) / float(bs.iloc[-20]) - 1) * 100, 1)
    except Exception as _e:
        _traza('main/retornos-spy', _e)
        spy_ret_1w = None; spy_ret_1m = None
    print(f'  SPY 1s: {spy_ret_1w}% | SPY 1m: {spy_ret_1m}%')

    print('\n▸ Datos macro (VIX, DXY, US10Y)...')
    macro = get_macro_data()
    if macro:
        vix = macro.get('^VIX', {})
        dxy = macro.get('DX-Y.NYB', {})
        usy = macro.get('^TNX', {})
        print(f'  VIX: {vix.get("current","?")} ({vix.get("chg_pct","?")}%) | DXY: {dxy.get("current","?")} | US10Y: {usy.get("current","?")}%')
    else:
        print('  Datos macro no disponibles')

    pt=list(set(t for grp in PERSONAL_WATCHLIST.values() for t in grp))

    # PUNTO 56 — put/call de Cboe, junto al resto del bloque macro. Se descarga y se
    # imprime AQUI a proposito: la primera version asignaba la variable despues del bloque
    # de setups pero la usaba antes, un NameError que no salto en Colab porque el espejo de
    # la celda del pipeline no llevaba ese print. Dato y log en el mismo sitio.
    putcall = get_putcall_cboe()
    if putcall:
        print('  Put/call (Cboe): ' + ' | '.join(f'{k}={v}' for k, v in sorted(putcall.items())))

    print(f'\n▸ Watchlist ({len(pt)} valores)...')
    cp,vp,hp,lp=download_prices(pt+['SPY'],period='1y')
    if not cp.empty:
        cache['wl'] = (cp,vp,hp,lp)
        pickle.dump(cache, open(CACHE_FILE,'wb'))
    elif os.path.exists(CACHE_FILE):
        print('  Usando cache de descarga anterior')
        cache = pickle.load(open(CACHE_FILE,'rb'))
        cp,vp,hp,lp = cache.get('wl',(cp,vp,hp,lp))
    pr=analyze_universe(PERSONAL_WATCHLIST,bs,cp,vp,hp,lp,spy_healthy=spy_ok,spy_score=spy_score_continuo)
    pgs=calc_groups(pr,is_sp=False); print(f'  OK {len(pr)} valores')
    # PUNTO 26 — nombrar a los tickers pedidos que no llegan al analisis (fallo de descarga,
    # historico insuficiente o delistado/renombrado — caso MMC->MRSH). Salir del "OK 31/32" mudo.
    faltan_wl = sorted(set(pt) - {r['ticker'] for r in pr})
    if faltan_wl: print(f'  Sin datos/analisis en watchlist: {", ".join(faltan_wl)}')

    print('\n▸ Lista S&P 500...')
    df_sp=pd.read_csv('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv')
    sbs={}
    for _,row in df_sp.iterrows():
        tk=row['Symbol'].replace('.','-')
        if tk not in pt: sbs.setdefault(SECTOR_LABELS.get(row['GICS Sector'],row['GICS Sector']),[]).append(tk)
    sa=list(set(t for grp in sbs.values() for t in grp))

    # Ampliar con Nasdaq 100 y SOX (valores no duplicados)
    nasdaq100_extra = [
        'ABNB','ADSK','ALGN','ASML','BIDU','BIIB',
        'CDNS','CHTR','CPRT','CTSH','DDOG','DLTR','DOCU','DXCM',
        'EA','EBAY','FAST','FSLR','FTNT','GFS',
        'IDXX','ILMN','KDP','KHC','LULU','MAR','MCHP','MDLZ',
        'MNST','MRNA','MRVL','NTES','ODFL','OKTA','ON','ORLY',
        'PANW','PAYX','PCAR','PDD','PYPL','QCOM','REGN','ROST',
        'SNPS','SWKS','TEAM','TMUS','TSCO',
        'TTD','TTWO','TXN','VRSK','VRSN','VRTX','WBD','XEL','ZM','ZS'
    ]
    sox_extra = [
        'ACLS','ADI','AEHR','AMAT','AMKR','ASML','COHU',
        'ENTG','ENVX','GFS','IPGP','KLAC','LRCX','MCHP',
        'MKSI','MPWR','MRVL','ONTO','POWI','QRVO','RMBS','SITM',
        'SWKS','SYNA','TER','TSM','UCTT','WOLF'
    ]
    todos_extra = list(set(nasdaq100_extra + sox_extra))
    ya_incluidos = set(pt + sa)
    for tk in todos_extra:
        if tk not in ya_incluidos:
            # Clasificar en sector apropiado
            if tk in sox_extra:
                sbs.setdefault('Semiconductores SOX', []).append(tk)
            else:
                sbs.setdefault('Nasdaq 100', []).append(tk)
    sa = list(set(t for grp in sbs.values() for t in grp))
    print(f'  Ampliado con Nasdaq 100 y SOX: {len(sa)} valores totales')
    print(f'▸ Descargando {len(sa)} valores (S&P 500 + Nasdaq 100 + SOX)...')
    cs,vs,hs,ls=download_prices(sa,period='1y')
    if not cs.empty:
        cache['sp'] = (cs,vs,hs,ls)
        pickle.dump(cache, open(CACHE_FILE,'wb'))
    elif 'sp' in cache:
        print('  Usando cache S&P 500')
        cs,vs,hs,ls = cache['sp']
    sr=analyze_universe(sbs,bs,cs,vs,hs,ls,spy_healthy=spy_ok,spy_score=spy_score_continuo)
    sgs=calc_groups(sr,is_sp=True); print(f'  OK {len(sr)} valores')
    faltan_uni = sorted(set(sa) - {r['ticker'] for r in sr})  # PUNTO 26
    if faltan_uni: print(f'  Sin datos/analisis en universo: {", ".join(faltan_uni)}')
    ar=pr+sr; all_groups=sorted(pgs+sgs,key=lambda x:x['score'],reverse=True)

    # PUNTO 8 — Amplitud de mercado (informativa, sobre el universo amplio S&P500+Nasdaq100+SOX)
    print('\n▸ Calculando amplitud de mercado...')
    breadth = calc_market_breadth(cs, bs)
    breadth = anotar_tendencia_en_breadth(breadth, macro, ts)   # P37b — contexto temporal para el informe
    if breadth.get('pct_sobre_mm50') is not None:
        print(f'  {breadth.get("pct_sobre_mm20","?")}% sobre MM20 | {breadth["pct_sobre_mm50"]}% sobre MM50 | {breadth["pct_sobre_mm200"]}% sobre MM200 | '
              f'Avance/Descenso: {breadth["avance"]}/{breadth["descenso"]} | '
              f'Nuevos max/min 52s: {breadth["nuevos_max_52s"]}/{breadth["nuevos_min_52s"]} | '
              f'McClellan: {breadth.get("mcclellan","?")} | MM200 SPY: {breadth["pendiente_mm200_spy"]}')
    else:
        print('  Amplitud no disponible (datos insuficientes)')

    # Obtener fundamentales de valores con setup valido
    print('\n▸ Obteniendo datos fundamentales...')
    valid_tickers = list(set(
        v['ticker'] for v in sorted(ar, key=lambda x: x['score'] or 0, reverse=True)
        if v.get('entry_range', {}).get('entry_lo')
        and (v.get('entry_range', {}).get('rr') or 0) >= 2.0
        and (v.get('adx') or 0) >= 20
        and v.get('riesgo') != 'ALTO'
    ))[:20]
    fundamentales = get_fundamentals(valid_tickers) if valid_tickers else {}
    print(f'  OK {len(fundamentales)} valores con datos fundamentales')

    print('\n▸ Actualizando historico de setups...')
    values_sorted = sorted(ar, key=lambda x: x['score'] or 0, reverse=True)
    setups_history, evaluaciones = update_setups_history(values_sorted, all_groups)
    upload_to_github('setups_history.json', json.dumps(clean_nan(setups_history), ensure_ascii=False))
    if evaluaciones:
        # RECALIBRADO (10/07) — la estadistica de rendimiento se calcula SOLO sobre los
        # checkpoints [5,10,20] (horizontes fijos comparables); el total diario se informa aparte
        chk = [e for e in evaluaciones if e.get('checkpoint')]
        n_ok = sum(1 for e in chk if e['resultado'] == 'target')
        n_stop = sum(1 for e in chk if e['resultado'] == 'stop')
        n_ab = sum(1 for e in chk if e['resultado'] == 'abierto')
        print(f'  Evaluaciones diarias: {len(evaluaciones)} | Checkpoints 5/10/20: {len(chk)} '
              f'(Target: {n_ok} | Stop: {n_stop} | Abierto: {n_ab})')
        # PUNTO 10 + NUEVO (27/06, RECALIBRADO 06/07 y 10/07) — alertas con jerarquia y
        # deduplicadas por ticker para presentacion (data.json conserva todas)
        # PUNTO 40 — resoluciones primero: un stop alcanzado es mas urgente que cualquier alerta.
        _resoluciones = resoluciones_por_ticker(evaluaciones)
        if _resoluciones:
            print(f'  🎯 SETUPS RESUELTOS ({len(_resoluciones)}): stop u objetivo alcanzado')
            for _r in _resoluciones:
                _que = 'STOP' if _r['resultado'] == 'stop' else 'OBJETIVO'
                print(f'     {_r["ticker"]}: setup del {_r["fecha_setup"]} | {_que} alcanzado | '
                      f'entrada ${_r["precio_entrada"]} -> ${_r["precio_actual"]} | ret={_r["ret_pct"]}%')
        alertas_cmf = dedup_alertas_por_ticker([e for e in evaluaciones if e.get('alerta_cmf')])
        alertas_st = dedup_alertas_por_ticker([e for e in evaluaciones if e.get('alerta_supertrend')])
        pre_rotos = [e for e in evaluaciones if e.get('supertrend_pre_roto')]
        if alertas_st:
            print(f'  🔴 ALERTAS SUPERTREND ({len(alertas_st)} setups con ruptura posterior a la entrada):')
            for a in alertas_st:
                d = a.get('supertrend_dias')
                etiq = ('ruptura NUEVA (esta sesion)' if (d is None or d <= 1)
                        else f'ruptura persistente ({d} sesiones)')
                print(f'     {a["ticker"]}: setup del {a["fecha_setup"]} | '
                      f'Supertrend bajista, nivel=${a["supertrend_nivel"]}, {etiq} | ret={a["ret_pct"]}% | '
                      f'Señal de salida total')
        if pre_rotos:
            print(f'  ℹ️  {len({a["ticker"] for a in pre_rotos})} tickers con Supertrend ya bajista AL CREARSE (sin alerta; '
                  f'registrados en data.json con supertrend_pre_roto=true): '
                  + ', '.join(sorted({a["ticker"] for a in pre_rotos})))
        cmf_pre = [e for e in evaluaciones if e.get('cmf_pre_negativo')]
        if cmf_pre:
            print(f'  ℹ️  {len({a["ticker"] for a in cmf_pre})} tickers con CMF ya negativo AL CREARSE (sin alerta blanda; '
                  f'cmf_pre_negativo=true en data.json): '
                  + ', '.join(sorted({a["ticker"] for a in cmf_pre})))
        if alertas_cmf:
            print(f'  ⚠️  ALERTAS CMF ({len(alertas_cmf)} setups con distribucion confirmada, CMF<0 en 3+ sesiones):')
            for a in alertas_cmf:
                print(f'     {a["ticker"]}: setup del {a["fecha_setup"]} | '
                      f'CMF negativo {a.get("cmf_dias_negativo")} sesiones (actual={a["cmf_actual"]}) | ret={a["ret_pct"]}% | '
                      f'Considerar reducir posicion 50%')

    print('\n▸ Generando analisis Claude...')
    macro = macro if 'macro' in dir() else {}
    # P43 — anotar resoluciones con su fecha de PRIMERA deteccion, antes del analisis,
    # para que el prompt pueda detallar solo las recientes y resumir las antiguas.
    resoluciones_anotadas = anotar_resoluciones(evaluaciones, macro, ts)
    data_tmp={'timestamp':ts,'mode':'S&P500 + Watchlist','groups':all_groups,'values':sorted(ar,key=lambda x:x['score'] or 0,reverse=True),'spy_healthy':spy_ok,'macro':macro,'fundamentales':fundamentales,'breadth':breadth,'evaluaciones':evaluaciones,'resoluciones':resoluciones_anotadas}
    analisis='Analisis no disponible. Ejecuta Colab para generar el analisis con Claude.'
    if ANTHROPIC_KEY:
        try:
            import anthropic as ant_test
            client_test = ant_test.Anthropic(api_key=ANTHROPIC_KEY.strip())
            models = client_test.models.list()
            print(f'  Conexion Anthropic OK — modelos disponibles: {len(list(models))}')
        except Exception as e:
            print(f'  Test conexion: {e}')
        for intento in range(3):
            try:
                analisis=generate_analysis(data_tmp,ANTHROPIC_KEY); print('  OK'); break
            except Exception as e:
                print(f'  Intento {intento+1}/3 fallido: {e}')
                if intento < 2:
                    import time; time.sleep(20)
    else:
        print('  ANTHROPIC_KEY no configurada — analisis omitido')
    # PUNTO 36 — validar integridad del informe generado
    # ARREGLO (02/08/2026): antes pasaba valid_final, que es LOCAL de generate_analysis y
    # nunca existe aqui — la comprobacion de tickers quedaba desactivada en silencio.
    _valido, _avisos = validar_informe(analisis, ar)
    if _avisos:
        for av in _avisos:
            print(f'  AVISO informe: {av}')
    # PUNTO 55 — si el informe vino vacio, conservar el anterior en vez de pisarlo.
    analisis, _rescatado, _no_publicar_data = rescatar_analisis_anterior(analisis, _avisos)
    history=update_history(all_groups, all_values=sorted(ar,key=lambda x:x['score'] or 0,reverse=True))

    print('\n▸ Generando alertas Telegram...')
    generate_alerts(data_tmp,history,spy_ok,bench_series=bs)
    data={'timestamp':ts,'mode':'S&P500 + Watchlist','groups':all_groups,'values':sorted(ar,key=lambda x:x['score'] or 0,reverse=True),'analisis':analisis,'spy_healthy':spy_ok,'evaluaciones':evaluaciones,'resoluciones':resoluciones_anotadas,'spy_ret_1w':spy_ret_1w,'spy_ret_1m':spy_ret_1m,'macro':macro,'breadth':breadth,'fundamentales':fundamentales if "fundamentales" in vars() else {}}
    print('\n▸ Subiendo a GitHub...')
    # RECALIBRADO (11/07) — un solo commit para data+history (un despliegue de Pages, sin
    # carreras). setups_history.json mantiene su subida temprana individual a proposito:
    # es el checkpoint de seguridad ANTES del analisis de Claude y de las alertas, para que
    # un fallo en ese tramo no pierda el registro del dia. Total: 2 commits/ejecucion (antes 3).
    # PUNTO 24 — persistir la serie diaria de amplitud (dataset de la calibracion de septiembre)
    breadth_history = actualizar_breadth_history(breadth, macro, ts)
    ficheros_subida = {
        'data.json':    json.dumps(clean_nan(data), ensure_ascii=False),
        'history.json': json.dumps(clean_nan(history), ensure_ascii=False),
    }
    if breadth_history is not None:
        ficheros_subida['breadth_history.json'] = json.dumps(clean_nan(breadth_history), ensure_ascii=False)
    # CURVA DE TIPOS (01/08) — serie diaria en el mismo commit unico, sin commit extra.
    rates_history = actualizar_rates_history(macro, ts)
    if rates_history is not None:
        ficheros_subida['rates_history.json'] = json.dumps(clean_nan(rates_history), ensure_ascii=False)
    # PUNTO 39 — historico permanente de checkpoints (los de data.json son efimeros).
    checkpoints_history = actualizar_checkpoints_history(evaluaciones, macro, ts)
    if checkpoints_history is not None:
        ficheros_subida['checkpoints_history.json'] = json.dumps(clean_nan(checkpoints_history), ensure_ascii=False)
    # PUNTO 43 — historico permanente de resoluciones con su fecha de deteccion.
    resoluciones_history = actualizar_resoluciones_history(evaluaciones, macro, ts)
    if resoluciones_history is not None:
        ficheros_subida['resoluciones_history.json'] = json.dumps(clean_nan(resoluciones_history), ensure_ascii=False)
    # PUNTO 56 — serie diaria de put/call, en el mismo commit unico.
    putcall_history = actualizar_putcall_history(putcall, macro, ts)
    if putcall_history is not None:
        ficheros_subida['putcall_history.json'] = json.dumps(clean_nan(putcall_history), ensure_ascii=False)
    # PUNTO 68 — historico de alertas con su desenlace.
    alertas_history = actualizar_alertas_history(evaluaciones, macro, ts)
    if alertas_history is not None:
        ficheros_subida['alertas_history.json'] = json.dumps(clean_nan(alertas_history), ensure_ascii=False)
    # PUNTO 75 — registro permanente de setups: una fila por setup creado, para siempre.
    # setups_history[-1] es la entrada de HOY (update_setups_history la añade al final).
    _setups_hoy = (setups_history[-1].get('setups') if setups_history else None) or []
    setups_registro = actualizar_setups_registro(_setups_hoy, values_sorted, macro, ts)
    if setups_registro is not None:
        ficheros_subida['setups_registro.json'] = json.dumps(clean_nan(setups_registro), ensure_ascii=False)
    # P57 — serie diaria de spy_health score continuo, en el mismo commit unico.
    try:
        _spy_close_val = float(bs.iloc[-1]) if bs is not None and len(bs) > 0 else None
        _ma200_val = float(bs.rolling(200).mean().iloc[-1]) if bs is not None and len(bs) >= 200 else None
    except Exception:
        _spy_close_val = None; _ma200_val = None
    spy_health_history = actualizar_spy_health_history(spy_score_continuo, spy_ok, _spy_close_val, _ma200_val, ts)
    if spy_health_history is not None:
        ficheros_subida['spy_health_history.json'] = json.dumps(clean_nan(spy_health_history), ensure_ascii=False)
    # P65 — no pisar data.json con un vuelco incompleto (ver data_json_publicable).
    _publicable, _motivo = data_json_publicable()
    if not _publicable:
        ficheros_subida.pop('data.json', None)
        print(f'  🔴 {_motivo}')
    # P78 — tampoco pisarlo con el informe vacio cuando no se ha podido comprobar que hay
    # al otro lado. El resto de ficheros suben igual: tienen dedupe por fecha.
    if _no_publicar_data and 'data.json' in ficheros_subida:
        ficheros_subida.pop('data.json', None)
        print('  🔴 data.json NO SE SUBE: el informe vino vacio y la lectura del anterior '
              'fallo, asi que se conserva intacto lo que hubiera en el repo.')
    upload_files_to_github(ficheros_subida)
    print('\n═══════════════════════════════════════')
    print('  TOP 5 TEMAS')
    print('═══════════════════════════════════════')
    # P63: los temas no comparables (pocos miembros) van aparte, no compiten en el ranking.
    _comparables = [g for g in all_groups if g.get('tema_comparable')]
    _no_comp = [g for g in all_groups if not g.get('tema_comparable')]
    for i,g in enumerate(_comparables[:5]): print(f'  #{i+1}  {g["score"]:5.1f}  {g["group"]}')
    if _no_comp:
        print(f'  (fuera de ranking, menos de {MIN_MIEMBROS_TEMA} valores: '
              + ', '.join(f'{g["group"]} {g["score"]:.1f} n={g["n"]}' for g in _no_comp[:5]) + ')')
    bos=[r for r in ar if r['breakout']]
    print(f'\n  Rupturas: {len(bos)} | SPY saludable: {spy_ok}')
    print(f'\n  Web: https://Carolo-III.github.io/scanner-temas')
    print('═══════════════════════════════════════\n')

if __name__ == '__main__':
    main()
