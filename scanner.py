"""
scanner.py — Script autonomo del Scanner de Temas Emergentes
Se ejecuta automaticamente via GitHub Actions cada noche de lunes a viernes.
Tambien puede ejecutarse manualmente: python scanner.py
"""

import math, warnings, json, base64, os
from datetime import datetime
from pathlib import Path
import numpy as np
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
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '5100549189')

PERSONAL_WATCHLIST = {
    'Semiconductores': ['ARM','AMD','MU','MTSI','POET','SMCI'],
    'Infraestructura AI': ['ANET','VRT','APLD','CORZ','IREN','CIFR','CRWV'],
    'Espacio y Defensa': ['RKLB','LUNR','ASTS','KTOS','BWXT'],
    'Cuantica': ['IONQ'],
    'Robotica AI': ['BBAI','TSLA'],
    'Minerales': ['MP','UAMY'],
    'Biotech': ['VKTX','ACRV','IBRX'],
    'Hardware': ['SNDK'],
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

def spy_health(bench_close):
    try:
        p = bench_close.dropna()
        if len(p) < 200: return True
        current = float(p.iloc[-1]); ma200 = float(p.rolling(200).mean().iloc[-1])
        healthy = current > ma200
        print(f'  SPY: ${round(current,2)} | MM200: ${round(ma200,2)} | {"ALCISTA" if healthy else "BAJISTA"}')
        return healthy
    except: return True

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
    """
    result = {'pct_sobre_mm50': None, 'pct_sobre_mm200': None, 'nuevos_max_52s': None,
              'nuevos_min_52s': None, 'avance': None, 'descenso': None,
              'pendiente_mm200_spy': None, 'n_valores': None}
    try:
        precios = universe_close.dropna(axis=1, how='all')
        if precios.empty:
            return result
        result['n_valores'] = precios.shape[1]

        # % sobre MM50 / MM200 (requiere al menos 50/200 sesiones de historico)
        if len(precios) >= 50:
            mm50 = precios.rolling(50).mean()
            ultimo = precios.iloc[-1]
            ultimo_mm50 = mm50.iloc[-1]
            valid50 = ultimo.notna() & ultimo_mm50.notna()
            if valid50.sum() > 0:
                result['pct_sobre_mm50'] = round((ultimo[valid50] > ultimo_mm50[valid50]).mean() * 100, 1)
        if len(precios) >= 200:
            mm200 = precios.rolling(200).mean()
            ultimo = precios.iloc[-1]
            ultimo_mm200 = mm200.iloc[-1]
            valid200 = ultimo.notna() & ultimo_mm200.notna()
            if valid200.sum() > 0:
                result['pct_sobre_mm200'] = round((ultimo[valid200] > ultimo_mm200[valid200]).mean() * 100, 1)

        # Nuevos maximos / minimos de 52 semanas (ventana de hasta 252 sesiones disponibles)
        ventana = min(252, len(precios))
        if ventana >= 20:  # umbral minimo para que el dato tenga algun sentido
            max_52s = precios.iloc[-ventana:].max()
            min_52s = precios.iloc[-ventana:].min()
            ultimo = precios.iloc[-1]
            validez = ultimo.notna() & max_52s.notna() & min_52s.notna()
            result['nuevos_max_52s'] = int((ultimo[validez] >= max_52s[validez]).sum())
            result['nuevos_min_52s'] = int((ultimo[validez] <= min_52s[validez]).sum())

        # Avance / descenso (variacion de la ultima sesion vs la anterior)
        if len(precios) >= 2:
            ret_diario = precios.pct_change().iloc[-1]
            ret_diario = ret_diario.dropna()
            result['avance'] = int((ret_diario > 0).sum())
            result['descenso'] = int((ret_diario < 0).sum())

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
    except Exception as e:
        print(f'  Aviso: calc_market_breadth fallo parcialmente: {e}')
    return result

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
                    chg     = round(current - prev, 2)
                    chg_pct = round((current/prev - 1)*100, 1)
                    result[tk] = {'current': current, 'prev': prev, 'chg': chg, 'chg_pct': chg_pct}
        return result
    except Exception as e:
        print(f'  Error macro: {e}')
        return {}

def get_fundamentals(tickers):
    result = {}
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info
            result[tk] = {
                'per_trailing':  round(info.get('trailingPE',  0) or 0, 1),
                'per_forward':   round(info.get('forwardPE',   0) or 0, 1),
                'peg':           round(info.get('pegRatio',    0) or 0, 2),
                'ev_ebitda':     round(info.get('enterpriseToEbitda', 0) or 0, 1),
                'margen_neto':   round((info.get('profitMargins', 0) or 0) * 100, 1),
                'roe':           round((info.get('returnOnEquity', 0) or 0) * 100, 1),
                'deuda_equity':  round(info.get('debtToEquity', 0) or 0, 2),
                'rev_growth':    round((info.get('revenueGrowth', 0) or 0) * 100, 1),
                'eps_trailing':  round(info.get('trailingEps', 0) or 0, 2),
                'eps_fwd':       round(info.get('forwardEps', 0) or 0, 2),
                'eps_growth':    round((info.get('earningsGrowth', 0) or 0) * 100, 1),
                'sector':        info.get('sector', ''),
                'mkt_cap_b':     round((info.get('marketCap', 0) or 0) / 1e9, 1),
            }
        except Exception as e:
            result[tk] = {}
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
            except Exception:
                pass
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
            except Exception:
                pass
        except Exception:
            pass
        result[tk] = entry
    return result

def rs_score(t, b, w):
    try:
        t, b = t.dropna(), b.dropna()
        if len(t) < w or len(b) < w: return None
        return round((t.iloc[-1]/t.iloc[-w]-1)*100 - (b.iloc[-1]/b.iloc[-w]-1)*100, 2)
    except: return None

def volume_zscore(v, w=20):
    try:
        v = v.dropna()
        if len(v) < w+1: return None
        m, s = v.iloc[-(w+1):-1].mean(), v.iloc[-(w+1):-1].std()
        return round((v.iloc[-1]-m)/s, 2) if s else 0.0
    except: return None

def detect_breakout(p, v, lb=50, vm=1.4):
    try:
        p, v = p.dropna(), v.dropna()
        if len(p) < lb: return {'breakout': False, 'days_ago': None, 'breakout_level': None}
        for d in range(1, 11):
            bh = p.iloc[-(lb+d):-(d+5)].max(); bvm = v.iloc[-(lb+d):-(d+5)].mean()
            if p.iloc[-d] > bh and v.iloc[-d] > vm*bvm:
                return {'breakout': True, 'days_ago': d, 'breakout_level': round(float(bh), 2)}
        return {'breakout': False, 'days_ago': None, 'breakout_level': round(float(p.iloc[-lb:-5].max()), 2)}
    except: return {'breakout': False, 'days_ago': None, 'breakout_level': None}

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
    except: return {'ma20':False,'ma50':False,'ma200':False}

def calc_atr(high, low, close, period=14):
    try:
        h = high.squeeze().dropna(); l = low.squeeze().dropna(); c = close.squeeze().dropna()
        if len(c) < period + 1: return None
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return round(float(atr), 4)
    except: return None

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
    except: return None

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
    except: return None, None, None

def calc_adx(high, low, close, period=14):
    try:
        h = high.squeeze().dropna(); l = low.squeeze().dropna(); c = close.squeeze().dropna()
        if len(c) < period * 2: return None
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        dm_plus  = (h - h.shift(1)).clip(lower=0)
        dm_minus = (l.shift(1) - l).clip(lower=0)
        dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
        dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
        atr14 = tr.rolling(period).mean()
        di_plus  = 100 * (dm_plus.rolling(period).mean()  / atr14)
        di_minus = 100 * (dm_minus.rolling(period).mean() / atr14)
        dx = 100 * ((di_plus - di_minus).abs() / (di_plus + di_minus))
        adx = dx.rolling(period).mean()
        return round(float(adx.iloc[-1]), 1)
    except: return None

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
    except: return None, None, None

def calc_cmf(high, low, close, volume, period=20):
    try:
        h = high.squeeze().dropna(); l = low.squeeze().dropna()
        c = close.squeeze().dropna(); v = volume.squeeze().dropna()
        if len(c) < period: return None
        hl_range = (h - l).replace(0, float('nan'))
        mfm = ((c - l) - (h - c)) / hl_range
        mfm = mfm.fillna(0)  # rango cero = sin presion compradora ni vendedora
        mfv = mfm * v
        vol_sum = v.rolling(period).sum()
        cmf = mfv.rolling(period).sum() / vol_sum.replace(0, float('nan'))
        val = float(cmf.iloc[-1])
        import math
        if math.isnan(val) or math.isinf(val): return None
        return round(val, 3)
    except: return None

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
    except: return None

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
    MEDIO: SCT entre 70-79 O RSI entre 68-70 O CMF entre 0.05-0.09 O ADX entre 17-19
    ALTO:  cualquier otro caso — se descarta antes de llegar a Claude (no entra en valid[])

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
    if 17 <= adx <= 19: motivos_medio.append(f'ADX en 17-19 ({adx})')
    if motivos_medio:
        return 'MEDIO', ' + '.join(motivos_medio)

    return 'ALTO', 'ninguna condicion de BAJO o MEDIO se cumple (no debe llegar a Claude)'

def price_stats(p, h, v_series):
    try:
        p=p.dropna()
        if len(p)<5: return {}
        current=float(p.iloc[-1]); w52=min(252,len(p)); high52=float(p.iloc[-w52:].max()); low52=float(p.iloc[-w52:].min())
        return {'price':round(current,2),'high52':round(high52,2),'low52':round(low52,2),
                'pct_from_high':round((current/high52-1)*100,1),
                'ret_1w':round((current/p.iloc[-5]-1)*100,1) if len(p)>=5 else None,
                'ret_1m':round((current/p.iloc[-20]-1)*100,1) if len(p)>=20 else None,
                'ret_3m':round((current/p.iloc[-65]-1)*100,1) if len(p)>=65 else None,
                'low5':round(float(p.iloc[-5:].min()),2) if len(p)>=5 else round(current*0.95,2)}
    except: return {}

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
    return target_final, target_parcial, rr_final

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
            # PUNTO 0+4: techo = maximo 52s; objetivo derivado de stop y R/B, nunca el techo directo
            techo = high52
            target, target_parcial, rr = _objetivo_escalonado(elo, ehi, stop, techo, ma200=ma200, high_recent=bl)
            tipo = 'Ruptura activa'
            # PUNTO 5: si la entrada exige un precio por encima del precio actual, es ruptura pendiente
            if elo > price:
                tipo = 'Ruptura pendiente'
            return {'tipo':tipo,'entry_lo':elo,'entry_hi':ehi,'stop':stop,'target':target,
                    'target_parcial':target_parcial,'rr':rr}
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
            # PUNTO 0+4: techo = maximo 52s; objetivo derivado de stop y R/B
            techo = high52
            target, target_parcial, rr = _objetivo_escalonado(elo, ehi, stop, techo, ma200=ma200, high_recent=None)
            # Caso 1: entry_lo <= price <= entry_hi -> operativo ahora ('Pullback MM50')
            # Caso 2: price < entry_lo -> esperar a que baje hasta la zona ('Ruptura pendiente')
            tipo = 'Pullback MM50' if price >= elo else 'Ruptura pendiente'
            return {'tipo':tipo,'entry_lo':elo,'entry_hi':ehi,'stop':stop,'target':target,
                    'target_parcial':target_parcial,'rr':rr}
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
            # PUNTO 0+4: techo = maximo 52s (sustituye al antiguo ma200*1.20 fijo)
            techo = high52
            target, target_parcial, rr = _objetivo_escalonado(elo, ehi, stop, techo, ma200=None, high_recent=None)
            tipo = 'Pullback MM200' if price >= elo else 'Ruptura pendiente'
            return {'tipo':tipo,'entry_lo':elo,'entry_hi':ehi,'stop':stop,'target':target,
                    'target_parcial':target_parcial,'rr':rr}
        if bl and price<=bl*1.03:
            return {'tipo':'En vigilancia','entry_lo':None,'entry_hi':None,'stop':None,'target':None,
                    'target_parcial':None,'rr':None,'nivel_ruptura':round(bl,2)}
        return {'tipo':'Sin setup','entry_lo':None,'entry_hi':None,'stop':None,'target':None,
                'target_parcial':None,'rr':None}
    except: return {}

def composite_score(r4, r13, vz, bo, ma, spy_healthy=True):
    s=0
    if r4 is not None: s+=30*min(1,max(0,(r4+30)/60))
    if r13 is not None: s+=20*min(1,max(0,(r13+50)/100))
    if vz is not None: s+=25*min(1,max(0,(vz+1)/4))
    if bo: s+=15
    if ma: s+=10*(sum([ma.get('ma20',False),ma.get('ma50',False),ma.get('ma200',False)])/3)
    if not spy_healthy: s=round(s*0.70,1)
    return round(s,1)

def analyze_universe(grps, bench, close_df, vol_df, high_df=None, low_df=None, spy_healthy=True):
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
            except: continue
            h=high_df[tk] if high_df is not None and tk in high_df.columns else p
            l=low_df[tk] if low_df is not None and tk in low_df.columns else p
            r4=rs_score(p,bench,20); r13=rs_score(p,bench,65)
            vz=volume_zscore(v) if not v.empty else None
            bi=detect_breakout(p,v) if not v.empty else {'breakout':False,'days_ago':None,'breakout_level':None}
            mah=ma_health(p); ps=price_stats(p,h,v)
            atr_pre=calc_atr(h,l,p) if not h.empty else None
            er=calc_entry_range(ps,bi,mah,atr_val=atr_pre)
            sc=composite_score(r4,r13,vz,bi['breakout'],mah,spy_healthy)
            # Score de Confirmacion Tecnica (SCT)
            atr_val = atr_pre  # ya calculado antes
            rsi_val = calc_rsi(p)
            macd_val, macd_sig, macd_hist = calc_macd(p)
            adx_val = calc_adx(h, l, p) if not h.empty else None
            pct_b, squeeze, bw = calc_bollinger(p)
            cmf_val = calc_cmf(h, l, p, v) if not h.empty and not v.empty else None
            obv_val = calc_obv(p, v) if not v.empty else None
            sct_val = calc_sct(adx_val, rsi_val, macd_hist, pct_b, squeeze, cmf_val, obv_val)
            # PUNTO 1 — RIESGO calculado en Python con la formula exacta (BAJO/MEDIO/ALTO)
            # Devuelve tambien el motivo exacto que dispara la clasificacion, para que Claude
            # lo use literalmente en el informe en vez de inferir una causa no verificada.
            riesgo_val, riesgo_motivo = calc_riesgo(sct_val, rsi_val, cmf_val, adx_val)
            res.append({'ticker':tk,'group':gn,'rs_4w':r4,'rs_13w':r13,'vol_z':vz,
                'breakout':bi['breakout'],'days_ago':bi['days_ago'],'breakout_level':bi.get('breakout_level'),
                'ma20':mah.get('ma20',False),'ma50':mah.get('ma50',False),'ma200':mah.get('ma200',False),
                'pct_ma50':mah.get('pct_ma50'),'pct_ma200':mah.get('pct_ma200'),
                'ma50_val':mah.get('ma50_val'),'ma200_val':mah.get('ma200_val'),
                'price':ps.get('price'),'high52':ps.get('high52'),'low52':ps.get('low52'),
                'pct_from_high':ps.get('pct_from_high'),'ret_1w':ps.get('ret_1w'),
                'ret_1m':ps.get('ret_1m'),'ret_3m':ps.get('ret_3m'),'entry_range':er,'score':sc,
                'sct':sct_val,'rsi':rsi_val,'adx':adx_val,'macd_hist':macd_hist,
                'cmf':cmf_val,'atr':atr_val,'squeeze':squeeze,'pct_b':pct_b,
                'riesgo':riesgo_val,'riesgo_motivo':riesgo_motivo})
    return res

def calc_groups(res, is_sp=False):
    gs={}
    for r in res: gs.setdefault(r['group'],[]).append(r)
    out=[]
    for gn,mb in gs.items():
        scores=[m['score'] for m in mb if m['score'] is not None]
        r4s=[m['rs_4w'] for m in mb if m['rs_4w'] is not None]
        mb_s=sorted(mb,key=lambda x:x['score'] or 0,reverse=True)
        out.append({'group':gn,'score':round(np.mean(scores),1) if scores else 0,
                    'rs_mean':round(np.mean(r4s),1) if r4s else 0,
                    'breakouts':sum(1 for m in mb if m['breakout']),
                    'n':len(mb),'is_sp':is_sp,'top3':mb_s[:3],'members':mb_s})
    return sorted(out,key=lambda x:x['score'],reverse=True)

def generate_analysis(data, anthropic_key):
    import anthropic as ant
    client=ant.Anthropic(api_key=anthropic_key.strip())
    groups=data['groups']; values=data['values']; ts=data['timestamp']; spy_ok=data.get('spy_healthy',True)
    strong=[g for g in groups if g['score']>=70]; emerging=[g for g in groups if 50<=g['score']<70]; weak=[g for g in groups if g['score']<30]
    # Datos macro para el prompt
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
            f'- US10Y (bono 10a): {usy.get("current","?")}% (cambio: {usy.get("chg","?")} pb)\n\n'
        )

    # PUNTO 8 — Amplitud de mercado (informativa, sobre universo amplio S&P500+Nasdaq100+SOX)
    breadth_txt = ''
    breadth = data.get('breadth', {})
    if breadth and breadth.get('pct_sobre_mm50') is not None:
        n_val = breadth.get('n_valores', '?')
        breadth_txt = (
            f'AMPLITUD DE MERCADO (sobre {n_val} valores del universo S&P500+Nasdaq100+SOX):\n'
            f'- % de valores por encima de su MM50: {breadth["pct_sobre_mm50"]}%\n'
            f'- % de valores por encima de su MM200: {breadth["pct_sobre_mm200"]}%\n'
            f'- Nuevos maximos de 52 semanas: {breadth.get("nuevos_max_52s","?")} | '
            f'Nuevos minimos de 52 semanas: {breadth.get("nuevos_min_52s","?")}\n'
            f'- Avance/Descenso (sesion actual): {breadth.get("avance","?")} valores suben, '
            f'{breadth.get("descenso","?")} bajan\n'
            f'- Pendiente MM200 del SPY: {breadth.get("pendiente_mm200_spy","?")}\n\n'
        )

    summary=f'DATOS — {ts}\nSPY sobre MM200: {"SI" if spy_ok else "NO"}\n\n' + macro_txt + breadth_txt + 'FUERTES:\n'
    for g in strong[:5]:
        leaders=', '.join(m['ticker']+'('+str(m['score'])+')' for m in g['top3'])
        summary+=f'- {g["group"]}: {g["score"]} | RS:{g["rs_mean"]}% | {leaders}\n'
    summary+='\nEMERGENTES:\n'
    for g in emerging[:5]: summary+=f'- {g["group"]}: {g["score"]} | RS:{g["rs_mean"]}%\n'
    fundamentales = data.get('fundamentales', {})
    # PUNTO 1 — descartar RIESGO=ALTO antes de pasar a Claude (no llegan setups de riesgo alto)
    # PUNTO 2 — filtro ADX>=20 obligatorio en setups validos
    valid=[v for v in values
           if v.get('entry_range',{}).get('entry_lo')
           and (v.get('entry_range',{}).get('rr') or 0)>=2.0
           and (v.get('sct') or 0)>=40
           and (v.get('adx') or 0)>=20
           and v.get('riesgo') != 'ALTO']
    # PUNTO 3 — ranking calculado en Python con tope R/B=5, no delegado a Claude.
    # Formula: SCT 40% + R/B normalizado (tope 5) 20% + fuerza sectorial del grupo 20% + CMF 20%
    group_score_map = {g['group']: g['score'] for g in groups}
    for v in valid:
        sct_n = (v.get('sct') or 0) / 100.0
        rb_n = min((v.get('entry_range',{}).get('rr') or 0), 5.0) / 5.0
        sector_n = (group_score_map.get(v['group'], 0)) / 100.0
        cmf_raw = v.get('cmf')
        cmf_n = min(max((cmf_raw if cmf_raw is not None else 0) / 0.3, 0), 1)  # 0.3 CMF ~ tope practico
        v['ranking'] = round((sct_n*0.40 + rb_n*0.20 + sector_n*0.20 + cmf_n*0.20) * 100, 1)
    valid = sorted(valid, key=lambda x: x.get('ranking', 0), reverse=True)

    # PUNTO 7 — earnings proximos + noticias, SOLO para los candidatos ya ordenados por ranking,
    # consultando un margen extra (hasta 8) para poder reemplazar los descartados por earnings
    # y seguir entregando hasta 5 setups cuando haya candidatos suficientes.
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
    valid = valid_final

    summary+='\nSETUPS VALIDOS CON ANALISIS FUNDAMENTAL (usa EXACTAMENTE estos niveles, RIESGO y RANKING ya calculados — no los recalcules):\n'
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
        summary+=f'- {v["ticker"]} ({v["group"]}): ${v.get("price","?")} | {er["tipo"]} | Entrada:${er["entry_lo"]}-${er["entry_hi"]} | Stop:${er["stop"]}'
        # PUNTO 4 — objetivo escalonado: parcial y final
        if er.get('target_parcial'): summary+=f' | Obj.parcial:${er["target_parcial"]}'
        summary+=f' | Obj.final:${er["target"]}'
        if ind_str: summary+=f' | {ind_str}'
        if er.get('rr'): summary+=f' | R/B:1:{er["rr"]}'
        summary+=f' | RIESGO:{v.get("riesgo","?")} (motivo exacto: {v.get("riesgo_motivo","?")}) | RANKING:{v.get("ranking","?")}'
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
            if f_parts: summary+=f'        FUND: {" | ".join(f_parts)}\n'
        # PUNTO 7 — noticias recientes (max 3, solo si existen)
        if v.get('noticias'):
            noticias_str = ' / '.join(f'"{n["titulo"]}" ({n["fuente"]}, {n["fecha"]})' for n in v['noticias'])
            summary+=f'        NOTICIAS: {noticias_str}\n'
        summary+='\n'
    if descartados_por_earnings:
        summary+='\nDESCARTADOS POR EARNINGS INMINENTE (no incluir en ENTRADAS, se puede mencionar brevemente si aporta contexto):\n'
        for v in descartados_por_earnings:
            summary+=f'- {v["ticker"]}: earnings en {v["dias_earnings"]} dias ({v["earnings_date"]})\n'
    extendidos=[v for v in values if v.get('entry_range',{}).get('tipo','').startswith('Extendido')]
    if extendidos:
        summary+='\nEXTENDIDOS:\n'
        for v in extendidos[:5]:
            er = v['entry_range']
            summary+=f'- {v["ticker"]}: ${v.get("price","?")} | {er.get("nota","Extendido")}'
            # PUNTO 6 — trigger concreto de entrada futura
            if er.get('trigger'): summary+=f' | TRIGGER: {er["trigger"]}'
            summary+='\n'
    summary+=f'\nFuertes:{len(strong)} Emergentes:{len(emerging)} Sin momentum:{len(weak)} Setups:{len(valid)}\n'
    aviso='ATENCION: SPY bajo MM200. Mercado bajista. ' if not spy_ok else ''
    prompt = (
        'Analista tecnico momentum. Informe ejecutivo espanol directo. ' + aviso +
        'Estructura: 1.MERCADO 2.TEMAS PRIORITARIOS 3.ENTRADAS(niveles exactos) 4.EXTENDIDOS 5.CONCLUSION. La CONCLUSION es OBLIGATORIO que termine con un parrafo separado titulado exactamente **RIESGOS DEL ESCENARIO** que liste en 3-4 puntos concisos que condiciones invalidarian la tesis alcista actual: perdida de MM200 por SPY, perdida de liderazgo sectorial, VIX elevado, deterioro de amplitud de mercado, resultados empresariales negativos proximos. Este parrafo NO es opcional. '
        'PUNTO 8 — AMPLITUD DE MERCADO: si el bloque AMPLITUD DE MERCADO aparece en los datos, usalo en la seccion MERCADO para matizar el regimen general (SPY>MM200 es una sola variable; la amplitud dice si ese movimiento esta respaldado por la mayoria de valores o solo por unos pocos). Interpreta los datos asi: % sobre MM50/MM200 por debajo del 50% mientras el SPY esta alcista indica un "rally estrecho" (pocos valores liderando, mercado fragil bajo la superficie); % por encima del 60-70% indica amplitud sana. Si nuevos minimos de 52 semanas superan a los nuevos maximos, es una señal de deterioro interno aunque el indice general suba. Esta informacion es CONTEXTO, no debe usarse para invalidar ni recalcular el RIESGO de los setups individuales (que depende solo de SCT/RSI/CMF/ADX del valor) — su unico rol es matizar la lectura del regimen de mercado en la seccion 1 y, si aplica, dar contenido especifico y verificable al punto de "deterioro de amplitud de mercado" en RIESGOS DEL ESCENARIO en lugar de mencionarlo de forma generica. '
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
        'PUNTO 6 — EXTENDIDOS CON DISTANCIA EXTREMA: si el campo TRIGGER de un valor en EXTENDIDOS empieza con "AVISO:", significa que la distancia hasta su MM50 es tan grande que el trigger de reentrada exigiria una caida brusca, probablemente incompatible con las condiciones de RSI/CMF exigidas. En ese caso NO redactes el trigger como un plan de reentrada normal ("vigilar cuando..."): explica que la corrección necesaria seria tan severa que invalidaria las propias condiciones de confirmacion, y que el valor debe tratarse como descartado salvo cambio sustancial de contexto — no como una oportunidad pendiente de vigilancia activa. '
        'PUNTO 7 — EARNINGS: si un setup incluye "EARNINGS EN X DIAS — AVISO OBLIGATORIO", debes mencionarlo explicitamente en el parrafo de ese setup con una advertencia clara: el informe de resultados puede provocar un gap de precio que el stop tecnico no protege (el stop se ejecuta al precio de apertura siguiente, no al nivel fijado). Sugiere reducir el tamaño de posicion o esperar a que pase el informe si el horizonte del trade puede solaparse con esa fecha. Si aparecen tickers en la lista "DESCARTADOS POR EARNINGS INMINENTE", debes mencionarlos brevemente en EXTENDIDOS o en una nota aparte como candidatos tecnicamente validos pero descartados por proximidad de resultados, indicando SIEMPRE los dias y la fecha exacta de earnings tal y como aparecen en los datos (ej. "earnings en 3 dias, el 2026-06-20") — nunca los menciones sin esa fecha concreta, y nunca les asignes niveles de entrada. Si un setup tiene noticias recientes (campo NOTICIAS), incorpora el titular mas relevante de forma natural en el texto si aporta contexto util (catalizador, riesgo, confirmacion), sin listarlas todas mecanicamente. '
        'Cada setup valido incluye un objetivo PARCIAL (resistencia inmediata, primera toma de beneficios parcial) y un objetivo FINAL (techo tecnico acotado a R/B 1:5 maximo). Menciona ambos: el parcial como punto donde valorar asegurar parte de la posicion, el final como objetivo de cierre completo. '
        'Ejemplos de tono correcto: '
        '"IONQ cotiza con RSI en 65, zona de momentum optimo sin sobrecompra, y ADX en 42, lo que confirma que la tendencia alcista tiene solidez suficiente para nuevas entradas." '
        '"El SCT de 78 sobre 100 indica que varios indicadores tecnicos apuntan en la misma direccion, lo que reduce el riesgo de una senal falsa." '
        'Para interpretar indicadores: RSI 55-70=momentum optimo, RSI>70=sobrecomprado con cautela; '
        'ADX>40=tendencia muy fuerte, ADX>25=tendencia confirmada, ADX<20=mercado lateral evitar (ya excluido del scanner); '
        'CMF>0.1=presion compradora sostenida (no afirmes que son instituciones, di acumulacion o presion compradora), CMF<0=presion vendedora; SCT>70=confirmacion tecnica alta; '
        'SQUEEZE=compresion de volatilidad previa a ruptura potente. '
        'FORMATO: en la seccion ENTRADAS, separa cada recomendacion con un salto de linea — una recomendacion por parrafo independiente. Escribe siempre el ticker con el simbolo dolar delante: $AZTA, $TECH, $CDW, etc. '
        'Al final de la seccion ENTRADAS incluye una tabla comparativa con columnas: TICKER | SETUP | R/B | SCT | RIESGO | RANKING. RIESGO y RANKING vienen YA CALCULADOS en los datos (campos RIESGO: y RANKING: de cada setup) — usalos exactamente como aparecen, NO los recalcules ni apliques ninguna formula propia sobre ellos. Cada setup incluye ademas "motivo exacto" entre parentesis junto al RIESGO: esa es la UNICA causa real que la formula evaluo para clasificarlo como BAJO o MEDIO. Cuando expliques por que un setup tiene ese RIESGO, usa literalmente ese motivo (reformulado en prosa natural, no copiado palabra por palabra) — NUNCA inventes ni infieras una causa distinta aunque suene plausible (por ejemplo, no atribuyas el RIESGO al tipo de setup —ruptura pendiente, pullback, etc.— salvo que el motivo exacto mencione ADX o RSI relacionados con eso; el tipo de setup y el RIESGO son cosas independientes). '
        'Aviso final obligatorio: "Este analisis no constituye asesoramiento financiero."\n\n' + summary
    )
    import httpx
    msg=client.messages.create(model='claude-opus-4-8',max_tokens=4000,messages=[{'role':'user','content':prompt}], timeout=httpx.Timeout(120.0, connect=30.0))
    return next(b.text for b in msg.content if hasattr(b, 'text'))

def get_github_file(filename):
    url=f'https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{filename}'
    r=requests.get(url,headers={'Authorization':f'token {GITHUB_TOKEN}'})
    if r.status_code==200:
        return json.loads(base64.b64decode(r.json()['content']).decode('utf-8')),r.json()['sha']
    return None,None

def update_setups_history(values, all_groups):
    today = datetime.now(madrid).strftime('%Y-%m-%d')
    history, _ = get_github_file('setups_history.json')
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
            })

    # Eliminar entrada de hoy si existe y añadir nueva
    history = [h for h in history if h['date'] != today]
    history.append({'date': today, 'setups': setups_hoy})
    history = sorted(history, key=lambda x: x['date'])[-30:]  # 30 dias

    # Evaluar setups anteriores con precios actuales
    current_prices = {v['ticker']: v.get('price') for v in values if v.get('price')}
    evaluaciones = []
    for entrada in history[:-1]:  # todos menos hoy
        dias = (datetime.now(madrid).date() - datetime.strptime(entrada['date'], '%Y-%m-%d').date()).days
        if dias not in [5, 10, 20]: continue
        for s in entrada['setups']:
            tk = s['ticker']
            precio_actual = current_prices.get(tk)
            if not precio_actual: continue
            precio_entrada = (s['entry_lo'] + s['entry_hi']) / 2
            ret_pct = round((precio_actual / precio_entrada - 1) * 100, 1)
            stop_tocado = precio_actual <= s['stop']
            target_tocado = precio_actual >= s['target']
            resultado = 'stop' if stop_tocado else ('target' if target_tocado else 'abierto')
            evaluaciones.append({
                'fecha_setup': entrada['date'],
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
            })

    return history, evaluaciones

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
    content_b64=base64.b64encode(content.encode('utf-8') if isinstance(content,str) else content).decode('utf-8')
    payload={'message':f'Actualizar {filename} - {datetime.now(madrid).strftime("%d/%m/%Y %H:%M")}','content':content_b64}
    if sha: payload['sha']=sha
    r=requests.put(url,headers=headers,json=payload)
    if r.status_code in [200,201]:
        print(f'  OK {filename}')
    else:
        print(f'  Error {filename}: HTTP {r.status_code} — {r.json().get("message","sin detalle")}')

def update_history(all_groups, all_values=None):
    today=datetime.now(madrid).strftime('%Y-%m-%d')
    history,_=get_github_file('history.json')
    if history is None: history=[]
    history=[h for h in history if h['date']!=today]
    entry={'date':today,'scores':{g['group']:g['score'] for g in all_groups},'ranks':{g['group']:i+1 for i,g in enumerate(all_groups)}}
    if all_values:
        entry['ma50_state']={v['ticker']:v.get('ma50',False) for v in all_values}
    # Guardar lider de cada grupo
    entry['leaders']={g['group']:g['top3'][0]['ticker'] if g['top3'] else None for g in all_groups}
    history.append(entry)
    return sorted(history,key=lambda x:x['date'])[-7:]

def send_telegram(message):
    if not TELEGRAM_TOKEN:
        print('  TELEGRAM_TOKEN no configurado — alerta omitida')
        return
    for intento in range(3):
        try:
            r=requests.post(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',json={'chat_id':TELEGRAM_CHAT_ID,'text':message},timeout=30)
            if r.status_code==200:
                print('  OK Telegram'); return
            else:
                print(f'  Error Telegram: {r.status_code}')
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
    sp_bo=[v for v in values if v['breakout'] and v['ticker'] not in wl and v.get('days_ago')==1 and (v.get('score') or 0)>=65 and (v.get('entry_range',{}).get('rr') or 0)>=2.0]
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
            msg = f'DIVERGENCIA SPY/AMPLITUD — {ts}\n\n'
            msg += f'SPY cerca de maximos de 52 semanas (${round(spy_current,2)})\n'
            msg += f'Solo {temas_verdes} de {len(groups)} temas en zona fuerte ({round(pct_verdes*100,0):.0f}%)\n'
            msg += 'El mercado sube concentrado en pocas megacaps. Extremar cautela.'
            msgs.append(msg)
    except Exception as e:
        pass

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
        for m in msgs: send_telegram(m)
    else:
        print('  Sin alertas nuevas hoy')

def main():
    print('\n═══════════════════════════════════════')
    print('  SCANNER DE TEMAS EMERGENTES')
    ts=datetime.now(madrid).strftime('%d/%m/%Y %H:%M')
    print(f'  {ts}\n═══════════════════════════════════════\n')
    import pickle, os
    CACHE_FILE = '/tmp/scanner_cache.pkl'
    cache = {}

    print('▸ SPY...')
    bc,bv,bh,bl=download_prices(['SPY'],period='1y'); bs=bc['SPY']; spy_ok=spy_health(bs)

    # Comparativa SPY semanal
    try:
        spy_ret_1w = round((float(bs.iloc[-1]) / float(bs.iloc[-5])  - 1) * 100, 1)
        spy_ret_1m = round((float(bs.iloc[-1]) / float(bs.iloc[-20]) - 1) * 100, 1)
    except:
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

    print(f'\n▸ Watchlist ({len(pt)} valores)...')
    cp,vp,hp,lp=download_prices(pt+['SPY'],period='1y')
    if not cp.empty:
        cache['wl'] = (cp,vp,hp,lp)
        pickle.dump(cache, open(CACHE_FILE,'wb'))
    elif os.path.exists(CACHE_FILE):
        print('  Usando cache de descarga anterior')
        cache = pickle.load(open(CACHE_FILE,'rb'))
        cp,vp,hp,lp = cache.get('wl',(cp,vp,hp,lp))
    pr=analyze_universe(PERSONAL_WATCHLIST,bs,cp,vp,hp,lp,spy_healthy=spy_ok)
    pgs=calc_groups(pr,is_sp=False); print(f'  OK {len(pr)} valores')

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
    sr=analyze_universe(sbs,bs,cs,vs,hs,ls,spy_healthy=spy_ok)
    sgs=calc_groups(sr,is_sp=True); print(f'  OK {len(sr)} valores')
    ar=pr+sr; all_groups=sorted(pgs+sgs,key=lambda x:x['score'],reverse=True)

    # PUNTO 8 — Amplitud de mercado (informativa, sobre el universo amplio S&P500+Nasdaq100+SOX)
    print('\n▸ Calculando amplitud de mercado...')
    breadth = calc_market_breadth(cs, bs)
    if breadth.get('pct_sobre_mm50') is not None:
        print(f'  {breadth["pct_sobre_mm50"]}% sobre MM50 | {breadth["pct_sobre_mm200"]}% sobre MM200 | '
              f'Avance/Descenso: {breadth["avance"]}/{breadth["descenso"]} | '
              f'Nuevos max/min 52s: {breadth["nuevos_max_52s"]}/{breadth["nuevos_min_52s"]} | '
              f'MM200 SPY: {breadth["pendiente_mm200_spy"]}')
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

    print('\n▸ Generando analisis Claude...')
    macro = macro if 'macro' in dir() else {}
    data_tmp={'timestamp':ts,'mode':'S&P500 + Watchlist','groups':all_groups,'values':sorted(ar,key=lambda x:x['score'] or 0,reverse=True),'spy_healthy':spy_ok,'macro':macro,'fundamentales':fundamentales,'breadth':breadth}
    analisis='Analisis no disponible. Ejecuta Colab para generar el analisis con Claude.'
    if ANTHROPIC_KEY:
        # Test de conexion antes de los reintentos
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
    history=update_history(all_groups, all_values=sorted(ar,key=lambda x:x['score'] or 0,reverse=True))

    print('\n▸ Actualizando historico de setups...')
    values_sorted = sorted(ar, key=lambda x: x['score'] or 0, reverse=True)
    setups_history, evaluaciones = update_setups_history(values_sorted, all_groups)
    upload_to_github('setups_history.json', json.dumps(clean_nan(setups_history), ensure_ascii=False))
    if evaluaciones:
        n_ok = sum(1 for e in evaluaciones if e['resultado'] == 'target')
        n_stop = sum(1 for e in evaluaciones if e['resultado'] == 'stop')
        n_ab = sum(1 for e in evaluaciones if e['resultado'] == 'abierto')
        print(f'  Evaluaciones: {len(evaluaciones)} | Target: {n_ok} | Stop: {n_stop} | Abierto: {n_ab}')
    print('\n▸ Generando alertas Telegram...')
    generate_alerts(data_tmp,history,spy_ok,bench_series=bs)
    data={'timestamp':ts,'mode':'S&P500 + Watchlist','groups':all_groups,'values':sorted(ar,key=lambda x:x['score'] or 0,reverse=True),'analisis':analisis,'spy_healthy':spy_ok,'evaluaciones':evaluaciones,'spy_ret_1w':spy_ret_1w,'spy_ret_1m':spy_ret_1m,'macro':macro,'fundamentales':fundamentales if "fundamentales" in vars() else {}}
    print('\n▸ Subiendo a GitHub...')
    upload_to_github('data.json',json.dumps(clean_nan(data),ensure_ascii=False))
    upload_to_github('history.json',json.dumps(clean_nan(history),ensure_ascii=False))
    print('\n═══════════════════════════════════════')
    print('  TOP 5 TEMAS')
    print('═══════════════════════════════════════')
    for i,g in enumerate(all_groups[:5]): print(f'  #{i+1}  {g["score"]:5.1f}  {g["group"]}')
    bos=[r for r in ar if r['breakout']]
    print(f'\n  Rupturas: {len(bos)} | SPY saludable: {spy_ok}')
    print(f'\n  Web: https://Carolo-III.github.io/scanner-temas')
    print('═══════════════════════════════════════\n')

if __name__ == '__main__':
    main()
