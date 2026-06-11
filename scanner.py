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
                'margen_neto':   round((info.get('profitMargins', 0) or 0) * 100, 1),
                'roe':           round((info.get('returnOnEquity', 0) or 0) * 100, 1),
                'deuda_equity':  round(info.get('debtToEquity', 0) or 0, 2),
                'rev_growth':    round((info.get('revenueGrowth', 0) or 0) * 100, 1),
                'eps_fwd':       round(info.get('forwardEps', 0) or 0, 2),
                'sector':        info.get('sector', ''),
                'mkt_cap_b':     round((info.get('marketCap', 0) or 0) / 1e9, 1),
            }
        except Exception as e:
            result[tk] = {}
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
    # CMF (15%) — flujo positivo
    if cmf is not None:
        if cmf >= 0.2:    score += 15
        elif cmf >= 0.05: score += 10
        elif cmf >= 0:    score += 5
    # OBV (10%) — en maximos junto con precio
    if obv_trend is not None:
        if obv_trend >= 0.98:  score += 10
        elif obv_trend >= 0.90: score += 6
        elif obv_trend >= 0.80: score += 3
    return round(score, 1)

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
            target=round(price+(price-bl if price>bl else price*0.10),2)
            rr=round((target-ehi)/(ehi-stop),1) if (ehi-stop)>0 else None
            return {'tipo':'Ruptura activa','entry_lo':elo,'entry_hi':ehi,'stop':stop,'target':target,'rr':rr}
        if bi.get('breakout') and bl and price>bl*1.15:
            return {'tipo':'Extendido — esperar pullback','entry_lo':None,'entry_hi':None,'stop':None,'target':None,'rr':None,
                    'nota':f'Precio {round((price/bl-1)*100,1)}% sobre ruptura. Esperar MM50=${round(ma50,2) if ma50 else "?"}'}
        if ma50 and abs(price/ma50-1)<0.04:
            elo=round(ma50*0.990,2); ehi=round(ma50*1.010,2); stop=round(ma50*0.950,2); target=round(high52,2)
            # Distancia minima stop: 1 ATR o 2%
            min_dist = max(atr_val if atr_val else 0, elo*0.02)
            if (elo - stop) < min_dist: stop=round(elo - min_dist, 2)
            rr=round((target-ehi)/(ehi-stop),1) if (ehi-stop)>0 else None
            return {'tipo':'Pullback MM50','entry_lo':elo,'entry_hi':ehi,'stop':stop,'target':target,'rr':rr}
        if ma200 and abs(price/ma200-1)<0.04:
            elo=round(ma200*0.990,2); ehi=round(ma200*1.010,2); stop=round(ma200*0.940,2); target=round(ma200*1.20,2)
            # Distancia minima stop: 1 ATR o 2%
            min_dist = max(atr_val if atr_val else 0, elo*0.02)
            if (elo - stop) < min_dist: stop=round(elo - min_dist, 2)
            rr=round((target-ehi)/(ehi-stop),1) if (ehi-stop)>0 else None
            return {'tipo':'Pullback MM200','entry_lo':elo,'entry_hi':ehi,'stop':stop,'target':target,'rr':rr}
        if bl and price<=bl*1.03:
            return {'tipo':'En vigilancia','entry_lo':None,'entry_hi':None,'stop':None,'target':None,'rr':None,'nivel_ruptura':round(bl,2)}
        return {'tipo':'Sin setup','entry_lo':None,'entry_hi':None,'stop':None,'target':None,'rr':None}
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
            res.append({'ticker':tk,'group':gn,'rs_4w':r4,'rs_13w':r13,'vol_z':vz,
                'breakout':bi['breakout'],'days_ago':bi['days_ago'],'breakout_level':bi.get('breakout_level'),
                'ma20':mah.get('ma20',False),'ma50':mah.get('ma50',False),'ma200':mah.get('ma200',False),
                'pct_ma50':mah.get('pct_ma50'),'pct_ma200':mah.get('pct_ma200'),
                'ma50_val':mah.get('ma50_val'),'ma200_val':mah.get('ma200_val'),
                'price':ps.get('price'),'high52':ps.get('high52'),'low52':ps.get('low52'),
                'pct_from_high':ps.get('pct_from_high'),'ret_1w':ps.get('ret_1w'),
                'ret_1m':ps.get('ret_1m'),'ret_3m':ps.get('ret_3m'),'entry_range':er,'score':sc,
                'sct':sct_val,'rsi':rsi_val,'adx':adx_val,'macd_hist':macd_hist,
                'cmf':cmf_val,'atr':atr_val,'squeeze':squeeze,'pct_b':pct_b})
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

    summary=f'DATOS — {ts}\nSPY sobre MM200: {"SI" if spy_ok else "NO"}\n\n' + macro_txt + 'FUERTES:\n'
    for g in strong[:5]:
        leaders=', '.join(m['ticker']+'('+str(m['score'])+')' for m in g['top3'])
        summary+=f'- {g["group"]}: {g["score"]} | RS:{g["rs_mean"]}% | {leaders}\n'
    summary+='\nEMERGENTES:\n'
    for g in emerging[:5]: summary+=f'- {g["group"]}: {g["score"]} | RS:{g["rs_mean"]}%\n'
    fundamentales = data.get('fundamentales', {})
    valid=[v for v in values if v.get('entry_range',{}).get('entry_lo') and (v.get('entry_range',{}).get('rr') or 0)>=2.0]
    summary+='\nSETUPS VALIDOS CON ANALISIS FUNDAMENTAL (usa EXACTAMENTE estos niveles):\n'
    for v in valid[:8]:
        er=v['entry_range']
        rsi_s = v.get("rsi"); adx_s = v.get("adx"); cmf_s = v.get("cmf"); sq_s = v.get("squeeze"); sct_s = v.get("sct")
        indicadores = []
        if rsi_s: indicadores.append(f"RSI:{rsi_s}")
        if adx_s: indicadores.append(f"ADX:{adx_s}")
        if cmf_s: indicadores.append(f"CMF:{cmf_s}")
        if sct_s: indicadores.append(f"SCT:{sct_s}")
        if sq_s: indicadores.append("SQUEEZE")
        ind_str = " | ".join(indicadores) if indicadores else ""
        summary+=f'- {v["ticker"]} ({v["group"]}): ${v.get("price","?")} | {er["tipo"]} | Entrada:${er["entry_lo"]}-${er["entry_hi"]} | Stop:${er["stop"]} | Obj:${er["target"]}'
        if ind_str: summary+=f' | {ind_str}'
        if er.get('rr'): summary+=f' | R/B:1:{er["rr"]}'
        # Datos fundamentales si disponibles
        fund = fundamentales.get(v['ticker'], {})
        if fund:
            f_parts = []
            if fund.get('per_forward'): f_parts.append(f'PERfwd:{fund["per_forward"]}x')
            if fund.get('peg'):         f_parts.append(f'PEG:{fund["peg"]}')
            if fund.get('margen_neto'): f_parts.append(f'Margen:{fund["margen_neto"]}%')
            if fund.get('roe'):         f_parts.append(f'ROE:{fund["roe"]}%')
            if fund.get('deuda_equity'):f_parts.append(f'D/E:{fund["deuda_equity"]}')
            if fund.get('rev_growth'):  f_parts.append(f'RevGrowth:{fund["rev_growth"]}%')
            if f_parts: summary+=f'        FUND: {" | ".join(f_parts)}\n'
        summary+='\n'
    extendidos=[v for v in values if v.get('entry_range',{}).get('tipo','').startswith('Extendido')]
    if extendidos:
        summary+='\nEXTENDIDOS:\n'
        for v in extendidos[:5]: summary+=f'- {v["ticker"]}: ${v.get("price","?")} | {v["entry_range"].get("nota","Extendido")}\n'
    summary+=f'\nFuertes:{len(strong)} Emergentes:{len(emerging)} Sin momentum:{len(weak)} Setups:{len(valid)}\n'
    aviso='ATENCION: SPY bajo MM200. Mercado bajista. ' if not spy_ok else ''
    prompt = (
        'Analista tecnico momentum. Informe ejecutivo espanol directo. ' + aviso +
        'Estructura: 1.MERCADO 2.TEMAS PRIORITARIOS 3.ENTRADAS(niveles exactos) 4.EXTENDIDOS 5.CONCLUSION. '
        'Escribe cada seccion de forma que sea comprensible tanto para un analista tecnico como para un inversor adulto con conocimientos generales de bolsa pero sin experiencia en analisis tecnico. '
        'No uses parrafos separados ni marcadores especiales para las explicaciones: integra el contexto y el significado directamente en el texto de cada seccion. '
        'Cuando menciones un indicador tecnico (RSI, ADX, SCT, CMF, MM50, etc.) explica brevemente en la misma frase que implica ese valor concreto para la decision. '
        'Para cada setup de entrada, integra el contexto fundamental (PER, margen, ROE, crecimiento) con el tecnico: explica si la valoracion apoya o cuestiona la entrada tecnica. '
        'Si el PER forward es elevado (>40x) pero el crecimiento de ingresos es alto (>30%), mencionalo como valoracion de crecimiento que requiere gestion de riesgo estricta. '
        'Si el margen neto es negativo, mencionalo brevemente pero sin descartar el setup si el momentum es solido. '
        'Ejemplos de tono correcto: '
        '"IONQ cotiza con RSI en 65, zona de momentum optimo sin sobrecompra, y ADX en 42, lo que confirma que la tendencia alcista tiene solidez suficiente para nuevas entradas." '
        '"El SCT de 78 sobre 100 indica que varios indicadores tecnicos apuntan en la misma direccion, lo que reduce el riesgo de una senal falsa." '
        'Para interpretar indicadores: RSI 55-70=momentum optimo, RSI>70=sobrecomprado con cautela; '
        'ADX>40=tendencia muy fuerte, ADX>25=tendencia confirmada, ADX<20=mercado lateral evitar; '
        'CMF>0.1=acumulacion institucional, CMF<0=distribucion; SCT>70=confirmacion tecnica alta; '
        'SQUEEZE=compresion de volatilidad previa a ruptura potente. '
        'FORMATO: en la seccion ENTRADAS, separa cada recomendacion con un salto de linea — una recomendacion por parrafo independiente. '
        'Aviso final obligatorio: "Este analisis no constituye asesoramiento financiero."\n\n' + summary
    )
    import httpx
    msg=client.messages.create(model='claude-fable-5',max_tokens=1500,messages=[{'role':'user','content':prompt}], timeout=httpx.Timeout(120.0, connect=30.0))
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
        if er.get('entry_lo') and (er.get('rr') or 0) >= 2.0:
            setups_hoy.append({
                'ticker':    v['ticker'],
                'group':     v['group'],
                'tipo':      er['tipo'],
                'price':     v.get('price'),
                'entry_lo':  er['entry_lo'],
                'entry_hi':  er['entry_hi'],
                'stop':      er['stop'],
                'target':    er['target'],
                'rr':        er.get('rr'),
                'score':     v.get('score'),
                'rs_4w':     v.get('rs_4w'),
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
        'ABNB','ADSK','ALGN','ANSS','ASML','BIDU','BIIB',
        'CDNS','CHTR','CPRT','CTSH','DDOG','DLTR','DOCU','DXCM',
        'EA','EBAY','FAST','FSLR','FTNT','GFS',
        'IDXX','ILMN','KDP','KHC','LULU','MAR','MCHP','MDLZ',
        'MNST','MRNA','MRVL','NTES','ODFL','OKTA','ON','ORLY',
        'PANW','PAYX','PCAR','PDD','PYPL','QCOM','REGN','ROST',
        'SNPS','SWKS','TEAM','TMUS','TSCO',
        'TTD','TTWO','TXN','VRSK','VRSN','VRTX','WBD','XEL','ZM','ZS'
    ]
    sox_extra = [
        'ACLS','ADI','AEHR','AMAT','AMKR','ASML','AZTA','COHU',
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
    # Obtener fundamentales de valores con setup valido
    print('\n▸ Obteniendo datos fundamentales...')
    valid_tickers = list(set(
        v['ticker'] for v in sorted(ar, key=lambda x: x['score'] or 0, reverse=True)
        if v.get('entry_range', {}).get('entry_lo') and (v.get('entry_range', {}).get('rr') or 0) >= 2.0
    ))[:20]
    fundamentales = get_fundamentals(valid_tickers) if valid_tickers else {}
    print(f'  OK {len(fundamentales)} valores con datos fundamentales')

    print('\n▸ Generando analisis Claude...')
    macro = macro if 'macro' in dir() else {}
    data_tmp={'timestamp':ts,'mode':'S&P500 + Watchlist','groups':all_groups,'values':sorted(ar,key=lambda x:x['score'] or 0,reverse=True),'spy_healthy':spy_ok,'macro':macro,'fundamentales':fundamentales}
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
