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
GITHUB_TOKEN     = os.environ.get('GITHUB_TOKEN_SCANNER', '')
ANTHROPIC_KEY    = os.environ.get('ANTHROPIC_KEY', '')
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '5100549189')

PERSONAL_WATCHLIST = {
    'Semiconductores': ['ARM','AMD','MU','MTSI','POET','SMCI'],
    'Infraestructura AI': ['ANET','VRT','APLD','CORZ','IREN','CIFR','CRWV'],
    'Espacio y Defensa': ['RKLB','LUNR','ASTS','KTOS','BWXT'],
    'Cuantica': ['IONQ'],
    'Robotica AI': ['BBAI','TSLA'],
    'Crypto Fintech': ['RDDT'],
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
        try:
            raw = yf.download(lote, period=period, interval='1d',
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                if len(lotes) > 1: print('vacio')
                continue
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
        except Exception as e:
            if len(lotes) > 1: print(f'error: {e}')
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

def calc_entry_range(ps, bi, mah):
    try:
        price=ps.get('price')
        if not price: return {}
        low5=ps.get('low5',price*0.95); high52=ps.get('high52',price)
        ma50=mah.get('ma50_val'); ma200=mah.get('ma200_val'); bl=bi.get('breakout_level')
        if bi.get('breakout') and bl and price<=bl*1.15:
            elo=round(price*0.995,2); ehi=round(price*1.020,2)
            stop=round(max(low5,price*0.930),2)
            if stop>=elo: stop=round(price*0.930,2)
            target=round(price+(price-bl if price>bl else price*0.10),2)
            rr=round((target-ehi)/(ehi-stop),1) if (ehi-stop)>0 else None
            return {'tipo':'Ruptura activa','entry_lo':elo,'entry_hi':ehi,'stop':stop,'target':target,'rr':rr}
        if bi.get('breakout') and bl and price>bl*1.15:
            return {'tipo':'Extendido — esperar pullback','entry_lo':None,'entry_hi':None,'stop':None,'target':None,'rr':None,
                    'nota':f'Precio {round((price/bl-1)*100,1)}% sobre ruptura. Esperar MM50=${round(ma50,2) if ma50 else "?"}'}
        if ma50 and abs(price/ma50-1)<0.04:
            elo=round(ma50*0.990,2); ehi=round(ma50*1.010,2); stop=round(ma50*0.950,2); target=round(high52,2)
            rr=round((target-ehi)/(ehi-stop),1) if (ehi-stop)>0 else None
            return {'tipo':'Pullback MM50','entry_lo':elo,'entry_hi':ehi,'stop':stop,'target':target,'rr':rr}
        if ma200 and abs(price/ma200-1)<0.04:
            elo=round(ma200*0.990,2); ehi=round(ma200*1.010,2); stop=round(ma200*0.940,2); target=round(ma200*1.20,2)
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
                if float(p.dropna().iloc[-1]) * float(v.dropna().iloc[-20:].mean() if len(v.dropna())>=20 else 0) < 2_000_000: continue
            except: pass
            h=high_df[tk] if high_df is not None and tk in high_df.columns else p
            l=low_df[tk] if low_df is not None and tk in low_df.columns else p
            r4=rs_score(p,bench,20); r13=rs_score(p,bench,65)
            vz=volume_zscore(v) if not v.empty else None
            bi=detect_breakout(p,v) if not v.empty else {'breakout':False,'days_ago':None,'breakout_level':None}
            mah=ma_health(p); ps=price_stats(p,h,v); er=calc_entry_range(ps,bi,mah)
            sc=composite_score(r4,r13,vz,bi['breakout'],mah,spy_healthy)
            res.append({'ticker':tk,'group':gn,'rs_4w':r4,'rs_13w':r13,'vol_z':vz,
                'breakout':bi['breakout'],'days_ago':bi['days_ago'],'breakout_level':bi.get('breakout_level'),
                'ma20':mah.get('ma20',False),'ma50':mah.get('ma50',False),'ma200':mah.get('ma200',False),
                'pct_ma50':mah.get('pct_ma50'),'pct_ma200':mah.get('pct_ma200'),
                'ma50_val':mah.get('ma50_val'),'ma200_val':mah.get('ma200_val'),
                'price':ps.get('price'),'high52':ps.get('high52'),'low52':ps.get('low52'),
                'pct_from_high':ps.get('pct_from_high'),'ret_1w':ps.get('ret_1w'),
                'ret_1m':ps.get('ret_1m'),'ret_3m':ps.get('ret_3m'),'entry_range':er,'score':sc})
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
    client=ant.Anthropic(api_key=anthropic_key)
    groups=data['groups']; values=data['values']; ts=data['timestamp']; spy_ok=data.get('spy_healthy',True)
    strong=[g for g in groups if g['score']>=70]; emerging=[g for g in groups if 50<=g['score']<70]; weak=[g for g in groups if g['score']<30]
    summary=f'DATOS — {ts}\nSPY sobre MM200: {"SI" if spy_ok else "NO"}\n\nFUERTES:\n'
    for g in strong[:5]:
        leaders=', '.join(m['ticker']+'('+str(m['score'])+')' for m in g['top3'])
        summary+=f'- {g["group"]}: {g["score"]} | RS:{g["rs_mean"]}% | {leaders}\n'
    summary+='\nEMERGENTES:\n'
    for g in emerging[:5]: summary+=f'- {g["group"]}: {g["score"]} | RS:{g["rs_mean"]}%\n'
    valid=[v for v in values if v.get('entry_range',{}).get('entry_lo') and (v.get('entry_range',{}).get('rr') or 0)>=2.0]
    summary+='\nSETUPS VALIDOS (usa EXACTAMENTE estos niveles):\n'
    for v in valid[:8]:
        er=v['entry_range']
        summary+=f'- {v["ticker"]} ({v["group"]}): ${v.get("price","?")} | {er["tipo"]} | Entrada:${er["entry_lo"]}-${er["entry_hi"]} | Stop:${er["stop"]} | Obj:${er["target"]}'
        if er.get('rr'): summary+=f' | R/B:1:{er["rr"]}'
        summary+='\n'
    extendidos=[v for v in values if v.get('entry_range',{}).get('tipo','').startswith('Extendido')]
    if extendidos:
        summary+='\nEXTENDIDOS:\n'
        for v in extendidos[:5]: summary+=f'- {v["ticker"]}: ${v.get("price","?")} | {v["entry_range"].get("nota","Extendido")}\n'
    summary+=f'\nFuertes:{len(strong)} Emergentes:{len(emerging)} Sin momentum:{len(weak)} Setups:{len(valid)}\n'
    aviso='ATENCION: SPY bajo MM200. Mercado bajista. ' if not spy_ok else ''
    prompt=('Analista tecnico momentum. Informe ejecutivo espanol directo. '+aviso+
            'Estructura: 1.MERCADO 2.TEMAS PRIORITARIOS 3.ENTRADAS(niveles exactos) 4.EXTENDIDOS 5.CONCLUSION. Max 400 palabras. '
            'Aviso final obligatorio: "Este analisis no constituye asesoramiento financiero."\n\n'+summary)
    msg=client.messages.create(model='claude-sonnet-4-5',max_tokens=1000,messages=[{'role':'user','content':prompt}])
    return msg.content[0].text

def get_github_file(filename):
    url=f'https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{filename}'
    r=requests.get(url,headers={'Authorization':f'token {GITHUB_TOKEN}'})
    if r.status_code==200:
        return json.loads(base64.b64decode(r.json()['content']).decode('utf-8')),r.json()['sha']
    return None,None

def upload_to_github(filename, content):
    url=f'https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{filename}'
    headers={'Authorization':f'token {GITHUB_TOKEN}','Content-Type':'application/json'}
    r=requests.get(url,headers=headers); sha=r.json().get('sha') if r.status_code==200 else None
    content_b64=base64.b64encode(content.encode('utf-8') if isinstance(content,str) else content).decode('utf-8')
    payload={'message':f'Actualizar {filename} - {datetime.now(madrid).strftime("%d/%m/%Y %H:%M")}','content':content_b64}
    if sha: payload['sha']=sha
    r=requests.put(url,headers=headers,json=payload)
    print(f'  {"OK" if r.status_code in [200,201] else "Error"} {filename}')

def update_history(all_groups):
    today=datetime.now(madrid).strftime('%Y-%m-%d')
    history,_=get_github_file('history.json')
    if history is None: history=[]
    history=[h for h in history if h['date']!=today]
    history.append({'date':today,'scores':{g['group']:g['score'] for g in all_groups},'ranks':{g['group']:i+1 for i,g in enumerate(all_groups)}})
    return sorted(history,key=lambda x:x['date'])[-7:]

def send_telegram(message):
    if not TELEGRAM_TOKEN: return
    try:
        r=requests.post(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',json={'chat_id':TELEGRAM_CHAT_ID,'text':message},timeout=10)
        print(f'  {"OK" if r.status_code==200 else "Error"} Telegram')
    except Exception as e: print(f'  Error Telegram: {e}')

def generate_alerts(data, history, spy_healthy=True):
    values=data['values']; groups=data['groups']; ts=data['timestamp']
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
    if msgs:
        for m in msgs: send_telegram(m)
    else:
        print('  Sin alertas nuevas hoy')

def main():
    print('\n═══════════════════════════════════════')
    print('  SCANNER DE TEMAS EMERGENTES')
    ts=datetime.now(madrid).strftime('%d/%m/%Y %H:%M')
    print(f'  {ts}\n═══════════════════════════════════════\n')
    print('▸ SPY...')
    bc,bv,bh,bl=download_prices(['SPY'],period='1y'); bs=bc['SPY']; spy_ok=spy_health(bs)
    pt=list(set(t for grp in PERSONAL_WATCHLIST.values() for t in grp))
    print(f'\n▸ Watchlist ({len(pt)} valores)...')
    cp,vp,hp,lp=download_prices(pt+['SPY'],period='1y')
    pr=analyze_universe(PERSONAL_WATCHLIST,bs,cp,vp,hp,lp,spy_healthy=spy_ok)
    pgs=calc_groups(pr,is_sp=False); print(f'  OK {len(pr)} valores')
    print('\n▸ Lista S&P 500...')
    df_sp=pd.read_csv('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv')
    sbs={}
    for _,row in df_sp.iterrows():
        tk=row['Symbol'].replace('.','-')
        if tk not in pt: sbs.setdefault(SECTOR_LABELS.get(row['GICS Sector'],row['GICS Sector']),[]).append(tk)
    sa=list(set(t for grp in sbs.values() for t in grp))
    print(f'▸ Descargando {len(sa)} valores S&P 500...')
    cs,vs,hs,ls=download_prices(sa,period='1y')
    sr=analyze_universe(sbs,bs,cs,vs,hs,ls,spy_healthy=spy_ok)
    sgs=calc_groups(sr,is_sp=True); print(f'  OK {len(sr)} valores')
    ar=pr+sr; all_groups=sorted(pgs+sgs,key=lambda x:x['score'],reverse=True)
    print('\n▸ Generando analisis Claude...')
    data_tmp={'timestamp':ts,'mode':'S&P500 + Watchlist','groups':all_groups,'values':sorted(ar,key=lambda x:x['score'] or 0,reverse=True),'spy_healthy':spy_ok}
    try:
        analisis=generate_analysis(data_tmp,ANTHROPIC_KEY); print('  OK')
    except Exception as e:
        print(f'  Aviso: {e}'); analisis='Analisis no disponible.'
    history=update_history(all_groups)
    print('\n▸ Generando alertas Telegram...')
    generate_alerts(data_tmp,history,spy_ok)
    data={'timestamp':ts,'mode':'S&P500 + Watchlist','groups':all_groups,'values':sorted(ar,key=lambda x:x['score'] or 0,reverse=True),'analisis':analisis,'spy_healthy':spy_ok}
    print('\n▸ Subiendo a GitHub...')
    upload_to_github('data.json',json.dumps(data,ensure_ascii=False))
    upload_to_github('history.json',json.dumps(history,ensure_ascii=False))
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
