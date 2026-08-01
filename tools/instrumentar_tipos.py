#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/instrumentar_tipos.py

Instrumentacion del tramo de tipos (punto 7). SOLO REGISTRA.

Construye y actualiza rates_history.json con nivel, variacion 20 sesiones
y pendientes de la curva. NO acopla nada al scoring ni al vector de regimen:
el objetivo es acumular serie propia antes de que ninguna decision dependa
de ella.

Diseño deliberado (mismo criterio que breadth_history.json / P29):
  - la fecha de cada entrada es la SESION del dato, no el reloj
  - dedupe por fecha; una entrada de cierre no es pisable por una provisional
  - reejecuciones del mismo dia son idempotentes

Tramos (yfinance):
  ^IRX  13 semanas    ^FVX  5 años
  ^TNX  10 años       ^TYX  30 años

Nota sobre el 2Y: Yahoo no expone un ticker fiable del 2 años. La pendiente
"corta" se aproxima con 5Y. Si se quiere el 2Y real hay que ir a FRED
(serie DGS2), lo que añade una dependencia externa. Decision pendiente.

Uso:
    python tools/instrumentar_tipos.py --solo-ultima    # uso diario
    python tools/instrumentar_tipos.py --sembrar 120   # siembra 120 sesiones
    python tools/instrumentar_tipos.py --resumen       # lee y resume, sin escribir
    python tools/instrumentar_tipos.py --simular fichero.json   # sin red
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

FICHERO = "rates_history.json"

TRAMOS = {
    "us3m": "^IRX",
    "us5y": "^FVX",
    "us10y": "^TNX",
    "us30y": "^TYX",
}

VENTANA_VARIACION = 20  # sesiones, mismo criterio que las alertas de regimen


# --------------------------------------------------------------------------
# Nucleo puro: sin red, testeable de forma aislada.
# --------------------------------------------------------------------------

def calcular_pendientes(niveles):
    """Pendientes de curva en puntos basicos.

    niveles: dict {clave_tramo: rendimiento en %}. Devuelve dict de
    pendientes; omite las que no se puedan calcular por falta de dato.
    """
    pendientes = {}
    pares = (
        ("p10y_5y", "us10y", "us5y"),
        ("p30y_10y", "us30y", "us10y"),
        ("p10y_3m", "us10y", "us3m"),
    )
    for nombre, largo, corto in pares:
        a, b = niveles.get(largo), niveles.get(corto)
        if a is None or b is None:
            continue
        pendientes[nombre] = round((a - b) * 100.0, 1)
    return pendientes


def variacion_pb(serie, ventana=VENTANA_VARIACION):
    """Variacion en puntos basicos entre el ultimo valor y el de hace
    `ventana` sesiones. serie: lista de floats en orden cronologico.
    Devuelve None si no hay historico suficiente.
    """
    if not serie or len(serie) <= ventana:
        return None
    return round((serie[-1] - serie[-1 - ventana]) * 100.0, 1)


def construir_serie_historica(series_por_tramo, es_cierre=False):
    """Construye una entrada por cada sesion disponible, no solo la ultima.

    series_por_tramo: dict {clave_tramo: [(fecha, valor), ...]} en orden
    cronologico. Los tramos pueden tener calendarios distintos; se toma la
    union de fechas y cada tramo aporta lo que tenga.

    La variacion a 20 sesiones de cada entrada se calcula con las sesiones
    de ESE tramo anteriores a la fecha, no con el calendario comun. Las
    primeras 20 entradas saldran sin variacion, que es lo correcto.
    """
    por_tramo = {}
    fechas = set()
    for clave, pares in series_por_tramo.items():
        ordenados = sorted(pares, key=lambda t: t[0])
        por_tramo[clave] = ordenados
        fechas.update(f for f, _ in ordenados)

    entradas = []
    for fecha in sorted(fechas):
        niveles, series_hasta = {}, {}
        for clave in sorted(por_tramo):
            hasta = [v for f, v in por_tramo[clave] if f <= fecha]
            if not hasta:
                continue
            # Solo aporta nivel si tiene dato en ESA sesion exacta.
            exacto = [v for f, v in por_tramo[clave] if f == fecha]
            if not exacto:
                continue
            niveles[clave] = exacto[-1]
            series_hasta[clave] = hasta
        if not niveles:
            continue
        entradas.append(
            construir_entrada(fecha, niveles, series_hasta, es_cierre=es_cierre))
    return entradas


def construir_entrada(fecha, niveles, series=None, es_cierre=False):
    """Ensambla una entrada de rates_history a partir de niveles y series."""
    series = series or {}
    entrada = {
        "fecha": fecha,
        "es_cierre": bool(es_cierre),
        "niveles": {k: round(v, 3) for k, v in sorted(niveles.items())
                    if v is not None},
        "variacion_20s_pb": {},
        "pendientes_pb": calcular_pendientes(niveles),
    }
    for clave, serie in sorted(series.items()):
        var = variacion_pb(serie)
        if var is not None:
            entrada["variacion_20s_pb"][clave] = var
    return entrada


def fusionar(historico, entrada):
    """Inserta o sustituye por fecha, respetando la precedencia de cierre.

    Devuelve (historico_ordenado, accion) donde accion es una de
    "insertada", "sustituida", "ignorada_por_cierre".
    """
    salida = []
    accion = "insertada"
    encontrada = False

    for previa in historico:
        if previa.get("fecha") != entrada["fecha"]:
            salida.append(previa)
            continue
        encontrada = True
        # Un cierre ya registrado no lo pisa una entrada provisional.
        if previa.get("es_cierre") and not entrada.get("es_cierre"):
            salida.append(previa)
            accion = "ignorada_por_cierre"
        else:
            salida.append(entrada)
            accion = "sustituida"

    if not encontrada:
        salida.append(entrada)

    salida.sort(key=lambda x: x.get("fecha", ""))
    return salida, accion


def cargar_historico(ruta):
    if not os.path.exists(ruta):
        return []
    with open(ruta, "r", encoding="utf-8") as fh:
        datos = json.load(fh)
    if not isinstance(datos, list):
        raise ValueError("%s no contiene una lista de entradas." % ruta)
    return datos


def guardar_historico(ruta, historico):
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(historico, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def resumir(historico):
    """Resumen legible de la serie acumulada."""
    if not historico:
        return "rates_history.json vacio o inexistente."

    lineas = []
    lineas.append("Entradas: %d  (%s a %s)" % (
        len(historico),
        historico[0].get("fecha", "?"),
        historico[-1].get("fecha", "?"),
    ))
    cierres = sum(1 for e in historico if e.get("es_cierre"))
    lineas.append("De ellas marcadas como cierre: %d" % cierres)

    ultima = historico[-1]
    lineas.append("\nUltima entrada (%s):" % ultima.get("fecha", "?"))
    for clave in sorted(ultima.get("niveles", {})):
        nivel = ultima["niveles"][clave]
        var = ultima.get("variacion_20s_pb", {}).get(clave)
        texto_var = "n/d" if var is None else ("%+.1f pb/20s" % var)
        lineas.append("  %-6s %6.3f%%   %s" % (clave, nivel, texto_var))

    pend = ultima.get("pendientes_pb", {})
    if pend:
        lineas.append("\n  Pendientes (pb):")
        for clave in sorted(pend):
            lineas.append("    %-10s %+7.1f" % (clave, pend[clave]))

    # Evolucion de la pendiente principal, si hay historico.
    serie_p = [(e.get("fecha"), e.get("pendientes_pb", {}).get("p30y_10y"))
               for e in historico]
    serie_p = [(f, v) for f, v in serie_p if v is not None]
    if len(serie_p) >= 2:
        lineas.append("\n  p30y_10y: %+.1f (%s) -> %+.1f (%s)  [delta %+.1f pb]"
                      % (serie_p[0][1], serie_p[0][0],
                         serie_p[-1][1], serie_p[-1][0],
                         serie_p[-1][1] - serie_p[0][1]))
    return "\n".join(lineas)


# --------------------------------------------------------------------------
# Capa de red: aislada, para poder probar el nucleo sin conexion.
# --------------------------------------------------------------------------

def descargar(dias):
    """Descarga los tramos via yfinance.

    Devuelve dict {clave_tramo: [(fecha_iso, valor), ...]} cronologico.
    Si yfinance no esta disponible o un tramo falla, se omite ese tramo y
    se sigue con los demas.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError(
            "yfinance no disponible. Instala con: pip install yfinance")

    series = {}

    for clave, ticker in sorted(TRAMOS.items()):
        try:
            hist = yf.Ticker(ticker).history(period="%dd" % max(dias, 40))
            if hist is None or hist.empty:
                print("  AVISO: %s (%s) sin datos, se omite" % (clave, ticker))
                continue
            pares = []
            for marca, valor in zip(hist.index, hist["Close"].tolist()):
                try:
                    valor = float(valor)
                except (TypeError, ValueError):
                    continue
                if valor != valor:  # NaN
                    continue
                pares.append((marca.strftime("%Y-%m-%d"), valor))
            if not pares:
                print("  AVISO: %s sin cierres validos, se omite" % clave)
                continue
            series[clave] = pares
            print("  OK %-6s %s  %.3f%%  (%d sesiones, %s a %s)"
                  % (clave, ticker, pares[-1][1], len(pares),
                     pares[0][0], pares[-1][0]))
        except Exception as exc:  # noqa: BLE001 - degradar, no abortar
            print("  AVISO: fallo en %s (%s): %s" % (clave, ticker, exc))

    if not series:
        raise RuntimeError("Ningun tramo devolvio datos.")
    return series


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Instrumentacion de tipos: actualiza rates_history.json")
    parser.add_argument("--fichero", default=FICHERO)
    parser.add_argument("--sembrar", type=int, default=60,
                        help="Dias de historico a pedir (min 40 por la ventana)")
    parser.add_argument("--cierre", action="store_true",
                        help="Marcar la entrada como cierre definitivo")
    parser.add_argument("--solo-ultima", action="store_true",
                        dest="solo_ultima",
                        help="Registrar solo la ultima sesion (uso diario)")
    parser.add_argument("--resumen", action="store_true",
                        help="Solo resumir la serie existente, sin escribir")
    parser.add_argument("--simular", default=None,
                        help="JSON {fecha, niveles, series} para probar sin red")
    args = parser.parse_args(argv)

    try:
        historico = cargar_historico(args.fichero)
    except (ValueError, json.JSONDecodeError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1

    if args.resumen:
        print(resumir(historico))
        return 0

    if args.simular:
        with open(args.simular, "r", encoding="utf-8") as fh:
            sim = json.load(fh)
        series = sim["series"]
    else:
        print("Descargando tramos de tipos...")
        try:
            series = descargar(args.sembrar)
        except RuntimeError as exc:
            print("ERROR: %s" % exc, file=sys.stderr)
            return 1

    entradas = construir_serie_historica(series, es_cierre=args.cierre)
    if not entradas:
        print("ERROR: no se pudo construir ninguna entrada.", file=sys.stderr)
        return 1

    if args.solo_ultima:
        entradas = entradas[-1:]

    resumen_acciones = {"insertada": 0, "sustituida": 0,
                        "ignorada_por_cierre": 0}
    for entrada in entradas:
        historico, accion = fusionar(historico, entrada)
        resumen_acciones[accion] += 1

    accion = "insertada"
    guardar_historico(args.fichero, historico)
    print("Procesadas %d sesiones: %d nuevas, %d actualizadas, %d ignoradas "
          "por precedencia de cierre. Total en fichero: %d entradas."
          % (len(entradas),
             resumen_acciones["insertada"],
             resumen_acciones["sustituida"],
             resumen_acciones["ignorada_por_cierre"],
             len(historico)))
    print()
    print(resumir(historico))
    return 0


if __name__ == "__main__":
    sys.exit(main())
