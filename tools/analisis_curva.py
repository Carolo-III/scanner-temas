#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/analisis_curva.py

Vista por EPISODIO de la serie de tipos (rates_history.json).

Responde la pregunta que quedo abierta el 01/08/2026: que hizo la curva
DURANTE el crash de momentum de junio-julio, no solo en los extremos de la
serie.

Criterio de diseño: los episodios se DERIVAN de la serie (extremos de cada
pendiente, tramos mensuales, mayores movimientos), no de fechas clavadas a
mano. Fijar "el maximo de junio" por memoria seria meter sesgo retrospectivo
en el analisis. Los hitos externos conocidos (FOMC, IPC) se pasan con
--hitos y se marcan aparte, sin mezclarse con lo derivado.

Uso:
    python tools/analisis_curva.py
    python tools/analisis_curva.py --hitos 2026-07-29:FOMC,2026-06-10:IPC
    python tools/analisis_curva.py --cruzar breadth_history.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PENDIENTES = ("p10y_3m", "p10y_5y", "p30y_10y")
TRAMOS = ("us3m", "us5y", "us10y", "us30y")


def cargar(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            "No encuentro %s. Ejecuta desde la raiz del repo." % ruta)
    with open(ruta, "r", encoding="utf-8") as fh:
        datos = json.load(fh)
    if not isinstance(datos, list):
        raise ValueError("%s no contiene una lista de entradas." % ruta)
    return sorted(datos, key=lambda e: e.get("fecha") or "")


def serie_de(historico, grupo, clave):
    """[(fecha, valor)] de una metrica, saltando entradas sin dato."""
    salida = []
    for e in historico:
        valor = (e.get(grupo) or {}).get(clave)
        if valor is not None:
            salida.append((e.get("fecha"), float(valor)))
    return salida


def extremos(serie):
    """(fecha_min, val_min, fecha_max, val_max) o None si la serie esta vacia."""
    if not serie:
        return None
    fmin, vmin = min(serie, key=lambda t: t[1])
    fmax, vmax = max(serie, key=lambda t: t[1])
    return fmin, vmin, fmax, vmax


def por_mes(serie):
    """Primer y ultimo valor de cada mes: {mes: (f_ini, v_ini, f_fin, v_fin)}."""
    meses = {}
    for fecha, valor in serie:
        mes = fecha[:7]
        if mes not in meses:
            meses[mes] = [fecha, valor, fecha, valor]
        else:
            meses[mes][2], meses[mes][3] = fecha, valor
    return {m: tuple(v) for m, v in sorted(meses.items())}


def mayores_movimientos(serie, ventana=10, top=3):
    """Los `top` tramos de `ventana` sesiones con mayor variacion absoluta.

    Devuelve [(f_ini, f_fin, delta)]. Los tramos se solapan; se filtran los
    que comparten mas de la mitad de la ventana con uno ya elegido, para no
    devolver tres veces el mismo episodio desplazado un dia.
    """
    if len(serie) <= ventana:
        return []
    candidatos = []
    for i in range(len(serie) - ventana):
        delta = serie[i + ventana][1] - serie[i][1]
        candidatos.append((i, serie[i][0], serie[i + ventana][0], delta))
    candidatos.sort(key=lambda t: abs(t[3]), reverse=True)

    elegidos = []
    for idx, f_ini, f_fin, delta in candidatos:
        if any(abs(idx - j) <= ventana // 2 for j, _, _, _ in elegidos):
            continue
        elegidos.append((idx, f_ini, f_fin, delta))
        if len(elegidos) >= top:
            break
    return [(f_ini, f_fin, delta) for _, f_ini, f_fin, delta in elegidos]


def valor_en(serie, fecha):
    """Valor en esa fecha exacta, o el ultimo anterior. None si no hay."""
    previo = None
    for f, v in serie:
        if f == fecha:
            return v, f
        if f < fecha:
            previo = (v, f)
        else:
            break
    return previo if previo else (None, None)


def parsear_hitos(texto):
    if not texto:
        return []
    hitos = []
    for parte in texto.split(","):
        parte = parte.strip()
        if not parte:
            continue
        if ":" in parte:
            fecha, etiqueta = parte.split(":", 1)
        else:
            fecha, etiqueta = parte, "hito"
        hitos.append((fecha.strip(), etiqueta.strip()))
    return sorted(hitos)


def informe(historico, hitos, cruce=None):
    print("=" * 70)
    print("CURVA DE TIPOS — VISTA POR EPISODIO")
    print("=" * 70)
    print("Entradas: %d  (%s a %s)" % (
        len(historico),
        historico[0].get("fecha", "?"),
        historico[-1].get("fecha", "?")))

    series_p = {c: serie_de(historico, "pendientes_pb", c) for c in PENDIENTES}
    series_n = {c: serie_de(historico, "niveles", c) for c in TRAMOS}

    # --- 1. Extremos derivados de la serie ---
    print("\n--- EXTREMOS DE CADA PENDIENTE (derivados, no fijados a mano) ---")
    for clave in PENDIENTES:
        ext = extremos(series_p[clave])
        if not ext:
            print("  %-10s sin datos" % clave)
            continue
        fmin, vmin, fmax, vmax = ext
        print("  %-10s min %+7.1f pb (%s)   max %+7.1f pb (%s)   recorrido %.1f pb"
              % (clave, vmin, fmin, vmax, fmax, vmax - vmin))

    print("\n--- EXTREMOS DE CADA TRAMO ---")
    for clave in TRAMOS:
        ext = extremos(series_n[clave])
        if not ext:
            print("  %-6s sin datos" % clave)
            continue
        fmin, vmin, fmax, vmax = ext
        print("  %-6s min %6.3f%% (%s)   max %6.3f%% (%s)   recorrido %.0f pb"
              % (clave, vmin, fmin, vmax, fmax, (vmax - vmin) * 100))

    # --- 2. Evolucion mensual ---
    print("\n--- EVOLUCION MENSUAL DE LAS PENDIENTES (pb) ---")
    meses_todos = sorted({f[:7] for f, _ in series_p.get("p10y_3m", [])}
                         | {f[:7] for f, _ in series_p.get("p30y_10y", [])})
    cabecera = "  %-9s" % "mes"
    for clave in PENDIENTES:
        cabecera += " %18s" % clave
    print(cabecera)
    for mes in meses_todos:
        linea = "  %-9s" % mes
        for clave in PENDIENTES:
            datos = por_mes(series_p[clave]).get(mes)
            if not datos:
                linea += " %18s" % "n/d"
                continue
            _, v_ini, _, v_fin = datos
            linea += " %8.1f->%6.1f" % (v_ini, v_fin)
        print(linea)

    # --- 3. Mayores movimientos: los episodios propiamente dichos ---
    print("\n--- MAYORES MOVIMIENTOS EN VENTANAS DE 10 SESIONES ---")
    for clave in PENDIENTES:
        movs = mayores_movimientos(series_p[clave])
        if not movs:
            print("  %-10s serie insuficiente" % clave)
            continue
        print("  %s:" % clave)
        for f_ini, f_fin, delta in movs:
            print("    %s -> %s   %+7.1f pb" % (f_ini, f_fin, delta))

    # --- 4. Hitos externos, si se han pasado ---
    if hitos:
        print("\n--- HITOS EXTERNOS (aportados por el usuario) ---")
        for fecha, etiqueta in hitos:
            print("  %s  %s" % (fecha, etiqueta))
            for clave in PENDIENTES:
                valor, f_real = valor_en(series_p[clave], fecha)
                if valor is None:
                    print("      %-10s sin dato" % clave)
                    continue
                nota = "" if f_real == fecha else "  (ultimo dato: %s)" % f_real
                print("      %-10s %+7.1f pb%s" % (clave, valor, nota))

    # --- 5. Cruce opcional con la amplitud ---
    if cruce:
        print("\n--- CRUCE CON AMPLITUD (solo fechas presentes en AMBAS series) ---")
        disp = {e.get("fecha"): e.get("dispersion_ratio")
                for e in cruce if e.get("dispersion_ratio") is not None}
        mcc = {e.get("fecha"): e.get("mcclellan")
               for e in cruce if e.get("mcclellan") is not None}
        comunes = sorted(set(disp) & {f for f, _ in series_p["p30y_10y"]})
        if not comunes:
            print("  Sin fechas comunes. Las series arrancan en momentos distintos.")
        else:
            print("  %d fechas comunes (%s a %s)"
                  % (len(comunes), comunes[0], comunes[-1]))
            print("  %-12s %10s %10s %12s %12s"
                  % ("fecha", "disp", "mccl", "p10y_3m", "p30y_10y"))
            p1 = dict(series_p["p10y_3m"])
            p2 = dict(series_p["p30y_10y"])
            for fecha in comunes:
                print("  %-12s %10.3f %10s %12s %12s" % (
                    fecha, disp[fecha],
                    ("%.1f" % mcc[fecha]) if fecha in mcc else "n/d",
                    ("%+.1f" % p1[fecha]) if fecha in p1 else "n/d",
                    ("%+.1f" % p2[fecha]) if fecha in p2 else "n/d"))
            print("\n  NOTA: %d fechas es muestra corta. Sirve para mirar la "
                  "coincidencia\n  temporal, no para inferir relacion."
                  % len(comunes))

    print("\n--- LECTURA ---")
    print("  Este script describe la serie; no infiere causalidad ni recomienda")
    print("  nada. Los episodios salen de los datos, no de fechas elegidas a")
    print("  posteriori. Toda la serie disponible es de un solo ciclo de tipos.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Vista por episodio de la curva de tipos.")
    parser.add_argument("--fichero", default="rates_history.json")
    parser.add_argument("--hitos", default=None,
                        help="FECHA:etiqueta separados por comas, "
                             "p.ej. 2026-07-29:FOMC")
    parser.add_argument("--cruzar", default=None,
                        help="Ruta a breadth_history.json para cruzar por fecha")
    args = parser.parse_args(argv)

    try:
        historico = cargar(args.fichero)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1

    if not historico:
        print("La serie esta vacia.", file=sys.stderr)
        return 1

    cruce = None
    if args.cruzar:
        try:
            cruce = cargar(args.cruzar)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print("AVISO: no se pudo leer %s (%s). Se omite el cruce."
                  % (args.cruzar, exc), file=sys.stderr)

    informe(historico, parsear_hitos(args.hitos), cruce)
    return 0


if __name__ == "__main__":
    sys.exit(main())
