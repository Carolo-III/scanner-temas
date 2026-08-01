#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/analisis_sensibilidad.py

Analisis de sensibilidad del umbral de dispersion (pregunta previa al P32).

Pregunta que responde: si el umbral de dispersion hubiera estado en otro
nivel, ¿cuantos setups habrian cambiado de lado, y con que consecuencia?

NO toca el pipeline. Se ejecuta a demanda sobre setups_history.json.
NO calibra nada ni recomienda un umbral: solo mide si el umbral es una
palanca viva o inerte. Si es inerte, el P32 no paga su coste.

Uso:
    python tools/analisis_sensibilidad.py
    python tools/analisis_sensibilidad.py --fichero setups_history.json
    python tools/analisis_sensibilidad.py --umbrales 0.28,0.30,0.32,0.35
    python tools/analisis_sensibilidad.py --diagnostico
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

# NO existe umbral de dispersion en produccion: el scanner registra la
# dispersion como instrumentacion, sin filtrar. Por eso la referencia por
# defecto es 0.0 (= todos los setups admitidos, estado real). La columna
# "cambian" se lee entonces como "cuantos se excluirian si hubiera filtro".
UMBRAL_ACTUAL = 0.0

UMBRALES_POR_DEFECTO = [0.26, 0.28, 0.30, 0.32, 0.35, 0.40]

# Nombres alternativos tolerados para cada campo, en orden de preferencia.
CLAVES_DISPERSION = ("dispersion_al_crear", "dispersion", "dispersion_creacion")
CLAVES_RETORNO = ("retorno_pct", "ret_pct", "retorno", "rendimiento_pct", "ret")
CLAVES_ESTADO = ("estado", "status", "resultado")
CLAVES_TICKER = ("ticker", "valor", "simbolo", "symbol")
CLAVES_TIPO = ("tipo", "setup", "tipo_setup", "categoria")
CLAVES_FECHA = ("fecha_creacion", "fecha", "date", "_fecha_snapshot",
                "fecha_alta", "creado")


def primera_clave(registro, candidatas):
    """Devuelve (clave, valor) de la primera candidata presente y no nula."""
    for clave in candidatas:
        if clave in registro and registro[clave] is not None:
            return clave, registro[clave]
    return None, None


def cargar_setups(ruta):
    """Carga setups_history.json y devuelve una lista plana de setups.

    Estructura real del repo: lista de snapshots diarios, cada uno
    {"date": "...", "setups": [ {...}, {...} ]}. Se aplana propagando
    la fecha del snapshot a cada setup como _fecha_snapshot.

    Tolera tambien lista plana de setups y dict de listas.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            "No encuentro %s. Ejecuta el script desde la raiz del repo "
            "o pasa --fichero con la ruta correcta." % ruta
        )

    with open(ruta, "r", encoding="utf-8") as fh:
        datos = json.load(fh)

    planos = []

    def absorber(elemento, fecha_heredada=None):
        if isinstance(elemento, list):
            for sub in elemento:
                absorber(sub, fecha_heredada)
            return
        if not isinstance(elemento, dict):
            return
        # Snapshot envoltorio: baja un nivel propagando la fecha.
        anidadas = [k for k in ("setups", "candidatos", "items")
                    if isinstance(elemento.get(k), list)]
        if anidadas:
            _, fecha = primera_clave(elemento, CLAVES_FECHA)
            for clave in anidadas:
                absorber(elemento[clave], fecha or fecha_heredada)
            return
        # Setup hoja.
        if fecha_heredada is not None:
            elemento = dict(elemento)
            elemento.setdefault("_fecha_snapshot", fecha_heredada)
        planos.append(elemento)

    absorber(datos)

    if not planos:
        raise ValueError(
            "No se ha podido interpretar la estructura de %s. "
            "Ejecuta con --diagnostico." % ruta
        )
    return planos


def diagnostico(registros, ruta):
    """Imprime que campos hay realmente en el fichero. Sin analisis."""
    print("=" * 68)
    print("DIAGNOSTICO DE ESQUEMA -- %s" % ruta)
    print("=" * 68)
    print("Registros totales: %d" % len(registros))
    if not registros:
        return

    contador = Counter()
    for reg in registros:
        if isinstance(reg, dict):
            contador.update(reg.keys())

    print("\nClaves presentes (clave: nº de registros que la traen):")
    for clave, n in contador.most_common():
        print("  %-28s %d" % (clave, n))

    print("\nPrimer registro completo:")
    print(json.dumps(registros[0], indent=2, ensure_ascii=False)[:1200])

    print("\nCobertura de dispersion_al_crear por fecha de snapshot:")
    por_fecha = {}
    for reg in registros:
        if not isinstance(reg, dict):
            continue
        _, f = primera_clave(reg, CLAVES_FECHA)
        _, d = primera_clave(reg, CLAVES_DISPERSION)
        clave = f or "?"
        con, tot = por_fecha.get(clave, (0, 0))
        por_fecha[clave] = (con + (1 if d is not None else 0), tot + 1)
    for f in sorted(por_fecha):
        con, tot = por_fecha[f]
        print("  %-12s %d/%d" % (f, con, tot))

    print("\nDeteccion de campos que necesita el analisis:")
    for etiqueta, candidatas in (
        ("dispersion", CLAVES_DISPERSION),
        ("retorno", CLAVES_RETORNO),
        ("estado", CLAVES_ESTADO),
        ("ticker", CLAVES_TICKER),
    ):
        encontradas = [c for c in candidatas if contador.get(c)]
        estado = encontradas[0] if encontradas else "NO ENCONTRADO"
        print("  %-12s -> %s" % (etiqueta, estado))


def extraer(registros):
    """Extrae los registros utilizables (los que traen dispersion).

    Devuelve (utilizables, n_sin_dispersion).
    """
    utilizables = []
    sin_dispersion = 0

    for reg in registros:
        if not isinstance(reg, dict):
            continue
        _, dispersion = primera_clave(reg, CLAVES_DISPERSION)
        if dispersion is None:
            sin_dispersion += 1
            continue
        try:
            dispersion = float(dispersion)
        except (TypeError, ValueError):
            sin_dispersion += 1
            continue

        _, retorno = primera_clave(reg, CLAVES_RETORNO)
        try:
            retorno = float(retorno) if retorno is not None else None
        except (TypeError, ValueError):
            retorno = None

        _, ticker = primera_clave(reg, CLAVES_TICKER)
        _, estado = primera_clave(reg, CLAVES_ESTADO)
        _, tipo = primera_clave(reg, CLAVES_TIPO)
        _, fecha = primera_clave(reg, CLAVES_FECHA)

        utilizables.append(
            {
                "ticker": ticker or "?",
                "dispersion": dispersion,
                "retorno": retorno,
                "estado": estado or "?",
                "tipo": tipo or "?",
                "fecha": fecha or "?",
            }
        )

    return utilizables, sin_dispersion


def percentil(valores_ordenados, p):
    """Percentil por interpolacion lineal. Sin dependencias externas."""
    if not valores_ordenados:
        return None
    if len(valores_ordenados) == 1:
        return valores_ordenados[0]
    pos = (len(valores_ordenados) - 1) * p
    bajo = int(pos)
    alto = min(bajo + 1, len(valores_ordenados) - 1)
    peso = pos - bajo
    return valores_ordenados[bajo] * (1 - peso) + valores_ordenados[alto] * peso


def resumen_distribucion(utilizables):
    """Distribucion de dispersion_al_crear. Es la salida que mas informa:
    si el rango observado es estrecho, el umbral es inerte por construccion.
    """
    valores = sorted(x["dispersion"] for x in utilizables)
    n = len(valores)
    if n == 0:
        return None

    media = sum(valores) / n
    return {
        "n": n,
        "min": valores[0],
        "p25": percentil(valores, 0.25),
        "mediana": percentil(valores, 0.50),
        "p75": percentil(valores, 0.75),
        "max": valores[-1],
        "media": media,
        "rango": valores[-1] - valores[0],
    }


def evaluar_umbral(utilizables, umbral):
    """Reparte los setups en admitidos/excluidos para un umbral dado."""
    admitidos = [x for x in utilizables if x["dispersion"] >= umbral]
    excluidos = [x for x in utilizables if x["dispersion"] < umbral]

    con_retorno = [x for x in admitidos if x["retorno"] is not None]
    ret_medio = (
        sum(x["retorno"] for x in con_retorno) / len(con_retorno)
        if con_retorno
        else None
    )
    return {
        "umbral": umbral,
        "n_admitidos": len(admitidos),
        "n_excluidos": len(excluidos),
        "admitidos": admitidos,
        "excluidos": excluidos,
        "ret_medio": ret_medio,
        "n_con_retorno": len(con_retorno),
    }


def cambios_de_lado(utilizables, umbral, referencia=UMBRAL_ACTUAL):
    """Setups que cambian de admitido a excluido (o viceversa) respecto
    al umbral de referencia. Esta es la metrica que decide si seguir.
    """
    cambian = []
    for x in utilizables:
        dentro_ref = x["dispersion"] >= referencia
        dentro_new = x["dispersion"] >= umbral
        if dentro_ref != dentro_new:
            cambian.append((x, "entra" if dentro_new else "sale"))
    return cambian


def fmt(valor, decimales=3, sufijo=""):
    return "n/d" if valor is None else ("%.*f%s" % (decimales, valor, sufijo))


def informe(utilizables, sin_dispersion, umbrales, referencia):
    print("=" * 68)
    print("ANALISIS DE SENSIBILIDAD DEL UMBRAL DE DISPERSION")
    print("=" * 68)
    print("Setups con dispersion_al_crear : %d" % len(utilizables))
    print("Setups sin el campo (pre-18/07): %d" % sin_dispersion)
    print("Umbral de referencia           : %.2f" % referencia)

    if not utilizables:
        print("\nSin setups instrumentados. Nada que analizar.")
        return

    dist = resumen_distribucion(utilizables)
    print("\n--- DISTRIBUCION DE dispersion_al_crear ---")
    print("  n       : %d" % dist["n"])
    print("  min     : %s" % fmt(dist["min"]))
    print("  p25     : %s" % fmt(dist["p25"]))
    print("  mediana : %s" % fmt(dist["mediana"]))
    print("  p75     : %s" % fmt(dist["p75"]))
    print("  max     : %s" % fmt(dist["max"]))
    print("  media   : %s" % fmt(dist["media"]))
    print("  rango   : %s" % fmt(dist["rango"]))

    print("\n--- REPARTO POR UMBRAL ---")
    print("  %-8s %-10s %-10s %-12s %-8s" % (
        "umbral", "admitidos", "excluidos", "ret.medio", "cambian"))
    for umbral in umbrales:
        ev = evaluar_umbral(utilizables, umbral)
        cambian = cambios_de_lado(utilizables, umbral, referencia)
        marca = "  <- actual" if abs(umbral - referencia) < 1e-9 else ""
        print("  %-8.2f %-10d %-10d %-12s %-8d%s" % (
            umbral,
            ev["n_admitidos"],
            ev["n_excluidos"],
            fmt(ev["ret_medio"], 2, "%"),
            len(cambian),
            marca,
        ))

    print("\n--- SETUPS QUE CAMBIAN DE LADO ---")
    hubo_cambios = False
    for umbral in umbrales:
        if abs(umbral - referencia) < 1e-9:
            continue
        cambian = cambios_de_lado(utilizables, umbral, referencia)
        if not cambian:
            continue
        hubo_cambios = True
        print("\n  Umbral %.2f (%d cambios):" % (umbral, len(cambian)))
        for x, sentido in sorted(cambian, key=lambda t: t[0]["dispersion"]):
            print("    %-6s %-5s disp=%s  ret=%s  [%s / %s]" % (
                x["ticker"],
                sentido,
                fmt(x["dispersion"]),
                fmt(x["retorno"], 2, "%"),
                x["tipo"],
                x["estado"],
            ))
    if not hubo_cambios:
        print("  Ninguno en el rango de umbrales evaluado.")

    # Lectura, no recomendacion.
    print("\n--- LECTURA ---")
    total_cambios = sum(
        len(cambios_de_lado(utilizables, u, referencia))
        for u in umbrales
        if abs(u - referencia) >= 1e-9
    )
    print("  Cambios de lado acumulados en todo el barrido: %d" % total_cambios)
    if dist["rango"] < 0.05:
        print("  AVISO: el rango observado de dispersion es %s (<0.05)."
              % fmt(dist["rango"]))
        print("  Toda la muestra vive en un intervalo estrecho: cualquier")
        print("  umbral fuera de [%s, %s] es extrapolacion, no calibracion."
              % (fmt(dist["min"]), fmt(dist["max"])))
    if total_cambios <= 3:
        print("  El umbral mueve muy pocas decisiones en esta muestra.")
        print("  Con n=%d no es concluyente en ningun sentido: relanzar"
              % dist["n"])
        print("  cuando la muestra haya crecido, no decidir ahora.")
    print("  Este script NO recomienda un umbral. Solo mide si el umbral")
    print("  es una palanca viva. La decision es tuya.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analisis de sensibilidad del umbral de dispersion.")
    parser.add_argument("--fichero", default="setups_history.json",
                        help="Ruta a setups_history.json")
    parser.add_argument("--umbrales", default=None,
                        help="Lista separada por comas, p.ej. 0.28,0.30,0.32")
    parser.add_argument("--referencia", type=float, default=UMBRAL_ACTUAL,
                        help="Umbral vigente contra el que se comparan cambios")
    parser.add_argument("--diagnostico", action="store_true",
                        help="Solo inspeccionar el esquema del fichero")
    args = parser.parse_args(argv)

    if args.umbrales:
        try:
            umbrales = sorted(float(u) for u in args.umbrales.split(","))
        except ValueError:
            print("ERROR: --umbrales mal formado.", file=sys.stderr)
            return 2
    else:
        umbrales = list(UMBRALES_POR_DEFECTO)

    try:
        registros = cargar_setups(args.fichero)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1

    if args.diagnostico:
        diagnostico(registros, args.fichero)
        return 0

    utilizables, sin_dispersion = extraer(registros)

    if not utilizables and registros:
        print("ERROR: ningun registro trae dispersion_al_crear reconocible.",
              file=sys.stderr)
        print("Relanza con --diagnostico para ver el esquema real.",
              file=sys.stderr)
        return 1

    informe(utilizables, sin_dispersion, umbrales, args.referencia)
    return 0


if __name__ == "__main__":
    sys.exit(main())
