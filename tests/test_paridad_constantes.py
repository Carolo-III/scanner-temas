# -*- coding: utf-8 -*-
"""
tests/test_paridad_constantes.py

PARIDAD DE CONSTANTES DE MODULO (06/08/2026)

La paridad AST que vigila el CI compara FUNCIONES. Las asignaciones de nivel de
modulo quedan fuera, y ahi viven cosas que cambian el comportamiento del scanner:
la watchlist personal, los umbrales, el calendario macro, los universos.

Se detecto al quitar el ticker ACRV: estaba tanto en scanner.py como en la celda de
definiciones del notebook, y haberlo quitado solo de uno habria dejado a Colab
analizando un valor que la Action ya no analiza — con el CI en verde, porque ninguna
funcion habia cambiado.

Comprobado el dia que se escribio este test: 20 constantes comunes, de las cuales
solo difieren las cinco credenciales, y difieren A PROPOSITO porque cada entorno las
lee de su propio origen (Colab Secrets frente a secrets de GitHub Actions).

Este test NO impone que las dos definiciones sean identicas en todo: impone que lo
sean en todo MENOS en esas cinco, y avisa si aparece una constante nueva solo en uno
de los dos lados.
"""

import ast
import json
import os

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANNER = os.path.join(RAIZ, 'scanner.py')
NOTEBOOK = os.path.join(RAIZ, 'Scanner_Temas_V001.ipynb')

# Difieren a proposito: cada entorno resuelve sus credenciales de otra forma.
EXCEPCIONES = {
    'ANTHROPIC_KEY', 'FMP_KEY', 'GITHUB_TOKEN',
    'TELEGRAM_TOKEN', 'TELEGRAM_CHAT_ID',
}


def _constantes(codigo):
    """{NOMBRE: valor_desnormalizado} de las asignaciones de nivel de modulo.

    Solo asignaciones simples a un unico nombre. Se comparan por su representacion
    fuente (ast.unparse), que ignora comentarios y formato pero distingue cualquier
    cambio real de valor.
    """
    arbol = ast.parse(codigo)
    salida = {}
    for nodo in arbol.body:
        if not isinstance(nodo, ast.Assign) or len(nodo.targets) != 1:
            continue
        destino = nodo.targets[0]
        if not isinstance(destino, ast.Name):
            continue
        try:
            salida[destino.id] = ast.unparse(nodo.value)
        except Exception:  # pragma: no cover - defensivo
            continue
    return salida


def _celda_definiciones():
    """Codigo de la celda que DEFINE las funciones del scanner.

    Se localiza por contenido (la que define analyze_universe) y no por indice: al
    eliminar la Celda 3 el 06/08/2026 todos los indices se desplazaron, y un test que
    dependiera de ellos habria fallado por la razon equivocada.
    """
    with open(NOTEBOOK, 'r', encoding='utf-8') as fh:
        nb = json.load(fh)
    for celda in nb['cells']:
        if celda.get('cell_type') != 'code':
            continue
        src = ''.join(celda['source'])
        try:
            arbol = ast.parse(src)
        except SyntaxError:
            continue
        definidas = {n.name for n in arbol.body if isinstance(n, ast.FunctionDef)}
        if 'analyze_universe' in definidas:
            return src
    pytest.fail('No se encontro la celda que define analyze_universe')


def test_constantes_comunes_coinciden():
    """Una divergencia aqui significa que la Action y Colab analizan cosas distintas."""
    en_scanner = _constantes(open(SCANNER, encoding='utf-8').read())
    en_celda = _constantes(_celda_definiciones())
    comunes = (set(en_scanner) & set(en_celda)) - EXCEPCIONES
    assert comunes, 'No se detecto ninguna constante comun: revisar el extractor'
    divergentes = sorted(k for k in comunes if en_scanner[k] != en_celda[k])
    detalle = '\n  '.join(
        '%s:\n    scanner.py: %s\n    notebook  : %s'
        % (k, en_scanner[k][:200], en_celda[k][:200]) for k in divergentes)
    assert not divergentes, (
        'Constantes de modulo con valor distinto en scanner.py y en el notebook '
        '(la paridad AST no las cubre):\n  ' + detalle)


def test_no_aparecen_constantes_solo_en_un_lado():
    """Una constante nueva en un solo lado suele ser un espejo a medio aplicar."""
    en_scanner = _constantes(open(SCANNER, encoding='utf-8').read())
    en_celda = _constantes(_celda_definiciones())
    solo_scanner = sorted(set(en_scanner) - set(en_celda) - EXCEPCIONES)
    solo_celda = sorted(set(en_celda) - set(en_scanner) - EXCEPCIONES)
    assert not solo_scanner, (
        'Constantes definidas solo en scanner.py: %s. Anadelas al notebook o, si son '
        'especificas del entorno, a EXCEPCIONES.' % solo_scanner)
    assert not solo_celda, (
        'Constantes definidas solo en el notebook: %s. Anadelas a scanner.py o, si son '
        'especificas del entorno, a EXCEPCIONES.' % solo_celda)


def test_las_excepciones_siguen_siendo_necesarias():
    """Guardia del propio test: si una excepcion deja de divergir, sobra en la lista.

    Evita que EXCEPCIONES crezca y acabe tapando divergencias reales.
    """
    en_scanner = _constantes(open(SCANNER, encoding='utf-8').read())
    en_celda = _constantes(_celda_definiciones())
    innecesarias = sorted(
        k for k in EXCEPCIONES
        if k in en_scanner and k in en_celda and en_scanner[k] == en_celda[k])
    assert not innecesarias, (
        'Estas constantes ya NO divergen, asi que sobran en EXCEPCIONES y estarian '
        'tapando futuras divergencias: %s' % innecesarias)
