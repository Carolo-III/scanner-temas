# -*- coding: utf-8 -*-
"""P89 (19/09/2026) — Reemplaza la paridad de constantes, que ya no tiene objeto.

Este test nacio el 06/08 porque la watchlist, los umbrales y el calendario vivian
DUPLICADOS en scanner.py y en el notebook, y quitar un ticker de un solo lado
dejaba a Colab analizando algo que la Action ya no analizaba, con el CI en verde.

Con el P89 esa duplicacion desaparece. Lo que queda es impedir que vuelva: si
alguien define de nuevo en el notebook una constante que pertenece a scanner.py,
esa definicion GANARIA sobre la del scanner si se ejecuta despues del exec, o
seria pisada silenciosamente si se ejecuta antes. Las dos cosas son trampas.
"""
import ast
import json
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANNER = os.path.join(RAIZ, 'scanner.py')
NOTEBOOK = os.path.join(RAIZ, 'Scanner_Temas_V001.ipynb')

# Nombres de trabajo de la propia celda de carga, que no son configuracion del
# scanner y por tanto no colisionan con nada.
PERMITIDAS_EN_NOTEBOOK = {'RAMA', 'URL'}


def _constantes_de_modulo(codigo):
    """{NOMBRE: fuente} de asignaciones simples de nivel superior en MAYUSCULAS."""
    salida = {}
    for nodo in ast.parse(codigo).body:
        if not isinstance(nodo, ast.Assign) or len(nodo.targets) != 1:
            continue
        destino = nodo.targets[0]
        if not isinstance(destino, ast.Name) or not destino.id.isupper():
            continue
        try:
            salida[destino.id] = ast.unparse(nodo.value)
        except Exception:  # pragma: no cover - defensivo
            continue
    return salida


def _celdas_codigo():
    with open(NOTEBOOK, 'r', encoding='utf-8') as fh:
        nb = json.load(fh)
    for i, celda in enumerate(nb['cells']):
        if celda.get('cell_type') != 'code':
            continue
        src = '\n'.join(l for l in ''.join(celda['source']).splitlines()
                        if not l.lstrip().startswith(('!', '%')))
        try:
            ast.parse(src)
        except SyntaxError:
            continue
        yield i, src


def test_el_notebook_no_redefine_constantes_del_scanner():
    en_scanner = set(_constantes_de_modulo(open(SCANNER, encoding='utf-8').read()))
    choques = []
    for i, src in _celdas_codigo():
        for nombre in _constantes_de_modulo(src):
            if nombre in en_scanner:
                choques.append('celda %d redefine %s' % (i, nombre))
    assert not choques, (
        'El notebook redefine constantes que pertenecen a scanner.py; segun el '
        'orden de ejecucion una de las dos gana en silencio: ' + '; '.join(choques))


def test_las_constantes_del_notebook_son_solo_de_carga():
    for i, src in _celdas_codigo():
        ajenas = set(_constantes_de_modulo(src)) - PERMITIDAS_EN_NOTEBOOK
        assert not ajenas, (
            'La celda %d define constantes que no son del mecanismo de carga: %s. '
            'La configuracion vive en scanner.py.' % (i, sorted(ajenas)))


def test_scanner_sigue_teniendo_su_configuracion():
    """Guardia inversa: si scanner.py se quedara sin constantes, los dos tests de
    arriba pasarian trivialmente y no estarian comprobando nada."""
    en_scanner = _constantes_de_modulo(open(SCANNER, encoding='utf-8').read())
    assert 'PERSONAL_WATCHLIST' in en_scanner, 'scanner.py no define PERSONAL_WATCHLIST'
    assert len(en_scanner) >= 15, (
        'Solo %d constantes de modulo en scanner.py: revisar el extractor'
        % len(en_scanner))
