# -*- coding: utf-8 -*-
"""PUNTO 21 — El test mas importante del repo: paridad AST scanner.py <-> Celda 2.

Los incidentes historicos de este proyecto (informes generados con codigo viejo)
vinieron de desincronizar el notebook y el script. Este test convierte la
verificacion manual de paridad en barrera automatica de CI: un push con los
codebases desincronizados falla en rojo antes de llegar a produccion.
Regla canonica: mismas funciones con cuerpos identicos en ambos; unicas
exclusivas permitidas: main (solo scanner.py) y _get_secret (solo Celda 2).
"""
import ast, json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
EXCLUSIVAS_SCANNER = {'main'}
EXCLUSIVAS_NOTEBOOK = {'_get_secret'}

def _funciones(src):
    """Firma completa + defaults + cuerpo. La version anterior comparaba solo el cuerpo
    (ast.Module(body=n.body)) y un cambio en un argumento por defecto — p. ej.
    umbral_aviso=0.70 -> 0.65 — pasaba invisible: agujero detectado por test negativo
    el 15/07/2026 y corregido comparando el nodo FunctionDef entero."""
    arbol = ast.parse(src)
    return {n.name: ast.dump(n, include_attributes=False)
            for n in ast.walk(arbol)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

def _celda(nb, i):
    return ''.join(nb['cells'][i]['source'])

def test_paridad_ast_scanner_notebook():
    scanner = (RAIZ / 'scanner.py').read_text()
    nb = json.loads((RAIZ / 'Scanner_Temas_V001.ipynb').read_text())
    a = _funciones(scanner)
    b = _funciones(_celda(nb, 2))
    comunes = set(a) & set(b)
    distintas = sorted(f for f in comunes if a[f] != b[f])
    assert not distintas, f'Cuerpos distintos entre scanner.py y Celda 2: {distintas}'
    solo_scanner = set(a) - set(b)
    solo_celda = set(b) - set(a)
    assert solo_scanner == EXCLUSIVAS_SCANNER, f'Funciones inesperadas solo en scanner.py: {sorted(solo_scanner - EXCLUSIVAS_SCANNER)}'
    assert solo_celda == EXCLUSIVAS_NOTEBOOK, f'Funciones inesperadas solo en Celda 2: {sorted(solo_celda - EXCLUSIVAS_NOTEBOOK)}'

def test_todas_las_celdas_de_codigo_parsean():
    nb = json.loads((RAIZ / 'Scanner_Temas_V001.ipynb').read_text())
    for i, c in enumerate(nb['cells']):
        if c.get('cell_type') == 'code':
            src = ''.join(c['source'])
            # las celdas Colab pueden llevar magics (!pip, %); se filtran lineas magicas
            limpio = '\n'.join(l for l in src.splitlines()
                               if not l.lstrip().startswith(('!', '%')))
            ast.parse(limpio)  # lanza SyntaxError si la celda esta rota
