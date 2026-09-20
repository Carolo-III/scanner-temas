# -*- coding: utf-8 -*-
"""P89 (19/09/2026) — Este fichero ya NO comprueba paridad: comprueba que NO haya
nada que emparejar.

Hasta el P89, la Celda 2 duplicaba las ~5.200 lineas de scanner.py y este test
verificaba que las dos copias fueran identicas. La duplicacion desaparecio: la
Celda 2 descarga scanner.py del repo y lo ejecuta, asi que solo hay una fuente de
verdad y la paridad es imposible de romper por construccion.

Lo que queda por vigilar es que nadie deshaga ese cambio pegando codigo otra vez
en el notebook, y que el mecanismo de carga conserve sus dos piezas criticas:
la guarda de __name__ y el volcado de secretos ANTES del exec.
"""
import ast, json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
NOTEBOOK = RAIZ / 'Scanner_Temas_V001.ipynb'

# La Celda 4 es un espejo manual de main() y la cubre test_celda4_superficie.py.
# Aqui solo interesa que ninguna celda vuelva a DEFINIR funciones del scanner.
MAX_FUNCIONES_POR_CELDA = 0


def _celdas_codigo():
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    for i, c in enumerate(nb['cells']):
        if c.get('cell_type') == 'code':
            yield i, ''.join(c['source'])


def _sin_magics(src):
    return '\n'.join(l for l in src.splitlines()
                     if not l.lstrip().startswith(('!', '%')))


def _celda_carga():
    """La celda que descarga y ejecuta scanner.py."""
    for i, src in _celdas_codigo():
        if 'raw.githubusercontent.com' in src and 'exec(' in src:
            return i, src
    raise AssertionError('Ninguna celda descarga y ejecuta scanner.py: '
                         'el notebook no sigue el modelo del P89')


def test_el_notebook_no_duplica_codigo_del_scanner():
    """Si alguien vuelve a pegar las definiciones en el notebook, reaparece el
    problema que el P89 elimino: dos copias que se desincronizan en silencio."""
    culpables = []
    for i, src in _celdas_codigo():
        try:
            arbol = ast.parse(_sin_magics(src))
        except SyntaxError:
            continue
        n = sum(1 for nodo in ast.walk(arbol)
                if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)))
        if n > MAX_FUNCIONES_POR_CELDA:
            culpables.append('celda %d define %d funcion(es)' % (i, n))
    assert not culpables, (
        'El notebook vuelve a definir funciones; con el P89 debe limitarse a '
        'descargar scanner.py: ' + '; '.join(culpables))


def test_la_celda_de_carga_protege_el_arranque_de_main():
    """scanner.py termina con `if __name__ == '__main__': main()` y en Colab
    __name__ YA vale '__main__'. Sin reasignarlo antes del exec, abrir el
    notebook lanzaria el pipeline entero sin que nadie lo pidiera."""
    _, src = _celda_carga()
    arbol = ast.parse(src)
    # Ojo: NO vale buscar "__name__" en cualquier sitio. La propia celda lo LEE
    # para restaurarlo despues (_nombre_previo = _g.get('__name__')), asi que hay
    # que exigir que aparezca como DESTINO de una asignacion. Con la version laxa
    # este test pasaba aunque se quitara la guarda (comprobado con test negativo).
    # Hay que ser preciso: la celda toca __name__ TRES veces — lo lee para
    # guardarlo, lo pone a un valor falso, y lo restaura. Solo la del medio es la
    # guarda. Se exige una asignacion a __name__ cuyo valor sea una CADENA
    # LITERAL distinta de '__main__' y que este ANTES del exec. Sin este nivel de
    # detalle el test pasaba aunque se quitara la guarda (comprobado).
    reasigna = any(
        isinstance(n, ast.Assign)
        and any('__name__' in ast.unparse(t) for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str)
        and n.value.value != '__main__'
        and src.find(ast.unparse(n)) < src.find('exec(')
        for n in ast.walk(arbol))
    assert reasigna, (
        'La celda de carga no reasigna __name__ antes de ejecutar scanner.py: '
        'el pipeline arrancaria solo al abrir el notebook')


def test_los_secretos_se_vuelcan_antes_de_ejecutar():
    """scanner.py resuelve GITHUB_TOKEN, ANTHROPIC_KEY, etc. como constantes de
    MODULO, es decir en el momento del exec. Si el volcado de Colab Secrets a
    os.environ ocurriera despues, todas quedarian vacias."""
    _, src = _celda_carga()
    pos_exec = src.find('exec(')
    for secreto in ('SCANNER_TOKEN', 'ANTHROPIC_KEY', 'TELEGRAM_TOKEN',
                    'TELEGRAM_CHAT_ID', 'FMP_KEY'):
        pos = src.find(secreto)
        assert pos != -1, 'La celda de carga no vuelca el secreto %s' % secreto
        assert pos < pos_exec, (
            '%s se vuelca DESPUES del exec: scanner.py lo leeria vacio' % secreto)


def test_todas_las_celdas_de_codigo_parsean():
    for i, src in _celdas_codigo():
        ast.parse(_sin_magics(src))
