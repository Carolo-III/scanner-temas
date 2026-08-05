# -*- coding: utf-8 -*-
"""
tests/test_celda4_superficie.py

TEST DE SUPERFICIE DE RIESGO DE LA CELDA 4 (02/08/2026)

La paridad AST del CI cubre scanner.py <-> Celda 2. La Celda 4 queda FUERA: son
espejos manuales de main(). Ese hueco ha dejado pasar fallos reales:

  - 27/07: Celda 4 sin la llamada a Telegram — semanas de silencio sin que
    nada avisara.
  - 02/08: main() llamaba validar_informe(analisis, valid_final) y la Celda 4
    validar_informe(analisis, valid[:5]) — DOS espejos distintos, ambos rotos,
    y el CI en verde.
  - 02/08: una rama sin la curva de tipos habria pasado el CI limpiamente,
    porque la paridad Celda2<->scanner seguia siendo correcta en ambos.

Este test no compara codigo linea a linea (la Celda 4 y main() difieren de forma
legitima: la celda no usa clean_nan, imprime cosas distintas). Compara la
SUPERFICIE: que orquesten las mismas piezas y persistan los mismos ficheros.

Tres comprobaciones:
  1. Mismo conjunto de ficheros subidos a GitHub.
  2. Mismas funciones de orquestacion llamadas.
  3. Mismos argumentos en las llamadas criticas (esto es lo que habria cazado
     el valid_final / valid).
"""

import ast
import json
import os

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANNER = os.path.join(RAIZ, 'scanner.py')
NOTEBOOK = os.path.join(RAIZ, 'Scanner_Temas_V001.ipynb')

# Funciones cuya presencia en AMBOS lados es obligatoria. Si se anade una nueva
# etapa al pipeline, se anade aqui: el test obliga a no olvidar el espejo.
ORQUESTACION = {
    'analyze_universe',
    'calc_groups',
    'calc_market_breadth',
    'get_macro_data',
    'get_fundamentals',
    'update_setups_history',
    'update_history',
    'generate_analysis',
    'validar_informe',
    'generate_alerts',
    'actualizar_breadth_history',
    'actualizar_rates_history',
    'anotar_tendencia_en_breadth',
    'actualizar_checkpoints_history',
    'anotar_resoluciones',
    'actualizar_resoluciones_history',
}

# Llamadas donde los ARGUMENTOS deben coincidir. Es la comprobacion que habria
# detectado el bug del 02/08.
ARGS_CRITICOS = {
    'validar_informe',
    'actualizar_breadth_history',
    'actualizar_rates_history',
    'actualizar_checkpoints_history',
    'anotar_resoluciones',
    'actualizar_resoluciones_history',
}

# Diferencias intencionadas: main() envuelve en clean_nan antes de serializar,
# la Celda 4 no. Se normaliza para no generar ruido permanente.
ENVOLTORIOS_IGNORADOS = {'clean_nan'}


def _cargar_celda(indice):
    with open(NOTEBOOK, 'r', encoding='utf-8') as fh:
        nb = json.load(fh)
    return ''.join(nb['cells'][indice]['source'])


def _celda_pipeline():
    """Devuelve el codigo de la celda que ejecuta el pipeline.

    Se localiza por contenido (la que llama a upload_files_to_github), no por
    indice fijo: si alguien reordena las celdas, el test sigue funcionando en
    vez de romperse por una razon equivocada.
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
        llamadas = {n.func.id for n in ast.walk(arbol)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        # La celda del pipeline LLAMA a upload_files_to_github sin DEFINIRLA
        # (la Celda 2 la define; buscar solo por texto cogeria esa).
        if 'upload_files_to_github' in llamadas and 'upload_files_to_github' not in definidas:
            return src
    pytest.fail('No se encontro ninguna celda que LLAME a upload_files_to_github '
                'sin definirla (la del pipeline)')


def _main_de_scanner():
    with open(SCANNER, 'r', encoding='utf-8') as fh:
        src = fh.read()
    arbol = ast.parse(src)
    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef) and nodo.name == 'main':
            return ast.get_source_segment(src, nodo)
    pytest.fail('scanner.py no define main()')


def _ficheros_subidos(codigo):
    """Claves asignadas a ficheros_subida + las del literal inicial."""
    arbol = ast.parse(codigo)
    claves = set()
    for nodo in ast.walk(arbol):
        # ficheros_subida['x.json'] = ...
        if (isinstance(nodo, ast.Subscript)
                and isinstance(nodo.value, ast.Name)
                and nodo.value.id == 'ficheros_subida'
                and isinstance(nodo.slice, ast.Constant)):
            claves.add(nodo.slice.value)
        # ficheros_subida = {'a.json': ..., 'b.json': ...}
        if isinstance(nodo, ast.Assign):
            for destino in nodo.targets:
                if (isinstance(destino, ast.Name)
                        and destino.id == 'ficheros_subida'
                        and isinstance(nodo.value, ast.Dict)):
                    for clave in nodo.value.keys:
                        if isinstance(clave, ast.Constant):
                            claves.add(clave.value)
    # upload_to_github('x.json', ...) — subida individual. main() la usa para
    # setups_history.json como checkpoint de seguridad ANTES del analisis de
    # Claude; la Celda 4 lo mete en el commit unico. Comportamiento distinto a
    # proposito (2 commits vs 1), pero el FICHERO se persiste en ambos: lo que
    # este test vigila es que no se pierda ninguno, no el numero de commits.
    for nodo in ast.walk(arbol):
        if (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name)
                and nodo.func.id in ('upload_to_github', 'upload_files_to_github')
                and nodo.args and isinstance(nodo.args[0], ast.Constant)
                and isinstance(nodo.args[0].value, str)
                and nodo.args[0].value.endswith('.json')):
            claves.add(nodo.args[0].value)
    return claves


def _llamadas(codigo):
    """{nombre_funcion: [firma_de_argumentos, ...]} para llamadas por nombre."""
    arbol = ast.parse(codigo)
    encontradas = {}
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name):
            args = [_normalizar(a) for a in nodo.args]
            args += ['%s=%s' % (k.arg, _normalizar(k.value))
                     for k in nodo.keywords if k.arg]
            encontradas.setdefault(nodo.func.id, []).append(tuple(sorted(args)))
    return encontradas


def _normalizar(nodo):
    """Texto del argumento, quitando envoltorios intencionados."""
    if (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name)
            and nodo.func.id in ENVOLTORIOS_IGNORADOS and nodo.args):
        return _normalizar(nodo.args[0])
    try:
        return ast.unparse(nodo)
    except Exception:
        return ast.dump(nodo)


# --------------------------------------------------------------------------

def test_mismos_ficheros_subidos():
    """Si main() persiste un fichero que la Celda 4 no, una ejecucion manual
    desde Colab dejaria de actualizarlo sin que nada avisara (y viceversa)."""
    en_main = _ficheros_subidos(_main_de_scanner())
    en_celda = _ficheros_subidos(_celda_pipeline())
    assert en_main, 'No se detecto ningun fichero de subida en main()'
    solo_main = en_main - en_celda
    solo_celda = en_celda - en_main
    assert not solo_main, (
        'Ficheros que main() sube y la Celda 4 NO: %s. '
        'Anade el espejo en la Celda 4.' % sorted(solo_main))
    assert not solo_celda, (
        'Ficheros que la Celda 4 sube y main() NO: %s.' % sorted(solo_celda))


def test_mismas_funciones_de_orquestacion():
    """Cada etapa del pipeline debe estar invocada en ambos lados."""
    en_main = set(_llamadas(_main_de_scanner()))
    en_celda = set(_llamadas(_celda_pipeline()))
    faltan_en_celda = (ORQUESTACION & en_main) - en_celda
    faltan_en_main = (ORQUESTACION & en_celda) - en_main
    assert not faltan_en_celda, (
        'Etapas llamadas en main() y ausentes en la Celda 4: %s'
        % sorted(faltan_en_celda))
    assert not faltan_en_main, (
        'Etapas llamadas en la Celda 4 y ausentes en main(): %s'
        % sorted(faltan_en_main))


def test_argumentos_de_llamadas_criticas_coinciden():
    """Misma funcion invocada con argumentos distintos en cada lado.

    Este es el caso del 02/08: validar_informe(analisis, valid_final) en main()
    frente a validar_informe(analisis, valid[:5]) en la Celda 4. Ambas rotas,
    de formas distintas, y el CI en verde.
    """
    llam_main = _llamadas(_main_de_scanner())
    llam_celda = _llamadas(_celda_pipeline())
    divergencias = []
    for nombre in sorted(ARGS_CRITICOS):
        if nombre not in llam_main or nombre not in llam_celda:
            continue  # lo cubre el test anterior
        if set(llam_main[nombre]) != set(llam_celda[nombre]):
            divergencias.append(
                '%s -> main%s vs Celda4%s'
                % (nombre, sorted(llam_main[nombre]), sorted(llam_celda[nombre])))
    assert not divergencias, (
        'Llamadas criticas con argumentos distintos:\n  ' + '\n  '.join(divergencias))


def test_cobertura_de_la_lista_de_orquestacion():
    """Guardia del propio test: si el pipeline crece y nadie amplia
    ORQUESTACION, este test avisa en vez de dar una falsa sensacion de
    cobertura. Solo mira funciones definidas en scanner.py."""
    with open(SCANNER, 'r', encoding='utf-8') as fh:
        src = fh.read()
    definidas = {n.name for n in ast.parse(src).body
                 if isinstance(n, ast.FunctionDef)}
    llamadas_main = set(_llamadas(_main_de_scanner())) & definidas
    # Auxiliares que no son etapas del pipeline y no necesitan espejo.
    auxiliares = {'clean_nan', '_traza', 'get_github_file', 'upload_files_to_github',
                  'upload_to_github', 'dedup_alertas_por_ticker', 'resoluciones_por_ticker', 'spy_health',
                  'download_prices', 'check_data_health', 'calc_pendientes_curva',
                  'construir_entrada_breadth', 'construir_entrada_rates',
                  'merge_breadth_entry', 'merge_rates_entry',
                  'construir_entradas_checkpoints', 'merge_checkpoints',
                  'construir_entradas_resoluciones', 'merge_resoluciones', '_dias_habiles_entre'}
    sin_vigilar = llamadas_main - ORQUESTACION - auxiliares
    assert not sin_vigilar, (
        'Funciones nuevas llamadas en main() y no vigiladas por este test: %s. '
        'Anadelas a ORQUESTACION (si son etapas del pipeline) o a la lista de '
        'auxiliares.' % sorted(sin_vigilar))
