Ver /mnt/user-data/outputs/scanner.py

Este fichero contiene:
- Filtro SCT≥40 en setups válidos
- Penalización CMF negativo: -8 puntos cuando CMF<0 y ADX>25
- Tabla comparativa con RIESGO actual (antes de mejoras)
- max_tokens=4000
- Prompt con $TICKER en ENTRADAS
- Lógica RIESGO mecánica (será corregida en próxima sesión)

CAMBIOS PENDIENTES PARA PRÓXIMA SESIÓN (puntos 0-6):
0. Corregir calc_entry_range() — objetivo derivado de stop×RB, no máximo 52s
1. Nueva función calc_riesgo() — BAJO/MEDIO/ALTO en Python antes de Claude
2. Añadir filtro ADX≥20 en valid=[]
3. Tope R/B=5 en fórmula RANKING
4. Objetivos escalonados (primer objetivo + final)
5. Distinguir pullback vs "Ruptura pendiente" en prompt
6. Triggers concretos para EXTENDIDOS en prompt