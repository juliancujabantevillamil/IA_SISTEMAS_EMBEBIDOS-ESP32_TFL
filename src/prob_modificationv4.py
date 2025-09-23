# desagrupar_probabilidades.py
# Autor: tú 🙂
# Propósito: limitar la cantidad de repeticiones de una probabilidad redondeada a N decimales,
#            manteniéndolas lo más cerca posible del valor original.
#
# I/O:
#   input  -> data_corregida1.csv
#   output -> data_corregida2.csv
#
# Supuestos:
#   - Separador ; y coma decimal (CSV estilo Excel en ES).
#   - La columna de probabilidad se llama "Resultado Inferencia".
#   - 0 y 1 quedan exentos del límite de repeticiones.
#
# Cómo funciona (resumen):
#   1) Convierte las probabilidades en "buckets" enteros a N decimales (ej. 4 -> escala 10000).
#   2) Cuenta repeticiones por bucket.
#   3) Si un bucket (≠0/1) excede el máximo permitido, mantiene las primeras K y
#      reubica las restantes en buckets cercanos con cupo libre, buscando primero
#      ±0.0001, luego ±0.0002, etc., hasta encontrar hueco.
#   4) Vuelve a flotantes y guarda el CSV.

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# ----------------- Configuración -----------------
INPUT_CSV  = Path("data_corregida1.csv")
OUTPUT_CSV = Path("data_corregida2.csv")

RI_COL = "Resultado Inferencia"   # nombre de la columna de probabilidad
DECIMALS = 4                      # redondeo objetivo
MAX_REPS = 5                      # máximo permitido por valor (salvo 0 y 1)
RNG_SEED = 42                     # para reproducibilidad (afecta el orden en búsquedas)
SEP = ";"
ENC = "utf-8-sig"

# ----------------- Utilidades -----------------
def sanitize_headers(cols):
    return (cols.str.replace("\ufeff", "", regex=False)
                .str.strip())

def to_bucket(x, scale):
    """Convierte float [0,1] a bucket entero con 'scale'=10**DECIMALS."""
    if pd.isna(x):
        return None
    # Limitar numéricamente a [0,1] por seguridad.
    x = min(max(float(x), 0.0), 1.0)
    # Redondeo al bucket (p.ej. 0.74612 -> 7461 si DECIMALS=4)
    return int(round(x * scale))

def from_bucket(b, scale):
    """Convierte bucket entero al float correspondiente."""
    return b / scale

def find_nearest_free_bucket(base_b, counts, max_reps, min_b, max_b, rng):
    """
    Busca el bucket libre más cercano a 'base_b' cuyo conteo actual sea < max_reps.
    Explora en anillos: ±1, ±2, ±3, ... (distancia en 'buckets'), aleatorizando el orden.
    """
    if counts[base_b] < max_reps:
        return base_b

    radius = 1
    # Para evitar bucles infinitos: como hay capacidad total de sobra, esto encuentra hueco.
    while True:
        candidates = []
        left  = base_b - radius
        right = base_b + radius
        if left >= min_b:
            candidates.append(left)
        if right <= max_b:
            candidates.append(right)

        rng.shuffle(candidates)
        for c in candidates:
            if counts[c] < max_reps:
                return c

        radius += 1
        # (En datasets enormes, podrías añadir un límite máximo de radius y fallback.)

# ----------------- Carga -----------------
np.random.seed(RNG_SEED)
rng = np.random.default_rng(RNG_SEED)

df = pd.read_csv(INPUT_CSV, sep=SEP, encoding=ENC, engine="python", decimal=",")
df.columns = sanitize_headers(df.columns)

# A numérico con coerce por si hay símbolos raros
df[RI_COL] = pd.to_numeric(df[RI_COL], errors="coerce")

# ----------------- Preparación -----------------
scale = 10 ** DECIMALS
min_bucket, max_bucket = 0, scale  # 0..10000 para 4 decimales
exempt_buckets = {min_bucket, max_bucket}  # 0 y 1 quedan exentos

# Serie de buckets (enteros). Guardamos también índices válidos.
buckets = df[RI_COL].apply(lambda x: to_bucket(x, scale))
valid_idx = buckets[buckets.notna()].index
buckets = buckets.astype("Int64")  # nullable int

# Conteos iniciales (Counter maneja enteros)
counts = Counter(int(b) for b in buckets.dropna().tolist())

# ----------------- Reasignación -----------------
# Recorremos bucket por bucket; si excede y no es exento, reasignamos los excedentes
# en índices (después de mantener MAX_REPS en ese bucket).
order = rng.permutation(len(valid_idx))  # aleatorizar orden de procesamiento global
idx_list = valid_idx.to_list()
idx_list = [idx_list[i] for i in order]

# Para acelerar, agrupamos índices por bucket actual
indices_por_bucket = {}
for i in idx_list:
    b = int(buckets[i])
    indices_por_bucket.setdefault(b, []).append(i)

ajustados = 0
excesos_iniciales = {
    b: c for b, c in counts.items()
    if b not in exempt_buckets and c > MAX_REPS
}

for b, c in excesos_iniciales.items():
    if b in exempt_buckets or c <= MAX_REPS:
        continue
    # Mantener primeras MAX_REPS en el mismo bucket, reasignar el resto
    ixs = indices_por_bucket.get(b, [])
    # Por si acaso (defensivo)
    if len(ixs) <= MAX_REPS:
        continue

    to_keep = ixs[:MAX_REPS]
    to_move = ixs[MAX_REPS:]  # excedentes

    for idx in to_move:
        # Buscar bucket cercano con cupo
        new_b = find_nearest_free_bucket(
            base_b=b,
            counts=counts,
            max_reps=MAX_REPS,
            min_b=min_bucket + 1,   # evitamos 0
            max_b=max_bucket - 1,   # evitamos 1
            rng=rng
        )
        # Si por alguna razón no encontró (muy improbable), caer a base_b (no cambia)
        if new_b is None:
            new_b = b

        # Aplicar el cambio
        if new_b != b:
            counts[b]    -= 1
            counts[new_b] += 1
            buckets.at[idx] = new_b
            ajustados += 1

# ----------------- Volver a float y guardar -----------------
# Pasar buckets a floats (redondeados a DECIMALS)
new_vals = buckets.apply(lambda bb: from_bucket(int(bb), scale) if pd.notna(bb) else np.nan)
df[RI_COL] = new_vals

# Guardar con coma decimal y 4 decimales
df.to_csv(OUTPUT_CSV, sep=SEP, index=False, decimal=",", float_format=f"%.{DECIMALS}f", encoding=ENC)

# ----------------- Reporte rápido -----------------
def resumen(series_float, decimals, max_reps, excluir={0.0, 1.0}):
    r = series_float.round(decimals)
    vc = r.value_counts()
    vc_filtrado = vc[~vc.index.isin(excluir)]
    excedentes = vc_filtrado[vc_filtrado > max_reps]
    return int(vc_filtrado.max() if not vc_filtrado.empty else 0), int(excedentes.sum())

max_rep_final, exceso_final = resumen(df[RI_COL], DECIMALS, MAX_REPS)
print("=== RESUMEN ===")
print(f"Filas totales: {len(df)}")
print(f"Máx. repetición final (sin 0/1): {max_rep_final}")
print(f"Filas que aún exceden el límite (sin 0/1): {exceso_final}")
print(f"Filas ajustadas: {ajustados}")
print(f"Archivo guardado en: {OUTPUT_CSV.resolve()}")
