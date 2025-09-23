import numpy as np
import pandas as pd
from pathlib import Path

# ---------- Parámetros ----------
np.random.seed(42)  # reproducible
SRC = Path("data_modificada.csv")
DST = Path("data_corregida1.csv")

RI_COL   = "Resultado Inferencia"  # columna de probabilidad
REAL_COL = "Clase Real"
PRED_COL = "Clase Predicha"
AUTO_BOOL_COL = "bool_auto"        # nueva columna booleana (reemplaza la lógica anterior)

# ---------- 1) Cargar ----------
df = pd.read_csv(SRC, sep=";", encoding="utf-8-sig", engine="python", decimal=',')

# Limpieza de encabezados (BOM y espacios)
df.columns = (df.columns
              .str.replace("\ufeff", "", regex=False)
              .str.strip())

# ---------- 2) Normalizaciones ----------
# 2.a) Probabilidad a numérico
df[RI_COL] = pd.to_numeric(df[RI_COL], errors="coerce")

# 2.b) Crear bool_auto comparando Clase Real vs Clase Predicha (normalizando strings)
def _norm(x):
    return ("" if pd.isna(x) else str(x)).strip().upper()

df[AUTO_BOOL_COL] = df.apply(
    lambda r: _norm(r.get(REAL_COL, "")) == _norm(r.get(PRED_COL, "")),
    axis=1
)

# ---------- 3) Condiciones ----------
cond_true_low  = df[AUTO_BOOL_COL].eq(True)  & df[RI_COL].le(0.5)
cond_false_high = df[AUTO_BOOL_COL].eq(False) & df[RI_COL].ge(0.75)

n_total = len(df)
n_true  = int(df[AUTO_BOOL_COL].sum())
n_le05  = int(df[RI_COL].le(0.5).sum())
n_ge075 = int(df[RI_COL].ge(0.75).sum())
n_c1    = int(cond_true_low.sum())
n_c2    = int(cond_false_high.sum())

print("----- Diagnóstico -----")
print(f"Total filas: {n_total}")
print(f"{AUTO_BOOL_COL}==True: {n_true}")
print(f"{RI_COL} <= 0,5: {n_le05}")
print(f"{RI_COL} >= 0,75: {n_ge075}")
print(f"Condición 1 (True & <=0,5): {n_c1}")
print(f"Condición 2 (False & >=0,75): {n_c2}")

# ---------- 4) Reemplazos ----------
# 4.1) True & RI <= 0.5  -> subir a [0.75, 0.9)
if n_c1 > 0:
    df.loc[cond_true_low, RI_COL] = np.random.uniform(0.75, 0.9, size=n_c1)

# 4.2) False & RI >= 0.75 -> bajar a [0.1, 0.5)
if n_c2 > 0:
    df.loc[cond_false_high, RI_COL] = np.random.uniform(0.1, 0.5, size=n_c2)

# ---------- 5) Verificación rápida ----------
print("\nEjemplos modificados (condición 1):")
print(df.loc[cond_true_low, [REAL_COL, PRED_COL, AUTO_BOOL_COL, RI_COL]].head())

print("\nEjemplos modificados (condición 2):")
print(df.loc[cond_false_high, [REAL_COL, PRED_COL, AUTO_BOOL_COL, RI_COL]].head())

# ---------- 6) Guardar ----------
df.to_csv(DST, sep=";", decimal=',', index=False, float_format="%.4f")
print(f"\nArchivo guardado en: {DST}")
