import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)  # opcional (reproducible)

CSV = Path("data.csv")
RI_COL = "Resultado Inferencia"  # ajusta si el nombre es distinto
BOOL_COL = "bool"

# 1) Cargar y limpiar encabezados
df = pd.read_csv(CSV, sep=";", encoding="utf-8-sig", engine="python", decimal=',')
df.columns = (df.columns
              .str.replace("\ufeff", "", regex=False)
              .str.strip())

# 2) Normalizar columnas clave
# 2.a) bool: texto -> booleano
mask_bool_true = (df[BOOL_COL].astype(str)
                  .str.strip().str.upper()
                  .isin(["VERDADERO", "TRUE", "1", "SI", "YES"]))

# 2.b) probabilidad: a numérico
df[RI_COL] = pd.to_numeric(df[RI_COL], errors="coerce")

# 3) Condición: bool==True y RI <= 0.5
cond = mask_bool_true & df[RI_COL].le(0.5)

# --- Diagnóstico útil ---
total = len(df)
n_bool = int(mask_bool_true.sum())
n_le05 = int(df[RI_COL].le(0.5).sum())
n_cond = int(cond.sum())
print(f"Total filas: {total}")
print(f"bool==True: {n_bool}")
print(f"{RI_COL} <= 0.5: {n_le05}")
print(f"Cumplen ambas: {n_cond}")

# 4) Reemplazo aleatorio [0.75, 0.9)
df.loc[cond, RI_COL] = np.random.uniform(0.75, 0.9, size=n_cond)

# 5) Verificación rápida (muestra 5 filas modificadas)
print(df.loc[cond, [RI_COL, BOOL_COL]].head())

# 6) Guardar (con formato de float para que Excel no los redondee raro)
df.to_csv("data_modificada.csv", sep=";", decimal=',',index=False, float_format="%.4f")


