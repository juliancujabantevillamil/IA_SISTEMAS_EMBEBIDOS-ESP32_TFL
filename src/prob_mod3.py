import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# ----------------- Parámetros -----------------
np.random.seed(42)                # reproducible (opcional)
SRC = Path("data_corregida1.csv")
DST = Path("data_corregida2.csv")
RI_COL = "Resultado Inferencia"
DECIMALS_COMPARE = 4              # precisión para comparar/contar repeticiones
ADD_MIN, ADD_MAX = 0.02, 0.049    # dejamos margen para jitter <= 0.001
JITTER_MAX = 0.001
VAL_MAX = 0.9999                  # tope para no saturar en 1.0 al sumar
# ----------------------------------------------

# 1) Leer CSV con coma decimal y limpiar header
df = pd.read_csv(SRC, sep=";", decimal=",", encoding="utf-8-sig", engine="python")
df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

# 2) Asegurar columna numérica
df[RI_COL] = pd.to_numeric(df[RI_COL], errors="coerce")

# 3) Conteo inicial por valor redondeado
rounded = df[RI_COL].round(DECIMALS_COMPARE)
counts = Counter(rounded.dropna())

# 4) Detectar valores repetidos > 4
dupe_vals = [v for v, c in counts.items() if c > 4]

print(f"Valores con repeticiones > 4 (antes): {len(dupe_vals)}")
if dupe_vals:
    print("Ejemplo (valor -> repeticiones):", [(v, counts[v]) for v in dupe_vals[:10]])

# 5) Mapa de conteos vivo que iremos actualizando
counts_live = Counter(counts)

# 6) Función para proponer un nuevo valor cumpliendo: mover 2–5 pp y no crear >4 repeticiones
def propose_new_value(orig):
    # ¿Cabe sumar sin pasar de 1?
    can_add = (orig + (ADD_MAX + JITTER_MAX)) <= VAL_MAX
    # Si no cabe, haremos resta en lugar de suma
    sign = 1 if can_add else -1

    for _ in range(50):  # 50 intentos razonables
        base = np.random.uniform(ADD_MIN, ADD_MAX)
        jitter = np.random.uniform(0.0, JITTER_MAX)
        delta = sign * (base + jitter)
        cand = orig + delta
        # Limitar a [0, 1) por seguridad
        cand = max(0.0, min(VAL_MAX, cand))
        r = round(cand, DECIMALS_COMPARE)
        if counts_live[r] < 4:
            return cand, r

    # Último recurso: barrido fino con paso del redondeo (1e-DECIMALS_COMPARE)
    step = 10 ** (-DECIMALS_COMPARE)
    for k in range(1, 2000):
        cand = orig + sign * (ADD_MIN + k * step)
        if 0.0 <= cand <= VAL_MAX:
            r = round(cand, DECIMALS_COMPARE)
            if counts_live[r] < 4:
                return cand, r

    # Si no encontramos nada (extremadamente raro), devolvemos el original
    return orig, round(orig, DECIMALS_COMPARE)

# 7) Para cada valor con >4 repeticiones, dejar 4 y modificar el resto
for val in dupe_vals:
    mask_group = rounded.eq(val)
    idxs = df.index[mask_group].tolist()
    if len(idxs) <= 4:
        continue
    # Deja los 4 primeros
    keep = idxs[:4]
    modify = idxs[4:]

    for i in modify:
        orig = float(df.at[i, RI_COL])
        new_val, new_r = propose_new_value(orig)

        # Actualiza DataFrame y conteos
        df.at[i, RI_COL] = new_val
        counts_live[val] -= 1
        counts_live[new_r] += 1

# 8) Verificación final
rounded_after = df[RI_COL].round(DECIMALS_COMPARE)
counts_after = Counter(rounded_after.dropna())
bad_after = {v: c for v, c in counts_after.items() if c > 4}

print(f"Valores con repeticiones > 4 (después): {len(bad_after)}")
if bad_after:
    print("Aún repetidos >4 (revisar manualmente):", list(bad_after.items())[:10])

# 9) Guardar con coma decimal y 4 decimales
df.to_csv(DST, sep=";", decimal=",", index=False, float_format="%.4f")
print(f"Archivo guardado: {DST}")
