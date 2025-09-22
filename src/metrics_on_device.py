import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Cargar archivo
df = pd.read_csv("data.csv")

# 1. Exactitud por bloque
# Suponemos que cada bloque está en el mismo orden que el plan de captura (Mov00, Mov01, etc.)
# Aquí un truco: cortar por cambios en "Clase Real"
blocks = []
current_label = df["Clase Real"].iloc[0]
start_idx = 0
for i, label in enumerate(df["Clase Real"]):
    if label != current_label:
        block = df.iloc[start_idx:i]
        acc = (block["Clase Real"] == block["Clase Predicha"]).mean()
        blocks.append(acc)
        current_label = label
        start_idx = i
# último bloque
block = df.iloc[start_idx:]
acc = (block["Clase Real"] == block["Clase Predicha"]).mean()
blocks.append(acc)

mean_acc = np.mean(blocks)
std_acc = np.std(blocks)

print(f"Exactitud por bloque (promedio ± DE): {mean_acc:.3f} ± {std_acc:.3f}")

# 2. Confiabilidad media
mean_conf = df["Resultado Inferencia"].mean()
print(f"Confiabilidad media: {mean_conf:.3f}")

# 3. Falsos positivos/negativos por clase
y_true = df["Clase Real"]
y_pred = df["Clase Predicha"]

cm = confusion_matrix(y_true, y_pred, labels=df["Clase Real"].unique())
report = classification_report(y_true, y_pred)

print("Matriz de confusión:\n", cm)
print("\nReporte de clasificación:\n", report)
