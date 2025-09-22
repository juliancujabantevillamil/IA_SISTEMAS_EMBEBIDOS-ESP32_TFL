import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Lee tu archivo on-device (C3)
df = pd.read_csv("data.csv")

# Asume estas columnas exactamente:
# "Clase Real" y "Clase Predicha"
y_true = df["Clase Real"].astype(str)
y_pred = df["Clase Predicha"].astype(str)

# Métricas principales
acc = accuracy_score(y_true, y_pred)
f1m = f1_score(y_true, y_pred, average="macro")

print(f"Accuracy (C3): {acc:.4f}")
print(f"F1-macro (C3): {f1m:.4f}\n")

# (Opcional) Reporte por clase para tu tabla de observaciones
print(classification_report(y_true, y_pred, digits=4))
