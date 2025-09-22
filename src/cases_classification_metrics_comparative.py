import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def eval_case(path, label):
    df = pd.read_csv(path)
    y_true = df["Clase Real"].astype(str)
    y_pred = df["Clase Predicha"].astype(str)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return {"Caso": label, "Accuracy": acc, "F1-macro": f1m}

# Archivos esperados
files = {
    "C1": "data_C1.csv",
    "C2": "data_C2.csv",
    "C3": "data.csv"
}

results = []
for label, path in files.items():
    try:
        results.append(eval_case(path, label))
    except FileNotFoundError:
        print(f" Archivo {path} no encontrado, salta este caso.")

table = pd.DataFrame(results)
print("\n=== Comparación entre casos (Ablation) ===\n")
print(table.to_string(index=False, float_format="%.4f"))
