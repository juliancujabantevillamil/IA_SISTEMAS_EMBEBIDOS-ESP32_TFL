import pandas as pd
df = pd.read_csv("data.csv")
print(df["Clase Predicha"].value_counts(dropna=False))
print("\nEjemplos con Clase Predicha vacía:\n", df[df["Clase Predicha"].isna() | (df["Clase Predicha"]=="")].head())
