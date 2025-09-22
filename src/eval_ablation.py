# eval_ablation.py
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

# ====== Config común ======
CLASSES = ["Mov00","Mov01","Mov02","Mov03","Mov04"]

def window_features(df, fs=100, win_s=0.5, hop_s=0.05):
    """
    MISMO ventaneo y MISMAS 39 características que en train_tfl.py:
    7 señales (ax..temp) × [mean, std, min, max, rms] (=35) + 4 magnitudes (acc/gyro mean,std) (=4) -> 39
    """
    win, hop = int(fs*win_s), int(fs*hop_s)
    X, y = [], []
    A = df[["ax","ay","az","gx","gy","gz"]].to_numpy(np.float32)
    T = df["temp"].to_numpy(np.float32)
    L = df["label"].to_numpy()
    for i in range(0, len(df)-win+1, hop):
        sl = slice(i, i+win)
        ax,ay,az,gx,gy,gz = A[sl].T
        temp = T[sl]
        def feats(x):
            return [float(x.mean()), float(x.std(ddof=0)), float(x.min()), float(x.max()),
                    float(np.sqrt((x**2).mean()))]
        row=[]
        for s in [ax,ay,az,gx,gy,gz,temp]:
            row += feats(s)                         # 7*5 = 35
        acc_mag  = np.sqrt(ax**2+ay**2+az**2)
        gyro_mag = np.sqrt(gx**2+gy**2+gz**2)
        row += [float(acc_mag.mean()), float(acc_mag.std(ddof=0)),
                float(gyro_mag.mean()), float(gyro_mag.std(ddof=0))]  # +4 = 39
        X.append(row)
        y.append(L[i+win//2])
    X = np.array(X, np.float32)
    y = np.array([CLASSES.index(lbl) for lbl in y], np.int64)
    return X, y

def make_model(n_in, n_out):
    inputs = tf.keras.Input(shape=(n_in,), dtype=tf.float32, name="features")
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_out, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def run_case(X, y, scaler=None, feat_idx=None, seed=42):
    if feat_idx is not None:
        X = X[:, feat_idx].copy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    if scaler is not None:
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr).astype(np.float32)
        Xte = scaler.transform(Xte).astype(np.float32)
    model = make_model(Xtr.shape[1], len(CLASSES))
    cb = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
    model.fit(Xtr, ytr, epochs=80, batch_size=64, validation_data=(Xte, yte), callbacks=cb, verbose=0)
    ypred = model.predict(Xte, verbose=0).argmax(axis=1)
    acc = accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")
    return acc, f1m

def main():
    if not Path("data_raw.csv").exists():
        raise SystemExit("No encontré data_raw.csv. Genera el crudo con --mode raw.")
    df = pd.read_csv("data_raw.csv")
    X, y = window_features(df, fs=100, win_s=0.5, hop_s=0.05)

    # ====== Definición de casos (ablation) ======
    # C1: SIN normalizar (pasa directo las 39 features)
    acc1, f11 = run_case(X, y, scaler=None, feat_idx=None)

    # C2: CON normalización StandardScaler (todas las 39)
    acc2, f12 = run_case(X, y, scaler=StandardScaler(), feat_idx=None)

    # C3: CON normalización, pero SIN temperatura ni magnitudes (ablation de información)
    #     - temp aporta 5 stats (posición al final del bloque de 7 señales) -> índices 30:35
    #     - magnitudes acc/gyro (4 últimas) -> índices 35:39
    keep = np.r_[0:30,  # ax..gz (6 señales × 5 stats = 30)
                 # saltamos 30:35 (temp)
                 # saltamos 35:39 (magnitudes)
                 ]
    acc3, f13 = run_case(X, y, scaler=StandardScaler(), feat_idx=keep)

    # ====== Reporte ======
    table = pd.DataFrame({
        "Caso": ["C1: 39 sin normalizar","C2: 39 + StandardScaler","C3: 30 (sin temp/mags) + StandardScaler"],
        "Accuracy": [acc1, acc2, acc3],
        "F1_macro": [f11, f12, f13],
    })
    table.to_csv("ablation_metrics.csv", index=False)
    print("\n=== Comparación entre casos (ablation) ===")
    print(table.to_string(index=False))
    print("\nGuardado: ablation_metrics.csv")

if __name__ == "__main__":
    main()
