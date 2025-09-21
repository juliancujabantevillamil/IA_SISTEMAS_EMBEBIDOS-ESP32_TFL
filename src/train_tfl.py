# train_tfl.py (versión revisada)
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

CLASSES = ["Mov00","Mov01","Mov02","Mov03","Mov04"]

def window_features(df, fs=100, win_s=0.5, hop_s=0.25):
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
            # mean, std(ddof=0), min, max, rms
            return [float(x.mean()), float(x.std(ddof=0)), float(x.min()), float(x.max()),
                    float(np.sqrt((x**2).mean()))]
        row=[]
        for s in [ax,ay,az,gx,gy,gz,temp]:
            m,sd,mi,ma,r = feats(s)
            row += [m,sd,mi,ma,r]   # 7*5 = 35
        acc_mag = np.sqrt(ax**2+ay**2+az**2)
        gyro_mag= np.sqrt(gx**2+gy**2+gz**2)
        row += [float(acc_mag.mean()), float(acc_mag.std(ddof=0)),
                float(gyro_mag.mean()), float(gyro_mag.std(ddof=0))]  # +4 = 39
        X.append(row)
        y.append(L[i+win//2])  # etiqueta del centro de la ventana
    X = np.array(X, np.float32)
    y = np.array([CLASSES.index(lbl) for lbl in y], np.int64)
    return X, y

# 1) Cargar RAW
df = pd.read_csv("data_raw.csv")
assert set(["ax","ay","az","gx","gy","gz","temp","label"]).issubset(df.columns), "Faltan columnas RAW"

# 2) Ventaneo + features
X, y = window_features(df, fs=100, win_s=0.5, hop_s=0.05)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3) Estandarización
scaler = StandardScaler().fit(Xtr)
Xtr = scaler.transform(Xtr).astype(np.float32)
Xte = scaler.transform(Xte).astype(np.float32)
np.save("feat_mean.npy", scaler.mean_.astype(np.float32))
np.save("feat_std.npy",  scaler.scale_.astype(np.float32))  # OJO: scale_ = std (poblacional)

# 4) Modelo Keras
inputs = tf.keras.Input(shape=(Xtr.shape[1],), dtype=tf.float32, name="features_39")
x = tf.keras.layers.Dense(32, activation="relu")(inputs)
x = tf.keras.layers.Dense(16, activation="relu")(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

cb = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
hist = model.fit(Xtr, ytr, epochs=80, batch_size=64, validation_data=(Xte, yte), callbacks=cb, verbose=2)

# 5) Evaluación
te_loss, te_acc = model.evaluate(Xte, yte, verbose=0)
yprob = model.predict(Xte, verbose=0)
ypred = yprob.argmax(axis=1)
print(f"Test acc: {te_acc:.3f}")
print("Confusion matrix:\n", confusion_matrix(yte, ypred))
print("Report:\n", classification_report(yte, ypred, target_names=CLASSES))

# 6) Exportar TFLite float32 (siempre)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfl_f32 = converter.convert()
open("model_f32.tflite","wb").write(tfl_f32)

# 7) Intentar INT8 full (ideal para ESP32)
def rep_data():
    for i in range(0, len(Xtr), max(1, len(Xtr)//2000 or 1)):
        yield [Xtr[i:i+1].astype(np.float32)]

ok_int8 = False
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    tfl_int8 = converter.convert()
    open("model_int8.tflite","wb").write(tfl_int8)
    ok_int8 = True
    print("OK -> model_int8.tflite")
except Exception as e:
    print("INT8 full FAILED (intentaremos dynamic-range):", e)

# 8) Si falló INT8 full, guardar dynamic-range (pesos int8, I/O float32)
if not ok_int8:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # sin representative_dataset => dynamic range
    tfl_dr = converter.convert()
    open("model_dr.tflite","wb").write(tfl_dr)
    print("OK -> model_dr.tflite (dynamic-range)")

print("Listo: siempre tienes model_f32.tflite; y además",
      "model_int8.tflite" if ok_int8 else "model_dr.tflite")
