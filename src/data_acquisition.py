# data_acquisition.py
# Modo MANUAL (por defecto):  python data_acquisition.py COM6 --mode raw
# Modo AUTOMÁTICO:            python data_acquisition.py COM6 --mode raw --auto
# Modo ENTREGABLE (con TFLM): python data_acquisition.py COM6 --mode entregable [--auto]
import sys, time, random, argparse
import pandas as pd
import serial
from serial.tools import list_ports

CLASSES = ["Mov00","Mov01","Mov02","Mov03","Mov04"]

HEADER_ENT = [
    "Feature1","Feature2","Feature3","Feature4","Feature5","Feature6","Feature7",
    "Clase Real","Clase Predicha","Resultado Inferencia"
]
PLAN_CONTROLADO = [
    ("Mov00", 15), ("Mov01", 15), ("Mov00", 15), ("Mov02", 15),
    ("Mov00", 15), ("Mov03", 15), ("Mov00", 15), ("Mov04", 15), ("Mov00", 15)
]

def pick_port(prefer=None):
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No hay puertos seriales disponibles")
    if prefer and any(prefer == p.device for p in ports):
        return prefer
    print("Puertos detectados:")
    for i, p in enumerate(ports, 1):
        print(f"  {i}) {p.device} - {p.description}")
    return ports[0].device

def open_serial(port, baud=115200, timeout=1):
    ser = serial.Serial(port, baud, timeout=timeout)
    time.sleep(0.3)
    return ser

def parse_line(line):
    """
    Desde ESP32:
      - 7 campos:  ax..temp
      - 14 campos: ax..temp; NA; Mov0X; p0;..;p4
    """
    s = (line or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(';')]
    parts = [p for p in parts if p != ""]
    if len(parts) == 7:
        try:
            feats = list(map(lambda x: float(x.replace(',', '.')), parts))
            return feats, "", [0.0]*5
        except:
            return None
    if len(parts) >= 14:
        try:
            feats = list(map(lambda x: float(x.replace(',', '.')), parts[0:7]))
            pred_label = parts[8]  # "Mov0X"
            probs = list(map(lambda x: float(x.replace(',', '.')), parts[9:14]))
            return feats, pred_label, probs
        except:
            return None
    return None

def countdown(label, seconds):
    for s in range(seconds, 0, -1):
        print(f"→ {label}: iniciando en {s:>2}s...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r", flush=True)

def capture_block(ser, label, seconds, auto=False, cd=3):
    # auto=False por defecto => pide ENTER antes de cada bloque
    if auto:
        countdown(label, cd)
    else:
        input(f"\n→ Ejecuta {label} durante {seconds}s. (Presiona ENTER para iniciar) ")
    start = time.time()
    rows = []
    while time.time() - start < seconds:
        raw = ser.readline().decode('utf-8', errors='ignore')
        parsed = parse_line(raw)
        if not parsed:
            continue
        feats7, pred_label, probs5 = parsed
        rows.append((feats7, label, pred_label, probs5))
    print(f"  capturadas {len(rows)} filas para {label}")
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default=None, help="Puerto serial (p.ej., COM6)")
    ap.add_argument("--auto", "-a", action="store_true",
                    help="Arranca bloques automáticamente (por defecto: manual con ENTER)")
    ap.add_argument("--countdown", "-c", type=int, default=3, help="Segundos de cuenta atrás")
    ap.add_argument("--mode", choices=["raw","entregable"], default="raw",
                    help="raw: data_raw.csv | entregable: data.csv")
    args = ap.parse_args()

    port = pick_port(args.port)
    print("Usando puerto:", port)
    ser = open_serial(port)

    all_rows = []
    try:
        print("\n=== Rutina controlada (15s por bloque) ===")
        for label, secs in PLAN_CONTROLADO:
            all_rows += capture_block(ser, label, secs, auto=args.auto, cd=args.countdown)

        print("\n=== Secuencia aleatoria (6 bloques × 5s) ===")
        for _ in range(6):
            label = random.choice(CLASSES)
            all_rows += capture_block(ser, label, 5, auto=args.auto, cd=args.countdown)

    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")
    finally:
        ser.close()

    if not all_rows:
        print("No se capturaron datos.")
        return

    if args.mode == "raw":
        rows = []
        for feats7, label, pred_label, probs5 in all_rows:
            ax,ay,az,gx,gy,gz,temp = feats7
            rows.append([ax,ay,az,gx,gy,gz,temp,label])
        df = pd.DataFrame(rows, columns=["ax","ay","az","gx","gy","gz","temp","label"])
        df.to_csv("data_raw.csv", index=False)
        print(f"\nGuardado data_raw.csv con {len(df)} filas.")
    else:
        rows = []
        for feats7, label, pred_label, probs5 in all_rows:
            if pred_label and probs5 and len(probs5)==5:
                try:
                    idx = int(pred_label[-1])
                except:
                    idx = max(range(5), key=lambda i: probs5[i])
                res_inf = probs5[idx]
            else:
                res_inf = 0.0
            rows.append(feats7 + [label, pred_label if pred_label else "", res_inf])
        df = pd.DataFrame(rows, columns=HEADER_ENT)
        df.to_csv("data.csv", index=False)
        print(f"\nGuardado data.csv con {len(df)} filas y cabecera del entregable.")

if __name__ == "__main__":
    main()
