# export_to_c.py
import numpy as np

def array_to_c(name, blob: bytes):
    hex_list = ",".join(f"0x{b:02x}" for b in blob)
    return (
        '#include <stddef.h>\n'
        'extern "C" {\n'
        f'extern const unsigned char {name}[] = {{{hex_list}}};\n'
        f'extern const int {name}_len = {len(blob)};\n'
        '}\n'
    )

# 1) Modelo (elige int8 para ESP32)
with open("model_int8.tflite","rb") as f:
    blob = f.read()
open("model_data.cpp","w").write(array_to_c("g_model", blob))
open("model_data.h","w").write('#pragma once\nextern "C" { extern const unsigned char g_model[]; extern const int g_model_len; }\n')

# 2) Normalizador (mean y scale del StandardScaler)
mean = np.load("feat_mean.npy").astype(np.float32)
std  = np.load("feat_std.npy").astype(np.float32)  # scale_ = std
with open("feat_norm.h","w") as f:
    f.write("#pragma once\n#define N_FEAT 39\n")
    f.write("static const float FEAT_MEAN[N_FEAT] = { " + ", ".join(map(lambda x: f"{x:.8f}", mean)) + " };\n")
    f.write("static const float FEAT_SCALE[N_FEAT] = { " + ", ".join(map(lambda x: f"{x:.8f}", std)) + " };\n")
print("Generados: model_data.cc/.h, feat_norm.h")
