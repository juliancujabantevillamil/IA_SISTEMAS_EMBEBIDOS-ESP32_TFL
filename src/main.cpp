// ==========================
// main.cpp  (ÚNICO archivo, dual-mode con salida a 100 Hz)
// - Si existen model_data.h/feat_norm.h => MODO CLASIFICACIÓN (TFLite Micro)
// - Si NO existen => MODO CAPTURA (7 campos) - NO requiere TFLM
//
// Serial (SIEMPRE a 100 Hz):
// - Modo CAPTURA:  ax;ay;az;gx;gy;gz;temp
// - Modo CLASIF :  ax;ay;az;gx;gy;gz;temp;NA;Mov0X;p0;p1;p2;p3;p4
//
// Notas:
// * La predicción (TFLM) se recalcula con ventanas de 0.5 s y hop de 0.25 s (4 Hz por defecto),
//   pero se "cachea" y se imprime en CADA muestra a 100 Hz. Así obtienes 1 predicción por fila.
// * Si necesitas que la predicción se actualice más rápido, baja HOP (p.ej., HOP=5 → 20 Hz; HOP=1 → 100 Hz).
// ==========================

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>   // sqrtf, lroundf

#define BAUD_RATE 115200 // Si tu serial se satura a 100 Hz con 14 campos, sube a 230400

// ---- (Opcional) OLED ----
#define USE_OLED 1
#if USE_OLED
  #include <Adafruit_SSD1306.h>
  Adafruit_SSD1306 display(128, 64, &Wire);
#endif

// --------- Sensado / RTOS ---------
Adafruit_MPU6050 mpu;
QueueHandle_t q_samples;   // muestras crudas @100Hz
QueueHandle_t q_preds;     // predicciones (~4-20 Hz, según HOP)
SemaphoreHandle_t i2cMutex;

struct Sample {
  float ax, ay, az, gx, gy, gz, temp;
  uint32_t t_ms;
};

struct Prediction {
  int   pred_class;      // 0..4
  float probs[5];
  uint32_t t_ms;
};

// --------- Frecuencia y ventana ---------
static const int   SAMPLE_RATE_HZ = 100;
static const float DT_MS = 1000.0f / SAMPLE_RATE_HZ;
static const int   WIN = 50;      // 0.5 s @100Hz
static const int   HOP = 25;      // 0.25 s (50% solape) => ~4 Hz de actualización de predicción

// Buffer circular para la ventana
static Sample ringBuf[WIN];
static int    ringHead = 0;
static int    nFilled  = 0;

// ==========================================================
// Detección de modelo/normalizador (activar modo TFLM)
// ==========================================================
#if __has_include("model_data.h") && __has_include("feat_norm.h") && __has_include("tensorflow/lite/micro/micro_interpreter.h")
  #define HAVE_TFLM 1
  #include "model_data.h"   // g_model, g_model_len
  #include "feat_norm.h"    // FEAT_MEAN[], FEAT_SCALE[], N_FEAT

  // ---- TensorFlow Lite Micro ----
  #include "tensorflow/lite/micro/micro_interpreter.h"
  #include "tensorflow/lite/micro/all_ops_resolver.h"
  #include "tensorflow/lite/micro/micro_error_reporter.h"  // necesario por MicroErrorReporter
  #include "tensorflow/lite/schema/schema_generated.h"
  // NO incluyas "tensorflow/lite/version.h"

  #ifndef N_FEAT
    #define N_FEAT 39
  #endif
  #define N_CLASS 5

  constexpr int kTensorArenaSize = 40 * 1024; // sube si AllocateTensors falla
  static uint8_t tensor_arena[kTensorArenaSize];

  static tflite::MicroInterpreter* interpreter = nullptr;
  static TfLiteTensor* input_tensor  = nullptr;
  static TfLiteTensor* output_tensor = nullptr;
  static bool model_is_int8 = true;

  // Predicción cacheada para imprimir a 100 Hz
  Prediction g_lastPred = {0,{0,0,0,0,0},0};
  volatile bool g_hasPred = false;
#else
  #define HAVE_TFLM 0
  // En modo captura no usamos TFLM ni 39 features; solo emitimos 7 campos.
#endif

// ==========================================================
// Utilidades de features (solo cuando hay TFLM)
// ==========================================================
#if HAVE_TFLM
static inline float mean_arr(const float *x, int n){ double s=0; for(int i=0;i<n;i++) s+=x[i]; return (float)(s/n); }
static inline float std_arr (const float *x, int n, float m){ double s=0; for(int i=0;i<n;i++){ double d=x[i]-m; s+=d*d; } return (float)sqrt(s/n); }
static inline float max_arr (const float *x, int n){ float v=x[0]; for(int i=1;i<n;i++) if (x[i]>v) v=x[i]; return v; }
static inline float min_arr (const float *x, int n){ float v=x[0]; for(int i=1;i<n;i++) if (x[i]<v) v=x[i]; return v; }
static inline float rms_arr (const float *x, int n){ double s=0; for(int i=0;i<n;i++) s+=x[i]*x[i]; return (float)sqrt(s/n); }

// 39 features = 7 señales×(mean,std,min,max,rms) + 4 de magnitudes (acc,gyro: mean,std)
static void extract_features(float feat[N_FEAT]) {
  float ax[WIN], ay[WIN], az[WIN], gx[WIN], gy[WIN], gz[WIN], tp[WIN];

  // reconstruir ventana en orden temporal
  for (int i=0; i<WIN; ++i){
    int idx = (ringHead - 1 - i + WIN) % WIN;
    ax[WIN-1-i] = ringBuf[idx].ax;
    ay[WIN-1-i] = ringBuf[idx].ay;
    az[WIN-1-i] = ringBuf[idx].az;
    gx[WIN-1-i] = ringBuf[idx].gx;
    gy[WIN-1-i] = ringBuf[idx].gy;
    gz[WIN-1-i] = ringBuf[idx].gz;
    tp[WIN-1-i] = ringBuf[idx].temp;
  }

  int k = 0;
  #define PUSH5(sig) { \
    float m = mean_arr(sig, WIN); \
    feat[k++] = m; \
    feat[k++] = std_arr(sig, WIN, m); \
    feat[k++] = min_arr(sig, WIN); \
    feat[k++] = max_arr(sig, WIN); \
    feat[k++] = rms_arr(sig, WIN); \
  }

  // 7×5 = 35
  PUSH5(ax); PUSH5(ay); PUSH5(az);
  PUSH5(gx); PUSH5(gy); PUSH5(gz);
  PUSH5(tp);

  // +4 magnitudes
  float acc_mag[WIN], gyro_mag[WIN];
  for (int i=0;i<WIN;i++){
    acc_mag[i]  = sqrtf(ax[i]*ax[i] + ay[i]*ay[i] + az[i]*az[i]);
    gyro_mag[i] = sqrtf(gx[i]*gx[i] + gy[i]*gy[i] + gz[i]*gz[i]);
  }
  float am_m = mean_arr(acc_mag, WIN);
  float am_s = std_arr (acc_mag, WIN, am_m);
  float gm_m = mean_arr(gyro_mag, WIN);
  float gm_s = std_arr (gyro_mag, WIN, gm_m);
  feat[k++] = am_m; feat[k++] = am_s; feat[k++] = gm_m; feat[k++] = gm_s;

  // z-score igual a StandardScaler (mean_, scale_)
  for(int i=0;i<N_FEAT;i++){
    feat[i] = (feat[i] - FEAT_MEAN[i]) / (FEAT_SCALE[i] + 1e-6f);
  }
}

// TFLM setup
static void tflm_setup(){
  const tflite::Model* model = tflite::GetModel(g_model);

  // Chequeo de versión solo si existe la macro
  #ifdef TFLITE_SCHEMA_VERSION
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Warning: TFLite schema version mismatch (continuando)");
  }
  #endif

  static tflite::AllOpsResolver resolver;

  // ---- API nueva: requiere ErrorReporter* y punteros opcionales ----
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,            // uint8_t*
      kTensorArenaSize,        // size_t
      error_reporter,          // ErrorReporter*
      /*resource_variables=*/nullptr,
      /*profiler=*/nullptr);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors FAILED (sube kTensorArenaSize a 60-80 KB)");
    while(true) delay(1000);
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);
  model_is_int8 = (input_tensor->type == kTfLiteInt8);
  Serial.printf("TFLM listo. in_type=%d, out_type=%d\n", input_tensor->type, output_tensor->type);
}

static void classify_tfl(const float feat_normed[N_FEAT], float probs_out[5], int &argmax){
  // entrada
  if (model_is_int8) {
    const float s = input_tensor->params.scale;
    const int   z = input_tensor->params.zero_point;
    for (int i=0;i<N_FEAT;i++){
      int q = (int)lroundf(feat_normed[i] / s) + z;
      if (q < -128) q = -128; if (q > 127) q = 127;
      input_tensor->data.int8[i] = (int8_t)q;
    }
  } else {
    for (int i=0;i<N_FEAT;i++) input_tensor->data.f[i] = feat_normed[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke FAILED");
    for(int c=0;c<5;c++) probs_out[c]=0.f;
    argmax=0; return;
  }

  // salida → probs
  if (output_tensor->type == kTfLiteInt8){
    const float s = output_tensor->params.scale;
    const int   z = output_tensor->params.zero_point;
    float sum=0;
    for (int c=0;c<5;c++){
      float p = s * (output_tensor->data.int8[c] - z);
      if (p < 0) p = 0; if (p > 1) p = 1;
      probs_out[c] = p; sum += p;
    }
    if (sum>0){ for(int c=0;c<5;c++) probs_out[c] /= sum; }
  } else {
    float sum=0;
    for (int c=0;c<5;c++){ probs_out[c] = output_tensor->data.f[c]; sum += probs_out[c]; }
    if (sum>0){ for (int c=0;c<5;c++) probs_out[c] /= sum; }
  }

  argmax = 0; float best = probs_out[0];
  for (int c=1;c<5;c++) if (probs_out[c]>best){ best=probs_out[c]; argmax=c; }
}
#endif // HAVE_TFLM

// ==========================================================
// Tareas
// ==========================================================
static void TaskAcquire(void *){
  TickType_t last = xTaskGetTickCount();
  const TickType_t period = pdMS_TO_TICKS(DT_MS);

  for(;;){
    sensors_event_t a, g, temp;
    xSemaphoreTake(i2cMutex, portMAX_DELAY);
    mpu.getEvent(&a,&g,&temp);
    xSemaphoreGive(i2cMutex);

    Sample s;
    s.ax=a.acceleration.x; s.ay=a.acceleration.y; s.az=a.acceleration.z;
    s.gx=g.gyro.x;         s.gy=g.gyro.y;         s.gz=g.gyro.z;
    s.temp=temp.temperature;
    s.t_ms = millis();

    // Empuja a la cola para ventaneo/clasificación
    xQueueSend(q_samples, &s, 0);

    // ===== Salida por Serial a 100 Hz =====
#if HAVE_TFLM
    // MODO CLASIFICACIÓN: 14 campos usando la última predicción disponible
    Serial.print(s.ax);   Serial.print(';');
    Serial.print(s.ay);   Serial.print(';');
    Serial.print(s.az);   Serial.print(';');
    Serial.print(s.gx);   Serial.print(';');
    Serial.print(s.gy);   Serial.print(';');
    Serial.print(s.gz);   Serial.print(';');
    Serial.print(s.temp); Serial.print(';');
    Serial.print("NA;");  // Clase Real la agrega el PC por bloque

    int cls = g_hasPred ? g_lastPred.pred_class : 0;
    Serial.print(String("Mov0") + String(cls) + ";");
    for (int c=0;c<5;c++){
      float pv = g_hasPred ? g_lastPred.probs[c] : 0.0f;
      Serial.print(pv, 6);
      Serial.print(c==4? '\n' : ';');
    }
#else
    // MODO CAPTURA: 7 campos
    Serial.print(s.ax);   Serial.print(';');
    Serial.print(s.ay);   Serial.print(';');
    Serial.print(s.az);   Serial.print(';');
    Serial.print(s.gx);   Serial.print(';');
    Serial.print(s.gy);   Serial.print(';');
    Serial.print(s.gz);   Serial.print(';');
    Serial.println(s.temp);
#endif

    vTaskDelayUntil(&last, period);
  }
}

static void TaskClassify(void *){
  int hopCnt=0;
  for(;;){
    Sample s;
    if (xQueueReceive(q_samples, &s, portMAX_DELAY)==pdTRUE){
      // actualizar ventana
      ringBuf[ringHead]=s;
      ringHead=(ringHead+1)%WIN;
      nFilled = nFilled < WIN ? nFilled+1 : WIN;

#if HAVE_TFLM
      if (nFilled==WIN){
        hopCnt++;
        if (hopCnt >= HOP){
          hopCnt=0;

          // ---------- Cálculo de predicción (según ventana) ----------
          float feat[N_FEAT];
          extract_features(feat);

          Prediction p;
          classify_tfl(feat, p.probs, p.pred_class);
          p.t_ms = s.t_ms;
          // Guardar en cola (para OLED) y en cache global (para impresión a 100 Hz)
          if (q_preds) xQueueSend(q_preds, &p, 0);
          g_lastPred = p;
          g_hasPred  = true;
        }
      }
#else
      // En modo captura no hay clasificación
      (void)hopCnt;
#endif
    }
  }
}

static void TaskDisplay(void *){
#if USE_OLED
  for(;;){
  #if HAVE_TFLM
    Prediction p;
    if (xQueueReceive(q_preds, &p, portMAX_DELAY)==pdTRUE){
      xSemaphoreTake(i2cMutex, portMAX_DELAY);
      display.clearDisplay();
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0,0);
      display.println("Movimiento:");
      display.setTextSize(2);
      display.printf("Mov0%d\n", p.pred_class);
      display.setTextSize(1);
      for (int c=0;c<5;c++){
        display.printf("P%02d: %.2f\n", c, p.probs[c]);
      }
      display.display();
      xSemaphoreGive(i2cMutex);
    }
  #else
    // En modo captura no hay predicciones: muestra estado básico
    xSemaphoreTake(i2cMutex, portMAX_DELAY);
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0,0);
    display.println("Modo: CAPTURA");
    display.println("@100Hz emitiendo 7 campos");
    display.display();
    xSemaphoreGive(i2cMutex);
    vTaskDelay(pdMS_TO_TICKS(500));
  #endif
  }
#else
  vTaskDelete(NULL);
#endif
}

// ==========================================================
// setup / loop
// ==========================================================
void setup(){
  Serial.begin(BAUD_RATE);
  Wire.begin(); // ESP32 SDA=21, SCL=22

  i2cMutex = xSemaphoreCreateMutex();

  xSemaphoreTake(i2cMutex, portMAX_DELAY);
  if (!mpu.begin()) { Serial.println("No se detecta MPU6050"); while(1) delay(10); }
  // Rangos/filtro (ajustables)
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);

#if USE_OLED
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)){
    Serial.println("SSD1306 no encontrado (continuando sin OLED)");
  } else {
    display.clearDisplay(); display.display();
  }
#endif
  xSemaphoreGive(i2cMutex);

#if HAVE_TFLM
  // Inicializa TFLM solo si el modelo está presente
  tflm_setup();
  q_preds   = xQueueCreate(16, sizeof(Prediction));
#else
  q_preds   = NULL; // no se usa en captura
#endif
  q_samples = xQueueCreate(128, sizeof(Sample));

  // Tareas (core 1)
  xTaskCreatePinnedToCore(TaskAcquire,  "acq",  4096, NULL, 2, NULL, 1);
  xTaskCreatePinnedToCore(TaskClassify, "clf",  8192, NULL, 2, NULL, 1);
  xTaskCreatePinnedToCore(TaskDisplay,  "disp", 4096, NULL, 1, NULL, 1);

#if HAVE_TFLM
  Serial.println("Modo: CLASIFICACIÓN (TFLM activo)");
#else
  Serial.println("Modo: CAPTURA (sin modelo)");
#endif
}

void loop(){ vTaskDelay(portMAX_DELAY); }

