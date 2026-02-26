# CuantumWiki - Tone Detection AI Network

Detector de tono de texto (Neutral, Positivo, Urgente, Molesto, Formal) con procesamiento de inferencia verdaderamente distribuido utilizando TinyML sobre microcontroladores ESP32.

Este proyecto implementa una red neuronal profunda con *Embedding*, *Global Average Pooling* y capazas *Dense* directamente en C puro (sin dependencias como TFLite Micro), distribuida a través de un bus I2C.

## 🚀 Arquitectura del Sistema

El proyecto tiene tres componentes principales que funcionan en cadena:

1. **Frontend Web (Tokenizador UI)**
   - Extrae el vocabulario, tokeniza el texto en el navegador y se comunica con el hardware vía **Web Serial API** usando JSONL.
   - Hosted directamente en [GitHub Pages](./docs/index.html). La wiki técnica está en [wiki.html](./docs/wiki.html).

2. **Coordinador Master (ESP32-S3)**
   - Actúa como puente entre la Web (Serial) y los nodos esclavos (I2C).
   - Encola peticiones, asigna trabajadores usando Round-Robin o selección directa.
   - Cuenta con una pantalla OLED para monitoreo, RTC, guardado de logs en tarjeta SD y sensor ambiental BME280.
   - *Directorio:* `firmware/coordinator_s3/`

3. **Workers de Inferencia (ESP32-C3 x4)**
   - Nodos esclavos I2C que ejecutan el modelo de Machine Learning (`model_weights.h`).
   - El modelo (1.3MB de código fuente que compila a ~450KB físicos de Flash) predice el tono basándose en los tokens y devuelve el resultado con su porcentaje de confianza al master.
   - *Directorio:* `firmware/worker_c3/`

---

## 🛠️ Instrucciones de Instalación y Uso

### 1. Despliegue de la Web UI
La web es 100% estática (HTML/CSS/JS). Para usarla, simplemente abre `docs/index.html` en un navegador compatible con **Web Serial API** (Chrome, Edge, Opera).
Para pruebas locales tipo servidor, puedes ejecutar:
```bash
cd docs
python -m http.server 8080
```
Y abrir `http://localhost:8080/`.

### 2. Flasheo del Firmware
El proyecto usa el framework Arduino. Necesitarás instalar las placas ESP32 en el Board Manager.

**A. Coordinador ESP32-S3:**
- Abre `firmware/coordinator_s3/coordinator_s3.ino`.
- Asegúrate de tener las librerías necesarias (*Adafruit BME280, Adafruit NeoPixel, U8g2, ArduinoJson, RTClib*).
- Selecciona tu placa ESP32-S3 y súbelo.

**B. Workers ESP32-C3 (¡CRÍTICO!):**
- Abre `firmware/worker_c3/worker_c3.ino`.
- **🚨 PRECAUCIÓN DE DIRECCIONES I2C:** Cada uno de los 4 nodos *DEBE* tener una dirección I2C única. Antes de compilar y subir el código a cada ESP32-C3 físico, debes cambiar esta línea:
  ```cpp
  // CAMBIAR ESTO POR NODO: 0x10, 0x11, 0x12, 0x13
  #define SLAVE_ADDRESS 0x10
  ```
- *Hardware Note:* Si conectas 4 esclavos al bus I2C del S3, es altamente recomendable usar **resistencias Pull-Up externas de 4.7kΩ** en las líneas SDA y SCL hacia 3.3V para garantizar la sincronización a 100kHz.

### 3. Pipeline de Entrenamiento TinyML
Si deseas re-entrenar el modelo neuronal con nuevos datos:
1. Instala Python y dependencias: `pip install tensorflow numpy pandas datasets regex`.
2. Ejecuta el pipeline:
   ```bash
   python firmware/train_tone_model.py
   ```
3. El script combinará datasets sintéticos con reales descargados de HuggingFace, balanceará las clases, entrenará la red con *Data Augmentation* y exportará dos archivos vitales:
   - `model/model_weights.h` (Copiado automáticamente a `worker_c3/`)
   - `model/vocab_web.js` (Copiado automáticamente a `docs/`)

Re-compila los C3 y recarga la web para aplicar los cambios del modelo.

---

## 📊 Especificaciones del Modelo
- **Vocabulario:** 3500 tokens (límite de memoria del C3).
- **Dimensiones:** Embedding(32) → GlobalAvgPool → Dense(64) → Dense(5).
- **Tamaño:** ~1.3MB en archivo header (`.h`), ocupando ~55% de la Flash en el binario compilado.
- **Precisión:** ~100% en validación con Datasets Mixtos limpios.
- **Clases:** `neutral`, `positivo`, `urgente`, `molesto`, `formal`.

---

## 📜 Licencia & Descargo de Responsabilidad

**Autor:** Alejandro Rebolledo ([arebolledo@udd.cl](mailto:arebolledo@udd.cl))

El código base se distribuye bajo la licencia **MIT** (ver archivo `LICENSE`).

**Disclaimer / Descargo de Responsabilidad:**  
Este software se proporciona "tal cual" (*AS IS*), sin ningún tipo de garantía expresa o implícita, incluyendo, pero no limitándose a, las garantías implícitas de comerciabilidad, idoneidad para un propósito particular y no infracción.  
En ningún caso el autor será responsable por cualquier reclamación, daño directo, indirecto, incidental, especial, ejemplar o consecuente (incluyendo, pero no limitándose a, la pérdida de uso, datos o beneficios; o interrupción del negocio) cualquiera que sea la causa y bajo cualquier teoría de responsabilidad, ya sea por contrato, responsabilidad estricta o agravio, que surja del uso de este software.

El uso, implementación o modificación de este sistema de IA y arquitectura distribuida para fines críticos, comerciales o de producción es responsabilidad entera y exclusiva del usuario final.
