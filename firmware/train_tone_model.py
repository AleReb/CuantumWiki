"""
CuantumWiki — TinyML Tone Detection Pipeline
=============================================

Complete pipeline: real dataset download → tokenizer → model → quantize → export weights

Usage:
  python train_tone_model.py

Output:
  model/vocab.json           - Shared vocabulary (web + firmware)
  model/model_weights.h      - Raw float weights for ESP32-C3 (NO TFLite needed)
  model/vocab_web.js         - JS vocabulary for web app
  model/training_report.txt  - Accuracy metrics

Labels: 0=neutral, 1=positivo, 2=urgente, 3=molesto, 4=formal
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================
MAX_LEN = 64          # Max tokens (must match web app and firmware)
VOCAB_SIZE = 3500     # Max vocabulary size (~450KB embed in flash)
EMBED_DIM = 32        # Embedding dimensions
HIDDEN_DIM = 64       # Hidden layer size
NUM_LABELS = 5        # neutral, positivo, urgente, molesto, formal
EPOCHS = 60           # 60 epochs is enough for balanced dataset
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
AUGMENT_FACTOR = 5    # Multiply dataset size by this factor
REAL_DATASET_SAMPLES = 1000  # Cap real samples to match oversampled synthetic

LABELS = ['neutral', 'positivo', 'urgente', 'molesto', 'formal']

OUTPUT_DIR = Path("model")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: DATASET
# ============================================================================

def _download_real_dataset():
    """Download Spanish sentiment data from multiple HuggingFace sources."""
    try:
        from datasets import load_dataset
        all_texts = {'positive': [], 'negative': [], 'neutral': []}

        # ── Source 1: tyqiangz/multilingual-sentiments ──
        print("  [1/3] tyqiangz/multilingual-sentiments...")
        try:
            ds1 = load_dataset("tyqiangz/multilingual-sentiments", "spanish",
                               trust_remote_code=True)
            for split in ds1:
                for row in ds1[split]:
                    text = (row.get('text') or '').strip().lower()
                    # Clean URLs and mentions
                    text = re.sub(r'http\S+|www\.\S+', '', text)
                    text = re.sub(r'@\w+', '', text)
                    if len(text) < 10:
                        continue
                    lbl = row.get('label', -1)
                    if isinstance(lbl, int):
                        lbl = ['positive', 'neutral', 'negative'][lbl]
                    if lbl in all_texts:
                        all_texts[lbl].append(text)
            print(f"    ✓ pos={len(all_texts['positive'])} neu={len(all_texts['neutral'])} neg={len(all_texts['negative'])}")
        except Exception as e:
            print(f"    ✗ {e}")

        # ── Source 2: mteb/SpanishSentimentClassification ──
        print("  [2/3] mteb/SpanishSentimentClassification...")
        try:
            ds2 = load_dataset("mteb/SpanishSentimentClassification",
                               trust_remote_code=True)
            for split in ds2:
                for row in ds2[split]:
                    text = (row.get('text') or '').strip().lower()
                    text = re.sub(r'http\S+|www\.\S+', '', text)
                    text = re.sub(r'@\w+', '', text)
                    if len(text) < 10:
                        continue
                    lbl = row.get('label', '')
                    if isinstance(lbl, int):
                        lbl = ['negative', 'neutral', 'positive'][lbl] if lbl < 3 else 'neutral'
                    lbl = str(lbl).lower().strip()
                    if lbl in ('pos', 'positive', 'positivo', 'p'):
                        all_texts['positive'].append(text)
                    elif lbl in ('neg', 'negative', 'negativo', 'n'):
                        all_texts['negative'].append(text)
                    elif lbl in ('neu', 'neutral', 'neutro', 'none'):
                        all_texts['neutral'].append(text)
            print(f"    ✓ pos={len(all_texts['positive'])} neu={len(all_texts['neutral'])} neg={len(all_texts['negative'])}")
        except Exception as e:
            print(f"    ✗ {e}")

        # ── Source 3: cardiffnlp/tweet_sentiment_multilingual ──
        print("  [3/3] cardiffnlp/tweet_sentiment_multilingual...")
        try:
            ds3 = load_dataset("cardiffnlp/tweet_sentiment_multilingual",
                               "spanish", trust_remote_code=True)
            for split in ds3:
                for row in ds3[split]:
                    text = (row.get('text') or '').strip().lower()
                    text = re.sub(r'http\S+|www\.\S+', '', text)
                    text = re.sub(r'@\w+', '', text)
                    if len(text) < 10:
                        continue
                    lbl = row.get('label', -1)
                    if isinstance(lbl, int):
                        lbl = ['negative', 'neutral', 'positive'][lbl]
                    if lbl in all_texts:
                        all_texts[lbl].append(text)
            print(f"    ✓ pos={len(all_texts['positive'])} neu={len(all_texts['neutral'])} neg={len(all_texts['negative'])}")
        except Exception as e:
            print(f"    ✗ {e}")

        total = sum(len(v) for v in all_texts.values())
        print(f"\n  Total real samples: {total}")
        if total < 50:
            return None
        return all_texts
    except Exception as e:
        print(f"  ⚠ Could not load datasets library: {e}")
        return None


def _synthetic_urgente():
    return [
        "necesito esto urgente para hoy", "ayuda rapido el sistema se cayo",
        "es critico resolver esto ahora mismo", "urgente el cliente esta esperando",
        "necesito una respuesta inmediata", "prioridad maxima resolver ya",
        "el servidor esta caido urgente", "hay que arreglar esto antes del mediodia",
        "emergencia de seguridad actuar rapido", "necesito aprobacion urgente del director",
        "el plazo vence hoy no puede esperar", "alerta critica en el sistema",
        "necesitamos solucion inmediata al problema", "el proyecto se retrasa urgente actuar",
        "por favor rapido que es para ya", "alarma el servicio no responde",
        "urgente revisar los logs del servidor", "necesito que alguien venga ahora",
        "el cliente llama furioso necesito solucion ya", "deadline en dos horas necesito ayuda",
        "critico error en produccion corregir ahora", "hay que desplegar el fix inmediatamente",
        "la base de datos se corrompio urgente", "prioridad uno el sistema esta fallando",
        "necesito los datos para la reunion de ahora", "emergencia cortaron el servicio electrico",
        "urgente el backup no se completo", "hay que escalar esto ya al supervisor",
        "el problema es grave actuar ya", "necesitamos mas gente urgente para el proyecto",
        "rapido que se acaba el tiempo", "el error afecta a todos los usuarios urgente",
        "necesito acceso inmediato al sistema", "urgente el pago no se proceso",
        "es una emergencia llamar al tecnico ya", "critico la api no responde desde hace una hora",
        "urgente necesito de documentos ahora", "inmediato arreglar el formulario",
        "necesitamos resolver esto antes que cierre", "atencion urgente solicitud de soporte critico",
        "alerta maxima vulnerabilidad detectada", "urgente perdida de datos en el servidor",
        "necesito que prioricen mi solicitud ya", "por favor atender esto es extremadamente urgente",
        "critico sistema caido miles de usuarios afectados", "hay que actuar rapido se viene la auditoria",
        "emergencia corte de servicio en produccion", "urgente necesito hablar con el responsable",
        "inmediatamente corregir el error de facturacion", "rapido revisar el sistema esta todo lento",
        "atencion inmediata se requiere accion ahora", "no puede esperar hay que resolver esto hoy",
        "el cliente amenaza con cancelar si no actuamos ya", "prioridad critica el deploy fallo",
        "necesito ayuda ahora mismo no puede esperar", "la situacion es critica actuar de inmediato",
        "el tiempo se agota necesitamos una solucion rapida", "alarma roja sistema comprometido",
        "hay que cerrar esta incidencia antes de las cinco", "no tenemos margen necesito respuesta ya",
        "emergencia sanitaria activar protocolo de inmediato", "urgente el contrato vence mañana",
        "necesito hablar con alguien ahora es una emergencia", "el equipo esta bloqueado necesitamos ayuda",
        "actualizacion critica instalar antes del fin de dia", "hay una fuga de datos actuar inmediatamente",
        "la produccion esta parada necesito solucion rapida", "urgente la certificacion expira esta semana",
        "corregir esto ahora antes de que escale mas", "necesitamos respaldo urgente para el evento de hoy",
        "critico el sistema de pagos esta fuera de linea", "la entrega es hoy y no esta lista urgente",
        "hay que notificar al equipo de guardia inmediatamente", "alerta de seguridad resolver cuanto antes",
        "urgente se cayeron todos los microservicios", "necesito autorizacion ya para proceder",
        "el incidente requiere atencion inmediata del equipo", "faltan minutos para el cierre necesito esto",
        "hay que migrar los datos urgente antes del corte", "emergencia todo el equipo debe conectarse ahora",
        "urgentisimo aprobar el presupuesto hoy sin falta", "necesito que escalen esto al nivel mas alto",
        "no hay tiempo que perder actuar de inmediato", "critico revisar la vulnerabilidad detectada ahora",
        "emergencia el respaldo se borro necesito recuperar", "necesito recursos urgente para este sprint",
        "el servidor de correo cayo urgente restaurar", "hay que parar el deploy hay un bug critico",
        "la plataforma esta inestable urgente estabilizar", "necesito la contraseña de admin es critico",
        "critico la latencia supera los diez segundos", "atencion urgente disco al noventa por ciento",
        # Parrafos largos
        "la situacion actual en el data center internacional requiere intervencion inmediata. multiples sistemas de refrigeracion han fallado simultaneamente provocando un aumento critico de temperatura que amenaza la integridad fisica de los servidores principales. si no actuamos en los proximos quince minutos perderemos toda la informacion",
        "el ataque cibernetico esta escalando a un ritmo alarmante y ha comprometido ya tres bases de datos de clientes corporativos. el equipo rojo confirma exfiltracion masiva en curso. necesitamos apagar los firewalls externos aislar la red interna y convocar al gabinete de crisis inmediatamente antes de que se propague",
        "este es un aviso de maxima prioridad ordenado por la direccion general. todos los empleados deben evacuar las instalaciones de la planta industrial debido a una fuga de gas toxico reportada en el sector tres. abandonen sus puestos de trabajo inmediatamente sin recoger pertenencias personales"
    ]


def _synthetic_formal():
    return [
        "estimado señor le informo que su solicitud fue recibida",
        "por medio de la presente me dirijo a usted",
        "adjunto remito el documento solicitado",
        "atentamente le saluda el departamento de recursos",
        "cordialmente le comunicamos la decision tomada",
        "estimada directora le hago llegar el informe",
        "de mi mayor consideracion paso a informarle",
        "me permito solicitar su valiosa colaboracion",
        "quedo a su disposicion para cualquier consulta",
        "sin otro particular le saluda muy atentamente",
        "es grato dirigirme a usted para comunicarle",
        "solicito formalmente la revision del expediente",
        "por la presente notifico la resolucion adoptada",
        "tengo el agrado de comunicarle que fue aprobado",
        "le escribo en relacion al asunto mencionado",
        "agradecere su amable respuesta a la brevedad",
        "hago de su conocimiento lo siguiente",
        "sirva la presente para hacer constar que",
        "en respuesta a su comunicacion del dia",
        "le informo que hemos procedido segun lo acordado",
        "me dirijo a usted respetuosamente para solicitar",
        "mediante el presente documento certifico que",
        "se le notifica que su tramite ha sido procesado",
        "queda a disposicion de la autoridad competente",
        "la presente tiene por objeto informarle sobre",
        "se adjunta la documentacion requerida para su revision",
        "ruego tenga a bien considerar mi solicitud",
        "hago referencia a la conversacion sostenida",
        "le comunico que el plazo ha sido extendido",
        "solicito su autorizacion para proceder",
        "quedo en espera de su amable respuesta",
        "reciba un cordial saludo de parte del consejo",
        "por intermedio del presente me permito consultar",
        "le expreso mi mas alta consideracion y respeto",
        "de acuerdo a la normativa vigente se establece que",
        "conforme a lo dispuesto en el articulo tercero",
        "la junta directiva ha resuelto lo siguiente",
        "le invito a participar de la sesion ordinaria",
        "el comite evaluador ha emitido su dictamen",
        "se le hace entrega del certificado correspondiente",
        "la direccion general comunica oficialmente que",
        "con el debido respeto me permito sugerir",
        "le agradezco de antemano su pronta atencion",
        "sin mas que agregar me despido atentamente",
        "esperando su favorable respuesta quedo de usted",
        "le remito la documentacion adjunta para su analisis",
        "me permito elevar a su consideracion la propuesta",
        "ruego se sirva dar tramite a la presente solicitud",
        "ante la instancia correspondiente presento este recurso",
        "cumpliendo con lo establecido en las disposiciones legales",
        "dejamos constancia de que se ha cumplido el procedimiento",
        "el suscrito certifica que la informacion es veraz",
        "en atencion a su solicitud procedemos a informarle",
        "hacemos llegar nuestra posicion institucional al respecto",
        "la gerencia ha determinado proceder conforme a derecho",
        "nos ponemos a su entera disposicion para lo que requiera",
        "por este conducto se notifica la resolucion administrativa",
        "se comunica a los interesados que el plazo ha sido fijado",
        "tenemos a bien informarle que su expediente fue aprobado",
        "mediante la presente manifestamos nuestra conformidad",
        "con fundamento en la legislacion aplicable resolvemos",
        "se exhorta a las partes a cumplir con lo acordado",
        "remitimos el acta de la sesion celebrada el dia",
        "la comision dictaminadora ha emitido su resolucion final",
        "con el objeto de formalizar el acuerdo se suscribe",
        "nos dirigimos a usted en calidad de representantes legales",
        "a quien corresponda se extiende la presente constancia",
        "previo analisis de los antecedentes se determina que",
        "nos referimos a su atenta comunicacion del pasado",
        "en cumplimiento de nuestras obligaciones informamos que",
        "queda debidamente notificado de la resolucion dictada",
        "la presente sirve como acuse de recibo de su documentacion",
        "se ha procedido conforme al reglamento interno vigente",
        "el directorio en pleno ha aprobado la mocion presentada",
        "respetuosamente sometemos a su consideracion el proyecto",
        "mediante acuerdo del consejo se aprobo por unanimidad",
        "se certifica para los efectos legales correspondientes",
        "hacemos de su conocimiento que el proceso ha concluido",
        "el tribunal ha resuelto en favor de la parte demandante",
        "acusamos recibo de su escrito de fecha indicada",
        "nos permitimos informar que la auditoria ha concluido",
        "el departamento juridico ha revisado y aprobado el contrato",
        "nos complace comunicar que su postulacion fue aceptada",
        "se convoca a sesion extraordinaria para tratar el asunto",
        "la secretaria general certifica la autenticidad del documento",
        "en ejercicio de las facultades conferidas se resuelve",
        "el consejo de administracion ha sesionado extraordinariamente",
        "se deja constancia del acuerdo alcanzado por las partes",
        "el ministerio publico ha emitido su pronunciamiento oficial",
        "mediante decreto se establece la nueva normativa aplicable",
        # Parrafos largos
        "la asamblea constituyente en sesion plenaria resolvio promulgar las modificaciones al capitulo tercero de la carta magna. las deliberaciones se extendieron por mas de quince horas consecutivas culminando en una votacion historica que aprobo la enmienda por amplia mayoria ratificando el compromiso del Estado con las instituciones democraticas",
        "con relacion al expediente administrativo numero cuatro cinco seis se informa que la contraloria procedera con el analisis exhaustivo de la documentacion contable presentada. el dictamen correspondiente sera notificado a las partes involucradas conforme a los plazos legales establecidos en la legislacion vigente",
    ]


def generate_dataset():
    """Build tone dataset: real Spanish data + synthetic urgente/formal."""

    print("=" * 60)
    print("STEP 1: Building tone dataset (real + synthetic)...")
    print("=" * 60)

    data = []
    real_data = _download_real_dataset()

    if real_data:
        label_map = {'positive': 1, 'negative': 3, 'neutral': 0}
        for src_label, our_label in label_map.items():
            texts = real_data.get(src_label, [])
            random.shuffle(texts)
            texts = texts[:REAL_DATASET_SAMPLES]
            for text in texts:
                data.append({'text': text, 'label': our_label})
            print(f"  {src_label} → {LABELS[our_label]}: {len(texts)} real samples")
            
        # Add long synthetic paragraphs to prevent length bias
        long_negatives = [
            "en 1978 la relacion entre chile y argentina alcanzo un punto critico que casi desencadena una guerra total por el control de las islas. la crisis diplomatica genero un escenario de tension extrema movilizando tropas hacia la frontera y creando una atmosfera de inminente conflicto belico",
            "este es indudablemente el peor servicio que he experimentado en toda mi vida. he estado intentando comunicarme con el soporte tecnico durante tres semanas seguidas sin recibir absolutamente ninguna respuesta coherente perdiendo dinero clientes y mi paciencia. la plataforma se cae constantemente destruyendo todo mi trabajo",
            "la catastrofe financiera de mil novecientos ochenta y dos provocó una de las contracciones economicas mas brutales en la historia reciente. el colapso sistemico de los bancos arruino los ahorros de millones causando un aumento desproporcionado del desempleo y una fractura social de dimensiones historicas"
        ]
        long_positives = [
            "la revolucion cientifica liderada por investigadores locales ha marcado un hito sin precedentes en la medicina moderna. tras decadas de estudio incansable el equipo logro desarrollar un tratamiento revolucionario que mejora dramaticamente la calidad de vida ofreciendo esperanza genuina y resultados extraordinarios",
            "estoy profundamente agradecido e impresionado con el increible nivel de profesionalismo entregado por todo el equipo. desde el primer dia superaron todas mis expectativas brindando un servicio impecable resolviendo cada minimo detalle con perfeccion y logrando un producto final verdaderamente majestuoso"
        ]
        for t in long_negatives: data.append({'text': t, 'label': 3})
        for t in long_positives: data.append({'text': t, 'label': 1})
    else:
        print("  ⚠ Fallback: synthetic neutral/positivo/molesto")
        for t in ["el clima hoy esta normal", "chile es un pais de america del sur",
                   "la temperatura es de veinte grados", "el tren sale a las ocho",
                   "santiago es la ciudad mas poblada", "el sistema solar tiene ocho planetas"]:
            data.append({'text': t, 'label': 0})
        for t in ["excelente trabajo bien hecho", "me encanta como quedo", "genial muchas gracias"]:
            data.append({'text': t, 'label': 1})
        for t in ["estoy harto de que no funcione", "terrible experiencia", "odio esperar"]:
            data.append({'text': t, 'label': 3})

    urgente = _synthetic_urgente()
    formal = _synthetic_formal()
    
    # Oversample synthetic to match REAL_DATASET_SAMPLES
    factor_urgente = max(1, REAL_DATASET_SAMPLES // len(urgente))
    urgente = (urgente * factor_urgente)[:REAL_DATASET_SAMPLES]
    
    factor_formal = max(1, REAL_DATASET_SAMPLES // len(formal))
    formal = (formal * factor_formal)[:REAL_DATASET_SAMPLES]

    for text in urgente:
        data.append({'text': text, 'label': 2})
    for text in formal:
        data.append({'text': text, 'label': 4})
    print(f"  urgente: {len(urgente)} (oversampled) | formal: {len(formal)} (oversampled)")

    df = pd.DataFrame(data)
    dataset_path = OUTPUT_DIR / "dataset_tone_es.csv"
    df.to_csv(dataset_path, index=False, encoding='utf-8')

    print(f"\n  Total dataset: {len(df)} samples")
    for i, label in enumerate(LABELS):
        print(f"    {label}: {len(df[df['label'] == i])}")
    return df


# ============================================================================
# STEP 1b: AUGMENT
# ============================================================================
def augment_dataset(df):
    """Augment dataset via word-level transformations."""
    print("\n  Augmenting dataset...")

    rng = np.random.RandomState(42)
    augmented = []

    for _, row in df.iterrows():
        text = row['text']
        label = row['label']
        words = text.split()

        augmented.append({'text': text, 'label': label})

        for _ in range(AUGMENT_FACTOR - 1):
            w = words.copy()
            r = rng.random()

            if r < 0.3 and len(w) > 2:
                i = rng.randint(0, len(w) - 1)
                j = rng.randint(0, len(w) - 1)
                w[i], w[j] = w[j], w[i]
            elif r < 0.5 and len(w) > 3:
                idx = rng.randint(0, len(w))
                w.pop(idx)
            elif r < 0.7:
                idx = rng.randint(0, len(w))
                w.insert(idx, w[idx])
            elif r < 0.85 and len(w) > 2:
                i = rng.randint(0, len(w) - 2)
                j = rng.randint(i + 1, min(i + 4, len(w)))
                w[i:j] = w[i:j][::-1]

            augmented.append({'text': ' '.join(w), 'label': label})

    aug_df = pd.DataFrame(augmented)
    print(f"  Augmented: {len(df)} → {len(aug_df)} samples")
    return aug_df


# ============================================================================
# STEP 2: BUILD TOKENIZER
# ============================================================================
def build_tokenizer(df):
    """Build word-level tokenizer compatible with web app."""

    print("\n" + "=" * 60)
    print("STEP 2: Building tokenizer...")
    print("=" * 60)

    word_counts = {}
    for text in df['text']:
        words = text.lower().split()
        for w in words:
            w = ''.join(c for c in w if c.isalnum() or c in 'áéíóúñü')
            if w:
                word_counts[w] = word_counts.get(w, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])

    vocab = {"[PAD]": 0}
    for i, (word, count) in enumerate(sorted_words):
        if i >= VOCAB_SIZE - 1:
            break
        vocab[word] = i + 1

    vocab_path = OUTPUT_DIR / "vocab.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Top 20 words: {[w for w, _ in sorted_words[:20]]}")
    return vocab


def tokenize_dataset(df, vocab):
    """Tokenize all texts and pad to MAX_LEN."""

    print("\n  Tokenizing dataset...")

    X = []
    for text in df['text']:
        words = text.lower().split()
        ids = []
        for w in words:
            w = ''.join(c for c in w if c.isalnum() or c in 'áéíóúñü')
            if w and w in vocab:
                ids.append(vocab[w])

        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
        while len(ids) < MAX_LEN:
            ids.append(0)
        X.append(ids)

    X = np.array(X, dtype=np.int32)
    y = np.array(df['label'].values, dtype=np.int32)

    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Class distribution: {np.bincount(y)}")
    return X, y


# ============================================================================
# STEP 3: BUILD MODEL
# ============================================================================
def build_model(vocab_size):
    """Build a small Keras model for tone detection."""

    print("\n" + "=" * 60)
    print("STEP 3: Building model...")
    print("=" * 60)

    actual_vocab = min(vocab_size, VOCAB_SIZE)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(actual_vocab, EMBED_DIM,
                                  input_length=MAX_LEN, name='embedding'),
        tf.keras.layers.GlobalAveragePooling1D(name='pool'),
        tf.keras.layers.Dense(HIDDEN_DIM, activation='relu', name='hidden'),
        tf.keras.layers.Dropout(0.1, name='dropout'),
        tf.keras.layers.Dense(NUM_LABELS, activation='softmax', name='output'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.build(input_shape=(None, MAX_LEN))
    model.summary()
    total_params = model.count_params()
    print(f"  Total params: {total_params} ({total_params * 4 / 1024:.1f} KB float32)")
    return model


def train_model(model, X, y):
    """Train with early stopping."""

    print("\n" + "=" * 60)
    print("STEP 3b: Training model...")
    print("=" * 60)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, min_delta=0.001
        )
    ]

    history = model.fit(
        X, y,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks, verbose=1
    )

    metrics = {
        'train_acc': float(history.history['accuracy'][-1]),
        'val_acc': float(history.history['val_accuracy'][-1]),
        'train_loss': float(history.history['loss'][-1]),
        'val_loss': float(history.history['val_loss'][-1]),
        'epochs_trained': len(history.history['loss']),
    }

    print(f"\n  Train acc:  {metrics['train_acc']:.4f}")
    print(f"  Val acc:    {metrics['val_acc']:.4f}")
    print(f"  Epochs:     {metrics['epochs_trained']}")
    return model, history, metrics


# ============================================================================
# STEP 4: QUANTIZE (for size reference)
# ============================================================================
def quantize_model(model, X):
    """Quantize to INT8 TFLite (size reference only)."""

    print("\n" + "=" * 60)
    print("STEP 4: Quantizing to INT8 TFLite...")
    print("=" * 60)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_gen():
        for i in range(min(200, len(X))):
            yield [X[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = OUTPUT_DIR / "tone_model.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"  TFLite: {len(tflite_model)} bytes ({len(tflite_model)/1024:.1f} KB)")

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input:  {input_details[0]['shape']}, {input_details[0]['dtype']}")
    print(f"  Output: {output_details[0]['shape']}, {output_details[0]['dtype']}")
    return tflite_model, input_details, output_details


# ============================================================================
# STEP 5: EXPORT RAW WEIGHTS (no TFLite dependency on ESP32)
# ============================================================================
def export_weights_header(model):
    """Export model weights as plain C float arrays."""

    print("\n" + "=" * 60)
    print("STEP 5: Exporting raw weights as C header...")
    print("=" * 60)

    emb_w = model.get_layer('embedding').get_weights()[0]
    d1_w = model.get_layer('hidden').get_weights()[0]
    d1_b = model.get_layer('hidden').get_weights()[1]
    d2_w = model.get_layer('output').get_weights()[0]
    d2_b = model.get_layer('output').get_weights()[1]

    print(f"  Embedding: {emb_w.shape}")
    print(f"  Dense1 W:  {d1_w.shape}, B: {d1_b.shape}")
    print(f"  Dense2 W:  {d2_w.shape}, B: {d2_b.shape}")

    def write_1d(f, name, arr):
        f.write(f"static const float {name}[{len(arr)}] = {{\n")
        for i in range(0, len(arr), 8):
            chunk = arr[i:i+8]
            f.write("  " + ", ".join(f"{v:.6f}f" for v in chunk) + ",\n")
        f.write("};\n\n")

    def write_2d(f, name, arr):
        rows, cols = arr.shape
        f.write(f"static const float {name}[{rows}][{cols}] = {{\n")
        for r in range(rows):
            row_str = ", ".join(f"{v:.6f}f" for v in arr[r])
            f.write(f"  {{{row_str}}},\n")
        f.write("};\n\n")

    header_path = OUTPUT_DIR / "model_weights.h"

    with open(header_path, 'w') as f:
        f.write("/* model_weights.h — Auto-generated for CuantumWiki\n")
        f.write(f" * Arch: Embed({emb_w.shape[0]},{emb_w.shape[1]}) → AvgPool → Dense({d1_w.shape[1]},relu) → Dense({d2_w.shape[1]},softmax)\n")
        f.write(f" * NO TFLite dependency!\n */\n\n")
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
        f.write(f"#define MODEL_VOCAB_SIZE  {emb_w.shape[0]}\n")
        f.write(f"#define MODEL_EMBED_DIM   {emb_w.shape[1]}\n")
        f.write(f"#define MODEL_INPUT_LEN   {MAX_LEN}\n")
        f.write(f"#define MODEL_HIDDEN_DIM  {d1_w.shape[1]}\n")
        f.write(f"#define MODEL_NUM_LABELS  {d2_w.shape[1]}\n\n")

        write_2d(f, "emb_weights", emb_w)
        write_2d(f, "dense1_weights", d1_w)
        write_1d(f, "dense1_bias", d1_b)
        write_2d(f, "dense2_weights", d2_w)
        write_1d(f, "dense2_bias", d2_b)

        f.write("#endif // MODEL_WEIGHTS_H\n")

    size_kb = header_path.stat().st_size / 1024
    print(f"  Weights header: {header_path} ({size_kb:.1f} KB)")
    return header_path


# ============================================================================
# STEP 6: WEB VOCAB
# ============================================================================
def export_web_vocab(vocab):
    """Export vocab as JS module for the web app."""

    print("\n" + "=" * 60)
    print("STEP 6: Exporting web-compatible vocab...")
    print("=" * 60)

    js_path = OUTPUT_DIR / "vocab_web.js"
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write("/* vocab_web.js — Auto-generated vocabulary */\n\n")
        f.write("const TRAINED_VOCAB = ")
        f.write(json.dumps(vocab, ensure_ascii=False, indent=2))
        f.write(";\n\n")
        f.write(f"const TRAINED_MAX_LEN = {MAX_LEN};\n")
        f.write(f"const TRAINED_LABELS = {json.dumps(LABELS)};\n")

    print(f"  Web vocab saved to: {js_path}")


# ============================================================================
# STEP 7: REPORT
# ============================================================================
def save_report(metrics, vocab_size, model_size_kb):
    report_path = OUTPUT_DIR / "training_report.txt"
    with open(report_path, 'w') as f:
        f.write("CuantumWiki — TinyML Tone Model Training Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Labels:           {', '.join(LABELS)}\n")
        f.write(f"Vocab size:       {vocab_size}\n")
        f.write(f"Max tokens:       {MAX_LEN}\n")
        f.write(f"Embedding dim:    {EMBED_DIM}\n")
        f.write(f"Hidden dim:       {HIDDEN_DIM}\n")
        f.write(f"Model size:       {model_size_kb:.1f} KB (INT8)\n\n")
        f.write(f"Train accuracy:      {metrics['train_acc']:.4f}\n")
        f.write(f"Validation accuracy: {metrics['val_acc']:.4f}\n")
        f.write(f"Train loss:          {metrics['train_loss']:.4f}\n")
        f.write(f"Validation loss:     {metrics['val_loss']:.4f}\n")
    print(f"\n  Report saved to: {report_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n╔" + "═" * 58 + "╗")
    print("║  CuantumWiki — TinyML Tone Model Training Pipeline      ║")
    print("╚" + "═" * 58 + "╝\n")

    df = generate_dataset()
    df = augment_dataset(df)

    vocab = build_tokenizer(df)
    X, y = tokenize_dataset(df, vocab)

    model = build_model(len(vocab))
    model, history, metrics = train_model(model, X, y)

    tflite_model, _, _ = quantize_model(model, X)

    export_weights_header(model)
    export_web_vocab(vocab)

    model_size_kb = len(tflite_model) / 1024
    save_report(metrics, len(vocab), model_size_kb)

    print("\n╔" + "═" * 58 + "╗")
    print("║  ✅ Pipeline complete!                                   ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Copy model/model_weights.h to firmware/worker_c3/")
    print(f"  Copy model/vocab_web.js to docs/")
    print(f"  Validation accuracy: {metrics['val_acc']:.1%}")
    print()


if __name__ == '__main__':
    main()
