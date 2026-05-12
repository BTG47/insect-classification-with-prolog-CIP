# Agente Neuro-Simbólico para Clasificación de Insectos

Proyecto final de Inteligencia Artificial. El sistema clasifica insectos a partir de imágenes combinando visión por computadora y razonamiento simbólico/probabilístico.

La arquitectura implementada integra tres componentes principales:

1. **YOLOv8n**: detecta partes visibles del insecto, como `antennae`, `body`, `insect`, `legs`, `mouthpart` y `wings`.
2. **MobileNetV3 Small multi-salida**: predice la especie y rasgos morfológicos desde la imagen del insecto.
3. **ProbLog / DeepProbLog-style reasoning**: convierte la evidencia visual a hechos lógicos y aplica reglas para producir una decisión final explicable.

> Nota importante: en la versión final del proyecto, YOLO y MobileNet funcionan como dos ramas visuales complementarias. YOLO aporta evidencia localizada de partes visibles y MobileNet aporta una predicción global de especie y rasgos. MobileNet no fue entrenado originalmente con recortes individuales generados por YOLO.

---

## 1. Objetivo

Construir un agente inteligente capaz de recibir una imagen de un insecto, extraer evidencia visual y tomar una decisión final usando reglas simbólicas.

El agente recibe un estado:

```text
s = {imagen, prediccion_mobilenet, rasgos, partes_detectadas}
```

Y genera una acción:

```text
a = clase_final
```

La decisión no depende únicamente de la etiqueta predicha por MobileNet. El sistema combina:

- confianza del clasificador visual;
- rasgos morfológicos predichos;
- partes detectadas por YOLO;
- reglas simbólicas de consistencia;
- salida final o estado de revisión.

---

## 2. Clases consideradas

La versión final del agente trabaja con seis clases principales:

| Clase simbólica | Especie / etiqueta visual    |
| ---------------- | ---------------------------- |
| `mosquito`     | `asian_tiger_mosquito`     |
| `bee`          | `honey_bee`                |
| `butterfly`    | `monarch_butterfly`        |
| `lady_beetle`  | `sevenspotted_lady_beetle` |
| `grasshopper`  | `carolina_grasshopper`     |
| `mantis`       | `european_mantid`          |

---

## 3. Arquitectura general

```text
Imagen de entrada
   ├── YOLOv8n
   │      └── partes detectadas: insect, body, legs, mouthpart, wings, antennae
   │
   ├── MobileNetV3 Small multi-head
   │      ├── especie predicha
   │      └── rasgos morfológicos
   │
   └── JSON de evidencia estructurada
          ├── cnn_insect(...)
          ├── trait(...)
          ├── part_summary(...)
          └── part_seen(...)
                 ↓
          ProbLog / razonador simbólico
                 ↓
          clase final + especie final + reglas activadas + explicación
```

El flujo completo se puede resumir como:

```text
Imagen → YOLO + MobileNet → JSON → ProbLog / reglas → decisión final explicable
```

---

## 4. Estructura del repositorio

La estructura principal esperada es:

```text
.
├── README.md
├── requirements.txt
├── run_demo.py
├── run_vision.py            
├── run_reasoning.py           
├── data/
│   ├── demo_images/               # imágenes para demo local
│   ├── demo_json/                 # JSONs precalculados para probar razonamiento
│   └── data_extraction/           # material del dataset / extracción
├── docs/
│   ├── Interface_Contract_v1.pdf
│   ├── ResumenProlog_PrimeraParte_VictorHugo.pdf
│   └── Reporte_técnico.pdf      
├── models/
│   ├── best_yolo_insect_parts.pt
│   ├── best_mobilenet_insect_multitask.pth
│   ├── label_maps.json
│   └── idx_maps.json
├── notebooks/
│   ├── 01_YOLO_entrenamiento_partes.ipynb
│   ├── 02_MobileNet_entrenamiento_traits.ipynb
│   └── 03_Generar_JSON_para_DeepProbLog.ipynb
├── results/
│   ├── summary_results.csv
│   ├── analysis_results.md
│   ├── mobilenet_metrics.json
│   ├── mobilenet_training_summary.json
│   ├── vision_outputs/
│   ├── reasoning_outputs/
│   └── demo_artifacts/
└── src/
    ├── vision/                    # pipeline visual, si está disponible
    ├── reasoning/
    │   ├── json_multiclass_rules.pl
    │   ├── run_batch_integration.py
    │   └── symbolic_reasoner.py   # si está disponible
    └── utils/
```

Los notebooks se conservan como evidencia del entrenamiento en Colab. La ejecución local se enfoca en inferencia y demo reproducible usando modelos ya entrenados.

---

## 5. Instalación

Se recomienda usar un ambiente limpio de Conda o `venv`.

### Opción A: Conda

```bash
conda create -n insect_ai python=3.10 -y
conda activate insect_ai
pip install -r requirements.txt
```

### Opción B: venv en Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Opción C: venv en Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 6. Dependencias principales

Las dependencias principales son:

```text
torch
torchvision
ultralytics
opencv-python
pillow
numpy
pandas
scikit-learn
matplotlib
problog
```

`problog` se usa para la capa simbólica cuando está disponible. Algunas versiones del demo incluyen un modo de respaldo en Python para evitar que la demo falle si ProbLog no está instalado correctamente.

---

## 7. Ejecución rápida

### 7.1 Demo principal desde imagen

Coloca una imagen en:

```text
data/demo_images/
```

Después ejecuta:

```bash
python run_demo.py --image data/demo_images/mosquito_demo.jpg
```

La demo ejecuta:

1. carga de imagen;
2. detección de partes con YOLO;
3. predicción de especie y rasgos con MobileNet;
4. generación del JSON estructurado;
5. razonamiento simbólico;
6. impresión de la decisión final;
7. guardado de resultados y artefactos.

### 7.2 Demo rápida desde JSON

Si solo se quiere probar la capa simbólica sin ejecutar YOLO ni MobileNet:

```bash
python run_demo.py --json data/demo_json/case1.json
```

O, si el script tiene un JSON por defecto:

```bash
python run_demo.py
```

### 7.3 Forzar razonamiento en Python

Si ProbLog no está instalado o causa conflictos, se puede usar el fallback:

```bash
python run_demo.py --image data/demo_images/mosquito_demo.jpg --force-python
```

---

## 8. Salida esperada

Una ejecución correcta imprime una salida similar a:

```text
=== Demo neuro-simbólica de clasificación de insectos ===

[1] Evidencia visual generada por MobileNet
- Especie CNN: asian_tiger_mosquito (...)
- Top-k especies:
  · asian_tiger_mosquito: ...
  · monarch_butterfly: ...
  · carolina_grasshopper: ...

[2] Rasgos morfológicos principales
- wing_count: one_pair (...)
- forewing_type: membranous (...)
- mouthpart_type: piercing_sucking (...)
- antenna_type: filiform (...)
- leg_specialization: generalist (...)
- body_shape: slender (...)
- waist_shape: no_waist (...)

[3] Partes detectadas por YOLO
- insect: count=..., max_confidence=...
- body: count=..., max_confidence=...
- legs: count=..., max_confidence=...
- mouthpart: count=..., max_confidence=...

[4] Decisión simbólica
- Caso válido: True
- Soporte visual YOLO: True
- Requiere revisión: False
- Clase final: mosquito
- Especie final: asian_tiger_mosquito

[5] Reglas activadas
- valid_case
- visual_support
- signature_mosquito
- final_class:mosquito
- final_species:asian_tiger_mosquito
```

---

## 9. Artefactos generados por la demo

Cuando se ejecuta desde imagen, la demo puede generar archivos en:

```text
results/
├── vision_outputs/
│   └── last_prediction.json
├── reasoning_outputs/
│   └── last_reasoning.json
└── demo_artifacts/
    ├── 01_input_image.jpg
    ├── 02_yolo_detections.jpg
    ├── 03_mobilenet_input_crop.jpg
    ├── 04_pipeline_trace.txt
    └── artifacts_manifest.json
```

Estos archivos sirven para demostrar visualmente qué hizo el agente:

- `01_input_image.jpg`: imagen original.
- `02_yolo_detections.jpg`: imagen con cajas detectadas por YOLO.
- `03_mobilenet_input_crop.jpg`: imagen o crop usado como entrada visual principal.
- `04_pipeline_trace.txt`: pasos internos del pipeline.
- `last_prediction.json`: evidencia visual estructurada.
- `last_reasoning.json`: salida simbólica final.

---

## 10. Resultados principales

| Componente      |                    Métrica | Resultado |
| --------------- | --------------------------: | --------: |
| YOLOv8n         |                   Precision |     0.682 |
| YOLOv8n         |                      Recall |     0.545 |
| YOLOv8n         |                       mAP50 |     0.591 |
| YOLOv8n         |                    mAP50-95 |     0.353 |
| MobileNet       |       Insect label accuracy |     0.971 |
| MobileNet       |         Wing count accuracy |     0.971 |
| MobileNet       |      Forewing type accuracy |     1.000 |
| MobileNet       |     Mouthpart type accuracy |     0.971 |
| MobileNet       |       Antenna type accuracy |     1.000 |
| MobileNet       | Leg specialization accuracy |     0.971 |
| MobileNet       |         Body shape accuracy |     0.971 |
| MobileNet       |        Waist shape accuracy |     1.000 |
| Agente completo |             Casos evaluados |        34 |
| Agente completo |           Accuracy de clase |    0.8824 |
| Agente completo |         Accuracy de especie |    0.8824 |
| Agente completo |             Macro precision |    0.9762 |

---

## 11. Cómo decide el agente

El razonador recibe evidencia estructurada y evalúa reglas simbólicas. Por ejemplo, un caso de mosquito puede apoyarse en rasgos como:

```text
wing_count = one_pair
forewing_type = membranous
mouthpart_type = piercing_sucking
body_shape = slender
```

Además, YOLO aporta soporte visual si detecta el insecto o partes relevantes. Si la confianza visual y las reglas son compatibles, el sistema produce una clase final. Si la evidencia es insuficiente, el agente puede marcar el caso como `review`.

---

## 12. Notebooks incluidos

Los notebooks documentan el desarrollo experimental:

| Notebook                                    | Función                                                       |
| ------------------------------------------- | -------------------------------------------------------------- |
| `01_YOLO_entrenamiento_partes.ipynb`      | Entrenamiento y evaluación de YOLO para partes del insecto.   |
| `02_MobileNet_entrenamiento_traits.ipynb` | Entrenamiento de MobileNet multi-salida para especie y rasgos. |
| `03_Generar_JSON_para_DeepProbLog.ipynb`  | Generación de JSONs de evidencia para el razonador.           |

No es necesario reentrenar los modelos para correr la demo local.

---

## 13. Limitaciones

- El dataset es pequeño, por lo que la generalización a imágenes muy distintas puede ser limitada.
- Algunas partes pequeñas, como `mouthpart` y `legs`, son más difíciles de detectar para YOLO.
- MobileNet predice rasgos desde la imagen completa, no desde crops especializados por parte.
- Las reglas simbólicas usan umbrales manuales.
- La integración usa evidencia ya generada por YOLO y MobileNet; no es entrenamiento neuro-simbólico extremo a extremo.
- Si cambia el esquema JSON, también debe actualizarse el convertidor a hechos lógicos.

---

## 14. Trabajo futuro

Una mejora natural sería implementar una arquitectura `region-aware` completa:

```text
YOLO detecta partes → se generan crops por región → modelos/cabezas especializadas predicen rasgos → ProbLog razona con evidencia localizada
```

Por ejemplo:

| Región detectada | Rasgo a predecir                  |
| ----------------- | --------------------------------- |
| alas              | `wing_count`, `forewing_type` |
| antenas           | `antenna_type`                  |
| patas             | `leg_specialization`            |
| cuerpo            | `body_shape`, `waist_shape`   |
| aparato bucal     | `mouthpart_type`                |

Esta extensión permitiría una explicación morfológica más localizada y sería una buena continuación para trabajo de investigación posterior.

---

## 15. Créditos

Proyecto desarrollado para el curso **LIS3082-2 — Inteligencia Artificial**, Universidad de las Américas Puebla.

Integrantes:

- Bruno Tarango Garay
- Miguel Ángel García Arrieta
- Diego Flores Martínez
- Víctor Hugo de la Calleja Mojica

---

## 16. Comandos útiles para Git

```bash
git status
git add .
git commit -m "Add final reproducible neuro-symbolic demo"
git push
```

Antes de hacer commit, se recomienda verificar que no se suban archivos temporales:

```bash
git status
```
