# Análisis de resultados de la capa simbólica

- Total de casos evaluados: **34**
- Accuracy de clase: **0.8824**
- Accuracy de especie: **0.8824**
- Macro precision por clase: **0.9762**

## Precision por clase

- mosquito: **1.0000**
- bee: **1.0000**
- grasshopper: **0.8571**
- mantis: **1.0000**
- butterfly: **1.0000**
- lady_beetle: **1.0000**

## Reglas más importantes en clasificaciones correctas

- valid_case: **30** activaciones
- visual_support: **30** activaciones
- signature_bee: **6** activaciones
- final_class:bee: **6** activaciones
- final_species:honey_bee: **6** activaciones
- signature_grasshopper: **6** activaciones
- final_class:grasshopper: **6** activaciones
- final_species:carolina_grasshopper: **6** activaciones
- signature_mosquito: **5** activaciones
- final_class:mosquito: **5** activaciones

## Limitaciones

- El sistema depende de que la estructura del JSON se mantenga estable.
- Los umbrales de las reglas fueron definidos manualmente.
- Las reglas usan traits ya calculados por el modelo visual; no se entrenaron predicados `nn(...)` dentro de DeepProbLog.
- Si faltan partes detectadas por YOLO o traits correctos, la inferencia simbólica puede fallar.
- El mapeo de especie real a clase simbólica se basa en el prefijo del `image_id`.

## Conclusión

La integración JSON -> facts -> reglas -> consulta funciona y produce inferencias interpretables. Además, permite registrar métricas, reglas activadas y limitaciones del modelo.
