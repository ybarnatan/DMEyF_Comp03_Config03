# 📘 DMEyF 2025 — Maestría en Explotación de Datos y Descubrimiento del Conocimiento (FCEN – UBA)

## Competencia 03 — Pipeline Completo de Modelado de Churn en Clientes Premium


Este proyecto corresponde a la Competencia 03 de la materia DMEyF 2025 y consiste en desarrollar un pipeline integral de modelado predictivo para estimar la probabilidad de churn (fuga) en clientes del segmento premium de un banco.

El modelo principal utilizado es zLightGBM, una variante personalizada de LightGBM que incorpora la lógica de canaritos para la detección de sobreajuste y la validación de integridad del pipeline.


## 🚀 Objetivo del Proyecto

 Construir un pipeline reproducible de punta a punta, que incluye:

* Feature engineering
* Entrenamiento del modelo zLightGBM
* Evaluación del modelo
* Generación de predicciones finales


## 🧠 Modelo Utilizado: zLightGBM 

zLightGBM es una adaptación de LightGBM que incorpora:

+ Canaritos para control de generalización
+ Ajustes específicos para alta dimensionalidad como el subsampleo.




## 🛠️ Pasos de Instalación y Ejecución

| Paso | Descripción | Comando |
|------|-------------|---------|
| 1 | Instalar Python | `sudo apt install -y python3.12-venv` |
| 2 | Clonar este repositorio | `git clone https://github.com/ybarnatan/DMEyF_Comp03_Config03.git` |
| 3 | Crear entorno virtual | `python3 -m venv .venv` |
| 4 | Activar entorno virtual | `source .venv/bin/activate` |
| 5 | Instalar dependencias | `pip install -r vm_requirements.txt` |
| 6 | Instalar zLightGBM (clonar repo LightGBM modificado) | ```bash\ncd\nrm -rf LightGBM\ngit clone --recursive https://github.com/dmecoyfin/LightGBM\n``` |
| 7 | Activar entorno y desinstalar LightGBM estándar | ```bash\nsource ~/.venv/bin/activate\npip uninstall --yes lightgbm\n``` |
| 8 | Instalar LightGBM modificado (zLightGBM) | ```bash\ncd ~/LightGBM\nsh ./build-python.sh install\n``` |
| 9 | Ejecutar pipeline completo cambiando el proceso principal en `config.yaml` | `python main.py` |


**Nota 1:** los procesos principales disponibles (punto 9), a correrse de modo secuencial y seteados en `config.yaml`, son:

* 1- `creacion_target_clase_ternaria`
* 2- `feat_eng`
* 3- `bayesiana`
* 4- `experimento` 
* 5- `prediccion_final` 


 **Nota 2:** Otros puntos de configuracion importantes en el `config.yaml` son:

* `SUBSAMPLEO`
* `SEMILLAS`
* `GANANCIA`
* `ESTIMULO`


 **Nota 3:** Proyecto diagramado para ser ejecutado en VM de Google Cloud o en local segun el parametro `IN_GCP` del `config.yaml`


## 📦 Resultado

#### Generando modelos particulares

Ejecutar proyecto desde `main.py` eligiendo el experimento correspondiente dentro de la carpeta `src_experimentos` especificando el modelo a ejecutar.





| Exp | Variables y Feat Eng             | Meses                        | Binaria | Subsampleo clase mayoritaria|
|-----|------------------------|------------------------------|---------|------------|
| 302 | Todas (percentiles)     | [2020, 2021)                 | 2       | 0.1        |
| 303 | Todas (percentiles)     | [2019, 2020)                 | 1       | 0.1        |
| 314c| Todas (percentiles)     | Post-Pandemia (202012 a 202104)| 1       | 0.1        |
| 321 | Todas                   | Todos                        | 1       | 0.05       |



+ Binaria 1: BAJA+1 y BAJA+2 juntos
+ Binaria 2: solo BAJA+2


#### Generando ensambles

Ejecutar el archivo `main.py` dentro de la carpeta `ensambles` especificando en `c_3_exp_ENSAMBLE_automatico.py`. Esto generará el modelo entrenado y el archivo de predicciones finales listo para submit de modo iterativo, para todas las combinaciones de modelos ya realizados y alojados en el Bucket de la VM de Google Cloud.



## 📊 Resultados

El pipeline produce:

+ Predicciones de churn para el conjunto de evaluación
+ Logs del proceso
+ Modelo entrenado 
+ Métricas internas del desempeño
+ Archivo final para submit


| Concepto | Detalle |
|----------|---------|
| **Combinacion de ensamble elegida (ver `.json`)** | `492` |
| **Modelos seleccionados para el ensamble** | `Exp302` , `Exp303`, `Exp 314c`, `Exp321` |
| **Ganancia meseta (+-500 clientes) — Mes Test 06** | `415 M (n clientes = 10407)` | 
| **Ganancia meseta (+-500 clientes) — Mes Test 07** | `432 M  (n clientes = 11597)` |
| **Ganancia estimada — Mes a predecir 09** | `Promedio del backtest en meses 06 y 07` |
| **Nro clientes estimulados — Mes a predecir 09** | `Umbral elegido a mano 11500` |


<img src="Entregable%20comp%2003/Ensamble 492 test 06.png" height ="400" width="600">
<img src="Entregable%20comp%2003/Ensamble 492 test 07.png" height ="400" width="600">


### 📅 Mes a predecir: Septiembre 2021

En el `config.yaml` se uso la configuracion seteada en el archivo para generar para todos los experimentos, a excepcion de los modelos "Exp303, 304, 305 y 314a", donde se uso:

+ SEMILLA: 80200
+ SEMILLAS: [80021, 80039, 80051, 80071, 80077]
