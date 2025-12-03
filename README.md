# 📘 DMEyF 2025 — Maestría en Explotación de Datos y Descubrimiento del Conocimiento (FCEN – UBA)

## Competencia 03 — Pipeline Completo de Modelado de Churn en Clientes Premium


Este proyecto corresponde a la Competencia 03 de la materia DMEyF 2025 y consiste en desarrollar un pipeline integral de modelado predictivo para estimar la probabilidad de churn (fuga) en clientes del segmento premium de un banco.

El modelo principal utilizado es zLightGBM, una variante personalizada de LightGBM que incorpora la lógica de canaritos para la detección de sobreajuste y la validación de integridad del pipeline.

---

# 🚀 Objetivo del Proyecto
## Construir un pipeline reproducible de punta a punta, que incluye:

* Feature engineering
* Entrenamiento del modelo zLightGBM
* Evaluación del modelo
* Generación de predicciones finales


---

# 🧠 Modelo Utilizado: zLightGBM
## zLightGBM es una adaptación de LightGBM que incorpora:

Canaritos para control de generalización
Ajustes específicos para alta dimensionalidad como el subsampleo.

---

# ▶️ Cómo ejecutar el pipeline completo


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
| 9 | Ejecutar pipeline completo | `python main.py` |

### 📦 Resultado

#### Generando modelos particulares

Ejecutar proyecto desde `main.py` eligiendo el experimento correspondiente dentro de la carpeta `src_experimentos` especificando el modelo a ejecutar.


#### Generando ensambles

Ejecutar el archivo `main.py` dentro de la carpeta `ensambles` especificando en `c_3_exp_ENSAMBLE_automatico.py`. Esto generará el modelo entrenado y el archivo de predicciones finales listo para submit de modo iterativo, para todas las combinaciones de modelos ya realizados y alojados en el Bucket de la VM de Google Cloud.

---


# 📊 Resultados

El pipeline produce:

Predicciones de churn para el conjunto de evaluación
Logs del proceso
Modelo entrenado 
Métricas internas del desempeño
Archivo final para submit

---

## 📊 Resultados del Modelo Ensamble

| Concepto | Detalle |
|----------|---------|
| **Modelos seleccionados para el ensamble** | ... |
| **Ganancia — Mes Test 06** | ... |
| **Ganancia — Mes Test 07** | ... |
| **Ganancia estimada — Mes a predecir 09** | Promedio del backtest en meses 06 y 07 |
| **Nro clientes estimulados — Mes a predecir 09** | ... |

*Nota:*

En el `config.yaml` se uso la siguiente configuracion para generar para todos los experimentos:



A excepcion de los modelos "303, 304, 305 y 314 a", donde se usaron:

.
