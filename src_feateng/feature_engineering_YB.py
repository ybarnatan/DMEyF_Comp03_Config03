#src/feature_engineering_YB.py
import pandas as pd
import numpy as np
import duckdb
import logging
from typing import List, Dict, Any
import gc
import re 

# Configuración del logger (simplificada)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def Feat_Eng_YB_digitalizacion(nombre_tabla: str, PATH_DATA_BASE_DB: str = "ruta/a/tu_base.duckdb") -> None:
    """
    Genera el Ratio de Digitalización (Transacciones Digitales / Transacciones Totales)
    directamente en la base DuckDB, sobrescribiendo la tabla original con la nueva columna.

    Cálculo:
        FE_RATIO_DIGITALIZACION = 
            (chomebanking_transacciones + cmobile_app_trx) /
            (chomebanking_transacciones + cmobile_app_trx + ccajas_transacciones)

    La función verifica primero si el feature ya existe en la tabla.

    Parámetros
    ----------
    nombre_tabla : str
        Nombre de la tabla existente en la base de datos DuckDB (e.g., 'df_completo').
    PATH_DATA_BASE_DB : str
        Ruta a la base de datos DuckDB.

    Retorna
    -------
    None
        Ejecuta el cálculo directamente sobre la base DuckDB.
    """

    FEATURE_NAME = 'FE_RATIO_DIGITALIZACION'
    logger.info("Comienzo feature de Digitalización")
    
    # --- FIX CRÍTICO: Validación de entrada para evitar el error de sintaxis ---
    if not isinstance(nombre_tabla, str) or not nombre_tabla.strip():
        error_msg = (
            "El parámetro 'nombre_tabla' debe ser una cadena de texto (string) no vacía que contenga "
            "solo el nombre de la tabla (ej: 'df_completo'). Parece que se le pasó un DataFrame o similar."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    # --------------------------------------------------------------------------

    # --- Lógica de Verificación (Adaptada para DuckDB) ---
    existing_cols = set()
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        
        # Consultar la información de la tabla para obtener las columnas
        # Se envuelve en un try/except interno por si la tabla no existe aún.
        try:
            cols_query = conn.execute(f"PRAGMA table_info('{nombre_tabla}')").df()
            existing_cols = set(cols_query['name'])
        except Exception:
            logger.warning(f"La tabla '{nombre_tabla}' no parece existir. Asumiendo que la columna no existe y procediendo.")

        conn.close()
    except Exception as e:
        logger.error(f"Error al conectar a DuckDB o consultar el esquema de la tabla. Ejecutando Feature por si acaso. Error: {e}")
        # Si falla la conexión, asumimos que debemos intentar crearlo


    if FEATURE_NAME.lower() in [col.lower() for col in existing_cols]:
        logger.info(f"Ya se hizo la Digitalización. Columna '{FEATURE_NAME}' ya existe.")
        return
    
    logger.info(f"Todavía no se hizo la Digitalización. Creando columna '{FEATURE_NAME}'.")
    
    # --- Definición de Columnas y Cálculo SQL ---
    
    # Columnas necesarias (se asume que existen, si no, DuckDB dará un error)
    digital_trx = ['chomebanking_transacciones', 'cmobile_app_trx']
    presential_trx = ['ccajas_transacciones']

    # --- FIX DE TIPO: Usamos CAST(col AS DOUBLE) dentro de COALESCE
    # para asegurar que estamos comparando un DOUBLE con el 0, resolviendo el Binder Error.
    # Crear expresiones SQL para la suma (usamos COALESCE para tratar NULL como 0)
    def build_sum_sql(cols: List[str]) -> str:
        # Aquí se aplica el CAST para evitar el error de mezcla de tipos (VARCHAR y INTEGER_LITERAL)
        return " + ".join(f"COALESCE(CAST({col} AS DOUBLE), 0)" for col in cols)

    digital_sum_sql = build_sum_sql(digital_trx)
    presential_sum_sql = build_sum_sql(presential_trx)
    
    # Suma total de transacciones (usada en el denominador)
    total_sum_sql = f"(({digital_sum_sql}) + ({presential_sum_sql}))"

    # Armar la consulta SQL final (usamos NULLIF para prevenir división por cero)
    # Ya no necesitamos CAST() en la parte superior del numerador, ya que digital_sum_sql ya es DOUBLE.
    sql_consulta = f"""
    CREATE OR REPLACE TABLE {nombre_tabla} AS
    SELECT 
        *,
        -- Cálculo: Digital / Total. El numerador ya es DOUBLE gracias al CAST en COALESCE.
        ({digital_sum_sql}) / 
        NULLIF({total_sum_sql}, 0)
        AS {FEATURE_NAME}
    FROM {nombre_tabla};
    """

    # logger.info(f"Consulta SQL generada:\n{sql_consulta}")
    
    # --- Ejecución sobre la base persistente ---
    logger.info(f"Conectando a la base DuckDB ({PATH_DATA_BASE_DB}) para ejecutar la consulta...")
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        conn.execute(sql_consulta)
        conn.close()
    except Exception as e:
        logger.error(f"Error al ejecutar la consulta en DuckDB. La causa más probable es que el nombre de la tabla ('{nombre_tabla}') sea incorrecto o que las columnas de transacciones no existan. Error subyacente: {e}")
        raise

    logger.info("Ejecución de Ratio de Digitalización finalizada correctamente ✅")
    



def Feat_Eng_YB_fx_exposure(nombre_tabla: str, PATH_DATA_BASE_DB: str = "ruta/a/tu_base.duckdb") -> None:
    """
    Genera el feature de Exposición a Moneda Extranjera (FX Exposure)
    como la proporción de saldos en dólares sobre el saldo total (dólares + pesos).
    
    Cálculo:
        FE_FX_EXPOSURE = 
            (mcaja_ahorro_dolares) /
            (mcaja_ahorro_dolares + mcaja_ahorro) 
            
    NOTA DE AJUSTE: Basado en el Binder Error anterior, se asume que la columna 
    de saldo en pesos/local para la caja de ahorro es 'mcaja_ahorro'.

    Parámetros
    ----------
    nombre_tabla : str
        Nombre de la tabla existente en la base de datos DuckDB (e.g., 'df_completo').
    PATH_DATA_BASE_DB : str
        Ruta a la base de datos DuckDB.
    """
    
    FEATURE_NAME = 'FE_FX_EXPOSURE'
    logger.info("Comienzo feature de Exposición FX")
    
    # Validación de entrada
    if not isinstance(nombre_tabla, str) or not nombre_tabla.strip():
        error_msg = "El parámetro 'nombre_tabla' debe ser una cadena de texto (string) no vacía."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # --- Lógica de Verificación (Adaptada para DuckDB) ---
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        cols_query = conn.execute(f"PRAGMA table_info('{nombre_tabla}')").df()
        existing_cols = set(cols_query['name'])
        conn.close()
        
        if FEATURE_NAME.lower() in [col.lower() for col in existing_cols]:
            logger.info(f"Ya se hizo la Exposición FX. Columna '{FEATURE_NAME}' ya existe.")
            return

    except Exception as e:
        logger.warning(f"Error al verificar la existencia de la columna '{FEATURE_NAME}'. Intentando crearla. Error: {e}")
    
    logger.info(f"Todavía no se hizo la Exposición FX. Creando columna '{FEATURE_NAME}'.")
    
    # --- Columnas de Saldo (Corregidas) ---
    # Columna en Dólares (Moneda Extranjera)
    col_dolares = 'mcaja_ahorro_dolares'
    # Columna en Pesos/Local (Corregida de 'mcaja_ahorro_pesos' a 'mcaja_ahorro')
    col_pesos_o_local = 'mcaja_ahorro' 

    # Función auxiliar para aplicar COALESCE y CAST a DOUBLE
    def build_cast_coalesce_sql(col: str) -> str:
        """Aplica COALESCE(CAST(col AS DOUBLE), 0) para manejar NULLs y tipos."""
        return f"COALESCE(CAST({col} AS DOUBLE), 0)"

    # Numerador: Saldo en Dólares (convertido y sin NULLs)
    numerator_sql = build_cast_coalesce_sql(col_dolares)
    
    # Denominador: Saldo Total (Dólares + Pesos)
    denominator_sql = f"({build_cast_coalesce_sql(col_dolares)} + {build_cast_coalesce_sql(col_pesos_o_local)})"

    # Armar la consulta SQL final (usamos NULLIF para prevenir división por cero)
    sql_consulta = f"""
    CREATE OR REPLACE TABLE {nombre_tabla} AS
    SELECT 
        *,
        -- Cálculo: Dólares / (Dólares + Pesos).
        {numerator_sql} / 
        NULLIF({denominator_sql}, 0)
        AS {FEATURE_NAME}
    FROM {nombre_tabla};
    """
    
    # logger.info(f"Consulta SQL generada:\n{sql_consulta}")
    
    # --- Ejecución sobre la base persistente ---
    logger.info(f"Conectando a la base DuckDB ({PATH_DATA_BASE_DB}) para ejecutar la consulta...")
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        # Se ejecuta la consulta
        conn.execute(sql_consulta)
        conn.close()
    except Exception as e:
        error_log = (
            f"Error al ejecutar la consulta en DuckDB. Verifica que la tabla '{nombre_tabla}' y las "
            f"columnas de exposición FX ('{col_dolares}' y '{col_pesos_o_local}') existan y sean numéricas. "
            f"Error: {e}"
        )
        logger.error(error_log)
        raise

    logger.info("Ejecución de Exposición FX finalizada correctamente ✅")


def Feat_Eng_YB_volatilidad(
    nombre_tabla: str, 
    VENTANA: int, 
    PATH_DATA_BASE_DB: str
) -> None:
    """
    Genera la Volatilidad Reciente (Desviación Estándar Móvil) de un conjunto 
    fijo de columnas clave, utilizando el tamaño de VENTANA pasado como argumento, 
    directamente en la base DuckDB.

    Cálculo:
        FE_STDDEV_{VENTANA}M_{col} = STDDEV_SAMP(col)
        OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes
              ROWS BETWEEN {VENTANA - 1} PRECEDING AND CURRENT ROW)

    Parámetros
    ----------
    nombre_tabla : str
        Nombre de la tabla existente en la base de datos DuckDB (e.g., 'df_completo').
    VENTANA : int
        Tamaño de la ventana móvil (en periodos/meses).
    PATH_DATA_BASE_DB : str
        Ruta a la base de datos DuckDB.

    Retorna
    -------
    None
        Ejecuta el cálculo directamente sobre la base DuckDB.
    """

    # --- CONSTANTES INTERNAS (Lista de columnas fijas) ---
    id_cliente = 'numero_de_cliente'
    id_periodo = 'foto_mes'
    
    # Lista de columnas numéricas clave para calcular la volatilidad.
    # Esta lista está definida internamente y no necesita pasarse como parámetro.
    COLUMNAS_VOLATILIDAD = [
        # Saldos y Pasivos
        'mpasivos_margen',           
        'mcaja_ahorro',              
        'mcuentas_saldo',            
        
        # Consumos y Transacciones
        'ctarjeta_debito'      
    ]

    logger.info(f"Comienzo feature de Volatilidad (Desviación Estándar Móvil) para VENTANA={VENTANA} y {len(COLUMNAS_VOLATILIDAD)} columnas.")

    # --- Validación de Entrada ---
    try:
        # Esto previene el error '<' not supported between instances of 'str' and 'int'
        VENTANA = int(VENTANA) 
    except ValueError:
        error_msg = f"El parámetro VENTANA debe ser un número entero, se recibió: {VENTANA}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not isinstance(nombre_tabla, str) or not nombre_tabla.strip():
        error_msg = "El parámetro 'nombre_tabla' debe ser una cadena de texto (string) no vacía."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if VENTANA < 2:
        error_msg = "La VENTANA debe ser al menos 2 para calcular una Desviación Estándar."
        logger.error(error_msg)
        raise ValueError(error_msg)


    # --- Lógica de Verificación (DuckDB PRAGMA) ---
    features_a_crear = [f"FE_STDDEV_{VENTANA}M_{col}" for col in COLUMNAS_VOLATILIDAD]
    features_existentes = set()
    
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        cols_query = conn.execute(f"PRAGMA table_info('{nombre_tabla}')").df()
        features_existentes = set(cols_query['name'])
        conn.close()
    except Exception as e:
        logger.error(f"Error al conectar a DuckDB o consultar el esquema de la tabla '{nombre_tabla}'. Se asume que las columnas deben ser creadas. Error: {e}")
        raise

    
    # Se filtra para ver qué features realmente necesitan ser creadas (comparación case-insensitive)
    features_pendientes = [feat for feat in features_a_crear if feat.lower() not in [col.lower() for col in features_existentes]]

    if not features_pendientes:
        logger.info(f"Ya se hicieron todas las features de Volatilidad para VENTANA={VENTANA}.")
        return
    
    logger.info(f"Faltan features de Volatilidad pendientes: {features_pendientes}")

    # --- Definición de SQL de la Ventana ---
    window_preceding = VENTANA - 1
    
    WINDOW_SPEC = f"""
        PARTITION BY {id_cliente} 
        ORDER BY {id_periodo} 
        ROWS BETWEEN {window_preceding} PRECEDING AND CURRENT ROW
    """
    
    # --- Generación de las Columnas de Desviación Estándar ---
    features_sql = []
    
    for col in COLUMNAS_VOLATILIDAD:
        feature_name = f"FE_STDDEV_{VENTANA}M_{col}"
        
        if feature_name in features_pendientes:
            # TRY_CAST: Asegura tipo numérico
            # STDDEV_SAMP: Desviación estándar muestral
            stddev_sql = f"""
            STDDEV_SAMP(TRY_CAST({col} AS DOUBLE)) 
            OVER ({WINDOW_SPEC}) AS {feature_name}
            """
            features_sql.append(stddev_sql)

    if not features_sql:
        logger.info("No hay columnas pendientes para generar. Finalizando.")
        return

    # --- Armar Consulta SQL de Creación (Patrón CREATE OR REPLACE) ---
    
    features_select_flat = ",\n 	".join(features_sql)

    sql_consulta = f"""
    CREATE OR REPLACE TABLE {nombre_tabla} AS
    SELECT 
        *,
        {features_select_flat}
    FROM {nombre_tabla};
    """
    
    
    # --- Ejecución sobre la base persistente ---
    logger.info(f"Conectando a la base DuckDB ({PATH_DATA_BASE_DB}) para ejecutar la consulta...")
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        conn.execute(sql_consulta)
        conn.close()
    except Exception as e:
        logger.error(f"Error al ejecutar la consulta en DuckDB. Error: {e}")
        raise

    logger.info(f"Ejecución de Features de Volatilidad (VENTANA={VENTANA}) finalizada correctamente ✅")





def Feat_Eng_YB_macro_event_dummies(
    nombre_tabla: str, 
    PATH_DATA_BASE_DB: str = "datasets/base_de_datos.duckdb" # Solo mantenemos los argumentos esenciales
) -> None:
    """
    Genera variables binarias (dummy) indicando si el 'foto_mes'
    es posterior o igual al mes de un evento macroeconómico,
    ejecutando directamente sobre la base DuckDB sin usar memoria.
    
    NOTA: Se asume que la columna de período es 'foto_mes'.

    Cálculo:
        FE_DUMMY_{nombre_evento} = CASE WHEN foto_mes >= mes_evento THEN 1 ELSE 0 END

    Parámetros
    ----------
    nombre_tabla : str
        Nombre de la tabla existente en la base de datos DuckDB (e.g., 'df_completo').
    PATH_DATA_BASE_DB : str
        Ruta a la base de datos DuckDB.

    Retorna
    -------
    None
        Ejecuta el cálculo directamente sobre la base DuckDB.
    """

    logger.info("Comienzo feature de Variables Binarias de Eventos Macroeconómicos.")
    
    # Columna de período hardcodeada para evitar errores de Binder
    id_periodo = 'foto_mes' 

    # Diccionario de eventos: Clave (string) es el nombre, Valor (int) es el mes
    eventos_macro = {
    # 2019: Transición de Gobierno y Cepo
    "PASO - Victoria Alberto-Cristina": 201908,
    "Fuerte devaluación post-PASO de Macri": 201908,
    "Reintroducción del Cepo Cambiario": 201909,
    "Elecciones Generales - Gana Alberto": 201910,
    "Ajuste del Cepo a USD 200": 201910,
    "Ley de Solidaridad Social e Impuesto PAÍS": 201912,

    # 2020: Pandemia y Deuda
    "Inicio COVID-19": 202003,
    "Caída histórica del PBI (Efecto Cuarentena)": 202004,
    "Acuerdo de reestructuración de la deuda con bonistas privados": 202008,
    "Ajuste al Dólar Ahorro (35% de percepción)": 202009, 

    # 2021: Continuación de la Crisis y Negociaciones con el FMI
    "Primeros acuerdos para el acceso a vacunas COVID-19": 202101,
    "Negociaciones iniciales con el FMI por nuevo programa": 202103,
    "Tensión inflacionaria y fuerte brecha cambiaria": 202108,
    }

    # --- Lógica de Verificación (DuckDB PRAGMA) ---
    features_existentes = set()
    
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        cols_query = conn.execute(f"PRAGMA table_info('{nombre_tabla}')").df()
        features_existentes = set(cols_query['name'])
        conn.close()
    except Exception as e:
        logger.error(f"Error al conectar a DuckDB o consultar el esquema de la tabla '{nombre_tabla}'. Asumiendo que las columnas deben ser creadas. Error: {e}")

    
    features_sql = []
    
    # Iterar sobre el diccionario para obtener el nombre y el mes del evento
    for nombre, mes in eventos_macro.items():
        
        # 1. Sanitización del nombre para la columna SQL (Mantiene la corrección de sintaxis)
        feature_suffix = nombre.replace(" ", "_").replace("-", "_")
        feature_suffix = re.sub(r'[^\w_]', '', feature_suffix).strip('_')
        feature_name = f"FE_DUMMY_{feature_suffix}"
        
        if feature_name.lower() not in [col.lower() for col in features_existentes]:
            # 2. Creación de la sentencia CASE WHEN en SQL
            # id_periodo ('foto_mes') ya está definido
            dummy_sql = f"""
            CASE 
                WHEN CAST({id_periodo} AS INTEGER) >= {mes} THEN 1 
                ELSE 0 
            END AS {feature_name}
            """
            features_sql.append(dummy_sql)
        else:
            logger.info(f"Saltando columna {feature_name} ya que existe.")

    if not features_sql:
        logger.info("No hay variables binarias pendientes para generar. Finalizando.")
        return

    # --- Armar Consulta SQL de Creación ---
    
    features_select = ",\n 	 	".join(features_sql)

    sql_consulta = f"""
    CREATE OR REPLACE TABLE {nombre_tabla} AS
    SELECT 
        *,
        {features_select}
    FROM {nombre_tabla};
    """

    # logger.info(f"Consulta SQL generada para Dummies Macroeconómicas:\n{sql_consulta}")
    
    # --- Ejecución sobre la base persistente ---
    logger.info(f"Conectando a la base DuckDB ({PATH_DATA_BASE_DB}) para ejecutar la consulta...")
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        conn.execute(sql_consulta)
        conn.close()
    except Exception as e:
        logger.error(f"Error al ejecutar la consulta en DuckDB. Verifica que la tabla '{nombre_tabla}' y la columna '{id_periodo}' existan. Error: {e}")
        raise

    logger.info("Ejecución de Features Dummies Macroeconómicas finalizada correctamente ✅")


def Feat_Eng_YB_product_count(
    nombre_tabla: str, 
    # MODIFICACIÓN: La ruta por defecto debe ser la correcta de depuración.
    PATH_DATA_BASE_DB: str = "datasets/base_de_datos.duckdb"
) -> None:
    """
    Genera el conteo de productos/servicios activos (valor > 0 y no nulo) 
    para las familias Visa y Master, de forma dinámica e in-place en DuckDB.

    La función detecta automáticamente las columnas de productos/servicios 
    buscando las palabras 'visa' o 'master' (case insensitive) en los nombres 
    de las columnas de la tabla.

    Parámetros
    ----------
    nombre_tabla : str
        Nombre de la tabla existente en la base de datos DuckDB (e.g., 'df_completo').
    PATH_DATA_BASE_DB : str
        Ruta a la base de datos DuckDB.

    Retorna
    -------
    None
        Ejecuta el cálculo directamente sobre la base DuckDB.
    """

    FEATURE_VISA = 'FE_COUNT_PROD_VISA'
    FEATURE_MASTER = 'FE_COUNT_PROD_MASTER'
    logger.info("Comienzo feature de Conteo de Productos/Servicios Activos (Búsqueda dinámica).")

    # --- 1. Lógica de Verificación y Detección de Columnas (DuckDB PRAGMA) ---
    features_existentes = set()
    all_cols = []
    
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        # Consultar información de la tabla para obtener la lista completa de columnas
        cols_query = conn.execute(f"PRAGMA table_info('{nombre_tabla}')").df()
        conn.close()
        
        features_existentes = set(cols_query['name'])
        all_cols = list(cols_query['name']) # Lista de todos los nombres de columna
        
    except Exception as e:
        logger.error(f"Error al conectar a DuckDB o consultar el esquema de la tabla '{nombre_tabla}'. Abortando. Error: {e}")
        return # Abortar si no podemos leer el esquema

    
    # 2. Detección Dinámica de Columnas (Case Insensitive)
    # Se añade un filtro para excluir las propias columnas de conteo si ya existieran
    visa_cols: List[str] = [
        col for col in all_cols 
        if 'visa' in col.lower() and col.lower() not in [FEATURE_VISA.lower(), FEATURE_MASTER.lower()]
    ]
    master_cols: List[str] = [
        col for col in all_cols 
        if 'master' in col.lower() and col.lower() not in [FEATURE_VISA.lower(), FEATURE_MASTER.lower()]
    ]

    if not visa_cols and not master_cols:
        logger.warning("No se encontraron columnas que contengan 'visa' o 'master'. No se generarán features de conteo.")
        return
        
    logger.info(f"Columnas VISA detectadas ({len(visa_cols)}): {visa_cols}")
    logger.info(f"Columnas MASTER detectadas ({len(master_cols)}): {master_cols}")

    # --- 3. Generación del SQL ---
    features_sql = []
    
    # Conteo de Productos VISA
    if FEATURE_VISA not in features_existentes and visa_cols:
        # SUM(CASE WHEN col > 0 THEN 1 ELSE 0 END) para cada columna
        # Usamos COALESCE para tratar nulos como 0 antes de la comparación
        visa_case_statements = [f"CASE WHEN COALESCE({col}, 0) > 0 THEN 1 ELSE 0 END" for col in visa_cols]
        visa_count_sql = " + ".join(visa_case_statements)
        features_sql.append(f"({visa_count_sql}) AS {FEATURE_VISA}")
        logger.info(f"Creando feature: {FEATURE_VISA}")
    else:
        logger.info(f"Saltando feature: {FEATURE_VISA} (ya existe o no hay columnas VISA).")
        
    # Conteo de Productos MASTER
    if FEATURE_MASTER not in features_existentes and master_cols:
        master_case_statements = [f"CASE WHEN COALESCE({col}, 0) > 0 THEN 1 ELSE 0 END" for col in master_cols]
        master_count_sql = " + ".join(master_case_statements)
        features_sql.append(f"({master_count_sql}) AS {FEATURE_MASTER}")
        logger.info(f"Creando feature: {FEATURE_MASTER}")
    else:
        logger.info(f"Saltando feature: {FEATURE_MASTER} (ya existe o no hay columnas MASTER).")

    if not features_sql:
        logger.info("No hay variables de conteo pendientes para generar. Finalizando.")
        return

    # --- 4. Armar y Ejecutar Consulta SQL ---
    
    features_select = ",\n 	 	".join(features_sql) # Se usa el formato de tabulación del código

    sql_consulta = f"""
    CREATE OR REPLACE TABLE {nombre_tabla} AS
    SELECT 
        *,
        {features_select}
    FROM {nombre_tabla};
    """

    
    # Ejecución sobre la base persistente
    logger.info(f"Conectando a la base DuckDB ({PATH_DATA_BASE_DB}) para ejecutar la consulta...")
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        conn.execute(sql_consulta)
        conn.close()
    except Exception as e:
        # Se asegura de que se muestra el error y la traza
        logger.error(f"Error al ejecutar la consulta en DuckDB. Error: {e}")
        raise

    logger.info("Ejecución de Features de Conteo de Productos finalizada correctamente ✅")
    

def Feat_Eng_YB_rentabilidad_tendencias(
    nombre_tabla: str, 
    id_cliente: str = 'numero_de_cliente', 
    id_periodo: str = 'foto_mes', 
    # CORRECCIÓN: Usar la ruta de depuración estándar
    PATH_DATA_BASE_DB: str = "datasets/base_de_datos.duckdb"
) -> None:
    """
    Calcula tendencias de rentabilidad del cliente para detectar deterioro en la relación,
    ejecutando directamente sobre la base DuckDB.

    Todas las configuraciones de features y ventanas están definidas dentro de esta función.

    Features generadas:
    - FE_DELTA_RENTABILIDAD: Cambio vs mes anterior.
    - FE_RATIO_RENTAB_VS_PROM: Rentabilidad actual vs promedio histórico (6 meses).
    - FE_SLOPE_RENTAB_3M: Pendiente de rentabilidad últimos 3 meses (tendencia).
    - FE_MESES_CAIDA_RENTAB: Conteo de caídas de rentabilidad en los últimos 4 meses (aproximación).

    Parámetros
    ----------
    nombre_tabla : str
        Nombre de la tabla existente en la base de datos DuckDB.
    id_cliente : str, opcional
        Columna de partición.
    id_periodo : str, opcional
        Columna de ordenamiento.
    PATH_DATA_BASE_DB : str
        Ruta a la base de datos DuckDB.

    Retorna
    -------
    None
        Ejecuta el cálculo directamente sobre la base DuckDB.

    Ejemplo de Llamada (Argumentos Nombrados para evitar errores):
    ----------------------------------------------------------
    # Asegúrate de que 'PATH_DATA_BASE_DB' esté definida previamente
    Feat_Eng_YB_rentabilidad_tendencias(
        nombre_tabla='df_completo', 
        PATH_DATA_BASE_DB=PATH_DATA_BASE_DB
    )
    """
    
    # =====================================================================
    #             !!! VALIDACIÓN DE SEGURIDAD !!!
    # Si la función es llamada sin argumentos nombrados y id_cliente
    # es sobrescrito por la ruta de la DB, lo corregimos.
    DEFAULT_CLIENT_ID = 'numero_de_cliente'
    if id_cliente.endswith('.duckdb'):
        logger.warning(
            f"El argumento 'id_cliente' se ha establecido como ruta de DB: '{id_cliente}'. "
            f"Esto probablemente se debe a una llamada posicional incorrecta. "
            f"Se está corrigiendo automáticamente a '{DEFAULT_CLIENT_ID}'."
        )
        id_cliente = DEFAULT_CLIENT_ID
    # =====================================================================
    
    # =====================================================================
    #             !!! CONFIGURACIÓN LOCAL DE FEATURES !!!
    # =====================================================================
    FEATURE_DELTA = 'FE_DELTA_RENTABILIDAD'
    FEATURE_RATIO = 'FE_RATIO_RENTAB_VS_PROM'
    FEATURE_SLOPE = 'FE_SLOPE_RENTAB_3M'
    FEATURE_CAIDA = 'FE_MESES_CAIDA_RENTAB'
    BASE_COL = 'mrentabilidad'
    WINDOW_SLOPE = 3
    WINDOW_PROM = 6 # Usaremos 6 meses para el promedio histórico
    # =====================================================================

    logger.info("Comienzo feature de Tendencias de Rentabilidad del Cliente.")
    
    # --- Lógica de Verificación (DuckDB PRAGMA) ---
    features_a_crear = [FEATURE_DELTA, FEATURE_RATIO, FEATURE_SLOPE, FEATURE_CAIDA]
    features_existentes = set()
    
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        cols_query = conn.execute(f"PRAGMA table_info('{nombre_tabla}')").df()
        features_existentes = set(cols_query['name'])
        conn.close()
    except Exception as e:
        logger.error(f"Error al conectar a DuckDB o consultar el esquema de la tabla. Abortando. Error: {e}")
        return

    features_pendientes = [feat for feat in features_a_crear if feat.lower() not in [col.lower() for col in features_existentes]]

    if not features_pendientes:
        logger.info("Ya se hicieron todos los features de Tendencias de Rentabilidad. Finalizando.")
        return
    
    logger.info(f"Features pendientes de creación: {features_pendientes}")

    # Especificación de Ventana para 3 meses (Slope)
    WINDOW_SPEC_3M = f"""
        PARTITION BY {id_cliente} 
        ORDER BY {id_periodo} 
        ROWS BETWEEN {WINDOW_SLOPE - 1} PRECEDING AND CURRENT ROW
    """
    # Especificación de Ventana para 6 meses (Promedio)
    WINDOW_SPEC_PROM = f"""
        PARTITION BY {id_cliente} 
        ORDER BY {id_periodo} 
        ROWS BETWEEN {WINDOW_PROM} PRECEDING AND 1 PRECEDING -- Promedio de los 6 meses ANTERIORES
    """
    # Sentencia SQL para la columna base (asegura tipo DOUBLE y maneja NULL)
    BASE_COL_CAST = f"CAST(COALESCE({BASE_COL}, 0) AS DOUBLE)"
    
    # --- Consulta Final usando CTE ---
    
    # Aproximación del Conteo de Caídas (Suma de banderas de caída en los últimos 4 meses)
    consecutive_fall_approx = f"""
        SUM(T1.caida_rentab_flag) OVER (PARTITION BY {id_cliente} ORDER BY {id_periodo} ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
    """
    
    # CORRECCIÓN CLAVE: Envolver el CTE y el SELECT en el CREATE OR REPLACE TABLE
    # para asegurar que DuckDB lo ejecute como una única sentencia de creación de tabla.
    sql_consulta = f"""
    CREATE OR REPLACE TABLE {nombre_tabla} AS
    -- CTE para cálculos intermedios (Lag, Bandera de Caída, Promedio Histórico)
    WITH T1_Features_Rentabilidad AS (
        SELECT 
            *,
            -- 1. Lag (Rentabilidad del mes anterior)
            LAG({BASE_COL_CAST}, 1) OVER (PARTITION BY {id_cliente} ORDER BY {id_periodo}) AS rentab_lag_1,
            
            -- 2. Bandera de Caída (1 si la rentabilidad actual es menor que la anterior)
            CASE WHEN {BASE_COL_CAST} < LAG({BASE_COL_CAST}, 1) OVER (PARTITION BY {id_cliente} ORDER BY {id_periodo}) THEN 1 ELSE 0 END AS caida_rentab_flag,

            -- 3. Promedio Histórico de Rentabilidad (6 meses anteriores)
            AVG({BASE_COL_CAST}) OVER ({WINDOW_SPEC_PROM}) AS rentab_prom_historico
        FROM {nombre_tabla}
    )
    
    SELECT 
        -- Mantener todas las columnas originales
        T1.* EXCLUDE (rentab_lag_1, caida_rentab_flag, rentab_prom_historico),

        -- a) Delta de Rentabilidad
        {BASE_COL_CAST} - T1.rentab_lag_1 AS {FEATURE_DELTA},
        
        -- b) Ratio Rentabilidad vs Promedio
        {BASE_COL_CAST} / NULLIF(T1.rentab_prom_historico, 0) AS {FEATURE_RATIO},
        
        -- c) Tendencia (Slope) 3 meses
        regr_slope({BASE_COL_CAST}, CAST({id_periodo} AS DOUBLE)) OVER ({WINDOW_SPEC_3M}) AS {FEATURE_SLOPE},
        
        -- d) Meses de Caída Consecutiva (Aproximación)
        {consecutive_fall_approx} AS {FEATURE_CAIDA}

    FROM T1_Features_Rentabilidad T1;
    """
    
    # --- Ejecución sobre la base persistente ---
    logger.info(f"Conectando a la base DuckDB ({PATH_DATA_BASE_DB}) para ejecutar la consulta...")
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        conn.execute(sql_consulta)
        conn.close()
    except Exception as e:
        logger.error(f"Error al ejecutar la consulta en DuckDB. Verifica que la tabla '{nombre_tabla}' y la columna '{BASE_COL}' existan y sean numéricas. Error: {e}")
        raise

    logger.info("Ejecución de Features de Tendencias de Rentabilidad finalizada correctamente ✅")
    



def Feat_Eng_YB_engagement_activity(
    nombre_tabla: str, 
    id_cliente: str = 'numero_de_cliente', 
    id_periodo: str = 'foto_mes', 
    PATH_DATA_BASE_DB: str = "datasets/base_de_datos.duckdb"
) -> None:
    """
    Mide el nivel de compromiso y actividad del cliente con el banco,
    ejecutando directamente sobre la base DuckDB.

    Features generadas:
    - FE_TOTAL_TRX_VOLUNTARIAS: Suma de todas las transacciones activas.
    - FE_RATIO_ENGAGEMENT_DIGITAL: Proporción de transacciones digitales vs totales.
    - FE_DELTA_ACTIVE_QUARTER: Cambio en actividad vs trimestre anterior.
    - FE_PCT_CAMBIO_TRX: Variación porcentual de transacciones vs mes anterior.

    Parámetros
    ----------
    nombre_tabla : str
        Nombre de la tabla existente en la base de datos DuckDB.
    id_cliente : str, opcional
        Columna de partición.
    id_periodo : str, opcional
        Columna de ordenamiento.
    PATH_DATA_BASE_DB : str
        Ruta a la base de datos DuckDB.

    Retorna
    -------
    None
        Ejecuta el cálculo directamente sobre la base DuckDB.
    """
    
    # =====================================================================
    #          !!! VALIDACIÓN DE SEGURIDAD !!!
    # Si la función es llamada sin argumentos nombrados (ej: Feat_Eng_YB_engagement_activity('df_completo', PATH_DATA_BASE_DB)),
    # el PATH_DATA_BASE_DB se asigna a id_cliente. Lo corregimos.
    # =====================================================================
    DEFAULT_CLIENT_ID = 'numero_de_cliente'
    if id_cliente.endswith('.duckdb') or ('datasets' in id_cliente and '/' in id_cliente):
        logger.warning(
            f"El argumento 'id_cliente' se ha establecido como ruta de DB: '{id_cliente}'. "
            f"Se está corrigiendo automáticamente a '{DEFAULT_CLIENT_ID}' para la partición SQL."
        )
        # Asumimos que el valor que entró por error en id_cliente es la ruta de la DB,
        # pero como el PATH_DATA_BASE_DB ya está al final, solo corregimos id_cliente.
        id_cliente = DEFAULT_CLIENT_ID
    # =====================================================================

    # =====================================================================
    #           !!! CONFIGURACIÓN LOCAL DE FEATURES !!!
    # =====================================================================
    # Nombres de las features a generar
    FEATURE_TOTAL_TRX = 'FE_TOTAL_TRX_VOLUNTARIAS'
    FEATURE_RATIO_DIG = 'FE_RATIO_ENGAGEMENT_DIGITAL'
    FEATURE_DELTA_Q = 'FE_DELTA_ACTIVE_QUARTER'
    FEATURE_PCT_CHG = 'FE_PCT_CAMBIO_TRX'
    
    # Columnas de transacciones base (AJUSTAR SEGÚN TU ESQUEMA REAL)
    DIGITAL_TRX_COLS = ['cmobile_app_trx', 'chomebanking_transacciones']
    OTRAS_TRX_COLS = ['ccajas_transacciones', 'mtransacciones_cajero']
    ALL_TRX_COLS = DIGITAL_TRX_COLS + OTRAS_TRX_COLS
    
    WINDOW_QUARTER = 3 # Ventana para el Delta Trimestral
    # =====================================================================

    logger.info("Comienzo feature de Engagement y Actividad del Cliente.")
    
    # --- 1. Lógica de Verificación (DuckDB PRAGMA) ---
    features_a_crear = [FEATURE_TOTAL_TRX, FEATURE_RATIO_DIG, FEATURE_DELTA_Q, FEATURE_PCT_CHG]
    features_existentes = set()
    
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        # Verificamos si las columnas base existen antes de continuar
        cols_query = conn.execute(f"PRAGMA table_info('{nombre_tabla}')").df()
        table_cols = set(cols_query['name'])
        conn.close()
        
        # Validación de columnas base (opcional pero recomendable)
        missing_cols = [col for col in ALL_TRX_COLS if col not in table_cols]
        if missing_cols:
             logger.error(f"Faltan columnas de transacciones necesarias en la tabla '{nombre_tabla}': {missing_cols}. Abortando.")
             return

        features_existentes = table_cols
        
    except Exception as e:
        logger.error(f"Error al conectar a DuckDB o consultar el esquema de la tabla. Abortando. Error: {e}")
        return

    # Usamos list comprehension para filtrar por existencia (case sensitive)
    features_pendientes = [feat for feat in features_a_crear if feat not in features_existentes]

    if not features_pendientes:
        logger.info("Ya se hicieron todos los features de Engagement. Finalizando.")
        return
    
    logger.info(f"Features pendientes de creación: {features_pendientes}")

    # --- 2. Sentencias SQL Base ---
    
    # Suma de Transacciones Digitales (Numerador del Ratio)
    digital_sum_sql = " + ".join(f"COALESCE({col}, 0)" for col in DIGITAL_TRX_COLS)
    
    # Suma de Todas las Transacciones Voluntarias (Base para el Total y Denominador)
    total_sum_sql = f"({digital_sum_sql}) + " + " + ".join(f"COALESCE({col}, 0)" for col in OTRAS_TRX_COLS)
    
    # Sentencia SQL para asegurar el tipo DOUBLE para cálculos de ratios y deltas
    TOTAL_TRX_CAST = f"CAST(({total_sum_sql}) AS DOUBLE)"
    
    # --- 3. Consulta Final usando CTE ---

    sql_consulta = f"""
    -- La sentencia CREATE OR REPLACE TABLE envuelve el CTE y el SELECT final.
    CREATE OR REPLACE TABLE {nombre_tabla} AS
    
    -- CTE para calcular el total de transacciones y el lag (requerido para Delta y Pct Change)
    WITH T1_Base_Activity AS (
        SELECT 
            *,
            -- 1. Total Transacciones Voluntarias (Temporal para el CTE)
            ({total_sum_sql}) AS total_trx_calc,
            
            -- 2. Lag de la Actividad (para Pct Change)
            LAG({TOTAL_TRX_CAST}, 1) OVER (PARTITION BY {id_cliente} ORDER BY {id_periodo}) AS trx_lag_1m,
            
            -- 3. Lag de la Actividad Trimestral (para Delta Trimestral)
            LAG({TOTAL_TRX_CAST}, {WINDOW_QUARTER}) OVER (PARTITION BY {id_cliente} ORDER BY {id_periodo}) AS trx_lag_q
        FROM {nombre_tabla}
    )
    
    SELECT 
        -- Mantener todas las columnas originales y la feature total
        T1.* EXCLUDE (trx_lag_1m, trx_lag_q, total_trx_calc),

        -- Nueva Columna: Total de Transacciones
        T1.total_trx_calc AS {FEATURE_TOTAL_TRX},

        -- a) Ratio Engagement Digital
        CAST(({digital_sum_sql}) AS DOUBLE) / NULLIF(T1.total_trx_calc, 0) AS {FEATURE_RATIO_DIG},
        
        -- b) Delta Active Quarter (Cambio vs trimestre anterior)
        T1.total_trx_calc - T1.trx_lag_q AS {FEATURE_DELTA_Q},
        
        -- c) Porcentaje de Cambio de Transacciones (vs mes anterior)
        (T1.total_trx_calc - T1.trx_lag_1m) / NULLIF(T1.trx_lag_1m, 0) AS {FEATURE_PCT_CHG}

    FROM T1_Base_Activity T1;
    """
        
    # logger.info(f"Consulta SQL generada para Engagement y Actividad:\n{sql_consulta}")

    # --- 4. Ejecución sobre la base persistente ---
    logger.info(f"Conectando a la base DuckDB ({PATH_DATA_BASE_DB}) para ejecutar la consulta...")
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        conn.execute(sql_consulta)
        conn.close()
    except Exception as e:
        logger.error(f"Error al ejecutar la consulta en DuckDB. Verifica que la tabla '{nombre_tabla}' y las columnas de transacciones existan y sean numéricas. Error: {e}")
        # Es fundamental propagar el error si ocurre una falla en la DB
        raise

    logger.info("Ejecución de Features de Engagement y Actividad finalizada correctamente ✅")
    
    

def Feat_Eng_YB_productos(PATH_DATA_BASE_DB):
    """
    Analiza la cartera de productos y su evolución para detectar desvinculación.
    
    Justificación de Negocio:
    - Clientes con más productos tienen menor tasa de churn (switching costs)
    - La pérdida de productos es indicador fuerte de abandono progresivo
    - Clientes mono-producto son más vulnerables a ofertas de competencia
    
    Features generadas:
    - delta_productos: cambio en cantidad de productos
    - total_productos_credito: cantidad de productos de crédito
    - total_productos_ahorro: cantidad de productos de ahorro
    - ratio_productos_credito_vs_ahorro: balance entre productos
    - tiene_productos_ancla: indica productos que retienen
    - meses_perdiendo_productos: meses consecutivos perdiendo productos
    - diversificacion_productos: inverso de concentración
    """
    logger.info("Comienzo Feature de productos")
    
    # Verificar si las columnas ya existen
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    columnas_existentes = conn.execute("SELECT * FROM df_completo LIMIT 0").df().columns.tolist()
    conn.close()
    
    columnas_nuevas = [
        'delta_productos',
        'total_productos_credito',
        'total_productos_ahorro',
        'ratio_productos_credito_vs_ahorro',
        'tiene_productos_ancla',
        'meses_perdiendo_productos',
        'diversificacion_productos'
    ]
    
    if any(col in columnas_existentes for col in columnas_nuevas):
        logger.info(f"Las columnas de productos ya existen. Se omite la creación.")
        return
    
    logger.info("Creando features de productos")
    
    sql = """
    CREATE OR REPLACE TABLE df_completo AS (
        SELECT *,
            -- Cambio en cantidad de productos
            cproductos - COALESCE(cproductos_lag_1, cproductos) AS delta_productos,
            
            -- Productos de crédito (generan compromiso de largo plazo)
            COALESCE(cprestamos_personales, 0) + 
            COALESCE(cprestamos_prendarios, 0) + 
            COALESCE(cprestamos_hipotecarios, 0) AS total_productos_credito,
            
            -- Productos de ahorro/inversión
            COALESCE(cplazo_fijo, 0) + 
            COALESCE(cinversion1, 0) + 
            COALESCE(cinversion2, 0) AS total_productos_ahorro,
            
            -- Ratio productos crédito vs ahorro
            CASE 
                WHEN (COALESCE(cplazo_fijo, 0) + COALESCE(cinversion1, 0) + COALESCE(cinversion2, 0)) > 0
                THEN (COALESCE(cprestamos_personales, 0) + COALESCE(cprestamos_prendarios, 0) + 
                      COALESCE(cprestamos_hipotecarios, 0)) * 1.0 /
                     (COALESCE(cplazo_fijo, 0) + COALESCE(cinversion1, 0) + COALESCE(cinversion2, 0))
                ELSE NULL
            END AS ratio_productos_credito_vs_ahorro,
            
            -- Tiene productos ancla (alta barrera de salida)
            CASE 
                WHEN COALESCE(cprestamos_hipotecarios, 0) > 0 
                    OR COALESCE(cpayroll_trx, 0) > 0 
                    OR (COALESCE(cinversion1, 0) + COALESCE(cinversion2, 0)) > 0
                THEN 1 
                ELSE 0 
            END AS tiene_productos_ancla,
            
            -- Meses consecutivos perdiendo productos
            CASE 
                WHEN cproductos < COALESCE(cproductos_lag_1, cproductos) 
                    AND COALESCE(cproductos_lag_1, cproductos) < COALESCE(cproductos_lag_2, cproductos)
                THEN 2
                WHEN cproductos < COALESCE(cproductos_lag_1, cproductos)
                THEN 1
                ELSE 0
            END AS meses_perdiendo_productos,
            
            -- Diversificación de productos (inverso de concentración)
            cproductos * 1.0 / GREATEST(
                COALESCE(ccuenta_corriente, 0),
                COALESCE(ccaja_ahorro, 0),
                COALESCE(ctarjeta_visa, 0),
                COALESCE(ctarjeta_master, 0),
                1
            ) AS diversificacion_productos
            
        FROM df_completo
    )
    """
    
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info("Feature productos finalizada")
    return


def Feat_Eng_YB_tarjetas_credito(PATH_DATA_BASE_DB):
    """
    Analiza uso y salud de tarjetas de crédito como indicador de lealtad.
    
    Justificación de Negocio:
    - Tarjetas de crédito son productos de alta rentabilidad y lealtad
    - La caída en uso de tarjetas precede al churn
    - Utilización del límite indica salud financiera del cliente
    - Mora es predictor fuerte de abandono
    
    Features generadas:
    - limite_compra_total: límite total de compra
    - consumo_tarjetas_total: consumo total en tarjetas
    - utilizacion_limite_total: % de uso del límite
    - delta_consumo_tarjetas: cambio en consumo total
    - ratio_pago_vs_consumo: capacidad de pago
    - tiene_mora_tarjetas: indicador de mora
    - tarjeta_en_cierre: tarjeta en proceso de cierre
    - tendencia_consumo_tarjetas_3m: tendencia de consumo
    """
    logger.info("Comienzo Feature de tarjetas de crédito")
    
    # Verificar si las columnas ya existen
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    columnas_existentes = conn.execute("SELECT * FROM df_completo LIMIT 0").df().columns.tolist()
    conn.close()
    
    columnas_nuevas = [
        'limite_compra_total',
        'consumo_tarjetas_total',
        'utilizacion_limite_total',
        'delta_consumo_tarjetas',
        'ratio_pago_vs_consumo',
        'tiene_mora_tarjetas',
        'tarjeta_en_cierre',
        'tendencia_consumo_tarjetas_3m'
    ]
    
    if any(col in columnas_existentes for col in columnas_nuevas):
        logger.info(f"Las columnas de tarjetas de crédito ya existen. Se omite la creación.")
        return
    
    logger.info("Creando features de tarjetas de crédito")
    
    sql = """
    CREATE OR REPLACE TABLE df_completo AS (
        SELECT *,
            -- Límite total disponible
            COALESCE(Visa_mlimitecompra, 0) + COALESCE(Master_mlimitecompra, 0) AS limite_compra_total,
            
            -- Consumo total en tarjetas
            COALESCE(Visa_mconsumototal, 0) + COALESCE(Master_mconsumototal, 0) AS consumo_tarjetas_total,
            
            -- Utilización del límite (muy importante para riesgo)
            CASE 
                WHEN (COALESCE(Visa_mlimitecompra, 0) + COALESCE(Master_mlimitecompra, 0)) > 0
                THEN (COALESCE(Visa_mconsumototal, 0) + COALESCE(Master_mconsumototal, 0)) * 1.0 /
                     (COALESCE(Visa_mlimitecompra, 0) + COALESCE(Master_mlimitecompra, 0))
                ELSE 0
            END AS utilizacion_limite_total,
            
            -- Cambio en consumo de tarjetas
            (COALESCE(Visa_mconsumototal, 0) + COALESCE(Master_mconsumototal, 0)) - 
            (COALESCE(Visa_mconsumototal_lag_1, 0) + COALESCE(Master_mconsumototal_lag_1, 0)) 
                AS delta_consumo_tarjetas,
            
            -- Ratio pago vs consumo (capacidad de pago)
            CASE 
                WHEN (COALESCE(Visa_mconsumototal, 0) + COALESCE(Master_mconsumototal, 0)) > 0
                THEN (COALESCE(Visa_mpagado, 0) + COALESCE(Master_mpagado, 0)) * 1.0 /
                     (COALESCE(Visa_mconsumototal, 0) + COALESCE(Master_mconsumototal, 0))
                ELSE NULL
            END AS ratio_pago_vs_consumo,
            
            -- Tiene mora en alguna tarjeta
            CASE 
                WHEN COALESCE(Visa_delinquency, 0) = 1 OR COALESCE(Master_delinquency, 0) = 1 
                THEN 1 
                ELSE 0 
            END AS tiene_mora_tarjetas,
            
            -- Tarjeta en proceso de cierre
            CASE 
                WHEN COALESCE(Visa_status, 0) IN (6, 7, 9) OR COALESCE(Master_status, 0) IN (6, 7, 9)
                THEN 1
                ELSE 0
            END AS tarjeta_en_cierre,
            
            -- Tendencia de consumo (últimos 3 meses)
            CASE 
                WHEN (COALESCE(Visa_mconsumototal_lag_2, 0) + COALESCE(Master_mconsumototal_lag_2, 0)) > 0
                THEN ((COALESCE(Visa_mconsumototal, 0) + COALESCE(Master_mconsumototal, 0)) - 
                      (COALESCE(Visa_mconsumototal_lag_2, 0) + COALESCE(Master_mconsumototal_lag_2, 0))) / 2.0
                ELSE 0
            END AS tendencia_consumo_tarjetas_3m
            
        FROM df_completo
    )
    """
    
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info("Feature tarjetas de crédito finalizada")
    return


def Feat_Eng_YB_flujo_fondos(PATH_DATA_BASE_DB):
    """
    Detecta patrones de salida de fondos que preceden al churn.
    
    Justificación de Negocio:
    - Clientes que retiran fondos están preparando el cierre de cuenta
    - Balance negativo de transferencias indica migración a otro banco
    - Caída en saldo total es señal temprana de abandono
    
    Features generadas:
    - saldo_total_cuentas: suma de todos los saldos
    - balance_neto_transferencias: diferencia entre recibidas y emitidas
    - ratio_transferencias_emitidas_vs_recibidas: ratio de transferencias
    - ratio_extracciones_vs_depositos: comportamiento de efectivo
    - delta_saldo_total: cambio en saldo total
    - tendencia_saldo_3m: evolución del saldo
    - meses_consecutivos_caida_saldo: meses con caída de saldo
    """
    logger.info("Comienzo Feature de flujo de fondos")
    
    # Verificar si las columnas ya existen
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    columnas_existentes = conn.execute("SELECT * FROM df_completo LIMIT 0").df().columns.tolist()
    conn.close()
    
    columnas_nuevas = [
        'saldo_total_cuentas',
        'balance_neto_transferencias',
        'ratio_transferencias_emitidas_vs_recibidas',
        'ratio_extracciones_vs_depositos',
        'delta_saldo_total',
        'tendencia_saldo_3m',
        'meses_consecutivos_caida_saldo'
    ]
    
    if any(col in columnas_existentes for col in columnas_nuevas):
        logger.info(f"Las columnas de flujo de fondos ya existen. Se omite la creación.")
        return
    
    logger.info("Creando features de flujo de fondos")
    
    sql = """
    CREATE OR REPLACE TABLE df_completo AS (
        SELECT *,
            -- Saldo total en cuentas (muy importante)
            COALESCE(mcuenta_corriente, 0) + 
            COALESCE(mcaja_ahorro, 0) + 
            COALESCE(mcaja_ahorro_adicional, 0) +
            COALESCE(mcaja_ahorro_dolares, 0) AS saldo_total_cuentas,
            
            -- Balance neto de transferencias (negativo = está sacando plata)
            COALESCE(mtransferencias_recibidas, 0) - COALESCE(mtransferencias_emitidas, 0) 
                AS balance_neto_transferencias,
            
            -- Ratio transferencias emitidas vs recibidas
            CASE 
                WHEN COALESCE(mtransferencias_recibidas, 0) > 0
                THEN COALESCE(mtransferencias_emitidas, 0) * 1.0 / mtransferencias_recibidas
                ELSE NULL
            END AS ratio_transferencias_emitidas_vs_recibidas,
            
            -- Ratio extracciones vs depósitos en cajero
            CASE 
                WHEN (COALESCE(mcheques_depositados, 0) + COALESCE(mpayroll, 0)) > 0
                THEN COALESCE(mextraccion_autoservicio, 0) * 1.0 /
                     (COALESCE(mcheques_depositados, 0) + COALESCE(mpayroll, 0))
                ELSE NULL
            END AS ratio_extracciones_vs_depositos,
            
            -- Cambio en saldo total
            (COALESCE(mcuenta_corriente, 0) + COALESCE(mcaja_ahorro, 0) + 
             COALESCE(mcaja_ahorro_adicional, 0)) -
            (COALESCE(mcuenta_corriente_lag_1, 0) + COALESCE(mcaja_ahorro_lag_1, 0) + 
             COALESCE(mcaja_ahorro_adicional_lag_1, 0)) AS delta_saldo_total,
            
            -- Tendencia de saldo (últimos 3 meses)
            CASE 
                WHEN (COALESCE(mcuenta_corriente_lag_2, 0) + COALESCE(mcaja_ahorro_lag_2, 0)) > 0
                THEN ((COALESCE(mcuenta_corriente, 0) + COALESCE(mcaja_ahorro, 0)) - 
                      (COALESCE(mcuenta_corriente_lag_2, 0) + COALESCE(mcaja_ahorro_lag_2, 0))) / 2.0
                ELSE 0
            END AS tendencia_saldo_3m,
            
            -- Placeholder para meses_consecutivos_caida_saldo (se calcula en segunda pasada)
            0 AS meses_consecutivos_caida_saldo
            
        FROM df_completo
    );
    
    -- Segunda pasada para calcular meses_consecutivos_caida_saldo
    CREATE OR REPLACE TABLE df_completo AS (
        SELECT * EXCLUDE (meses_consecutivos_caida_saldo),
            CASE 
                WHEN saldo_total_cuentas < LAG(saldo_total_cuentas, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)
                    AND LAG(saldo_total_cuentas, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) < 
                        LAG(saldo_total_cuentas, 2) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)
                THEN 2
                WHEN saldo_total_cuentas < LAG(saldo_total_cuentas, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)
                THEN 1
                ELSE 0
            END AS meses_consecutivos_caida_saldo
        FROM df_completo
    )
    """
    
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info("Feature flujo de fondos finalizada")
    return


def Feat_Eng_YB_vip_satisfaccion(PATH_DATA_BASE_DB):
    '''
    Detecta cambios en status VIP y señales de insatisfacción.
    
    Justificación de Negocio:
    - Pérdida de status VIP indica deterioro de relación
    - Incremento en comisiones sin aumento de beneficios genera insatisfacción
    - Clientes que dejan de usar beneficios premium están en riesgo
    
    Features generadas:
    - delta_status_vip: cambio en clasificación VIP
    - ratio_comisiones_vs_rentabilidad: peso de comisiones
    - descuentos_totales: total de descuentos obtenidos
    - delta_descuentos_totales: cambio en descuentos
    - ratio_descuentos_vs_comisiones: value for money
    - delta_comisiones_mantenimiento: incremento en comisiones
    - uso_servicios_premium: uso de servicios premium
    '''
    logger.info("Comienzo Feature de VIP y satisfacción")
    
    # Verificar si las columnas ya existen
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    columnas_existentes = conn.execute("SELECT * FROM df_completo LIMIT 0").df().columns.tolist()
    conn.close()
    
    columnas_nuevas = [
        'delta_status_vip',
        'ratio_comisiones_vs_rentabilidad',
        'descuentos_totales',
        'delta_descuentos_totales',
        'ratio_descuentos_vs_comisiones',
        'delta_comisiones_mantenimiento',
        'uso_servicios_premium'
    ]
    
    if any(col in columnas_existentes for col in columnas_nuevas):
        logger.info(f"Las columnas de VIP y satisfacción ya existen. Se omite la creación.")
        return
    
    logger.info("Creando features de VIP y satisfacción")
    
    sql = """
    CREATE OR REPLACE TABLE df_completo AS (
        SELECT *,
            -- Cambio en status VIP (muy relevante)
            cliente_vip - COALESCE(cliente_vip_lag_1, cliente_vip) AS delta_status_vip,
            
            -- Ratio comisiones vs rentabilidad (si paga mucho y genera poco, se va)
            CASE 
                WHEN COALESCE(mrentabilidad, 0) > 0
                THEN COALESCE(mcomisiones, 0) * 1.0 / mrentabilidad
                ELSE NULL
            END AS ratio_comisiones_vs_rentabilidad,
            
            -- Total de descuentos obtenidos (beneficios premium)
            COALESCE(mcajeros_propios_descuentos, 0) + 
            COALESCE(mtarjeta_visa_descuentos, 0) + 
            COALESCE(mtarjeta_master_descuentos, 0) AS descuentos_totales,
            
            -- Cambio en descuentos (dejó de usar beneficios)
            (COALESCE(mcajeros_propios_descuentos, 0) + COALESCE(mtarjeta_visa_descuentos, 0) + 
             COALESCE(mtarjeta_master_descuentos, 0)) -
            (COALESCE(mcajeros_propios_descuentos_lag_1, 0) + COALESCE(mtarjeta_visa_descuentos_lag_1, 0) + 
             COALESCE(mtarjeta_master_descuentos_lag_1, 0)) AS delta_descuentos_totales,
            
            -- Ratio descuentos vs comisiones (value for money percibido)
            CASE 
                WHEN COALESCE(mcomisiones, 0) > 0
                THEN (COALESCE(mcajeros_propios_descuentos, 0) + COALESCE(mtarjeta_visa_descuentos, 0) + 
                      COALESCE(mtarjeta_master_descuentos, 0)) * 1.0 / mcomisiones
                ELSE NULL
            END AS ratio_descuentos_vs_comisiones,
            
            -- Incremento en comisiones de mantenimiento
            COALESCE(mcomisiones_mantenimiento, 0) - COALESCE(mcomisiones_mantenimiento_lag_1, 0) 
                AS delta_comisiones_mantenimiento,
            
                      -- Uso de servicios premium (forex, cajas seguridad, seguros)
            (COALESCE(cforex, 0) + COALESCE(ccaja_seguridad, 0) + COALESCE(cseguro_vida, 0) + 
             COALESCE(cseguro_auto, 0) + COALESCE(cseguro_vivienda, 0)) AS uso_servicios_premium
            
        FROM df_completo
    )
    """
    
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info("Feature VIP y satisfacción finalizada")
    return



def Feat_Eng_YB_estabilidad(PATH_DATA_BASE_DB):
    """
    Mide estabilidad financiera y madurez de la relación con el banco.
    
    Justificación de Negocio:
    - Clientes más antiguos tienen menor tasa de churn
    - Acreditación de haberes genera dependencia y estabilidad
    - Clientes jóvenes son más propensos a cambiar de banco
    - Pérdida de acreditación de sueldo es predictor fuerte de churn
    - Débitos automáticos generan switching costs
    
    Features generadas:
    - tiene_acreditacion_sueldo: cliente bancarizado
    - delta_transacciones_payroll: cambio en transacciones de payroll
    - perdio_acreditacion_sueldo: perdió acreditación
    - ratio_edad_vs_antiguedad: madurez de la relación
    - delta_monto_payroll: cambio en monto de payroll
    - total_debitos_automaticos: cantidad total de débitos automáticos
    - delta_debitos_automaticos: cambio en débitos automáticos
    - cliente_nuevo_premium: cliente reciente en premium
    """
    logger.info("Comienzo Feature de estabilidad")
    
    # Verificar si las columnas ya existen
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    columnas_existentes = conn.execute("SELECT * FROM df_completo LIMIT 0").df().columns.tolist()
    conn.close()
    
    columnas_nuevas = [
        'tiene_acreditacion_sueldo',
        'delta_transacciones_payroll',
        'perdio_acreditacion_sueldo',
        'ratio_edad_vs_antiguedad',
        'delta_monto_payroll',
        'total_debitos_automaticos',
        'delta_debitos_automaticos',
        'cliente_nuevo_premium'
    ]
    
    if any(col in columnas_existentes for col in columnas_nuevas):
        logger.info(f"Las columnas de estabilidad ya existen. Se omite la creación.")
        return
    
    logger.info("Creando features de estabilidad")
    
    sql = """
    CREATE OR REPLACE TABLE df_completo AS (
        SELECT *,
            -- Tiene acreditación de sueldo (muy importante para retención)
            CASE 
                WHEN COALESCE(cpayroll_trx, 0) > 0 OR COALESCE(cpayroll2_trx, 0) > 0 
                THEN 1 
                ELSE 0 
            END AS tiene_acreditacion_sueldo,
            
            -- Cambio en acreditación de sueldo
            (COALESCE(cpayroll_trx, 0) + COALESCE(cpayroll2_trx, 0)) - 
            (COALESCE(cpayroll_trx_lag_1, 0) + COALESCE(cpayroll2_trx_lag_1, 0)) AS delta_transacciones_payroll,
            
            -- Perdió la acreditación de sueldo
            CASE 
                WHEN (COALESCE(cpayroll_trx_lag_1, 0) + COALESCE(cpayroll2_trx_lag_1, 0)) > 0 
                    AND (COALESCE(cpayroll_trx, 0) + COALESCE(cpayroll2_trx, 0)) = 0
                THEN 1
                ELSE 0
            END AS perdio_acreditacion_sueldo,
            
            -- Ratio edad vs antigüedad (cliente maduro vs nuevo)
            CASE 
                WHEN COALESCE(cliente_antiguedad, 0) > 0
                THEN COALESCE(cliente_edad, 0) * 1.0 / cliente_antiguedad
                ELSE NULL
            END AS ratio_edad_vs_antiguedad,
            
            -- Cambio en monto de payroll
            (COALESCE(mpayroll, 0) + COALESCE(mpayroll2, 0)) - 
            (COALESCE(mpayroll_lag_1, 0) + COALESCE(mpayroll2_lag_1, 0)) AS delta_monto_payroll,
            
            -- Cantidad de débitos automáticos (genera dependencia)
            COALESCE(ccuenta_debitos_automaticos, 0) + 
            COALESCE(ctarjeta_visa_debitos_automaticos, 0) + 
            COALESCE(ctarjeta_master_debitos_automaticos, 0) AS total_debitos_automaticos,
            
            -- Cambio en débitos automáticos (perdió servicios)
            (COALESCE(ccuenta_debitos_automaticos, 0) + COALESCE(ctarjeta_visa_debitos_automaticos, 0) + 
             COALESCE(ctarjeta_master_debitos_automaticos, 0)) -
            (COALESCE(ccuenta_debitos_automaticos_lag_1, 0) + COALESCE(ctarjeta_visa_debitos_automaticos_lag_1, 0) + 
             COALESCE(ctarjeta_master_debitos_automaticos_lag_1, 0)) AS delta_debitos_automaticos,
            
            -- Cliente nuevo premium (más propenso a churn)
            CASE 
                WHEN COALESCE(cliente_antiguedad, 0) <= 12 
                THEN 1 
                ELSE 0 
            END AS cliente_nuevo_premium
            
        FROM df_completo
    )
    """
    
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info("Feature estabilidad finalizada")
    return


def Feat_Eng_YB_indicios_de_abandono(
    PATH_DATA_BASE_DB: str,
    nombre_tabla: str = 'df_completo' 
) -> None:
    """
    Crea indicadores compuestos de señales tempranas de abandono.
    
    Justificación de Negocio:
    - Combina múltiples señales débiles en indicadores fuertes de churn
    - Score de riesgo de abandono basado en comportamiento
    - Clientes con múltiples señales simultáneas tienen altísimo riesgo
    
    Features generadas:
    - score_señales_abandono: score compuesto de riesgo
    - tiene_multiples_señales_abandono: múltiples señales simultáneas
    - perdio_producto_principal: perdió producto core
    - caida_actividad_severa: caída drástica en actividad

    Parámetros
    ----------
    PATH_DATA_BASE_DB : str
        Ruta a la base de datos DuckDB.
    nombre_tabla : str, opcional
        Nombre de la tabla existente en la base de datos DuckDB (e.g., 'df_completo').
    
    Retorna
    -------
    None
        Ejecuta el cálculo directamente sobre la base DuckDB.
    """
    logger.info(f"Comienzo Feature de señales de abandono para la tabla '{nombre_tabla}'.")
    
    # Definición de features a crear
    features_a_crear = [
        'score_señales_abandono',
        'tiene_multiples_señales_abandono',
        'perdio_producto_principal',
        'caida_actividad_severa'
    ]

    # Columnas requeridas para los cálculos (dependencias de otras funciones)
    columnas_requeridas = [
        'delta_productos', 'meses_perdiendo_productos', 'delta_saldo_total', 
        'meses_consecutivos_caida_saldo', 'tiene_mora_tarjetas', 'tarjeta_en_cierre', 
        'perdio_acreditacion_sueldo', 'delta_status_vip', 'delta_debitos_automaticos',
        'pct_cambio_transacciones', 'ctarjeta_visa', 'ctarjeta_visa_lag_1',
        'ctarjeta_master', 'ctarjeta_master_lag_1', 'ccuenta_corriente', 'ccuenta_corriente_lag_1'
    ]
    
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        # 1. Obtener columnas existentes con PRAGMA
        cols_query = conn.execute(f"PRAGMA table_info('{nombre_tabla}')").df()
        columnas_existentes = set(cols_query['name'])
        conn.close()
    except Exception as e:
        logger.error(f"Error al conectar a DuckDB o consultar el esquema de la tabla '{nombre_tabla}'. Abortando. Error: {e}")
        return
        
    # 2. Verificar dependencias
    missing_required = [col for col in columnas_requeridas if col not in columnas_existentes]
    
    if missing_required:
        logger.warning(f"Faltan columnas requeridas para 'indicios de abandono' en '{nombre_tabla}': {missing_required}. Ejecutar primero las funciones de feature engineering previas.")
        return
    
    # 3. Verificar si los features ya existen
    features_pendientes = [feat for feat in features_a_crear if feat not in columnas_existentes]
    
    if not features_pendientes:
        logger.info(f"Las columnas de señales de abandono ya existen en '{nombre_tabla}'. Se omite la creación.")
        return
    
    logger.info("Creando features de señales de abandono")
    
    # Utilizamos el parámetro nombre_tabla en la consulta SQL
    sql = f"""
    CREATE OR REPLACE TABLE {nombre_tabla} AS (
        SELECT *,
            -- Score compuesto de señales de abandono (suma de flags ponderadas)
            (CASE WHEN delta_productos < 0 THEN 1 ELSE 0 END +
             CASE WHEN meses_perdiendo_productos >= 2 THEN 2 ELSE 0 END +
             CASE WHEN delta_saldo_total < -10000 THEN 2 ELSE 0 END +
             CASE WHEN meses_consecutivos_caida_saldo >= 2 THEN 1 ELSE 0 END +
             CASE WHEN tiene_mora_tarjetas = 1 THEN 2 ELSE 0 END +
             CASE WHEN tarjeta_en_cierre = 1 THEN 3 ELSE 0 END +
             CASE WHEN perdio_acreditacion_sueldo = 1 THEN 3 ELSE 0 END +
             CASE WHEN delta_status_vip < 0 THEN 2 ELSE 0 END +
             CASE WHEN delta_debitos_automaticos < 0 THEN 1 ELSE 0 END +
             CASE WHEN pct_cambio_transacciones < -0.5 THEN 2 ELSE 0 END) AS score_señales_abandono,
            
            -- Tiene múltiples señales simultáneas (3 o más señales no ponderadas)
            CASE 
                WHEN (CASE WHEN delta_productos < 0 THEN 1 ELSE 0 END +
                      CASE WHEN delta_saldo_total < 0 THEN 1 ELSE 0 END +
                      CASE WHEN tiene_mora_tarjetas = 1 THEN 1 ELSE 0 END +
                      CASE WHEN tarjeta_en_cierre = 1 THEN 1 ELSE 0 END +
                      CASE WHEN perdio_acreditacion_sueldo = 1 THEN 1 ELSE 0 END +
                      CASE WHEN delta_status_vip < 0 THEN 1 ELSE 0 END) >= 3
                THEN 1
                ELSE 0
            END AS tiene_multiples_señales_abandono,
            
            -- Perdió producto principal (tarjeta, cuenta o payroll)
            CASE 
                WHEN (COALESCE(ctarjeta_visa, 0) = 0 AND COALESCE(ctarjeta_visa_lag_1, 0) > 0)
                     OR (COALESCE(ctarjeta_master, 0) = 0 AND COALESCE(ctarjeta_master_lag_1, 0) > 0)
                     OR (COALESCE(ccuenta_corriente, 0) = 0 AND COALESCE(ccuenta_corriente_lag_1, 0) > 0)
                     OR perdio_acreditacion_sueldo = 1
                THEN 1
                ELSE 0
            END AS perdio_producto_principal,
            
            -- Caída de actividad severa (más del 50% de caída en transacciones)
            CASE 
                WHEN pct_cambio_transacciones < -0.5 
                THEN 1
                ELSE 0
            END AS caida_actividad_severa
            
        FROM {nombre_tabla}
    )
    """
    

    # --- 4. Ejecución sobre la base persistente ---
    logger.info(f"Conectando a la base DuckDB ({PATH_DATA_BASE_DB}) para ejecutar la consulta...")
    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)
        conn.execute(sql)
        conn.close()
    except Exception as e:
        logger.error(f"Error al ejecutar la consulta en DuckDB. Verifica las dependencias y la sintaxis SQL. Error: {e}")
        # Es fundamental propagar el error si ocurre una falla en la DB
        raise
    
    logger.info("Feature señales de abandono finalizada ✅")
    
    
    
    
def Feat_Eng_YB_Normalizacion_x_edad(nombre_tabla: str,PATH_DATA_BASE_DB: str = "ruta/a/tu_base.duckdb") -> None:
    """
    Genera automáticamente nuevas variables del tipo:
        variable_sobre_edad = variable / cliente_edad

    ✔ Lo aplica a TODAS las columnas numéricas
    ✔ Excluye cliente_edad
    ✔ Excluye columnas que ya existen con _sobre_edad
    ✔ Genera SQL dinámico
    ✔ Sobrescribe la tabla completa
    
    Basado en:   https://github.com/dmecoyfin/dmeyf2025/blob/main/src/zLightGBM/apo-506.ipynb
    """

    logger.info("=== Comienzo Normalización masiva sobre edad ===")

    # Validación
    if not isinstance(nombre_tabla, str) or not nombre_tabla.strip():
        raise ValueError("nombre_tabla debe ser un string válido.")

    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)

        # Obtenemos metadatos
        info = conn.execute(f"PRAGMA table_info('{nombre_tabla}')").df()

        # Columnas numéricas detectadas por tipo
        numeric_types = {"INTEGER", "BIGINT", "DOUBLE", "DECIMAL", "FLOAT", "REAL"}
        numeric_cols = [
            row["name"]
            for _, row in info.iterrows()
            if row["type"].upper() in numeric_types
        ]

        # Excluir edad y columnas ya creadas
        cols_to_process = [
            col for col in numeric_cols
            if col.lower() != "cliente_edad"
            and not col.lower().endswith("_sobre_edad")
        ]

        if len(cols_to_process) == 0:
            logger.info("No hay columnas numéricas para normalizar.")
            conn.close()
            return

        logger.info(f"Columnas que serán normalizadas sobre edad: {cols_to_process}")

        # Crear expresiones SQL dinámicamente
        nuevas_cols_sql = []
        for col in cols_to_process:
            new_col = f"{col}_sobre_edad"
            expr = (
                f"COALESCE({col}, 0)::DOUBLE / "
                f"NULLIF(cliente_edad, 0)::DOUBLE AS {new_col}"
            )
            nuevas_cols_sql.append(expr)

        nuevas_columnas_sql = ",\n        ".join(nuevas_cols_sql)

        # Consulta final: sobrescribe tabla agregando todas las columnas nuevas
        sql = f"""
        CREATE OR REPLACE TABLE {nombre_tabla} AS
        SELECT
            *,
            {nuevas_columnas_sql}
        FROM {nombre_tabla};
        """

        #logger.info(f"SQL generado:\n{sql}")
        conn.execute(sql)
        conn.close()

    except Exception as e:
        logger.error(f"Error normalizando columnas sobre edad: {e}")
        raise

    logger.info("=== Normalización masiva sobre edad completada correctamente ✅ ===")
    
def Feat_Eng_YB_Normalizacion_x_edad(
        nombre_tabla: str,
        columnas: list[str],
        PATH_DATA_BASE_DB: str = "ruta/a/tu_base.duckdb"
    ) -> None:
    """
    Genera automáticamente nuevas variables del tipo:
        variable_sobre_edad = variable / cliente_edad

    ✔ Ahora trabaja SOLO sobre las columnas indicadas en `columnas`
    ✔ Excluye cliente_edad aunque esté en la lista
    ✔ Excluye columnas que ya existen con _sobre_edad
    ✔ Genera SQL dinámico
    ✔ Sobrescribe la tabla completa
    """

    logger.info("=== Comienzo Normalización por edad sobre columnas seleccionadas ===")

    # Validaciones
    if not isinstance(nombre_tabla, str) or not nombre_tabla.strip():
        raise ValueError("nombre_tabla debe ser un string válido.")

    if not isinstance(columnas, (list, set, tuple)):
        raise ValueError("columnas debe ser una lista de strings.")

    try:
        conn = duckdb.connect(PATH_DATA_BASE_DB)

        # Columnas efectivamente presentes en la tabla
        info = conn.execute(f"PRAGMA table_info('{nombre_tabla}')").df()
        columnas_existentes = set(info["name"].tolist())

        # Filtrar columnas válidas
        cols_to_process = []

        for col in columnas:
            if col not in columnas_existentes:
                logger.warning(f"Columna {col} no existe en {nombre_tabla}, se omite.")
                continue

            if col.lower() == "cliente_edad":
                logger.warning("No se puede normalizar cliente_edad. Se omite.")
                continue

            if col.lower().endswith("_sobre_edad"):
                logger.warning(f"Columna {col} ya parece normalizada (_sobre_edad). Se omite.")
                continue

            cols_to_process.append(col)

        if len(cols_to_process) == 0:
            logger.info("No hay columnas para normalizar.")
            conn.close()
            return

        logger.info(f"Columnas que serán normalizadas sobre edad: {cols_to_process}")

        # Crear expresiones SQL dinámicamente
        nuevas_cols_sql = []
        for col in cols_to_process:
            new_col = f"{col}_sobre_edad"
            expr = (
                f"COALESCE({col}, 0)::DOUBLE / "
                f"NULLIF(cliente_edad, 0)::DOUBLE AS {new_col}"
            )
            nuevas_cols_sql.append(expr)

        nuevas_columnas_sql = ",\n        ".join(nuevas_cols_sql)

        # Consulta dinámica
        sql = f"""
        CREATE OR REPLACE TABLE {nombre_tabla} AS
        SELECT
            *,
            {nuevas_columnas_sql}
        FROM {nombre_tabla};
        """

        conn.execute(sql)
        conn.close()

    except Exception as e:
        logger.error(f"Error normalizando columnas sobre edad: {e}")
        raise

    logger.info("=== Normalización por edad completada correctamente ✅ ===")



    # === 
    # CORRO  EL PAQUETE DE FUNCOINES JUNTO 
    # ===



def ejecutar_todo_Feat_Eng_YB(PATH_DATA_BASE_DB, VENTANA):
    

    try:
        logger.info("Ejecutando Feat_Eng_YB_digitalizacion...")
        Feat_Eng_YB_digitalizacion('df_completo', PATH_DATA_BASE_DB)
        logger.info("Feat_Eng_YB_digitalizacion completado correctamente.")
    except Exception as e:
        logger.error(f"Error en Feat_Eng_YB_digitalizacion: {e}")
  

    try:
        logger.info("Ejecutando Feat_Eng_YB_fx_exposure...")
        Feat_Eng_YB_fx_exposure('df_completo', PATH_DATA_BASE_DB)
        logger.info("Feat_Eng_YB_fx_exposure completado correctamente.")
    except Exception as e:
        logger.error(f"Error en Feat_Eng_YB_fx_exposure: {e}")


    try:
        logger.info("Ejecutando Feat_Eng_YB_volatilidad...")
        Feat_Eng_YB_volatilidad('df_completo', VENTANA, PATH_DATA_BASE_DB)
        logger.info("Feat_Eng_YB_volatilidad completado correctamente.")
    except Exception as e:
        logger.error(f"Error en Feat_Eng_YB_volatilidad: {e}")

   
    try:
        logger.info("Ejecutando Feat_Eng_YB_macro_event_dummies...")
        Feat_Eng_YB_macro_event_dummies('df_completo', PATH_DATA_BASE_DB)
        
        logger.info("Feat_Eng_YB_macro_event_dummies completado correctamente.")
    except Exception as e:
        logger.error(f"Error en Feat_Eng_YB_macro_event_dummies: {e}")

    
    try:
        logger.info("Ejecutando Feat_Eng_YB_product_count...")
        Feat_Eng_YB_product_count('df_completo', PATH_DATA_BASE_DB)
        logger.info("Feat_Eng_YB_product_count completado correctamente.")
    except Exception as e:
        logger.error(f"Error en Feat_Eng_YB_product_count: {e}")

 
    try:
        logger.info("Ejecutando Feat_Eng_YB_rentabilidad_tendencias...")
        Feat_Eng_YB_rentabilidad_tendencias('df_completo', PATH_DATA_BASE_DB)
        logger.info("Feat_Eng_YB_rentabilidad_tendencias completado correctamente.")
    except Exception as e:
        logger.error(f"Error en Feat_Eng_YB_rentabilidad_tendencias: {e}")
    
    
    try:
        logger.info("Ejecutando Feat_Eng_YB_engagement_activity...")
        Feat_Eng_YB_engagement_activity('df_completo', PATH_DATA_BASE_DB)
        logger.info("Feat_Eng_YB_engagement_activity completado correctamente.")
    except Exception as e:
        logger.error(f"Error en Feat_Eng_YB_engagement_activity: {e}")
   

    try:
        logger.info("Ejecutando Feat_Eng_YB_indicios_de_abandono...")
        Feat_Eng_YB_indicios_de_abandono(PATH_DATA_BASE_DB)
        logger.info("Feat_Eng_YB_indicios_de_abandono completado correctamente.")
    except Exception as e:
        logger.error(f"Error en Feat_Eng_YB_indicios_de_abandono: {e}")
    
    
    
