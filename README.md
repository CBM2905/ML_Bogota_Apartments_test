## ML_Bogota_Apartments_test — README (en español)

## Resumen rápido
Proyecto de Machine Learning para análisis y modelado de precios / oportunidades de apartamentos en Bogotá. Contiene scripts para EDA, preprocesamiento, ingeniería de características, pipeline de entrenamiento/validación, y módulos para generar oportunidades/insights.

## Estructura del proyecto
- `README.md` — (este documento)
- `requirements.txt` — dependencias del proyecto.
- `data/` — datos brutos y/o preparados (CSV, parquet, etc.).
- `notebooks/` — notebooks exploratorios y experimentos.
- `results/` — artefactos de salida: modelos, métricas, figuras, reportes.
- `src/` — código fuente:
  - `__init__.py`
  - `eda.py` — análisis exploratorio de datos (visualizaciones, estadísticas).
  - `preprocessing.py` — limpieza y transformaciones base.
  - `features.py` — generación/selección de características.
  - `pipeline.py` — orquestador del flujo (ejecución secuencial del pipeline).
  - `model.py` — definición/entrenamiento/evaluación del modelo.
  - `opportunities.py` — lógica para detectar y exportar oportunidades (insights).
  - `model.py` y `pipeline.py` suelen producir y guardar artefactos en `results/`.


## Pipeline completo — pasos y dónde está cada parte
A continuación explico el pipeline de extremo a extremo y qué archivo(s) implementan cada paso.

1. Ingesta / carga de datos
	- Archivo: puede estar en `src/pipeline.py` o notebooks.
	- Qué hace: leer CSV/parquet desde `data/`, validar esquema mínimo, particionar train/test si aplica.

2. Análisis exploratorio (EDA)
	- Archivo: `src/eda.py` y notebooks en `notebooks/`
	- Qué hace: describir distribuciones, detectar outliers, visualizar correlaciones y relaciones precio vs. features.

3. Limpieza y preprocesamiento
	- Archivo: `src/preprocessing.py`
	- Qué hace: manejo de valores faltantes, corrección de tipos, normalización o transformaciones básicas (ej. convertir strings a categorías, parsear ubicaciones).

4. Ingeniería de características
	- Archivo: `src/features.py`
	- Qué hace: crear nuevas variables (precio/m2, distancia a puntos de interés si está disponible), codificar categóricas, seleccionar/filtrar features.

5. Pipeline de entrenamiento / orquestación
	- Archivo: `src/pipeline.py`
	- Qué hace: ejecutar en orden los pasos anteriores, entrenar modelos llamando a `src/model.py`, evaluar en conjunto de validación, guardar artefactos.

6. Modelado y evaluación
	- Archivo: `src/model.py`
	- Qué hace: definir modelos (ej. scikit-learn, XGBoost), hiperparámetros, cross-validation, métricas (MAE, RMSE, R^2). Guardar el mejor modelo.

7. Detección de oportunidades / Post-procesamiento
	- Archivo: `src/opportunities.py`
	- Qué hace: usar predicciones y reglas de negocio para identificar listings interesantes (p.ej. precio por debajo de predicción, inmuebles con potencial de rentabilidad).

8. Resultados y reporting
	- Archivos/Carpetas: `results/`, notebooks para reporte.
	- Qué hace: guardar métricas, gráficos y CSVs listos para análisis o dashboards.

isarlo/editarlo primero?

