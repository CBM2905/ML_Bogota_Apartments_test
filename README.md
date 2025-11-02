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

## Contrato mínimo (inputs / outputs / errores)
- Inputs:
  - Datos crudos en `data/` (CSV/parquet). Formato esperado: filas = anuncios, columnas = features como precio, área, ubicación, habitaciones, etc.
- Outputs:
  - Modelos entrenados en `results/models/`
  - Reportes y métricas en `results/metrics/` y `results/figures/`
  - CSV con oportunidades en `results/opportunities/`
- Errores típicos:
  - Formato de columnas cambiado (columnas faltantes o renombradas).
  - Valores NaN en columnas críticas sin manejo.
  - Versiones de dependencias incompatibles (ver `requirements.txt`).

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

## Cómo ejecutar (Windows PowerShell)
Sugerencia de flujo local rápido. Ajusta rutas/variables según tu entorno.

1. Crear y activar entorno virtual:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:
```powershell
pip install -r requirements.txt
```

3. Ejecutar el pipeline principal (si `src/pipeline.py` contiene un main):
```powershell
python .\src\pipeline.py
# o si el paquete es ejecutable:
python -m src.pipeline
```

4. Ejecutar notebooks (opcional):
- Abre `jupyter lab` o `jupyter notebook` y carga los notebooks en `notebooks/` para reproducir EDA/experimentos.

5. Revisar outputs:
- Modelos y artefactos en `results/`.
- Métricas en `results/metrics/`.
- Oportunidades exportadas en `results/opportunities/`.

Nota: Si `src/pipeline.py` requiere argumentos, revisa su docstring o abre el archivo para ver la interfaz (por ejemplo `--data-path`, `--output-dir`, `--mode=train|predict`).

## Comprobaciones y calidad (recomendado)
- Lint/format: usar `flake8` o `ruff` y `black`.
- Tests: crear tests unitarios (pytest) para:
  - Funciones de `preprocessing.py` (manejando NaNs).
  - Generación de features (`features.py`).
  - Rutas principales de `pipeline.py` (mockear I/O).
- CI: agregar GitHub Actions con pasos: install, lint, test.

## Contrato de funciones y casos límite (breve)
- Inputs esperados: DataFrame/payload con columnas mínimas: `price`, `area`, `bedrooms`, `location`.
- Outputs esperados: DataFrame con columna `predicted_price` y métricas (MAE).
- Casos límite:
  - Filas con `area == 0` o nulas → filtrar o imputar.
  - Valores atípicos extremadamente altos → winsorize o truncar.
  - Columnas categóricas con cardinalidad alta → agrupar categorías raras.

## Ejemplo de flujo (resumen paso a paso)
1. Colocar raw data en `data/`.
2. Correr `python .\src\pipeline.py` (ingesta → preprocessing → features → train).
3. Revisar `results/` para métricas y modelo.
4. Ejecutar `python .\src\opportunities.py` (o función equivalente) para generar CSVs de oportunidades.

## Buenas prácticas y recomendaciones
- Versiona los datos importantes o usa hashes para trazabilidad.
- Usa `mlflow` o `tensorboard` para tracking de experimentos.
- Guarda metadatos del modelo (versión, fecha, métricas) junto al artefacto.
- Serializa transformadores (por ejemplo `sklearn` Pipelines) junto al modelo para reproducibilidad.

## Próximos pasos sugeridos (rápido)
- Añadir tests básicos (pytest) y configurarlos en CI.
- Documentar las funciones públicas en `src/` con docstrings.
- Crear scripts CLI más robustos (click/argparse) en `src/pipeline.py`.
- Preparar un notebook de demo en `notebooks/` con un "runbook" paso a paso.

## Resumen y verificación
- Qué hice: Redacté el README en español y mapeé el pipeline completo con los archivos relevantes del repositorio (`src/`).
- No modifiqué archivos en el repo; te entrego el contenido listo para copiar a `README.md`.
- Próxima acción que puedo hacer si quieres: 1) crear el `README.md` en el repo, 2) añadir un `Makefile` o scripts de ejecución, 3) implementar tests básicos. Indica cuál prefieres y lo hago.

¿Quieres que agregue este `README.md` directamente al repositorio (lo creo en la raíz) o prefieres revisarlo/editarlo primero?

