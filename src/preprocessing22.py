"""
Módulo de preprocesamiento mejorado para Bogotá Apartments
Pipeline completo para preparación de datos para ML
"""

import pandas as pd
import numpy as np
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import re

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BogotaApartmentsPreprocessor:
    """Clase mejorada para preprocesamiento de datos inmobiliarios de Bogotá"""
    
    def __init__(self):
        self.original_shape = None
        self.final_shape = None
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # Mapeo para antigüedad
        self.antiguedad_map = {
            'MENOS DE 1 ANO': 0.5,
            'ENTRE 1 Y 5 ANOS': 3,
            'ENTRE 0 Y 5 ANOS': 2.5,
            'ENTRE 5 Y 10 ANOS': 7.5,
            'ENTRE 10 Y 20 ANOS': 15,
            'MAS DE 20 ANOS': 25,
            'EN CONSTRUCCION': 0,
            'ESTRENAR': 0
        }
    
    def load_excel_data(self, file_path):
        """
        Cargar datos desde archivo Excel con logging
        
        Args:
            file_path (str): Ruta al archivo Excel
            
        Returns:
            pd.DataFrame: DataFrame con los datos cargados
        """
        logger.info(f"📥 Cargando datos desde: {file_path}")
        
        try:
            df = pd.read_excel(file_path)
            self.original_shape = df.shape
            logger.info(f"✅ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
            return df
        except Exception as e:
            logger.error(f"❌ Error cargando Excel: {e}")
            raise
    
    def clean_numeric_columns(self, df):
        """
        Convertir columnas numéricas con comas a puntos y transformar a float
        
        Args:
            df (pd.DataFrame): DataFrame original
            
        Returns:
            pd.DataFrame: DataFrame con columnas numéricas limpias
        """
        logger.info("🔧 Limpiando columnas numéricas...")
        df_clean = df.copy()
        
        # Identificar columnas numéricas potenciales
        numeric_candidates = df_clean.select_dtypes(include=['object']).columns
        
        for col in numeric_candidates:
            # Verificar si la columna contiene números con comas
            if df_clean[col].astype(str).str.contains(r'\d+,\d+', na=False).any():
                try:
                    # Convertir comas a puntos y luego a numérico
                    df_clean[col] = (
                        df_clean[col]
                        .astype(str)
                        .str.replace(',', '.', regex=False)
                        .str.replace(r'[^\d.-]', '', regex=True)  # Remover caracteres no numéricos
                    )
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    logger.info(f"  ✅ Columna {col} convertida a numérica")
                except Exception as e:
                    logger.warning(f"  ⚠️ No se pudo convertir {col}: {e}")
        
        return df_clean
    
    def remove_useless_columns(self, df, threshold=0.5):
        """
        Eliminar columnas con más del threshold% de valores nulos
        
        Args:
            df (pd.DataFrame): DataFrame a limpiar
            threshold (float): Umbral de nulos para eliminar (0.5 = 50%)
            
        Returns:
            pd.DataFrame: DataFrame sin columnas inútiles
        """
        logger.info("🗑️ Eliminando columnas con muchos valores nulos...")
        df_clean = df.copy()
        
        null_ratio = df_clean.isnull().sum() / len(df_clean)
        columns_to_drop = null_ratio[null_ratio > threshold].index.tolist()
        
        if columns_to_drop:
            df_clean = df_clean.drop(columns=columns_to_drop)
            logger.info(f"  ✅ Columnas eliminadas: {columns_to_drop}")
        else:
            logger.info("  ℹ️ No se encontraron columnas con más del 50% de valores nulos")
        
        return df_clean
    
    def filter_valid_records(self, df):
        """
        Filtrar registros válidos según criterios específicos
        
        Args:
            df (pd.DataFrame): DataFrame a filtrar
            
        Returns:
            pd.DataFrame: DataFrame filtrado
        """
        logger.info("🎯 Filtrando registros válidos...")
        df_filtered = df.copy()
        initial_count = len(df_filtered)
        
        # Aplicar filtros secuencialmente
        filters_applied = 0
        
        if 'precio_venta' in df_filtered.columns:
            mask = df_filtered['precio_venta'] > 0
            df_filtered = df_filtered[mask]
            filters_applied += 1
            logger.info(f"  ✅ Filtro precio_venta > 0: {len(df_filtered)} registros")
        
        if 'area' in df_filtered.columns:
            mask = df_filtered['area'] > 0
            df_filtered = df_filtered[mask]
            filters_applied += 1
            logger.info(f"  ✅ Filtro area > 0: {len(df_filtered)} registros")
        
        if 'latitud' in df_filtered.columns:
            mask = df_filtered['latitud'].between(4.4, 4.9)
            df_filtered = df_filtered[mask]
            filters_applied += 1
            logger.info(f"  ✅ Filtro latitud [4.4, 4.9]: {len(df_filtered)} registros")
        
        if 'longitud' in df_filtered.columns:
            mask = df_filtered['longitud'].between(-74.3, -74.0)
            df_filtered = df_filtered[mask]
            filters_applied += 1
            logger.info(f"  ✅ Filtro longitud [-74.3, -74.0]: {len(df_filtered)} registros")
        
        final_count = len(df_filtered)
        retention_rate = (final_count / initial_count) * 100
        logger.info(f"📊 Retención después de filtrado: {final_count}/{initial_count} ({retention_rate:.1f}%)")
        
        return df_filtered
    
    def impute_missing_values(self, df):
        """
        Imputar valores faltantes según tipo de variable
        
        Args:
            df (pd.DataFrame): DataFrame con valores faltantes
            
        Returns:
            pd.DataFrame: DataFrame sin valores faltantes
        """
        logger.info("🔄 Imputando valores faltantes...")
        df_imputed = df.copy()
        
        # Separar columnas numéricas y categóricas
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns
        
        # Imputar numéricas con mediana
        for col in numeric_cols:
            if df_imputed[col].isnull().any():
                median_val = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(median_val)
                logger.info(f"  ✅ Numérica {col}: imputado con mediana {median_val:.2f}")
        
        # Imputar categóricas con moda o "DESCONOCIDO"
        for col in categorical_cols:
            if df_imputed[col].isnull().any():
                if df_imputed[col].notna().any():  # Si hay al menos un valor no nulo
                    mode_val = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else "DESCONOCIDO"
                    df_imputed[col] = df_imputed[col].fillna(mode_val)
                    logger.info(f"  ✅ Categórica {col}: imputado con moda '{mode_val}'")
                else:  # Si toda la columna es nula
                    df_imputed[col] = df_imputed[col].fillna("DESCONOCIDO")
                    logger.info(f"  ✅ Categórica {col}: imputado con 'DESCONOCIDO'")
        
        remaining_nulls = df_imputed.isnull().sum().sum()
        if remaining_nulls == 0:
            logger.info("✅ Todos los valores faltantes han sido imputados")
        else:
            logger.warning(f"⚠️ Aún quedan {remaining_nulls} valores nulos")
        
        return df_imputed
    
    def transform_skewed_variables(self, df):
        """
        Aplicar transformación log1p a variables sesgadas
        
        Args:
            df (pd.DataFrame): DataFrame con variables sesgadas
            
        Returns:
            pd.DataFrame: DataFrame con variables transformadas
        """
        logger.info("📈 Transformando variables sesgadas con log1p...")
        df_transformed = df.copy()
        
        skewed_vars = ['precio_venta', 'area', 'administracion', 'precio_arriendo']
        
        for var in skewed_vars:
            if var in df_transformed.columns:
                # Verificar que no haya valores negativos antes de aplicar log
                if (df_transformed[var] < 0).any():
                    logger.warning(f"  ⚠️ {var} tiene valores negativos, ajustando...")
                    min_val = df_transformed[var].min()
                    if min_val < 0:
                        df_transformed[var] = df_transformed[var] - min_val + 1
                
                df_transformed[f'log_{var}'] = np.log1p(df_transformed[var])
                logger.info(f"  ✅ {var} transformado a log_{var}")
        
        return df_transformed
    
    def handle_outliers(self, df):
        """
        Aplicar winsorization a variables con outliers
        
        Args:
            df (pd.DataFrame): DataFrame con outliers
            
        Returns:
            pd.DataFrame: DataFrame con outliers tratados
        """
        logger.info("📊 Manejo de outliers con winsorization...")
        df_clean = df.copy()
        
        # Winsorize precio_venta
        if 'precio_venta' in df_clean.columns:
            df_clean['precio_venta'] = mstats.winsorize(
                df_clean['precio_venta'], limits=[0.01, 0.01]
            )
            logger.info("  ✅ precio_venta winsorizado (1%-99%)")
        
        # Winsorize precio_m2 si existe
        if 'precio_m2' in df_clean.columns:
            df_clean['precio_m2'] = mstats.winsorize(
                df_clean['precio_m2'], limits=[0.01, 0.01]
            )
            logger.info("  ✅ precio_m2 winsorizado (1%-99%)")
        
        return df_clean
    
    def preprocess_antiguedad(self, df):
        """
        Convertir categorías de antigüedad a valores numéricos
        
        Args:
            df (pd.DataFrame): DataFrame con columna antiguedad
            
        Returns:
            pd.DataFrame: DataFrame con antiguedad numérica
        """
        logger.info("🕒 Preprocesando antigüedad...")
        df_processed = df.copy()
        
        if 'antiguedad' in df_processed.columns:
            # Mapear categorías a valores numéricos
            df_processed['antiguedad_num'] = (
                df_processed['antiguedad']
                .str.upper()
                .map(self.antiguedad_map)
            )
            
            # Imputar valores no mapeados con la mediana
            median_antiguedad = df_processed['antiguedad_num'].median()
            df_processed['antiguedad_num'] = df_processed['antiguedad_num'].fillna(median_antiguedad)
            
            logger.info(f"  ✅ Antigüedad convertida a numérica (mediana: {median_antiguedad})")
        
        return df_processed
    
    def create_features(self, df):
        """
        Ingeniería de características avanzada
        
        Args:
            df (pd.DataFrame): DataFrame base
            
        Returns:
            pd.DataFrame: DataFrame con nuevas características
        """
        logger.info("🛠️ Creando características de ingeniería...")
        df_enhanced = df.copy()
        
        # 1. Precio por m2
        if all(col in df_enhanced.columns for col in ['precio_venta', 'area']):
            df_enhanced['precio_m2'] = df_enhanced['precio_venta'] / df_enhanced['area']
            logger.info("  ✅ precio_m2 creado")
        
        # 2. Score de amenities
        amenities_pattern = r'jacuzzi|gimnasio|ascensor|conjunto_cerrado|piscina|salon_comunal|terraza|vigilancia|chimenea|mascotas'
        amenities_cols = [col for col in df_enhanced.columns if re.search(amenities_pattern, col, re.IGNORECASE)]
        
        if amenities_cols:
            # Convertir a binario y sumar
            for col in amenities_cols:
                df_enhanced[col] = pd.to_numeric(df_enhanced[col], errors='coerce').fillna(0)
            
            df_enhanced['amenities_score'] = df_enhanced[amenities_cols].sum(axis=1)
            logger.info(f"  ✅ amenities_score creado con {len(amenities_cols)} amenities")
        
        # 3. Ratios de densidad
        if 'area' in df_enhanced.columns:
            if 'banos' in df_enhanced.columns:
                df_enhanced['banos_por_area'] = df_enhanced['banos'] / df_enhanced['area']
            
            if 'habitaciones' in df_enhanced.columns:
                df_enhanced['habitaciones_por_area'] = df_enhanced['habitaciones'] / df_enhanced['area']
            
            if 'parqueaderos' in df_enhanced.columns:
                df_enhanced['parqueaderos_por_area'] = df_enhanced['parqueaderos'] / df_enhanced['area']
            
            logger.info("  ✅ Ratios por área creados")
        
        # 4. Indicadores de lujo
        if all(col in df_enhanced.columns for col in ['precio_m2', 'localidad']):
            # Calcular mediana de precio_m2 por localidad
            mediana_localidad = df_enhanced.groupby('localidad')['precio_m2'].transform('median')
            df_enhanced['indicador_lujo'] = (df_enhanced['precio_m2'] > mediana_localidad).astype(int)
            logger.info("  ✅ Indicador de lujo creado")
        
        # 5. Estrato + localidad
        if all(col in df_enhanced.columns for col in ['estrato', 'localidad']):
            df_enhanced['estrato_localidad'] = (
                df_enhanced['estrato'].astype(str) + "_" + df_enhanced['localidad']
            )
            logger.info("  ✅ estrato_localidad creado")
        
        return df_enhanced
    
    def encode_categorical_variables(self, df):
        """
        Codificar variables categóricas (one-hot y target encoding)
        
        Args:
            df (pd.DataFrame): DataFrame con variables categóricas
            
        Returns:
            pd.DataFrame: DataFrame con variables codificadas
        """
        logger.info("🔠 Codificando variables categóricas...")
        df_encoded = df.copy()
        
        # Agrupar localidades minoritarias
        if 'localidad' in df_encoded.columns:
            localidad_counts = df_encoded['localidad'].value_counts()
            minor_localidades = localidad_counts[localidad_counts < 10].index
            df_encoded['localidad'] = df_encoded['localidad'].replace(minor_localidades, 'OTROS')
            logger.info(f"  ✅ Localidades agrupadas: {len(minor_localidades)} en 'OTROS'")
        
        # One-hot encoding para variables con pocas categorías
        low_cardinality_vars = []
        for col in df_encoded.select_dtypes(include=['object']).columns:
            if df_encoded[col].nunique() <= 10:
                low_cardinality_vars.append(col)
        
        if low_cardinality_vars:
            df_encoded = pd.get_dummies(df_encoded, columns=low_cardinality_vars, prefix=low_cardinality_vars)
            logger.info(f"  ✅ One-hot encoding aplicado a: {low_cardinality_vars}")
        
        # Para variables de alta cardinalidad, usar frecuencia encoding
        high_cardinality_vars = ['barrio']  # Ejemplo
        for col in high_cardinality_vars:
            if col in df_encoded.columns:
                freq_encoding = df_encoded[col].value_counts(normalize=True)
                df_encoded[f'{col}_freq_encoded'] = df_encoded[col].map(freq_encoding)
                logger.info(f"  ✅ Frecuencia encoding aplicado a: {col}")
        
        return df_encoded
    
    def scale_numeric_variables(self, df):
        """
        Escalar variables numéricas para ML
        
        Args:
            df (pd.DataFrame): DataFrame con variables numéricas
            
        Returns:
            pd.DataFrame: DataFrame con variables escaladas
        """
        logger.info("⚖️ Escalando variables numéricas...")
        df_scaled = df.copy()
        
        # Identificar columnas numéricas (excluyendo las que ya fueron transformadas)
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        
        # Excluir columnas que ya están en escala logarítmica o son binarias
        exclude_patterns = ['log_', '_encoded', 'indicador_', 'dummy_']
        cols_to_scale = [
            col for col in numeric_cols 
            if not any(pattern in col for pattern in exclude_patterns)
            and df_scaled[col].nunique() > 2  # Excluir variables binarias
        ]
        
        if cols_to_scale:
            # Aplicar StandardScaler
            scaled_values = self.scaler.fit_transform(df_scaled[cols_to_scale])
            df_scaled[cols_to_scale] = scaled_values
            
            # Crear nombres para las columnas escaladas
            scaled_cols = [f'scaled_{col}' for col in cols_to_scale]
            df_scaled[scaled_cols] = scaled_values
            
            logger.info(f"  ✅ {len(cols_to_scale)} variables escaladas con StandardScaler")
        
        return df_scaled
    
    def run_complete_pipeline(self, file_path):
        """
        Ejecutar pipeline completo de preprocesamiento
        
        Args:
            file_path (str): Ruta al archivo Excel
            
        Returns:
            pd.DataFrame: DataFrame preprocesado listo para ML
        """
        logger.info("🚀 INICIANDO PIPELINE COMPLETO DE PREPROCESAMIENTO")
        
        try:
            # 1. Carga de datos
            df = self.load_excel_data(file_path)
            
            # 2. Limpieza de columnas numéricas
            df = self.clean_numeric_columns(df)
            
            # 3. Eliminación de columnas inútiles
            df = self.remove_useless_columns(df)
            
            # 4. Filtrado de registros válidos
            df = self.filter_valid_records(df)
            
            # 5. Imputación de valores faltantes
            df = self.impute_missing_values(df)
            
            # 6. Transformación de variables sesgadas
            df = self.transform_skewed_variables(df)
            
            # 7. Manejo de outliers
            df = self.handle_outliers(df)
            
            # 8. Preprocesamiento de antigüedad
            df = self.preprocess_antiguedad(df)
            
            # 9. Ingeniería de características
            df = self.create_features(df)
            
            # 10. Codificación de variables categóricas
            df = self.encode_categorical_variables(df)
            
            # 11. Escalado de variables numéricas
            df = self.scale_numeric_variables(df)
            
            self.final_shape = df.shape
            logger.info(f"🎉 PIPELINE COMPLETADO EXITOSAMENTE")
            logger.info(f"📊 RESUMEN: {self.original_shape[0]} → {self.final_shape[0]} registros")
            logger.info(f"📊 RESUMEN: {self.original_shape[1]} → {self.final_shape[1]} columnas")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error en el pipeline: {e}")
            raise

# Función de conveniencia para uso rápido
def preprocess_bogota_apartments(file_path):
    """
    Función simple para ejecutar el pipeline completo
    
    Args:
        file_path (str): Ruta al archivo Excel
        
    Returns:
        pd.DataFrame: DataFrame preprocesado
    """
    processor = BogotaApartmentsPreprocessor()
    return processor.run_complete_pipeline(file_path)

if __name__ == "__main__":
    # Ejemplo de uso
    sample_file = "bogota_apartments.xlsx"
    try:
        processed_data = preprocess_bogota_apartments(sample_file)
        print(f"✅ Datos preprocesados: {processed_data.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")