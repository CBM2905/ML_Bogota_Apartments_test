

# %% [markdown]
# ## 🔧 MÓDULO: preprocessing.py

# %%


"""
Módulo de preprocesamiento para Bogotá Apartments
Optimizado para datos en formato Excel
"""

import pandas as pd
import numpy as np
import re

class DataPreprocessor:
    """Clase para preprocesamiento de datos inmobiliarios desde Excel"""
    
    def __init__(self):
        self.essential_columns = [
            'codigo', 'tipo_propiedad', 'precio_venta', 'area', 'habitaciones', 
            'banos', 'administracion', 'parqueaderos', 'sector', 'estrato', 
            'antiguedad', 'latitud', 'longitud', 'localidad', 'barrio'
        ]
        
        self.amenities_columns = [
            'jacuzzi', 'gimnasio', 'ascensor', 'conjunto_cerrado', 'piscina',
            'salon_comunal', 'terraza', 'vigilancia', 'chimenea', 'permite_mascotas'
        ]
    
    def load_excel_data(self, file_path):
        """
        Cargar datos desde archivo Excel
        
        Args:
            file_path (str): Ruta al archivo Excel
            
        Returns:
            pd.DataFrame: DataFrame con los datos
        """
        print(f"📥 Cargando datos desde: {file_path}")
        
        try:
            df = pd.read_excel(file_path)
            print(f"✅ Excel cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
            return df
        except Exception as e:
            print(f"❌ Error cargando Excel: {e}")
            return pd.DataFrame()
    
    def clean_numeric_columns(self, df):
        """
        Limpiar columnas numéricas (manejar comas como decimales)
        
        Args:
            df (pd.DataFrame): DataFrame original
            
        Returns:
            pd.DataFrame: DataFrame con columnas numéricas limpias
        """
        df_clean = df.copy()
        
        # Columnas que pueden tener comas como separador decimal
        numeric_columns = ['latitud', 'longitud', 'distancia_estacion_tm_m', 'distancia_parque_m']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                # Convertir comas a puntos y luego a numérico
                df_clean[col] = df_clean[col].astype(str).str.replace(',', '.')
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                print(f"✅ Columna {col} convertida a numérica")
        
        return df_clean
    
    def handle_missing_values(self, df):
        """
        Manejar valores faltantes de forma robusta
        
        Args:
            df (pd.DataFrame): DataFrame con valores faltantes
            
        Returns:
            pd.DataFrame: DataFrame sin valores faltantes
        """
        df_clean = df.copy()
        
        # Estrategias por tipo de columna
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        print(f"🔧 Imputando {len(numeric_cols)} variables numéricas...")
        print(f"🔧 Imputando {len(categorical_cols)} variables categóricas...")
        
        # Imputar numéricas con mediana
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Imputar categóricas con moda
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                if df_clean[col].notna().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                else:
                    df_clean[col] = df_clean[col].fillna('DESCONOCIDO')
        
        remaining_nulls = df_clean.isnull().sum().sum()
        print(f"✅ Valores nulos restantes: {remaining_nulls}")
        
        return df_clean
    
    def filter_valid_records(self, df):
        """
        Filtrar registros válidos para el modelo
        
        Args:
            df (pd.DataFrame): DataFrame a filtrar
            
        Returns:
            pd.DataFrame: DataFrame filtrado
        """
        df_filtered = df.copy()
        initial_count = len(df_filtered)
        
        # Filtros esenciales
        masks = []
        
        if 'precio_venta' in df_filtered.columns:
            masks.append(df_filtered['precio_venta'] > 0)
        
        if 'area' in df_filtered.columns:
            masks.append(df_filtered['area'] > 0)
        
        if 'latitud' in df_filtered.columns and 'longitud' in df_filtered.columns:
            masks.append(df_filtered['latitud'].between(4.4, 4.9))
            masks.append(df_filtered['longitud'].between(-74.3, -74.0))
        
        # Aplicar todos los filtros
        if masks:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask &= mask
            df_filtered = df_filtered[combined_mask]
        
        final_count = len(df_filtered)
        print(f"✅ Registros filtrados: {final_count}/{initial_count} ({final_count/initial_count*100:.1f}%)")
        
        return df_filtered
    
    def preprocess_antiguedad(self, df):
        """
        Convertir antigüedad categórica a numérica
        
        Args:
            df (pd.DataFrame): DataFrame con columna antiguedad
            
        Returns:
            pd.DataFrame: DataFrame con antiguedad numérica
        """
        df_clean = df.copy()
        
        if 'antiguedad' in df_clean.columns:
            antiguedad_map = {
                'ENTRE 0 Y 5 ANOS': 2.5,
                'ENTRE 5 Y 10 ANOS': 7.5, 
                'ENTRE 10 Y 20 ANOS': 15,
                'MAS DE 20 ANOS': 25,
                'MENOS DE 1 ANO': 0.5
            }
            
            df_clean['antiguedad_num'] = df_clean['antiguedad'].map(antiguedad_map)
            df_clean['antiguedad_num'] = df_clean['antiguedad_num'].fillna(10)
            
            print("✅ Antigüedad convertida a numérica")
        
        return df_clean
    
    def create_essential_features(self, df):
        """
        Crear características esenciales para el modelo
        
        Args:
            df (pd.DataFrame): DataFrame base
            
        Returns:
            pd.DataFrame: DataFrame con características adicionales
        """
        df_enhanced = df.copy()
        
        # Precio por m2
        if all(col in df_enhanced.columns for col in ['precio_venta', 'area']):
            df_enhanced['precio_m2'] = df_enhanced['precio_venta'] / df_enhanced['area']
        
        # Score de amenities
        available_amenities = [col for col in self.amenities_columns if col in df_enhanced.columns]
        if available_amenities:
            for col in available_amenities:
                df_enhanced[col] = pd.to_numeric(df_enhanced[col], errors='coerce').fillna(0)
            df_enhanced['amenities_score'] = df_enhanced[available_amenities].sum(axis=1)
        
        # Estrato localidad
        if all(col in df_enhanced.columns for col in ['estrato', 'localidad']):
            df_enhanced['estrato_localidad'] = df_enhanced['estrato'].astype(str) + "_" + df_enhanced['localidad']
        
        print("✅ Características esenciales creadas")
        return df_enhanced
    
    def run_pipeline(self, file_path):
        """
        Ejecutar pipeline completo de preprocesamiento
        
        Args:
            file_path (str): Ruta al archivo Excel
            
        Returns:
            pd.DataFrame: DataFrame preprocesado
        """
        print("🚀 Iniciando pipeline de preprocesamiento...")
        
        # 1. Cargar datos
        df = self.load_excel_data(file_path)
        if df.empty:
            return df
        
        # 2. Limpiar columnas numéricas
        df = self.clean_numeric_columns(df)
        
        # 3. Filtrar registros válidos
        df = self.filter_valid_records(df)
        
        # 4. Preprocesar antigüedad
        df = self.preprocess_antiguedad(df)
        
        # 5. Crear características
        df = self.create_essential_features(df)
        
        # 6. Manejar valores faltantes
        df = self.handle_missing_values(df)
        
        print(f"✅ Pipeline completado: {df.shape[0]} registros, {df.shape[1]} columnas")
        return df

# Función de conveniencia
def preprocess_data(file_path):
    """Función simple para preprocesar datos"""
    processor = DataPreprocessor()
    return processor.run_pipeline(file_path)