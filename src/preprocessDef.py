import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

class BogotaApartmentsPreprocessor:
    """
    Pipeline completo de preprocesamiento para dataset de apartamentos en Bogotá
    Incluye exportación a Excel del dataset preprocesado
    """
    
    def __init__(self):
        self.numeric_features = ['area', 'habitaciones', 'banos', 'estrato', 
                               'parqueaderos', 'administracion', 'latitud', 'longitud']
        self.categorical_features = ['localidad', 'barrio', 'antiguedad', 'tipo_propiedad']
        self.target = 'precio_venta'
        self.preprocessor = None
        self.is_fitted = False
        self.processed_df = None
        
    def filter_inconsistent_records(self, df):
        """Filtrado de registros inconsistentes"""
        print("🎯 Filtrando registros inconsistentes...")
        initial_count = len(df)
        
        # Eliminar filas con precio_venta faltante
        df = df.dropna(subset=[self.target])
        
        # Aplicar filtros de calidad
        filters = (
            (df['area'] > 0) &
            (df['estrato'].between(1, 6)) &
            (df['latitud'].between(4.55, 4.85)) &
            (df['longitud'].between(-74.2, -74.0))
        )
        
        df_filtered = df[filters].copy()
        final_count = len(df_filtered)
        
        print(f"✅ Registros después de filtrado: {final_count}/{initial_count} "
              f"({final_count/initial_count*100:.1f}%)")
        
        return df_filtered
    
    def remove_price_arriendo(self, df):
        """Eliminar columna precio_arriendo"""
        if 'precio_arriendo' in df.columns:
            df = df.drop(columns=['precio_arriendo'])
            print("✅ Columna precio_arriendo eliminada")
        return df
    
    def handle_missing_values(self, df):
        """Manejo de valores faltantes"""
        print("🔄 Manejo de valores faltantes...")
        
        # Barrio: reemplazar con "Desconocido"
        if 'barrio' in df.columns:
            df['barrio'] = df['barrio'].fillna('Desconocido')
            print("✅ Barrio: valores faltantes reemplazados con 'Desconocido'")
        
        # Antiguedad: imputar con moda
        if 'antiguedad' in df.columns:
            mode_antiguedad = df['antiguedad'].mode()[0] if not df['antiguedad'].mode().empty else 'Desconocido'
            df['antiguedad'] = df['antiguedad'].fillna(mode_antiguedad)
            print(f"✅ Antiguedad: valores faltantes imputados con moda '{mode_antiguedad}'")
        
        return df
    
    def create_engineered_features(self, df):
        """Ingeniería de características"""
        print("🛠️ Creando características de ingeniería...")
        
        # Precio por m2
        if all(col in df.columns for col in ['precio_venta', 'area']):
            df['precio_m2'] = df['precio_venta'] / df['area']
            print("✅ precio_m2 creado")
        
        # Ratios de densidad
        if 'area' in df.columns:
            if 'banos' in df.columns:
                df['banos_por_area'] = df['banos'] / df['area']
            
            if 'habitaciones' in df.columns:
                df['habitaciones_por_area'] = df['habitaciones'] / df['area']
            
            print("✅ Ratios por área creados")
        
        # Score de amenities
        amenities_cols = [col for col in df.columns if any(amenity in col.lower() 
                         for amenity in ['jacuzzi', 'gimnasio', 'piscina', 'ascensor', 
                                       'conjunto_cerrado', 'salon_comunal', 'terraza', 
                                       'vigilancia', 'chimenea', 'mascotas'])]
        
        if amenities_cols:
            for col in amenities_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['amenities_score'] = df[amenities_cols].sum(axis=1)
            print(f"✅ amenities_score creado con {len(amenities_cols)} amenities")
        
        return df
    
    def filter_extreme_outliers(self, df, columns, lower_percentile=1, upper_percentile=99):
        """Filtrar outliers extremos usando IQR"""
        print("📊 Filtrado de outliers extremos...")
        initial_count = len(df)
        
        for col in columns:
            if col in df.columns:
                lower_bound = df[col].quantile(lower_percentile / 100)
                upper_bound = df[col].quantile(upper_percentile / 100)
                
                # Filtrar outliers
                mask = df[col].between(lower_bound, upper_bound)
                df = df[mask]
                
                outliers_removed = initial_count - len(df)
                print(f"✅ {col}: {outliers_removed} outliers removidos "
                      f"({lower_percentile}%-{upper_percentile}%)")
        
        final_count = len(df)
        print(f"📈 Registros después de filtrado de outliers: {final_count}/{initial_count} "
              f"({final_count/initial_count*100:.1f}%)")
        
        return df

    def apply_log_transformation(self, df, columns):
        """Aplicar transformación log1p a columnas específicas"""
        print("📈 Aplicando transformación logarítmica...")
        df_transformed = df.copy()
        
        for col in columns:
            if col in df_transformed.columns:
                # Asegurar que no hay valores negativos
                min_val = df_transformed[col].min()
                if min_val <= 0:
                    df_transformed[col] = df_transformed[col] - min_val + 1
                df_transformed[f'log_{col}'] = np.log1p(df_transformed[col])
                print(f"✅ {col} transformado a log_{col}")
        
        return df_transformed

    def truncate_outliers(self, df, columns, percentile=99):
        """Truncar outliers en percentil específico"""
        print("📊 Truncando outliers...")
        df_truncated = df.copy()
        
        for col in columns:
            if col in df_truncated.columns:
                trunc_value = df_truncated[col].quantile(percentile / 100)
                df_truncated[col] = np.where(df_truncated[col] > trunc_value, trunc_value, df_truncated[col])
                print(f"✅ {col} truncado en percentil {percentile}%")
        
        return df_truncated

    def apply_target_encoding(self, df, target_col):
        """Aplicar Target Encoding a localidad"""
        print("🔤 Aplicando Target Encoding a localidad...")
        df_encoded = df.copy()
        
        if 'localidad' in df_encoded.columns:
            # Calcular promedio de precio_venta por localidad
            target_means = df_encoded.groupby('localidad')[target_col].mean().to_dict()
            df_encoded['localidad_encoded'] = df_encoded['localidad'].map(target_means)
            
            # Imputar con media global si hay valores nulos
            global_mean = df_encoded[target_col].mean()
            df_encoded['localidad_encoded'] = df_encoded['localidad_encoded'].fillna(global_mean)
            
            print("✅ Target Encoding aplicado a localidad")
        
        return df_encoded

    def apply_frequency_encoding(self, df, threshold=0.01):
        """Aplicar Frequency Encoding a barrio con agrupación de categorías raras"""
        print("🔤 Aplicando Frequency Encoding a barrio...")
        df_encoded = df.copy()
        
        if 'barrio' in df_encoded.columns:
            # Calcular frecuencias
            freq_series = df_encoded['barrio'].value_counts(normalize=True)
            
            # Identificar categorías a mantener (frecuencia > threshold)
            categories_to_keep = set(freq_series[freq_series > threshold].index)
            
            # Agrupar categorías raras
            df_encoded['barrio_grouped'] = df_encoded['barrio'].apply(
                lambda x: x if x in categories_to_keep else 'Otros'
            )
            
            # Aplicar frequency encoding
            freq_map = df_encoded['barrio_grouped'].value_counts(normalize=True).to_dict()
            df_encoded['barrio_freq_encoded'] = df_encoded['barrio_grouped'].map(freq_map)
            
            print(f"✅ Frequency Encoding aplicado a barrio ({len(categories_to_keep)} categorías conservadas)")
        
        return df_encoded

    def apply_onehot_encoding(self, df, columns):
        """Aplicar One-Hot Encoding a columnas categóricas"""
        print("🔤 Aplicando One-Hot Encoding...")
        df_encoded = df.copy()
        
        for col in columns:
            if col in df_encoded.columns:
                # Aplicar one-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                # Eliminar columna original
                df_encoded = df_encoded.drop(columns=[col])
                
                print(f"✅ One-Hot Encoding aplicado a {col} ({len(dummies.columns)} categorías)")
        
        return df_encoded

    def scale_numeric_features(self, df, columns):
        """Escalar características numéricas con RobustScaler"""
        print("⚖️ Escalando características numéricas...")
        df_scaled = df.copy()
        
        scaler = RobustScaler()
        
        for col in columns:
            if col in df_scaled.columns:
                # Escalar y mantener en nueva columna
                scaled_values = scaler.fit_transform(df_scaled[[col]])
                df_scaled[f'scaled_{col}'] = scaled_values
                print(f"✅ {col} escalado con RobustScaler")
        
        return df_scaled

    def save_processed_excel(self, df, output_path):
        """Guardar DataFrame procesado en Excel"""
        print(f"💾 Guardando datos procesados en: {output_path}")
        
        # Crear Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Guardar dataset completo
            df.to_excel(writer, sheet_name='Datos_Procesados', index=False)
            
            # Crear hoja de resumen
            summary_data = {
                'Métrica': [
                    'Total Registros',
                    'Total Características',
                    'Características Numéricas',
                    'Características Categóricas',
                    'Variable Objetivo',
                    'Precio Promedio',
                    'Área Promedio'
                ],
                'Valor': [
                    len(df),
                    len(df.columns),
                    len([col for col in df.columns if df[col].dtype in ['int64', 'float64']]),
                    len([col for col in df.columns if df[col].dtype == 'object']),
                    'precio_venta',
                    f"${df['precio_venta'].mean():,.0f}" if 'precio_venta' in df.columns else 'N/A',
                    f"{df['area'].mean():.1f} m²" if 'area' in df.columns else 'N/A'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Resumen', index=False)
            
            # Crear hoja de transformaciones aplicadas
            transformations = [
                'Filtrado de registros inconsistentes',
                'Eliminación de precio_arriendo',
                'Imputación de valores faltantes',
                'Transformación logarítmica (precio_venta, area, administracion)',
                'Truncamiento de outliers',
                'Target Encoding (localidad)',
                'Frequency Encoding (barrio)',
                'One-Hot Encoding (antiguedad, tipo_propiedad)',
                'Escalado RobustScaler',
                'Ingeniería de características'
            ]
            
            transform_df = pd.DataFrame({'Transformación_Aplicada': transformations})
            transform_df.to_excel(writer, sheet_name='Transformaciones', index=False)
        
        print(f"✅ Excel guardado exitosamente: {output_path}")
        print(f"📊 Hojas incluidas: Datos_Procesados, Resumen, Transformaciones")

def complete_preprocessing_pipeline(file_path, filter_outliers=True, save_excel=True, output_path="datos_preprocesados.xlsx"):
    """
    Pipeline completo de preprocesamiento con exportación a Excel
    
    Args:
        file_path (str): Ruta al archivo Excel original
        filter_outliers (bool): Si aplicar filtrado de outliers extremos
        save_excel (bool): Si guardar el resultado en Excel
        output_path (str): Ruta para guardar el Excel procesado
    
    Returns:
        tuple: (X_processed, y, processed_df, feature_names)
    """
    print("🚀 INICIANDO PIPELINE COMPLETO DE PREPROCESAMIENTO")
    print("=" * 60)
    
    # 1. Cargar datos
    print("📥 Cargando datos...")
    df = pd.read_excel(file_path)
    print(f"✅ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
    
    # 2. Inicializar preprocessor
    processor = BogotaApartmentsPreprocessor()
    
    # 3. Filtrado de registros inconsistentes
    df = processor.filter_inconsistent_records(df)
    
    # 4. Eliminar precio_arriendo
    df = processor.remove_price_arriendo(df)
    
    # 5. Manejo de valores faltantes
    df = processor.handle_missing_values(df)
    
    # 6. Ingeniería de características
    df = processor.create_engineered_features(df)
    
    # 7. Filtrado de outliers extremos (opcional)
    if filter_outliers:
        outlier_columns = ['precio_venta', 'area', 'administracion']
        df = processor.filter_extreme_outliers(df, outlier_columns)
    
    # 8. Aplicar KNN Imputer para administración (si hay valores faltantes)
    if 'administracion' in df.columns and df['administracion'].isnull().any():
        print("🔧 Aplicando KNN Imputer para administración...")
        knn_imputer = KNNImputer(n_neighbors=5)
        knn_features = ['estrato', 'area', 'precio_venta', 'latitud', 'longitud']
        available_features = [f for f in knn_features if f in df.columns]
        
        if available_features:
            # Crear dataset temporal para KNN
            temp_df = df[available_features + ['administracion']].copy()
            temp_imputed = knn_imputer.fit_transform(temp_df)
            
            # Actualizar administración
            admin_idx = temp_df.columns.get_loc('administracion')
            df['administracion'] = temp_imputed[:, admin_idx]
            print("✅ KNN Imputer aplicado a administración")
    
    # 9. Aplicar transformaciones logarítmicas
    log_columns = ['precio_venta', 'area', 'administracion']
    df = processor.apply_log_transformation(df, log_columns)
    
    # 10. Truncar outliers en variables específicas
    truncate_columns = ['parqueaderos', 'banos', 'habitaciones']
    df = processor.truncate_outliers(df, truncate_columns, percentile=99)
    
    # 11. Aplicar encodings
    df = processor.apply_target_encoding(df, 'precio_venta')
    df = processor.apply_frequency_encoding(df)
    
    # 12. Aplicar One-Hot Encoding
    onehot_columns = ['antiguedad', 'tipo_propiedad']
    df = processor.apply_onehot_encoding(df, onehot_columns)
    
    # 13. Escalar características numéricas
    numeric_columns_to_scale = [
        'area', 'habitaciones', 'banos', 'estrato', 'parqueaderos', 
        'administracion', 'latitud', 'longitud', 'precio_m2',
        'banos_por_area', 'habitaciones_por_area', 'amenities_score',
        'localidad_encoded', 'barrio_freq_encoded'
    ]
    # Filtrar columnas que existen
    numeric_columns_to_scale = [col for col in numeric_columns_to_scale if col in df.columns]
    df = processor.scale_numeric_features(df, numeric_columns_to_scale)
    
    # 14. Preparar datos para modelado
    print("📊 Preparando datos para modelado...")
    
    # Separar características y target
    feature_columns = [col for col in df.columns if col != 'precio_venta' and not col.startswith('log_')]
    X = df[feature_columns]
    y = df['precio_venta']  # Target original
    
    # Para modelado, usar el target transformado
    y_transformed = df['log_precio_venta'] if 'log_precio_venta' in df.columns else y
    
    print(f"✅ Dataset final: {X.shape[0]} muestras, {X.shape[1]} características")
    
    # 15. Guardar DataFrame procesado
    processor.processed_df = df
    
    if save_excel:
        processor.save_processed_excel(df, output_path)
    
    # 16. Preparar arrays para modelado
    X_processed = X.select_dtypes(include=[np.number]).values  # Solo características numéricas
    feature_names = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"🎯 Características para modelado: {len(feature_names)}")
    print(f"📈 Variable objetivo: {len(y_transformed)} muestras")
    
    return X_processed, y_transformed, df, feature_names

# Función de conveniencia para uso rápido
def preprocess_bogota_apartments(file_path, filter_outliers=True, save_excel=True, output_path="datos_preprocesados.xlsx"):
    """
    Función simple para ejecutar el pipeline completo
    """
    return complete_preprocessing_pipeline(file_path, filter_outliers, save_excel, output_path)

# Ejemplo de uso
if __name__ == "__main__":
    # Ejecutar pipeline completo
    try:
        file_path = "bogota_apartments.xlsx"
        X_processed, y, processed_df, feature_names = preprocess_bogota_apartments(
            file_path, 
            save_excel=True,
            output_path="apartamentos_bogota_preprocesados.xlsx"
        )
        
        print("\n🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 50)
        print(f"📊 Datos preprocesados: {X_processed.shape}")
        print(f"🎯 Variable objetivo: {len(y)} muestras")
        print(f"🔧 Características: {len(feature_names)}")
        print(f"💾 Excel guardado: apartamentos_bogota_preprocesados.xlsx")
        
        # Mostrar primeras características
        print("\n📋 Primeras 10 características:")
        for i, name in enumerate(feature_names[:10]):
            print(f"  {i+1}. {name}")
            
        # Mostrar estructura del Excel
        print("\n📁 ESTRUCTURA DEL EXCEL GUARDADO:")
        print("  • Hoja 'Datos_Procesados': Dataset completo preprocesado")
        print("  • Hoja 'Resumen': Métricas y estadísticas del dataset")
        print("  • Hoja 'Transformaciones': Lista de transformaciones aplicadas")
            
    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")
        import traceback
        traceback.print_exc()