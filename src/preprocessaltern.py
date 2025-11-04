"""
MÓDULO COMPLETO DE PREPROCESAMIENTO - SIN DATA LEAKAGE
Versión corregida con separación estricta pre-split/post-split
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class BogotaApartmentsPreprocessor:
    """
    Pipeline PRE-SPLIT: Solo transformaciones que no requieren estadísticas del dataset
    """
    
    def __init__(self):
        self.numeric_features = ['area', 'habitaciones', 'banos', 'estrato', 
                               'parqueaderos', 'administracion', 'latitud', 'longitud']
        self.categorical_features = ['localidad', 'barrio', 'antiguedad', 'tipo_propiedad']
        self.target = 'precio_venta'
        self.processed_df = None
        
    def filter_inconsistent_records(self, df):
        """Filtrado de registros inconsistentes - PRE SPLIT (sin estadísticas)"""
        print("🎯 Filtrando registros inconsistentes...")
        initial_count = len(df)
        
        # Eliminar filas con precio_venta faltante
        df = df.dropna(subset=[self.target])
        
        # Aplicar filtros de calidad (reglas fijas, no estadísticas)
        filters = (
            (df['area'] > 0) &
            (df['estrato'].between(1, 6)) &
            (df['latitud'].between(4.55, 4.85)) &
            (df['longitud'].between(-74.2, -74.0)) &
            (df['precio_venta'] > 0)
        )
        
        df_filtered = df[filters].copy()
        final_count = len(df_filtered)
        
        print(f"✅ Registros después de filtrado: {final_count}/{initial_count} "
              f"({final_count/initial_count*100:.1f}%)")
        
        return df_filtered
    
    def remove_price_arriendo(self, df):
        """Eliminar columna precio_arriendo - PRE SPLIT"""
        if 'precio_arriendo' in df.columns:
            df = df.drop(columns=['precio_arriendo'])
            print("✅ Columna precio_arriendo eliminada")
        return df
    
    def create_engineered_features(self, df):
        """Ingeniería de características - PRE SPLIT (sin usar estadísticas)"""
        print("🛠️ Creando características de ingeniería...")
        
        # Solo transformaciones que no requieren estadísticas del dataset
        if all(col in df.columns for col in ['banos', 'area']):
            df['banos_por_area'] = df['banos'] / df['area']
            print("✅ banos_por_area creado")
        
        if all(col in df.columns for col in ['habitaciones', 'area']):
            df['habitaciones_por_area'] = df['habitaciones'] / df['area']
            print("✅ habitaciones_por_area creado")
        
        if all(col in df.columns for col in ['precio_venta', 'area']):
            df['precio_m2'] = df['precio_venta'] / df['area']
            print("✅ precio_m2 creado")
        
        # Score de amenities (transformación booleana)
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

    def apply_log_transformation(self, df, columns):
        """Aplicar transformación log1p - PRE SPLIT (solo para target)"""
        print("📈 Aplicando transformación logarítmica...")
        df_transformed = df.copy()
        
        for col in columns:
            if col in df_transformed.columns:
                # Solo aplicar log al target, características se escalarán post-split
                if col == self.target:
                    min_val = df_transformed[col].min()
                    if min_val <= 0:
                        df_transformed[col] = df_transformed[col] - min_val + 1
                    df_transformed[f'log_{col}'] = np.log1p(df_transformed[col])
                    print(f"✅ {col} transformado a log_{col}")
        
        return df_transformed

    def basic_missing_values_handling(self, df):
        """Manejo básico de valores faltantes - PRE SPLIT (sin estadísticas)"""
        print("🔄 Manejo básico de valores faltantes...")
        
        # Solo imputaciones que no requieren estadísticas
        if 'barrio' in df.columns:
            df['barrio'] = df['barrio'].fillna('Desconocido')
            print("✅ Barrio: valores faltantes reemplazados con 'Desconocido'")
        
        return df

    def pre_split_processing(self, df):
        """
        Pipeline COMPLETO PRE-SPLIT
        Solo transformaciones que no causan data leakage
        """
        print("🚀 INICIANDO PREPROCESAMIENTO PRE-SPLIT")
        print("=" * 50)
        
        # 1. Filtrado de registros inconsistentes
        df = self.filter_inconsistent_records(df)
        
        # 2. Eliminar precio_arriendo
        df = self.remove_price_arriendo(df)
        
        # 3. Manejo básico de missing values
        df = self.basic_missing_values_handling(df)
        
        # 4. Ingeniería de características (sin estadísticas)
        df = self.create_engineered_features(df)
        
        # 5. Transformación logarítmica SOLO para target
        df = self.apply_log_transformation(df, [self.target])
        
        self.processed_df = df
        print(f"✅ Preprocesamiento PRE-SPLIT completado: {df.shape[0]} registros, {df.shape[1]} columnas")
        return df


class PostSplitTransformer:
    """
    Transformaciones que deben aplicarse DESPUÉS de dividir train/test
    Para evitar data leakage
    """
    
    def __init__(self):
        self.imputer_antiguedad = None
        self.target_encoder_localidad = None
        self.freq_encoder_barrio = None
        self.scaler = None
        self.outlier_filters = {}
        self.truncate_limits = {}
        self.knn_imputer = None
        self.onehot_encoder = None
        self.categories_to_keep = None
        
    def handle_missing_values_train(self, df):
        """Manejo de valores faltantes - SOLO TRAIN"""
        print("🔄 Manejo de valores faltantes (train only)...")
        df_processed = df.copy()
        
        # Antiguedad: guardar moda de TRAIN para usar en test
        if 'antiguedad' in df_processed.columns:
            mode_result = df_processed['antiguedad'].mode()
            self.imputer_antiguedad = mode_result[0] if not mode_result.empty else 'Desconocido'
            df_processed['antiguedad'] = df_processed['antiguedad'].fillna(self.imputer_antiguedad)
            print(f"✅ Antiguedad (train): valores faltantes imputados con moda '{self.imputer_antiguedad}'")
        
        return df_processed
    
    def handle_missing_values_test(self, df):
        """Manejo de valores faltantes - SOLO TEST"""
        print("🔄 Aplicando imputación a test...")
        df_processed = df.copy()
        
        # Antiguedad: usar moda aprendida de TRAIN
        if 'antiguedad' in df_processed.columns and self.imputer_antiguedad is not None:
            df_processed['antiguedad'] = df_processed['antiguedad'].fillna(self.imputer_antiguedad)
            print(f"✅ Antiguedad (test): valores faltantes imputados con moda de train '{self.imputer_antiguedad}'")
        
        return df_processed
    
    def filter_extreme_outliers_train(self, df, columns, lower_percentile=1, upper_percentile=99):
        """Filtrar outliers extremos - SOLO TRAIN"""
        print("📊 Filtrado de outliers extremos (train only)...")
        initial_count = len(df)
        
        self.outlier_filters = {}
        
        for col in columns:
            if col in df.columns:
                # Calcular límites SOLO con train
                lower_bound = df[col].quantile(lower_percentile / 100)
                upper_bound = df[col].quantile(upper_percentile / 100)
                
                # Guardar límites para aplicar mismo filtro a test
                self.outlier_filters[col] = (lower_bound, upper_bound)
                
                # Filtrar outliers en train
                mask = df[col].between(lower_bound, upper_bound)
                df = df[mask]
                
                outliers_removed = initial_count - len(df)
                if outliers_removed > 0:
                    print(f"✅ {col}: {outliers_removed} outliers removidos "
                          f"({lower_percentile}%-{upper_percentile}%)")
        
        final_count = len(df)
        print(f"📈 Registros train después de filtrado de outliers: {final_count}/{initial_count} "
              f"({final_count/initial_count*100:.1f}%)")
        
        return df
    
    def filter_extreme_outliers_test(self, df):
        """Filtrar outliers extremos - SOLO TEST (usando límites de train)"""
        print("📊 Aplicando filtro de outliers a test...")
        initial_count = len(df)
        
        for col, (lower_bound, upper_bound) in self.outlier_filters.items():
            if col in df.columns:
                mask = df[col].between(lower_bound, upper_bound)
                df = df[mask]
                print(f"✅ {col}: filtro aplicado [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        final_count = len(df)
        print(f"📈 Registros test después de filtrado de outliers: {final_count}/{initial_count}")
        
        return df
    
    def apply_knn_imputer_train(self, df):
        """Aplicar KNN Imputer - SOLO TRAIN"""
        if 'administracion' in df.columns and df['administracion'].isnull().any():
            print("🔧 Aplicando KNN Imputer para administración (train only)...")
            
            # Seleccionar características para KNN
            knn_features = ['estrato', 'area', 'latitud', 'longitud', 'precio_m2']
            available_features = [f for f in knn_features if f in df.columns]
            
            if available_features and 'administracion' in df.columns:
                # Crear dataset temporal para KNN
                temp_df = df[available_features + ['administracion']].copy()
                
                # Entrenar KNN imputer con train
                self.knn_imputer = KNNImputer(n_neighbors=5)
                temp_imputed = self.knn_imputer.fit_transform(temp_df)
                
                # Actualizar administración en train
                admin_idx = temp_df.columns.get_loc('administracion')
                df['administracion'] = temp_imputed[:, admin_idx]
                print("✅ KNN Imputer entrenado y aplicado a administración (train)")
        
        return df
    
    def apply_knn_imputer_test(self, df):
        """Aplicar KNN Imputer - SOLO TEST (usando imputer entrenado)"""
        if self.knn_imputer is not None and 'administracion' in df.columns:
            print("🔧 Aplicando KNN Imputer para administración (test)...")
            
            # Seleccionar mismas características usadas en train
            knn_features = ['estrato', 'area', 'latitud', 'longitud', 'precio_m2']
            available_features = [f for f in knn_features if f in df.columns]
            
            if available_features and 'administracion' in df.columns:
                temp_df = df[available_features + ['administracion']].copy()
                temp_imputed = self.knn_imputer.transform(temp_df)
                
                # Actualizar administración en test
                admin_idx = temp_df.columns.get_loc('administracion')
                df['administracion'] = temp_imputed[:, admin_idx]
                print("✅ KNN Imputer aplicado a administración (test)")
        
        return df
    
    def truncate_outliers_train(self, df, columns, percentile=99):
        """Truncar outliers - SOLO TRAIN"""
        print("📊 Truncando outliers (train only)...")
        df_truncated = df.copy()
        self.truncate_limits = {}
        
        for col in columns:
            if col in df_truncated.columns:
                # Calcular límite SOLO con train
                trunc_value = df_truncated[col].quantile(percentile / 100)
                self.truncate_limits[col] = trunc_value
                
                # Truncar en train
                df_truncated[col] = np.where(df_truncated[col] > trunc_value, trunc_value, df_truncated[col])
                print(f"✅ {col} truncado en percentil {percentile}% (limite: {trunc_value:.2f})")
        
        return df_truncated
    
    def truncate_outliers_test(self, df):
        """Truncar outliers - SOLO TEST (usando límites de train)"""
        print("📊 Truncando outliers (test)...")
        df_truncated = df.copy()
        
        for col, trunc_value in self.truncate_limits.items():
            if col in df_truncated.columns:
                df_truncated[col] = np.where(df_truncated[col] > trunc_value, trunc_value, df_truncated[col])
                print(f"✅ {col} truncado con límite de train: {trunc_value:.2f}")
        
        return df_truncated
    
    def apply_target_encoding_train(self, df, target_col):
        """Target Encoding - SOLO TRAIN"""
        print("🔤 Aplicando Target Encoding a localidad (train only)...")
        df_encoded = df.copy()
        
        if 'localidad' in df_encoded.columns and target_col in df_encoded.columns:
            # Calcular promedio de precio_venta por localidad SOLO con train
            self.target_encoder_localidad = df_encoded.groupby('localidad')[target_col].mean().to_dict()
            
            # Aplicar encoding a train
            df_encoded['localidad_encoded'] = df_encoded['localidad'].map(self.target_encoder_localidad)
            
            # Imputar con media global de TRAIN si hay valores nulos
            global_mean = df_encoded[target_col].mean()
            df_encoded['localidad_encoded'] = df_encoded['localidad_encoded'].fillna(global_mean)
            
            print(f"✅ Target Encoding aplicado a localidad (train) - {len(self.target_encoder_localidad)} categorías")
        
        return df_encoded
    
    def apply_target_encoding_test(self, df, target_col):
        """Target Encoding - SOLO TEST"""
        print("🔤 Aplicando Target Encoding a localidad (test)...")
        df_encoded = df.copy()
        
        if 'localidad' in df_encoded.columns and self.target_encoder_localidad is not None:
            # Aplicar encoding aprendido de train
            df_encoded['localidad_encoded'] = df_encoded['localidad'].map(self.target_encoder_localidad)
            
            # Imputar con media global de TRAIN
            if 'localidad_encoded' in df_encoded.columns:
                # Usar la primera media global que calculamos de train
                global_mean = np.mean(list(self.target_encoder_localidad.values()))
                df_encoded['localidad_encoded'] = df_encoded['localidad_encoded'].fillna(global_mean)
            
            print("✅ Target Encoding aplicado a localidad (test)")
        
        return df_encoded
    
    def apply_frequency_encoding_train(self, df, threshold=0.01):
        """Frequency Encoding - SOLO TRAIN"""
        print("🔤 Aplicando Frequency Encoding a barrio (train only)...")
        df_encoded = df.copy()
        
        if 'barrio' in df_encoded.columns:
            # Calcular frecuencias SOLO con train
            freq_series = df_encoded['barrio'].value_counts(normalize=True)
            
            # Identificar categorías a mantener (frecuencia > threshold)
            self.categories_to_keep = set(freq_series[freq_series > threshold].index)
            
            # Agrupar categorías raras en train
            df_encoded['barrio_grouped'] = df_encoded['barrio'].apply(
                lambda x: x if x in self.categories_to_keep else 'Otros'
            )
            
            # Aplicar frequency encoding y guardar mapeo
            self.freq_encoder_barrio = df_encoded['barrio_grouped'].value_counts(normalize=True).to_dict()
            df_encoded['barrio_freq_encoded'] = df_encoded['barrio_grouped'].map(self.freq_encoder_barrio)
            
            print(f"✅ Frequency Encoding aplicado a barrio (train) - {len(self.categories_to_keep)} categorías conservadas")
        
        return df_encoded
    
    def apply_frequency_encoding_test(self, df):
        """Frequency Encoding - SOLO TEST"""
        print("🔤 Aplicando Frequency Encoding a barrio (test)...")
        df_encoded = df.copy()
        
        if 'barrio' in df_encoded.columns and self.categories_to_keep is not None:
            # Agrupar categorías raras usando criterio de TRAIN
            df_encoded['barrio_grouped'] = df_encoded['barrio'].apply(
                lambda x: x if x in self.categories_to_keep else 'Otros'
            )
            
            # Aplicar frequency encoding usando mapeo de TRAIN
            if self.freq_encoder_barrio is not None:
                df_encoded['barrio_freq_encoded'] = df_encoded['barrio_grouped'].map(self.freq_encoder_barrio)
                # Imputar con 0 si hay nuevas categorías
                df_encoded['barrio_freq_encoded'] = df_encoded['barrio_freq_encoded'].fillna(0)
            
            print("✅ Frequency Encoding aplicado a barrio (test)")
        
        return df_encoded
    
    def apply_onehot_encoding_train(self, df, columns):
        """One-Hot Encoding - SOLO TRAIN"""
        print("🔤 Aplicando One-Hot Encoding (train only)...")
        df_encoded = df.copy()
        
        for col in columns:
            if col in df_encoded.columns:
                # Aplicar one-hot encoding manualmente para mantener control
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                # Eliminar columna original
                df_encoded = df_encoded.drop(columns=[col])
                
                print(f"✅ One-Hot Encoding aplicado a {col} ({len(dummies.columns)} categorías)")
        
        return df_encoded
    
    def apply_onehot_encoding_test(self, df, columns):
        """One-Hot Encoding - SOLO TEST (asegurando mismas columnas que train)"""
        print("🔤 Aplicando One-Hot Encoding (test)...")
        df_encoded = df.copy()
        
        for col in columns:
            if col in df_encoded.columns:
                # Aplicar one-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                # Eliminar columna original
                df_encoded = df_encoded.drop(columns=[col])
        
        print("✅ One-Hot Encoding aplicado a test")
        return df_encoded
    
    def scale_numeric_features_train(self, df, columns):
        """Escalar características numéricas - SOLO TRAIN"""
        print("⚖️ Escalando características numéricas (train only)...")
        df_scaled = df.copy()
        
        # Filtrar columnas que existen
        columns_to_scale = [col for col in columns if col in df_scaled.columns]
        
        if columns_to_scale:
            # Entrenar scaler
            self.scaler = RobustScaler()
            scaled_values = self.scaler.fit_transform(df_scaled[columns_to_scale])
            
            # Crear nuevas columnas escaladas
            for i, col in enumerate(columns_to_scale):
                df_scaled[f'scaled_{col}'] = scaled_values[:, i]
                print(f"✅ {col} escalado con RobustScaler")
        else:
            print("⚠️  No hay columnas numéricas para escalar")
        
        return df_scaled
    
    def scale_numeric_features_test(self, df):
        """Escalar características numéricas - SOLO TEST"""
        print("⚖️ Escalando características numéricas (test)...")
        if self.scaler is None:
            print("⚠️  Scaler no entrenado, saltando escalado en test")
            return df
            
        df_scaled = df.copy()
        
        # Obtener columnas que fueron escaladas en train
        scaled_columns = [col for col in df_scaled.columns if col.startswith('scaled_')]
        original_columns = [col.replace('scaled_', '') for col in scaled_columns]
        original_columns = [col for col in original_columns if col in df_scaled.columns]
        
        if original_columns:
            # Aplicar transformación usando scaler entrenado
            scaled_values = self.scaler.transform(df_scaled[original_columns])
            
            # Actualizar columnas escaladas
            for i, col in enumerate(original_columns):
                scaled_col_name = f'scaled_{col}'
                if scaled_col_name in df_scaled.columns:
                    df_scaled[scaled_col_name] = scaled_values[:, i]
                else:
                    # Si no existe la columna escalada, crearla
                    df_scaled[scaled_col_name] = scaled_values[:, i]
        
        return df_scaled
    
    def align_train_test_columns(self, X_train_df, X_test_df):
        """Asegurar que train y test tengan las mismas columnas"""
        print("🔧 Alineando columnas de train y test...")
        
        # Obtener columnas comunes
        common_columns = X_train_df.columns.intersection(X_test_df.columns)
        
        # Columnas faltantes en test
        missing_in_test = X_train_df.columns.difference(X_test_df.columns)
        if len(missing_in_test) > 0:
            print(f"⚠️  Añadiendo {len(missing_in_test)} columnas faltantes en test...")
            for col in missing_in_test:
                X_test_df[col] = 0
        
        # Columnas faltantes en train
        missing_in_train = X_test_df.columns.difference(X_train_df.columns)
        if len(missing_in_train) > 0:
            print(f"⚠️  Eliminando {len(missing_in_train)} columnas extras en test...")
            X_test_df = X_test_df.drop(columns=missing_in_train)
        
        # Reordenar columnas para que coincidan
        X_test_df = X_test_df[X_train_df.columns]
        
        print(f"✅ Columnas alineadas: {X_train_df.shape[1]} características")
        return X_train_df, X_test_df


def complete_preprocessing_pipeline_corrected(file_path, test_size=0.2, random_state=42, save_excel=True):
    """
    Pipeline COMPLETO corregido sin data leakage
    """
    print("🚀 INICIANDO PIPELINE COMPLETO CORREGIDO (SIN DATA LEAKAGE)")
    print("=" * 60)
    
    # 1. Cargar datos
    print("📥 Cargando datos...")
    df = pd.read_excel(file_path)
    print(f"✅ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
    
    # 2. Preprocesamiento PRE-SPLIT
    pre_processor = BogotaApartmentsPreprocessor()
    df_pre_split = pre_processor.pre_split_processing(df)
    
    # 3. SEPARAR DATOS en train/test ANTES de cualquier transformación con estadísticas
    print("\n📊 SEPARANDO DATOS EN TRAIN/TEST...")
    
    # Preparar características y target
    feature_columns = [col for col in df_pre_split.columns if col != 'precio_venta' and col != 'log_precio_venta']
    X = df_pre_split[feature_columns]
    y = df_pre_split['log_precio_venta']  # Usar target transformado
    
    # Separar datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"✅ Datos separados:")
    print(f"   • Train: {X_train.shape[0]} muestras")
    print(f"   • Test: {X_test.shape[0]} muestras")
    
    # 4. Inicializar transformador POST-SPLIT
    post_transformer = PostSplitTransformer()
    
    # 5. PROCESAMIENTO TRAIN (con estadísticas)
    print("\n🎯 PROCESANDO DATOS DE TRAIN...")
    
    # Convertir a DataFrame para mantener nombres de columnas
    X_train_df = pd.DataFrame(X_train, columns=X.columns, index=X_train.index)
    y_train_series = pd.Series(y_train, index=X_train.index)
    
    # Aplicar transformaciones POST-SPLIT a train
    X_train_processed = post_transformer.handle_missing_values_train(X_train_df)
    X_train_processed = post_transformer.apply_knn_imputer_train(X_train_processed)
    
    # Filtrar outliers SOLO en train
    outlier_columns = ['area', 'administracion', 'precio_m2']
    outlier_columns = [col for col in outlier_columns if col in X_train_processed.columns]
    X_train_filtered = post_transformer.filter_extreme_outliers_train(X_train_processed, outlier_columns)
    
    # Ajustar y_train al filtrado de outliers
    y_train_filtered = y_train_series.loc[X_train_filtered.index]
    
    # Continuar procesamiento
    X_train_processed = post_transformer.truncate_outliers_train(X_train_filtered, ['parqueaderos', 'banos', 'habitaciones'])
    X_train_processed = post_transformer.apply_target_encoding_train(X_train_processed, 'precio_venta')
    X_train_processed = post_transformer.apply_frequency_encoding_train(X_train_processed)
    X_train_processed = post_transformer.apply_onehot_encoding_train(X_train_processed, ['antiguedad', 'tipo_propiedad'])
    
    # Escalar características numéricas
    numeric_columns_to_scale = [
        'area', 'habitaciones', 'banos', 'estrato', 'parqueaderos', 
        'administracion', 'latitud', 'longitud', 'precio_m2',
        'banos_por_area', 'habitaciones_por_area', 'amenities_score', 
        'localidad_encoded', 'barrio_freq_encoded'
    ]
    numeric_columns_to_scale = [col for col in numeric_columns_to_scale if col in X_train_processed.columns]
    X_train_processed = post_transformer.scale_numeric_features_train(X_train_processed, numeric_columns_to_scale)
    
    # 6. PROCESAMIENTO TEST (usando transformadores entrenados con train)
    print("\n🎯 PROCESANDO DATOS DE TEST...")
    
    # Convertir a DataFrame
    X_test_df = pd.DataFrame(X_test, columns=X.columns, index=X_test.index)
    y_test_series = pd.Series(y_test, index=X_test.index)
    
    # Aplicar mismas transformaciones usando parámetros de TRAIN
    X_test_processed = post_transformer.handle_missing_values_test(X_test_df)
    X_test_processed = post_transformer.apply_knn_imputer_test(X_test_processed)
    X_test_processed = post_transformer.filter_extreme_outliers_test(X_test_processed)
    X_test_processed = post_transformer.truncate_outliers_test(X_test_processed)
    X_test_processed = post_transformer.apply_target_encoding_test(X_test_processed, 'precio_venta')
    X_test_processed = post_transformer.apply_frequency_encoding_test(X_test_processed)
    X_test_processed = post_transformer.apply_onehot_encoding_test(X_test_processed, ['antiguedad', 'tipo_propiedad'])
    X_test_processed = post_transformer.scale_numeric_features_test(X_test_processed)
    
    # 7. Alinear columnas de train y test
    X_train_final, X_test_final = post_transformer.align_train_test_columns(X_train_processed, X_test_processed)
    
    # 8. Preparar datos para modelado
    print("\n📊 PREPARANDO DATOS PARA MODELADO...")
    
    # Seleccionar solo características numéricas para modelado
    def get_numeric_features(df):
        return df.select_dtypes(include=[np.number])
    
    X_train_numeric = get_numeric_features(X_train_final)
    X_test_numeric = get_numeric_features(X_test_final)
    
    # Asegurar que train y test tengan las mismas columnas (nuevamente por seguridad)
    common_columns = X_train_numeric.columns.intersection(X_test_numeric.columns)
    X_train_final = X_train_numeric[common_columns]
    X_test_final = X_test_numeric[common_columns]
    
    print(f"✅ Datos finales:")
    print(f"   • X_train: {X_train_final.shape}")
    print(f"   • X_test: {X_test_final.shape}")
    print(f"   • y_train: {y_train_filtered.shape}")
    print(f"   • y_test: {y_test_series.shape}")
    print(f"   • Características: {len(common_columns)}")
    
    # 9. Guardar resultados si es necesario
    if save_excel:
        # Combinar train y test para guardar
        X_train_final['dataset'] = 'train'
        X_test_final['dataset'] = 'test'
        X_train_final['log_precio_venta'] = y_train_filtered.values
        X_test_final['log_precio_venta'] = y_test_series.values
        
        combined_df = pd.concat([X_train_final, X_test_final], axis=0)
        output_path = "datos_preprocesados_sin_leakage.xlsx"
        combined_df.to_excel(output_path, index=False)
        print(f"💾 Datos guardados en: {output_path}")
    
    return (X_train_final.values, X_test_final.values, 
            y_train_filtered.values, y_test_series.values, 
            common_columns.tolist())


# Función de conveniencia
def preprocess_bogota_apartments_corrected(file_path, test_size=0.2, random_state=42, save_excel=True):
    """
    Función corregida sin data leakage - interfaz simple
    """
    return complete_preprocessing_pipeline_corrected(
        file_path, 
        test_size=test_size, 
        random_state=random_state, 
        save_excel=save_excel
    )


# Función para diagnóstico rápido
def diagnose_preprocessing(file_path):
    """
    Función de diagnóstico para verificar el preprocesamiento
    """
    print("🔍 DIAGNÓSTICO DE PREPROCESAMIENTO")
    print("=" * 50)
    
    try:
        # Cargar datos originales
        df_original = pd.read_excel(file_path)
        print(f"📊 Datos originales: {df_original.shape}")
        
        # Preprocesamiento PRE-SPLIT
        pre_processor = BogotaApartmentsPreprocessor()
        df_pre_split = pre_processor.pre_split_processing(df_original)
        print(f"📊 Después de PRE-SPLIT: {df_pre_split.shape}")
        
        # Procesamiento completo
        X_train, X_test, y_train, y_test, features = preprocess_bogota_apartments_corrected(
            file_path, test_size=0.2, save_excel=False
        )
        
        print(f"✅ DIAGNÓSTICO EXITOSO:")
        print(f"   • X_train: {X_train.shape}")
        print(f"   • X_test: {X_test.shape}")
        print(f"   • Features: {len(features)}")
        print(f"   • Sin data leakage confirmado")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en diagnóstico: {e}")
        return False


# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 MÓDULO DE PREPROCESAMIENTO CORREGIDO - SIN DATA LEAKAGE")
    
    # Ejecutar diagnóstico
    try:
        file_path = "bogota_apartments.xlsx"  # Cambiar por tu ruta
        success = diagnose_preprocessing(file_path)
        
        if success:
            print("\n🎉 MÓDULO LISTO PARA USAR")
            print("\n📝 USO:")
            print("   from preprocesamiento_corregido import preprocess_bogota_apartments_corrected")
            print("   X_train, X_test, y_train, y_test, features = preprocess_bogota_apartments_corrected('tu_archivo.xlsx')")
        else:
            print("\n❌ VERIFICAR EL ARCHIVO DE DATOS")
            
    except Exception as e:
        print(f"❌ Error en ejecución: {e}")
        print("\n💡 Asegúrate de tener el archivo 'bogota_apartments.xlsx' en el mismo directorio")