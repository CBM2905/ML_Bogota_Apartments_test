"""
MÓDULO COMPLETO DE PREPROCESAMIENTO - VERSIÓN FINAL CORREGIDA (Se esta usando este)
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class BogotaApartmentsPreprocessor:
    """
    Pipeline PRE-SPLIT: Solo transformaciones que no requieren estadísticas del dataset
    """
    
    def __init__(self):
        self.target = 'precio_venta'
        self.processed_df = None
        
    def filter_inconsistent_records(self, df):
        """Filtrado de registros inconsistentes - PRE SPLIT"""
        print("🎯 Filtrando registros inconsistentes...")
        initial_count = len(df)
        
        df = df.dropna(subset=[self.target])
        
        filters = (
            (df['area'] > 0) &
            (df['estrato'].between(1, 6)) &
            (df['latitud'].between(4.55, 4.85)) &
            (df['longitud'].between(-74.2, -74.0)) &
            (df['precio_venta'] > 0)
        )
        
        df_filtered = df[filters].copy()
        final_count = len(df_filtered)
        
        print(f"✅ Registros después de filtrado: {final_count}/{initial_count} ({final_count/initial_count*100:.1f}%)")
        return df_filtered
    
    def remove_price_arriendo(self, df):
        """Eliminar columna precio_arriendo"""
        if 'precio_arriendo' in df.columns:
            df = df.drop(columns=['precio_arriendo'])
            print("✅ Columna precio_arriendo eliminada")
        return df
    
    def create_engineered_features(self, df):
        """Ingeniería de características - PRE SPLIT"""
        print("🛠️ Creando características de ingeniería...")
        
        if all(col in df.columns for col in ['banos', 'area']):
            df['banos_por_area'] = df['banos'] / df['area']
        
        if all(col in df.columns for col in ['habitaciones', 'area']):
            df['habitaciones_por_area'] = df['habitaciones'] / df['area']
        
        if all(col in df.columns for col in ['precio_venta', 'area']):
            df['precio_m2'] = df['precio_venta'] / df['area']
        
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

    def apply_log_transformation(self, df):
        """Aplicar transformación log1p - SOLO para target"""
        print("📈 Aplicando transformación logarítmica...")
        df_transformed = df.copy()
        
        if self.target in df_transformed.columns:
            min_val = df_transformed[self.target].min()
            if min_val <= 0:
                df_transformed[self.target] = df_transformed[self.target] - min_val + 1
            df_transformed[f'log_{self.target}'] = np.log1p(df_transformed[self.target])
            print(f"✅ {self.target} transformado a log_{self.target}")
        
        return df_transformed

    def basic_missing_values_handling(self, df):
        """Manejo básico de valores faltantes - PRE SPLIT"""
        print("🔄 Manejo básico de valores faltantes...")
        
        if 'barrio' in df.columns:
            df['barrio'] = df['barrio'].fillna('Desconocido')
            print("✅ Barrio: valores faltantes reemplazados con 'Desconocido'")
        
        return df

    def pre_split_processing(self, df):
        """Pipeline COMPLETO PRE-SPLIT"""
        print("🚀 INICIANDO PREPROCESAMIENTO PRE-SPLIT")
        print("=" * 50)
        
        df = self.filter_inconsistent_records(df)
        df = self.remove_price_arriendo(df)
        df = self.basic_missing_values_handling(df)
        df = self.create_engineered_features(df)
        df = self.apply_log_transformation(df)
        
        self.processed_df = df
        print(f"✅ Preprocesamiento PRE-SPLIT completado: {df.shape[0]} registros, {df.shape[1]} columnas")
        return df


class PostSplitTransformer:
    """Transformaciones POST-SPLIT para evitar data leakage"""
    
    def __init__(self):
        self.imputer_antiguedad = None
        self.target_encoder_localidad = None
        self.freq_encoder_barrio = None
        self.scaler = None
        self.outlier_filters = {}
        self.truncate_limits = {}
        self.knn_imputer = None
        self.medians = {}
        
    def handle_missing_values_train(self, df):
        """Manejo de valores faltantes - SOLO TRAIN"""
        print("🔄 Manejo de valores faltantes (train only)...")
        
        if 'antiguedad' in df.columns:
            mode_result = df['antiguedad'].mode()
            self.imputer_antiguedad = mode_result[0] if not mode_result.empty else 'Desconocido'
            df['antiguedad'] = df['antiguedad'].fillna(self.imputer_antiguedad)
            print(f"✅ Antiguedad (train): imputados con moda '{self.imputer_antiguedad}'")
        
        return df
    
    def handle_missing_values_test(self, df):
        """Manejo de valores faltantes - SOLO TEST"""
        if 'antiguedad' in df.columns and self.imputer_antiguedad is not None:
            df['antiguedad'] = df['antiguedad'].fillna(self.imputer_antiguedad)
        
        return df
    
    def filter_extreme_outliers_train(self, df, columns, lower_percentile=1, upper_percentile=99):
        """Filtrar outliers extremos - SOLO TRAIN"""
        print("📊 Filtrado de outliers extremos (train only)...")
        initial_count = len(df)
        
        self.outlier_filters = {}
        
        for col in columns:
            if col in df.columns:
                lower_bound = df[col].quantile(lower_percentile / 100)
                upper_bound = df[col].quantile(upper_percentile / 100)
                self.outlier_filters[col] = (lower_bound, upper_bound)
                
                mask = df[col].between(lower_bound, upper_bound)
                df = df[mask]
        
        final_count = len(df)
        print(f"📈 Registros train después de filtrado: {final_count}/{initial_count} ({final_count/initial_count*100:.1f}%)")
        return df
    
    def filter_extreme_outliers_test(self, df):
        """Filtrar outliers extremos - SOLO TEST"""
        for col, (lower_bound, upper_bound) in self.outlier_filters.items():
            if col in df.columns:
                mask = df[col].between(lower_bound, upper_bound)
                df = df[mask]
        
        return df
    
    def apply_knn_imputer_train(self, df):
        """Aplicar KNN Imputer - SOLO TRAIN"""
        if 'administracion' in df.columns and df['administracion'].isnull().any():
            print("🔧 Aplicando KNN Imputer para administración (train only)...")
            
            knn_features = ['estrato', 'area', 'latitud', 'longitud', 'precio_m2']
            available_features = [f for f in knn_features if f in df.columns]
            
            if available_features and 'administracion' in df.columns:
                temp_df = df[available_features + ['administracion']].copy()
                self.knn_imputer = KNNImputer(n_neighbors=5)
                temp_imputed = self.knn_imputer.fit_transform(temp_df)
                admin_idx = temp_df.columns.get_loc('administracion')
                df['administracion'] = temp_imputed[:, admin_idx]
                print("✅ KNN Imputer entrenado y aplicado a administración (train)")
        
        return df
    
    def apply_knn_imputer_test(self, df):
        """Aplicar KNN Imputer - SOLO TEST"""
        if self.knn_imputer is not None and 'administracion' in df.columns:
            knn_features = ['estrato', 'area', 'latitud', 'longitud', 'precio_m2']
            available_features = [f for f in knn_features if f in df.columns]
            
            if available_features and 'administracion' in df.columns:
                temp_df = df[available_features + ['administracion']].copy()
                temp_imputed = self.knn_imputer.transform(temp_df)
                admin_idx = temp_df.columns.get_loc('administracion')
                df['administracion'] = temp_imputed[:, admin_idx]
        
        return df
    
    def truncate_outliers_train(self, df, columns, percentile=99):
        """Truncar outliers - SOLO TRAIN"""
        print("📊 Truncando outliers (train only)...")
        self.truncate_limits = {}
        
        for col in columns:
            if col in df.columns:
                trunc_value = df[col].quantile(percentile / 100)
                self.truncate_limits[col] = trunc_value
                df[col] = np.where(df[col] > trunc_value, trunc_value, df[col])
                print(f"✅ {col} truncado en percentil {percentile}%")
        
        return df
    
    def truncate_outliers_test(self, df):
        """Truncar outliers - SOLO TEST"""
        for col, trunc_value in self.truncate_limits.items():
            if col in df.columns:
                df[col] = np.where(df[col] > trunc_value, trunc_value, df[col])
        
        return df
    
    def apply_target_encoding_train(self, df, target_col):
        """Target Encoding - SOLO TRAIN"""
        print("🔤 Aplicando Target Encoding a localidad (train only)...")
        
        if 'localidad' in df.columns and target_col in df.columns:
            self.target_encoder_localidad = df.groupby('localidad')[target_col].mean().to_dict()
            df['localidad_encoded'] = df['localidad'].map(self.target_encoder_localidad)
            global_mean = df[target_col].mean()
            df['localidad_encoded'] = df['localidad_encoded'].fillna(global_mean)
            print(f"✅ Target Encoding aplicado a localidad (train)")
        
        return df
    
    def apply_target_encoding_test(self, df):
        """Target Encoding - SOLO TEST"""
        if 'localidad' in df.columns and self.target_encoder_localidad is not None:
            df['localidad_encoded'] = df['localidad'].map(self.target_encoder_localidad)
            if 'localidad_encoded' in df.columns:
                global_mean = np.mean(list(self.target_encoder_localidad.values()))
                df['localidad_encoded'] = df['localidad_encoded'].fillna(global_mean)
        
        return df
    
    def apply_frequency_encoding_train(self, df, threshold=0.01):
        """Frequency Encoding - SOLO TRAIN"""
        print("🔤 Aplicando Frequency Encoding a barrio (train only)...")
        
        if 'barrio' in df.columns:
            freq_series = df['barrio'].value_counts(normalize=True)
            self.categories_to_keep = set(freq_series[freq_series > threshold].index)
            
            df['barrio_grouped'] = df['barrio'].apply(
                lambda x: x if x in self.categories_to_keep else 'Otros'
            )
            
            self.freq_encoder_barrio = df['barrio_grouped'].value_counts(normalize=True).to_dict()
            df['barrio_freq_encoded'] = df['barrio_grouped'].map(self.freq_encoder_barrio)
            print(f"✅ Frequency Encoding aplicado a barrio (train)")
        
        return df
    
    def apply_frequency_encoding_test(self, df):
        """Frequency Encoding - SOLO TEST"""
        if 'barrio' in df.columns and self.categories_to_keep is not None:
            df['barrio_grouped'] = df['barrio'].apply(
                lambda x: x if x in self.categories_to_keep else 'Otros'
            )
            
            if self.freq_encoder_barrio is not None:
                df['barrio_freq_encoded'] = df['barrio_grouped'].map(self.freq_encoder_barrio)
                df['barrio_freq_encoded'] = df['barrio_freq_encoded'].fillna(0)
        
        return df
    
    def apply_onehot_encoding(self, df, columns):
        """One-Hot Encoding - aplicado igual a train y test"""
        print("🔤 Aplicando One-Hot Encoding...")
        
        for col in columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                print(f"✅ One-Hot Encoding aplicado a {col}")
        
        return df
    
    def scale_numeric_features_train(self, df, columns):
        """Escalar características numéricas - SOLO TRAIN"""
        print("⚖️ Escalando características numéricas (train only)...")
        
        columns_to_scale = [col for col in columns if col in df.columns]
        
        if columns_to_scale:
            self.scaler = RobustScaler()
            scaled_values = self.scaler.fit_transform(df[columns_to_scale])
            
            for i, col in enumerate(columns_to_scale):
                df[f'scaled_{col}'] = scaled_values[:, i]
        
        return df
    
    def scale_numeric_features_test(self, df):
        """Escalar características numéricas - SOLO TEST"""
        if self.scaler is None:
            return df
            
        scaled_columns = [col for col in df.columns if col.startswith('scaled_')]
        original_columns = [col.replace('scaled_', '') for col in scaled_columns]
        original_columns = [col for col in original_columns if col in df.columns]
        
        if original_columns:
            scaled_values = self.scaler.transform(df[original_columns])
            for i, col in enumerate(original_columns):
                scaled_col_name = f'scaled_{col}'
                if scaled_col_name in df.columns:
                    df[scaled_col_name] = scaled_values[:, i]
        
        return df
    
    def apply_final_imputation_train(self, df):
        """Imputación final para eliminar cualquier NaN restante - SOLO TRAIN"""
        print("🔧 Aplicando imputación final (train)...")
        
        # Identificar columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Verificar si hay NaN
            nan_before = df[numeric_cols].isnull().sum().sum()
            if nan_before > 0:
                print(f"   • Encontrados {nan_before} valores NaN para imputar")
                
                # ESTRATEGIA SEGURA: imputar columna por columna
                for col in numeric_cols:
                    if df[col].isnull().any():
                        # Calcular la mediana de la columna (excluyendo NaN)
                        median_val = df[col].median()
                        # Si la mediana es NaN (columna completamente vacía), usar 0
                        if pd.isna(median_val):
                            median_val = 0
                            print(f"   • Columna {col} completamente NaN, usando 0")
                        # Imputar los NaN con la mediana
                        df[col].fillna(median_val, inplace=True)
                        # Guardar la mediana para usar en test
                        self.medians[col] = median_val
                
                nan_after = df[numeric_cols].isnull().sum().sum()
                print(f"✅ Eliminados {nan_before - nan_after} valores NaN con imputación por mediana")
            else:
                print("✅ No se encontraron NaN para imputar")
                # Guardar medianas para test
                for col in numeric_cols:
                    self.medians[col] = df[col].median()
        
        return df
    
    def apply_final_imputation_test(self, df):
        """Imputación final para eliminar cualquier NaN restante - SOLO TEST"""
        if hasattr(self, 'medians') and self.medians:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                for col in numeric_cols:
                    if df[col].isnull().any() and col in self.medians:
                        df[col].fillna(self.medians[col], inplace=True)
        
        return df
    
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
    Pipeline COMPLETO corregido sin data leakage y sin NaN
    """
    print("🚀 INICIANDO PIPELINE COMPLETO CORREGIDO (SIN DATA LEAKAGE Y SIN NaN)")
    print("=" * 60)
    
    # 1. Cargar datos
    print("📥 Cargando datos...")
    df = pd.read_excel(file_path)
    print(f"✅ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
    
    # 2. Preprocesamiento PRE-SPLIT
    pre_processor = BogotaApartmentsPreprocessor()
    df_pre_split = pre_processor.pre_split_processing(df)
    
    # 3. SEPARAR DATOS en train/test
    print("\n📊 SEPARANDO DATOS EN TRAIN/TEST...")
    
    feature_columns = [col for col in df_pre_split.columns if col != 'precio_venta' and col != 'log_precio_venta']
    X = df_pre_split[feature_columns]
    y = df_pre_split['log_precio_venta']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"✅ Datos separados: Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # 4. Inicializar transformador POST-SPLIT
    post_transformer = PostSplitTransformer()
    
    # 5. PROCESAMIENTO TRAIN
    print("\n🎯 PROCESANDO DATOS DE TRAIN...")
    
    X_train_df = pd.DataFrame(X_train, columns=X.columns, index=X_train.index)
    y_train_series = pd.Series(y_train, index=X_train.index)
    
    X_train_processed = post_transformer.handle_missing_values_train(X_train_df)
    X_train_processed = post_transformer.apply_knn_imputer_train(X_train_processed)
    
    outlier_columns = ['area', 'administracion', 'precio_m2']
    outlier_columns = [col for col in outlier_columns if col in X_train_processed.columns]
    X_train_filtered = post_transformer.filter_extreme_outliers_train(X_train_processed, outlier_columns)
    y_train_filtered = y_train_series.loc[X_train_filtered.index]
    
    X_train_processed = post_transformer.truncate_outliers_train(X_train_filtered, ['parqueaderos', 'banos', 'habitaciones'])
    X_train_processed = post_transformer.apply_target_encoding_train(X_train_processed, 'precio_venta')
    X_train_processed = post_transformer.apply_frequency_encoding_train(X_train_processed)
    X_train_processed = post_transformer.apply_onehot_encoding(X_train_processed, ['antiguedad', 'tipo_propiedad'])
    
    numeric_columns_to_scale = [
        'area', 'habitaciones', 'banos', 'estrato', 'parqueaderos', 
        'administracion', 'latitud', 'longitud', 'precio_m2',
        'banos_por_area', 'habitaciones_por_area', 'amenities_score', 
        'localidad_encoded', 'barrio_freq_encoded'
    ]
    numeric_columns_to_scale = [col for col in numeric_columns_to_scale if col in X_train_processed.columns]
    X_train_processed = post_transformer.scale_numeric_features_train(X_train_processed, numeric_columns_to_scale)
    
    # IMPUTACIÓN FINAL PARA ELIMINAR NaN - MÉTODO SEGURO
    X_train_processed = post_transformer.apply_final_imputation_train(X_train_processed)
    
    # 6. PROCESAMIENTO TEST
    print("\n🎯 PROCESANDO DATOS DE TEST...")
    
    X_test_df = pd.DataFrame(X_test, columns=X.columns, index=X_test.index)
    y_test_series = pd.Series(y_test, index=X_test.index)
    
    X_test_processed = post_transformer.handle_missing_values_test(X_test_df)
    X_test_processed = post_transformer.apply_knn_imputer_test(X_test_processed)
    X_test_processed = post_transformer.filter_extreme_outliers_test(X_test_processed)
    y_test_series = y_test_series.loc[X_test_processed.index]
    
    X_test_processed = post_transformer.truncate_outliers_test(X_test_processed)
    X_test_processed = post_transformer.apply_target_encoding_test(X_test_processed)
    X_test_processed = post_transformer.apply_frequency_encoding_test(X_test_processed)
    X_test_processed = post_transformer.apply_onehot_encoding(X_test_processed, ['antiguedad', 'tipo_propiedad'])
    X_test_processed = post_transformer.scale_numeric_features_test(X_test_processed)
    
    # IMPUTACIÓN FINAL PARA ELIMINAR NaN
    X_test_processed = post_transformer.apply_final_imputation_test(X_test_processed)
    
    # 7. Alinear columnas
    X_train_final, X_test_final = post_transformer.align_train_test_columns(X_train_processed, X_test_processed)
    
    # 8. Preparar datos para modelado
    print("\n📊 PREPARANDO DATOS PARA MODELADO...")
    
    def get_numeric_features(df):
        return df.select_dtypes(include=[np.number])
    
    X_train_numeric = get_numeric_features(X_train_final)
    X_test_numeric = get_numeric_features(X_test_final)
    
    common_columns = X_train_numeric.columns.intersection(X_test_numeric.columns)
    X_train_final = X_train_numeric[common_columns]
    X_test_final = X_test_numeric[common_columns]
    
    # 9. ELIMINAR COLUMNAS DE METADATOS Y VERIFICAR NaN
    columns_to_remove = ['dataset']
    for col in columns_to_remove:
        if col in X_train_final.columns:
            X_train_final = X_train_final.drop(columns=[col])
        if col in X_test_final.columns:
            X_test_final = X_test_final.drop(columns=[col])
    
    # VERIFICACIÓN FINAL DE NaN - MÉTODO MÁS ROBUSTO
    print("🔍 VERIFICACIÓN FINAL DE CALIDAD...")
    
    # Si aún hay NaN, usar fillna de manera segura
    nan_check_train = X_train_final.isnull().sum().sum()
    nan_check_test = X_test_final.isnull().sum().sum()
    
    if nan_check_train > 0:
        print(f"⚠️  Eliminando {nan_check_train} NaN restantes en train con fillna(0)")
        X_train_final = X_train_final.fillna(0)
    
    if nan_check_test > 0:
        print(f"⚠️  Eliminando {nan_check_test} NaN restantes en test con fillna(0)")
        X_test_final = X_test_final.fillna(0)
    
    # Asegurar que solo tenemos características numéricas
    X_train_final = X_train_final.select_dtypes(include=[np.number])
    X_test_final = X_test_final.select_dtypes(include=[np.number])
    
    # Recalcular common_columns después de la limpieza
    common_columns = X_train_final.columns.intersection(X_test_final.columns)
    X_train_final = X_train_final[common_columns]
    X_test_final = X_test_final[common_columns]
    
    print(f"✅ Datos finales:")
    print(f"   • X_train: {X_train_final.shape}")
    print(f"   • X_test: {X_test_final.shape}")
    print(f"   • y_train: {y_train_filtered.shape}")
    print(f"   • y_test: {y_test_series.shape}")
    print(f"   • Características: {len(common_columns)}")
    
    # VERIFICACIÓN EXTRA DE CALIDAD
    print("🔍 VERIFICACIÓN FINAL DE CALIDAD:")
    print(f"   • NaN en X_train: {X_train_final.isnull().sum().sum()}")
    print(f"   • NaN en X_test: {X_test_final.isnull().sum().sum()}")
    print(f"   • Infinitos en X_train: {np.isinf(X_train_final.values).sum()}")
    print(f"   • Infinitos en X_test: {np.isinf(X_test_final.values).sum()}")
    
    # 10. Guardar resultados
    if save_excel:
        try:
            X_train_save = X_train_final.copy()
            X_test_save = X_test_final.copy()
            
            X_train_save['dataset'] = 'train'
            X_test_save['dataset'] = 'test'
            X_train_save['log_precio_venta'] = y_train_filtered.values
            X_test_save['log_precio_venta'] = y_test_series.values
            
            combined_df = pd.concat([X_train_save, X_test_save], axis=0)
            output_path = "datos_preprocesados_sin_leakage.xlsx"
            combined_df.to_excel(output_path, index=False)
            print(f"💾 Datos guardados en: {output_path}")
        except Exception as e:
            print(f"⚠️  Error guardando Excel: {e}")
    
    return (X_train_final.values, X_test_final.values, 
            y_train_filtered.values, y_test_series.values, 
            common_columns.tolist())


# Función de conveniencia
def preprocess_bogota_apartments_corrected(file_path, test_size=0.2, random_state=42, save_excel=True):
    """
    Función corregida sin data leakage y sin NaN
    """
    return complete_preprocessing_pipeline_corrected(
        file_path, 
        test_size=test_size, 
        random_state=random_state, 
        save_excel=save_excel
    )