"""
MÓDULO DE ENTRENAMIENTO Y EVALUACIÓN DE MODELOS CORREGIDO
Manejo robusto de valores NaN y otros problemas de datos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import traceback
from datetime import datetime
import os

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class RobustModelTrainer:
    """
    Clase corregida para entrenar y evaluar modelos con manejo robusto de datos
    """
    
    def __init__(self, random_state=42):
        """
        Inicializar el entrenador de modelos
        
        Args:
            random_state (int): Semilla para reproducibilidad
        """
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
        self.predictions = {}
        self.feature_importance = {}
        self.imputer = None
        self.scaler = None
        
    def check_and_clean_data(self, X, y):
        """
        Verificar y limpiar datos de entrada de valores NaN/infinitos
        
        Args:
            X (array-like): Características
            y (array-like): Variable objetivo
            
        Returns:
            tuple: X_clean, y_clean
        """
        print("🔍 Verificando y limpiando datos...")
        
        # Convertir a arrays numpy si son DataFrames
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        # Verificar dimensiones
        if len(X) != len(y):
            raise ValueError(f"Dimensiones inconsistentes: X {len(X)} vs y {len(y)}")
        
        # Verificar y limpiar valores infinitos
        X = np.where(np.isfinite(X), X, np.nan)
        y = np.where(np.isfinite(y), y, np.nan)
        
        # Contar valores NaN antes de la limpieza
        nan_count_x = np.isnan(X).sum()
        nan_count_y = np.isnan(y).sum()
        
        print(f"📊 Valores NaN encontrados: X={nan_count_x}, y={nan_count_y}")
        
        # Eliminar filas donde y es NaN
        if nan_count_y > 0:
            valid_mask = ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
            print(f"✅ Eliminadas {np.sum(~valid_mask)} filas con NaN en y")
        
        # Crear y ajustar imputador para X
        if nan_count_x > 0:
            self.imputer = SimpleImputer(strategy='median')
            X = self.imputer.fit_transform(X)
            print("✅ Valores NaN en X imputados con mediana")
        else:
            self.imputer = SimpleImputer(strategy='median')
            self.imputer.fit(X)  # Ajustar por si acaso
            
        # Escalar características
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        print("✅ Características escaladas con StandardScaler")
        
        # Verificación final
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("❌ Aún hay valores NaN después de la limpieza")
            
        print(f"✅ Datos limpios: {X.shape[0]} muestras, {X.shape[1]} características")
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """
        Dividir datos en conjuntos de entrenamiento y prueba
        
        Args:
            X (array-like): Características
            y (array-like): Variable objetivo
            test_size (float): Proporción para test
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("📊 Dividiendo datos en train/test...")
        
        # Limpiar datos primero
        X_clean, y_clean = self.check_and_clean_data(X, y)
            
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=test_size, random_state=self.random_state, shuffle=True
        )
        
        print(f"✅ Datos divididos:")
        print(f"   • Entrenamiento: {X_train.shape[0]} muestras")
        print(f"   • Prueba: {X_test.shape[0]} muestras")
        print(f"   • Características: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """
        Inicializar modelos con sus hiperparámetros
        """
        print("🤖 Inicializando modelos...")
        
        self.models = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {},
                'description': 'Regresión lineal básica'
            },
            'Ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.1, 1, 10, 100, 1000]
                },
                'description': 'Regresión Ridge con regularización L2'
            },
            'XGBoost': {
                'model': XGBRegressor(
                    random_state=self.random_state,
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    verbosity=0,
                    n_jobs=-1
                ),
                'params': {},
                'description': 'XGBoost con parámetros optimizados'
            },
            'RandomForest': {
                'model': RandomForestRegressor(
                    random_state=self.random_state,
                    n_estimators=100,
                    max_depth=10,
                    n_jobs=-1
                ),
                'params': {},
                'description': 'Random Forest robusto'
            }
        }
        
        print("✅ Modelos inicializados:")
        for name, config in self.models.items():
            print(f"   • {name}: {config['description']}")
    
    def train_ridge_with_gridsearch(self, X_train, y_train):
        """
        Entrenar modelo Ridge con búsqueda de hiperparámetros robusta
        
        Args:
            X_train (array): Datos de entrenamiento
            y_train (array): Target de entrenamiento
            
        Returns:
            Ridge: Mejor modelo Ridge entrenado
        """
        print("🎯 Entrenando Ridge con GridSearch...")
        
        try:
            ridge = Ridge(random_state=self.random_state)
            grid_search = GridSearchCV(
                ridge,
                param_grid={'alpha': [0.1, 1, 10, 100, 1000]},
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0,
                error_score='raise'  # Para debug
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"✅ Mejor alpha para Ridge: {grid_search.best_params_['alpha']}")
            print(f"✅ Mejor score: {-grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"❌ Error en Ridge GridSearch: {e}")
            print("🔄 Usando Ridge con alpha=1.0 como fallback...")
            # Retornar modelo por defecto en caso de error
            ridge_fallback = Ridge(alpha=1.0, random_state=self.random_state)
            ridge_fallback.fit(X_train, y_train)
            return ridge_fallback
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """
        Calcular métricas de evaluación
        
        Args:
            y_true (array): Valores reales
            y_pred (array): Predicciones
            model_name (str): Nombre del modelo
            
        Returns:
            dict: Diccionario con métricas
        """
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            print(f"📈 {model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculando métricas para {model_name}: {e}")
            return {'RMSE': np.inf, 'MAE': np.inf, 'R2': -np.inf}
    
    def train_single_model(self, model_name, config, X_train, X_test, y_train, y_test):
        """
        Entrenar un solo modelo con manejo robusto de errores
        
        Args:
            model_name (str): Nombre del modelo
            config (dict): Configuración del modelo
            X_train (array): Datos de entrenamiento
            X_test (array): Datos de prueba
            y_train (array): Target de entrenamiento
            y_test (array): Target de prueba
        """
        try:
            print(f"\n🎯 Entrenando {model_name}...")
            
            if model_name == 'Ridge':
                # Entrenamiento especial para Ridge con GridSearch
                model = self.train_ridge_with_gridsearch(X_train, y_train)
            else:
                # Entrenamiento normal para otros modelos
                model = config['model']
                model.fit(X_train, y_train)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            metrics = self.calculate_metrics(y_test, y_pred, model_name)
            
            # Guardar resultados
            self.models[model_name]['trained_model'] = model
            self.metrics[model_name] = metrics
            self.predictions[model_name] = {
                'y_true': y_test,
                'y_pred': y_pred
            }
            
            # Calcular importancia de características si el modelo lo soporta
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Para modelos lineales, usar coeficientes absolutos como importancia
                self.feature_importance[model_name] = np.abs(model.coef_)
            
            print(f"✅ {model_name} entrenado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error entrenando {model_name}: {e}")
            # traceback.print_exc()  # Comentado para evitar output muy largo
            return False
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Entrenar todos los modelos y evaluarlos
        
        Args:
            X_train (array): Datos de entrenamiento
            X_test (array): Datos de prueba
            y_train (array): Target de entrenamiento
            y_test (array): Target de prueba
        """
        print("\n🚀 INICIANDO ENTRENAMIENTO DE MODELOS")
        print("=" * 60)
        
        self.metrics = {}
        self.predictions = {}
        self.feature_importance = {}
        
        successful_models = 0
        
        for model_name, config in self.models.items():
            success = self.train_single_model(model_name, config, X_train, X_test, y_train, y_test)
            if success:
                successful_models += 1
        
        print(f"\n📊 Resumen: {successful_models}/{len(self.models)} modelos entrenados exitosamente")
        
        if successful_models == 0:
            raise RuntimeError("❌ Ningún modelo pudo ser entrenado")
    
    def plot_predictions_vs_actual(self, model_name, y_true, y_pred):
        """
        Generar gráfico de predicciones vs valores reales
        
        Args:
            model_name (str): Nombre del modelo
            y_true (array): Valores reales
            y_pred (array): Predicciones
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Scatter plot
            plt.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predicciones')
            
            # Línea de identidad perfecta
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Línea de identidad')
            
            plt.xlabel('Valores Reales')
            plt.ylabel('Predicciones')
            plt.title(f'{model_name} - Predicciones vs Valores Reales')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Añadir métricas en el gráfico
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            plt.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error generando gráfico para {model_name}: {e}")
    
    def generate_all_plots(self):
        """
        Generar todos los gráficos de predicciones vs reales
        """
        print("\n📊 Generando gráficos de evaluación...")
        
        plots_generated = 0
        for model_name, pred_data in self.predictions.items():
            try:
                self.plot_predictions_vs_actual(
                    model_name, 
                    pred_data['y_true'], 
                    pred_data['y_pred']
                )
                plots_generated += 1
            except Exception as e:
                print(f"❌ Error generando gráfico para {model_name}: {e}")
        
        print(f"✅ {plots_generated} gráficos generados exitosamente")
    
    def plot_feature_importance(self, feature_names, top_n=15):
        """
        Graficar importancia de características para modelos que lo soportan
        
        Args:
            feature_names (list): Nombres de las características
            top_n (int): Número top de características a mostrar
        """
        if not self.feature_importance:
            print("⚠️  No hay modelos con importancia de características")
            return
        
        try:
            n_models = len(self.feature_importance)
            fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
            fig.suptitle('IMPORTANCIA DE CARACTERÍSTICAS POR MODELO', fontsize=16, fontweight='bold')
            
            if n_models == 1:
                axes = [axes]
            
            for idx, (model_name, importance) in enumerate(self.feature_importance.items()):
                if len(importance) != len(feature_names):
                    print(f"⚠️  Dimensiones no coinciden para {model_name}: importancia {len(importance)} vs features {len(feature_names)}")
                    continue
                    
                # Crear DataFrame para mejor visualización
                feat_imp_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=True).tail(top_n)
                
                # Graficar
                axes[idx].barh(feat_imp_df['feature'], feat_imp_df['importance'], color='skyblue')
                axes[idx].set_title(f'{model_name}', fontweight='bold')
                axes[idx].set_xlabel('Importancia')
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error generando gráfico de importancia: {e}")
    
    def save_models(self, model_dir='modelos_entrenados'):
        """
        Guardar modelos entrenados usando joblib
        
        Args:
            model_dir (str): Directorio donde guardar los modelos
        """
        print(f"\n💾 Guardando modelos en '{model_dir}'...")
        
        # Crear directorio si no existe
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"✅ Directorio '{model_dir}' creado")
        
        saved_models = []
        
        for model_name, config in self.models.items():
            if 'trained_model' in config:
                try:
                    # Nombre del archivo con timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{model_name}_{timestamp}.joblib"
                    filepath = os.path.join(model_dir, filename)
                    
                    # Guardar modelo
                    joblib.dump(config['trained_model'], filepath)
                    saved_models.append((model_name, filepath))
                    
                    print(f"✅ {model_name} guardado en: {filepath}")
                    
                except Exception as e:
                    print(f"❌ Error guardando {model_name}: {e}")
        
        # Guardar métricas en un archivo CSV
        if self.metrics:
            metrics_df = pd.DataFrame(self.metrics).T
            metrics_path = os.path.join(model_dir, f'metricas_{timestamp}.csv')
            metrics_df.to_csv(metrics_path)
            print(f"✅ Métricas guardadas en: {metrics_path}")
            
        # Guardar preprocesadores
        if self.imputer:
            imputer_path = os.path.join(model_dir, f'imputer_{timestamp}.joblib')
            joblib.dump(self.imputer, imputer_path)
            print(f"✅ Imputer guardado en: {imputer_path}")
            
        if self.scaler:
            scaler_path = os.path.join(model_dir, f'scaler_{timestamp}.joblib')
            joblib.dump(self.scaler, scaler_path)
            print(f"✅ Scaler guardado en: {scaler_path}")
        
        return saved_models
    
    def print_final_metrics(self):
        """
        Imprimir métricas finales de todos los modelos en formato tabular
        """
        if not self.metrics:
            print("⚠️  No hay métricas para mostrar")
            return
        
        print("\n" + "=" * 70)
        print("📊 MÉTRICAS FINALES DE TODOS LOS MODELOS")
        print("=" * 70)
        
        # Crear DataFrame con las métricas
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df = metrics_df.round(4)
        
        # Mostrar tabla formateada
        print(metrics_df.to_string())
        
        # Encontrar mejor modelo por R²
        if len(metrics_df) > 0:
            best_model = metrics_df['R2'].idxmax()
            best_r2 = metrics_df.loc[best_model, 'R2']
            
            print(f"\n🏆 MEJOR MODELO: {best_model} (R² = {best_r2:.4f})")
    
    def train_and_evaluate_models(self, X, y, feature_names=None, save_models=True):
        """
        Función principal que ejecuta todo el pipeline
        
        Args:
            X (array-like): Características
            y (array-like): Variable objetivo
            feature_names (list): Nombres de las características
            save_models (bool): Si guardar los modelos en disco
            
        Returns:
            dict: Diccionario con métricas de todos los modelos
        """
        print("🚀 INICIANDO ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
        print("=" * 60)
        
        try:
            # 1. Dividir datos
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # 2. Inicializar modelos
            self.initialize_models()
            
            # 3. Entrenar y evaluar modelos
            self.train_models(X_train, X_test, y_train, y_test)
            
            # 4. Generar gráficos
            self.generate_all_plots()
            
            # 5. Graficar importancia de características si hay nombres
            if feature_names is not None:
                self.plot_feature_importance(feature_names)
            
            # 6. Guardar modelos
            if save_models:
                self.save_models()
            
            # 7. Mostrar métricas finales
            self.print_final_metrics()
            
            print("\n🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
            
            return self.metrics
            
        except Exception as e:
            print(f"❌ Error en el pipeline de entrenamiento: {e}")
            traceback.print_exc()
            return {}

# Función de conveniencia para uso rápido
def train_and_evaluate_models(X, y, feature_names=None, random_state=42, save_models=True):
    """
    Función principal para entrenar y evaluar modelos
    
    Args:
        X (array-like): Características preprocesadas
        y (array-like): Variable objetivo
        feature_names (list): Nombres de las características
        random_state (int): Semilla para reproducibilidad
        save_models (bool): Si guardar los modelos
        
    Returns:
        dict: Métricas de evaluación por modelo
    """
    trainer = RobustModelTrainer(random_state=random_state)
    metrics = trainer.train_and_evaluate_models(X, y, feature_names, save_models)
    return metrics

# Función para diagnóstico de datos
def diagnose_data_issues(X, y):
    """
    Diagnosticar problemas comunes en los datos
    
    Args:
        X (array-like): Características
        y (array-like): Variable objetivo
    """
    print("🔍 DIAGNÓSTICO DE DATOS")
    print("=" * 50)
    
    # Convertir a arrays numpy si es necesario
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    print(f"📊 Dimensiones: X {X.shape}, y {y.shape}")
    print(f"🔢 NaN en X: {np.isnan(X).sum()}")
    print(f"🔢 NaN en y: {np.isnan(y).sum()}")
    print(f"∞ Infinitos en X: {np.isinf(X).sum()}")
    print(f"∞ Infinitos en y: {np.isinf(y).sum()}")
    
    # Verificar rangos
    if len(X) > 0:
        print(f"📈 Rango de X: [{X.min():.4f}, {X.max():.4f}]")
    if len(y) > 0:
        print(f"📈 Rango de y: [{y.min():.4f}, {y.max():.4f}]")
    
    # Verificar variabilidad
    if len(X) > 0 and X.shape[1] > 0:
        std_dev = np.std(X, axis=0)
        zero_std_features = np.sum(std_dev == 0)
        if zero_std_features > 0:
            print(f"⚠️  Características con desviación estándar cero: {zero_std_features}")

# Ejemplo de uso
if __name__ == "__main__":
    # Este bloque se ejecuta solo cuando el módulo se corre directamente
    print("🔧 MÓDULO DE ENTRENAMIENTO DE MODELOS - VERSIÓN CORREGIDA")
    print("📝 Ejecuta train_and_evaluate_models(X, y) para usar el módulo")
    
    # Ejemplo de uso con datos dummy (para testing)
    try:
        from sklearn.datasets import make_regression
        
        print("\n🧪 EJECUTANDO EJEMPLO CON DATOS DUMMY...")
        
        # Generar datos de ejemplo
        X_dummy, y_dummy = make_regression(
            n_samples=1000, 
            n_features=10, 
            noise=0.1, 
            random_state=42
        )
        
        # Añadir algunos NaN para probar la robustez
        X_dummy[5, 2] = np.nan
        X_dummy[10, 5] = np.nan
        
        feature_names_dummy = [f'feature_{i}' for i in range(X_dummy.shape[1])]
        
        # Ejecutar diagnóstico
        diagnose_data_issues(X_dummy, y_dummy)
        
        # Ejecutar entrenamiento
        metrics = train_and_evaluate_models(
            X_dummy, 
            y_dummy, 
            feature_names=feature_names_dummy,
            save_models=False
        )
        
    except Exception as e:
        print(f"❌ Error en ejemplo: {e}")
        traceback.print_exc()