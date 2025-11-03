"""
MÓDULO COMPLETO DE EVALUACIÓN Y COMPARACIÓN DE MODELOS
Incluye gráficos avanzados y comparaciones detalladas entre modelos
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
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class ComprehensiveModelTrainer:
    """
    Clase completa para entrenar, evaluar y comparar modelos con visualizaciones avanzadas
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
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        
    def check_and_clean_data(self, X, y):
        """
        Verificar y limpiar datos de entrada de valores NaN/infinitos
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
            self.imputer.fit(X)
            
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
        """
        print("📊 Dividiendo datos en train/test...")
        
        # Limpiar datos primero
        X_clean, y_clean = self.check_and_clean_data(X, y)
            
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=test_size, random_state=self.random_state, shuffle=True
        )
        
        self.X_test = X_test
        self.y_test = y_test
        
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
                'description': 'Regresión lineal básica',
                'color': '#1f77b4'
            },
            'Ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {'alpha': [0.1, 1, 10, 100, 1000]},
                'description': 'Regresión Ridge con regularización L2',
                'color': '#ff7f0e'
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
                'description': 'XGBoost con parámetros optimizados',
                'color': '#2ca02c'
            },
            'RandomForest': {
                'model': RandomForestRegressor(
                    random_state=self.random_state,
                    n_estimators=100,
                    max_depth=10,
                    n_jobs=-1
                ),
                'params': {},
                'description': 'Random Forest robusto',
                'color': '#d62728'
            }
        }
        
        print("✅ Modelos inicializados:")
        for name, config in self.models.items():
            print(f"   • {name}: {config['description']}")
    
    def train_ridge_with_gridsearch(self, X_train, y_train):
        """
        Entrenar modelo Ridge con búsqueda de hiperparámetros
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
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"✅ Mejor alpha para Ridge: {grid_search.best_params_['alpha']}")
            print(f"✅ Mejor score: {-grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"❌ Error en Ridge GridSearch: {e}")
            print("🔄 Usando Ridge con alpha=1.0 como fallback...")
            ridge_fallback = Ridge(alpha=1.0, random_state=self.random_state)
            ridge_fallback.fit(X_train, y_train)
            return ridge_fallback
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """
        Calcular métricas de evaluación avanzadas
        """
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Métricas adicionales
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            residuals = y_true - y_pred
            residual_std = np.std(residuals)
            
            metrics = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape,
                'Residual_Std': residual_std,
                'Max_Error': np.max(np.abs(residuals))
            }
            
            print(f"📈 {model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculando métricas para {model_name}: {e}")
            return {'RMSE': np.inf, 'MAE': np.inf, 'R2': -np.inf, 'MAPE': np.inf, 'Residual_Std': np.inf, 'Max_Error': np.inf}
    
    def train_single_model(self, model_name, config, X_train, X_test, y_train, y_test):
        """
        Entrenar un solo modelo con manejo robusto de errores
        """
        try:
            print(f"\n🎯 Entrenando {model_name}...")
            
            if model_name == 'Ridge':
                model = self.train_ridge_with_gridsearch(X_train, y_train)
            else:
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
                'y_pred': y_pred,
                'residuals': y_test - y_pred
            }
            
            # Calcular importancia de características
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                self.feature_importance[model_name] = np.abs(model.coef_)
            
            print(f"✅ {model_name} entrenado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error entrenando {model_name}: {e}")
            return False
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Entrenar todos los modelos y evaluarlos
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
    
    def plot_comprehensive_comparison(self):
        """
        Gráfico completo de comparación entre todos los modelos
        """
        if not self.metrics:
            print("⚠️  No hay métricas para comparar")
            return
        
        print("\n📊 Generando gráficos comprehensivos de comparación...")
        
        # Crear figura con múltiples subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)
        
        # 1. Comparación de métricas principales
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_metrics_comparison(ax1)
        
        # 2. Predicciones vs reales para todos los modelos
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_all_predictions_vs_actual([ax2, ax3, ax4])
        
        # 3. Residuales para todos los modelos
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_all_residuals([ax5, ax6, ax7])
        
        plt.tight_layout()
        plt.show()
    
    def _plot_metrics_comparison(self, ax):
        """
        Gráfico de comparación de métricas entre modelos
        """
        metrics_df = pd.DataFrame(self.metrics).T
        
        # Preparar datos para el gráfico
        models = list(metrics_df.index)
        x = np.arange(len(models))
        width = 0.2
        
        # Gráfico de barras para múltiples métricas
        metrics_to_plot = ['RMSE', 'MAE', 'R2']
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        for i, metric in enumerate(metrics_to_plot):
            values = metrics_df[metric].values
            if metric == 'R2':
                # Para R², usar escala diferente
                values = values * 10  # Escalar para mejor visualización
            ax.bar(x + i*width, values, width, label=metric, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel('Valores de Métricas')
        ax.set_title('Comparación de Métricas entre Modelos', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for i, model in enumerate(models):
            for j, metric in enumerate(metrics_to_plot):
                value = metrics_df.loc[model, metric]
                if metric == 'R2':
                    value = value  # Valor original para el texto
                ax.text(i + j*width, metrics_df.loc[model, metric] + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_all_predictions_vs_actual(self, axes):
        """
        Gráfico de predicciones vs valores reales para múltiples modelos
        """
        model_keys = list(self.predictions.keys())[:3]  # Tomar primeros 3 modelos
        
        for idx, model_name in enumerate(model_keys):
            if idx < len(axes):
                pred_data = self.predictions[model_name]
                y_true, y_pred = pred_data['y_true'], pred_data['y_pred']
                
                ax = axes[idx]
                ax.scatter(y_true, y_pred, alpha=0.6, 
                          color=self.models[model_name]['color'], s=20)
                
                # Línea de identidad
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                ax.set_xlabel('Valores Reales')
                ax.set_ylabel('Predicciones')
                ax.set_title(f'{model_name}\nPredicciones vs Reales', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Añadir R² en el gráfico
                r2 = self.metrics[model_name]['R2']
                ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
                       transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_all_residuals(self, axes):
        """
        Gráfico de residuales para múltiples modelos
        """
        model_keys = list(self.predictions.keys())[:3]  # Tomar primeros 3 modelos
        
        for idx, model_name in enumerate(model_keys):
            if idx < len(axes):
                pred_data = self.predictions[model_name]
                residuals = pred_data['residuals']
                y_pred = pred_data['y_pred']
                
                ax = axes[idx]
                ax.scatter(y_pred, residuals, alpha=0.6, 
                          color=self.models[model_name]['color'], s=20)
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
                
                ax.set_xlabel('Predicciones')
                ax.set_ylabel('Residuales')
                ax.set_title(f'{model_name}\nAnálisis de Residuales', fontsize=10)
                ax.grid(True, alpha=0.3)
    
    def plot_metrics_radar_chart(self):
        """
        Gráfico radar chart para comparación visual de métricas
        """
        if len(self.metrics) < 2:
            print("⚠️  Se necesitan al menos 2 modelos para el radar chart")
            return
        
        # Preparar datos para radar chart
        metrics_df = pd.DataFrame(self.metrics).T
        models = metrics_df.index.tolist()
        
        # Normalizar métricas (mejor si es más cercano a 1)
        normalized_df = metrics_df.copy()
        for col in ['R2']:
            normalized_df[col] = metrics_df[col]  # R² ya está en escala 0-1
        
        for col in ['RMSE', 'MAE', 'MAPE']:
            # Invertir para que mejor sea más cercano a 1
            normalized_df[col] = 1 / (1 + metrics_df[col])
        
        # Métricas para el radar
        radar_metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el círculo
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for model in models:
            values = normalized_df.loc[model, radar_metrics].values.tolist()
            values += values[:1]  # Cerrar el círculo
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=model, color=self.models[model]['color'])
            ax.fill(angles, values, alpha=0.1, color=self.models[model]['color'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Comparación de Modelos - Radar Chart\n(Valores normalizados, mejor = más cercano a 1)', 
                    size=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.show()
    
    def plot_error_distribution(self):
        """
        Distribución de errores para todos los modelos
        """
        if not self.predictions:
            print("⚠️  No hay predicciones para graficar")
            return
        
        plt.figure(figsize=(12, 6))
        
        for model_name, pred_data in self.predictions.items():
            errors = np.abs(pred_data['y_true'] - pred_data['y_pred'])
            plt.hist(errors, bins=50, alpha=0.6, 
                    label=model_name, color=self.models[model_name]['color'])
        
        plt.xlabel('Error Absoluto')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Errores Absolutos por Modelo', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Escala logarítmica para mejor visualización
        plt.show()
    
    def plot_cumulative_error(self):
        """
        Gráfico de error acumulado por modelo
        """
        if not self.predictions:
            print("⚠️  No hay predicciones para graficar")
            return
        
        plt.figure(figsize=(12, 6))
        
        for model_name, pred_data in self.predictions.items():
            errors = np.abs(pred_data['y_true'] - pred_data['y_pred'])
            sorted_errors = np.sort(errors)
            cumulative = np.cumsum(sorted_errors) / np.sum(sorted_errors)
            
            plt.plot(sorted_errors, cumulative, 
                    label=model_name, color=self.models[model_name]['color'], linewidth=2)
        
        plt.xlabel('Error Absoluto')
        plt.ylabel('Error Acumulado Normalizado')
        plt.title('Distribución Acumulativa de Errores', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_feature_importance_comparison(self, top_n=10):
        """
        Comparación de importancia de características entre modelos
        """
        if not self.feature_importance or self.feature_names is None:
            print("⚠️  No hay datos de importancia de características")
            return
        
        # Crear DataFrame con importancias
        importance_df = pd.DataFrame()
        for model_name, importance in self.feature_importance.items():
            if len(importance) == len(self.feature_names):
                importance_df[model_name] = importance
        
        if importance_df.empty:
            print("⚠️  No se pudo crear DataFrame de importancias")
            return
        
        # Tomar top N características más importantes en promedio
        importance_df['feature'] = self.feature_names
        importance_df['mean_importance'] = importance_df.drop('feature', axis=1).mean(axis=1)
        top_features = importance_df.nlargest(top_n, 'mean_importance')['feature'].tolist()
        
        # Filtrar solo top features
        plot_df = importance_df[importance_df['feature'].isin(top_features)]
        
        # Crear gráfico
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Gráfico 1: Heatmap de importancias
        heatmap_data = plot_df.drop(['feature', 'mean_importance'], axis=1).T
        heatmap_data.columns = plot_df['feature']
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=axes[0], cbar_kws={'label': 'Importancia'})
        axes[0].set_title('Heatmap de Importancia de Características\n(Top {} Features)'.format(top_n), 
                         fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Gráfico 2: Barras de importancia promedio
        plot_df.set_index('feature')['mean_importance'].sort_values().plot(
            kind='barh', ax=axes[1], color='skyblue', alpha=0.7
        )
        axes[1].set_title('Importancia Promedio de Características\n(Top {} Features)'.format(top_n), 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Importancia Promedio')
        
        plt.tight_layout()
        plt.show()
    
    def plot_residual_analysis(self):
        """
        Análisis completo de residuales para todos los modelos
        """
        if not self.predictions:
            print("⚠️  No hay predicciones para análisis de residuales")
            return
        
        n_models = len(self.predictions)
        fig, axes = plt.subplots(n_models, 3, figsize=(18, 5 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, pred_data) in enumerate(self.predictions.items()):
            residuals = pred_data['residuals']
            y_pred = pred_data['y_pred']
            
            # Histograma de residuales
            axes[idx, 0].hist(residuals, bins=50, alpha=0.7, 
                             color=self.models[model_name]['color'], edgecolor='black')
            axes[idx, 0].axvline(0, color='red', linestyle='--')
            axes[idx, 0].set_title(f'{model_name} - Distribución de Residuales')
            axes[idx, 0].set_xlabel('Residuales')
            axes[idx, 0].set_ylabel('Frecuencia')
            
            # QQ plot para normalidad
            stats.probplot(residuals, dist="norm", plot=axes[idx, 1])
            axes[idx, 1].set_title(f'{model_name} - QQ Plot (Normalidad)')
            
            # Residuales vs predicciones
            axes[idx, 2].scatter(y_pred, residuals, alpha=0.6, 
                                color=self.models[model_name]['color'])
            axes[idx, 2].axhline(0, color='red', linestyle='--')
            axes[idx, 2].set_title(f'{model_name} - Residuales vs Predicciones')
            axes[idx, 2].set_xlabel('Predicciones')
            axes[idx, 2].set_ylabel('Residuales')
        
        plt.tight_layout()
        plt.show()
    
    def generate_model_performance_dashboard(self):
        """
        Generar dashboard completo de rendimiento de modelos
        """
        print("\n📈 GENERANDO DASHBOARD COMPLETO DE MODELOS...")
        
        # 1. Gráfico de comparación comprehensiva
        self.plot_comprehensive_comparison()
        
        # 2. Radar chart de métricas
        self.plot_metrics_radar_chart()
        
        # 3. Distribución de errores
        self.plot_error_distribution()
        
        # 4. Error acumulado
        self.plot_cumulative_error()
        
        # 5. Análisis de residuales
        self.plot_residual_analysis()
        
        # 6. Importancia de características (si está disponible)
        if self.feature_importance and self.feature_names:
            self.plot_feature_importance_comparison()
    
    def save_models(self, model_dir='modelos_entrenados'):
        """
        Guardar modelos entrenados y métricas
        """
        print(f"\n💾 Guardando modelos en '{model_dir}'...")
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        saved_models = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, config in self.models.items():
            if 'trained_model' in config:
                try:
                    filename = f"{model_name}_{timestamp}.joblib"
                    filepath = os.path.join(model_dir, filename)
                    joblib.dump(config['trained_model'], filepath)
                    saved_models.append((model_name, filepath))
                    print(f"✅ {model_name} guardado en: {filepath}")
                except Exception as e:
                    print(f"❌ Error guardando {model_name}: {e}")
        
        # Guardar métricas
        if self.metrics:
            metrics_df = pd.DataFrame(self.metrics).T
            metrics_path = os.path.join(model_dir, f'metricas_{timestamp}.csv')
            metrics_df.to_csv(metrics_path)
            print(f"✅ Métricas guardadas en: {metrics_path}")
            
            # Guardar gráfico de métricas
            plt.figure(figsize=(10, 6))
            metrics_df[['RMSE', 'MAE', 'R2']].plot(kind='bar', alpha=0.7)
            plt.title('Comparación de Métricas por Modelo')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f'metricas_comparacion_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return saved_models
    
    def print_detailed_report(self):
        """
        Imprimir reporte detallado de todos los modelos
        """
        if not self.metrics:
            print("⚠️  No hay métricas para reportar")
            return
        
        print("\n" + "=" * 80)
        print("📊 REPORTE DETALLADO DE MODELOS")
        print("=" * 80)
        
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df = metrics_df.round(4)
        
        print("\nMÉTRICAS DETALLADAS:")
        print(metrics_df.to_string())
        
        # Análisis comparativo
        print("\n🏆 ANÁLISIS COMPARATIVO:")
        
        # Mejor modelo por cada métrica
        best_rmse = metrics_df['RMSE'].idxmin()
        best_mae = metrics_df['MAE'].idxmin()
        best_r2 = metrics_df['R2'].idxmax()
        best_mape = metrics_df['MAPE'].idxmin()
        
        print(f"• Mejor RMSE: {best_rmse} ({metrics_df.loc[best_rmse, 'RMSE']:.4f})")
        print(f"• Mejor MAE: {best_mae} ({metrics_df.loc[best_mae, 'MAE']:.4f})")
        print(f"• Mejor R²: {best_r2} ({metrics_df.loc[best_r2, 'R2']:.4f})")
        print(f"• Mejor MAPE: {best_mape} ({metrics_df.loc[best_mape, 'MAPE']:.2f}%)")
        
        # Modelo recomendado (basado en R² principalmente)
        best_overall = best_r2
        print(f"\n🎯 MODELO RECOMENDADO: {best_overall}")
        print(f"   Justificación: Mayor poder predictivo (R² = {metrics_df.loc[best_overall, 'R2']:.4f})")
    
    def train_and_evaluate_models(self, X, y, feature_names=None, save_models=True):
        """
        Función principal que ejecuta todo el pipeline
        """
        print("🚀 INICIANDO ENTRENAMIENTO Y EVALUACIÓN COMPLETA DE MODELOS")
        print("=" * 60)
        
        self.feature_names = feature_names
        
        try:
            # 1. Dividir datos
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # 2. Inicializar modelos
            self.initialize_models()
            
            # 3. Entrenar y evaluar modelos
            self.train_models(X_train, X_test, y_train, y_test)
            
            # 4. Generar dashboard completo
            self.generate_model_performance_dashboard()
            
            # 5. Guardar modelos
            if save_models:
                self.save_models()
            
            # 6. Imprimir reporte detallado
            self.print_detailed_report()
            
            print("\n🎉 ENTRENAMIENTO Y EVALUACIÓN COMPLETADOS EXITOSAMENTE")
            
            return self.metrics
            
        except Exception as e:
            print(f"❌ Error en el pipeline de entrenamiento: {e}")
            traceback.print_exc()
            return {}

# Función de conveniencia para uso rápido
def train_and_evaluate_models(X, y, feature_names=None, random_state=42, save_models=True):
    """
    Función principal para entrenar y evaluar modelos
    """
    trainer = ComprehensiveModelTrainer(random_state=random_state)
    metrics = trainer.train_and_evaluate_models(X, y, feature_names, save_models)
    return metrics

# Función para diagnóstico de datos
def diagnose_data_issues(X, y):
    """
    Diagnosticar problemas comunes en los datos
    """
    print("🔍 DIAGNÓSTICO DE DATOS")
    print("=" * 50)
    
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    print(f"📊 Dimensiones: X {X.shape}, y {y.shape}")
    print(f"🔢 NaN en X: {np.isnan(X).sum()}")
    print(f"🔢 NaN en y: {np.isnan(y).sum()}")
    print(f"∞ Infinitos en X: {np.isinf(X).sum()}")
    print(f"∞ Infinitos en y: {np.isinf(y).sum()}")
    
    if len(X) > 0:
        print(f"📈 Rango de X: [{X.min():.4f}, {X.max():.4f}]")
    if len(y) > 0:
        print(f"📈 Rango de y: [{y.min():.4f}, {y.max():.4f}]")

# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 MÓDULO COMPLETO DE EVALUACIÓN DE MODELOS")
    print("📝 Ejecuta train_and_evaluate_models(X, y) para usar el módulo")
    
    # Ejemplo con datos dummy
    try:
        from sklearn.datasets import make_regression
        
        print("\n🧪 EJECUTANDO EJEMPLO CON DATOS DUMMY...")
        
        X_dummy, y_dummy = make_regression(
            n_samples=1000, n_features=10, noise=0.1, random_state=42
        )
        
        feature_names_dummy = [f'feature_{i}' for i in range(X_dummy.shape[1])]
        
        diagnose_data_issues(X_dummy, y_dummy)
        
        metrics = train_and_evaluate_models(
            X_dummy, y_dummy, 
            feature_names=feature_names_dummy,
            save_models=False
        )
        
    except Exception as e:
        print(f"❌ Error en ejemplo: {e}")
        traceback.print_exc()