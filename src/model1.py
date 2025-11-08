"""
MÓDULO UNIFICADO DE MODELADO - COMBINACIÓN OPTIMIZADA Y COMPLETA
Integra entrenamiento rápido con evaluación comprehensiva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.impute import SimpleImputer
import joblib
import warnings
import traceback
from datetime import datetime
import os
import time
from scipy import stats

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class UnifiedModelTrainer:
    """
    Entrenador unificado que combina velocidad y evaluación comprehensiva
    """
    
    def __init__(self, random_state=42, n_splits=5, mode='balanced'):
        """
        Inicializar el entrenador unificado
        
        Args:
            random_state (int): Semilla para reproducibilidad
            n_splits (int): Número de folds para validación cruzada
            mode (str): 'fast', 'balanced', o 'comprehensive'
        """
        self.random_state = random_state
        self.n_splits = n_splits
        self.mode = mode
        self.models = {}
        self.metrics = {}
        self.cv_results = {}
        self.predictions = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
        # Configurar según el modo
        self._configure_mode()
    
    def _configure_mode(self):
        """Configurar parámetros según el modo seleccionado"""
        if self.mode == 'fast':
            self.n_estimators = 100
            self.cv_folds = 3
            self.grid_search_cv = 2
            self.verbose = 0
        elif self.mode == 'balanced':
            self.n_estimators = 200
            self.cv_folds = 5
            self.grid_search_cv = 3
            self.verbose = 1
        else:  # comprehensive
            self.n_estimators = 300
            self.cv_folds = 5
            self.grid_search_cv = 5
            self.verbose = 1
    
    def initialize_models(self):
        """
        Inicializar modelos según el modo seleccionado
        """
        print(f"🤖 Inicializando modelos en modo {self.mode.upper()}...")
        
        base_config = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {},
                'description': 'Regresión lineal',
                'color': '#1f77b4'
            },
            'Ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {'alpha': [0.1, 1, 10, 100]},
                'description': 'Regresión Ridge',
                'color': '#ff7f0e'
            },
            'Lasso': {
                'model': Lasso(random_state=self.random_state, max_iter=5000),
                'params': {'alpha': [0.001, 0.01, 0.1, 1]},
                'description': 'Regresión Lasso',
                'color': '#9467bd'
            },
            'RandomForest': {
                'model': RandomForestRegressor(
                    random_state=self.random_state,
                    n_jobs=-1,
                    n_estimators=self.n_estimators
                ),
                'params': {
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'description': f'Random Forest ({self.n_estimators} trees)',
                'color': '#d62728'
            },
            'XGBoost': {
                'model': XGBRegressor(
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=0,
                    n_estimators=self.n_estimators
                ),
                'params': {
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0]
                },
                'description': f'XGBoost ({self.n_estimators} trees)',
                'color': '#2ca02c'
            }
        }
        
        # Ajustar configuración según el modo
        if self.mode == 'fast':
            # Modelos más simples para velocidad
            self.models = {
                'LinearRegression': base_config['LinearRegression'],
                'Ridge': base_config['Ridge'],
                'RandomForest': base_config['RandomForest'],
                'XGBoost': base_config['XGBoost']
            }
        elif self.mode == 'balanced':
            self.models = base_config
        else:  # comprehensive
            self.models = base_config
            # Añadir modelos adicionales para modo comprehensivo
            self.models['GradientBoosting'] = {
                'model': GradientBoostingRegressor(
                    random_state=self.random_state,
                    n_estimators=self.n_estimators
                ),
                'params': {
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 4],
                    'subsample': [0.8, 1.0]
                },
                'description': f'Gradient Boosting ({self.n_estimators} trees)',
                'color': '#8c564b'
            }
            self.models['ElasticNet'] = {
                'model': ElasticNet(random_state=self.random_state, max_iter=5000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1],
                    'l1_ratio': [0.1, 0.5, 0.9]
                },
                'description': 'Elastic Net',
                'color': '#e377c2'
            }
        
        print(f"✅ {len(self.models)} modelos inicializados:")
        for name, config in self.models.items():
            print(f"   • {name}: {config['description']}")
    
    def clean_data(self, X, y):
        """
        Limpieza robusta de datos
        """
        # Convertir a numpy si es necesario
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Manejar valores infinitos
        X = np.where(np.isfinite(X), X, np.nan)
        y = np.where(np.isfinite(y), y, np.nan)
        
        # Eliminar filas con NaN en y
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Eliminar filas con demasiados NaN en X
        nan_mask_x = np.isnan(X).sum(axis=1) / X.shape[1] < 0.5  # Menos del 50% NaN
        X = X[nan_mask_x]
        y = y[nan_mask_x]
        
        # Imputar valores faltantes
        if len(X) > 0:
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        return X, y
    
    def perform_cross_validation(self, model, X, y, model_name):
        """
        Validación cruzada robusta
        """
        try:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            cv_scores = cross_val_score(
                model, X, y, cv=kf, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1
            )
            
            cv_rmse_scores = np.sqrt(-cv_scores)
            
            cv_results = {
                'cv_rmse_mean': cv_rmse_scores.mean(),
                'cv_rmse_std': cv_rmse_scores.std(),
                'cv_rmse_scores': cv_rmse_scores,
                'cv_r2_mean': cross_val_score(model, X, y, cv=kf, scoring='r2').mean()
            }
            
            print(f"   ✅ CV RMSE: {cv_results['cv_rmse_mean']:.4f} ± {cv_results['cv_rmse_std']:.4f}")
            print(f"   ✅ CV R²: {cv_results['cv_r2_mean']:.4f}")
            
            return cv_results
            
        except Exception as e:
            print(f"   ❌ Error en CV para {model_name}: {e}")
            return None
    
    def perform_hyperparameter_tuning(self, model_name, config, X_train, y_train):
        """
        Búsqueda de hiperparámetros optimizada
        """
        if not config['params']:
            return config['model']
        
        print(f"   ⚡ Sintonizando {model_name}...")
        
        try:
            grid_search = GridSearchCV(
                config['model'],
                param_grid=config['params'],
                cv=self.grid_search_cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=self.verbose
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"   ✅ Mejores parámetros: {grid_search.best_params_}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"   ❌ Error en sintonización de {model_name}: {e}")
            return config['model']
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, model_name):
        """
        Calcular métricas comprehensivas en escala original
        """
        try:
            # Convertir de log scale a escala original
            y_true_orig = np.expm1(y_true)
            y_pred_orig = np.expm1(y_pred)
            
            # Métricas principales
            rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
            mae = mean_absolute_error(y_true_orig, y_pred_orig)
            r2 = r2_score(y_true_orig, y_pred_orig)
            
            # Métricas adicionales
            mape = np.mean(np.abs((y_true_orig - y_pred_orig) / np.maximum(y_true_orig, 1))) * 100
            residuals = y_true_orig - y_pred_orig
            residual_std = np.std(residuals)
            max_error = np.max(np.abs(residuals))
            
            # Métricas relativas
            mean_price = np.mean(y_true_orig)
            rmse_relative = rmse / mean_price * 100
            mae_relative = mae / mean_price * 100
            
            metrics = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape,
                'Residual_Std': residual_std,
                'Max_Error': max_error,
                'RMSE_Relative_%': rmse_relative,
                'MAE_Relative_%': mae_relative,
                'Mean_Price': mean_price
            }
            
            print(f"   📊 {model_name} - R²: {r2:.4f}, RMSE: ${rmse:,.0f}, MAPE: {mape:.2f}%")
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error calculando métricas para {model_name}: {e}")
            return {
                'RMSE': np.inf, 'MAE': np.inf, 'R2': -np.inf, 'MAPE': np.inf,
                'Residual_Std': np.inf, 'Max_Error': np.inf,
                'RMSE_Relative_%': np.inf, 'MAE_Relative_%': np.inf,
                'Mean_Price': np.inf
            }
    
    def train_single_model(self, model_name, config, X_train, X_test, y_train, y_test):
        """
        Entrenamiento robusto de un solo modelo
        """
        try:
            print(f"\n🎯 Entrenando {model_name}...")
            start_time = time.time()
            
            # Limpieza de datos
            X_train_clean, y_train_clean = self.clean_data(X_train, y_train)
            X_test_clean, y_test_clean = self.clean_data(X_test, y_test)
            
            # Verificar datos suficientes
            if len(X_train_clean) < 10:
                print(f"   ❌ Datos insuficientes para entrenar {model_name}")
                return False
            
            # Ajuste de hiperparámetros
            if config['params']:
                model = self.perform_hyperparameter_tuning(model_name, config, X_train_clean, y_train_clean)
            else:
                model = config['model']
            
            # Validación cruzada
            cv_results = self.perform_cross_validation(model, X_train_clean, y_train_clean, model_name)
            
            # Entrenamiento final
            model.fit(X_train_clean, y_train_clean)
            
            # Predicciones
            y_pred_train = model.predict(X_train_clean)
            y_pred_test = model.predict(X_test_clean)
            
            # Métricas
            metrics_train = self.calculate_comprehensive_metrics(y_train_clean, y_pred_train, f"{model_name} (Train)")
            metrics_test = self.calculate_comprehensive_metrics(y_test_clean, y_pred_test, f"{model_name} (Test)")
            
            # Calcular overfitting
            overfit_gap_r2 = metrics_train['R2'] - metrics_test['R2']
            overfit_gap_rmse = metrics_test['RMSE'] - metrics_train['RMSE']
            
            # Guardar resultados
            self.models[model_name]['trained_model'] = model
            self.metrics[model_name] = {
                'train': metrics_train,
                'test': metrics_test,
                'cv': cv_results,
                'overfit_gap_r2': overfit_gap_r2,
                'overfit_gap_rmse': overfit_gap_rmse,
                'training_time': time.time() - start_time
            }
            
            self.predictions[model_name] = {
                'y_true_train': y_train_clean,
                'y_pred_train': y_pred_train,
                'y_true_test': y_test_clean,
                'y_pred_test': y_pred_test,
                'residuals_train': y_train_clean - y_pred_train,
                'residuals_test': y_test_clean - y_pred_test
            }
            
            # Importancia de características
            self._calculate_feature_importance(model, model_name)
            
            print(f"   ✅ {model_name} completado en {self.metrics[model_name]['training_time']:.1f}s")
            print(f"   📈 Overfitting R²: {overfit_gap_r2:.4f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error entrenando {model_name}: {str(e)[:100]}...")
            return False
    
    def _calculate_feature_importance(self, model, model_name):
        """Calcular importancia de características"""
        try:
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                self.feature_importance[model_name] = np.abs(model.coef_)
        except:
            pass
    
    def train_all_models(self, X_train, X_test, y_train, y_test, feature_names=None):
        """
        Entrenar todos los modelos
        """
        print(f"\n🚀 INICIANDO ENTRENAMIENTO EN MODO {self.mode.upper()}")
        print("=" * 60)
        
        self.feature_names = feature_names
        self.metrics = {}
        self.predictions = {}
        self.feature_importance = {}
        
        successful_models = 0
        
        for model_name, config in self.models.items():
            success = self.train_single_model(model_name, config, X_train, X_test, y_train, y_test)
            if success:
                successful_models += 1
        
        print(f"\n📊 Resultado: {successful_models}/{len(self.models)} modelos exitosos")
        
        if successful_models > 0:
            self._select_best_model()
        else:
            raise RuntimeError("❌ Ningún modelo pudo ser entrenado")
    
    def _select_best_model(self):
        """Seleccionar el mejor modelo basado en múltiples criterios"""
        try:
            # Puntuar modelos basado en R² test y bajo overfitting
            model_scores = {}
            
            for name, metrics in self.metrics.items():
                test_r2 = metrics['test']['R2']
                overfit_penalty = min(metrics['overfit_gap_r2'] * 10, 0.5)  # Penalizar overfitting
                score = test_r2 - overfit_penalty
                model_scores[name] = score
            
            self.best_model_name = max(model_scores, key=model_scores.get)
            self.best_model = self.models[self.best_model_name]['trained_model']
            
            best_metrics = self.metrics[self.best_model_name]['test']
            
            print(f"\n🏆 MEJOR MODELO: {self.best_model_name}")
            print(f"   • R²: {best_metrics['R2']:.4f}")
            print(f"   • RMSE: ${best_metrics['RMSE']:,.0f}")
            print(f"   • MAPE: {best_metrics['MAPE']:.2f}%")
            
        except Exception as e:
            print(f"⚠️  Error seleccionando mejor modelo: {e}")
            # Fallback al primer modelo exitoso
            for name in self.metrics.keys():
                self.best_model_name = name
                self.best_model = self.models[name]['trained_model']
                break
    
    def generate_comprehensive_report(self):
        """
        Generar reporte comprehensivo con visualizaciones
        """
        print("\n📈 GENERANDO REPORTE COMPREHENSIVO...")
        
        # 1. Comparación básica de modelos
        self.plot_model_comparison()
        
        # 2. Análisis de overfitting
        self.plot_overfitting_analysis()
        
        # 3. Predicciones vs reales
        self.plot_predictions_vs_actual()
        
        # 4. Análisis de residuales
        self.plot_residual_analysis()
        
        # 5. Importancia de características (si está disponible)
        if self.feature_importance and self.feature_names:
            self.plot_feature_importance()
        
        # 6. Métricas detalladas
        self.plot_detailed_metrics()
    
    def plot_model_comparison(self):
        """Gráfico de comparación de modelos"""
        if not self.metrics:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        models = list(self.metrics.keys())
        colors = [self.models[model]['color'] for model in models]
        
        # R² Test
        test_r2 = [self.metrics[model]['test']['R2'] for model in models]
        bars1 = ax1.bar(models, test_r2, color=colors, alpha=0.7)
        ax1.set_title('R² Score (Test)', fontweight='bold')
        ax1.set_ylabel('R²')
        ax1.tick_params(axis='x', rotation=45)
        
        # RMSE Test
        test_rmse = [self.metrics[model]['test']['RMSE'] for model in models]
        bars2 = ax2.bar(models, test_rmse, color=colors, alpha=0.7)
        ax2.set_title('RMSE (Test)', fontweight='bold')
        ax2.set_ylabel('RMSE ($)')
        ax2.tick_params(axis='x', rotation=45)
        
        # MAPE Test
        test_mape = [self.metrics[model]['test']['MAPE'] for model in models]
        bars3 = ax3.bar(models, test_mape, color=colors, alpha=0.7)
        ax3.set_title('MAPE (Test)', fontweight='bold')
        ax3.set_ylabel('MAPE (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Tiempo de entrenamiento
        train_times = [self.metrics[model]['training_time'] for model in models]
        bars4 = ax4.bar(models, train_times, color=colors, alpha=0.7)
        ax4.set_title('Tiempo de Entrenamiento', fontweight='bold')
        ax4.set_ylabel('Tiempo (s)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for ax, bars, values, fmt in [
            (ax1, bars1, test_r2, '.3f'),
            (ax2, bars2, test_rmse, ',.0f'),
            (ax3, bars3, test_mape, '.1f'),
            (ax4, bars4, train_times, '.1f')
        ]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:{fmt}}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_overfitting_analysis(self):
        """Análisis de overfitting"""
        if not self.metrics:
            return
        
        models = list(self.metrics.keys())
        train_r2 = [self.metrics[model]['train']['R2'] for model in models]
        test_r2 = [self.metrics[model]['test']['R2'] for model in models]
        overfit_gaps = [self.metrics[model]['overfit_gap_r2'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # R² Train vs Test
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, train_r2, width, label='Train', alpha=0.7)
        ax1.bar(x + width/2, test_r2, width, label='Test', alpha=0.7)
        ax1.set_xlabel('Modelos')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R²: Train vs Test', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gap de overfitting
        bars = ax2.bar(models, overfit_gaps, alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Modelos')
        ax2.set_ylabel('Gap de Overfitting (R²)')
        ax2.set_title('Análisis de Overfitting', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Colorear barras de overfitting
        for bar, gap in zip(bars, overfit_gaps):
            color = 'green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red'
            bar.set_color(color)
            
            # Añadir valor
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{gap:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self):
        """Predicciones vs valores reales para el mejor modelo"""
        if not self.predictions or not self.best_model_name:
            return
        
        pred_data = self.predictions[self.best_model_name]
        y_true_test = np.expm1(pred_data['y_true_test'])
        y_pred_test = np.expm1(pred_data['y_pred_test'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot
        ax1.scatter(y_true_test, y_pred_test, alpha=0.6, s=20)
        min_val = min(y_true_test.min(), y_pred_test.min())
        max_val = max(y_true_test.max(), y_pred_test.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax1.set_xlabel('Valor Real ($)')
        ax1.set_ylabel('Predicción ($)')
        ax1.set_title(f'{self.best_model_name}\nPredicciones vs Reales', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Añadir R²
        r2 = self.metrics[self.best_model_name]['test']['R2']
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', 
                transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Distribución de errores
        errors = y_true_test - y_pred_test
        ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_xlabel('Error de Predicción ($)')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Errores', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residual_analysis(self):
        """Análisis de residuales para el mejor modelo"""
        if not self.predictions or not self.best_model_name:
            return
        
        pred_data = self.predictions[self.best_model_name]
        residuals_test = pred_data['residuals_test']
        y_pred_test = pred_data['y_pred_test']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuales vs predicciones
        ax1.scatter(y_pred_test, residuals_test, alpha=0.6, s=20)
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Predicciones (log scale)')
        ax1.set_ylabel('Residuales (log scale)')
        ax1.set_title('Residuales vs Predicciones', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Histograma de residuales
        ax2.hist(residuals_test, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_xlabel('Residuales (log scale)')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Residuales', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # QQ plot
        stats.probplot(residuals_test, dist="norm", plot=ax3)
        ax3.set_title('QQ Plot - Normalidad de Residuales', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Residuales ordenados
        ax4.plot(range(len(residuals_test)), np.sort(residuals_test), alpha=0.7)
        ax4.axhline(y=0, color='red', linestyle='--')
        ax4.set_xlabel('Índice Ordenado')
        ax4.set_ylabel('Residuales (log scale)')
        ax4.set_title('Residuales Ordenados', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n=15):
        """Importancia de características"""
        if not self.feature_importance or not self.feature_names:
            return
        
        # Usar importancia del mejor modelo o promediar
        if self.best_model_name in self.feature_importance:
            importance = self.feature_importance[self.best_model_name]
        else:
            # Promediar importancias de todos los modelos
            all_importances = list(self.feature_importance.values())
            importance = np.mean(all_importances, axis=0)
        
        if len(importance) != len(self.feature_names):
            return
        
        # Crear DataFrame y ordenar
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'], alpha=0.7)
        plt.xlabel('Importancia')
        plt.title(f'Top {top_n} Características Más Importantes', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_detailed_metrics(self):
        """Métricas detalladas para todos los modelos"""
        if not self.metrics:
            return
        
        metrics_list = []
        for model_name, metrics in self.metrics.items():
            row = {
                'Modelo': model_name,
                'R² Train': metrics['train']['R2'],
                'R² Test': metrics['test']['R2'],
                'Overfit Gap': metrics['overfit_gap_r2'],
                'RMSE Test': metrics['test']['RMSE'],
                'MAE Test': metrics['test']['MAE'],
                'MAPE Test': metrics['test']['MAPE'],
                'Tiempo (s)': metrics['training_time']
            }
            metrics_list.append(row)
        
        metrics_df = pd.DataFrame(metrics_list)
        
        # Crear tabla visual
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=metrics_df.round(4).values,
            colLabels=metrics_df.columns,
            cellLoc='center',
            loc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('Métricas Detalladas por Modelo', fontweight='bold', fontsize=14)
        plt.show()
    
    def print_comprehensive_summary(self):
        """
        Resumen comprehensivo en consola
        """
        if not self.metrics:
            return
        
        print("\n" + "=" * 100)
        print("📊 REPORTE COMPREHENSIVO DE MODELOS")
        print("=" * 100)
        
        # Encabezado
        print(f"\n{'Modelo':<20} {'R² Train':<10} {'R² Test':<10} {'Overfit Gap':<12} {'RMSE Test':<15} {'MAPE (%)':<10} {'Tiempo(s)':<10}")
        print("-" * 100)
        
        for model_name, metrics in self.metrics.items():
            train_r2 = metrics['train']['R2']
            test_r2 = metrics['test']['R2']
            overfit_gap = metrics['overfit_gap_r2']
            rmse = metrics['test']['RMSE']
            mape = metrics['test']['MAPE']
            tiempo = metrics['training_time']
            
            print(f"{model_name:<20} {train_r2:<10.4f} {test_r2:<10.4f} {overfit_gap:<12.4f} ${rmse:<14,.0f} {mape:<10.2f} {tiempo:<10.1f}")
        
        # Mejor modelo y recomendaciones
        if self.best_model_name:
            best = self.metrics[self.best_model_name]
            print(f"\n🏆 MODELO RECOMENDADO: {self.best_model_name}")
            print(f"   • R² Test: {best['test']['R2']:.4f}")
            print(f"   • RMSE: ${best['test']['RMSE']:,.0f}")
            print(f"   • MAPE: {best['test']['MAPE']:.2f}%")
            print(f"   • Overfitting: {'BAJO' if best['overfit_gap_r2'] < 0.05 else 'MODERADO' if best['overfit_gap_r2'] < 0.1 else 'ALTO'}")
            print(f"   • Tiempo entrenamiento: {best['training_time']:.1f}s")
            
            # Interpretación del error
            mean_price = best['test']['Mean_Price']
            rmse_relative = best['test']['RMSE_Relative_%']
            print(f"   • Error relativo: {rmse_relative:.1f}% del precio promedio (${mean_price:,.0f})")
    
    def save_models_and_results(self, output_dir='resultados_modelos'):
        """
        Guardar modelos, métricas y resultados
        """
        print(f"\n💾 Guardando resultados en '{output_dir}'...")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Guardar modelos entrenados
        saved_models = []
        for model_name, config in self.models.items():
            if 'trained_model' in config:
                try:
                    filename = f"{model_name}_{timestamp}.joblib"
                    filepath = os.path.join(output_dir, filename)
                    joblib.dump(config['trained_model'], filepath)
                    saved_models.append((model_name, filepath))
                except Exception as e:
                    print(f"❌ Error guardando {model_name}: {e}")
        
        # 2. Guardar métricas
        if self.metrics:
            metrics_data = []
            for model_name, metrics in self.metrics.items():
                row = {
                    'model': model_name,
                    'r2_train': metrics['train']['R2'],
                    'r2_test': metrics['test']['R2'],
                    'overfit_gap': metrics['overfit_gap_r2'],
                    'rmse_test': metrics['test']['RMSE'],
                    'mae_test': metrics['test']['MAE'],
                    'mape_test': metrics['test']['MAPE'],
                    'training_time': metrics['training_time']
                }
                metrics_data.append(row)
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = os.path.join(output_dir, f'metricas_{timestamp}.csv')
            metrics_df.to_csv(metrics_path, index=False)
            print(f"✅ Métricas guardadas: {metrics_path}")
        
        # 3. Guardar mejor modelo por separado
        if self.best_model:
            best_model_path = os.path.join(output_dir, f'mejor_modelo_{timestamp}.joblib')
            joblib.dump(self.best_model, best_model_path)
            print(f"✅ Mejor modelo guardado: {best_model_path}")
        
        # 4. Guardar reporte de características
        if self.feature_names:
            features_df = pd.DataFrame({'feature': self.feature_names})
            features_path = os.path.join(output_dir, f'caracteristicas_{timestamp}.csv')
            features_df.to_csv(features_path, index=False)
        
        print(f"✅ Resultados guardados exitosamente en {output_dir}")
        return saved_models
    
    def train_complete_pipeline(self, X_train, X_test, y_train, y_test, feature_names=None):
        """
        Pipeline completo de entrenamiento y evaluación
        """
        print(f"🚀 INICIANDO PIPELINE COMPLETO - MODO {self.mode.upper()}")
        print("=" * 60)
        
        try:
            # 1. Inicializar modelos
            self.initialize_models()
            
            # 2. Entrenar todos los modelos
            self.train_all_models(X_train, X_test, y_train, y_test, feature_names)
            
            # 3. Generar reporte comprehensivo
            self.generate_comprehensive_report()
            
            # 4. Imprimir resumen
            self.print_comprehensive_summary()
            
            # 5. Guardar resultados
            self.save_models_and_results()
            
            print("\n✅ PIPELINE COMPLETADO EXITOSAMENTE")
            
            return self.metrics
            
        except Exception as e:
            print(f"❌ Error en el pipeline: {e}")
            traceback.print_exc()
            return {}


# FUNCIONES DE CONVENIENCIA

def train_unified_models(X_train, X_test, y_train, y_test, feature_names=None, 
                        mode='balanced', random_state=42):
    """
    Función principal unificada para entrenamiento de modelos
    """
    trainer = UnifiedModelTrainer(random_state=random_state, mode=mode)
    metrics = trainer.train_complete_pipeline(
        X_train, X_test, y_train, y_test, feature_names
    )
    return metrics, trainer

def quick_model_comparison(X_train, X_test, y_train, y_test, feature_names=None):
    """
    Comparación rápida de modelos básicos
    """
    print("⚡ COMPARACIÓN RÁPIDA DE MODELOS BÁSICOS")
    
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            print(f"\nEntrenando {name}...")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Métricas en escala original
            y_true_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred)
            
            rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
            r2 = r2_score(y_true_orig, y_pred_orig)
            mape = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
            
            results[name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAPE': mape,
                'time': time.time() - start_time
            }
            
            print(f"✅ {name} - R²: {r2:.4f}, RMSE: ${rmse:,.0f}, MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"❌ Error en {name}: {e}")
    
    # Mostrar comparación
    print("\n🏆 COMPARACIÓN FINAL:")
    for name, res in sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True):
        print(f"   • {name}: R²={res['R2']:.4f}, RMSE=${res['RMSE']:,.0f}, MAPE={res['MAPE']:.2f}%")
    
    return results

# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 MÓDULO UNIFICADO DE MODELADO")
    print("📝 Usa train_unified_models() para entrenamiento completo")
    
    # Ejemplo con datos dummy
    try:
        from sklearn.datasets import make_regression
        
        print("\n🧪 EJECUTANDO EJEMPLO CON DATOS DUMMY...")
        
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Probar modo rápido
        metrics, trainer = train_unified_models(
            X_train, X_test, y_train, y_test,
            feature_names=feature_names,
            mode='fast'
        )
        
    except Exception as e:
        print(f"❌ Error en ejemplo: {e}")