"""
MÓDULO ULTIMATE PARA DATOS PREPROCESADOS
Solo entrenamiento y evaluación - Sin preprocesamiento
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
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

class UltimateModelTrainerPreprocessed:
    """
    Entrenador ultimate para datos YA PREPROCESADOS
    Solo entrenamiento, validación cruzada y evaluación
    """
    
    def __init__(self, random_state=42, n_splits=5, fast_mode=True):
        self.random_state = random_state
        self.n_splits = n_splits
        self.fast_mode = fast_mode
        self.models = {}
        self.metrics = {}
        self.cv_results = {}
        self.predictions = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def initialize_models_optimized(self):
        """
        Inicializar modelos con configuración balanceada
        """
        print("🚀 Inicializando modelos optimizados...")
        
        if self.fast_mode:
            # CONFIGURACIÓN RÁPIDA PERO COMPLETA
            self.models = {
                'LinearRegression': {
                    'model': LinearRegression(),
                    'params': {},
                    'description': 'Regresión lineal rápida',
                    'color': '#1f77b4',
                    'use_cv': False
                },
                'Ridge_Optimized': {
                    'model': Ridge(random_state=self.random_state),
                    'params': {
                        'alpha': [0.1, 1, 10, 100]
                    },
                    'description': 'Ridge con regularización L2',
                    'color': '#ff7f0e',
                    'use_cv': True
                },
                'RandomForest_Optimized': {
                    'model': RandomForestRegressor(
                        random_state=self.random_state,
                        n_jobs=-1
                    ),
                    'params': {
                        'n_estimators': [100, 150],
                        'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5]
                    },
                    'description': 'Random Forest optimizado',
                    'color': '#d62728',
                    'use_cv': True
                },
                'XGBoost_Optimized': {
                    'model': XGBRegressor(
                        random_state=self.random_state,
                        n_jobs=-1,
                        verbosity=0
                    ),
                    'params': {
                        'n_estimators': [100, 150],
                        'max_depth': [3, 4, 5],
                        'learning_rate': [0.05, 0.1],
                        'subsample': [0.8, 1.0]
                    },
                    'description': 'XGBoost con regularización',
                    'color': '#2ca02c',
                    'use_cv': True
                }
            }
        else:
            # CONFIGURACIÓN COMPLETA
            self.models = {
                'LinearRegression': {
                    'model': LinearRegression(),
                    'params': {},
                    'description': 'Regresión lineal',
                    'color': '#1f77b4',
                    'use_cv': False
                },
                'Ridge_CV': {
                    'model': Ridge(random_state=self.random_state),
                    'params': {
                        'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
                    },
                    'description': 'Ridge con validación cruzada',
                    'color': '#ff7f0e',
                    'use_cv': True
                },
                'Lasso_CV': {
                    'model': Lasso(random_state=self.random_state, max_iter=5000),
                    'params': {
                        'alpha': [0.001, 0.01, 0.1, 1, 10]
                    },
                    'description': 'Lasso con validación cruzada',
                    'color': '#9467bd',
                    'use_cv': True
                },
                'RandomForest_Advanced': {
                    'model': RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [5, 10, 15, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'description': 'Random Forest avanzado',
                    'color': '#d62728',
                    'use_cv': True
                },
                'XGBoost_Advanced': {
                    'model': XGBRegressor(random_state=self.random_state, n_jobs=-1, verbosity=0),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 4, 5, 6],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'subsample': [0.8, 0.9, 1.0]
                    },
                    'description': 'XGBoost avanzado',
                    'color': '#2ca02c',
                    'use_cv': True
                }
            }
        
        print(f"✅ {len(self.models)} modelos inicializados en modo {'RÁPIDO' if self.fast_mode else 'COMPLETO'}")

    def perform_robust_cross_validation(self, model, X, y, model_name):
        """
        Validación cruzada robusta con múltiples métricas
        """
        print(f"   🔄 Validación cruzada ({self.n_splits} folds)...")
        
        try:
            # Usar menos folds en modo rápido
            cv_folds = 3 if self.fast_mode else min(5, self.n_splits)
            
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # Calcular múltiples métricas en CV
            cv_rmse_scores = cross_val_score(model, X, y, cv=kf, 
                                           scoring='neg_mean_squared_error', n_jobs=-1)
            cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
            
            # Convertir a métricas positivas
            cv_rmse_scores = np.sqrt(-cv_rmse_scores)
            
            cv_results = {
                'cv_rmse_mean': cv_rmse_scores.mean(),
                'cv_rmse_std': cv_rmse_scores.std(),
                'cv_r2_mean': cv_r2_scores.mean(),
                'cv_r2_std': cv_r2_scores.std(),
                'cv_rmse_scores': cv_rmse_scores,
                'cv_r2_scores': cv_r2_scores
            }
            
            print(f"   ✅ CV RMSE: {cv_results['cv_rmse_mean']:.4f} ± {cv_results['cv_rmse_std']:.4f}")
            print(f"   ✅ CV R²: {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
            
            return cv_results
            
        except Exception as e:
            print(f"   ❌ Error en validación cruzada: {e}")
            return None

    def perform_fast_hyperparameter_tuning(self, model_name, config, X_train, y_train):
        """
        Búsqueda rápida pero efectiva de hiperparámetros
        """
        if not config['params']:
            return config['model']
        
        print(f"   🎛️  Sintonizando {model_name}...")
        
        try:
            # Configuración optimizada para velocidad
            cv_folds = 2 if self.fast_mode else 3
            
            grid_search = GridSearchCV(
                config['model'],
                param_grid=config['params'],
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"   ✅ Mejores parámetros encontrados")
            print(f"   📊 Mejor score CV: {-grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"   ❌ Error en sintonización: {e}")
            return config['model']

    def calculate_comprehensive_metrics(self, y_true, y_pred, model_name, dataset_type):
        """
        Calcular métricas comprehensivas en ambas escalas
        """
        try:
            # Métricas en escala logarítmica
            rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
            mae_log = mean_absolute_error(y_true, y_pred)
            r2_log = r2_score(y_true, y_pred)
            
            # Convertir a escala original (asumiendo que y está en log)
            y_true_orig = np.expm1(y_true)
            y_pred_orig = np.expm1(y_pred)
            
            # Métricas en escala original
            rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
            mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
            r2_orig = r2_score(y_true_orig, y_pred_orig)
            
            # MAPE y métricas adicionales
            mape = np.mean(np.abs((y_true_orig - y_pred_orig) / np.maximum(y_true_orig, 1))) * 100
            residuals = y_true - y_pred
            residual_std = np.std(residuals)
            max_error = np.max(np.abs(residuals))
            
            metrics = {
                'RMSE_log': rmse_log,
                'MAE_log': mae_log,
                'R2_log': r2_log,
                'RMSE_original': rmse_orig,
                'MAE_original': mae_orig,
                'R2_original': r2_orig,
                'MAPE_%': mape,
                'Residual_Std': residual_std,
                'Max_Error': max_error
            }
            
            print(f"   📈 {model_name} ({dataset_type}) - R²: {r2_orig:.4f}, RMSE: ${rmse_orig:,.0f}, MAPE: {mape:.2f}%")
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error calculando métricas: {e}")
            return {'RMSE_original': np.inf, 'R2_original': -np.inf, 'MAPE_%': np.inf}

    def train_single_model_ultimate(self, model_name, config, X_train, X_test, y_train, y_test):
        """
        Entrenamiento completo de un modelo con CV y análisis avanzado
        """
        try:
            print(f"\n🎯 Entrenando {model_name}...")
            start_time = time.time()
            
            # Sintonización de hiperparámetros
            if config['use_cv'] and config['params']:
                model = self.perform_fast_hyperparameter_tuning(model_name, config, X_train, y_train)
            else:
                model = config['model']
            
            # Validación cruzada robusta
            cv_results = self.perform_robust_cross_validation(model, X_train, y_train, model_name)
            
            # Entrenar modelo final
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calcular métricas comprehensivas
            metrics_train = self.calculate_comprehensive_metrics(y_train, y_pred_train, model_name, "Train")
            metrics_test = self.calculate_comprehensive_metrics(y_test, y_pred_test, model_name, "Test")
            
            # Calcular overfitting gap
            overfit_gap = metrics_train['R2_original'] - metrics_test['R2_original']
            
            # Guardar resultados
            self.models[model_name]['trained_model'] = model
            self.metrics[model_name] = {
                'train': metrics_train,
                'test': metrics_test,
                'cv': cv_results,
                'overfit_gap': overfit_gap,
                'training_time': time.time() - start_time
            }
            
            self.predictions[model_name] = {
                'y_true_train': y_train,
                'y_pred_train': y_pred_train,
                'y_true_test': y_test,
                'y_pred_test': y_pred_test,
                'residuals_train': y_train - y_pred_train,
                'residuals_test': y_test - y_pred_test
            }
            
            # Calcular importancia de características
            self._calculate_feature_importance(model, model_name)
            
            print(f"   ✅ {model_name} completado en {self.metrics[model_name]['training_time']:.1f}s")
            print(f"   📊 Gap de overfitting: {overfit_gap:.4f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error en {model_name}: {str(e)[:100]}...")
            return False

    def _calculate_feature_importance(self, model, model_name):
        """
        Calcular importancia de características para modelos que lo soportan
        """
        try:
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                self.feature_importance[model_name] = np.abs(model.coef_)
            else:
                self.feature_importance[model_name] = None
        except:
            self.feature_importance[model_name] = None

    def train_all_models_preprocessed(self, X_train, X_test, y_train, y_test, feature_names=None):
        """
        Entrenar todos los modelos con datos YA PREPROCESADOS
        """
        print("\n🚀 INICIANDO ENTRENAMIENTO ULTIMATE CON DATOS PREPROCESADOS")
        print("=" * 70)
        
        self.feature_names = feature_names
        self.metrics = {}
        self.predictions = {}
        self.feature_importance = {}
        
        # Inicializar modelos
        self.initialize_models_optimized()
        
        successful_models = 0
        
        for model_name, config in self.models.items():
            success = self.train_single_model_ultimate(model_name, config, X_train, X_test, y_train, y_test)
            if success:
                successful_models += 1
        
        print(f"\n📊 Resumen: {successful_models}/{len(self.models)} modelos exitosos")
        
        if successful_models == 0:
            raise RuntimeError("❌ Ningún modelo completado")
        
        # Seleccionar mejor modelo
        self._select_best_model_ultimate()

    def _select_best_model_ultimate(self):
        """
        Selección avanzada del mejor modelo considerando múltiples factores
        """
        try:
            model_scores = {}
            
            for model_name, metrics in self.metrics.items():
                # Factores para la selección
                test_r2 = metrics['test']['R2_original']
                overfit_gap = metrics['overfit_gap']
                cv_stability = 0
                
                if metrics['cv'] is not None:
                    # Penalizar alta variabilidad en CV
                    cv_stability = 1 - (metrics['cv']['cv_r2_std'] / max(0.1, abs(metrics['cv']['cv_r2_mean'])))
                
                # Score compuesto (60% test performance, 20% CV stability, 20% low overfitting)
                composite_score = (0.6 * test_r2 + 
                                 0.2 * cv_stability + 
                                 0.2 * (1 - min(overfit_gap, 0.3)/0.3))
                
                model_scores[model_name] = composite_score
            
            self.best_model_name = max(model_scores, key=model_scores.get)
            self.best_model = self.models[self.best_model_name]['trained_model']
            
            best_metrics = self.metrics[self.best_model_name]
            print(f"\n🏆 MEJOR MODELO: {self.best_model_name}")
            print(f"   • R² (test): {best_metrics['test']['R2_original']:.4f}")
            print(f"   • RMSE: ${best_metrics['test']['RMSE_original']:,.0f}")
            print(f"   • Score compuesto: {model_scores[self.best_model_name]:.4f}")
            print(f"   • Overfitting: {'BAJO' if best_metrics['overfit_gap'] < 0.05 else 'MODERADO' if best_metrics['overfit_gap'] < 0.1 else 'ALTO'}")
            
        except Exception as e:
            print(f"⚠️  Error seleccionando mejor modelo: {e}")
            # Fallback simple
            test_r2_scores = {name: metrics['test']['R2_original'] for name, metrics in self.metrics.items()}
            self.best_model_name = max(test_r2_scores, key=test_r2_scores.get)
            self.best_model = self.models[self.best_model_name]['trained_model']

    # =========================================================================
    # GRÁFICOS Y VISUALIZACIONES AVANZADAS
    # =========================================================================

    def plot_comprehensive_comparison(self):
        """
        Gráfico completo de comparación entre modelos
        """
        if not self.metrics:
            print("⚠️  No hay métricas para comparar")
            return
        
        print("\n📊 Generando gráficos comprehensivos...")
        
        # Crear figura con múltiples subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3)
        
        # 1. Comparación de métricas principales
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_metrics_comparison(ax1)
        
        # 2. Predicciones vs reales
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_predictions_vs_actual([ax2, ax3, ax4])
        
        plt.tight_layout()
        plt.show()

    def _plot_metrics_comparison(self, ax):
        """
        Gráfico de comparación de métricas entre modelos
        """
        models = list(self.metrics.keys())
        
        # Preparar datos
        test_r2 = [self.metrics[model]['test']['R2_original'] for model in models]
        test_rmse = [self.metrics[model]['test']['RMSE_original'] for model in models]
        overfit_gaps = [self.metrics[model]['overfit_gap'] for model in models]
        colors = [self.models[model]['color'] for model in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        # Gráfico de barras múltiples
        bars1 = ax.bar(x - width, test_r2, width, label='R² Test', alpha=0.8)
        bars2 = ax.bar(x, overfit_gaps, width, label='Overfit Gap', alpha=0.8)
        
        # Escalar RMSE para mejor visualización
        rmse_scaled = [rmse / max(test_rmse) for rmse in test_rmse]
        bars3 = ax.bar(x + width, rmse_scaled, width, label='RMSE (escalado)', alpha=0.8)
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel('Valores')
        ax.set_title('Comparación Completa de Modelos', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Añadir valores
        for i, (r2, gap, rmse) in enumerate(zip(test_r2, overfit_gaps, test_rmse)):
            ax.text(i - width, r2 + 0.01, f'{r2:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i, gap + 0.01, f'{gap:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, rmse_scaled[i] + 0.01, f'${rmse:,.0f}', ha='center', va='bottom', fontsize=8)

    def _plot_predictions_vs_actual(self, axes):
        """
        Gráfico de predicciones vs valores reales para modelos top
        """
        # Seleccionar 3 mejores modelos por R²
        model_scores = {name: metrics['test']['R2_original'] for name, metrics in self.metrics.items()}
        top_models = sorted(model_scores, key=model_scores.get, reverse=True)[:3]
        
        for idx, model_name in enumerate(top_models):
            if idx < len(axes):
                pred_data = self.predictions[model_name]
                y_true = np.expm1(pred_data['y_true_test'])
                y_pred = np.expm1(pred_data['y_pred_test'])
                
                ax = axes[idx]
                ax.scatter(y_true, y_pred, alpha=0.6, 
                          color=self.models[model_name]['color'], s=30)
                
                # Línea de identidad
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                
                ax.set_xlabel('Valores Reales ($)')
                ax.set_ylabel('Predicciones ($)')
                ax.set_title(f'{model_name}\nPredicciones vs Reales', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Añadir métricas
                r2 = self.metrics[model_name]['test']['R2_original']
                rmse = self.metrics[model_name]['test']['RMSE_original']
                ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = ${rmse:,.0f}', 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    def plot_cv_performance(self):
        """
        Gráfico de rendimiento en validación cruzada
        """
        models_with_cv = [name for name, metrics in self.metrics.items() 
                         if metrics['cv'] is not None]
        
        if not models_with_cv:
            print("⚠️  No hay resultados de CV para graficar")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² de CV
        cv_r2_means = [self.metrics[name]['cv']['cv_r2_mean'] for name in models_with_cv]
        cv_r2_stds = [self.metrics[name]['cv']['cv_r2_std'] for name in models_with_cv]
        colors = [self.models[name]['color'] for name in models_with_cv]
        
        bars1 = ax1.bar(models_with_cv, cv_r2_means, yerr=cv_r2_stds, 
                       capsize=5, color=colors, alpha=0.7)
        ax1.set_title('R² - Validación Cruzada\n(Promedio ± Desviación Estándar)', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, mean, std in zip(bars1, cv_r2_means, cv_r2_stds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        # RMSE de CV
        cv_rmse_means = [self.metrics[name]['cv']['cv_rmse_mean'] for name in models_with_cv]
        cv_rmse_stds = [self.metrics[name]['cv']['cv_rmse_std'] for name in models_with_cv]
        
        bars2 = ax2.bar(models_with_cv, cv_rmse_means, yerr=cv_rmse_stds,
                       capsize=5, color=colors, alpha=0.7)
        ax2.set_title('RMSE - Validación Cruzada\n(Promedio ± Desviación Estándar)', 
                     fontsize=14, fontweight='bold')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

    def plot_overfitting_analysis(self):
        """
        Análisis detallado de overfitting
        """
        if not self.metrics:
            return
        
        models = list(self.metrics.keys())
        train_r2 = [self.metrics[model]['train']['R2_original'] for model in models]
        test_r2 = [self.metrics[model]['test']['R2_original'] for model in models]
        overfit_gaps = [self.metrics[model]['overfit_gap'] for model in models]
        colors = [self.models[model]['color'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² Train vs Test
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, train_r2, width, label='Train', alpha=0.7)
        bars2 = ax1.bar(x + width/2, test_r2, width, label='Test', alpha=0.7)
        
        ax1.set_xlabel('Modelos')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Comparación R²: Train vs Test', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gap de overfitting con colores según severidad
        bars3 = ax2.bar(models, overfit_gaps, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Modelos')
        ax2.set_ylabel('Gap de Overfitting (Train R² - Test R²)')
        ax2.set_title('Análisis de Overfitting', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Colorear según severidad de overfitting
        for bar, gap in zip(bars3, overfit_gaps):
            if gap < 0.02:
                bar.set_color('green')
            elif gap < 0.05:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.show()

    def generate_ultimate_report(self):
        """
        Generar reporte completo con todos los análisis
        """
        print("\n📈 GENERANDO REPORTE ULTIMATE...")
        
        # 1. Comparación comprehensiva
        self.plot_comprehensive_comparison()
        
        # 2. Rendimiento en CV
        self.plot_cv_performance()
        
        # 3. Análisis de overfitting
        self.plot_overfitting_analysis()

    def print_detailed_report(self):
        """
        Imprimir reporte detallado en consola
        """
        if not self.metrics:
            print("⚠️  No hay métricas para reportar")
            return
        
        print("\n" + "=" * 100)
        print("📊 REPORTE ULTIMATE DETALLADO")
        print("=" * 100)
        
        # Crear tabla resumen
        summary_data = []
        for model_name, model_metrics in self.metrics.items():
            cv_info = ""
            if model_metrics['cv']:
                cv_info = f"{model_metrics['cv']['cv_r2_mean']:.4f}±{model_metrics['cv']['cv_r2_std']:.4f}"
            
            summary_data.append({
                'Modelo': model_name,
                'R² Train': f"{model_metrics['train']['R2_original']:.4f}",
                'R² Test': f"{model_metrics['test']['R2_original']:.4f}",
                'R² CV': cv_info,
                'Overfit Gap': f"{model_metrics['overfit_gap']:.4f}",
                'RMSE Test': f"${model_metrics['test']['RMSE_original']:,.0f}",
                'MAPE': f"{model_metrics['test']['MAPE_%']:.2f}%",
                'Tiempo (s)': f"{model_metrics['training_time']:.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nRESUMEN COMPARATIVO:")
        print(summary_df.to_string(index=False))
        
        # Análisis del mejor modelo
        if self.best_model_name:
            print(f"\n🏆 MODELO RECOMENDADO: {self.best_model_name}")
            best_metrics = self.metrics[self.best_model_name]
            print(f"   • R² (test): {best_metrics['test']['R2_original']:.4f}")
            print(f"   • RMSE: ${best_metrics['test']['RMSE_original']:,.0f}")
            print(f"   • MAPE: {best_metrics['test']['MAPE_%']:.2f}%")
            print(f"   • Gap overfitting: {best_metrics['overfit_gap']:.4f}")
            
            if best_metrics['cv']:
                print(f"   • R² CV: {best_metrics['cv']['cv_r2_mean']:.4f} ± {best_metrics['cv']['cv_r2_std']:.4f}")
            
            # Interpretación del overfitting
            overfit_gap = best_metrics['overfit_gap']
            if overfit_gap < 0.02:
                print("   ✅ EXCELENTE: Overfitting mínimo")
            elif overfit_gap < 0.05:
                print("   👍 BUENO: Overfitting bajo")
            elif overfit_gap < 0.1:
                print("   📊 MODERADO: Overfitting aceptable")
            else:
                print("   🔄 ALTO: Considerar más regularización")

    def save_models_and_report(self, model_dir='modelos_ultimate_preprocessed'):
        """
        Guardar modelos entrenados y reporte completo
        """
        print(f"\n💾 Guardando modelos en '{model_dir}'...")
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_models = []
        
        # Guardar cada modelo
        for model_name, config in self.models.items():
            if 'trained_model' in config:
                try:
                    filename = f"{model_name}_{timestamp}.joblib"
                    filepath = os.path.join(model_dir, filename)
                    joblib.dump(config['trained_model'], filepath)
                    saved_models.append((model_name, filepath))
                    print(f"✅ {model_name} guardado")
                except Exception as e:
                    print(f"❌ Error guardando {model_name}: {e}")
        
        # Guardar métricas detalladas
        if self.metrics:
            metrics_data = []
            for model_name, model_metrics in self.metrics.items():
                row = {'Modelo': model_name}
                
                # Métricas de train
                for key, value in model_metrics['train'].items():
                    row[f'Train_{key}'] = value
                
                # Métricas de test
                for key, value in model_metrics['test'].items():
                    row[f'Test_{key}'] = value
                
                # Métricas de CV
                if model_metrics['cv']:
                    for key, value in model_metrics['cv'].items():
                        if key not in ['cv_rmse_scores', 'cv_r2_scores']:
                            row[f'CV_{key}'] = value
                
                row['Overfit_Gap'] = model_metrics['overfit_gap']
                row['Training_Time_s'] = model_metrics['training_time']
                
                metrics_data.append(row)
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = os.path.join(model_dir, f'metricas_ultimate_{timestamp}.csv')
            metrics_df.to_csv(metrics_path, index=False)
            print(f"✅ Métricas guardadas")
        
        return saved_models

    def train_and_evaluate_preprocessed(self, X_train, X_test, y_train, y_test, 
                                      feature_names=None, save_models=True):
        """
        Función principal para datos YA PREPROCESADOS
        """
        print("🚀 INICIANDO PIPELINE ULTIMATE CON DATOS PREPROCESADOS")
        print("=" * 70)

        try:
            # 1. Entrenar todos los modelos
            self.train_all_models_preprocessed(X_train, X_test, y_train, y_test, feature_names)

            # 2. Generar reporte completo
            self.generate_ultimate_report()

            # 3. Guardar modelos
            if save_models:
                self.save_models_and_report()

            # 4. Imprimir reporte detallado
            self.print_detailed_report()

            print("\n🎉 PIPELINE ULTIMATE COMPLETADO EXITOSAMENTE")

            return self.metrics

        except Exception as e:
            print(f"❌ Error en el pipeline ultimate: {e}")
            traceback.print_exc()
            return {}


# FUNCIONES DE CONVENIENCIA PARA DATOS PREPROCESADOS

def train_ultimate_preprocessed(X_train, X_test, y_train, y_test, feature_names=None, 
                               fast_mode=True, random_state=42, save_models=True):
    """
    Función principal simplificada para datos YA PREPROCESADOS
    """
    trainer = UltimateModelTrainerPreprocessed(random_state=random_state, fast_mode=fast_mode)
    metrics = trainer.train_and_evaluate_preprocessed(
        X_train, X_test, y_train, y_test, 
        feature_names=feature_names, 
        save_models=save_models
    )
    return metrics, trainer

def quick_ultimate_preprocessed(X_train, X_test, y_train, y_test, feature_names=None):
    """
    Entrenamiento ultra-rápido para datos preprocesados
    """
    print("⚡ ENTRENAMIENTO ULTRA-RÁPIDO ULTIMATE (PREPROCESADO)")
    
    trainer = UltimateModelTrainerPreprocessed(fast_mode=True, n_splits=3)
    metrics = trainer.train_and_evaluate_preprocessed(
        X_train, X_test, y_train, y_test, 
        feature_names=feature_names, 
        save_models=False
    )
    return metrics, trainer