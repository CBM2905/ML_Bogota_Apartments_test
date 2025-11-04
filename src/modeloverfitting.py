"""
MÓDULO OPTIMIZADO - ENTRENAMIENTO RÁPIDO CON VALIDACIÓN CRUZADA (se esta usando)
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

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class FastModelTrainer:
    """
    Entrenador optimizado para velocidad con validación cruzada
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
        self.final_imputer = SimpleImputer(strategy='median')
        
    def initialize_models_fast(self):
        """
        Inicializar modelos con configuración optimizada para velocidad
        """
        print("🚀 Inicializando modelos en modo rápido...")
        
        if self.fast_mode:
            # CONFIGURACIÓN RÁPIDA - Menos hiperparámetros
            self.models = {
                'LinearRegression': {
                    'model': LinearRegression(),
                    'params': {},
                    'description': 'Regresión lineal rápida',
                    'color': '#1f77b4',
                    'use_cv': False  # No necesita CV
                },
                'Ridge_Fast': {
                    'model': Ridge(random_state=self.random_state),
                    'params': {
                        'alpha': [0.1, 1, 10]  # Solo 3 valores
                    },
                    'description': 'Ridge optimizado',
                    'color': '#ff7f0e',
                    'use_cv': True
                },
                'RandomForest_Fast': {
                    'model': RandomForestRegressor(
                        random_state=self.random_state,
                        n_jobs=-1,
                        n_estimators=100  # Fijo para velocidad
                    ),
                    'params': {
                        'max_depth': [5, 10, None],  # Solo 3 valores
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    },
                    'description': 'Random Forest rápido',
                    'color': '#d62728',
                    'use_cv': True
                },
                'XGBoost_Fast': {
                    'model': XGBRegressor(
                        random_state=self.random_state,
                        n_jobs=-1,
                        verbosity=0,
                        n_estimators=100  # Fijo para velocidad
                    ),
                    'params': {
                        'max_depth': [3, 4, 5],
                        'learning_rate': [0.05, 0.1],
                        'subsample': [0.8, 1.0]
                    },
                    'description': 'XGBoost rápido',
                    'color': '#2ca02c',
                    'use_cv': True
                }
            }
        else:
            # CONFIGURACIÓN COMPLETA (original reducida)
            self.models = {
                'LinearRegression': {
                    'model': LinearRegression(),
                    'params': {},
                    'description': 'Regresión lineal',
                    'color': '#1f77b4',
                    'use_cv': False
                },
                'Ridge': {
                    'model': Ridge(random_state=self.random_state),
                    'params': {
                        'alpha': [0.01, 0.1, 1, 10, 100]
                    },
                    'description': 'Ridge con CV',
                    'color': '#ff7f0e',
                    'use_cv': True
                },
                'Lasso': {
                    'model': Lasso(random_state=self.random_state, max_iter=5000),
                    'params': {
                        'alpha': [0.001, 0.01, 0.1, 1]
                    },
                    'description': 'Lasso con CV',
                    'color': '#9467bd',
                    'use_cv': True
                },
                'RandomForest': {
                    'model': RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
                    'params': {
                        'n_estimators': [100, 150],
                        'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5]
                    },
                    'description': 'Random Forest',
                    'color': '#d62728',
                    'use_cv': True
                },
                'XGBoost': {
                    'model': XGBRegressor(random_state=self.random_state, n_jobs=-1, verbosity=0),
                    'params': {
                        'n_estimators': [100, 150],
                        'max_depth': [3, 4, 5],
                        'learning_rate': [0.05, 0.1]
                    },
                    'description': 'XGBoost',
                    'color': '#2ca02c',
                    'use_cv': True
                }
            }
        
        print(f"✅ {len(self.models)} modelos inicializados en modo {'RÁPIDO' if self.fast_mode else 'COMPLETO'}")
        for name, config in self.models.items():
            print(f"   • {name}: {config['description']}")
    
    def clean_data(self, X, y):
        """
        Limpieza optimizada de datos
        """
        # Conversión rápida a numpy
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        # Eliminar filas con NaN usando operaciones vectorizadas
        valid_mask = ~np.isnan(y)
        if valid_mask.any():
            X = X[valid_mask]
            y = y[valid_mask]
            
        # Eliminar NaN en X
        nan_mask_x = np.isnan(X).any(axis=1)
        if nan_mask_x.any():
            X = X[~nan_mask_x]
            y = y[~nan_mask_x]
            
        # Imputación rápida
        if len(X) > 0:
            X = self.final_imputer.fit_transform(X)
            
        return X, y
    
    def perform_fast_cross_validation(self, model, X, y, model_name):
        """
        Validación cruzada optimizada
        """
        try:
            # Usar menos folds en modo rápido
            cv_folds = 3 if self.fast_mode else min(5, self.n_splits)
            
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # Calcular solo RMSE para mayor velocidad
            cv_scores = cross_val_score(model, X, y, cv=kf, 
                                      scoring='neg_mean_squared_error', n_jobs=-1)
            
            cv_rmse_scores = np.sqrt(-cv_scores)
            
            cv_results = {
                'cv_rmse_mean': cv_rmse_scores.mean(),
                'cv_rmse_std': cv_rmse_scores.std(),
                'cv_rmse_scores': cv_rmse_scores
            }
            
            print(f"   ✅ CV RMSE: {cv_results['cv_rmse_mean']:.4f} ± {cv_results['cv_rmse_std']:.4f}")
            
            return cv_results
            
        except Exception as e:
            print(f"   ❌ Error en CV rápida: {e}")
            return None
    
    def perform_fast_hyperparameter_tuning(self, model_name, config, X_train, y_train):
        """
        Búsqueda rápida de hiperparámetros
        """
        if not config['params']:
            return config['model']
        
        print(f"   ⚡ Sintonización rápida para {model_name}...")
        
        try:
            # Configuración rápida para GridSearch
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
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"   ❌ Error en sintonización rápida: {e}")
            return config['model']
    
    def calculate_fast_metrics(self, y_true, y_pred, model_name):
        """
        Métricas rápidas pero informativas
        """
        try:
            # Convertir a escala original
            y_true_orig = np.expm1(y_true)
            y_pred_orig = np.expm1(y_pred)
            
            # Solo métricas esenciales
            rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
            r2_orig = r2_score(y_true_orig, y_pred_orig)
            mape = np.mean(np.abs((y_true_orig - y_pred_orig) / np.maximum(y_true_orig, 1))) * 100
            
            metrics = {
                'RMSE_original': rmse_orig,
                'R2_original': r2_orig,
                'MAPE_%': mape
            }
            
            print(f"   📊 {model_name} - R²: {r2_orig:.4f}, RMSE: ${rmse_orig:,.0f}")
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error en métricas rápidas: {e}")
            return {'RMSE_original': np.inf, 'R2_original': -np.inf, 'MAPE_%': np.inf}
    
    def train_single_model_fast(self, model_name, config, X_train, X_test, y_train, y_test):
        """
        Entrenamiento rápido de un solo modelo
        """
        try:
            print(f"\n⚡ Entrenando {model_name}...")
            start_time = time.time()
            
            # Limpieza rápida
            X_train_clean, y_train_clean = self.clean_data(X_train, y_train)
            X_test_clean, y_test_clean = self.clean_data(X_test, y_test)
            
            # Verificación rápida de datos
            if len(X_train_clean) < 5:
                print(f"   ❌ Datos insuficientes")
                return False
            
            # Sintonización rápida si es necesario
            if config['use_cv'] and config['params']:
                model = self.perform_fast_hyperparameter_tuning(model_name, config, X_train_clean, y_train_clean)
            else:
                model = config['model']
            
            # Validación cruzada rápida
            cv_results = self.perform_fast_cross_validation(model, X_train_clean, y_train_clean, model_name)
            
            # Entrenamiento final
            model.fit(X_train_clean, y_train_clean)
            
            # Predicciones
            y_pred_train = model.predict(X_train_clean)
            y_pred_test = model.predict(X_test_clean)
            
            # Métricas rápidas
            metrics_train = self.calculate_fast_metrics(y_train_clean, y_pred_train, f"{model_name} (Train)")
            metrics_test = self.calculate_fast_metrics(y_test_clean, y_pred_test, f"{model_name} (Test)")
            
            # Calcular overfitting gap simple
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
                'y_true_test': y_test_clean,
                'y_pred_test': y_pred_test
            }
            
            print(f"   ✅ {model_name} completado en {self.metrics[model_name]['training_time']:.1f}s")
            print(f"   📈 Gap overfitting: {overfit_gap:.4f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error rápido en {model_name}: {str(e)[:100]}...")
            return False
    
    def train_all_models_fast(self, X_train, X_test, y_train, y_test, feature_names=None):
        """
        Entrenamiento paralelo optimizado de todos los modelos
        """
        print(f"\n🚀 INICIANDO ENTRENAMIENTO RÁPIDO")
        print("=" * 50)
        
        self.feature_names = feature_names
        self.metrics = {}
        self.predictions = {}
        
        successful_models = 0
        
        for model_name, config in self.models.items():
            success = self.train_single_model_fast(model_name, config, X_train, X_test, y_train, y_test)
            if success:
                successful_models += 1
        
        print(f"\n📊 Resultado: {successful_models}/{len(self.models)} modelos exitosos")
        
        if successful_models > 0:
            self._select_best_model_fast()
        else:
            raise RuntimeError("❌ Ningún modelo completado")
    
    def _select_best_model_fast(self):
        """
        Selección rápida del mejor modelo
        """
        try:
            # Simple: mejor R² en test
            test_r2_scores = {name: metrics['test']['R2_original'] 
                            for name, metrics in self.metrics.items()}
            
            self.best_model_name = max(test_r2_scores, key=test_r2_scores.get)
            self.best_model = self.models[self.best_model_name]['trained_model']
            
            best_r2 = self.metrics[self.best_model_name]['test']['R2_original']
            best_rmse = self.metrics[self.best_model_name]['test']['RMSE_original']
            
            print(f"\n🏆 MEJOR MODELO: {self.best_model_name}")
            print(f"   • R²: {best_r2:.4f}")
            print(f"   • RMSE: ${best_rmse:,.0f}")
            
        except Exception as e:
            print(f"⚠️  Error seleccionando mejor modelo: {e}")
            # Fallback al primer modelo exitoso
            for name in self.metrics.keys():
                self.best_model_name = name
                self.best_model = self.models[name]['trained_model']
                break
    
    def plot_fast_comparison(self):
        """
        Gráfico rápido de comparación
        """
        if not self.metrics:
            return
        
        # Gráfico simple de comparación de R²
        models = list(self.metrics.keys())
        test_r2 = [self.metrics[model]['test']['R2_original'] for model in models]
        colors = [self.models[model]['color'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, test_r2, color=colors, alpha=0.7)
        
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.ylabel('R² Score (Test)')
        plt.title('Comparación Rápida de Modelos', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Añadir valores
        for bar, r2 in zip(bars, test_r2):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{r2:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_fast_overfitting(self):
        """
        Gráfico rápido de análisis de overfitting
        """
        if not self.metrics:
            return
        
        models = list(self.metrics.keys())
        train_r2 = [self.metrics[model]['train']['R2_original'] for model in models]
        test_r2 = [self.metrics[model]['test']['R2_original'] for model in models]
        overfit_gaps = [self.metrics[model]['overfit_gap'] for model in models]
        
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
        ax2.set_ylabel('Gap de Overfitting')
        ax2.set_title('Análisis de Overfitting', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Colorear barras de overfitting
        for bar, gap in zip(bars, overfit_gaps):
            color = 'green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red'
            bar.set_color(color)
        
        plt.tight_layout()
        plt.show()
    
    def generate_fast_report(self):
        """
        Reporte rápido pero informativo
        """
        print("\n📈 GENERANDO REPORTE RÁPIDO...")
        
        # 1. Comparación simple
        self.plot_fast_comparison()
        
        # 2. Análisis de overfitting
        self.plot_fast_overfitting()
    
    def print_fast_summary(self):
        """
        Resumen rápido en consola
        """
        if not self.metrics:
            return
        
        print("\n" + "=" * 80)
        print("📊 REPORTE RÁPIDO DE MODELOS")
        print("=" * 80)
        
        # Tabla simple
        print(f"\n{'Modelo':<20} {'R² Train':<10} {'R² Test':<10} {'Overfit Gap':<12} {'RMSE Test':<15} {'Tiempo(s)':<10}")
        print("-" * 80)
        
        for model_name, metrics in self.metrics.items():
            train_r2 = metrics['train']['R2_original']
            test_r2 = metrics['test']['R2_original']
            overfit_gap = metrics['overfit_gap']
            rmse = metrics['test']['RMSE_original']
            tiempo = metrics['training_time']
            
            print(f"{model_name:<20} {train_r2:<10.4f} {test_r2:<10.4f} {overfit_gap:<12.4f} ${rmse:<14,.0f} {tiempo:<10.1f}")
        
        # Mejor modelo
        if self.best_model_name:
            best = self.metrics[self.best_model_name]
            print(f"\n🏆 RECOMENDACIÓN: {self.best_model_name}")
            print(f"   • R² Test: {best['test']['R2_original']:.4f}")
            print(f"   • RMSE: ${best['test']['RMSE_original']:,.0f}")
            print(f"   • Overfitting: {'BAJO' if best['overfit_gap'] < 0.05 else 'MODERADO' if best['overfit_gap'] < 0.1 else 'ALTO'}")
    
    def train_fast_pipeline(self, X_train, X_test, y_train, y_test, feature_names=None, save_models=False):
        """
        Pipeline completo optimizado
        """
        print("🚀 INICIANDO PIPELINE RÁPIDO")
        print("=" * 50)
        
        try:
            # 1. Inicializar modelos rápidos
            self.initialize_models_fast()
            
            # 2. Entrenamiento rápido
            self.train_all_models_fast(X_train, X_test, y_train, y_test, feature_names)
            
            # 3. Reporte rápido
            self.generate_fast_report()
            
            # 4. Resumen
            self.print_fast_summary()
            
            print("\n✅ PIPELINE RÁPIDO COMPLETADO")
            
            return self.metrics
            
        except Exception as e:
            print(f"❌ Error en pipeline rápido: {e}")
            return {}


# FUNCIONES DE CONVENIENCIA OPTIMIZADAS

def train_fast_models(X_train, X_test, y_train, y_test, feature_names=None, 
                     fast_mode=True, random_state=42):
    """
    Función principal optimizada para entrenamiento rápido
    """
    trainer = FastModelTrainer(random_state=random_state, fast_mode=fast_mode)
    metrics = trainer.train_fast_pipeline(
        X_train, X_test, y_train, y_test, 
        feature_names=feature_names
    )
    return metrics, trainer

def train_quick_baseline(X_train, X_test, y_train, y_test):
    """
    Entrenamiento ultra-rápido con solo 2 modelos
    """
    print("⚡ ENTRENAMIENTO ULTRA-RÁPIDO (2 modelos)")
    
    # Solo los modelos más rápidos
    models = {
        'Ridge_Fast': Ridge(alpha=1.0),
        'RandomForest_Fast': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }
    
    metrics = {}
    
    for name, model in models.items():
        try:
            print(f"\nEntrenando {name}...")
            start_time = time.time()
            
            # Entrenamiento directo sin CV
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            
            # Métricas rápidas
            y_true_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred)
            
            rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
            r2 = r2_score(y_true_orig, y_pred_orig)
            
            metrics[name] = {
                'R2': r2,
                'RMSE': rmse,
                'time': time.time() - start_time
            }
            
            print(f"✅ {name} - R²: {r2:.4f}, RMSE: ${rmse:,.0f}")
            
        except Exception as e:
            print(f"❌ Error en {name}: {e}")
    
    return metrics

# Diagnóstico rápido
def quick_diagnostic(metrics):
    """
    Diagnóstico ultra-rápido
    """
    if not metrics:
        return
    
    print("\n🔍 DIAGNÓSTICO RÁPIDO")
    print("=" * 40)
    
    best_model = max(metrics.items(), key=lambda x: x[1]['R2'])
    print(f"🏆 Mejor modelo: {best_model[0]} (R²: {best_model[1]['R2']:.4f})")
    
    for name, results in metrics.items():
        print(f"   • {name}: R²={results['R2']:.4f}, RMSE=${results['RMSE']:,.0f}")