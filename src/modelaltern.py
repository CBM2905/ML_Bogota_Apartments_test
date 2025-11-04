"""
MÓDULO COMPLETO DE MODELOS - VERSIÓN CORREGIDA SIN DATA LEAKAGE
Optimizado para trabajar con el preprocesamiento post-split
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import warnings
import traceback
from datetime import datetime
import os
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class ComprehensiveModelTrainer:
    """
    Clase ACTUALIZADA para entrenar modelos con datos preprocesados sin leakage
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
        self.predictions = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """
        Inicializar modelos con hiperparámetros optimizados
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
            'Lasso': {
                'model': Lasso(random_state=self.random_state),
                'params': {'alpha': [0.001, 0.01, 0.1, 1, 10]},
                'description': 'Regresión Lasso con selección de características',
                'color': '#9467bd'
            },
            'RandomForest': {
                'model': RandomForestRegressor(
                    random_state=self.random_state,
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                },
                'description': 'Random Forest robusto',
                'color': '#d62728'
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
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'description': 'XGBoost optimizado',
                'color': '#2ca02c'
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(
                    random_state=self.random_state,
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.05, 0.1, 0.2]
                },
                'description': 'Gradient Boosting tradicional',
                'color': '#8c564b'
            }
        }
        
        print("✅ Modelos inicializados:")
        for name, config in self.models.items():
            print(f"   • {name}: {config['description']}")
    
    def calculate_metrics(self, y_true, y_pred, model_name, return_dict=True):
        """
        Calcular métricas de evaluación (en escala logarítmica y conversión a original)
        """
        try:
            # Métricas en escala logarítmica
            rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
            mae_log = mean_absolute_error(y_true, y_pred)
            r2_log = r2_score(y_true, y_pred)
            
            # Convertir a escala original (exponenciar)
            y_true_orig = np.expm1(y_true)
            y_pred_orig = np.expm1(y_pred)
            
            # Métricas en escala original
            rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
            mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
            r2_orig = r2_score(y_true_orig, y_pred_orig)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
            
            # Residual analysis
            residuals = y_true - y_pred
            residual_std = np.std(residuals)
            
            metrics = {
                'RMSE_log': rmse_log,
                'MAE_log': mae_log,
                'R2_log': r2_log,
                'RMSE_original': rmse_orig,
                'MAE_original': mae_orig,
                'R2_original': r2_orig,
                'MAPE_%': mape,
                'Residual_Std': residual_std,
                'Max_Error': np.max(np.abs(residuals))
            }
            
            print(f"📈 {model_name}:")
            print(f"   • R² (log): {r2_log:.4f}")
            print(f"   • R² (original): {r2_orig:.4f}")
            print(f"   • RMSE (original): ${rmse_orig:,.0f}")
            print(f"   • MAE (original): ${mae_orig:,.0f}")
            print(f"   • MAPE: {mape:.2f}%")
            
            return metrics if return_dict else (rmse_log, mae_log, r2_log)
            
        except Exception as e:
            print(f"❌ Error calculando métricas para {model_name}: {e}")
            return {'RMSE_log': np.inf, 'MAE_log': np.inf, 'R2_log': -np.inf} if return_dict else (np.inf, np.inf, -np.inf)
    
    def train_single_model(self, model_name, config, X_train, X_test, y_train, y_test):
        """
        Entrenar un solo modelo con manejo robusto de errores
        """
        try:
            print(f"\n🎯 Entrenando {model_name}...")
            
            model = config['model']
            
            # Entrenar modelo base
            model.fit(X_train, y_train)
            
            # Realizar predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calcular métricas para train y test
            metrics_train = self.calculate_metrics(y_train, y_pred_train, f"{model_name} (Train)", return_dict=True)
            metrics_test = self.calculate_metrics(y_test, y_pred_test, f"{model_name} (Test)", return_dict=True)
            
            # Guardar resultados
            self.models[model_name]['trained_model'] = model
            self.metrics[model_name] = {
                'train': metrics_train,
                'test': metrics_test,
                'overfit_gap': metrics_train['R2_log'] - metrics_test['R2_log']  # Medida de overfitting
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
            
            print(f"✅ {model_name} entrenado exitosamente")
            print(f"   • Overfit gap: {self.metrics[model_name]['overfit_gap']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error entrenando {model_name}: {e}")
            traceback.print_exc()
            return False
    
    def _calculate_feature_importance(self, model, model_name):
        """
        Calcular importancia de características según el tipo de modelo
        """
        try:
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                self.feature_importance[model_name] = np.abs(model.coef_)
            else:
                self.feature_importance[model_name] = None
        except Exception as e:
            print(f"⚠️ No se pudo calcular importancia para {model_name}: {e}")
            self.feature_importance[model_name] = None
    
    def perform_hyperparameter_tuning(self, model_name, config, X_train, y_train):
        """
        Realizar búsqueda de hiperparámetros con GridSearchCV
        """
        try:
            if not config['params']:
                print(f"⏭️  Sin hiperparámetros para {model_name}, usando modelo por defecto")
                return config['model']
            
            print(f"🎛️  Realizando GridSearch para {model_name}...")
            
            grid_search = GridSearchCV(
                config['model'],
                param_grid=config['params'],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"✅ Mejores parámetros para {model_name}: {grid_search.best_params_}")
            print(f"✅ Mejor score: {-grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"❌ Error en GridSearch para {model_name}: {e}")
            print("🔄 Usando modelo por defecto...")
            return config['model']
    
    def train_models(self, X_train, X_test, y_train, y_test, feature_names=None, tune_hyperparams=False):
        """
        Entrenar todos los modelos
        """
        print("\n🚀 INICIANDO ENTRENAMIENTO DE MODELOS")
        print("=" * 60)
        
        self.feature_names = feature_names
        self.metrics = {}
        self.predictions = {}
        self.feature_importance = {}
        
        successful_models = 0
        
        for model_name, config in self.models.items():
            # Realizar tuning de hiperparámetros si se solicita
            if tune_hyperparams:
                tuned_model = self.perform_hyperparameter_tuning(model_name, config, X_train, y_train)
                config['model'] = tuned_model
            
            # Entrenar modelo
            success = self.train_single_model(model_name, config, X_train, X_test, y_train, y_test)
            if success:
                successful_models += 1
        
        print(f"\n📊 Resumen: {successful_models}/{len(self.models)} modelos entrenados exitosamente")
        
        if successful_models == 0:
            raise RuntimeError("❌ Ningún modelo pudo ser entrenado")
        
        # Identificar mejor modelo
        self._select_best_model()
    
    def _select_best_model(self):
        """
        Seleccionar el mejor modelo basado en R² de test
        """
        try:
            test_r2_scores = {}
            for model_name, metrics in self.metrics.items():
                test_r2_scores[model_name] = metrics['test']['R2_original']
            
            self.best_model_name = max(test_r2_scores, key=test_r2_scores.get)
            self.best_model = self.models[self.best_model_name]['trained_model']
            
            print(f"\n🏆 MEJOR MODELO: {self.best_model_name}")
            print(f"   • R² (test): {test_r2_scores[self.best_model_name]:.4f}")
            
        except Exception as e:
            print(f"⚠️  Error seleccionando mejor modelo: {e}")
            self.best_model_name = None
            self.best_model = None
    
    def plot_model_comparison(self):
        """
        Gráfico de comparación entre modelos
        """
        if not self.metrics:
            print("⚠️  No hay métricas para comparar")
            return
        
        print("\n📊 Generando gráficos de comparación...")
        
        # Preparar datos para comparación
        models = list(self.metrics.keys())
        test_r2 = [self.metrics[model]['test']['R2_original'] for model in models]
        test_rmse = [self.metrics[model]['test']['RMSE_original'] for model in models]
        overfit_gaps = [self.metrics[model]['overfit_gap'] for model in models]
        colors = [self.models[model]['color'] for model in models]
        
        # Crear figura
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Comparación de R²
        bars1 = axes[0, 0].bar(models, test_r2, color=colors, alpha=0.7)
        axes[0, 0].set_title('Comparación de R² (Test - Escala Original)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        # Añadir valores en las barras
        for bar, value in zip(bars1, test_r2):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Comparación de RMSE
        bars2 = axes[0, 1].bar(models, test_rmse, color=colors, alpha=0.7)
        axes[0, 1].set_title('Comparación de RMSE (Test - Escala Original)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('RMSE ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        # Añadir valores en las barras
        for bar, value in zip(bars2, test_rmse):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                           f'${value:,.0f}', ha='center', va='bottom')
        
        # 3. Gap de overfitting
        bars3 = axes[1, 0].bar(models, overfit_gaps, color=colors, alpha=0.7)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Gap de Overfitting (R² Train - R² Test)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Overfit Gap')
        axes[1, 0].tick_params(axis='x', rotation=45)
        # Añadir valores en las barras
        for bar, value in zip(bars3, overfit_gaps):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 4. MAPE
        test_mape = [self.metrics[model]['test']['MAPE_%'] for model in models]
        bars4 = axes[1, 1].bar(models, test_mape, color=colors, alpha=0.7)
        axes[1, 1].set_title('Error Porcentual Absoluto Medio (MAPE)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        # Añadir valores en las barras
        for bar, value in zip(bars4, test_mape):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, top_n=3):
        """
        Gráfico de predicciones vs valores reales para los mejores modelos
        """
        if not self.predictions:
            print("⚠️  No hay predicciones para graficar")
            return
        
        # Seleccionar top N modelos por R²
        model_scores = {}
        for model_name in self.predictions.keys():
            if model_name in self.metrics:
                model_scores[model_name] = self.metrics[model_name]['test']['R2_original']
        
        top_models = sorted(model_scores, key=model_scores.get, reverse=True)[:top_n]
        
        fig, axes = plt.subplots(1, top_n, figsize=(5*top_n, 5))
        if top_n == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(top_models):
            pred_data = self.predictions[model_name]
            y_true = np.expm1(pred_data['y_true_test'])  # Convertir a escala original
            y_pred = np.expm1(pred_data['y_pred_test'])  # Convertir a escala original
            
            ax = axes[idx]
            scatter = ax.scatter(y_true, y_pred, alpha=0.6, 
                               color=self.models[model_name]['color'], s=30)
            
            # Línea de perfecta predicción
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Valor Real ($)')
            ax.set_ylabel('Predicción ($)')
            ax.set_title(f'{model_name}\nR² = {self.metrics[model_name]["test"]["R2_original"]:.3f}', 
                        fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Formatear ejes en formato de dinero
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.show()
    
    def plot_residual_analysis(self, top_n=3):
        """
        Análisis de residuales para los mejores modelos
        """
        if not self.predictions:
            print("⚠️  No hay predicciones para analizar")
            return
        
        # Seleccionar top N modelos
        model_scores = {}
        for model_name in self.predictions.keys():
            if model_name in self.metrics:
                model_scores[model_name] = self.metrics[model_name]['test']['R2_original']
        
        top_models = sorted(model_scores, key=model_scores.get, reverse=True)[:top_n]
        
        fig, axes = plt.subplots(top_n, 2, figsize=(12, 4*top_n))
        if top_n == 1:
            axes = axes.reshape(1, -1)
        
        for idx, model_name in enumerate(top_models):
            pred_data = self.predictions[model_name]
            residuals = pred_data['residuals_test']
            y_pred = pred_data['y_pred_test']
            
            # Histograma de residuales
            axes[idx, 0].hist(residuals, bins=50, alpha=0.7, 
                             color=self.models[model_name]['color'], edgecolor='black')
            axes[idx, 0].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[idx, 0].set_title(f'{model_name} - Distribución de Residuales', fontweight='bold')
            axes[idx, 0].set_xlabel('Residuales (log scale)')
            axes[idx, 0].set_ylabel('Frecuencia')
            
            # Residuales vs predicciones
            axes[idx, 1].scatter(y_pred, residuals, alpha=0.6, 
                                color=self.models[model_name]['color'], s=30)
            axes[idx, 1].axhline(0, color='red', linestyle='--', linewidth=2)
            axes[idx, 1].set_title(f'{model_name} - Residuales vs Predicciones', fontweight='bold')
            axes[idx, 1].set_xlabel('Predicciones (log scale)')
            axes[idx, 1].set_ylabel('Residuales (log scale)')
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n=15):
        """
        Gráfico de importancia de características para modelos que lo soportan
        """
        if not self.feature_importance or self.feature_names is None:
            print("⚠️  No hay datos de importancia de características")
            return
        
        models_with_importance = [name for name, imp in self.feature_importance.items() 
                                 if imp is not None and len(imp) == len(self.feature_names)]
        
        if not models_with_importance:
            print("⚠️  Ningún modelo tiene importancia de características calculable")
            return
        
        # Tomar el mejor modelo con importancia de características
        best_model_with_imp = None
        for model_name in models_with_importance:
            if model_name == self.best_model_name:
                best_model_with_imp = model_name
                break
        
        if best_model_with_imp is None and models_with_importance:
            best_model_with_imp = models_with_importance[0]
        
        importance_values = self.feature_importance[best_model_with_imp]
        feature_imp_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_imp_df['feature'], feature_imp_df['importance'], 
                color='skyblue', alpha=0.7)
        plt.xlabel('Importancia')
        plt.title(f'Importancia de Características - {best_model_with_imp}\n(Top {top_n} Features)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
    
    def generate_comprehensive_report(self):
        """
        Generar reporte completo de todos los modelos
        """
        print("\n📈 GENERANDO REPORTE COMPLETO...")
        
        # 1. Gráfico de comparación
        self.plot_model_comparison()
        
        # 2. Predicciones vs reales
        self.plot_predictions_vs_actual()
        
        # 3. Análisis de residuales
        self.plot_residual_analysis()
        
        # 4. Importancia de características
        self.plot_feature_importance()
    
    def save_models(self, model_dir='modelos_entrenados'):
        """
        Guardar modelos entrenados y métricas
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
                    print(f"✅ {model_name} guardado en: {filepath}")
                except Exception as e:
                    print(f"❌ Error guardando {model_name}: {e}")
        
        # Guardar métricas
        if self.metrics:
            # Crear DataFrame con métricas
            metrics_data = []
            for model_name, model_metrics in self.metrics.items():
                row = {'Modelo': model_name}
                row.update({f'Train_{k}': v for k, v in model_metrics['train'].items()})
                row.update({f'Test_{k}': v for k, v in model_metrics['test'].items()})
                row['Overfit_Gap'] = model_metrics['overfit_gap']
                metrics_data.append(row)
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = os.path.join(model_dir, f'metricas_detalladas_{timestamp}.csv')
            metrics_df.to_csv(metrics_path, index=False)
            print(f"✅ Métricas guardadas en: {metrics_path}")
            
            # Guardar resumen ejecutivo
            summary_path = os.path.join(model_dir, f'resumen_ejecutivo_{timestamp}.txt')
            with open(summary_path, 'w') as f:
                f.write("RESUMEN EJECUTIVO - MODELOS ENTRENADOS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Mejor modelo: {self.best_model_name}\n")
                f.write(f"R² test (original): {self.metrics[self.best_model_name]['test']['R2_original']:.4f}\n")
                f.write(f"RMSE test (original): ${self.metrics[self.best_model_name]['test']['RMSE_original']:,.0f}\n")
                f.write(f"MAPE: {self.metrics[self.best_model_name]['test']['MAPE_%']:.2f}%\n\n")
                
                f.write("COMPARACIÓN DE MODELOS:\n")
                for model_name in self.metrics.keys():
                    f.write(f"- {model_name}: R² = {self.metrics[model_name]['test']['R2_original']:.4f}, "
                           f"RMSE = ${self.metrics[model_name]['test']['RMSE_original']:,.0f}\n")
            
            print(f"✅ Resumen ejecutivo guardado en: {summary_path}")
        
        return saved_models
    
    def print_detailed_report(self):
        """
        Imprimir reporte detallado en consola
        """
        if not self.metrics:
            print("⚠️  No hay métricas para reportar")
            return
        
        print("\n" + "=" * 100)
        print("📊 REPORTE DETALLADO DE MODELOS")
        print("=" * 100)
        
        # Crear tabla resumen
        summary_data = []
        for model_name, model_metrics in self.metrics.items():
            summary_data.append({
                'Modelo': model_name,
                'R² Train': f"{model_metrics['train']['R2_original']:.4f}",
                'R² Test': f"{model_metrics['test']['R2_original']:.4f}",
                'Overfit Gap': f"{model_metrics['overfit_gap']:.4f}",
                'RMSE Test': f"${model_metrics['test']['RMSE_original']:,.0f}",
                'MAPE': f"{model_metrics['test']['MAPE_%']:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nRESUMEN COMPARATIVO:")
        print(summary_df.to_string(index=False))
        
        # Análisis del mejor modelo
        if self.best_model_name:
            print(f"\n🏆 MODELO RECOMENDADO: {self.best_model_name}")
            best_metrics = self.metrics[self.best_model_name]['test']
            print(f"   • R² (test): {best_metrics['R2_original']:.4f}")
            print(f"   • RMSE: ${best_metrics['RMSE_original']:,.0f}")
            print(f"   • MAE: ${best_metrics['MAE_original']:,.0f}")
            print(f"   • MAPE: {best_metrics['MAPE_%']:.2f}%")
            
            # Interpretación del performance
            if best_metrics['R2_original'] > 0.8:
                print("   ✅ EXCELENTE: El modelo explica más del 80% de la varianza")
            elif best_metrics['R2_original'] > 0.7:
                print("   👍 BUENO: El modelo explica más del 70% de la varianza")
            elif best_metrics['R2_original'] > 0.6:
                print("   📊 ACEPTABLE: El modelo explica más del 60% de la varianza")
            else:
                print("   🔄 MEJORABLE: Considerar ajustar características o modelo")
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, feature_names=None, 
                          tune_hyperparams=False, save_models=True):
        """
        Función principal que ejecuta todo el pipeline
        """
        print("🚀 INICIANDO ENTRENAMIENTO Y EVALUACIÓN COMPLETA")
        print("=" * 60)
        
        self.feature_names = feature_names
        
        try:
            # 1. Inicializar modelos
            self.initialize_models()
            
            # 2. Entrenar y evaluar modelos
            self.train_models(X_train, X_test, y_train, y_test, feature_names, tune_hyperparams)
            
            # 3. Generar reporte completo
            self.generate_comprehensive_report()
            
            # 4. Guardar modelos
            if save_models:
                self.save_models()
            
            # 5. Imprimir reporte detallado
            self.print_detailed_report()
            
            print("\n🎉 ENTRENAMIENTO Y EVALUACIÓN COMPLETADOS EXITOSAMENTE")
            
            return self.metrics
            
        except Exception as e:
            print(f"❌ Error en el pipeline de entrenamiento: {e}")
            traceback.print_exc()
            return {}

# Función de conveniencia para uso rápido
def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names=None, 
                             random_state=42, tune_hyperparams=False, save_models=True):
    """
    Función principal simplificada
    """
    trainer = ComprehensiveModelTrainer(random_state=random_state)
    metrics = trainer.train_and_evaluate(
        X_train, X_test, y_train, y_test, 
        feature_names=feature_names,
        tune_hyperparams=tune_hyperparams,
        save_models=save_models
    )
    return metrics

# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 MÓDULO DE MODELOS - VERSIÓN CORREGIDA")
    print("📝 Ejecuta train_and_evaluate_models(X_train, X_test, y_train, y_test) para usar el módulo")
    
    # Ejemplo con datos dummy
    try:
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        
        print("\n🧪 EJECUTANDO EJEMPLO CON DATOS DUMMY...")
        
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        metrics = train_and_evaluate_models(
            X_train, X_test, y_train, y_test,
            feature_names=feature_names,
            save_models=False
        )
        
    except Exception as e:
        print(f"❌ Error en ejemplo: {e}")