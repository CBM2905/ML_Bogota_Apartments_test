"""
Utilidades adicionales para evaluación y análisis de modelos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compare_models_metrics(metrics_dict):
    """
    Comparar métricas de múltiples modelos visualmente
    
    Args:
        metrics_dict (dict): Diccionario con métricas por modelo
    """
    if not metrics_dict:
        print("⚠️  No hay métricas para comparar")
        return
    
    # Convertir a DataFrame
    metrics_df = pd.DataFrame(metrics_dict).T
    
    # Crear gráfico de comparación
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('COMPARACIÓN DE MÉTRICAS POR MODELO', fontsize=16, fontweight='bold')
    
    # Gráfico de RMSE
    axes[0, 0].bar(metrics_df.index, metrics_df['RMSE'], color='lightcoral', alpha=0.7)
    axes[0, 0].set_title('RMSE por Modelo')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Gráfico de MAE
    axes[0, 1].bar(metrics_df.index, metrics_df['MAE'], color='lightblue', alpha=0.7)
    axes[0, 1].set_title('MAE por Modelo')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Gráfico de R²
    axes[1, 0].bar(metrics_df.index, metrics_df['R2'], color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('R² por Modelo')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Gráfico de comparación múltiple
    metrics_df[['RMSE', 'MAE']].plot(kind='bar', ax=axes[1, 1], alpha=0.7)
    axes[1, 1].set_title('Comparación RMSE vs MAE')
    axes[1, 1].set_ylabel('Valor')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return metrics_df

def analyze_residuals(y_true, y_pred, model_name):
    """
    Analizar y visualizar residuos del modelo
    
    Args:
        y_true (array): Valores reales
        y_pred (array): Predicciones
        model_name (str): Nombre del modelo
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'ANÁLISIS DE RESIDUALES - {model_name}', fontweight='bold')
    
    # Histograma de residuos
    axes[0].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title('Distribución de Residuales')
    axes[0].set_xlabel('Residuales')
    axes[0].set_ylabel('Frecuencia')
    
    # Residuales vs Predicciones
    axes[1].scatter(y_pred, residuals, alpha=0.6, color='orange')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title('Residuales vs Predicciones')
    axes[1].set_xlabel('Predicciones')
    axes[1].set_ylabel('Residuales')
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas de residuales
    residual_stats = {
        'Media': np.mean(residuals),
        'Desviación Estándar': np.std(residuals),
        'Mínimo': np.min(residuals),
        'Máximo': np.max(residuals),
        'Porcentaje dentro de ±1 SD': np.mean(np.abs(residuals) <= np.std(residuals)) * 100
    }
    
    print(f"📊 Estadísticas de Residuales - {model_name}:")
    for stat, value in residual_stats.items():
        print(f"   • {stat}: {value:.4f}")
    
    return residual_stats

def generate_model_report(metrics_dict, output_path="reporte_modelos.txt"):
    """
    Generar reporte completo de modelos en archivo de texto
    
    Args:
        metrics_dict (dict): Métricas por modelo
        output_path (str): Ruta del archivo de salida
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("REPORTE DE MODELOS - BOGOTÁ APARTMENTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"FECHA: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if not metrics_dict:
            f.write("No hay modelos entrenados para reportar.\n")
            return
        
        # Encontrar mejor modelo
        metrics_df = pd.DataFrame(metrics_dict).T
        best_model = metrics_df['R2'].idxmax()
        best_r2 = metrics_df.loc[best_model, 'R2']
        
        f.write("RESUMEN EJECUTIVO:\n")
        f.write(f"• Mejor modelo: {best_model} (R² = {best_r2:.4f})\n")
        f.write(f"• Total de modelos evaluados: {len(metrics_dict)}\n\n")
        
        f.write("MÉTRICAS DETALLADAS POR MODELO:\n")
        f.write("-" * 50 + "\n")
        
        for model_name, metrics in metrics_dict.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  • RMSE: {metrics['RMSE']:.4f}\n")
            f.write(f"  • MAE: {metrics['MAE']:.4f}\n")
            f.write(f"  • R²: {metrics['R2']:.4f}\n")
        
        f.write(f"\nRECOMENDACIONES:\n")
        f.write(f"• Modelo recomendado para producción: {best_model}\n")
        f.write(f"• Justificación: Mayor poder predictivo (R² = {best_r2:.4f})\n")
        
        # Análisis de rendimiento relativo
        f.write(f"\nANÁLISIS DE RENDIMIENTO RELATIVO:\n")
        for metric in ['RMSE', 'MAE']:
            best_val = metrics_df[metric].min()
            best_model_metric = metrics_df[metric].idxmin()
            f.write(f"• Mejor {metric}: {best_model_metric} ({best_val:.4f})\n")
    
    print(f"✅ Reporte guardado en: {output_path}")

# Función para cargar y usar modelos guardados
def load_and_predict(model_path, X_new):
    """
    Cargar modelo guardado y hacer predicciones
    
    Args:
        model_path (str): Ruta al modelo guardado
        X_new (array): Nuevos datos para predecir
        
    Returns:
        array: Predicciones
    """
    try:
        model = joblib.load(model_path)
        predictions = model.predict(X_new)
        print(f"✅ Predicciones realizadas con modelo: {os.path.basename(model_path)}")
        return predictions
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None

if __name__ == "__main__":
    # Ejemplo de uso de las utilidades
    print("🔧 MÓDULO DE UTILIDADES PARA EVALUACIÓN DE MODELOS")