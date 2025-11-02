"""
Módulo de Análisis Exploratorio de Datos para Bogotá Apartments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium

class DataExplorer:
    """Clase para análisis exploratorio de datos"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def dataset_overview(self, df):
        """
        Resumen general del dataset
        
        Args:
            df (pd.DataFrame): DataFrame a analizar
        """
        print("=" * 50)
        print("📊 ANÁLISIS EXPLORATORIO DE DATOS")
        print("=" * 50)
        
        print(f"📈 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
        
        # Tipos de datos
        dtype_counts = df.dtypes.value_counts()
        print(f"🔧 Tipos de datos:")
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columnas")
        
        # Valores nulos
        null_summary = df.isnull().sum()
        null_columns = null_summary[null_summary > 0]
        if len(null_columns) > 0:
            print(f"🔍 Columnas con valores nulos:")
            for col, null_count in null_columns.items():
                pct = (null_count / len(df)) * 100
                print(f"   {col}: {null_count} nulos ({pct:.1f}%)")
    
    def price_analysis(self, df):
        """
        Análisis de precios
        
        Args:
            df (pd.DataFrame): DataFrame con columna precio_venta
        """
        if 'precio_venta' not in df.columns:
            return
        
        prices = df['precio_venta'].dropna()
        
        print(f"\n💰 ANÁLISIS DE PRECIOS:")
        print(f"   Mínimo: ${prices.min():,.0f} COP")
        print(f"   Máximo: ${prices.max():,.0f} COP")
        print(f"   Promedio: ${prices.mean():,.0f} COP")
        print(f"   Mediana: ${prices.median():,.0f} COP")
        
        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histograma
        axes[0].hist(prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('Distribución de Precios')
        axes[0].set_xlabel('Precio (COP)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].grid(True, alpha=0.3)
        
        # Boxplot
        axes[1].boxplot(prices)
        axes[1].set_title('Boxplot de Precios')
        axes[1].set_ylabel('Precio (COP)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def location_analysis(self, df):
        """
        Análisis por ubicación
        
        Args:
            df (pd.DataFrame): DataFrame con datos de ubicación
        """
        if 'localidad' not in df.columns:
            return
        
        print(f"\n📍 ANÁLISIS POR LOCALIDAD:")
        
        # Precios por localidad
        localidad_stats = df.groupby('localidad').agg({
            'precio_venta': ['count', 'mean', 'median'],
            'area': 'mean',
            'estrato': 'mean'
        }).round(2)
        
        localidad_stats.columns = ['cantidad', 'precio_promedio', 'precio_mediano', 'area_promedio', 'estrato_promedio']
        localidad_stats = localidad_stats.sort_values('precio_promedio', ascending=False)
        
        print("🏙️ Precios por localidad (Top 10):")
        display(localidad_stats.head(10))
        
        # Gráfico
        plt.figure(figsize=(12, 6))
        top_localidades = localidad_stats.head(10).index
        df_top = df[df['localidad'].isin(top_localidades)]
        
        sns.boxplot(data=df_top, x='localidad', y='precio_venta')
        plt.title('Distribución de Precios por Localidad (Top 10)')
        plt.xlabel('Localidad')
        plt.ylabel('Precio de Venta (COP)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self, df):
        """
        Análisis de correlaciones
        
        Args:
            df (pd.DataFrame): DataFrame con variables numéricas
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return
        
        # Matriz de correlación
        correlation_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        plt.show()
        
        # Correlaciones con precio_venta
        if 'precio_venta' in numeric_df.columns:
            price_correlations = correlation_matrix['precio_venta'].sort_values(ascending=False)
            print(f"\n🔗 Correlaciones con precio_venta:")
            for col, corr in price_correlations.head(10).items():
                if col != 'precio_venta':
                    print(f"   {col}: {corr:.3f}")
    
    def run_complete_analysis(self, df):
        """
        Ejecutar análisis completo
        
        Args:
            df (pd.DataFrame): DataFrame a analizar
        """
        self.dataset_overview(df)
        self.price_analysis(df)
        self.location_analysis(df)
        self.correlation_analysis(df)
        
        print("\n✅ Análisis exploratorio completado")

# Función de conveniencia
def perform_eda(df):
    """Ejecutar EDA completo"""
    explorer = DataExplorer()
    explorer.run_complete_analysis(df)
    return explorer