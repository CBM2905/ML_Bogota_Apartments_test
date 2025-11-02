# %% [markdown]
# ## 📊 MÓDULO EDA MEJORADO: Análisis Exploratorio Automatizado

# %%

"""
Módulo EDA Mejorado para Bogotá Apartments
Análisis Exploratorio de Datos Automatizado, Interactivo y Profesional
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy import stats as scipy_stats
import warnings
import json
import os
from datetime import datetime
import folium
from folium import plugins

# Configuración de estilo profesional
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class EnhancedDataExplorer:
    """
    Clase para Análisis Exploratorio de Datos Automatizado y Profesional
    """
    
    def __init__(self, results_dir='bogota_apartments_ml/results/eda'):
        """
        Inicializar el explorador de datos
        
        Args:
            results_dir (str): Directorio para guardar resultados
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.insights = []
        self.summary_stats = {}
        
        print("🔍 Inicializando Explorador de Datos Mejorado...")
    
    def dataset_overview(self, df):
        """
        Resumen completo y profesional del dataset
        
        Args:
            df (pd.DataFrame): DataFrame a analizar
        """
        print("=" * 80)
        print("📊 ANÁLISIS EXPLORATORIO COMPLETO - BOGOTÁ APARTMENTS")
        print("=" * 80)
        
        # Información básica
        print(f"📈 DIMENSIONES: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        print(f"💾 MEMORIA: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Resumen de tipos de datos
        print("\n🎯 RESUMEN DE TIPOS DE DATOS:")
        dtype_summary = df.dtypes.value_counts()
        for dtype, count in dtype_summary.items():
            print(f"   {dtype}: {count} columnas")
        
        # Análisis de valores nulos
        self._analyze_missing_values(df)
        
        # Estadísticas descriptivas básicas
        self._basic_descriptive_stats(df)
        
        return self._create_overview_dataframe(df)
    
    def _analyze_missing_values(self, df):
        """Análisis profundo de valores faltantes"""
        print("\n🔍 ANÁLISIS DE VALORES FALTANTES:")
        
        null_summary = df.isnull().sum()
        null_percentage = (null_summary / len(df)) * 100
        
        # Columnas con valores nulos
        null_columns = null_summary[null_summary > 0]
        
        if len(null_columns) > 0:
            print("   Columnas con valores nulos:")
            for col, null_count in null_columns.items():
                pct = null_percentage[col]
                print(f"   • {col}: {null_count:,} nulos ({pct:.1f}%)")
                
                # Insight automático
                if pct > 50:
                    self.insights.append(f"⚠️ La columna '{col}' tiene más del 50% de valores nulos ({pct:.1f}%)")
                elif pct > 20:
                    self.insights.append(f"📝 La columna '{col}' tiene {pct:.1f}% de valores nulos, considerar imputación")
        else:
            print("   ✅ No hay valores nulos en el dataset")
            self.insights.append("✅ Dataset completo sin valores nulos")
    
    def _basic_descriptive_stats(self, df):
        """Estadísticas descriptivas básicas"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            print(f"\n📊 ESTADÍSTICAS DESCRIPTIVAS ({len(numeric_cols)} variables numéricas):")
            
            # Seleccionar columnas clave para mostrar
            key_columns = ['precio_venta', 'area', 'habitaciones', 'banos', 'estrato', 'precio_m2']
            available_keys = [col for col in key_columns if col in numeric_cols]
            
            if available_keys:
                stats_df = df[available_keys].describe()
                display(stats_df.round(2))
    
    def _create_overview_dataframe(self, df):
        """Crear DataFrame de resumen del dataset"""
        overview_data = []
        
        for col in df.columns:
            col_info = {
                'columna': col,
                'tipo': str(df[col].dtype),
                'no_nulos': df[col].count(),
                'nulos': df[col].isnull().sum(),
                '%_nulos': (df[col].isnull().sum() / len(df)) * 100,
                'valores_unicos': df[col].nunique()
            }
            
            if np.issubdtype(df[col].dtype, np.number):
                col_info.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'media': df[col].mean(),
                    'mediana': df[col].median()
                })
            else:
                col_info['valor_mas_frecuente'] = df[col].mode()[0] if not df[col].mode().empty else 'N/A'
            
            overview_data.append(col_info)
        
        return pd.DataFrame(overview_data)
    
    def advanced_statistical_analysis(self, df):
        """
        Análisis estadístico avanzado
        
        Args:
            df (pd.DataFrame): DataFrame a analizar
        """
        print("\n" + "=" * 60)
        print("📈 ANÁLISIS ESTADÍSTICO AVANZADO")
        print("=" * 60)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("❌ No hay variables numéricas para análisis estadístico")
            return
        
        # Análisis de distribución
        self._distribution_analysis(df, numeric_cols)
        
        # Análisis de correlación mejorado
        self._enhanced_correlation_analysis(df)
        
        # Análisis de outliers
        self._comprehensive_outlier_analysis(df, numeric_cols)
    
    def _distribution_analysis(self, df, numeric_cols):
        """Análisis de distribución de variables numéricas"""
        print("\n📊 ANÁLISIS DE DISTRIBUCIÓN:")
        
        distribution_data = []
        
        for col in numeric_cols[:10]:  # Limitar a las primeras 10 columnas
            data = df[col].dropna()
            
            # Estadísticas de distribución
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            normality_test = stats.normaltest(data)
            
            dist_info = {
                'variable': col,
                'asimetria': skewness,
                'curtosis': kurtosis,
                'p_valor_normalidad': normality_test.pvalue,
                'es_normal': normality_test.pvalue > 0.05
            }
            
            distribution_data.append(dist_info)
            
            # Insights automáticos
            if abs(skewness) > 1:
                direction = "derecha" if skewness > 0 else "izquierda"
                self.insights.append(f"📊 La variable '{col}' tiene distribución sesgada a la {direction} (asimetría: {skewness:.2f})")
        
        dist_df = pd.DataFrame(distribution_data)
        display(dist_df.round(4))
    
    def _enhanced_correlation_analysis(self, df):
        """
        Análisis de correlación mejorado con visualizaciones avanzadas
        """
        print("\n🔗 ANÁLISIS DE CORRELACIÓN AVANZADO")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            print("❌ No hay suficientes variables numéricas para análisis de correlación")
            return
        
        # 1. Matriz de correlación completa
        correlation_matrix = numeric_df.corr()
        
        # Crear máscara para el triángulo superior
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Heatmap mejorado
        plt.figure(figsize=(14, 12))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True, 
                   fmt='.2f',
                   cbar_kws={"shrink": .8},
                   annot_kws={"size": 8})
        plt.title('🔗 MATRIZ DE CORRELACIÓN - Variables Numéricas\n', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Top correlaciones con precio_venta
        if 'precio_venta' in correlation_matrix.columns:
            self._price_correlation_analysis(correlation_matrix, numeric_df)
        
        # 3. Matriz de correlación clustermap
        self._create_clustered_correlation(df, numeric_df)
    
    def _price_correlation_analysis(self, correlation_matrix, numeric_df):
        """Análisis específico de correlaciones con precio"""
        price_correlations = correlation_matrix['precio_venta'].sort_values(ascending=False)
        
        # Filtrar correlaciones significativas (excluyendo precio_venta consigo mismo)
        significant_correlations = price_correlations[price_correlations.index != 'precio_venta']
        top_correlations = significant_correlations.head(10)
        
        print(f"\n🎯 TOP 10 CORRELACIONES CON PRECIO_VENTA:")
        for i, (variable, corr) in enumerate(top_correlations.items(), 1):
            print(f"   {i:2d}. {variable:25} : {corr:+.3f}")
            
            # Insights automáticos
            if abs(corr) > 0.5:
                direction = "positiva" if corr > 0 else "negativa"
                self.insights.append(f"💰 '{variable}' tiene correlación {direction} fuerte con el precio (r={corr:.2f})")
        
        # Gráfico de barras horizontal para top correlaciones
        plt.figure(figsize=(12, 8))
        top_correlations.plot(kind='barh', color=['green' if x > 0 else 'red' for x in top_correlations.values])
        plt.title('Top 10 Correlaciones con Precio de Venta', fontsize=14, fontweight='bold')
        plt.xlabel('Coeficiente de Correlación')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/top_price_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_clustered_correlation(self, df, numeric_df):
        """Crear clustermap de correlaciones"""
        if len(numeric_df.columns) > 3:  # Solo si hay suficientes variables
            print("\n🌳 Generando clustermap de correlaciones...")
            
            # Seleccionar solo algunas variables para no saturar
            important_cols = ['precio_venta', 'area', 'habitaciones', 'banos', 'estrato', 
                            'precio_m2', 'amenities_score', 'administracion', 'parqueaderos']
            available_cols = [col for col in important_cols if col in numeric_df.columns]
            
            if len(available_cols) > 3:
                plt.figure(figsize=(12, 10))
                sns.clustermap(numeric_df[available_cols].corr(), 
                              annot=True, 
                              cmap='RdBu_r', 
                              center=0,
                              fmt='.2f',
                              figsize=(12, 10))
                plt.title('Clustermap de Correlaciones', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{self.results_dir}/correlation_clustermap.png', dpi=300, bbox_inches='tight')
                plt.show()
    
    def _comprehensive_outlier_analysis(self, df, numeric_cols):
        """
        Análisis completo de outliers usando múltiples métodos
        """
        print("\n📊 ANÁLISIS COMPLETO DE OUTLIERS")
        
        outlier_summary = []
        key_columns = ['precio_venta', 'area', 'precio_m2', 'habitaciones', 'banos']
        available_keys = [col for col in key_columns if col in numeric_cols]
        
        for col in available_keys:
            data = df[col].dropna()
            
            # Método IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # Método Z-score (3 desviaciones estándar)
            z_scores = np.abs(stats.zscore(data))
            z_outliers = data[z_scores > 3]
            
            outlier_info = {
                'variable': col,
                'total_valores': len(data),
                'outliers_iqr': len(iqr_outliers),
                '%_outliers_iqr': (len(iqr_outliers) / len(data)) * 100,
                'outliers_zscore': len(z_outliers),
                '%_outliers_zscore': (len(z_outliers) / len(data)) * 100
            }
            
            outlier_summary.append(outlier_info)
            
            # Insights automáticos
            if len(iqr_outliers) / len(data) > 0.05:  # Más del 5% son outliers
                self.insights.append(f"🚨 La variable '{col}' tiene {len(iqr_outliers):,} outliers ({len(iqr_outliers)/len(data)*100:.1f}%)")
        
        outlier_df = pd.DataFrame(outlier_summary)
        display(outlier_df.round(2))
        
        # Visualización de outliers
        self._visualize_outliers(df, available_keys)
    
    def _visualize_outliers(self, df, columns):
        """Visualización de outliers para variables clave"""
        if len(columns) == 0:
            return
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, col in enumerate(columns):
            if i < len(axes):
                # Boxplot
                df.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'Outliers - {col}', fontweight='bold')
                axes[i].grid(True, alpha=0.3)
        
        # Ocultar ejes vacíos
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/outliers_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def price_distribution_analysis(self, df):
        """
        Análisis completo de distribución de precios
        """
        if 'precio_venta' not in df.columns:
            print("❌ No se encuentra la columna 'precio_venta'")
            return
        
        print("\n" + "=" * 60)
        print("💰 ANÁLISIS DE DISTRIBUCIÓN DE PRECIOS")
        print("=" * 60)
        
        prices = df['precio_venta'].dropna()
        
        # Estadísticas detalladas
        print(f"📊 ESTADÍSTICAS DE PRECIOS:")
        stats_dict = {
            'Mínimo': f"${prices.min():,.0f} COP",
            'Máximo': f"${prices.max():,.0f} COP", 
            'Promedio': f"${prices.mean():,.0f} COP",
            'Mediana': f"${prices.median():,.0f} COP",
            'Desviación Estándar': f"${prices.std():,.0f} COP",
            'Coef. Variación': f"{(prices.std() / prices.mean()) * 100:.1f}%",
            'Asimetría': f"{stats.skew(prices):.2f}",
            'Curtosis': f"{stats.kurtosis(prices):.2f}"
        }
        
        for stat, value in stats_dict.items():
            print(f"   {stat}: {value}")
        
        # Visualizaciones múltiples
        self._create_price_visualizations(df, prices)
        
        # Análisis de precios por m2
        if 'precio_m2' in df.columns:
            self._price_per_sqm_analysis(df)
    
    def _create_price_visualizations(self, df, prices):
        """Crear visualizaciones múltiples para precios"""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Histograma lineal
        plt.subplot(2, 3, 1)
        plt.hist(prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribución de Precios (Lineal)', fontweight='bold')
        plt.xlabel('Precio (COP)')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        
        # 2. Histograma logarítmico
        plt.subplot(2, 3, 2)
        log_prices = np.log1p(prices)
        plt.hist(log_prices, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Distribución Logarítmica de Precios', fontweight='bold')
        plt.xlabel('Log(Precio + 1)')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        
        # 3. Boxplot
        plt.subplot(2, 3, 3)
        plt.boxplot(prices)
        plt.title('Boxplot de Precios', fontweight='bold')
        plt.ylabel('Precio (COP)')
        plt.grid(True, alpha=0.3)
        
        # 4. QQ plot
        plt.subplot(2, 3, 4)
        stats.probplot(log_prices, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normalidad)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 5. Distribución acumulativa
        plt.subplot(2, 3, 5)
        sorted_prices = np.sort(prices)
        y_vals = np.arange(len(sorted_prices)) / float(len(sorted_prices))
        plt.plot(sorted_prices, y_vals)
        plt.title('Distribución Acumulativa', fontweight='bold')
        plt.xlabel('Precio (COP)')
        plt.ylabel('Probabilidad Acumulada')
        plt.grid(True, alpha=0.3)
        
        # 6. Densidad KDE
        plt.subplot(2, 3, 6)
        sns.kdeplot(prices, fill=True, color='purple', alpha=0.7)
        plt.title('Estimación de Densidad (KDE)', fontweight='bold')
        plt.xlabel('Precio (COP)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/price_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _price_per_sqm_analysis(self, df):
        """Análisis de precios por metro cuadrado"""
        print(f"\n📏 ANÁLISIS DE PRECIO POR m²:")
        
        precio_m2 = df['precio_m2'].dropna()
        
        print(f"   Precio m² promedio: ${precio_m2.mean():,.0f} COP")
        print(f"   Precio m² mediano: ${precio_m2.median():,.0f} COP")
        print(f"   Rango: ${precio_m2.min():,.0f} - ${precio_m2.max():,.0f} COP")
        
        # Distribución por localidad
        if 'localidad' in df.columns:
            precio_por_localidad = df.groupby('localidad')['precio_m2'].agg(['mean', 'median', 'count']).round(0)
            precio_por_localidad = precio_por_localidad.sort_values('mean', ascending=False)
            
            print(f"\n🏙️  PRECIO m² POR LOCALIDAD (Top 10):")
            display(precio_por_localidad.head(10))
            
            # Insight automático
            top_localidad = precio_por_localidad.index[0]
            top_precio = precio_por_localidad['mean'].iloc[0]
            avg_precio = precio_m2.mean()
            diferencia_pct = ((top_precio - avg_precio) / avg_precio) * 100
            
            self.insights.append(
                f"🏆 {top_localidad} tiene el precio m² más alto (${top_precio:,.0f} COP), "
                f"un {diferencia_pct:+.1f}% sobre el promedio"
            )
    
    def categorical_analysis(self, df):
        """
        Análisis avanzado de variables categóricas
        """
        print("\n" + "=" * 60)
        print("📝 ANÁLISIS DE VARIABLES CATEGÓRICAS")
        print("=" * 60)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            print("❌ No hay variables categóricas para analizar")
            return
        
        print(f"🔍 Analizando {len(categorical_cols)} variables categóricas...")
        
        # Análisis por tipo de variable categórica importante
        important_categorical = ['localidad', 'barrio', 'tipo_propiedad', 'antiguedad']
        available_categorical = [col for col in important_categorical if col in categorical_cols]
        
        for col in available_categorical:
            self._analyze_single_categorical(df, col)
    
    def _analyze_single_categorical(self, df, column):
        """Análisis individual de variable categórica"""
        print(f"\n📊 ANÁLISIS DE: {column.upper()}")
        
        value_counts = df[column].value_counts()
        value_pct = df[column].value_counts(normalize=True) * 100
        
        print(f"   Valores únicos: {df[column].nunique()}")
        print(f"   Top 5 categorías:")
        
        for i, (categoria, count) in enumerate(value_counts.head().items()):
            pct = value_pct[categoria]
            print(f"     {i+1}. {categoria}: {count:,} ({pct:.1f}%)")
        
        # Análisis de precios por categoría si existe precio_venta
        if 'precio_venta' in df.columns:
            precio_por_categoria = df.groupby(column)['precio_venta'].agg(['mean', 'median', 'count']).round(0)
            precio_por_categoria = precio_por_categoria.sort_values('mean', ascending=False)
            
            print(f"\n💰 PRECIO PROMEDIO POR {column.upper()} (Top 10):")
            display(precio_por_categoria.head(10))
            
            # Visualización
            self._create_categorical_visualizations(df, column, value_counts)
    
    def _create_categorical_visualizations(self, df, column, value_counts):
        """Crear visualizaciones para variables categóricas"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top categorías (bar chart)
        top_categories = value_counts.head(10)
        axes[0, 0].barh(range(len(top_categories)), top_categories.values)
        axes[0, 0].set_yticks(range(len(top_categories)))
        axes[0, 0].set_yticklabels(top_categories.index)
        axes[0, 0].set_title(f'Top 10 {column.title()}', fontweight='bold')
        axes[0, 0].set_xlabel('Frecuencia')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribución de precios por categoría (boxplot)
        if 'precio_venta' in df.columns:
            # Tomar solo las top categorías para no saturar
            top_cats = value_counts.head(8).index
            df_top = df[df[column].isin(top_cats)]
            
            sns.boxplot(data=df_top, x=column, y='precio_venta', ax=axes[0, 1])
            axes[0, 1].set_title(f'Distribución de Precios por {column.title()}', fontweight='bold')
            axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
            axes[0, 1].set_ylabel('Precio (COP)')
        
        # 3. Violin plot
        if 'precio_venta' in df.columns and len(top_cats) > 0:
            sns.violinplot(data=df_top, x=column, y='precio_venta', ax=axes[1, 0])
            axes[1, 0].set_title(f'Distribución Violin - {column.title()}', fontweight='bold')
            axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
            axes[1, 0].set_ylabel('Precio (COP)')
        
        # 4. Precio promedio por categoría
        if 'precio_venta' in df.columns:
            avg_prices = df.groupby(column)['precio_venta'].mean().sort_values(ascending=False).head(10)
            axes[1, 1].barh(range(len(avg_prices)), avg_prices.values)
            axes[1, 1].set_yticks(range(len(avg_prices)))
            axes[1, 1].set_yticklabels(avg_prices.index)
            axes[1, 1].set_title(f'Precio Promedio por {column.title()}', fontweight='bold')
            axes[1, 1].set_xlabel('Precio Promedio (COP)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/categorical_analysis_{column}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def geographic_analysis(self, df):
        """
        Análisis geográfico con visualizaciones avanzadas
        """
        if not all(col in df.columns for col in ['latitud', 'longitud']):
            print("❌ No se encuentran coordenadas para análisis geográfico")
            return
        
        print("\n" + "=" * 60)
        print("🗺️  ANÁLISIS GEOGRÁFICO AVANZADO")
        print("=" * 60)
        
        # Filtrar coordenadas válidas en Bogotá
        df_geo = df.dropna(subset=['latitud', 'longitud']).copy()
        df_geo = df_geo[
            (df_geo['latitud'].between(4.4, 4.9)) & 
            (df_geo['longitud'].between(-74.3, -74.0))
        ]
        
        print(f"📍 {len(df_geo)} propiedades con coordenadas válidas en Bogotá")
        
        # Crear múltiples visualizaciones geográficas
        self._create_geographic_visualizations(df_geo)
        
        # Análisis de densidad
        self._density_analysis(df_geo)
    
    def _create_geographic_visualizations(self, df):
        """Crear visualizaciones geográficas múltiples"""
        # 1. Scatter plot básico
        plt.figure(figsize=(15, 12))
        
        if 'precio_venta' in df.columns:
            # Usar precio para colorear
            scatter = plt.scatter(df['longitud'], df['latitud'], 
                                c=df['precio_venta'], 
                                cmap='viridis', 
                                alpha=0.6, 
                                s=10)
            plt.colorbar(scatter, label='Precio de Venta (COP)')
            title_suffix = ' (Coloreado por Precio)'
        else:
            plt.scatter(df['longitud'], df['latitud'], alpha=0.6, s=10)
            title_suffix = ''
        
        plt.title(f'Distribución Geográfica de Propiedades{title_suffix}', fontweight='bold')
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.grid(True, alpha=0.3)
        
        # Añadir referencia de Bogotá
        plt.axhline(y=4.6097, color='red', linestyle='--', alpha=0.7, label='Centro Bogotá')
        plt.axvline(x=-74.0817, color='red', linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/geographic_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Mapa de calor si hay suficientes puntos
        if len(df) > 100:
            self._create_heatmap(df)
    
    def _create_heatmap(self, df):
        """Crear mapa de calor de precios"""
        print("\n🔥 Generando mapa de calor...")
        
        # Mapa de Folium
        bogota_center = [4.6097, -74.0817]
        m = folium.Map(location=bogota_center, zoom_start=11, tiles='OpenStreetMap')
        
        # Datos para heatmap (muestrear si es muy grande)
        sample_size = min(2000, len(df))
        df_sample = df.sample(sample_size, random_state=42)
        
        heat_data = [[row['latitud'], row['longitud'], row.get('precio_venta', 1)] 
                    for idx, row in df_sample.iterrows()]
        
        plugins.HeatMap(heat_data, 
                       gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'},
                       min_opacity=0.5,
                       max_opacity=0.8,
                       radius=15).add_to(m)
        
        # Guardar mapa
        m.save(f'{self.results_dir}/price_heatmap.html')
        print(f"✅ Mapa de calor guardado: {self.results_dir}/price_heatmap.html")
        
        # Mostrar mapa en notebook
        display(m)
    
    def _density_analysis(self, df):
        """Análisis de densidad geográfica"""
        if 'localidad' in df.columns:
            print(f"\n🏙️  DENSIDAD POR LOCALIDAD:")
            
            densidad_por_localidad = df['localidad'].value_counts()
            display(densidad_por_localidad.head(10))
            
            # Insight automático
            top_localidad = densidad_por_localidad.index[0]
            count_top = densidad_por_localidad.iloc[0]
            pct_top = (count_top / len(df)) * 100
            
            self.insights.append(
                f"📈 {top_localidad} concentra el {pct_top:.1f}% de las propiedades ({count_top:,} propiedades)"
            )
    
    def automated_insights_generation(self, df):
        """
        Generar insights automáticos basados en el análisis
        """
        print("\n" + "=" * 60)
        print("💡 INSIGHTS AUTOMÁTICOS GENERADOS")
        print("=" * 60)
        
        # Insights de precio
        self._generate_price_insights(df)
        
        # Insights geográficos
        self._generate_geographic_insights(df)
        
        # Insights de características
        self._generate_feature_insights(df)
        
        # Mostrar todos los insights
        print(f"\n🎯 SE GENERARON {len(self.insights)} INSIGHTS:")
        for i, insight in enumerate(self.insights, 1):
            print(f"   {i:2d}. {insight}")
        
        # Guardar insights
        self._save_insights()
    
    def _generate_price_insights(self, df):
        """Generar insights relacionados con precios"""
        if 'precio_venta' in df.columns:
            precio_promedio = df['precio_venta'].mean()
            precio_mediano = df['precio_venta'].median()
            
            if precio_promedio > precio_mediano:
                self.insights.append("💰 La distribución de precios está sesgada a la derecha (promedio > mediana)")
            
            # Rango intercuartílico de precios
            Q1 = df['precio_venta'].quantile(0.25)
            Q3 = df['precio_venta'].quantile(0.75)
            iqr = Q3 - Q1
            self.insights.append(f"📊 El 50% central de precios está entre ${Q1:,.0f} y ${Q3:,.0f} COP (IQR: ${iqr:,.0f} COP)")
    
    def _generate_geographic_insights(self, df):
        """Generar insights geográficos"""
        if all(col in df.columns for col in ['localidad', 'precio_venta']):
            precio_por_localidad = df.groupby('localidad')['precio_venta'].mean().sort_values(ascending=False)
            
            if len(precio_por_localidad) >= 2:
                mas_cara = precio_por_localidad.index[0]
                mas_economica = precio_por_localidad.index[-1]
                diferencia = precio_por_localidad.iloc[0] - precio_por_localidad.iloc[-1]
                diferencia_pct = (diferencia / precio_por_localidad.iloc[-1]) * 100
                
                self.insights.append(
                    f"🏆 {mas_cara} es {diferencia_pct:.0f}% más cara que {mas_economica} "
                    f"(${diferencia:,.0f} COP de diferencia)"
                )
    
    def _generate_feature_insights(self, df):
        """Generar insights de características"""
        # Insight de amenities
        if 'amenities_score' in df.columns:
            avg_amenities = df['amenities_score'].mean()
            if avg_amenities < 2:
                self.insights.append("🏊 El score promedio de amenities es bajo, sugiriendo propiedades básicas")
            elif avg_amenities > 4:
                self.insights.append("🏊 El score promedio de amenities es alto, indicando propiedades premium")
        
        # Insight de estrato
        if 'estrato' in df.columns:
            estrato_mode = df['estrato'].mode()[0]
            self.insights.append(f"🏘️ El estrato más común es {estrato_mode}")
    
    def _save_insights(self):
        """Guardar insights en archivo JSON"""
        insights_data = {
            'timestamp': datetime.now().isoformat(),
            'total_insights': len(self.insights),
            'insights': self.insights
        }
        
        insights_file = f'{self.results_dir}/automated_insights.json'
        with open(insights_file, 'w', encoding='utf-8') as f:
            json.dump(insights_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Insights guardados en: {insights_file}")
    
    def create_comprehensive_report(self, df):
        """
        Crear reporte completo del EDA
        
        Args:
            df (pd.DataFrame): DataFrame a analizar
        """
        print("🚀 INICIANDO ANÁLISIS EXPLORATORIO COMPLETO...")
        
        # Ejecutar todos los análisis
        self.dataset_overview(df)
        self.advanced_statistical_analysis(df)
        self.price_distribution_analysis(df)
        self.categorical_analysis(df)
        self.geographic_analysis(df)
        self.automated_insights_generation(df)
        
        print("\n" + "=" * 80)
        print("🎉 ANÁLISIS EXPLORATORIO COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        print(f"📁 Resultados guardados en: {self.results_dir}")
        print(f"💡 Insights generados: {len(self.insights)}")
        print("=" * 80)

# Función de conveniencia
def perform_enhanced_eda(df, results_dir='bogota_apartments_ml/results/eda'):
    """
    Función conveniente para ejecutar EDA completo
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        results_dir (str): Directorio para resultados
        
    Returns:
        EnhancedDataExplorer: Instancia del explorador
    """
    explorer = EnhancedDataExplorer(results_dir)
    explorer.create_comprehensive_report(df)
    return explorer