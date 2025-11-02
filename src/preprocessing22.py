"""
Módulo de Análisis Exploratorio de Datos (EDA) para Bogotá Apartments
Análisis completo de variables originales sin transformaciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import warnings

# Configuración
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BogotaApartmentsEDA:
    """Clase para análisis exploratorio de datos de apartamentos en Bogotá"""
    
    def __init__(self):
        self.df = None
        self.original_columns = [
            'precio_venta', 'area', 'habitaciones', 'banos', 'estrato', 
            'parqueaderos', 'administracion', 'localidad', 'barrio', 'antiguedad'
        ]
        self.numeric_columns = []
        self.categorical_columns = []
        
    def load_data(self, file_path):
        """
        Cargar datos desde archivo Excel
        
        Args:
            file_path (str): Ruta al archivo Excel
        """
        logger.info(f"📥 Cargando datos desde: {file_path}")
        try:
            self.df = pd.read_excel(file_path)
            logger.info(f"✅ Datos cargados: {self.df.shape[0]} registros, {self.df.shape[1]} columnas")
            
            # Filtrar solo columnas originales que existan en el dataset
            available_columns = [col for col in self.original_columns if col in self.df.columns]
            self.df = self.df[available_columns]
            
            # Clasificar columnas
            self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
            
            logger.info(f"📊 Columnas numéricas: {len(self.numeric_columns)}")
            logger.info(f"📊 Columnas categóricas: {len(self.categorical_columns)}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            raise
    
    def dataset_overview(self):
        """Resumen general del dataset"""
        logger.info("📋 Generando resumen del dataset...")
        
        print("=" * 80)
        print("📊 RESUMEN GENERAL DEL DATASET - BOGOTÁ APARTMENTS")
        print("=" * 80)
        
        # Información básica
        print(f"📈 Dimensiones: {self.df.shape[0]} registros, {self.df.shape[1]} columnas")
        print(f"💰 Variable objetivo: precio_venta")
        print("\n")
        
        # Tipos de datos
        print("📝 TIPOS DE DATOS:")
        print(self.df.dtypes)
        print("\n")
        
        return self.df.shape
    
    def missing_values_analysis(self):
        """Análisis de valores faltantes"""
        logger.info("🔍 Analizando valores faltantes...")
        
        print("=" * 80)
        print("🔍 ANÁLISIS DE VALORES FALTANTES")
        print("=" * 80)
        
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Columna': missing_data.index,
            'Valores_Faltantes': missing_data.values,
            'Porcentaje': missing_percent.values
        })
        
        missing_df = missing_df[missing_df['Valores_Faltantes'] > 0].sort_values('Porcentaje', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.round(2))
            
            # Gráfico de valores faltantes
            plt.figure(figsize=(12, 6))
            missing_plot_data = missing_df[missing_df['Porcentaje'] > 0]
            
            if len(missing_plot_data) > 0:
                bars = plt.bar(missing_plot_data['Columna'], missing_plot_data['Porcentaje'], 
                              color='salmon', alpha=0.7)
                plt.title('Porcentaje de Valores Faltantes por Columna', fontsize=14, fontweight='bold')
                plt.xlabel('Columnas')
                plt.ylabel('Porcentaje de Valores Faltantes (%)')
                plt.xticks(rotation=45)
                
                # Añadir etiquetas en las barras
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.show()
        else:
            print("✅ No hay valores faltantes en el dataset")
        
        print("\n")
        return missing_df
    
    def numeric_descriptive_stats(self):
        """Estadísticas descriptivas para variables numéricas"""
        logger.info("📈 Calculando estadísticas descriptivas numéricas...")
        
        print("=" * 80)
        print("📊 ESTADÍSTICAS DESCRIPTIVAS - VARIABLES NUMÉRICAS")
        print("=" * 80)
        
        if self.numeric_columns:
            stats_df = self.df[self.numeric_columns].describe(percentiles=[.25, .5, .75, .95]).T
            stats_df['cv'] = (stats_df['std'] / stats_df['mean']) * 100  # Coeficiente de variación
            stats_df['skew'] = self.df[self.numeric_columns].skew()
            stats_df['kurtosis'] = self.df[self.numeric_columns].kurtosis()
            
            # Formatear para mejor presentación
            formatted_stats = stats_df.round(2)
            print(formatted_stats)
            print("\n")
            
            return stats_df
        else:
            print("❌ No hay variables numéricas para analizar")
            return pd.DataFrame()
    
    def numeric_distribution_analysis(self):
        """Análisis de distribución y outliers para variables numéricas"""
        logger.info("📊 Analizando distribuciones numéricas...")
        
        if not self.numeric_columns:
            logger.warning("⚠️ No hay variables numéricas para analizar")
            return
        
        print("=" * 80)
        print("📊 ANÁLISIS DE DISTRIBUCIÓN - VARIABLES NUMÉRICAS")
        print("=" * 80)
        
        n_numeric = len(self.numeric_columns)
        n_cols = 3
        n_rows = (n_numeric + n_cols - 1) // n_cols
        
        # Crear subplots para histogramas y boxplots
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(18, n_rows * 8))
        axes = axes.flatten()
        
        outlier_counts_iqr = {}
        outlier_counts_zscore = {}
        
        for i, col in enumerate(self.numeric_columns):
            # Histograma
            ax_hist = axes[i * 2]
            self.df[col].hist(bins=30, ax=ax_hist, color='skyblue', alpha=0.7, edgecolor='black')
            ax_hist.set_title(f'Distribución de {col}', fontweight='bold')
            ax_hist.set_xlabel(col)
            ax_hist.set_ylabel('Frecuencia')
            
            # Añadir líneas de media y mediana
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.2f}')
            ax_hist.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_val:.2f}')
            ax_hist.legend()
            
            # Boxplot
            ax_box = axes[i * 2 + 1]
            self.df.boxplot(column=col, ax=ax_box, color='lightgreen')
            ax_box.set_title(f'Boxplot de {col}', fontweight='bold')
            
            # Detección de outliers
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_iqr = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_counts_iqr[col] = len(outliers_iqr)
            
            # Outliers por z-score (abs(z) > 3)
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers_zscore = self.df[z_scores > 3]
            outlier_counts_zscore[col] = len(outliers_zscore)
            
            print(f"📌 {col}:")
            print(f"   - Outliers (IQR): {outlier_counts_iqr[col]} ({outlier_counts_iqr[col]/len(self.df)*100:.1f}%)")
            print(f"   - Outliers (Z-score > 3): {outlier_counts_zscore[col]} ({outlier_counts_zscore[col]/len(self.df)*100:.1f}%)")
        
        # Ocultar ejes vacíos
        for j in range(len(self.numeric_columns) * 2, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Resumen de outliers
        print("\n📌 RESUMEN DE OUTLIERS:")
        outlier_summary = pd.DataFrame({
            'Variable': list(outlier_counts_iqr.keys()),
            'Outliers_IQR': list(outlier_counts_iqr.values()),
            'Porcentaje_IQR': [f"{(count/len(self.df))*100:.1f}%" for count in outlier_counts_iqr.values()],
            'Outliers_ZScore': list(outlier_counts_zscore.values()),
            'Porcentaje_ZScore': [f"{(count/len(self.df))*100:.1f}%" for count in outlier_counts_zscore.values()]
        })
        print(outlier_summary)
        
        return outlier_counts_iqr, outlier_counts_zscore
    
    def correlation_analysis(self):
        """Análisis de correlación entre variables numéricas"""
        logger.info("🔗 Analizando correlaciones...")
        
        if len(self.numeric_columns) < 2:
            logger.warning("⚠️ No hay suficientes variables numéricas para análisis de correlación")
            return
        
        print("=" * 80)
        print("🔗 ANÁLISIS DE CORRELACIÓN")
        print("=" * 80)
        
        # Matriz de correlación
        correlation_matrix = self.df[self.numeric_columns].corr()
        
        # Mostrar matriz numérica
        print("Matriz de Correlación:")
        print(correlation_matrix.round(3))
        print("\n")
        
        # Heatmap de correlación
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={'shrink': .8})
        
        plt.title('Matriz de Correlación - Variables Numéricas', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Correlaciones fuertes con precio_venta
        if 'precio_venta' in self.numeric_columns:
            price_correlations = correlation_matrix['precio_venta'].sort_values(ascending=False)
            print("🔗 CORRELACIONES CON PRECIO_VENTA:")
            for var, corr in price_correlations.items():
                if var != 'precio_venta':
                    strength = "FUERTE" if abs(corr) > 0.5 else "MODERADA" if abs(corr) > 0.3 else "DÉBIL"
                    print(f"   {var}: {corr:.3f} ({strength})")
        
        return correlation_matrix
    
    def categorical_analysis(self):
        """Análisis de variables categóricas"""
        logger.info("📊 Analizando variables categóricas...")
        
        if not self.categorical_columns:
            logger.warning("⚠️ No hay variables categóricas para analizar")
            return
        
        print("=" * 80)
        print("📊 ANÁLISIS DE VARIABLES CATEGÓRICAS")
        print("=" * 80)
        
        categorical_stats = {}
        
        for col in self.categorical_columns:
            print(f"\n📌 ANÁLISIS DE: {col.upper()}")
            print("-" * 40)
            
            # Estadísticas básicas
            n_categories = self.df[col].nunique()
            n_missing = self.df[col].isnull().sum()
            value_counts = self.df[col].value_counts()
            
            print(f"Categorías únicas: {n_categories}")
            print(f"Valores faltantes: {n_missing}")
            print(f"Top 5 categorías más frecuentes:")
            
            # Top 5 categorías
            top_5 = value_counts.head(5)
            for category, count in top_5.items():
                percentage = (count / len(self.df)) * 100
                print(f"   - {category}: {count} ({percentage:.1f}%)")
            
            # Precio promedio por categoría (si existe precio_venta)
            if 'precio_venta' in self.df.columns:
                price_by_category = self.df.groupby(col)['precio_venta'].agg(['mean', 'median', 'count'])
                price_by_category = price_by_category.sort_values('mean', ascending=False)
                
                print(f"\n💰 Precio promedio por {col}:")
                top_5_prices = price_by_category.head(5)
                for category, row in top_5_prices.iterrows():
                    print(f"   - {category}: ${row['mean']:,.0f} (mediana: ${row['median']:,.0f}, n={row['count']})")
            
            categorical_stats[col] = {
                'n_categories': n_categories,
                'n_missing': n_missing,
                'value_counts': value_counts
            }
        
        # Visualización de variables categóricas
        n_categorical = len(self.categorical_columns)
        if n_categorical > 0:
            n_cols = 2
            n_rows = (n_categorical + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 6))
            if n_categorical == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(self.categorical_columns):
                if i < len(axes):
                    # Tomar top 10 categorías para visualización
                    top_categories = self.df[col].value_counts().head(10)
                    
                    if len(top_categories) > 0:
                        bars = axes[i].bar(top_categories.index.astype(str), top_categories.values, 
                                         color='lightcoral', alpha=0.7)
                        axes[i].set_title(f'Distribución de {col}', fontweight='bold')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frecuencia')
                        axes[i].tick_params(axis='x', rotation=45)
                        
                        # Añadir etiquetas en las barras
                        for bar in bars:
                            height = bar.get_height()
                            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                       f'{height}', ha='center', va='bottom', fontsize=8)
            
            # Ocultar ejes vacíos
            for j in range(len(self.categorical_columns), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        
        return categorical_stats
    
    def price_analysis_by_category(self):
        """Análisis detallado de precios por categorías"""
        logger.info("💰 Analizando precios por categorías...")
        
        if 'precio_venta' not in self.df.columns or not self.categorical_columns:
            return
        
        print("=" * 80)
        print("💰 ANÁLISIS DE PRECIOS POR CATEGORÍAS")
        print("=" * 80)
        
        for col in self.categorical_columns:
            if self.df[col].nunique() <= 15:  # Solo para variables con pocas categorías
                print(f"\n📊 PRECIOS POR {col.upper()}:")
                
                price_stats = self.df.groupby(col)['precio_venta'].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).round(0).sort_values('mean', ascending=False)
                
                print(price_stats)
                
                # Boxplot de precios por categoría
                plt.figure(figsize=(12, 6))
                order = price_stats.index.tolist()
                sns.boxplot(data=self.df, x=col, y='precio_venta', order=order)
                plt.title(f'Distribución de Precios por {col}', fontweight='bold')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
    
    def generate_complete_report(self, file_path):
        """
        Generar reporte completo de EDA
        
        Args:
            file_path (str): Ruta al archivo Excel
        """
        logger.info("🚀 INICIANDO REPORTE COMPLETO DE EDA")
        
        try:
            # 1. Cargar datos
            self.load_data(file_path)
            
            # 2. Resumen general
            self.dataset_overview()
            
            # 3. Análisis de valores faltantes
            self.missing_values_analysis()
            
            # 4. Estadísticas descriptivas numéricas
            self.numeric_descriptive_stats()
            
            # 5. Análisis de distribución y outliers
            self.numeric_distribution_analysis()
            
            # 6. Análisis de correlación
            self.correlation_analysis()
            
            # 7. Análisis categórico
            self.categorical_analysis()
            
            # 8. Análisis de precios por categoría
            self.price_analysis_by_category()
            
            logger.info("🎉 REPORTE DE EDA COMPLETADO EXITOSAMENTE")
            
        except Exception as e:
            logger.error(f"❌ Error en el reporte EDA: {e}")
            raise

# Función de conveniencia para uso rápido
def generate_eda_report(file_path):
    """
    Función simple para generar reporte EDA completo
    
    Args:
        file_path (str): Ruta al archivo Excel
    """
    eda = BogotaApartmentsEDA()
    eda.generate_complete_report(file_path)

if __name__ == "__main__":
    # Ejemplo de uso
    sample_file = "bogota_apartments.xlsx"
    try:
        generate_eda_report(sample_file)
    except Exception as e:
        print(f"❌ Error: {e}")