"""
MÓDULO DE ANÁLISIS EXPLORATORIO (EDA) PARA BOGOTÁ APARTMENTS
Análisis completo, automático y profesional del dataset inmobiliario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import warnings
from matplotlib.gridspec import GridSpec
import os

warnings.filterwarnings('ignore')

class BogotaApartmentsEDA:
    """
    Clase para análisis exploratorio completo de dataset de apartamentos en Bogotá
    """
    
    def __init__(self, file_path=None):
        """
        Inicializar el analizador EDA
        
        Args:
            file_path (str): Ruta al archivo Excel (opcional)
        """
        self.df = None
        self.file_path = file_path
        self.numeric_columns = []
        self.categorical_columns = []
        self.analysis_results = {}
        
        # Configuración de estilo
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3F7CAC']
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Cargar datos si se proporciona file_path
        if file_path:
            self.load_data(file_path)
    
    def load_data(self, file_path):
        """
        Cargar datos desde archivo Excel
        
        Args:
            file_path (str): Ruta al archivo Excel
        """
        self.logger.info(f"📥 Cargando datos desde: {file_path}")
        try:
            self.df = pd.read_excel(file_path)
            self.file_path = file_path
            
            # Definir columnas de interés
            self.original_columns = [
                'precio_venta', 'area', 'habitaciones', 'banos', 'estrato', 
                'parqueaderos', 'administracion', 'localidad', 'barrio', 'antiguedad',
                'latitud', 'longitud', 'tipo_propiedad', 'precio_arriendo'
            ]
            
            # Filtrar solo columnas que existan
            available_columns = [col for col in self.original_columns if col in self.df.columns]
            self.df = self.df[available_columns]
            
            # Clasificar columnas
            self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
            
            self.logger.info(f"✅ Datos cargados: {self.df.shape[0]:,} registros, {self.df.shape[1]} columnas")
            self.logger.info(f"📊 Numéricas: {len(self.numeric_columns)}, Categóricas: {len(self.categorical_columns)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando datos: {e}")
            raise
    
    def generate_complete_report(self, save_plots=False, plot_dir='eda_plots'):
        """
        Generar reporte EDA completo
        
        Args:
            save_plots (bool): Guardar gráficos en archivos
            plot_dir (str): Directorio para guardar gráficos
        """
        self.logger.info("🚀 INICIANDO REPORTE EDA COMPLETO")
        
        if self.df is None:
            self.logger.error("❌ No hay datos cargados. Use load_data() primero.")
            return
        
        # Crear directorio para gráficos si es necesario
        if save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        try:
            # 1. Resumen general
            self._print_dataset_overview()
            
            # 2. Análisis de valores faltantes
            missing_analysis = self._analyze_missing_values()
            
            # 3. Análisis numérico
            numeric_stats = self._analyze_numeric_variables(save_plots, plot_dir)
            
            # 4. Análisis categórico
            categorical_stats = self._analyze_categorical_variables(save_plots, plot_dir)
            
            # 5. Análisis de correlaciones
            correlation_analysis = self._analyze_correlations(save_plots, plot_dir)
            
            # 6. Análisis de relaciones con precio
            price_analysis = self._analyze_price_relationships(save_plots, plot_dir)
            
            # 7. Detección de problemas
            issues = self._detect_potential_issues(missing_analysis, numeric_stats)
            
            # 8. Recomendaciones
            self._provide_preprocessing_recommendations(issues, numeric_stats, missing_analysis)
            
            # 9. Resumen ejecutivo
            self._generate_executive_summary(missing_analysis, numeric_stats, issues)
            
            self.logger.info("🎉 REPORTE EDA COMPLETADO EXITOSAMENTE")
            
            return {
                'missing_analysis': missing_analysis,
                'numeric_stats': numeric_stats,
                'categorical_stats': categorical_stats,
                'correlation_analysis': correlation_analysis,
                'issues': issues
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error en el reporte EDA: {e}")
            raise
    
    def _print_dataset_overview(self):
        """Imprimir resumen general del dataset"""
        print("=" * 80)
        print("🏢 ANÁLISIS EXPLORATORIO - BOGOTÁ APARTMENTS")
        print("=" * 80)
        
        print(f"📊 DIMENSIONES: {self.df.shape[0]:,} registros × {self.df.shape[1]} columnas")
        print(f"🎯 VARIABLE OBJETIVO: precio_venta")
        print(f"💾 MEMORIA: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        print("\n📝 TIPOS DE DATOS:")
        type_counts = self.df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            print(f"   • {dtype}: {count} columnas")
        
        print("\n🔍 COLUMNAS DISPONIBLES:")
        print(f"   • Numéricas ({len(self.numeric_columns)}): {', '.join(self.numeric_columns)}")
        print(f"   • Categóricas ({len(self.categorical_columns)}): {', '.join(self.categorical_columns)}")
        print("=" * 80)
    
    def _analyze_missing_values(self):
        """Analizar valores faltantes"""
        self.logger.info("🔍 Analizando valores faltantes...")
        
        print("\n" + "🔍 ANÁLISIS DE VALORES FALTANTES")
        print("-" * 50)
        
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Valores_Faltantes': missing_data,
            'Porcentaje': missing_percent
        }).sort_values('Porcentaje', ascending=False)
        
        # Filtrar solo columnas con valores faltantes
        missing_df = missing_df[missing_df['Valores_Faltantes'] > 0]
        
        if len(missing_df) > 0:
            print("📋 COLUMNAS CON VALORES FALTANTES:")
            for col, row in missing_df.iterrows():
                print(f"   ⚠️  {col}: {row['Valores_Faltantes']} ({row['Porcentaje']:.1f}%)")
            
            # Visualización
            self._plot_missing_values(missing_df)
        else:
            print("✅ No hay valores faltantes en el dataset")
        
        return missing_df
    
    def _plot_missing_values(self, missing_df):
        """Visualizar valores faltantes"""
        plt.figure(figsize=(12, 6))
        
        # Tomar top 15 columnas con missing values
        plot_data = missing_df.head(15)
        
        bars = plt.bar(plot_data.index, plot_data['Porcentaje'], 
                      color=self.colors[0], alpha=0.7, edgecolor='black')
        
        plt.title('Porcentaje de Valores Faltantes por Columna', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Porcentaje Faltante (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Añadir etiquetas
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_numeric_variables(self, save_plots=False, plot_dir='eda_plots'):
        """Análisis exhaustivo de variables numéricas"""
        self.logger.info("📈 Analizando variables numéricas...")
        
        if not self.numeric_columns:
            self.logger.warning("⚠️ No hay variables numéricas para analizar")
            return {}
        
        print("\n" + "📊 ANÁLISIS DE VARIABLES NUMÉRICAS")
        print("-" * 50)
        
        # Estadísticas extendidas
        stats_df = self.df[self.numeric_columns].describe(percentiles=[.01, .25, .5, .75, .95, .99]).T
        
        # Calcular estadísticas adicionales
        stats_df['skewness'] = self.df[self.numeric_columns].skew()
        stats_df['kurtosis'] = self.df[self.numeric_columns].kurtosis()
        stats_df['cv'] = (stats_df['std'] / stats_df['mean']) * 100
        stats_df['iqr'] = stats_df['75%'] - stats_df['25%']
        
        # Detección de outliers
        outlier_stats = {}
        for col in self.numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_stats[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100,
                'bounds': (lower_bound, upper_bound)
            }
        
        stats_df['outliers_count'] = [outlier_stats[col]['count'] for col in self.numeric_columns]
        stats_df['outliers_percent'] = [outlier_stats[col]['percentage'] for col in self.numeric_columns]
        
        # Mostrar estadísticas formateadas
        display_stats = stats_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 
                                 'skewness', 'cv', 'outliers_percent']].round(3)
        print(display_stats)
        
        # Identificar variables problemáticas
        self._identify_problematic_numeric_variables(stats_df)
        
        # Visualizaciones
        self._plot_numeric_distributions(save_plots, plot_dir)
        
        return {
            'descriptive_stats': stats_df,
            'outlier_analysis': outlier_stats
        }
    
    def _identify_problematic_numeric_variables(self, stats_df):
        """Identificar variables numéricas problemáticas"""
        print("\n🚨 VARIABLES NUMÉRICAS PROBLEMÁTICAS:")
        
        high_skew = stats_df[abs(stats_df['skewness']) > 2]
        high_cv = stats_df[stats_df['cv'] > 100]
        high_outliers = stats_df[stats_df['outliers_percent'] > 5]
        
        if len(high_skew) > 0:
            print("   • Alta asimetría (|skew| > 2):")
            for col in high_skew.index:
                print(f"     - {col}: skewness = {high_skew.loc[col, 'skewness']:.2f}")
        
        if len(high_cv) > 0:
            print("   • Alta dispersión (CV > 100%):")
            for col in high_cv.index:
                print(f"     - {col}: CV = {high_cv.loc[col, 'cv']:.1f}%")
        
        if len(high_outliers) > 0:
            print("   • Muchos outliers (>5%):")
            for col in high_outliers.index:
                print(f"     - {col}: {high_outliers.loc[col, 'outliers_percent']:.1f}% outliers")
    
    def _plot_numeric_distributions(self, save_plots=False, plot_dir='eda_plots'):
        """Visualizar distribuciones numéricas"""
        key_numeric = ['precio_venta', 'area', 'habitaciones', 'banos', 'administracion']
        available_numeric = [col for col in key_numeric if col in self.numeric_columns]
        
        if not available_numeric:
            return
        
        n_cols = min(3, len(available_numeric))
        n_rows = (len(available_numeric) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(18, n_rows * 8))
        fig.suptitle('DISTRIBUCIÓN DE VARIABLES NUMÉRICAS CLAVE', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(available_numeric):
            row_hist = (i // n_cols) * 2
            row_box = row_hist + 1
            col_pos = i % n_cols
            
            # Histograma
            if n_rows == 1:
                ax_hist = axes[col_pos] if n_cols > 1 else axes
                ax_box = axes[col_pos + n_cols] if n_cols > 1 else axes
            else:
                ax_hist = axes[row_hist, col_pos]
                ax_box = axes[row_box, col_pos]
            
            # Histograma con KDE
            self.df[col].hist(bins=30, ax=ax_hist, color=self.colors[0], 
                             alpha=0.7, edgecolor='black')
            ax_hist.set_title(f'Distribución de {col}', fontweight='bold')
            ax_hist.set_xlabel(col)
            ax_hist.set_ylabel('Frecuencia')
            
            # Añadir estadísticas
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                           label=f'Media: {mean_val:,.0f}')
            ax_hist.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                           label=f'Mediana: {median_val:,.0f}')
            ax_hist.legend()
            
            # Boxplot
            self.df.boxplot(column=col, ax=ax_box, color=self.colors[1])
            ax_box.set_title(f'Boxplot de {col}', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{plot_dir}/numeric_distributions.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _analyze_categorical_variables(self, save_plots=False, plot_dir='eda_plots'):
        """Análisis de variables categóricas"""
        self.logger.info("📊 Analizando variables categóricas...")
        
        if not self.categorical_columns:
            self.logger.warning("⚠️ No hay variables categóricas para analizar")
            return {}
        
        print("\n" + "📝 ANÁLISIS DE VARIABLES CATEGÓRICAS")
        print("-" * 50)
        
        categorical_stats = {}
        
        for col in self.categorical_columns:
            print(f"\n📌 {col.upper()}:")
            
            value_counts = self.df[col].value_counts()
            n_categories = self.df[col].nunique()
            n_missing = self.df[col].isnull().sum()
            
            print(f"   • Categorías únicas: {n_categories}")
            print(f"   • Valores faltantes: {n_missing}")
            print(f"   • Top 5 categorías:")
            
            top_5 = value_counts.head(5)
            for category, count in top_5.items():
                percentage = (count / len(self.df)) * 100
                print(f"     - {category}: {count} ({percentage:.1f}%)")
            
            # Precio promedio por categoría si existe precio_venta
            if 'precio_venta' in self.df.columns:
                price_stats = self.df.groupby(col)['precio_venta'].agg(['mean', 'count']).round(0)
                top_prices = price_stats.nlargest(3, 'mean')
                
                if len(top_prices) > 0:
                    print(f"   💰 Top 3 categorías por precio:")
                    for category, row in top_prices.iterrows():
                        print(f"     - {category}: ${row['mean']:,.0f} (n={row['count']})")
            
            categorical_stats[col] = {
                'n_categories': n_categories,
                'n_missing': n_missing,
                'value_counts': value_counts,
                'top_categories': top_5
            }
        
        # Visualizaciones
        self._plot_categorical_distributions(save_plots, plot_dir)
        
        return categorical_stats
    
    def _plot_categorical_distributions(self, save_plots=False, plot_dir='eda_plots'):
        """Visualizar distribuciones categóricas"""
        key_categorical = ['tipo_propiedad', 'estrato', 'antiguedad', 'localidad']
        available_categorical = [col for col in key_categorical if col in self.categorical_columns]
        
        if not available_categorical:
            return
        
        n_cols = 2
        n_rows = (len(available_categorical) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
        fig.suptitle('DISTRIBUCIÓN DE VARIABLES CATEGÓRICAS CLAVE', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(available_categorical):
            row = i // n_cols
            col_pos = i % n_cols
            
            if n_rows == 1:
                ax = axes[col_pos]
            else:
                ax = axes[row, col_pos]
            
            # Tomar top 10 categorías
            top_categories = self.df[col].value_counts().head(10)
            
            bars = ax.bar(top_categories.index.astype(str), top_categories.values, 
                         color=self.colors[i % len(self.colors)], alpha=0.7, edgecolor='black')
            
            ax.set_title(f'Distribución de {col}', fontweight='bold', pad=20)
            ax.set_xlabel(col)
            ax.set_ylabel('Frecuencia')
            ax.tick_params(axis='x', rotation=45)
            
            # Añadir etiquetas
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:,}', ha='center', va='bottom', fontsize=9)
        
        # Ocultar ejes vacíos
        for i in range(len(available_categorical), n_rows * n_cols):
            row = i // n_cols
            col_pos = i % n_cols
            if n_rows == 1:
                axes[col_pos].set_visible(False)
            else:
                axes[row, col_pos].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{plot_dir}/categorical_distributions.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _analyze_correlations(self, save_plots=False, plot_dir='eda_plots'):
        """Análisis de correlaciones entre variables numéricas"""
        self.logger.info("🔗 Analizando correlaciones...")
        
        if len(self.numeric_columns) < 2:
            self.logger.warning("⚠️ No hay suficientes variables numéricas para análisis de correlación")
            return {}
        
        print("\n" + "🔗 ANÁLISIS DE CORRELACIONES")
        print("-" * 50)
        
        # Matriz de correlación
        correlation_matrix = self.df[self.numeric_columns].corr()
        
        print("📊 MATRIZ DE CORRELACIÓN (Pearson):")
        print(correlation_matrix.round(3))
        
        # Correlaciones fuertes con precio_venta
        if 'precio_venta' in correlation_matrix.columns:
            price_correlations = correlation_matrix['precio_venta'].sort_values(ascending=False)
            
            print("\n💪 CORRELACIONES CON PRECIO_VENTA:")
            for var, corr in price_correlations.items():
                if var != 'precio_venta':
                    strength = "FUERTE ↗" if corr > 0.7 else "MODERADA →" if corr > 0.3 else "DÉBIL ↘"
                    print(f"   • {var}: {corr:.3f} ({strength})")
        
        # Visualización
        self._plot_correlation_heatmap(correlation_matrix, save_plots, plot_dir)
        
        # Detectar correlaciones altas
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print("\n🚨 CORRELACIONES MUY ALTAS (>0.8):")
            for var1, var2, corr in high_corr_pairs:
                print(f"   ⚠️  {var1} ↔ {var2}: r = {corr:.3f}")
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_correlation_pairs': high_corr_pairs
        }
    
    def _plot_correlation_heatmap(self, corr_matrix, save_plots=False, plot_dir='eda_plots'):
        """Visualizar matriz de correlación"""
        plt.figure(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': .8},
                   linewidths=0.5)
        
        plt.title('MATRIZ DE CORRELACIÓN - VARIABLES NUMÉRICAS', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{plot_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _analyze_price_relationships(self, save_plots=False, plot_dir='eda_plots'):
        """Analizar relaciones con precio_venta"""
        self.logger.info("💰 Analizando relaciones con precio...")
        
        if 'precio_venta' not in self.df.columns:
            self.logger.warning("⚠️ Variable precio_venta no encontrada")
            return {}
        
        print("\n" + "💰 RELACIONES CON PRECIO DE VENTA")
        print("-" * 50)
        
        # Variables para análisis de relaciones
        relationship_vars = ['area', 'habitaciones', 'banos', 'administracion']
        available_relationship_vars = [col for col in relationship_vars if col in self.numeric_columns]
        
        if not available_relationship_vars:
            return {}
        
        n_cols = 2
        n_rows = (len(available_relationship_vars) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 6))
        fig.suptitle('RELACIONES CON PRECIO DE VENTA', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        correlation_results = {}
        
        for i, var in enumerate(available_relationship_vars):
            row = i // n_cols
            col_pos = i % n_cols
            
            if n_rows == 1:
                ax = axes[col_pos]
            else:
                ax = axes[row, col_pos]
            
            # Scatter plot
            ax.scatter(self.df[var], self.df['precio_venta'], 
                      alpha=0.6, color=self.colors[2], s=50)
            ax.set_xlabel(var)
            ax.set_ylabel('Precio Venta')
            ax.set_title(f'{var} vs Precio', fontweight='bold')
            
            # Calcular correlación
            valid_data = self.df[[var, 'precio_venta']].dropna()
            if len(valid_data) > 1:
                corr = valid_data[var].corr(valid_data['precio_venta'])
                correlation_results[var] = corr
                
                # Añadir línea de tendencia
                z = np.polyfit(valid_data[var], valid_data['precio_venta'], 1)
                p = np.poly1d(z)
                ax.plot(valid_data[var], p(valid_data[var]), "r--", alpha=0.8, linewidth=2)
                
                # Añadir texto de correlación
                ax.text(0.05, 0.95, f'Correlación: {corr:.3f}', 
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Ocultar ejes vacíos
        for i in range(len(available_relationship_vars), n_rows * n_cols):
            row = i // n_cols
            col_pos = i % n_cols
            if n_rows == 1:
                axes[col_pos].set_visible(False)
            else:
                axes[row, col_pos].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{plot_dir}/price_relationships.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return correlation_results
    
    def _detect_potential_issues(self, missing_analysis, numeric_stats):
        """Detección sistemática de problemas potenciales"""
        self.logger.info("🚨 Detectando problemas potenciales...")
        
        print("\n" + "🚨 DETECCIÓN DE PROBLEMAS POTENCIALES")
        print("-" * 50)
        
        issues = {
            'missing_values': [],
            'outliers': [],
            'skewness': [],
            'high_correlation': [],
            'categorical_issues': []
        }
        
        # 1. Valores faltantes críticos
        if len(missing_analysis) > 0:
            high_missing = missing_analysis[missing_analysis['Porcentaje'] > 30]
            if len(high_missing) > 0:
                issues['missing_values'].append("Columnas con >30% valores faltantes:")
                for col in high_missing.index:
                    issues['missing_values'].append(f"  - {col}: {high_missing.loc[col, 'Porcentaje']:.1f}%")
        
        # 2. Outliers extremos
        if hasattr(numeric_stats, 'index'):
            high_outliers = numeric_stats[numeric_stats['outliers_percent'] > 10]
            if len(high_outliers) > 0:
                issues['outliers'].append("Variables con >10% outliers:")
                for col in high_outliers.index:
                    issues['outliers'].append(f"  - {col}: {high_outliers.loc[col, 'outliers_percent']:.1f}%")
        
        # 3. Distribuciones sesgadas
        if hasattr(numeric_stats, 'index'):
            high_skew = numeric_stats[abs(numeric_stats['skewness']) > 3]
            if len(high_skew) > 0:
                issues['skewness'].append("Variables muy sesgadas (|skew| > 3):")
                for col in high_skew.index:
                    issues['skewness'].append(f"  - {col}: skewness = {high_skew.loc[col, 'skewness']:.2f}")
        
        # 4. Problemas categóricos
        for col in self.categorical_columns:
            n_categories = self.df[col].nunique()
            if n_categories > 50:
                issues['categorical_issues'].append(f"  - {col}: {n_categories} categorías (demasiadas)")
        
        # Mostrar problemas detectados
        for category, problem_list in issues.items():
            if problem_list:
                print(f"\n🔴 {category.upper().replace('_', ' ')}:")
                for problem in problem_list:
                    print(f"   {problem}")
        
        if all(len(v) == 0 for v in issues.values()):
            print("✅ No se detectaron problemas críticos")
        
        return issues
    
    def _provide_preprocessing_recommendations(self, issues, numeric_stats, missing_analysis):
        """Proporcionar recomendaciones de preprocesamiento"""
        self.logger.info("🎯 Generando recomendaciones...")
        
        print("\n" + "🎯 RECOMENDACIONES DE PREPROCESAMIENTO")
        print("-" * 50)
        
        print("1️⃣ MANEJO DE VALORES FALTANTES:")
        if issues['missing_values']:
            high_missing = missing_analysis[missing_analysis['Porcentaje'] > 50]
            if len(high_missing) > 0:
                print("   🗑️  ELIMINAR columnas con >50% faltantes:")
                for col in high_missing.index:
                    print(f"     • {col}")
            
            moderate_missing = missing_analysis[(missing_analysis['Porcentaje'] > 5) & 
                                              (missing_analysis['Porcentaje'] <= 50)]
            if len(moderate_missing) > 0:
                print("   🔧 IMPUTAR columnas con 5-50% faltantes:")
                for col in moderate_missing.index:
                    if col in self.numeric_columns:
                        print(f"     • {col}: Imputar con mediana")
                    else:
                        print(f"     • {col}: Imputar con moda o 'DESCONOCIDO'")
        else:
            print("   ✅ No se requieren acciones para valores faltantes")
        
        print("\n2️⃣ TRANSFORMACIÓN DE VARIABLES:")
        if issues['skewness'] and hasattr(numeric_stats, 'index'):
            skewed_vars = numeric_stats[abs(numeric_stats['skewness']) > 2].index
            if len(skewed_vars) > 0:
                print("   📈 APLICAR transformación logarítmica a:")
                for var in skewed_vars:
                    print(f"     • {var}")
        
        if issues['outliers'] and hasattr(numeric_stats, 'index'):
            outlier_vars = numeric_stats[numeric_stats['outliers_percent'] > 5].index
            if len(outlier_vars) > 0:
                print("   📊 APLICAR winsorization (1%-99%) a:")
                for var in outlier_vars:
                    print(f"     • {var}")
        
        print("\n3️⃣ CODIFICACIÓN DE VARIABLES CATEGÓRICAS:")
        for col in self.categorical_columns:
            n_categories = self.df[col].nunique()
            if n_categories <= 10:
                print(f"   🔤 {col}: One-Hot Encoding ({n_categories} categorías)")
            elif n_categories <= 20:
                print(f"   🔤 {col}: Target Encoding ({n_categories} categorías)")
            else:
                print(f"   🔤 {col}: Agrupar + Frequency Encoding ({n_categories} categorías)")
        
        print("\n4️⃣ FILTRADO Y LIMPIEZA:")
        print("   🎯 Aplicar filtros básicos:")
        print("     • precio_venta > 0")
        print("     • area > 0")
        print("     • Coordenadas dentro de Bogotá")
        print("     • Estrato entre 1 y 6")
        
        print("\n5️⃣ INGENIERÍA DE CARACTERÍSTICAS:")
        print("   🛠️  Crear nuevas variables:")
        print("     • precio_m2 = precio_venta / area")
        print("     • amenities_score = suma de amenities")
        print("     • ratios: banos_por_area, habitaciones_por_area")
    
    def _generate_executive_summary(self, missing_analysis, numeric_stats, issues):
        """Generar resumen ejecutivo final"""
        print("\n" + "⭐" * 50)
        print("⭐ RESUMEN EJECUTIVO - DIAGNÓSTICO FINAL")
        print("⭐" * 50)
        
        # Calcular métricas clave
        total_issues = sum(len(v) for v in issues.values())
        data_quality_score = max(0, 10 - total_issues * 0.5)
        
        print(f"\n📊 MÉTRICAS CLAVE:")
        print(f"   • Registros totales: {self.df.shape[0]:,}")
        print(f"   • Variables analizadas: {self.df.shape[1]}")
        print(f"   • Calidad de datos: {data_quality_score:.1f}/10")
        
        if 'precio_venta' in self.df.columns:
            price_stats = self.df['precio_venta'].describe()
            print(f"   • Rango de precios: ${price_stats['min']:,.0f} - ${price_stats['max']:,.0f}")
        
        print(f"\n🚨 PROBLEMAS IDENTIFICADOS: {total_issues}")
        for category, problem_list in issues.items():
            if problem_list:
                print(f"   • {category}: {len(problem_list)}")
        
        print(f"\n💡 RECOMENDACIONES PRIORITARIAS:")
        print("   1. Limpieza de outliers en variables clave")
        print("   2. Transformación de variables sesgadas")
        print("   3. Imputación inteligente de valores faltantes")
        print("   4. Codificación apropiada de categóricas")
        
        print(f"\n📈 PREPARACIÓN PARA MODELADO:")
        if data_quality_score >= 7:
            print("   ✅ CALIDAD ALTA: Listo para preprocesamiento estándar")
        elif data_quality_score >= 5:
            print("   ⚠️  CALIDAD MEDIA: Requiere preprocesamiento moderado")
        else:
            print("   🔴 CALIDAD BAJA: Requiere limpieza extensiva")
        
        print(f"\n🎯 PRÓXIMOS PASOS:")
        print("   1. Implementar pipeline de preprocesamiento")
        print("   2. Validar calidad después de limpieza")
        print("   3. Realizar feature engineering")
        print("   4. Entrenar modelos baseline")


# Función de conveniencia para uso rápido
def run_complete_eda(file_path, save_plots=False):
    """
    Ejecutar análisis EDA completo con una sola función
    
    Args:
        file_path (str): Ruta al archivo Excel
        save_plots (bool): Guardar gráficos en archivos
    
    Returns:
        dict: Resultados del análisis
    """
    eda = BogotaApartmentsEDA(file_path)
    return eda.generate_complete_report(save_plots=save_plots)


if __name__ == "__main__":
    # Ejemplo de uso
    sample_file = "bogota_apartments.xlsx"
    
    try:
        # Uso simple
        results = run_complete_eda(sample_file, save_plots=True)
        print("✅ Análisis EDA completado exitosamente")
        
    except Exception as e:
        print(f"❌ Error en el análisis EDA: {e}")