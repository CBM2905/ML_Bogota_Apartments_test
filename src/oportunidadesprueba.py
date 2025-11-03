"""
MÓDULO CORREGIDO - CON REVERSIÓN DE TRANSFORMACIÓN LOGARÍTMICA
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

class ProcessedDataTradingSignalGenerator:
    """
    Generador de señales de trading CORREGIDO - Revierte transformación logarítmica
    """
    
    def __init__(self, model_path, use_log_transformation=True):
        """
        Inicializar el generador de señales
        
        Args:
            model_path (str): Ruta al modelo XGBoost guardado
            use_log_transformation (bool): Si el modelo predice log(precio)
        """
        self.model_path = model_path
        self.model = None
        self.results_df = None
        self.signals_summary = None
        self.use_log_transformation = use_log_transformation
        
        # Cargar modelo
        self._load_model()

    def reverse_log_transformation(self, values):
        """
        Revertir transformación logarítmica: exp(valores) - 1 para log1p
        """
        if self.use_log_transformation:
            return np.expm1(values)  # Revierte log1p
        else:
            return values  # Si no hay transformación, devolver original

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

    def _load_model(self):
        """Cargar modelo entrenado"""
        print("🔄 Cargando modelo...")
        
        try:
            self.model = joblib.load(self.model_path)
            print(f"✅ Modelo cargado: {os.path.basename(self.model_path)}")
            print(f"📊 Tipo de modelo: {type(self.model).__name__}")
            print(f"🔁 Transformación logarítmica: {'SÍ' if self.use_log_transformation else 'NO'}")
                
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise

    def generate_signals_from_processed(self, X_processed, y, feature_names=None, 
                                      dates=None, threshold=0.05):
        """
        Generar señales de trading CORREGIDO - Con reversión de transformación
        """
        print("🎯 Generando señales de trading (CON REVERSIÓN LOGARÍTMICA)...")
        
        # Validar inputs
        self._validate_inputs(X_processed, y)
        
        # Realizar predicciones
        print("🔮 Realizando predicciones...")
        try:
            y_pred_log = self.model.predict(X_processed)
            print(f"✅ {len(y_pred_log)} predicciones realizadas")
            
            # REVERTIR TRANSFORMACIÓN LOGARÍTMICA
            print("🔄 Revirtiendo transformación logarítmica...")
            y_pred_original = self.reverse_log_transformation(y_pred_log)
            y_original = self.reverse_log_transformation(y)
            
            print(f"💰 Ejemplo de reversión:")
            print(f"   - Predicción (log): {y_pred_log[0]:.4f} → Precio: ${y_pred_original[0]:,.0f}")
            print(f"   - Actual (log): {y[0]:.4f} → Precio: ${y_original[0]:,.0f}")
            
        except Exception as e:
            print(f"❌ Error en las predicciones: {e}")
            raise
        
        # Crear DataFrame de resultados CON PRECIOS REALES
        self._create_results_dataframe(y_original, y_pred_original, dates, feature_names)
        
        # Generar señales basadas en el umbral
        self._apply_trading_signals(threshold)
        
        # Generar resumen de señales
        self._generate_signals_summary()
        
        print("✅ Señales de trading generadas exitosamente (con precios reales)")
        return self.results_df

    # El resto de los métodos permanecen iguales pero usarán precios reales
    def _create_results_dataframe(self, y, y_pred, dates, feature_names):
        """Crear DataFrame con resultados de predicciones EN PRECIOS REALES"""
        # Crear datos básicos CON PRECIOS REALES
        results_data = {
            'precio_actual': y,  # Ahora en pesos reales
            'prediccion': y_pred  # Ahora en pesos reales
        }
        
        # Añadir fechas si están disponibles
        if dates is not None:
            if hasattr(dates, 'values'):
                dates = dates.values
            results_data['fecha'] = dates
        else:
            results_data['fecha'] = pd.date_range(start='2023-01-01', periods=len(y), freq='D')
        
        # Añadir nombres de características si están disponibles
        if feature_names is not None and len(feature_names) <= 10:
            for i, feature in enumerate(feature_names):
                if i < X_processed.shape[1]:
                    results_data[f'feature_{feature}'] = X_processed[:, i]
        
        self.results_df = pd.DataFrame(results_data)
        self.results_df['fecha'] = pd.to_datetime(self.results_df['fecha'])

    # Los métodos _apply_trading_signals, _generate_signals_summary, etc. permanecen iguales
    # pero ahora trabajarán con precios reales en lugar de log-precios

    def _apply_trading_signals(self, threshold):
        """
        Aplicar lógica de señales de trading CON PRECIOS REALES
        """
        # Calcular diferencia porcentual entre predicción y precio actual (REALES)
        price_diff_pct = (self.results_df['prediccion'] - self.results_df['precio_actual']) 
        price_diff_pct = price_diff_pct / self.results_df['precio_actual']
        
        # Aplicar lógica de señales
        signals = []
        confidence = []
        
        for diff in price_diff_pct:
            if diff > threshold:
                signals.append('COMPRA')
                confidence.append('ALTA' if diff > threshold * 2 else 'MEDIA')
            elif diff < -threshold:
                signals.append('VENTA')
                confidence.append('ALTA' if diff < -threshold * 2 else 'MEDIA')
            else:
                signals.append('MANTENER')
                confidence.append('BAJA')
        
        self.results_df['señal'] = signals
        self.results_df['confianza'] = confidence
        self.results_df['diferencia_porcentual'] = price_diff_pct
        self.results_df['umbral_aplicado'] = threshold
        
        print(f"📊 Umbral aplicado: {threshold*100:.1f}%")
        print(f"💰 Precios en escala REAL (no logarítmica)")

    # Los métodos restantes (plot, save, etc.) permanecen iguales

# FUNCIONES CORREGIDAS PARA IDENTIFICAR PROPIEDADES ESPECÍFICAS

def generate_signals_with_property_details(model_path, original_dataframe, X_processed, y, 
                                         threshold=0.05, save_results=True, use_log_transformation=True):
    """
    VERSIÓN CORREGIDA - Con reversión de transformación logarítmica
    """
    print("🎯 GENERANDO SEÑALES CON PROPIEDADES ESPECÍFICAS Y PRECIOS REALES")
    print("=" * 70)
    
    try:
        # 1. Inicializar generador CORREGIDO
        generator = ProcessedDataTradingSignalGenerator(model_path, use_log_transformation)
        X_clean, y_clean = generator.check_and_clean_data(X_processed, y)
        
        # 2. Generar predicciones Y REVERTIR TRANSFORMACIÓN
        print("🔮 Realizando predicciones y revirtiendo transformación...")
        predictions_log = generator.model.predict(X_clean)
        
        # Revertir a precios reales
        if use_log_transformation:
            predictions_real = np.expm1(predictions_log)
            y_real = np.expm1(y_clean)
            print("✅ Transformación logarítmica revertida (log1p → expm1)")
        else:
            predictions_real = predictions_log
            y_real = y_clean
            print("✅ Sin transformación logarítmica - usando precios reales")
        
        # 3. Crear DataFrame COMBINADO con información original Y PRECIOS REALES
        results_with_properties = original_dataframe.copy()
        results_with_properties = results_with_properties.iloc[:len(predictions_real)].copy()
        
        # Añadir predicciones y cálculos EN PRECIOS REALES
        results_with_properties['prediccion_modelo'] = predictions_real
        results_with_properties['precio_actual_calculado'] = y_real
        
        # Calcular diferencia porcentual con valores REALES
        results_with_properties['diferencia_porcentual'] = (
            (results_with_properties['prediccion_modelo'] - results_with_properties['precio_actual_calculado']) / 
            results_with_properties['precio_actual_calculado'] * 100
        )
        
        # 4. Aplicar señales de trading
        conditions = [
            results_with_properties['diferencia_porcentual'] > threshold * 100,
            results_with_properties['diferencia_porcentual'] < -threshold * 100
        ]
        choices = ['COMPRA', 'VENTA']
        results_with_properties['señal'] = np.select(conditions, choices, default='MANTENER')
        
        # 5. Calcular confianza
        results_with_properties['confianza'] = 'BAJA'
        results_with_properties.loc[
            abs(results_with_properties['diferencia_porcentual']) > threshold * 200, 'confianza'
        ] = 'ALTA'
        results_with_properties.loc[
            (abs(results_with_properties['diferencia_porcentual']) > threshold * 100) & 
            (abs(results_with_properties['diferencia_porcentual']) <= threshold * 200), 'confianza'
        ] = 'MEDIA'
        
        # 6. FILTRAR SOLO LAS OPORTUNIDADES (COMPRA/VENTA)
        oportunidades_df = results_with_properties[
            results_with_properties['señal'].isin(['COMPRA', 'VENTA'])
        ].copy()
        
        # 7. Generar reportes específicos CON PRECIOS REALES
        _generate_detailed_opportunities_report(oportunidades_df)
        _plot_property_specific_analysis(oportunidades_df, results_with_properties)
        
        # 8. Guardar resultados
        if save_results:
            _save_property_opportunities(oportunidades_df, results_with_properties)
        
        print(f"\n🎉 GENERACIÓN DE SEÑALES CON PRECIOS REALES COMPLETADA")
        print(f"💡 Todas las cifras están en PESOS COLOMBIANOS reales")
        
        return oportunidades_df, results_with_properties
        
    except Exception as e:
        print(f"❌ Error generando señales con precios reales: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def _generate_detailed_opportunities_report(oportunidades_df):
    """Generar reporte detallado con propiedades específicas Y PRECIOS REALES"""
    print("\n" + "🔥" * 80)
    print("🔥 REPORTE DE OPORTUNIDADES INMOBILIARIAS - PRECIOS REALES (PESOS)")
    print("🔥" * 80)
    
    # Oportunidades de COMPRA
    compras = oportunidades_df[oportunidades_df['señal'] == 'COMPRA']
    if not compras.empty:
        print(f"\n🏆 TOP 10 PROPIEDADES PARA COMPRAR (Mayor Potencial):")
        top_compras = compras.nlargest(10, 'diferencia_porcentual')
        
        for i, (idx, propiedad) in enumerate(top_compras.iterrows(), 1):
            print(f"\n{i}. 🏠 {propiedad.get('barrio', 'N/A')} - {propiedad.get('direccion', 'N/A')}")
            print(f"   💰 PRECIO ACTUAL: ${propiedad['precio_actual_calculado']:,.0f}")
            print(f"   📈 VALOR REAL: ${propiedad['prediccion_modelo']:,.0f}")
            print(f"   🎯 OPORTUNIDAD: +{propiedad['diferencia_porcentual']:.1f}%")
            print(f"   📏 Área: {propiedad.get('area', 'N/A')}m² | 🏢 Estrato: {propiedad.get('estrato', 'N/A')}")
            print(f"   ✅ Confianza: {propiedad['confianza']}")
            
            # Calcular ganancia potencial en pesos
            ganancia_potencial = propiedad['prediccion_modelo'] - propiedad['precio_actual_calculado']
            print(f"   💸 GANANCIA POTENCIAL: ${ganancia_potencial:,.0f}")
    
    # Oportunidades de VENTA
    ventas = oportunidades_df[oportunidades_df['señal'] == 'VENTA']
    if not ventas.empty:
        print(f"\n💸 TOP 10 PROPIEDADES PARA VENDER (Sobrevaloradas):")
        top_ventas = ventas.nsmallest(10, 'diferencia_porcentual')
        
        for i, (idx, propiedad) in enumerate(top_ventas.iterrows(), 1):
            print(f"\n{i}. 🏠 {propiedad.get('barrio', 'N/A')} - {propiedad.get('direccion', 'N/A')}")
            print(f"   💰 PRECIO ACTUAL: ${propiedad['precio_actual_calculado']:,.0f}")
            print(f"   📉 VALOR REAL: ${propiedad['prediccion_modelo']:,.0f}")
            print(f"   ⚠️  SOBREPRECIO: {propiedad['diferencia_porcentual']:.1f}%")
            print(f"   📏 Área: {propiedad.get('area', 'N/A')}m² | 🏢 Estrato: {propiedad.get('estrato', 'N/A')}")
            print(f"   ✅ Confianza: {propiedad['confianza']}")
            
            # Calcular sobreprecio en pesos
            sobreprecio = propiedad['precio_actual_calculado'] - propiedad['prediccion_modelo']
            print(f"   💰 SOBREPRECIO ACTUAL: ${sobreprecio:,.0f}")

# Las funciones _plot_property_specific_analysis y _save_property_opportunities permanecen iguales
# pero ahora mostrarán precios reales

def _plot_property_specific_analysis(oportunidades_df, todas_propiedades):
    """Generar gráficos específicos de propiedades CON PRECIOS REALES"""
    print("\n📊 Generando gráficos de análisis con precios reales...")
    
    # 1. Top propiedades para comprar (Gráfico de barras)
    plt.figure(figsize=(14, 8))
    compras = oportunidades_df[oportunidades_df['señal'] == 'COMPRA']
    if not compras.empty:
        top_10_compras = compras.nlargest(10, 'diferencia_porcentual')
        
        # Crear etiquetas con información de propiedad
        labels = []
        for idx, prop in top_10_compras.iterrows():
            barrio = prop.get('barrio', 'N/A')[:15]
            precio_actual = prop['precio_actual_calculado'] / 1e6  # Convertir a millones para mejor visualización
            labels.append(f"{barrio}\n${precio_actual:.0f}M")
        
        bars = plt.barh(labels, top_10_compras['diferencia_porcentual'], color='green', alpha=0.7)
        plt.title('TOP 10 PROPIEDADES PARA COMPRAR - PRECIOS REALES', fontweight='bold')
        plt.xlabel('Potencial de Ganancia (%)')
        
        # Añadir etiquetas con precios reales
        for i, bar in enumerate(bars):
            precio_actual = top_10_compras.iloc[i]['precio_actual_calculado']
            precio_real = top_10_compras.iloc[i]['prediccion_modelo']
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f"Actual: ${precio_actual:,.0f}\nReal: ${precio_real:,.0f}", 
                    va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

def _save_property_opportunities(oportunidades_df, todas_propiedades):
    """Guardar oportunidades con información de propiedades Y PRECIOS REALES"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Guardar CSV con todas las oportunidades
    oportunidades_path = f"oportunidades_inmobiliarias_reales_{timestamp}.csv"
    
    # Seleccionar columnas relevantes para el CSV
    columnas_relevantes = [
        'direccion', 'barrio', 'localidad', 'precio_actual_calculado', 'prediccion_modelo',
        'diferencia_porcentual', 'señal', 'confianza', 'area', 'estrato',
        'habitaciones', 'banos', 'parqueaderos', 'administracion'
    ]
    
    # Filtrar columnas que existan
    columnas_disponibles = [col for col in columnas_relevantes if col in oportunidades_df.columns]
    
    # Crear columna de ganancia/sobreprecio en pesos
    if 'precio_actual_calculado' in oportunidades_df.columns and 'prediccion_modelo' in oportunidades_df.columns:
        oportunidades_df['diferencia_pesos'] = oportunidades_df['prediccion_modelo'] - oportunidades_df['precio_actual_calculado']
        columnas_disponibles.append('diferencia_pesos')
    
    oportunidades_df[columnas_disponibles].to_csv(oportunidades_path, index=False, encoding='utf-8')
    print(f"✅ Oportunidades con precios reales guardadas en: {oportunidades_path}")
    
    # 2. Guardar Excel con hojas separadas
    excel_path = f"reporte_oportunidades_reales_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Hoja 1: Todas las oportunidades
        oportunidades_df[columnas_disponibles].to_excel(writer, sheet_name='Oportunidades', index=False)
        
        # Hoja 2: Solo compras (ordenadas por potencial)
        compras_df = oportunidades_df[oportunidades_df['señal'] == 'COMPRA']
        if not compras_df.empty:
            compras_df[columnas_disponibles].sort_values('diferencia_porcentual', ascending=False).to_excel(
                writer, sheet_name='Mejores_Compras', index=False
            )
        
        # Hoja 3: Solo ventas (ordenadas por sobreprecio)
        ventas_df = oportunidades_df[oportunidades_df['señal'] == 'VENTA']
        if not ventas_df.empty:
            ventas_df[columnas_disponibles].sort_values('diferencia_porcentual', ascending=True).to_excel(
                writer, sheet_name='Mejores_Ventas', index=False
            )
    
    print(f"✅ Reporte Excel con precios reales guardado en: {excel_path}")
    print("📁 Hojas incluidas: Oportunidades, Mejores_Compras, Mejores_Ventas")
    print("💰 TODOS los precios están en PESOS COLOMBIANOS reales")

# Ejemplo de uso CORREGIDO
if __name__ == "__main__":
    # Configuración de ejemplo
    MODEL_PATH = "modelos_entrenados/XGBoost_20251103_165700.joblib"
    
    print("🔧 MÓDULO CORREGIDO - SEÑALES CON PRECIOS REALES")
    print("📝 Ejecuta generate_signals_with_property_details() para usar el módulo")
    
    # Ejemplo de uso CORREGIDO
    try:
        # Suponiendo que tienes tus datos
        oportunidades, todas_propiedades = generate_signals_with_property_details(
            model_path=MODEL_PATH,
            original_dataframe=df_original,  # Tu DataFrame original
            X_processed=X_processed,         # Características preprocesadas
            y=y,                            # Variable objetivo (posiblemente log-transformada)
            threshold=0.05,                 # 5% de diferencia
            use_log_transformation=True     # ¡IMPORTANTE! Especificar si hay transformación
        )
        
        if oportunidades is not None:
            print(f"\n📋 Resumen de oportunidades encontradas:")
            print(f"   • Compras recomendadas: {len(oportunidades[oportunidades['señal'] == 'COMPRA'])}")
            print(f"   • Ventas recomendadas: {len(oportunidades[oportunidades['señal'] == 'VENTA'])}")
            print(f"   • Todas las cifras en PESOS COLOMBIANOS reales")
            
    except Exception as e:
        print(f"❌ Error en el ejemplo: {e}")
        import traceback
        traceback.print_exc()