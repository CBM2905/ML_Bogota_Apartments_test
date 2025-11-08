"""
MÓDULO UNIFICADO DE OPORTUNIDADES INMOBILIARIAS - INTEGRADO CON MODELADO UNIFICADO
Genera señales de compra/venta basado en predicciones del modelo unificado
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
import traceback

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class UnifiedTradingSignalGenerator:
    """
    Generador de señales de trading integrado con el módulo unificado de modelado
    """
    
    def __init__(self, model_trainer=None, model_path=None, use_log_transformation=True):
        """
        Inicializar el generador de señales
        
        Args:
            model_trainer: Instancia de UnifiedModelTrainer ya entrenada
            model_path: Ruta al modelo guardado (alternativa)
            use_log_transformation: Si el modelo predice log(precio)
        """
        self.model_trainer = model_trainer
        self.model_path = model_path
        self.use_log_transformation = use_log_transformation
        self.model = None
        self.results_df = None
        self.opportunities_df = None
        self.signals_summary = None
        
        # Cargar modelo
        self._load_model()

    def _load_model(self):
        """Cargar modelo desde trainer o archivo"""
        print("🔄 Cargando modelo para señales...")
        
        try:
            if self.model_trainer is not None and self.model_trainer.best_model is not None:
                self.model = self.model_trainer.best_model
                model_name = self.model_trainer.best_model_name
                print(f"✅ Modelo cargado desde trainer: {model_name}")
            elif self.model_path is not None:
                self.model = joblib.load(self.model_path)
                print(f"✅ Modelo cargado desde archivo: {os.path.basename(self.model_path)}")
            else:
                raise ValueError("❌ No se proporcionó modelo trainer ni ruta de modelo")
                
            print(f"📊 Tipo de modelo: {type(self.model).__name__}")
            print(f"🔁 Transformación logarítmica: {'SÍ' if self.use_log_transformation else 'NO'}")
                
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise

    def reverse_log_transformation(self, values):
        """
        Revertir transformación logarítmica de forma segura
        """
        if self.use_log_transformation:
            # Asegurar que no hay valores extremos que causen overflow
            values = np.clip(values, -100, 100)
            return np.expm1(values)
        else:
            return values

    def prepare_features(self, X, feature_names=None):
        """
        Preparar características para predicción
        """
        print("🔧 Preparando características...")
        
        # Convertir a numpy si es DataFrame
        if hasattr(X, 'values'):
            X = X.values
        
        # Manejar valores faltantes
        nan_mask = np.isnan(X)
        if nan_mask.any():
            print(f"⚠️  Encontrados {nan_mask.sum()} valores NaN, imputando...")
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Verificar valores infinitos
        inf_mask = ~np.isfinite(X)
        if inf_mask.any():
            print(f"⚠️  Encontrados {inf_mask.sum()} valores infinitos, corrigiendo...")
            X = np.where(inf_mask, np.nan, X)
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        print(f"✅ Características preparadas: {X.shape}")
        return X

    def generate_trading_signals(self, X_processed, y_true, feature_names=None, 
                               property_data=None, threshold=0.05, min_confidence='MEDIA'):
        """
        Generar señales de trading con reversión logarítmica
        """
        print("🎯 GENERANDO SEÑALES DE TRADING UNIFICADAS...")
        print("=" * 60)
        
        try:
            # 1. Preparar datos
            X_clean = self.prepare_features(X_processed, feature_names)
            
            # 2. Realizar predicciones
            print("🔮 Realizando predicciones...")
            y_pred_log = self.model.predict(X_clean)
            
            # 3. REVERTIR TRANSFORMACIÓN LOGARÍTMICA
            print("🔄 Revirtiendo transformación logarítmica...")
            y_pred_real = self.reverse_log_transformation(y_pred_log)
            y_true_real = self.reverse_log_transformation(y_true)
            
            # 4. Crear DataFrame de resultados
            results_data = {
                'precio_actual_real': y_true_real,
                'prediccion_real': y_pred_real
            }
            
            # Añadir información de propiedades si está disponible
            if property_data is not None:
                property_cols = ['direccion', 'barrio', 'localidad', 'area', 'estrato', 
                               'habitaciones', 'banos', 'parqueaderos', 'administracion']
                available_cols = [col for col in property_cols if col in property_data.columns]
                
                for col in available_cols:
                    results_data[col] = property_data[col].values[:len(y_true_real)]
            
            self.results_df = pd.DataFrame(results_data)
            
            # 5. Calcular diferencias y generar señales
            self._calculate_trading_signals(threshold, min_confidence)
            
            # 6. Filtrar oportunidades
            self._filter_opportunities()
            
            # 7. Generar reportes
            self._generate_comprehensive_report()
            
            print("✅ Señales de trading generadas exitosamente")
            return self.opportunities_df, self.results_df
            
        except Exception as e:
            print(f"❌ Error generando señales: {e}")
            traceback.print_exc()
            return None, None

    def _calculate_trading_signals(self, threshold, min_confidence):
        """Calcular señales de trading basadas en umbrales"""
        print("📊 Calculando señales de trading...")
        
        # Calcular diferencia porcentual
        self.results_df['diferencia_porcentual'] = (
            (self.results_df['prediccion_real'] - self.results_df['precio_actual_real']) / 
            self.results_df['precio_actual_real'] * 100
        )
        
        # Calcular diferencia absoluta en pesos
        self.results_df['diferencia_pesos'] = (
            self.results_df['prediccion_real'] - self.results_df['precio_actual_real']
        )
        
        # Aplicar lógica de señales
        conditions = [
            self.results_df['diferencia_porcentual'] > threshold * 100,
            self.results_df['diferencia_porcentual'] < -threshold * 100
        ]
        choices = ['COMPRA', 'VENTA']
        self.results_df['señal'] = np.select(conditions, choices, default='MANTENER')
        
        # Calcular confianza
        self.results_df['confianza'] = 'BAJA'
        high_cond = abs(self.results_df['diferencia_porcentual']) > threshold * 200
        medium_cond = (abs(self.results_df['diferencia_porcentual']) > threshold * 100) & \
                     (abs(self.results_df['diferencia_porcentual']) <= threshold * 200)
        
        self.results_df.loc[medium_cond, 'confianza'] = 'MEDIA'
        self.results_df.loc[high_cond, 'confianza'] = 'ALTA'
        
        print(f"📈 Umbral aplicado: {threshold*100:.1f}%")
        print(f"💰 Todas las cifras en PESOS COLOMBIANOS reales")

    def _filter_opportunities(self):
        """Filtrar solo las oportunidades de trading"""
        print("🎯 Filtrando oportunidades...")
        
        # Filtrar por señal y confianza
        signal_mask = self.results_df['señal'].isin(['COMPRA', 'VENTA'])
        self.opportunities_df = self.results_df[signal_mask].copy()
        
        # Ordenar por potencial
        self.opportunities_df = self.opportunities_df.sort_values(
            'diferencia_porcentual', 
            ascending=False if self.opportunities_df['señal'].iloc[0] == 'COMPRA' else True
        )
        
        print(f"✅ Encontradas {len(self.opportunities_df)} oportunidades")

    def _generate_comprehensive_report(self):
        """Generar reporte comprehensivo de oportunidades"""
        print("\n📈 GENERANDO REPORTE COMPLETO...")
        
        # 1. Estadísticas generales
        self._print_summary_statistics()
        
        # 2. Reporte detallado de oportunidades
        self._print_detailed_opportunities()
        
        # 3. Gráficos de análisis
        self._plot_opportunity_analysis()
        
        # 4. Resumen ejecutivo
        self._print_executive_summary()

    def _print_summary_statistics(self):
        """Imprimir estadísticas generales"""
        print("\n📊 ESTADÍSTICAS GENERALES")
        print("=" * 50)
        
        total_propiedades = len(self.results_df)
        oportunidades = len(self.opportunities_df)
        compras = len(self.opportunities_df[self.opportunities_df['señal'] == 'COMPRA'])
        ventas = len(self.opportunities_df[self.opportunities_df['señal'] == 'VENTA'])
        
        print(f"🏠 Total propiedades analizadas: {total_propiedades}")
        print(f"🎯 Oportunidades identificadas: {oportunidades} ({oportunidades/total_propiedades*100:.1f}%)")
        print(f"🛒 Compras recomendadas: {compras}")
        print(f"💰 Ventas recomendadas: {ventas}")
        
        if compras > 0:
            avg_gain = self.opportunities_df[self.opportunities_df['señal'] == 'COMPRA']['diferencia_porcentual'].mean()
            max_gain = self.opportunities_df[self.opportunities_df['señal'] == 'COMPRA']['diferencia_porcentual'].max()
            print(f"📈 Ganancia promedio en compras: +{avg_gain:.1f}%")
            print(f"🚀 Máxima ganancia potencial: +{max_gain:.1f}%")
        
        if ventas > 0:
            avg_overprice = abs(self.opportunities_df[self.opportunities_df['señal'] == 'VENTA']['diferencia_porcentual'].mean())
            max_overprice = abs(self.opportunities_df[self.opportunities_df['señal'] == 'VENTA']['diferencia_porcentual'].min())
            print(f"📉 Sobreprecio promedio en ventas: {avg_overprice:.1f}%")
            print(f"⚠️  Máximo sobreprecio: {max_overprice:.1f}%")

    def _print_detailed_opportunities(self):
        """Imprimir reporte detallado de oportunidades"""
        print("\n🔥 OPORTUNIDADES DETALLADAS")
        print("=" * 60)
        
        # Oportunidades de COMPRA
        compras = self.opportunities_df[self.opportunities_df['señal'] == 'COMPRA']
        if not compras.empty:
            print(f"\n🏆 TOP 5 PROPIEDADES PARA COMPRAR:")
            top_compras = compras.nlargest(5, 'diferencia_porcentual')
            
            for i, (idx, prop) in enumerate(top_compras.iterrows(), 1):
                self._print_property_details(i, prop, 'COMPRA')
        
        # Oportunidades de VENTA
        ventas = self.opportunities_df[self.opportunities_df['señal'] == 'VENTA']
        if not ventas.empty:
            print(f"\n💸 TOP 5 PROPIEDADES PARA VENDER:")
            top_ventas = ventas.nsmallest(5, 'diferencia_porcentual')
            
            for i, (idx, prop) in enumerate(top_ventas.iterrows(), 1):
                self._print_property_details(i, prop, 'VENTA')

    def _print_property_details(self, index, propiedad, tipo):
        """Imprimir detalles de una propiedad"""
        barrio = propiedad.get('barrio', 'N/A')
        direccion = propiedad.get('direccion', 'N/A')
        precio_actual = propiedad['precio_actual_real']
        precio_predicho = propiedad['prediccion_real']
        diferencia_pct = propiedad['diferencia_porcentual']
        diferencia_pesos = propiedad['diferencia_pesos']
        confianza = propiedad['confianza']
        
        print(f"\n{index}. 🏠 {barrio} - {direccion}")
        print(f"   💰 PRECIO ACTUAL: ${precio_actual:,.0f}")
        print(f"   📊 VALOR PREDICHO: ${precio_predicho:,.0f}")
        
        if tipo == 'COMPRA':
            print(f"   🎯 OPORTUNIDAD: +{diferencia_pct:.1f}%")
            print(f"   💸 GANANCIA POTENCIAL: ${diferencia_pesos:,.0f}")
        else:
            print(f"   ⚠️  SOBREPRECIO: {diferencia_pct:.1f}%")
            print(f"   💰 SOBREPRECIO ACTUAL: ${abs(diferencia_pesos):,.0f}")
        
        # Información adicional de la propiedad
        if 'area' in propiedad:
            print(f"   📏 Área: {propiedad['area']}m²")
        if 'estrato' in propiedad:
            print(f"   🏢 Estrato: {propiedad['estrato']}")
        if 'habitaciones' in propiedad:
            print(f"   🛏️ Habitaciones: {propiedad['habitaciones']}")
        
        print(f"   ✅ Confianza: {confianza}")

    def _plot_opportunity_analysis(self):
        """Generar gráficos de análisis de oportunidades"""
        print("\n📊 Generando gráficos de análisis...")
        
        if self.opportunities_df.empty:
            print("⚠️  No hay oportunidades para graficar")
            return
        
        # Crear figura con múltiples subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribución de oportunidades por tipo
        signal_counts = self.opportunities_df['señal'].value_counts()
        colors = ['green' if x == 'COMPRA' else 'red' for x in signal_counts.index]
        ax1.bar(signal_counts.index, signal_counts.values, color=colors, alpha=0.7)
        ax1.set_title('Distribución de Oportunidades por Tipo', fontweight='bold')
        ax1.set_ylabel('Número de Propiedades')
        
        # 2. Top oportunidades de compra
        compras = self.opportunities_df[self.opportunities_df['señal'] == 'COMPRA']
        if not compras.empty:
            top_compras = compras.nlargest(8, 'diferencia_porcentual')
            bars = ax2.barh(range(len(top_compras)), top_compras['diferencia_porcentual'], color='green', alpha=0.7)
            ax2.set_yticks(range(len(top_compras)))
            ax2.set_yticklabels([f"Prop {i+1}" for i in range(len(top_compras))])
            ax2.set_title('Top Oportunidades de COMPRA', fontweight='bold')
            ax2.set_xlabel('Ganancia Potencial (%)')
            
            # Añadir valores en las barras
            for i, bar in enumerate(bars):
                ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'+{bar.get_width():.1f}%', va='center', fontweight='bold')
        
        # 3. Distribución de confianza
        confidence_counts = self.opportunities_df['confianza'].value_counts()
        ax3.pie(confidence_counts.values, labels=confidence_counts.index, autopct='%1.1f%%', 
               colors=['lightcoral', 'gold', 'lightgreen'])
        ax3.set_title('Distribución por Nivel de Confianza', fontweight='bold')
        
        # 4. Oportunidades por estrato (si disponible)
        if 'estrato' in self.opportunities_df.columns:
            estrato_opps = self.opportunities_df.groupby('estrato').size()
            ax4.bar(estrato_opps.index.astype(str), estrato_opps.values, alpha=0.7, color='skyblue')
            ax4.set_title('Oportunidades por Estrato', fontweight='bold')
            ax4.set_xlabel('Estrato')
            ax4.set_ylabel('Número de Oportunidades')
        
        plt.tight_layout()
        plt.show()

    def _print_executive_summary(self):
        """Imprimir resumen ejecutivo"""
        print("\n" + "⭐" * 60)
        print("⭐ RESUMEN EJECUTIVO - OPORTUNIDADES INMOBILIARIAS")
        print("⭐" * 60)
        
        compras = self.opportunities_df[self.opportunities_df['señal'] == 'COMPRA']
        ventas = self.opportunities_df[self.opportunities_df['señal'] == 'VENTA']
        
        print(f"\n📈 OPORTUNIDADES DE COMPRA: {len(compras)} propiedades")
        if not compras.empty:
            total_inversion = compras['precio_actual_real'].sum()
            total_ganancia_potencial = compras['diferencia_pesos'].sum()
            roi_promedio = compras['diferencia_porcentual'].mean()
            
            print(f"   💰 Inversión total requerida: ${total_inversion:,.0f}")
            print(f"   💸 Ganancia potencial total: ${total_ganancia_potencial:,.0f}")
            print(f"   📊 ROI promedio: +{roi_promedio:.1f}%")
            
            # Mejor oportunidad
            mejor_compra = compras.nlargest(1, 'diferencia_porcentual').iloc[0]
            print(f"   🏆 Mejor oportunidad: +{mejor_compra['diferencia_porcentual']:.1f}%")
        
        print(f"\n📉 OPORTUNIDADES DE VENTA: {len(ventas)} propiedades")
        if not ventas.empty:
            total_sobreprecio = abs(ventas['diferencia_pesos']).sum()
            sobreprecio_promedio = abs(ventas['diferencia_porcentual']).mean()
            
            print(f"   💰 Sobreprecio total identificado: ${total_sobreprecio:,.0f}")
            print(f"   📊 Sobreprecio promedio: {sobreprecio_promedio:.1f}%")
            
            # Propiedad más sobrevalorada
            peor_venta = ventas.nsmallest(1, 'diferencia_porcentual').iloc[0]
            print(f"   ⚠️  Propiedad más sobrevalorada: {peor_venta['diferencia_porcentual']:.1f}%")

    def save_opportunities_report(self, output_dir='reportes_oportunidades'):
        """
        Guardar reporte completo de oportunidades
        """
        print(f"\n💾 Guardando reporte en '{output_dir}'...")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Guardar CSV con todas las oportunidades
        if self.opportunities_df is not None and not self.opportunities_df.empty:
            csv_path = os.path.join(output_dir, f'oportunidades_detalladas_{timestamp}.csv')
            
            # Seleccionar columnas relevantes
            columnas_base = ['precio_actual_real', 'prediccion_real', 'diferencia_porcentual', 
                           'diferencia_pesos', 'señal', 'confianza']
            columnas_propiedad = ['direccion', 'barrio', 'localidad', 'area', 'estrato', 
                                'habitaciones', 'banos', 'parqueaderos', 'administracion']
            
            columnas_guardar = columnas_base + [col for col in columnas_propiedad if col in self.opportunities_df.columns]
            
            self.opportunities_df[columnas_guardar].to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✅ Oportunidades guardadas: {csv_path}")
        
        # 2. Guardar Excel con hojas separadas
        excel_path = os.path.join(output_dir, f'reporte_oportunidades_{timestamp}.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Hoja 1: Resumen ejecutivo
            summary_data = self._create_summary_data()
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Resumen_Ejecutivo', index=False)
            
            # Hoja 2: Todas las oportunidades
            if self.opportunities_df is not None and not self.opportunities_df.empty:
                self.opportunities_df[columnas_guardar].to_excel(writer, sheet_name='Todas_Oportunidades', index=False)
            
            # Hoja 3: Mejores compras
            if self.opportunities_df is not None and not self.opportunities_df.empty:
                compras = self.opportunities_df[self.opportunities_df['señal'] == 'COMPRA']
                if not compras.empty:
                    compras[columnas_guardar].sort_values('diferencia_porcentual', ascending=False).to_excel(
                        writer, sheet_name='Mejores_Compras', index=False
                    )
            
            # Hoja 4: Mejores ventas
            if self.opportunities_df is not None and not self.opportunities_df.empty:
                ventas = self.opportunities_df[self.opportunities_df['señal'] == 'VENTA']
                if not ventas.empty:
                    ventas[columnas_guardar].sort_values('diferencia_porcentual', ascending=True).to_excel(
                        writer, sheet_name='Mejores_Ventas', index=False
                    )
        
        print(f"✅ Reporte Excel guardado: {excel_path}")
        print("📁 Hojas incluidas: Resumen_Ejecutivo, Todas_Oportunidades, Mejores_Compras, Mejores_Ventas")
        
        return excel_path

    def _create_summary_data(self):
        """Crear datos para el resumen ejecutivo"""
        compras = self.opportunities_df[self.opportunities_df['señal'] == 'COMPRA']
        ventas = self.opportunities_df[self.opportunities_df['señal'] == 'VENTA']
        
        summary_data = {
            'Métrica': [
                'Total Propiedades Analizadas',
                'Oportunidades Identificadas',
                'Compras Recomendadas',
                'Ventas Recomendadas',
                'Inversión Total Requerida (Compras)',
                'Ganancia Potencial Total (Compras)',
                'ROI Promedio (Compras)',
                'Sobreprecio Total Identificado (Ventas)',
                'Sobreprecio Promedio (Ventas)'
            ],
            'Valor': [
                len(self.results_df) if self.results_df is not None else 0,
                len(self.opportunities_df) if self.opportunities_df is not None else 0,
                len(compras),
                len(ventas),
                f"${compras['precio_actual_real'].sum():,.0f}" if not compras.empty else "$0",
                f"${compras['diferencia_pesos'].sum():,.0f}" if not compras.empty else "$0",
                f"+{compras['diferencia_porcentual'].mean():.1f}%" if not compras.empty else "0%",
                f"${abs(ventas['diferencia_pesos']).sum():,.0f}" if not ventas.empty else "$0",
                f"{abs(ventas['diferencia_porcentual']).mean():.1f}%" if not ventas.empty else "0%"
            ]
        }
        
        return summary_data


# FUNCIONES DE CONVENIENCIA

def generate_unified_trading_signals(model_trainer, X_processed, y_true, property_data=None, 
                                   threshold=0.05, use_log_transformation=True, save_report=True):
    """
    Función principal para generar señales de trading unificadas
    """
    print("🚀 INICIANDO GENERACIÓN DE SEÑALES UNIFICADAS")
    print("=" * 60)
    
    try:
        # Inicializar generador
        generator = UnifiedTradingSignalGenerator(
            model_trainer=model_trainer,
            use_log_transformation=use_log_transformation
        )
        
        # Generar señales
        oportunidades, resultados = generator.generate_trading_signals(
            X_processed=X_processed,
            y_true=y_true,
            property_data=property_data,
            threshold=threshold
        )
        
        # Guardar reporte
        if save_report and oportunidades is not None and not oportunidades.empty:
            report_path = generator.save_opportunities_report()
            print(f"📄 Reporte guardado en: {report_path}")
        
        return oportunidades, resultados, generator
        
    except Exception as e:
        print(f"❌ Error en generación de señales: {e}")
        traceback.print_exc()
        return None, None, None

def quick_opportunity_analysis(model_path, X_processed, y_true, property_data=None, 
                             threshold=0.05, use_log_transformation=True):
    """
    Análisis rápido de oportunidades
    """
    print("⚡ ANÁLISIS RÁPIDO DE OPORTUNIDADES")
    
    try:
        # Cargar modelo y generar señales
        generator = UnifiedTradingSignalGenerator(
            model_path=model_path,
            use_log_transformation=use_log_transformation
        )
        
        oportunidades, _ = generator.generate_trading_signals(
            X_processed=X_processed,
            y_true=y_true,
            property_data=property_data,
            threshold=threshold
        )
        
        if oportunidades is not None:
            print(f"\n🎯 RESUMEN RÁPIDO:")
            print(f"   • Compras identificadas: {len(oportunidades[oportunidades['señal'] == 'COMPRA'])}")
            print(f"   • Ventas identificadas: {len(oportunidades[oportunidades['señal'] == 'VENTA'])}")
            print(f"   • Umbral aplicado: {threshold*100:.1f}%")
            
            # Mostrar mejor oportunidad de compra
            compras = oportunidades[oportunidades['señal'] == 'COMPRA']
            if not compras.empty:
                mejor_compra = compras.nlargest(1, 'diferencia_porcentual').iloc[0]
                print(f"   🏆 Mejor compra: +{mejor_compra['diferencia_porcentual']:.1f}%")
        
        return oportunidades
        
    except Exception as e:
        print(f"❌ Error en análisis rápido: {e}")
        return None


# Ejemplo de uso
if __name__ == "__main__":
    print("🔧 MÓDULO UNIFICADO DE OPORTUNIDADES INMOBILIARIAS")
    print("📝 Usa generate_unified_trading_signals() para análisis completo")
    
    # Ejemplo de uso
    try:
        # Suponiendo que tienes un modelo entrenado y datos
        oportunidades, resultados, generator = generate_unified_trading_signals(
            model_trainer=model_trainer_entrenado,  # Tu UnifiedModelTrainer entrenado
            X_processed=X_test,                     # Características preprocesadas
            y_true=y_test,                         # Valores reales (posiblemente log-transformados)
            property_data=df_original,             # DataFrame original con info de propiedades
            threshold=0.05,                        # 5% de diferencia
            use_log_transformation=True           # Especificar si hay transformación log
        )
        
        if oportunidades is not None:
            print(f"\n✅ Análisis completado exitosamente")
            print(f"📊 {len(oportunidades)} oportunidades identificadas")
            
    except Exception as e:
        print(f"❌ Error en ejemplo: {e}")
        traceback.print_exc()