#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programador de Entrenamientos - GuruInversor

Sistema de programación automática para entrenamiento incremental que:
- Programa reentrenamientos automáticos
- Monitorea la salud de los modelos
- Ejecuta tareas de mantenimiento
- Genera reportes periódicos
"""

import os
import sys
import logging
import schedule
import threading
import time
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json

# Importar componentes del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.incremental_trainer import IncrementalTrainer, IncrementalConfig

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Configuración para el programador de entrenamientos."""
    
    # Horarios de ejecución
    daily_check_time: str = "06:00"  # Hora para verificación diaria
    weekly_retrain_day: str = "sunday"  # Día para reentrenamiento semanal
    weekly_retrain_time: str = "02:00"  # Hora para reentrenamiento semanal
    
    # Configuración de monitoreo
    health_check_interval_hours: int = 6  # Intervalo de verificación de salud
    performance_alert_threshold: float = 0.20  # Umbral para alertas de rendimiento
    
    # Listas de tickers
    active_tickers: List[str] = None  # Lista de tickers activos
    priority_tickers: List[str] = None  # Tickers de alta prioridad
    
    # Configuración de reportes
    generate_daily_reports: bool = True
    generate_weekly_reports: bool = True
    report_directory: str = "ml/results/reports"
    
    # Configuración de alertas
    enable_alerts: bool = True
    alert_log_file: str = "ml/results/incremental_logs/alerts.log"
    
    def __post_init__(self):
        """Inicializar valores por defecto."""
        if self.active_tickers is None:
            self.active_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        if self.priority_tickers is None:
            self.priority_tickers = ["AAPL", "MSFT"]
        
        # Crear directorios
        Path(self.report_directory).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.alert_log_file)).mkdir(parents=True, exist_ok=True)


class TrainingScheduler:
    """
    Programador automático de entrenamientos incrementales.
    
    Maneja la programación y ejecución automática de:
    - Verificaciones de salud de modelos
    - Reentrenamientos incrementales
    - Generación de reportes
    - Alertas de rendimiento
    """
    
    def __init__(self, 
                 scheduler_config: SchedulerConfig = None,
                 incremental_config: IncrementalConfig = None):
        """
        Inicializar programador de entrenamientos.
        
        Args:
            scheduler_config: Configuración del programador
            incremental_config: Configuración del entrenador incremental
        """
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.incremental_trainer = IncrementalTrainer(incremental_config)
        
        # Estado del programador
        self.is_running = False
        self.scheduler_thread = None
        self.last_execution_times = {}
        self.execution_history = []
        
        # Configurar logging
        self._setup_logging()
        
        # Configurar programaciones
        self._setup_schedules()
        
        logger.info("⏰ TrainingScheduler inicializado")
        logger.info(f"📊 Tickers activos: {len(self.scheduler_config.active_tickers)}")
        logger.info(f"⭐ Tickers prioritarios: {len(self.scheduler_config.priority_tickers)}")
    def _setup_logging(self):
        """Configurar logging específico para el programador."""
        log_file = Path(self.scheduler_config.alert_log_file)
        
        # Evitar duplicar handlers
        handler_exists = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
            for h in logger.handlers
        )
        
        if not handler_exists:
            try:
                alert_handler = logging.FileHandler(log_file, encoding='utf-8')
                alert_handler.setLevel(logging.WARNING)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                alert_handler.setFormatter(formatter)
                
                logger.addHandler(alert_handler)
            except Exception as e:
                logger.warning(f"No se pudo configurar logging de alertas: {e}")
    
    def _setup_schedules(self):
        """Configurar todas las programaciones automáticas."""
        # Verificación diaria de salud
        schedule.every().day.at(self.scheduler_config.daily_check_time).do(
            self._log_execution, "daily_health_check", self._daily_health_check
        )
        
        # Reentrenamiento semanal
        getattr(schedule.every(), self.scheduler_config.weekly_retrain_day).at(
            self.scheduler_config.weekly_retrain_time
        ).do(
            self._log_execution, "weekly_retrain", self._weekly_retrain
        )
        
        # Verificación de salud cada X horas
        schedule.every(self.scheduler_config.health_check_interval_hours).hours.do(
            self._log_execution, "health_check", self._periodic_health_check
        )
        
        # Reportes diarios
        if self.scheduler_config.generate_daily_reports:
            schedule.every().day.at("23:30").do(
                self._log_execution, "daily_report", self._generate_daily_report
            )
        
        # Reportes semanales
        if self.scheduler_config.generate_weekly_reports:
            schedule.every().sunday.at("23:45").do(
                self._log_execution, "weekly_report", self._generate_weekly_report
            )
        
        logger.info("📅 Programaciones configuradas exitosamente")
    
    def _log_execution(self, task_name: str, task_function: Callable):
        """Wrapper para logging de ejecuciones."""
        start_time = datetime.now()
        logger.info(f"▶️ Ejecutando tarea programada: {task_name}")
        
        try:
            result = task_function()
            
            execution_record = {
                'task_name': task_name,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'success': True,
                'result': result
            }
            
            self.execution_history.append(execution_record)
            self.last_execution_times[task_name] = start_time.isoformat()
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Tarea {task_name} completada en {duration:.1f}s")
            
        except Exception as e:
            execution_record = {
                'task_name': task_name,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            
            self.execution_history.append(execution_record)
            logger.error(f"❌ Error en tarea {task_name}: {e}")
        
        # Mantener solo últimas 100 ejecuciones
        self.execution_history = self.execution_history[-100:]
    
    def _daily_health_check(self) -> Dict[str, any]:
        """Verificación diaria de salud de todos los modelos."""
        logger.info("🏥 Iniciando verificación diaria de salud")
        
        health_results = {
            'total_tickers': len(self.scheduler_config.active_tickers),
            'healthy': [],
            'needs_attention': [],
            'critical': [],
            'summary': {}
        }
        
        for ticker in self.scheduler_config.active_tickers:
            try:
                # Verificar necesidad de reentrenamiento
                check_result = self.incremental_trainer.check_retrain_need(ticker)
                
                if not check_result['needs_retrain']:
                    health_results['healthy'].append(ticker)
                elif any('degradación' in reason.lower() for reason in check_result['reasons']):
                    health_results['critical'].append(ticker)
                else:
                    health_results['needs_attention'].append(ticker)
                
            except Exception as e:
                logger.error(f"Error verificando salud de {ticker}: {e}")
                health_results['critical'].append(ticker)
        
        # Generar resumen
        health_results['summary'] = {
            'healthy_count': len(health_results['healthy']),
            'attention_count': len(health_results['needs_attention']),
            'critical_count': len(health_results['critical']),
            'health_rate': len(health_results['healthy']) / health_results['total_tickers']
        }
        
        # Generar alertas si es necesario
        if health_results['critical']:
            self._generate_alert(
                f"⚠️ {len(health_results['critical'])} modelos en estado crítico: "
                f"{', '.join(health_results['critical'])}"
            )
        
        logger.info(f"🏥 Verificación de salud completada: "
                   f"{health_results['summary']['healthy_count']} saludables, "
                   f"{health_results['summary']['attention_count']} necesitan atención, "
                   f"{health_results['summary']['critical_count']} críticos")
        
        return health_results
    
    def _weekly_retrain(self) -> Dict[str, any]:
        """Reentrenamiento semanal de modelos prioritarios."""
        logger.info("🔄 Iniciando reentrenamiento semanal")
        
        # Entrenar modelos prioritarios
        priority_result = self.incremental_trainer.batch_retrain(
            self.scheduler_config.priority_tickers, force=False
        )
        
        # Verificar y entrenar otros modelos que lo necesiten
        other_tickers = [t for t in self.scheduler_config.active_tickers 
                        if t not in self.scheduler_config.priority_tickers]
        
        other_result = self.incremental_trainer.batch_retrain(
            other_tickers, force=False
        )
        
        # Combinar resultados
        weekly_result = {
            'priority_tickers': priority_result,
            'other_tickers': other_result,
            'total_retrained': (priority_result['summary']['retrained_count'] + 
                              other_result['summary']['retrained_count']),
            'total_processed': (len(self.scheduler_config.priority_tickers) + 
                              len(other_tickers))
        }
        
        logger.info(f"🔄 Reentrenamiento semanal completado: "
                   f"{weekly_result['total_retrained']} modelos reentrenados")
        
        return weekly_result
    
    def _periodic_health_check(self) -> Dict[str, any]:
        """Verificación periódica de salud (cada X horas)."""
        logger.info("🔍 Verificación periódica de salud")
        
        # Verificar solo tickers prioritarios para monitoreo frecuente
        quick_check = {
            'checked_tickers': self.scheduler_config.priority_tickers,
            'issues_found': [],
            'immediate_action_needed': []
        }
        
        for ticker in self.scheduler_config.priority_tickers:
            try:
                check_result = self.incremental_trainer.check_retrain_need(ticker)
                
                if check_result['needs_retrain']:
                    quick_check['issues_found'].append({
                        'ticker': ticker,
                        'reasons': check_result['reasons']
                    })
                    
                    # Si hay degradación severa, marcar para acción inmediata
                    if any('degradación' in reason.lower() for reason in check_result['reasons']):
                        quick_check['immediate_action_needed'].append(ticker)
                
            except Exception as e:
                logger.error(f"Error en verificación periódica de {ticker}: {e}")
                quick_check['immediate_action_needed'].append(ticker)
        
        # Ejecutar reentrenamiento inmediato si es necesario
        if quick_check['immediate_action_needed']:
            logger.warning(f"⚠️ Acción inmediata requerida para: {quick_check['immediate_action_needed']}")
            
            immediate_retrain = self.incremental_trainer.batch_retrain(
                quick_check['immediate_action_needed'], force=True
            )
            quick_check['immediate_retrain_result'] = immediate_retrain
        
        return quick_check
    
    def _generate_daily_report(self) -> str:
        """Generar reporte diario."""
        logger.info("📊 Generando reporte diario")
        
        # Generar reporte del entrenador incremental
        incremental_report = self.incremental_trainer.generate_incremental_report()
        
        # Agregar información del programador
        scheduler_info = [
            "\n📅 INFORMACIÓN DEL PROGRAMADOR",
            "-" * 40,
            f"Última verificación diaria: {self.last_execution_times.get('daily_health_check', 'N/A')}",
            f"Última verificación periódica: {self.last_execution_times.get('health_check', 'N/A')}",
            f"Último reentrenamiento semanal: {self.last_execution_times.get('weekly_retrain', 'N/A')}",
            f"Tickers activos: {len(self.scheduler_config.active_tickers)}",
            f"Tickers prioritarios: {len(self.scheduler_config.priority_tickers)}",
        ]
        
        # Ejecuciones recientes
        recent_executions = [e for e in self.execution_history 
                           if datetime.fromisoformat(e['start_time']) > datetime.now() - timedelta(days=1)]
        
        if recent_executions:
            scheduler_info.append(f"\nEjecuciones últimas 24h: {len(recent_executions)}")
            failed_executions = [e for e in recent_executions if not e['success']]
            if failed_executions:
                scheduler_info.append(f"Ejecuciones fallidas: {len(failed_executions)}")
        
        full_report = incremental_report + "\n" + "\n".join(scheduler_info)
        
        # Guardar reporte
        report_file = Path(self.scheduler_config.report_directory) / f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        logger.info(f"📊 Reporte diario guardado en: {report_file}")
        
        return str(report_file)
    
    def _generate_weekly_report(self) -> str:
        """Generar reporte semanal."""
        logger.info("📊 Generando reporte semanal")
        
        # Estadísticas de la semana
        week_start = datetime.now() - timedelta(days=7)
        weekly_executions = [e for e in self.execution_history 
                           if datetime.fromisoformat(e['start_time']) > week_start]
        
        # Análisis semanal
        weekly_analysis = [
            "📈 ANÁLISIS SEMANAL",
            "=" * 60,
            f"Período: {week_start.strftime('%Y-%m-%d')} a {datetime.now().strftime('%Y-%m-%d')}",
            f"Total ejecuciones: {len(weekly_executions)}",
        ]
        
        if weekly_executions:
            successful = [e for e in weekly_executions if e['success']]
            failed = [e for e in weekly_executions if not e['success']]
            
            weekly_analysis.extend([
                f"Ejecuciones exitosas: {len(successful)}",
                f"Ejecuciones fallidas: {len(failed)}",
                f"Tasa de éxito: {len(successful)/len(weekly_executions):.1%}",
            ])
            
            # Tipos de tareas ejecutadas
            task_types = {}
            for execution in weekly_executions:
                task_name = execution['task_name']
                task_types[task_name] = task_types.get(task_name, 0) + 1
            
            weekly_analysis.append("\nTareas ejecutadas:")
            for task, count in task_types.items():
                weekly_analysis.append(f"  - {task}: {count}")
        
        # Combinar con reporte incremental
        incremental_report = self.incremental_trainer.generate_incremental_report()
        full_weekly_report = "\n".join(weekly_analysis) + "\n\n" + incremental_report
        
        # Guardar reporte
        report_file = Path(self.scheduler_config.report_directory) / f"weekly_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(full_weekly_report)
        
        logger.info(f"📊 Reporte semanal guardado en: {report_file}")
        
        return str(report_file)
    
    def _generate_alert(self, message: str):
        """Generar alerta."""
        if self.scheduler_config.enable_alerts:
            logger.warning(f"🚨 ALERTA: {message}")
    
    def start(self):
        """Iniciar el programador de entrenamientos."""
        if self.is_running:
            logger.warning("⚠️ El programador ya está ejecutándose")
            return
        
        self.is_running = True
        
        def run_scheduler():
            logger.info("🚀 Programador de entrenamientos iniciado")
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Verificar cada minuto
            logger.info("⏸️ Programador de entrenamientos detenido")
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("⏰ TrainingScheduler iniciado en hilo separado")
    
    def stop(self):
        """Detener el programador de entrenamientos."""
        if not self.is_running:
            logger.warning("⚠️ El programador no está ejecutándose")
            return
        
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("⏸️ TrainingScheduler detenido")
    
    def get_status(self) -> Dict[str, any]:
        """Obtener estado del programador."""
        return {
            'is_running': self.is_running,
            'last_execution_times': self.last_execution_times.copy(),
            'recent_executions': len([e for e in self.execution_history 
                                    if datetime.fromisoformat(e['start_time']) > datetime.now() - timedelta(hours=24)]),
            'total_executions': len(self.execution_history),
            'active_tickers': len(self.scheduler_config.active_tickers),
            'next_scheduled_tasks': [str(job) for job in schedule.jobs]
        }
    def execute_manual_task(self, task_name: str) -> Dict[str, any]:
        """Ejecutar tarea manualmente."""
        task_mapping = {
            'health_check': self._daily_health_check,
            'retrain': self._weekly_retrain,
            'daily_report': self._generate_daily_report,
            'weekly_report': self._generate_weekly_report
        }
        
        if task_name not in task_mapping:
            return {'success': False, 'error': f'Tarea no reconocida: {task_name}'}
        
        try:
            logger.info(f"🔧 Ejecutando tarea manual: {task_name}")
            
            # Usar _log_execution para registrar en el historial
            self._log_execution(f"manual_{task_name}", task_mapping[task_name])
            
            # Ejecutar la tarea para obtener el resultado
            result = task_mapping[task_name]()
            return {'success': True, 'result': result}
        except Exception as e:
            logger.error(f"❌ Error en ejecución manual de {task_name}: {e}")
            return {'success': False, 'error': str(e)}


# Función de utilidad para uso directo
def create_training_scheduler(scheduler_config: SchedulerConfig = None, 
                            incremental_config: IncrementalConfig = None) -> TrainingScheduler:
    """
    Crear instancia de programador de entrenamientos.
    
    Args:
        scheduler_config: Configuración del programador
        incremental_config: Configuración del entrenador incremental
        
    Returns:
        Instancia de TrainingScheduler
    """
    return TrainingScheduler(scheduler_config, incremental_config)


if __name__ == "__main__":
    # Ejemplo de uso
    scheduler = create_training_scheduler()
    
    # Mostrar estado
    print("Estado del programador:", scheduler.get_status())
    
    # Ejecutar verificación manual
    result = scheduler.execute_manual_task('health_check')
    print("Resultado verificación manual:", result)
