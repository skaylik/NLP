"""
Модуль для итогового анализа и выбора лучшей модели
Этап 7: Практический анализ всех этапов (3-6)
"""

import numpy as np
import pandas as pd
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import warnings
import traceback
import logging
from collections import Counter

warnings.filterwarnings('ignore')

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Импорт классов из предыдущих этапов
try:
    from classical_classifiers import ClassicalClassifier
except ImportError:
    ClassicalClassifier = None

try:
    from neural_classifiers import NeuralClassifier
except ImportError:
    NeuralClassifier = None


class ModelPerformanceAnalyzer:
    """Анализатор производительности моделей"""
    
    @staticmethod
    def analyze_performance(metrics: Dict, task_type: str = "category") -> Dict:
        """Анализ производительности модели"""
        analysis = {
            'performance_level': 'unknown',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Определяем используемые метрики
        f1_score = metrics.get('f1', metrics.get('f1_macro', metrics.get('f1_micro', 0)))
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', metrics.get('precision_macro', 0))
        recall = metrics.get('recall', metrics.get('recall_macro', 0))
        
        # Определяем уровень производительности
        if f1_score >= 0.9:
            analysis['performance_level'] = 'excellent'
            analysis['strengths'].append('Отличная общая производительность')
        elif f1_score >= 0.8:
            analysis['performance_level'] = 'good'
            analysis['strengths'].append('Хорошая производительность')
        elif f1_score >= 0.7:
            analysis['performance_level'] = 'satisfactory'
            analysis['strengths'].append('Удовлетворительная производительность')
        elif f1_score >= 0.5:
            analysis['performance_level'] = 'poor'
            analysis['weaknesses'].append('Низкая производительность')
        else:
            analysis['performance_level'] = 'very_poor'
            analysis['weaknesses'].append('Очень низкая производительность')
        
        # Анализ баланса precision/recall
        if precision > 0 and recall > 0:
            precision_recall_ratio = abs(precision - recall)
            if precision_recall_ratio > 0.2:
                if precision > recall:
                    analysis['weaknesses'].append('Высокая точность, но низкая полнота (много false negatives)')
                    analysis['recommendations'].append('Увеличить полноту, возможно снизить порог классификации')
                else:
                    analysis['weaknesses'].append('Высокая полнота, но низкая точность (много false positives)')
                    analysis['recommendations'].append('Увеличить точность, возможно повысить порог классификации')
            else:
                analysis['strengths'].append('Сбалансированная точность и полнота')
        
        # Для multi-label задач
        if task_type == 'categories' or metrics.get('is_multi_label', False):
            hamming_loss = metrics.get('hamming_loss', 1.0)
            if hamming_loss > 0.3:
                analysis['weaknesses'].append('Высокий Hamming Loss для multi-label задачи')
                analysis['recommendations'].append('Рассмотреть другие подходы к multi-label классификации')
            else:
                analysis['strengths'].append('Низкий Hamming Loss для multi-label задачи')
        
        # Общие рекомендации
        if f1_score < 0.7:
            analysis['recommendations'].append('Попробовать другие алгоритмы или улучшить признаки')
        if accuracy - f1_score > 0.1:
            analysis['recommendations'].append('Разница между accuracy и F1 указывает на дисбаланс классов')
        
        return analysis


class FinalModelSelector:
    """Класс для выбора финальной модели из всех этапов"""
    
    def __init__(self):
        self.all_models = {}
        self.all_results = {}
        self.champion_model = None
        self.champion_metrics = {}
        self.champion_stage = ""
        self.champion_name = ""
        
    def collect_models_from_stages(self, stage_outputs: Dict) -> Dict:
        """Сбор моделей и результатов из всех этапов"""
        collected_data = {
            'stage3': self._collect_stage3_models(stage_outputs),
            'stage4': self._collect_stage4_models(stage_outputs),
            'stage5': self._collect_stage5_models(stage_outputs),
            'stage6': self._collect_stage6_models(stage_outputs)
        }
        
        # Удаляем пустые этапы
        collected_data = {k: v for k, v in collected_data.items() if v}
        
        # Объединяем все модели
        all_models = {}
        for stage, models in collected_data.items():
            for model_name, (model, metrics) in models.items():
                full_name = f"{stage}_{model_name}"
                all_models[full_name] = {
                    'model': model,
                    'metrics': metrics,
                    'stage': stage,
                    'original_name': model_name
                }
        
        self.all_models = all_models
        return collected_data
    
    def _collect_stage3_models(self, stage_outputs: Dict) -> Dict:
        """Сбор моделей из этапа 3 (классические модели)"""
        models = {}
        
        # 1. Модели из сравнения (comparator)
        if 'comparator_results' in stage_outputs:
            try:
                comparator = stage_outputs['comparator_results']
                if hasattr(comparator, 'models'):
                    for model_name, model_info in comparator.models.items():
                        if hasattr(model_info, 'evaluate'):
                            # Тестовая оценка
                            try:
                                test_metrics = model_info.history.get('evaluation', {})
                                if test_metrics:
                                    models[f"classical_{model_name}"] = (model_info, test_metrics)
                            except:
                                pass
            except Exception as e:
                logger.warning(f"Ошибка при сборе моделей этапа 3: {e}")
        
        # 2. Лучшая модель этапа 3
        if 'best_classical_model' in stage_outputs:
            try:
                model = stage_outputs['best_classical_model']
                metrics = stage_outputs.get('best_classical_metrics', {})
                models['stage3_best'] = (model, metrics)
            except:
                pass
        
        return models
    
    def _collect_stage4_models(self, stage_outputs: Dict) -> Dict:
        """Сбор моделей из этапа 4 (нейросетевые модели)"""
        models = {}
        
        # 1. Модели из нейросетевого компаратора
        if 'neural_results' in stage_outputs:
            try:
                neural_results = stage_outputs['neural_results']
                if isinstance(neural_results, dict):
                    for model_name, result in neural_results.items():
                        if 'model' in result and 'metrics' in result:
                            models[f"neural_{model_name}"] = (result['model'], result['metrics'])
            except Exception as e:
                logger.warning(f"Ошибка при сборе нейросетевых моделей: {e}")
        
        # 2. Лучшая нейросетевая модель
        if 'best_neural_model' in stage_outputs:
            try:
                model = stage_outputs['best_neural_model']
                metrics = stage_outputs.get('best_neural_metrics', {})
                models['stage4_best'] = (model, metrics)
            except:
                pass
        
        return models
    
    def _collect_stage5_models(self, stage_outputs: Dict) -> Dict:
        """Сбор моделей из этапа 5 (балансировка)"""
        models = {}
        
        if 'balancing_results' in stage_outputs:
            try:
                balance_results = stage_outputs['balancing_results']
                for method, result in balance_results.items():
                    if 'model' in result and 'metrics' in result:
                        models[f"balanced_{method}"] = (result['model'], result['metrics'])
            except Exception as e:
                logger.warning(f"Ошибка при сборе сбалансированных моделей: {e}")
        
        return models
    
    def _collect_stage6_models(self, stage_outputs: Dict) -> Dict:
        """Сбор моделей из этапа 6 (настройка гиперпараметров)"""
        models = {}
        
        if 'tuning_results' in stage_outputs:
            try:
                tuning_results = stage_outputs['tuning_results']
                if 'best_model' in tuning_results:
                    model = tuning_results['best_model']
                    metrics = tuning_results.get('evaluation', {}).get('metrics', {})
                    models['tuned_best'] = (model, metrics)
            except Exception as e:
                logger.warning(f"Ошибка при сборе настроенных моделей: {e}")
        
        return models
    
    def select_champion_model(self, criterion: str = 'f1', task_type: str = "category") -> Tuple[Any, Dict, str]:
        """Выбор чемпионской модели по заданному критерию"""
        if not self.all_models:
            raise ValueError("Нет моделей для выбора. Сначала соберите модели.")
        
        best_score = -1
        champion_key = None
        
        for model_key, model_info in self.all_models.items():
            metrics = model_info['metrics']
            
            # Выбираем оценку в зависимости от критерия и типа задачи
            if task_type == 'categories' or metrics.get('is_multi_label', False):
                # Для multi-label задач
                if criterion == 'f1':
                    score = metrics.get('f1_micro', metrics.get('f1', 0))
                elif criterion == 'accuracy':
                    score = metrics.get('accuracy', 0)
                elif criterion == 'hamming':
                    score = -metrics.get('hamming_loss', 1)  # Чем меньше, тем лучше
                else:
                    score = metrics.get(criterion, 0)
            else:
                # Для обычных задач
                score = metrics.get(criterion, 0)
            
            if score > best_score:
                best_score = score
                champion_key = model_key
        
        if champion_key:
            champion_info = self.all_models[champion_key]
            self.champion_model = champion_info['model']
            self.champion_metrics = champion_info['metrics']
            self.champion_stage = champion_info['stage']
            self.champion_name = champion_info['original_name']
            
            logger.info(f"Чемпионская модель: {champion_key} (счет: {best_score:.4f})")
            return self.champion_model, self.champion_metrics, champion_key
        
        raise ValueError("Не удалось выбрать чемпионскую модель")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Создание таблицы сравнения всех моделей"""
        rows = []
        
        for model_key, model_info in self.all_models.items():
            metrics = model_info['metrics']
            
            row = {
                'Model': model_key,
                'Stage': model_info['stage'],
                'F1': metrics.get('f1', metrics.get('f1_macro', metrics.get('f1_micro', 'N/A'))),
                'Accuracy': metrics.get('accuracy', 'N/A'),
                'Precision': metrics.get('precision', metrics.get('precision_macro', 'N/A')),
                'Recall': metrics.get('recall', metrics.get('recall_macro', 'N/A'))
            }
            
            # Для multi-label добавляем специфичные метрики
            if metrics.get('is_multi_label', False):
                row['Hamming Loss'] = metrics.get('hamming_loss', 'N/A')
                row['Jaccard Score'] = metrics.get('jaccard_score', 'N/A')
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Сортируем по F1 score
        if 'F1' in df.columns:
            df['F1_numeric'] = pd.to_numeric(df['F1'], errors='coerce')
            df = df.sort_values('F1_numeric', ascending=False)
            df = df.drop('F1_numeric', axis=1)
        
        return df
    
    def generate_performance_report(self) -> Dict:
        """Генерация полного отчета о производительности"""
        if not self.champion_model:
            raise ValueError("Сначала нужно выбрать чемпионскую модель")
        
        # Анализ производительности
        performance_analysis = ModelPerformanceAnalyzer.analyze_performance(
            self.champion_metrics,
            task_type='categories' if self.champion_metrics.get('is_multi_label', False) else 'category'
        )
        
        # Собираем информацию о модели
        model_info = self._extract_model_info(self.champion_model)
        
        # Создаем отчет
        report = {
            'timestamp': datetime.now().isoformat(),
            'champion_model': {
                'name': self.champion_name,
                'stage': self.champion_stage,
                'type': model_info['model_type'],
                'parameters': model_info.get('parameters', {}),
                'performance_level': performance_analysis['performance_level']
            },
            'performance_metrics': self.champion_metrics,
            'performance_analysis': performance_analysis,
            'comparison_summary': {
                'total_models': len(self.all_models),
                'models_by_stage': self._count_models_by_stage(),
                'best_f1_score': float(self.champion_metrics.get('f1', self.champion_metrics.get('f1_macro', 0)))
            },
            'recommendations': {
                'model_improvement': performance_analysis['recommendations'],
                'deployment': self._generate_deployment_recommendations(),
                'monitoring': self._generate_monitoring_recommendations()
            }
        }
        
        return report
    
    def _extract_model_info(self, model) -> Dict:
        """Извлечение информации о модели"""
        info = {
            'model_type': 'unknown',
            'parameters': {}
        }
        
        model_class = type(model).__name__.lower()
        
        # Определяем тип модели
        if 'classical' in model_class or hasattr(model, 'model_type'):
            info['model_type'] = 'classical'
            if hasattr(model, 'model_type'):
                info['model_type'] = model.model_type
        elif 'neural' in model_class or 'cnn' in model_class or 'rnn' in model_class:
            info['model_type'] = 'neural'
        elif 'transformer' in model_class:
            info['model_type'] = 'transformer'
        elif 'pipeline' in model_class:
            info['model_type'] = 'pipeline'
        
        # Извлекаем параметры
        try:
            if hasattr(model, 'get_params'):
                info['parameters'] = model.get_params()
            elif hasattr(model, 'model') and hasattr(model.model, 'get_params'):
                info['parameters'] = model.model.get_params()
        except:
            pass
        
        return info
    
    def _count_models_by_stage(self) -> Dict:
        """Подсчет моделей по этапам"""
        counts = {}
        for model_info in self.all_models.values():
            stage = model_info['stage']
            counts[stage] = counts.get(stage, 0) + 1
        return counts
    
    def _generate_deployment_recommendations(self) -> List[str]:
        """Генерация рекомендаций по развертыванию"""
        recommendations = []
        
        if self.champion_metrics.get('is_multi_label', False):
            recommendations.append("Для multi-label задачи рассмотрите пороговую оптимизацию для каждого класса")
        
        f1_score = self.champion_metrics.get('f1', self.champion_metrics.get('f1_macro', 0))
        if f1_score >= 0.8:
            recommendations.append("Модель готова к продакшен-развертыванию")
            recommendations.append("Рекомендуется A/B тестирование с текущей системой")
        elif f1_score >= 0.7:
            recommendations.append("Модель требует дополнительной валидации перед развертыванием")
            recommendations.append("Рассмотрите канареечное развертывание (canary release)")
        else:
            recommendations.append("Требуется значительное улучшение перед развертыванием")
            recommendations.append("Рассмотрите использование модели только для рекомендаций")
        
        return recommendations
    
    def _generate_monitoring_recommendations(self) -> List[str]:
        """Генерация рекомендаций по мониторингу"""
        return [
            "Настройте мониторинг метрик качества (accuracy, precision, recall)",
            "Отслеживайте дрейф данных (data drift)",
            "Настройте алерты при падении качества",
            "Реализуйте периодическое переобучение модели"
        ]
    
    def save_champion_model(self, filepath: str) -> None:
        """Сохранение чемпионской модели"""
        if not self.champion_model:
            raise ValueError("Нет чемпионской модели для сохранения")
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.champion_model,
                    'metrics': self.champion_metrics,
                    'stage': self.champion_stage,
                    'name': self.champion_name,
                    'timestamp': datetime.now().isoformat()
                }, f)
            logger.info(f"Чемпионская модель сохранена в {filepath}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}")
            raise


class PracticalInsightsGenerator:
    """Генератор практических инсайтов"""
    
    @staticmethod
    def generate_insights(comparison_df: pd.DataFrame, champion_metrics: Dict) -> Dict:
        """Генерация практических инсайтов"""
        insights = {
            'best_algorithm': '',
            'effectiveness_of_techniques': {},
            'data_insights': [],
            'practical_advice': []
        }
        
        # Определяем лучший алгоритм
        if not comparison_df.empty and 'Model' in comparison_df.columns:
            top_model = comparison_df.iloc[0]['Model']
            insights['best_algorithm'] = top_model
            
            # Анализируем эффективность техник
            insights['effectiveness_of_techniques'] = PracticalInsightsGenerator._analyze_techniques(comparison_df)
        
        # Генерация инсайтов о данных
        insights['data_insights'] = PracticalInsightsGenerator._generate_data_insights(champion_metrics)
        
        # Практические советы
        insights['practical_advice'] = PracticalInsightsGenerator._generate_practical_advice(champion_metrics)
        
        return insights
    
    @staticmethod
    def _analyze_techniques(df: pd.DataFrame) -> Dict:
        """Анализ эффективности различных техник"""
        techniques = {}
        
        # Анализ по типам моделей
        for _, row in df.iterrows():
            model_name = str(row.get('Model', '')).lower()
            
            # Определяем техники по названию модели
            if 'classical' in model_name or 'logistic' in model_name or 'random' in model_name:
                techniques.setdefault('classical_ml', []).append(float(row.get('F1', 0)))
            elif 'neural' in model_name or 'cnn' in model_name or 'rnn' in model_name:
                techniques.setdefault('neural_networks', []).append(float(row.get('F1', 0)))
            elif 'transformer' in model_name:
                techniques.setdefault('transformers', []).append(float(row.get('F1', 0)))
            elif 'balanced' in model_name:
                techniques.setdefault('balancing', []).append(float(row.get('F1', 0)))
            elif 'tuned' in model_name:
                techniques.setdefault('hyperparameter_tuning', []).append(float(row.get('F1', 0)))
        
        # Вычисляем среднюю эффективность
        effectiveness = {}
        for technique, scores in techniques.items():
            if scores:
                effectiveness[technique] = {
                    'average_score': np.mean(scores),
                    'best_score': np.max(scores),
                    'count': len(scores)
                }
        
        return effectiveness
    
    @staticmethod
    def _generate_data_insights(metrics: Dict) -> List[str]:
        """Генерация инсайтов о данных"""
        insights = []
        
        if metrics.get('is_multi_label', False):
            insights.append("Задача является multi-label классификацией")
            hamming_loss = metrics.get('hamming_loss', 1.0)
            if hamming_loss < 0.2:
                insights.append("Низкий Hamming loss указывает на хорошее разделение классов")
            else:
                insights.append("Высокий Hamming loss может указывать на сложность задачи или перекрытие классов")
        else:
            insights.append("Задача является single-label классификацией")
        
        # Анализ точности и полноты
        precision = metrics.get('precision', metrics.get('precision_macro', 0))
        recall = metrics.get('recall', metrics.get('recall_macro', 0))
        
        if abs(precision - recall) > 0.15:
            if precision > recall:
                insights.append("Модель более точна, но пропускает некоторые случаи")
            else:
                insights.append("Модель улавливает большинство случаев, но с ошибками")
        else:
            insights.append("Хороший баланс между точностью и полнотой")
        
        return insights
    
    @staticmethod
    def _generate_practical_advice(metrics: Dict) -> List[str]:
        """Генерация практических советов"""
        advice = []
        
        f1_score = metrics.get('f1', metrics.get('f1_macro', metrics.get('f1_micro', 0)))
        
        if f1_score >= 0.8:
            advice.append("Модель показывает отличные результаты и готова к промышленному использованию")
            advice.append("Рекомендуется сосредоточиться на оптимизации производительности и масштабируемости")
        elif f1_score >= 0.7:
            advice.append("Модель показывает хорошие результаты, но есть возможности для улучшения")
            advice.append("Рассмотрите сбор дополнительных данных или улучшение признаков")
        elif f1_score >= 0.6:
            advice.append("Модель требует существенного улучшения перед использованием в production")
            advice.append("Попробуйте ансамблирование моделей или другие алгоритмы")
        else:
            advice.append("Требуется пересмотр подхода к решению задачи")
            advice.append("Рассмотрите упрощение задачи или сбор большего количества данных")
        
        # Советы по multi-label задачам
        if metrics.get('is_multi_label', False):
            advice.append("Для multi-label задачи рассмотрите использование цепочек классификаторов (Classifier Chains)")
            advice.append("Попробуйте адаптивные пороги для каждого класса")
        
        return advice


class DeploymentPreparer:
    """Подготовка модели к развертыванию"""
    
    @staticmethod
    def prepare_for_deployment(model, model_info: Dict, output_dir: str = "./deployment") -> Dict:
        """Подготовка модели к развертыванию"""
        import os
        import json
        import pickle
        
        os.makedirs(output_dir, exist_ok=True)
        
        deployment_info = {
            'model_file': os.path.join(output_dir, 'model.pkl'),
            'metadata_file': os.path.join(output_dir, 'metadata.json'),
            'requirements_file': os.path.join(output_dir, 'requirements.txt'),
            'api_example': os.path.join(output_dir, 'api_example.py')
        }
        
        # 1. Сохраняем модель
        try:
            with open(deployment_info['model_file'], 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.warning(f"Не удалось сохранить модель: {e}")
            # Пробуем сохранить только веса для нейросетевых моделей
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), deployment_info['model_file'])
        
        # 2. Сохраняем метаданные
        metadata = {
            'model_type': model_info.get('model_type', 'unknown'),
            'creation_date': datetime.now().isoformat(),
            'input_features': model_info.get('input_features', 'unknown'),
            'output_classes': model_info.get('output_classes', []),
            'performance': model_info.get('performance', {}),
            'dependencies': DeploymentPreparer._get_dependencies()
        }
        
        with open(deployment_info['metadata_file'], 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 3. Создаем файл requirements
        DeploymentPreparer._create_requirements_file(deployment_info['requirements_file'])
        
        # 4. Создаем пример API
        DeploymentPreparer._create_api_example(deployment_info['api_example'], model_info)
        
        return deployment_info
    
    @staticmethod
    def _get_dependencies() -> List[str]:
        """Получение зависимостей"""
        dependencies = [
            "numpy>=1.19.0",
            "pandas>=1.2.0",
            "scikit-learn>=0.24.0",
            "python>=3.8"
        ]
        
        # Проверяем наличие дополнительных библиотек
        try:
            import torch
            dependencies.append("torch>=1.9.0")
        except:
            pass
        
        try:
            import tensorflow
            dependencies.append("tensorflow>=2.6.0")
        except:
            pass
        
        try:
            import xgboost
            dependencies.append("xgboost>=1.5.0")
        except:
            pass
        
        return dependencies
    
    @staticmethod
    def _create_requirements_file(filepath: str):
        """Создание файла requirements"""
        with open(filepath, 'w') as f:
            f.write("# Requirements for model deployment\n")
            for dep in DeploymentPreparer._get_dependencies():
                f.write(f"{dep}\n")
    
    @staticmethod
    def _create_api_example(filepath: str, model_info: Dict):
        """Создание примера API"""
        example_code = '''"""
Пример API для развертывания модели
"""

import pickle
import numpy as np
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Загрузка модели
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None

app = FastAPI(title="Model Deployment API")

class PredictionRequest(BaseModel):
    features: List[float]
    
class PredictionResponse(BaseModel):
    prediction: List[float]
    confidence: float
    class_labels: List[str]

@app.get("/")
def read_root():
    return {"message": "Model API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Предсказание на основе входных признаков"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Преобразуем входные данные
        features_array = np.array(request.features).reshape(1, -1)
        
        # Получаем предсказание
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)[0]
            prediction = probabilities.tolist()
            confidence = float(np.max(probabilities))
        elif hasattr(model, 'predict'):
            prediction = model.predict(features_array)[0]
            confidence = 1.0
            prediction = [float(prediction)]
        else:
            raise HTTPException(status_code=500, detail="Model doesn't support prediction")
        
        # Возвращаем результат
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            class_labels=[f"Class_{i}" for i in range(len(prediction))]
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(example_code)


def create_final_analysis_pipeline() -> Dict:
    """Создание пайплайна для финального анализа"""
    return {
        'selector': FinalModelSelector(),
        'insights_generator': PracticalInsightsGenerator(),
        'deployment_preparer': DeploymentPreparer()
    }


def perform_complete_analysis(stage_outputs: Dict, output_dir: str = "./results") -> Dict:
    """
    Выполнение полного анализа всех этапов
    
    Args:
        stage_outputs: Словарь с результатами всех этапов (3-6)
        output_dir: Директория для сохранения результатов
    
    Returns:
        Полный отчет анализа
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Начинаем финальный анализ всех этапов...")
    
    # 1. Создаем пайплайн анализа
    pipeline = create_final_analysis_pipeline()
    selector = pipeline['selector']
    
    # 2. Собираем модели из всех этапов
    collected_models = selector.collect_models_from_stages(stage_outputs)
    logger.info(f"Собрано моделей: {sum(len(m) for m in collected_models.values())}")
    
    # 3. Выбираем чемпионскую модель
    try:
        champion_model, champion_metrics, champion_key = selector.select_champion_model()
        logger.info(f"Выбрана чемпионская модель: {champion_key}")
    except Exception as e:
        logger.error(f"Ошибка при выборе чемпионской модели: {e}")
        # Используем первую доступную модель как fallback
        if selector.all_models:
            champion_key = list(selector.all_models.keys())[0]
            champion_info = selector.all_models[champion_key]
            champion_model = champion_info['model']
            champion_metrics = champion_info['metrics']
            logger.info(f"Используем модель {champion_key} как fallback")
        else:
            raise ValueError("Нет доступных моделей для анализа")
    
    # 4. Создаем таблицу сравнения
    comparison_df = selector.create_comparison_table()
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
    logger.info(f"Таблица сравнения сохранена: {comparison_path}")
    
    # 5. Генерируем отчет о производительности
    performance_report = selector.generate_performance_report()
    report_path = os.path.join(output_dir, 'performance_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(performance_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Отчет о производительности сохранен: {report_path}")
    
    # 6. Генерируем практические инсайты
    insights = pipeline['insights_generator'].generate_insights(comparison_df, champion_metrics)
    insights_path = os.path.join(output_dir, 'practical_insights.json')
    with open(insights_path, 'w', encoding='utf-8') as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    logger.info(f"Практические инсайты сохранены: {insights_path}")
    
    # 7. Сохраняем чемпионскую модель
    model_path = os.path.join(output_dir, 'champion_model.pkl')
    try:
        selector.save_champion_model(model_path)
        logger.info(f"Чемпионская модель сохранена: {model_path}")
    except Exception as e:
        logger.warning(f"Не удалось сохранить чемпионскую модель: {e}")
    
    # 8. Подготавливаем к развертыванию (опционально)
    deployment_dir = os.path.join(output_dir, 'deployment')
    try:
        model_info = {
            'model_type': selector.champion_stage,
            'input_features': 'text_features',
            'output_classes': ['class_0', 'class_1'],  # Заменить на реальные классы
            'performance': champion_metrics
        }
        deployment_info = pipeline['deployment_preparer'].prepare_for_deployment(
            champion_model, model_info, deployment_dir
        )
        logger.info(f"Файлы для развертывания сохранены в: {deployment_dir}")
    except Exception as e:
        logger.warning(f"Не удалось подготовить файлы для развертывания: {e}")
        deployment_info = {}
    
    # 9. Создаем сводный отчет
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_models_analyzed': len(selector.all_models),
        'champion_model': {
            'name': selector.champion_name,
            'stage': selector.champion_stage,
            'key': champion_key,
            'f1_score': champion_metrics.get('f1', champion_metrics.get('f1_macro', 0))
        },
        'best_algorithm': insights.get('best_algorithm', 'unknown'),
        'effectiveness_summary': insights.get('effectiveness_of_techniques', {}),
        'files_generated': {
            'comparison_table': comparison_path,
            'performance_report': report_path,
            'practical_insights': insights_path,
            'champion_model': model_path if os.path.exists(model_path) else None,
            'deployment_files': list(deployment_info.values()) if deployment_info else []
        },
        'recommendations': {
            'next_steps': [
                "Протестировать чемпионскую модель на новых данных",
                "Оптимизировать модель для продакшена",
                "Настроить мониторинг качества предсказаний"
            ]
        }
    }
    
    summary_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info("Финальный анализ успешно завершен!")
    
    return summary


if __name__ == "__main__":
    print("✅ Модуль для финального анализа (этап 7) успешно загружен")
    print("\nФункции:")
    print("  1. FinalModelSelector - выбор лучшей модели из всех этапов")
    print("  2. PracticalInsightsGenerator - генерация практических инсайтов")
    print("  3. DeploymentPreparer - подготовка модели к развертыванию")
    print("  4. perform_complete_analysis - полный анализ всех этапов")