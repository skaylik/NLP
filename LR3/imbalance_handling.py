"""
Модуль для борьбы с дисбалансом классов - адаптированная версия для интеграции
с результатами предыдущих этапов (3-4) из Streamlit приложения (этап 5)
"""

import pandas as pd
import numpy as np
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Импорты с обработкой ошибок
try:
    from sklearn.utils import compute_class_weight
    from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn не установлен")

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning("imbalanced-learn не установлен")


class ClassBalanceAnalyzer:
    """Анализатор дисбаланса классов (упрощенный)"""
    
    def __init__(self, verbose=True, max_samples=10000):
        self.verbose = verbose
        self.max_samples = max_samples
        self.class_distribution = None
        self.balance_metrics = None
        
    def analyze_class_distribution(self, y, class_names=None):
        """
        Анализ распределения классов
        
        Args:
            y: метки классов
            class_names: имена классов
            
        Returns:
            Словарь с метриками дисбаланса
        """
        if self.verbose:
            logger.info(f"Анализ распределения классов. Всего образцов: {len(y)}")
        
        start_time = time.time()
        
        # Ограничение выборки для анализа
        if len(y) > self.max_samples:
            if self.verbose:
                logger.info(f"Выборка слишком большая ({len(y)}). Использую случайную подвыборку.")
            indices = np.random.choice(len(y), self.max_samples, replace=False)
            y = np.array(y)[indices] if isinstance(y, list) else y[indices]
        
        # Подсчет количества примеров в каждом классе
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            # Мультилабельная классификация
            class_counts = y.sum(axis=0)
            class_dist = {f"tag_{i}": int(count) for i, count in enumerate(class_counts)}
            n_classes = len(class_counts)
        else:
            # Обычная классификация
            unique_classes, class_counts = np.unique(y, return_counts=True)
            if class_names is not None and len(class_names) == len(unique_classes):
                class_dist = {class_names[i]: int(count) for i, (cls, count) in enumerate(zip(unique_classes, class_counts))}
            else:
                class_dist = {str(cls): int(count) for cls, count in zip(unique_classes, class_counts)}
            n_classes = len(unique_classes)
        
        # Вычисление метрик дисбаланса
        total_samples = len(y)
        class_counts_list = list(class_dist.values())
        
        # Максимальный и минимальный размер классов
        max_class_count = max(class_counts_list)
        min_class_count = min(class_counts_list)
        
        # Отношение максимального к минимальному (Imbalance Ratio)
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
        
        # Стандартное отклонение размеров классов
        std_dev = np.std(class_counts_list)
        
        # Коэффициент вариации
        coef_variation = std_dev / np.mean(class_counts_list) if np.mean(class_counts_list) > 0 else float('inf')
        
        # Классификация дисбаланса
        if imbalance_ratio < 2:
            imbalance_level = "Сбалансированный"
        elif imbalance_ratio < 10:
            imbalance_level = "Небольшой дисбаланс"
        elif imbalance_ratio < 50:
            imbalance_level = "Умеренный дисбаланс"
        else:
            imbalance_level = "Сильный дисбаланс"
        
        # Сохранение результатов
        self.class_distribution = class_dist
        self.balance_metrics = {
            'total_samples': total_samples,
            'n_classes': n_classes,
            'class_distribution': class_dist,
            'max_class_count': max_class_count,
            'min_class_count': min_class_count,
            'imbalance_ratio': imbalance_ratio,
            'std_dev': std_dev,
            'coef_variation': coef_variation,
            'imbalance_level': imbalance_level
        }
        
        if self.verbose:
            elapsed_time = time.time() - start_time
            logger.info(f"Анализ завершен за {elapsed_time:.2f} секунд")
            logger.info(f"Количество классов: {n_classes}, Коэффициент дисбаланса: {imbalance_ratio:.2f}")
        
        return self.balance_metrics
    
    def get_class_weights(self, y, method='balanced'):
        """
        Вычисление весов классов
        
        Args:
            y: метки классов
            method: метод вычисления весов
            
        Returns:
            Словарь весов классов
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn не установлен. Возвращаю равные веса.")
            unique_classes = np.unique(y)
            return {cls: 1.0 for cls in unique_classes}
        
        if method == 'balanced':
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            return {cls: weight for cls, weight in zip(classes, weights)}
        else:
            # Простой метод: обратные частоты
            unique_classes, counts = np.unique(y, return_counts=True)
            total = len(y)
            weights = total / (len(unique_classes) * counts)
            return {cls: weight for cls, weight in zip(unique_classes, weights)}


class FastSamplingBalancer:
    """Быстрая балансировка через сэмплирование"""
    
    def __init__(self, method='random_oversample', random_state=42, max_samples=5000):
        self.method = method
        self.random_state = random_state
        self.max_samples = max_samples
        self.sampler = None
        
    def fit_resample(self, X, y):
        """
        Применение метода сэмплирования
        
        Args:
            X: признаки
            y: метки
            
        Returns:
            Сбалансированные X, y
        """
        logger.info(f"Применение метода сэмплирования: {self.method}")
        logger.info(f"Исходная форма: X={X.shape}, y={y.shape}")
        
        start_time = time.time()
        
        # Ограничение выборки для сэмплирования
        if len(y) > self.max_samples:
            logger.info(f"Выборка слишком большая ({len(y)}). Использую подвыборку.")
            indices = np.random.choice(len(y), self.max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        try:
            if self.method == 'random_oversample' and IMBLEARN_AVAILABLE:
                self.sampler = RandomOverSampler(random_state=self.random_state)
                X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            elif self.method == 'random_undersample' and IMBLEARN_AVAILABLE:
                self.sampler = RandomUnderSampler(random_state=self.random_state)
                X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            elif self.method == 'smote' and IMBLEARN_AVAILABLE:
                try:
                    self.sampler = SMOTE(random_state=self.random_state)
                    X_resampled, y_resampled = self.sampler.fit_resample(X, y)
                except:
                    # Fallback к RandomOverSampler если SMOTE не работает
                    logger.warning("SMOTE не сработал. Использую RandomOverSampler.")
                    self.sampler = RandomOverSampler(random_state=self.random_state)
                    X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            else:
                # Простая случайная перевыборка
                logger.warning("imbalanced-learn недоступен. Использую простую случайную перевыборку.")
                X_resampled, y_resampled = self._simple_random_oversample(X, y)
                
        except Exception as e:
            logger.error(f"Ошибка при сэмплировании: {e}. Возвращаю исходные данные.")
            X_resampled, y_resampled = X, y
        
        elapsed_time = time.time() - start_time
        logger.info(f"Сэмплирование завершено за {elapsed_time:.2f} секунд")
        logger.info(f"Итоговая форма: X={X_resampled.shape}, y={y_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def _simple_random_oversample(self, X, y):
        """Простая случайная перевыборка"""
        unique_classes, class_counts = np.unique(y, return_counts=True)
        max_count = np.max(class_counts)
        
        X_resampled = []
        y_resampled = []
        
        for cls in unique_classes:
            mask = (y == cls)
            X_cls = X[mask]
            y_cls = y[mask]
            
            n_samples = len(X_cls)
            if n_samples < max_count:
                # Дублируем существующие образцы
                n_needed = max_count - n_samples
                indices = np.random.choice(n_samples, n_needed, replace=True)
                X_resampled.append(np.vstack([X_cls, X_cls[indices]]))
                y_resampled.append(np.hstack([y_cls, y_cls[indices]]))
            else:
                X_resampled.append(X_cls)
                y_resampled.append(y_cls)
        
        X_balanced = np.vstack(X_resampled)
        y_balanced = np.hstack(y_resampled)
        
        return X_balanced, y_balanced


class ImbalanceHandler:
    """Обработчик дисбаланса классов (интегрированный с предыдущими этапами)"""
    
    def __init__(self, random_state=42, language='rus', max_samples=5000):
        self.random_state = random_state
        self.language = language
        self.max_samples = max_samples
        np.random.seed(random_state)
        
        # Инициализация компонентов
        self.analyzer = ClassBalanceAnalyzer(verbose=False)
        logger.info(f"Инициализирован ImbalanceHandler (язык: {language}, max_samples: {max_samples})")
    
    def analyze_imbalance(self, labels, class_names=None):
        """
        Анализ дисбаланса
        
        Args:
            labels: метки классов
            class_names: имена классов
            
        Returns:
            Отчет о дисбалансе
        """
        logger.info("Анализ дисбаланса классов")
        
        if isinstance(labels, list):
            labels = np.array(labels)
        
        # Ограничиваем выборку
        if len(labels) > self.max_samples:
            indices = np.random.choice(len(labels), self.max_samples, replace=False)
            labels_sample = labels[indices]
        else:
            labels_sample = labels
        
        # Анализ
        report = self.analyzer.analyze_class_distribution(labels_sample, class_names)
        
        # Добавление рекомендаций
        imbalance_ratio = report['imbalance_ratio']
        
        if imbalance_ratio < 2:
            recommendations = [
                "✅ **Дисбаланс незначительный**",
                "Специальные методы балансировки не требуются",
                "Можно использовать стандартное обучение моделей"
            ]
        elif imbalance_ratio < 10:
            recommendations = [
                "⚠️ **Небольшой дисбаланс**",
                "Рекомендуется использовать веса классов (class_weight='balanced')",
                "Можно применить RandomOverSampling для миноритарных классов"
            ]
        elif imbalance_ratio < 50:
            recommendations = [
                "⚠️ **Умеренный дисбаланс**",
                "Необходимо применение методов борьбы с дисбалансом",
                "Рекомендуется SMOTE или простая аугментация текстов"
            ]
        else:
            recommendations = [
                "🚨 **Сильный дисбаланс**",
                "Требуется комплексный подход",
                "Рекомендуется комбинация методов: взвешивание + сэмплирование",
                "Рассмотрите возможность агрегирования редких классов"
            ]
        
        report['recommendations'] = recommendations
        logger.info("Анализ дисбаланса завершен")
        
        return report
    
    def apply_balancing(self, X, y, method='class_weight', balancing_params=None):
        """
        Применение метода балансировки
        
        Args:
            X: признаки
            y: метки
            method: метод балансировки
            balancing_params: параметры балансировки
            
        Returns:
            Сбалансированные данные и информация
        """
        logger.info(f"Применение метода балансировки: {method}")
        
        if balancing_params is None:
            balancing_params = {}
        
        info = {
            'method': method,
            'original_shape': X.shape,
            'original_distribution': dict(Counter(y)),
            'start_time': time.time()
        }
        
        try:
            # Ограничиваем размер данных для обработки
            if len(y) > self.max_samples:
                logger.info(f"Данные слишком большие ({len(y)}). Использую подвыборку.")
                indices = np.random.choice(len(y), self.max_samples, replace=False)
                X = X[indices]
                y = y[indices]
            
            if method == 'class_weight':
                # Вычисление весов
                class_weights = self.analyzer.get_class_weights(y, method='balanced')
                info['class_weights'] = class_weights
                info['balanced_shape'] = X.shape
                info['status'] = 'success'
                return X, y, info
            
            elif method == 'random_oversample':
                # Случайная перевыборка
                sampler = FastSamplingBalancer(method='random_oversample')
                X_balanced, y_balanced = sampler.fit_resample(X, y)
                info['balanced_shape'] = X_balanced.shape
                info['status'] = 'success'
                return X_balanced, y_balanced, info
            
            elif method == 'random_undersample':
                # Случайная недовыборка
                sampler = FastSamplingBalancer(method='random_undersample')
                X_balanced, y_balanced = sampler.fit_resample(X, y)
                info['balanced_shape'] = X_balanced.shape
                info['status'] = 'success'
                return X_balanced, y_balanced, info
            
            elif method == 'smote' and IMBLEARN_AVAILABLE:
                # SMOTE
                sampler = FastSamplingBalancer(method='smote')
                X_balanced, y_balanced = sampler.fit_resample(X, y)
                info['balanced_shape'] = X_balanced.shape
                info['status'] = 'success'
                return X_balanced, y_balanced, info
            
            else:
                # Для других методов используем простую перевыборку
                logger.info(f"Метод {method} не поддерживается. Использую random_oversample.")
                sampler = FastSamplingBalancer(method='random_oversample')
                X_balanced, y_balanced = sampler.fit_resample(X, y)
                
                info['balanced_shape'] = X_balanced.shape
                info['status'] = 'fallback'
                info['fallback_method'] = 'random_oversample'
                
                return X_balanced, y_balanced, info
                
        except Exception as e:
            logger.error(f"Ошибка при применении метода {method}: {str(e)}")
            info['status'] = 'error'
            info['error'] = str(e)
            
            return X, y, info


class FastClassWeightBalancer:
    """Быстрое взвешивание классов"""
    
    def __init__(self, method='balanced', max_classes=100):
        self.method = method
        self.max_classes = max_classes
        self.class_weights = None
        self.analyzer = ClassBalanceAnalyzer(verbose=False)
        
    def fit(self, y):
        """Вычисление весов классов"""
        self.class_weights = self.analyzer.get_class_weights(y, self.method)
        return self


class SamplingBalancer:
    """Балансировка через сэмплирование"""
    
    def __init__(self, method='random_oversample', random_state=42):
        self.method = method
        self.random_state = random_state
        self.sampler = FastSamplingBalancer(method=method, random_state=random_state)
        
    def fit_resample(self, X, y):
        """Применение сэмплирования"""
        return self.sampler.fit_resample(X, y)


class TextAugmenter:
    """Заглушка для аугментации текстов"""
    def __init__(self, language='rus', **kwargs):
        self.language = language
    
    def augment(self, texts, labels, n_augment=1):
        """Простая аугментация - возвращаем исходные данные"""
        return texts, labels


# Функции для совместимости
def create_imbalance_report(y, class_names=None):
    """Создание отчета о дисбалансе"""
    handler = ImbalanceHandler()
    return handler.analyze_imbalance(y, class_names)


def visualize_imbalance_comparison(original_y, balanced_y, 
                                 original_label="Оригинальные",
                                 balanced_label="Сбалансированные"):
    """Визуализация сравнения распределений"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("Plotly не установлен. Пропускаю визуализацию.")
        return None
    
    try:
        # Быстрый подсчет
        orig_unique, orig_counts = np.unique(original_y, return_counts=True)
        balanced_unique, balanced_counts = np.unique(balanced_y, return_counts=True)
        
        # Объединяем все классы
        all_classes = sorted(set(np.concatenate([orig_unique, balanced_unique])))
        
        # Преобразуем в строки для Plotly
        all_classes_str = [str(cls) for cls in all_classes]
        
        orig_values = []
        balanced_values = []
        
        for cls in all_classes:
            if cls in orig_unique:
                idx = np.where(orig_unique == cls)[0][0]
                orig_values.append(int(orig_counts[idx]))
            else:
                orig_values.append(0)
                
            if cls in balanced_unique:
                idx = np.where(balanced_unique == cls)[0][0]
                balanced_values.append(int(balanced_counts[idx]))
            else:
                balanced_values.append(0)
        
        fig = go.Figure(data=[
            go.Bar(name=original_label, x=all_classes_str, y=orig_values),
            go.Bar(name=balanced_label, x=all_classes_str, y=balanced_values)
        ])
        
        fig.update_layout(
            title='Сравнение распределения классов',
            xaxis_title='Классы',
            yaxis_title='Количество образцов',
            barmode='group',
            height=400
        )
        
        return fig
        
    except Exception as e:
        logger.warning(f"Ошибка при создании визуализации: {e}")
        return None


def get_available_balancing_methods():
    """Получение списка доступных методов балансировки"""
    methods = {
        'none': 'Без балансировки (базовый вариант)',
        'class_weight': 'Взвешивание классов',
        'random_oversample': 'Случайная перевыборка',
        'random_undersample': 'Случайная недовыборка',
    }
    
    if IMBLEARN_AVAILABLE:
        methods['smote'] = 'SMOTE'
    
    return methods


def get_available_augmentation_methods(language='rus'):
    """Получение списка доступных методов аугментации"""
    return {
        'simple_augmentation': 'Простая аугментация (перемешивание слов)',
    }


# Для совместимости
ClassBalanceAnalyzer = ClassBalanceAnalyzer
ImbalanceHandler = ImbalanceHandler
ClassWeightBalancer = FastClassWeightBalancer
SamplingBalancer = SamplingBalancer


if __name__ == "__main__":
    # Тестирование модуля
    print("✅ Модуль imbalance_handling успешно загружен")
    print(f"Доступные библиотеки:")
    print(f"  Scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
    print(f"  Imbalanced-learn: {'✅' if IMBLEARN_AVAILABLE else '❌'}")
    
    # Пример использования
    y_test = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
    handler = ImbalanceHandler()
    report = handler.analyze_imbalance(y_test)
    print(f"\nПример отчета: {report['imbalance_level']} (коэффициент: {report['imbalance_ratio']:.2f})")