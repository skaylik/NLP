"""
Расширенный модуль для настройки гиперпараметров и оценки моделей
Этап 6: Настройка гиперпараметров - совместим с этапами 3-5
"""

import numpy as np
import pandas as pd
import warnings
import time
import json
import copy
import traceback
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from abc import ABC, abstractmethod
import inspect

warnings.filterwarnings('ignore')

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Импорты с обработкой ошибок
SKLEARN_AVAILABLE = False
OPTUNA_AVAILABLE = False
HYPEROPT_AVAILABLE = False

try:
    import sklearn
    from sklearn.model_selection import (
        StratifiedKFold, TimeSeriesSplit, GroupKFold, 
        GridSearchCV, RandomizedSearchCV, cross_val_score
    )
    from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report, roc_auc_score,
        average_precision_score, log_loss, make_scorer
    )
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logger.error(f"scikit-learn не установлен: {e}")
    # Создаем заглушки для типизации
    class BaseEstimator:
        pass
    class ClassifierMixin:
        pass
    class Pipeline:
        pass

# Попытка импорта Optuna для байесовской оптимизации
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("Optuna не установлена. Bayesian Optimization недоступен.")

# Попытка импорта Hyperopt
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    logger.warning("Hyperopt не установлен.")


# ============================================================
# ОБЕРТКИ ДЛЯ ВСЕХ ТИПОВ МОДЕЛЕЙ ИЗ ЭТАПОВ 3-5
# ============================================================

class UniversalModelWrapper(BaseEstimator, ClassifierMixin):
    """Универсальная обертка для всех типов моделей из этапов 3-5"""
    
    def __init__(self, model=None, **params):
        """
        Инициализация обертки для любой модели.
        
        Args:
            model: Исходная модель (ClassicalClassifier, NeuralClassifier, и т.д.)
            **params: Параметры для установки в модель
        """
        if model is None:
            # Если модель не передана, создаем простую модель по умолчанию
            logger.warning("Модель не передана. Создаю LogisticRegression по умолчанию.")
            model = LogisticRegression()
            
        self.model = model
        self.params = params
        self.model_type = self._detect_model_type(model)
        self._is_multi_label = hasattr(model, 'is_multi_label') and model.is_multi_label
        
        if params:
            self.set_params(**params)
        
        logger.info(f"Создана универсальная обертка для модели типа: {self.model_type}")
    
    def _detect_model_type(self, model):
        """Определение типа модели"""
        model_class_name = model.__class__.__name__.lower()
        
        if 'classical' in model_class_name:
            return 'classical'
        elif 'neural' in model_class_name or 'cnn' in model_class_name or 'rnn' in model_class_name:
            return 'neural'
        elif 'transformer' in model_class_name:
            return 'transformer'
        elif 'ensemble' in model_class_name:
            return 'ensemble'
        elif 'logistic' in model_class_name:
            return 'logistic'
        elif 'randomforest' in model_class_name:
            return 'random_forest'
        elif 'svm' in model_class_name:
            return 'svm'
        elif 'xgboost' in model_class_name:
            return 'xgboost'
        else:
            return 'unknown'
    
    def get_params(self, deep=True):
        """Получение параметров модели"""
        params = self.params.copy()
        
        if deep:
            # Рекурсивно получаем параметры вложенных объектов
            if hasattr(self.model, 'get_params'):
                model_params = self.model.get_params(deep=deep)
                params.update({f'model__{k}': v for k, v in model_params.items()})
        
        return params
    
    def set_params(self, **params):
        """Установка параметров модели"""
        model_params = {}
        other_params = {}
        
        for key, value in params.items():
            if key.startswith('model__'):
                model_params[key[7:]] = value
            else:
                other_params[key] = value
        
        # Обновляем параметры модели
        if model_params:
            if hasattr(self.model, 'set_params'):
                try:
                    self.model.set_params(**model_params)
                except Exception as e:
                    logger.warning(f"Не удалось установить параметры модели: {e}")
            else:
                # Пробуем установить параметры через атрибуты
                for key, value in model_params.items():
                    if hasattr(self.model, key):
                        try:
                            setattr(self.model, key, value)
                        except:
                            pass
        
        # Обновляем локальные параметры
        self.params.update(other_params)
        
        return self
    
    def fit(self, X, y, **fit_params):
        """Обучение модели"""
        try:
            if hasattr(self.model, 'fit'):
                # Проверяем сигнатуру метода fit
                fit_signature = inspect.signature(self.model.fit)
                fit_params_names = list(fit_signature.parameters.keys())
                
                # Подготовка параметров для fit
                model_fit_params = {}
                for key, value in fit_params.items():
                    if key in fit_params_names:
                        model_fit_params[key] = value
                
                if 'X_val' in fit_params_names and 'X_val' in fit_params:
                    # Если модель поддерживает validation данные
                    if 'y_val' in fit_params:
                        self.model.fit(X, y, X_val=fit_params['X_val'], y_val=fit_params['y_val'], **model_fit_params)
                    else:
                        self.model.fit(X, y, X_val=fit_params['X_val'], **model_fit_params)
                else:
                    self.model.fit(X, y, **model_fit_params)
            else:
                logger.error("Модель не имеет метода fit")
                raise AttributeError("Model does not have fit method")
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            raise
        
        return self
    
    def predict(self, X):
        """Предсказание классов"""
        if hasattr(self.model, 'predict'):
            try:
                return self.model.predict(X)
            except Exception as e:
                logger.error(f"Ошибка при предсказании: {e}")
                raise
        else:
            raise AttributeError("Model does not have predict method")
    
    def predict_proba(self, X):
        """Предсказание вероятностей"""
        if hasattr(self.model, 'predict_proba'):
            try:
                return self.model.predict_proba(X)
            except Exception:
                # Если нет predict_proba, возвращаем псевдовероятности
                y_pred = self.predict(X)
                n_classes = len(np.unique(y_pred))
                n_samples = len(X)
                
                proba = np.zeros((n_samples, n_classes))
                for i, pred in enumerate(y_pred):
                    proba[i, int(pred)] = 1.0
                return proba
        else:
            # Возвращаем псевдовероятности
            y_pred = self.predict(X)
            n_classes = len(np.unique(y_pred))
            n_samples = len(X)
            
            proba = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(y_pred):
                proba[i, int(pred)] = 1.0
            return proba
    
    def score(self, X, y):
        """Оценка точности модели"""
        if hasattr(self.model, 'score'):
            return self.model.score(X, y)
        else:
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred)


def ensure_sklearn_compatible(model):
    """
    Обеспечивает совместимость любой модели со scikit-learn.
    
    Args:
        model: Исходная модель
        
    Returns:
        Sklearn-совместимая модель
    """
    # Проверяем, совместима ли уже модель
    if hasattr(model, 'get_params') and hasattr(model, 'set_params'):
        return model
    
    # Если нет, оборачиваем
    logger.info("Обнаружена несовместимая модель. Применяем универсальную обертку...")
    return UniversalModelWrapper(model=model)


# ============================================================
# БАЗОВЫЕ КЛАССЫ
# ============================================================

class BaseCrossValidator(ABC):
    """Базовый класс для кросс-валидации"""
    
    @abstractmethod
    def get_cv_splits(self, X, y=None, groups=None):
        """Получение сплитов для кросс-валидации"""
        pass


class StratifiedKFoldCV(BaseCrossValidator):
    """Stratified K-Fold для сохранения баланса классов"""
    
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def get_cv_splits(self, X, y=None, groups=None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn не установлен")
        return StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        ).split(X, y)


class TimeSeriesCV(BaseCrossValidator):
    """Временное разделение для временных рядов"""
    
    def __init__(self, n_splits=5, max_train_size=None, test_size=None):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
    
    def get_cv_splits(self, X, y=None, groups=None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn не установлен")
        return TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=self.max_train_size,
            test_size=self.test_size
        ).split(X, y)


class GroupKFoldCV(BaseCrossValidator):
    """Group K-Fold для данных с групповой структурой"""
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
    
    def get_cv_splits(self, X, y=None, groups=None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn не установлен")
        if groups is None:
            raise ValueError("GroupKFold требует параметр groups")
        return GroupKFold(n_splits=self.n_splits).split(X, y, groups)


class BaseHyperparameterOptimizer(ABC):
    """Базовый класс для оптимизации гиперпараметров"""
    
    @abstractmethod
    def optimize(self, estimator, X, y, param_space, cv, scoring):
        """Оптимизация гиперпараметров"""
        pass


class GridSearchOptimizer(BaseHyperparameterOptimizer):
    """Grid Search для полного перебора"""
    
    def __init__(self, n_jobs=-1, verbose=0):
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def optimize(self, estimator, X, y, param_space, cv, scoring):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn не установлен")
        
        # Обеспечиваем совместимость
        estimator = ensure_sklearn_compatible(estimator)
        
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_space,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        grid_search.fit(X, y)
        return grid_search


class RandomSearchOptimizer(BaseHyperparameterOptimizer):
    """Random Search для случайного поиска"""
    
    def __init__(self, n_iter=50, n_jobs=-1, verbose=0, random_state=42):
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
    
    def optimize(self, estimator, X, y, param_space, cv, scoring):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn не установлен")
        
        # Обеспечиваем совместимость
        estimator = ensure_sklearn_compatible(estimator)
        
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_space,
            n_iter=self.n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state
        )
        random_search.fit(X, y)
        return random_search


class BayesianOptimizer(BaseHyperparameterOptimizer):
    """Bayesian Optimization с использованием Optuna"""
    
    def __init__(self, n_trials=50, timeout=None, n_jobs=-1, random_state=42):
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def optimize(self, estimator, X, y, param_space, cv, scoring):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna не установлена. Установите: pip install optuna")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn не установлен")
        
        # Обеспечиваем совместимость
        estimator = ensure_sklearn_compatible(estimator)
        
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state)
        )
        
        def objective(trial):
            params = {}
            for param_name, param_config in param_space.items():
                if isinstance(param_config, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
                elif isinstance(param_config, dict) and 'type' in param_config:
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                else:
                    # Если конфиг не в ожидаемом формате, используем как есть
                    params[param_name] = param_config
            
            estimator_clone = clone(estimator)
            estimator_clone.set_params(**params)
            
            scores = cross_val_score(
                estimator_clone, X, y,
                cv=cv, scoring=scoring, n_jobs=self.n_jobs
            )
            return scores.mean()
        
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Создаем лучшую модель
        best_params = study.best_params
        best_estimator = clone(estimator)
        best_estimator.set_params(**best_params)
        best_estimator.fit(X, y)
        
        # Создаем результат
        class OptunaResult:
            def __init__(self, estimator, params, score):
                self.best_estimator_ = estimator
                self.best_params_ = params
                self.best_score_ = score
        
        return OptunaResult(best_estimator, best_params, study.best_value)


# ============================================================
# КЛАСС ДЛЯ РЕГУЛЯРИЗАЦИИ (совместим с этапами 3-5)
# ============================================================

class RegularizationManager:
    """Менеджер для применения методов регуляризации"""
    
    @staticmethod
    def apply_l1_l2(model, alpha=0.01, l1_ratio=0.5):
        """Применение L1 и L2 регуляризации для линейных моделей"""
        if hasattr(model, 'penalty'):
            model.set_params(penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio)
        return model
    
    @staticmethod
    def apply_dropout(model, dropout_rate=0.5):
        """Применение Dropout для нейросетей"""
        if hasattr(model, 'dropout'):
            model.set_params(dropout=dropout_rate)
        elif hasattr(model, 'dropout_rate'):
            model.set_params(dropout_rate=dropout_rate)
        return model
    
    @staticmethod
    def apply_weight_decay(model, weight_decay=0.01):
        """Применение Weight Decay для трансформеров"""
        if hasattr(model, 'weight_decay'):
            model.set_params(weight_decay=weight_decay)
        return model
    
    @staticmethod
    def apply_early_stopping(model, patience=10, min_delta=0.001):
        """Настройка Early Stopping"""
        if hasattr(model, 'early_stopping'):
            model.set_params(
                early_stopping=True,
                early_stopping_patience=patience,
                early_stopping_min_delta=min_delta
            )
        return model
    
    @staticmethod
    def apply_class_weights(model, class_weights):
        """Применение весов классов"""
        if hasattr(model, 'class_weight'):
            model.set_params(class_weight=class_weights)
        return model


# ============================================================
# КОМПЛЕКСНЫЙ ОЦЕНЩИК МОДЕЛЕЙ (совместим с этапами 3-5)
# ============================================================

class ComprehensiveModelEvaluator:
    """Комплексная оценка моделей с поддержкой всех типов из этапов 3-5"""
    
    def __init__(self, metrics=None):
        self.metrics = metrics or [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'precision_micro', 'recall_micro', 'f1_micro',
            'roc_auc', 'pr_auc', 'log_loss'
        ]
        self.results = {}
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None, classes=None, is_multi_label=False):
        """Вычисление всех метрик с учетом типа задачи"""
        metrics_dict = {}
        
        if not SKLEARN_AVAILABLE:
            return metrics_dict
        
        # Базовые метрики
        if 'accuracy' in self.metrics:
            try:
                metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
            except:
                metrics_dict['accuracy'] = 0
        
        if is_multi_label:
            # Для multi-label задач
            try:
                # Преобразуем к бинарному формату если нужно
                if len(y_true.shape) == 1 or y_true.shape[1] == 1:
                    # Простая multi-label обработка
                    unique_labels = np.unique(y_true)
                    y_true_binary = np.zeros((len(y_true), len(unique_labels)))
                    y_pred_binary = np.zeros((len(y_pred), len(unique_labels)))
                    
                    for i, label in enumerate(unique_labels):
                        y_true_binary[:, i] = (y_true == label).astype(int)
                        y_pred_binary[:, i] = (y_pred == label).astype(int)
                else:
                    y_true_binary = y_true
                    y_pred_binary = y_pred
                
                # Hamming loss для multi-label
                from sklearn.metrics import hamming_loss
                metrics_dict['hamming_loss'] = hamming_loss(y_true_binary, y_pred_binary)
                
                # Precision, Recall, F1 для multi-label
                try:
                    metrics_dict['precision_micro'] = precision_score(y_true_binary, y_pred_binary, 
                                                                     average='micro', zero_division=0)
                    metrics_dict['recall_micro'] = recall_score(y_true_binary, y_pred_binary, 
                                                               average='micro', zero_division=0)
                    metrics_dict['f1_micro'] = f1_score(y_true_binary, y_pred_binary, 
                                                       average='micro', zero_division=0)
                    
                    metrics_dict['precision_macro'] = precision_score(y_true_binary, y_pred_binary, 
                                                                     average='macro', zero_division=0)
                    metrics_dict['recall_macro'] = recall_score(y_true_binary, y_pred_binary, 
                                                               average='macro', zero_division=0)
                    metrics_dict['f1_macro'] = f1_score(y_true_binary, y_pred_binary, 
                                                       average='macro', zero_division=0)
                except:
                    pass
                
            except Exception as e:
                logger.warning(f"Ошибка при расчете multi-label метрик: {e}")
        
        else:
            # Для single-label задач
            # Макро усреднение
            if 'precision_macro' in self.metrics:
                try:
                    metrics_dict['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                except:
                    metrics_dict['precision_macro'] = 0
                    
            if 'recall_macro' in self.metrics:
                try:
                    metrics_dict['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                except:
                    metrics_dict['recall_macro'] = 0
                    
            if 'f1_macro' in self.metrics:
                try:
                    metrics_dict['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
                except:
                    metrics_dict['f1_macro'] = 0
            
            # Микро усреднение
            if 'precision_micro' in self.metrics:
                try:
                    metrics_dict['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
                except:
                    metrics_dict['precision_micro'] = 0
                    
            if 'recall_micro' in self.metrics:
                try:
                    metrics_dict['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
                except:
                    metrics_dict['recall_micro'] = 0
                    
            if 'f1_micro' in self.metrics:
                try:
                    metrics_dict['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
                except:
                    metrics_dict['f1_micro'] = 0
        
        # ROC-AUC (только для бинарной классификации)
        if 'roc_auc' in self.metrics and y_pred_proba is not None:
            try:
                if classes is not None and len(classes) == 2:
                    metrics_dict['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                elif classes is not None:
                    metrics_dict['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                else:
                    metrics_dict['roc_auc'] = 0.0
            except Exception:
                metrics_dict['roc_auc'] = 0.0
        
        # PR-AUC (для задач с дисбалансом)
        if 'pr_auc' in self.metrics and y_pred_proba is not None:
            try:
                if classes is not None and len(classes) == 2:
                    metrics_dict['pr_auc'] = average_precision_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics_dict['pr_auc'] = 0.0  # Для многоклассовой сложнее
            except Exception:
                metrics_dict['pr_auc'] = 0.0
        
        # Log Loss
        if 'log_loss' in self.metrics and y_pred_proba is not None:
            try:
                metrics_dict['log_loss'] = log_loss(y_true, y_pred_proba)
            except Exception:
                metrics_dict['log_loss'] = float('inf')
        
        return metrics_dict
    
    def evaluate(self, model, X_test, y_test, X_train=None, y_train=None):
        """Оценка модели (совместим с моделями из этапов 3-5)"""
        start_time = time.time()
        
        results = {
            'metrics': {},
            'confusion_matrix': None,
            'classification_report': None,
            'feature_importance': None,
            'calibration_curve': None,
            'prediction_time': 0,
            'model_type': 'unknown'
        }
        
        try:
            # Определяем тип модели
            model_wrapper = ensure_sklearn_compatible(model)
            model_type = getattr(model_wrapper, 'model_type', 'unknown')
            results['model_type'] = model_type
            
            # Проверяем, является ли задача multi-label
            is_multi_label = False
            if hasattr(model, 'is_multi_label'):
                is_multi_label = model.is_multi_label
            elif hasattr(model_wrapper, '_is_multi_label'):
                is_multi_label = model_wrapper._is_multi_label
            
            # Предсказания
            y_pred = model_wrapper.predict(X_test)
            
            # Вероятности (если доступно)
            y_pred_proba = None
            if hasattr(model_wrapper, 'predict_proba'):
                try:
                    y_pred_proba = model_wrapper.predict_proba(X_test)
                except Exception:
                    pass
            
            # Вычисление метрик
            classes = np.unique(y_test) if not is_multi_label else None
            results['metrics'] = self._calculate_metrics(
                y_test, y_pred, y_pred_proba, classes, is_multi_label
            )
            
            # Матрица ошибок (только для single-label)
            if not is_multi_label and SKLEARN_AVAILABLE:
                try:
                    results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
                    
                    # Classification report
                    results['classification_report'] = classification_report(
                        y_test, y_pred, output_dict=True, zero_division=0
                    )
                except Exception:
                    pass
            
            # Важность признаков
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                results['feature_importance'] = model.coef_.tolist()
            
            # Время предсказания
            results['prediction_time'] = time.time() - start_time
            
            # Кривая калибровки (если есть вероятности и не multi-label)
            if (not is_multi_label and y_pred_proba is not None and 
                classes is not None and len(classes) == 2 and SKLEARN_AVAILABLE):
                try:
                    from sklearn.calibration import calibration_curve
                    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
                    results['calibration_curve'] = {
                        'prob_true': prob_true.tolist(),
                        'prob_pred': prob_pred.tolist()
                    }
                except Exception:
                    pass
        
        except Exception as e:
            logger.error(f"Ошибка при оценке модели: {e}")
            results['error'] = str(e)
        
        self.results = results
        return results
    
    def create_comprehensive_report(self, evaluation_results, tuning_info=None):
        """Создание комплексного отчета"""
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'metrics': evaluation_results.get('metrics', {}),
            'model_type': evaluation_results.get('model_type', 'unknown'),
            'model_performance': self._assess_performance(evaluation_results.get('metrics', {})),
            'recommendations': self._generate_recommendations(evaluation_results),
            'tuning_info': tuning_info
        }
        return report
    
    def _assess_performance(self, metrics):
        """Оценка производительности модели"""
        f1_macro = metrics.get('f1_macro', 0)
        f1_micro = metrics.get('f1_micro', 0)
        accuracy = metrics.get('accuracy', 0)
        
        # Используем лучшую доступную метрику
        best_f1 = max(f1_macro, f1_micro) if f1_macro > 0 or f1_micro > 0 else accuracy
        
        if best_f1 >= 0.99:
            return "Отличная производительность (возможна утечка данных или слишком простые данные)"
        elif best_f1 >= 0.9:
            return "Отличная производительность"
        elif best_f1 >= 0.8:
            return "Хорошая производительность"
        elif best_f1 >= 0.7:
            return "Удовлетворительная производительность"
        elif best_f1 >= 0.5:
            return "Слабая производительность"
        else:
            return "Очень слабая производительность"
    
    def _generate_recommendations(self, evaluation_results):
        """Генерация рекомендаций"""
        metrics = evaluation_results.get('metrics', {})
        recommendations = []
        
        f1_macro = metrics.get('f1_macro', 0)
        f1_micro = metrics.get('f1_micro', 0)
        accuracy = metrics.get('accuracy', 0)
        
        best_f1 = max(f1_macro, f1_micro) if f1_macro > 0 or f1_micro > 0 else accuracy
        
        if best_f1 >= 0.99:
            recommendations.append("⚠️ F1-Score = 1.0. Возможна утечка данных или слишком простая задача. Проверьте данные.")
        
        if best_f1 < 0.7:
            recommendations.append("Рекомендуется улучшить балансировку классов или использовать другие признаки")
        
        if 'hamming_loss' in metrics and metrics['hamming_loss'] > 0.3:
            recommendations.append("Высокий Hamming Loss для multi-label задачи. Рассмотрите другие подходы к классификации")
        
        if abs(f1_macro - f1_micro) > 0.1 and f1_macro > 0 and f1_micro > 0:
            recommendations.append("Большая разница между macro и micro F1. Возможен дисбаланс классов")
        
        return recommendations


# ============================================================
# ГЛАВНЫЙ КЛАСС ДЛЯ НАСТРОЙКИ И ОЦЕНКИ (интегрирован с этапами 3-5)
# ============================================================

class AdvancedModelTuner:
    """Продвинутый тюнер для настройки и оценки моделей из этапов 3-5"""
    
    def __init__(self, 
                 cv_strategy='stratified',
                 cv_splits=5,
                 optimizer_type='random',
                 n_trials=50,
                 scoring='f1_macro',
                 regularization_params=None,
                 metrics=None,
                 n_jobs=-1,
                 random_state=42):
        
        self.cv_strategy = cv_strategy
        self.cv_splits = cv_splits
        self.optimizer_type = optimizer_type
        self.n_trials = n_trials
        self.scoring = scoring
        self.regularization_params = regularization_params or {}
        self.metrics = metrics
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Инициализация компонентов
        self.cv = self._create_cv_strategy()
        self.optimizer = self._create_optimizer()
        self.evaluator = ComprehensiveModelEvaluator(metrics=metrics)
        self.regularizer = RegularizationManager()
        
        logger.info(f"Инициализирован AdvancedModelTuner (интегрирован с этапами 3-5): "
                   f"CV={cv_strategy}, Оптимизатор={optimizer_type}")
    
    def _create_cv_strategy(self):
        """Создание стратегии кросс-валидации"""
        if self.cv_strategy == 'stratified':
            return StratifiedKFoldCV(n_splits=self.cv_splits, random_state=self.random_state)
        elif self.cv_strategy == 'timeseries':
            return TimeSeriesCV(n_splits=self.cv_splits)
        elif self.cv_strategy == 'group':
            return GroupKFoldCV(n_splits=self.cv_splits)
        else:
            raise ValueError(f"Неизвестная стратегия CV: {self.cv_strategy}")
    
    def _create_optimizer(self):
        """Создание оптимизатора гиперпараметров"""
        if self.optimizer_type == 'grid':
            return GridSearchOptimizer(n_jobs=self.n_jobs, verbose=0)
        elif self.optimizer_type == 'random':
            return RandomSearchOptimizer(
                n_iter=self.n_trials,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )
        elif self.optimizer_type == 'bayesian':
            if not OPTUNA_AVAILABLE:
                logger.warning("Optuna не доступна. Использую Random Search.")
                return RandomSearchOptimizer(n_iter=self.n_trials, n_jobs=self.n_jobs)
            return BayesianOptimizer(n_trials=self.n_trials, n_jobs=self.n_jobs)
        else:
            raise ValueError(f"Неизвестный оптимизатор: {self.optimizer_type}")
    
    def _get_param_space(self, model_type, original_model):
        """Получение пространства параметров для разных типов моделей"""
        
        # Базовые пространства параметров для классических моделей
        classical_spaces = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['lbfgs', 'liblinear', 'saga'],
                'max_iter': [100, 200, 500],
                'l1_ratio': [0, 0.15, 0.5, 0.85, 1]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto'],
                'degree': [2, 3, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'subsample': [0.5, 0.7, 1.0],
                'colsample_bytree': [0.5, 0.7, 1.0]
            },
            'classical_classifier': {
                'learning_rate': [0.001, 0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'n_estimators': [50, 100, 200],
                'subsample': [0.5, 0.7, 1.0]
            }
        }
        
        # Пространства для нейросетевых моделей
        neural_spaces = {
            'neural_classifier': {
                'learning_rate': [1e-4, 1e-3, 1e-2],
                'batch_size': [16, 32, 64],
                'epochs': [10, 20, 30]
            },
            'mlp': {
                'learning_rate': [1e-4, 1e-3, 1e-2],
                'hidden_dims': [[128], [256], [128, 64]],
                'dropout': [0.1, 0.3, 0.5]
            },
            'cnn': {
                'learning_rate': [1e-4, 1e-3, 1e-2],
                'num_filters': [64, 128, 256],
                'filter_sizes': [[3, 4, 5], [2, 3, 4]],
                'dropout': [0.1, 0.3, 0.5]
            },
            'rnn': {
                'learning_rate': [1e-4, 1e-3, 1e-2],
                'hidden_dim': [64, 128, 256],
                'num_layers': [1, 2, 3],
                'dropout': [0.1, 0.3, 0.5]
            },
            'transformer': {
                'learning_rate': [2e-5, 5e-5, 1e-4],
                'batch_size': [8, 16, 32],
                'dropout': [0.1, 0.2, 0.3]
            }
        }
        
        # Определяем, к какому типу относится модель
        if model_type in ['classical', 'logistic', 'logistic_regression']:
            return classical_spaces['logistic_regression']
        elif model_type in ['random_forest']:
            return classical_spaces['random_forest']
        elif model_type in ['svm']:
            return classical_spaces['svm']
        elif model_type in ['xgboost']:
            return classical_spaces['xgboost']
        elif model_type in ['classical_classifier']:
            return classical_spaces['classical_classifier']
        
        elif model_type in ['neural', 'mlp']:
            return neural_spaces['mlp']
        elif model_type in ['cnn']:
            return neural_spaces['cnn']
        elif model_type in ['rnn']:
            return neural_spaces['rnn']
        elif model_type in ['transformer']:
            return neural_spaces['transformer']
        else:
            # Возвращаем базовое пространство параметров
            return classical_spaces['random_forest']
    
    def _detect_model_type(self, model):
        """Определение типа модели"""
        # Если модель уже обернута в UniversalModelWrapper
        if hasattr(model, 'model_type'):
            return model.model_type
        
        # Определяем по имени класса
        model_name = str(type(model)).lower()
        
        # Проверяем на классические модели
        if 'logistic' in model_name:
            return 'logistic_regression'
        elif 'randomforest' in model_name:
            return 'random_forest'
        elif 'svm' in model_name or 'svc' in model_name:
            return 'svm'
        elif 'xgboost' in model_name:
            return 'xgboost'
        elif 'classical' in model_name:
            return 'classical_classifier'
        
        # Проверяем на нейросетевые модели
        elif 'neural' in model_name:
            return 'neural'
        elif 'mlp' in model_name:
            return 'mlp'
        elif 'cnn' in model_name:
            return 'cnn'
        elif 'rnn' in model_name or 'lstm' in model_name or 'gru' in model_name:
            return 'rnn'
        elif 'transformer' in model_name:
            return 'transformer'
        
        else:
            return 'unknown'
    
    def _prepare_model_for_optimization(self, model):
        """Подготовка модели для оптимизации (обеспечение совместимости)"""
        # Обеспечиваем совместимость модели
        if not hasattr(model, 'get_params') or not hasattr(model, 'set_params'):
            logger.info("Модель не совместима со scikit-learn. Применяем универсальную обертку...")
            model = UniversalModelWrapper(model=model)
        return model
    
    def tune_and_evaluate(self, model, X_train, y_train, X_test, y_test, 
                          groups=None, model_type=None, stage_outputs=None,
                          task_name=None, balancing_info=None):
        """Полная настройка и оценка модели (интегрирована с этапами 3-5)"""
        logger.info("Начало комплексной настройки и оценки модели")
        logger.info(f"Размеры данных: Train={X_train.shape}, Test={X_test.shape}")
        
        results = {
            'tuning': None,
            'evaluation': None,
            'report': None,
            'success': False
        }
        
        try:
            # 0. Проверка модели
            if model is None:
                logger.error("Модель для настройки не предоставлена")
                raise ValueError("Model cannot be None")
            
            # Проверяем F1-Score исходной модели
            model_wrapper = UniversalModelWrapper(model=model)
            model_wrapper.fit(X_train, y_train)
            y_pred = model_wrapper.predict(X_test)
            original_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            logger.info(f"Исходный F1-Score модели: {original_f1:.4f}")
            
            if original_f1 >= 0.99:
                logger.warning("⚠️ Исходный F1-Score = 1.0. Возможна утечка данных или слишком простая задача.")
            
            # 1. Определение типа модели
            if model_type is None:
                model_type = self._detect_model_type(model)
            logger.info(f"Определен тип модели: {model_type}")
            
            # 2. Применение регуляризации (если возможно)
            try:
                model = self._apply_regularization(model, model_type)
            except Exception as e:
                logger.warning(f"Не удалось применить регуляризацию: {e}")
            
            # 3. Подготовка данных для CV
            cv_splits = list(self.cv.get_cv_splits(X_train, y_train, groups))
            
            # 4. Получение пространства параметров
            original_model = model.model if hasattr(model, 'model') else model
            param_space = self._get_param_space(model_type, original_model)
            
            # 5. Применение информации из предыдущих этапов
            estimator_to_use = self._prepare_model_for_optimization(model)
            param_space_for_optimizer = param_space
            
            # 6. Оптимизация гиперпараметров
            logger.info(f"Запуск оптимизации ({self.optimizer_type})...")
            
            # Проверяем, является ли задача multi-label
            is_multi_label = False
            if hasattr(model, '_is_multi_label'):
                is_multi_label = model._is_multi_label
            elif hasattr(getattr(model, 'model', None), 'is_multi_label'):
                is_multi_label = model.model.is_multi_label
            
            # Выбираем scoring в зависимости от типа задачи
            scoring = self.scoring
            if is_multi_label and 'micro' not in scoring and 'macro' not in scoring:
                scoring = 'f1_micro'  # По умолчанию для multi-label
            
            optimization_result = self.optimizer.optimize(
                estimator=estimator_to_use,
                X=X_train,
                y=y_train,
                param_space=param_space_for_optimizer,
                cv=cv_splits,
                scoring=scoring
            )
            
            # 7. Получение лучшей модели
            best_model = optimization_result.best_estimator_
            best_params = optimization_result.best_params_
            best_score = optimization_result.best_score_
            
            # 8. Обучение лучшей модели на всех данных
            best_model.fit(X_train, y_train)
            
            # 9. Оценка на тестовых данных
            logger.info("Оценка лучшей модели...")
            evaluation_results = self.evaluator.evaluate(best_model, X_test, y_test, X_train, y_train)
            
            # 10. Создание отчета
            tuning_info = {
                'optimizer_type': self.optimizer_type,
                'cv_strategy': self.cv_strategy,
                'best_params': best_params,
                'best_score': best_score,
                'param_space_size': len(param_space_for_optimizer),
                'model_type': model_type,
                'is_multi_label': is_multi_label,
                'task_name': task_name,
                'original_f1': original_f1,
                'tuned_f1': evaluation_results['metrics'].get('f1_macro', 0)
            }
            
            report = self.evaluator.create_comprehensive_report(evaluation_results, tuning_info)
            
            # 11. Сохранение результатов
            results.update({
                'tuning': {
                    'best_model': best_model,
                    'best_params': best_params,
                    'best_score': best_score,
                    'cv_strategy': self.cv_strategy,
                    'optimizer_type': self.optimizer_type,
                    'model_type': model_type,
                    'original_f1': original_f1
                },
                'evaluation': evaluation_results,
                'report': report,
                'success': True
            })
            
            logger.info("Комплексная настройка завершена успешно")
            
        except Exception as e:
            logger.error(f"Ошибка при настройке: {e}")
            logger.error(traceback.format_exc())
            results['error'] = str(e)
            
            # Fallback: простая оценка исходной модели
            try:
                logger.info("Попытка простой оценки исходной модели...")
                model_wrapper = UniversalModelWrapper(model=model)
                model_wrapper.fit(X_train, y_train)
                evaluation_results = self.evaluator.evaluate(model_wrapper, X_test, y_test)
                results['evaluation'] = evaluation_results
                results['report'] = self.evaluator.create_comprehensive_report(evaluation_results)
                results['fallback'] = True
            except Exception as fallback_error:
                logger.error(f"Ошибка при fallback оценке: {fallback_error}")
        
        return results
    
    def _apply_regularization(self, model, model_type):
        """Применение методов регуляризации"""
        if model_type in ['logistic', 'logistic_regression']:
            model = self.regularizer.apply_l1_l2(
                model,
                alpha=self.regularization_params.get('alpha', 0.01),
                l1_ratio=self.regularization_params.get('l1_ratio', 0.5)
            )
        elif model_type in ['random_forest', 'xgboost', 'classical']:
            # Для деревьев регуляризация через параметры
            pass
        elif model_type in ['neural', 'mlp', 'cnn', 'rnn', 'transformer']:
            # Для нейросетей и трансформеров
            model = self.regularizer.apply_dropout(
                model,
                dropout_rate=self.regularization_params.get('dropout_rate', 0.5)
            )
            model = self.regularizer.apply_weight_decay(
                model,
                weight_decay=self.regularization_params.get('weight_decay', 0.01)
            )
            model = self.regularizer.apply_early_stopping(
                model,
                patience=self.regularization_params.get('patience', 10),
                min_delta=self.regularization_params.get('min_delta', 0.001)
            )
        
        return model


# ============================================================
# ИНТЕГРАЦИОННЫЕ ФУНКЦИИ ДЛЯ STREAMLIT
# ============================================================

def create_tuning_pipeline(cv_strategy='stratified', optimizer_type='random',
                          n_trials=50, cv_splits=5, scoring='f1_macro',
                          metrics=None, n_jobs=-1, random_state=42):
    """Создание пайплайна для настройки"""
    return AdvancedModelTuner(
        cv_strategy=cv_strategy,
        cv_splits=cv_splits,
        optimizer_type=optimizer_type,
        n_trials=n_trials,
        scoring=scoring,
        metrics=metrics,
        n_jobs=n_jobs,
        random_state=random_state
    )


def analyze_model_stability(model, X, y, n_bootstrap=10, random_state=42):
    """Анализ стабильности модели"""
    try:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn не установлен")
            
        scores = []
        n_samples = len(X)
        
        for i in range(n_bootstrap):
            indices = np.random.RandomState(random_state + i).choice(
                n_samples, n_samples, replace=True
            )
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Создаем копию модели
            model_wrapper = UniversalModelWrapper(model=model)
            model_wrapper.fit(X_boot, y_boot)
            
            y_pred = model_wrapper.predict(X_boot)
            score = f1_score(y_boot, y_pred, average='macro', zero_division=0)
            scores.append(score)
        
        return {
            'bootstrap_scores': scores,
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'confidence_interval': [
                float(np.percentile(scores, 2.5)),
                float(np.percentile(scores, 97.5))
            ]
        }
    except Exception as e:
        logger.error(f"Ошибка при анализе стабильности: {e}")
        return {
            'bootstrap_scores': [],
            'mean_score': 0,
            'std_score': 0,
            'confidence_interval': [0, 0]
        }


def tune_model_for_task(model, X_train, y_train, X_test, y_test, 
                       stage_outputs=None, task_name=None, balancing_info=None):
    """
    Упрощенная функция для настройки модели для конкретной задачи
    
    Args:
        model: модель из этапа 3 или 4
        X_train, y_train: обучающие данные
        X_test, y_test: тестовые данные
        stage_outputs: выходные данные предыдущих этапов
        task_name: название задачи
        balancing_info: информация о балансировке
        
    Returns:
        Результаты настройки
    """
    logger.info(f"Настройка модели для задачи: {task_name}")
    
    # Создаем тюнер с настройками по умолчанию
    tuner = AdvancedModelTuner(
        cv_strategy='stratified',
        cv_splits=5,
        optimizer_type='random',
        n_trials=30,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42
    )
    
    # Выполняем настройку
    results = tuner.tune_and_evaluate(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_name=task_name,
        stage_outputs=stage_outputs,
        balancing_info=balancing_info
    )
    
    return results


# Для совместимости
CrossValidationManager = {
    'stratified': StratifiedKFoldCV,
    'timeseries': TimeSeriesCV,
    'group': GroupKFoldCV
}

HyperparameterOptimizer = {
    'grid': GridSearchOptimizer,
    'random': RandomSearchOptimizer,
    'bayesian': BayesianOptimizer
}


if __name__ == "__main__":
    print("✅ Расширенный модуль для настройки гиперпараметров (этап 6) успешно загружен")
    print("   Интегрирован с этапами 3-5:")
    print("   - Классические модели (этап 3)")
    print("   - Нейросетевые модели (этап 4)")
    print("   - Методы балансировки (этап 5)")
    print(f"\nДоступные библиотеки:")
    print(f"  Scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
    print(f"  Optuna: {'✅' if OPTUNA_AVAILABLE else '❌'}")
    print(f"  Hyperopt: {'✅' if HYPEROPT_AVAILABLE else '❌'}")