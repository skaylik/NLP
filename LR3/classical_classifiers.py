"""
Модуль для классических методов классификации с поддержкой multi-label.
Этап 3: Реализация классических ML-моделей классификации.
Интегрируется с TextDataProcessor из text_preprocessing.py.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import json
import platform
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

warnings.filterwarnings('ignore')

# Определяем операционную систему
IS_WINDOWS = platform.system() == 'Windows'

# Базовые модели
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC, SVC
    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
    from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Градиентный бустинг
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Метрики и утилиты
if SKLEARN_AVAILABLE:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, classification_report, confusion_matrix,
        roc_auc_score, log_loss, hamming_loss, jaccard_score
    )
    from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
    from sklearn.pipeline import Pipeline

# Вспомогательная функция для преобразования данных
def convert_to_dense(X):
    """
    Преобразование разреженных матриц в плотные
    
    Args:
        X: матрица признаков (может быть разреженной)
        
    Returns:
        Плотная матрица numpy
    """
    if X is None:
        return None
    
    # Проверяем, является ли матрица разреженной
    if hasattr(X, 'toarray'):
        return X.toarray()
    elif hasattr(X, 'todense'):
        return X.todense()
    elif isinstance(X, np.ndarray):
        return X
    elif isinstance(X, pd.DataFrame):
        return X.values
    else:
        # Пробуем преобразовать в numpy array
        try:
            return np.array(X)
        except:
            return X

class ClassicalClassifier:
    """Базовый класс для классических классификаторов с поддержкой multi-label"""
    
    def __init__(self, model_type: str = 'logistic', **kwargs):
        """
        Инициализация классификатора
        
        Args:
            model_type: тип модели ('logistic', 'svm', 'random_forest', 'xgboost',
                         'lightgbm', 'catboost', 'naive_bayes', 'knn')
            **kwargs: гиперпараметры модели
        """
        self.model_type = model_type
        self.model = None
        self.label_encoder = None
        self.is_trained = False
        self.history = {}
        self.task_type = kwargs.get('task_type', 'category')
        self.is_multi_label = kwargs.get('multi_label', False)  # Новый параметр для multi-label
        
        # Для multi-label задач
        self.mlb = None  # MultiLabelBinarizer
        self.n_classes = None
        
        self._init_model(**kwargs)
    
    def _init_model(self, **kwargs):
        """Инициализация модели по типу с учетом multi-label"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn необходим для базовых моделей")
        
        # Для multi-label задач используем специальные обертки
        if self.is_multi_label:
            base_model = None
            
            if self.model_type == 'logistic':
                base_model = LogisticRegression(
                    penalty=kwargs.get('penalty', 'l2'),
                    C=kwargs.get('C', 1.0),
                    solver=kwargs.get('solver', 'lbfgs'),
                    max_iter=kwargs.get('max_iter', 1000),
                    multi_class=kwargs.get('multi_class', 'auto'),
                    random_state=kwargs.get('random_state', 42)
                )
                # Для multi-label используем OneVsRestClassifier
                self.model = OneVsRestClassifier(base_model)
            
            elif self.model_type == 'svm_linear':
                base_model = LinearSVC(
                    C=kwargs.get('C', 1.0),
                    max_iter=kwargs.get('max_iter', 1000),
                    random_state=kwargs.get('random_state', 42)
                )
                self.model = OneVsRestClassifier(base_model)
            
            elif self.model_type == 'svm_rbf':
                base_model = SVC(
                    C=kwargs.get('C', 1.0),
                    kernel='rbf',
                    gamma=kwargs.get('gamma', 'scale'),
                    probability=kwargs.get('probability', True),
                    random_state=kwargs.get('random_state', 42)
                )
                self.model = OneVsRestClassifier(base_model)
            
            elif self.model_type == 'random_forest':
                # RandomForest поддерживает multi-label через multi_output=True
                self.model = RandomForestClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', None),
                    min_samples_split=kwargs.get('min_samples_split', 2),
                    min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                    random_state=kwargs.get('random_state', 42),
                    n_jobs=kwargs.get('n_jobs', -1)
                )
            
            elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                # XGBoost с поддержкой multi-label
                self.model = xgb.XGBClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', 6),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=kwargs.get('random_state', 42),
                    n_jobs=kwargs.get('n_jobs', -1),
                    objective='binary:logistic'  # Для multi-label
                )
            
            elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                # LightGBM с поддержкой multi-label
                self.model = lgb.LGBMClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', -1),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=kwargs.get('random_state', 42),
                    n_jobs=kwargs.get('n_jobs', -1),
                    objective='binary'  # Для multi-label
                )
            
            elif self.model_type == 'catboost' and CATBOOST_AVAILABLE:
                # CatBoost с поддержкой multi-label
                self.model = cb.CatBoostClassifier(
                    iterations=kwargs.get('iterations', 100),
                    depth=kwargs.get('depth', 6),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=kwargs.get('random_state', 42),
                    verbose=kwargs.get('verbose', False),
                    loss_function='Logloss'  # Для multi-label
                )
            
            elif self.model_type == 'naive_bayes':
                # Для multi-label используем OneVsRestClassifier
                base_model = MultinomialNB()
                self.model = OneVsRestClassifier(base_model)
            
            elif self.model_type == 'knn':
                base_model = KNeighborsClassifier(
                    n_neighbors=kwargs.get('n_neighbors', 5),
                    n_jobs=kwargs.get('n_jobs', -1)
                )
                self.model = OneVsRestClassifier(base_model)
            
            else:
                raise ValueError(f"Неподдерживаемый тип модели для multi-label: {self.model_type}")
        
        else:
            # Обычная классификация (single-label)
            if self.model_type == 'logistic':
                self.model = LogisticRegression(
                    penalty=kwargs.get('penalty', 'l2'),
                    C=kwargs.get('C', 1.0),
                    solver=kwargs.get('solver', 'lbfgs'),
                    max_iter=kwargs.get('max_iter', 1000),
                    multi_class=kwargs.get('multi_class', 'auto'),
                    random_state=kwargs.get('random_state', 42)
                )
            
            elif self.model_type == 'svm_linear':
                self.model = LinearSVC(
                    C=kwargs.get('C', 1.0),
                    max_iter=kwargs.get('max_iter', 1000),
                    random_state=kwargs.get('random_state', 42)
                )
            
            elif self.model_type == 'svm_rbf':
                self.model = SVC(
                    C=kwargs.get('C', 1.0),
                    kernel='rbf',
                    gamma=kwargs.get('gamma', 'scale'),
                    probability=kwargs.get('probability', True),
                    random_state=kwargs.get('random_state', 42)
                )
            
            elif self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', None),
                    min_samples_split=kwargs.get('min_samples_split', 2),
                    min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                    random_state=kwargs.get('random_state', 42),
                    n_jobs=kwargs.get('n_jobs', -1)
                )
            
            elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                self.model = xgb.XGBClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', 6),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=kwargs.get('random_state', 42),
                    n_jobs=kwargs.get('n_jobs', -1)
                )
            
            elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                self.model = lgb.LGBMClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', -1),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=kwargs.get('random_state', 42),
                    n_jobs=kwargs.get('n_jobs', -1)
                )
            
            elif self.model_type == 'catboost' and CATBOOST_AVAILABLE:
                self.model = cb.CatBoostClassifier(
                    iterations=kwargs.get('iterations', 100),
                    depth=kwargs.get('depth', 6),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=kwargs.get('random_state', 42),
                    verbose=kwargs.get('verbose', False)
                )
            
            elif self.model_type == 'naive_bayes':
                self.model = GaussianNB()
            
            elif self.model_type == 'knn':
                self.model = KNeighborsClassifier(
                    n_neighbors=kwargs.get('n_neighbors', 5),
                    n_jobs=kwargs.get('n_jobs', -1)
                )
            
            else:
                raise ValueError(f"Неподдерживаемый тип модели: {self.model_type}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Обучение модели
        
        Args:
            X_train: признаки для обучения
            y_train: метки для обучения
            X_val: признаки для валидации (опционально)
            y_val: метки для валидации (опционально)
        """
        # Конвертируем данные в плотные если нужно
        X_train = convert_to_dense(X_train)
        if X_val is not None:
            X_val = convert_to_dense(X_val)
        
        # Для multi-label задач
        if self.is_multi_label:
            # Убедимся, что y_train имеет правильную форму
            # Ожидаем бинарную матрицу или список списков
            if len(y_train.shape) == 1 or (len(y_train.shape) == 2 and y_train.shape[1] == 1):
                # Если y_train - одномерный массив, преобразуем в бинарную матрицу
                try:
                    # Пробуем интерпретировать как список списков
                    if isinstance(y_train[0], list) or (isinstance(y_train[0], np.ndarray) and len(y_train[0].shape) > 0):
                        # Используем MultiLabelBinarizer
                        from sklearn.preprocessing import MultiLabelBinarizer
                        self.mlb = MultiLabelBinarizer()
                        y_train_binary = self.mlb.fit_transform(y_train)
                        if y_val is not None:
                            y_val_binary = self.mlb.transform(y_val)
                    else:
                        # Это обычная классификация, но помеченная как multi-label
                        # Преобразуем в бинарную матрицу с одним классом
                        unique_labels = np.unique(y_train)
                        self.n_classes = len(unique_labels)
                        y_train_binary = np.zeros((len(y_train), self.n_classes))
                        for i, label in enumerate(unique_labels):
                            y_train_binary[:, i] = (y_train == label).astype(int)
                        
                        if y_val is not None:
                            y_val_binary = np.zeros((len(y_val), self.n_classes))
                            for i, label in enumerate(unique_labels):
                                y_val_binary[:, i] = (y_val == label).astype(int)
                except:
                    # Простой случай бинарной классификации
                    self.n_classes = 2
                    y_train_binary = np.column_stack([1 - y_train, y_train])
                    if y_val is not None:
                        y_val_binary = np.column_stack([1 - y_val, y_val])
            else:
                # Уже бинарная матрица
                y_train_binary = y_train
                if y_val is not None:
                    y_val_binary = y_val
            
            self.n_classes = y_train_binary.shape[1] if len(y_train_binary.shape) > 1 else 1
            
            # Обучение модели
            start_time = datetime.now()
            
            # Особые случаи для некоторых моделей
            if self.model_type == 'catboost' and CATBOOST_AVAILABLE and X_val is not None:
                self.model.fit(X_train, y_train_binary,
                              eval_set=(X_val, y_val_binary),
                              verbose=False)
            elif self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
                self.model.fit(X_train, y_train_binary,
                              eval_set=[(X_val, y_val_binary)],
                              verbose=False)
            else:
                self.model.fit(X_train, y_train_binary)
            
            training_time = (datetime.now() - start_time).total_seconds()
        
        else:
            # Обычная классификация (single-label)
            # Кодируем метки если нужно
            if y_train.dtype == object or len(np.unique(y_train)) > 2:
                self.label_encoder = LabelEncoder()
                y_train_encoded = self.label_encoder.fit_transform(y_train)
                if y_val is not None:
                    y_val_encoded = self.label_encoder.transform(y_val)
            else:
                y_train_encoded = y_train
                y_val_encoded = y_val if y_val is not None else None
            
            self.n_classes = len(np.unique(y_train_encoded))
            
            # Обучение модели
            start_time = datetime.now()
            
            # Особые случаи для некоторых моделей
            if self.model_type == 'catboost' and CATBOOST_AVAILABLE and X_val is not None:
                self.model.fit(X_train, y_train_encoded,
                              eval_set=(X_val, y_val_encoded),
                              verbose=False)
            elif self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
                self.model.fit(X_train, y_train_encoded,
                              eval_set=[(X_val, y_val_encoded)],
                              verbose=False)
            else:
                self.model.fit(X_train, y_train_encoded)
            
            training_time = (datetime.now() - start_time).total_seconds()
        
        # Сохраняем историю
        self.history['training_time'] = training_time
        self.history['train_samples'] = len(X_train)
        self.history['train_features'] = X_train.shape[1]
        self.history['n_classes'] = self.n_classes
        if X_val is not None:
            self.history['val_samples'] = len(X_val)
        
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание классов"""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")
        
        # Конвертируем данные в плотные если нужно
        X = convert_to_dense(X)
        
        if self.is_multi_label:
            # Для multi-label возвращаем бинарную матрицу
            if hasattr(self.model, 'predict_proba'):
                # Используем порог 0.5 для преобразования вероятностей в бинарные метки
                y_proba = self.model.predict_proba(X)
                # Для OneVsRestClassifier predict_proba возвращает список массивов
                if isinstance(y_proba, list):
                    y_pred = np.zeros((len(X), len(y_proba)))
                    for i, proba in enumerate(y_proba):
                        y_pred[:, i] = (proba[:, 1] > 0.5).astype(int)
                else:
                    y_pred = (y_proba > 0.5).astype(int)
            else:
                y_pred = self.model.predict(X)
        else:
            # Обычная классификация
            y_pred = self.model.predict(X)
            
            # Декодируем метки если нужно
            if self.label_encoder is not None:
                y_pred = self.label_encoder.inverse_transform(y_pred)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей"""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")
        
        # Конвертируем данные в плотные если нужно
        X = convert_to_dense(X)
        
        if self.is_multi_label:
            # Для multi-label возвращаем вероятности для каждого класса
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)
                # Для OneVsRestClassifier возвращаем список массивов
                return proba
            else:
                # Если нет predict_proba, возвращаем бинарные предсказания как вероятности
                y_pred = self.predict(X)
                return y_pred.astype(float)
        else:
            # Обычная классификация
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)
            else:
                # Для моделей без predict_proba (например, LinearSVC)
                proba = np.zeros((len(X), self.n_classes))
            
            return proba
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict:
        """
        Оценка модели
        
        Args:
            X: признаки
            y_true: истинные метки
            
        Returns:
            словарь с метриками
        """
        y_pred = self.predict(X)
        
        # Для multi-label задач
        if self.is_multi_label:
            # Преобразуем данные к правильному формату
            y_true_binary = self._ensure_binary_matrix(y_true)
            y_pred_binary = self._ensure_binary_matrix(y_pred)
            
            # Проверяем формы
            if y_true_binary.shape != y_pred_binary.shape:
                # Приводим к одинаковой размерности
                n_samples = max(y_true_binary.shape[0], y_pred_binary.shape[0])
                n_classes = max(y_true_binary.shape[1] if len(y_true_binary.shape) > 1 else 1, 
                            y_pred_binary.shape[1] if len(y_pred_binary.shape) > 1 else 1)
                
                y_true_fixed = np.zeros((n_samples, n_classes), dtype=int)
                y_pred_fixed = np.zeros((n_samples, n_classes), dtype=int)
                
                if len(y_true_binary.shape) > 1:
                    y_true_fixed[:y_true_binary.shape[0], :y_true_binary.shape[1]] = y_true_binary
                else:
                    y_true_fixed[:len(y_true_binary), 0] = y_true_binary
                
                if len(y_pred_binary.shape) > 1:
                    y_pred_fixed[:y_pred_binary.shape[0], :y_pred_binary.shape[1]] = y_pred_binary
                else:
                    y_pred_fixed[:len(y_pred_binary), 0] = y_pred_binary
                
                y_true_binary = y_true_fixed
                y_pred_binary = y_pred_fixed
            
            # Вычисляем метрики с защитой от ошибок
            try:
                # Подмножественная точность (Subset Accuracy)
                accuracy = accuracy_score(y_true_binary, y_pred_binary)
                
                # Hamming Loss
                hamming = hamming_loss(y_true_binary, y_pred_binary)
                
                # Jaccard Score (среднее по образцам)
                jaccard = jaccard_score(y_true_binary, y_pred_binary, 
                                    average='samples', zero_division=0)
                
                # Precision и Recall (micro average)
                try:
                    precision_micro = precision_score(y_true_binary, y_pred_binary, 
                                                    average='micro', zero_division=0)
                    recall_micro = recall_score(y_true_binary, y_pred_binary, 
                                            average='micro', zero_division=0)
                    f1_micro = f1_score(y_true_binary, y_pred_binary, 
                                    average='micro', zero_division=0)
                except:
                    precision_micro = 0
                    recall_micro = 0
                    f1_micro = 0
                
                # Precision и Recall (macro average)
                try:
                    precision_macro = precision_score(y_true_binary, y_pred_binary, 
                                                    average='macro', zero_division=0)
                    recall_macro = recall_score(y_true_binary, y_pred_binary, 
                                            average='macro', zero_division=0)
                    f1_macro = f1_score(y_true_binary, y_pred_binary, 
                                    average='macro', zero_division=0)
                except:
                    precision_macro = 0
                    recall_macro = 0
                    f1_macro = 0
                
                # Precision и Recall (samples average)
                try:
                    precision_samples = precision_score(y_true_binary, y_pred_binary, 
                                                    average='samples', zero_division=0)
                    recall_samples = recall_score(y_true_binary, y_pred_binary, 
                                                average='samples', zero_division=0)
                    f1_samples = f1_score(y_true_binary, y_pred_binary, 
                                        average='samples', zero_division=0)
                except:
                    precision_samples = 0
                    recall_samples = 0
                    f1_samples = 0
                
                # Основная метрика для сравнения
                main_f1 = f1_micro if f1_micro > 0 else f1_samples
                
                metrics = {
                    'accuracy': accuracy,
                    'hamming_loss': hamming,
                    'jaccard_score': jaccard,
                    'precision_samples': precision_samples,
                    'recall_samples': recall_samples,
                    'f1_samples': f1_samples,
                    'precision_macro': precision_macro,
                    'recall_macro': recall_macro,
                    'f1_macro': f1_macro,
                    'precision_micro': precision_micro,
                    'recall_micro': recall_micro,
                    'f1_micro': f1_micro,
                    'precision': precision_micro,  # Для отображения
                    'recall': recall_micro,        # Для отображения
                    'f1': main_f1,
                    'is_multi_label': True,
                    'success': True
                }
                
            except Exception as e:
                # Резервный вариант для очень маленьких данных
                accuracy = accuracy_score(y_true_binary.flatten(), y_pred_binary.flatten())
                metrics = {
                    'accuracy': accuracy,
                    'precision': accuracy,
                    'recall': accuracy,
                    'f1': accuracy,
                    'is_multi_label': True,
                    'success': True,
                    'note': 'Использованы упрощенные метрики'
                }
        
        else:
            # Обычная классификация (single-label)
            # Кодируем метки если нужно
            if y_true.dtype == object or len(np.unique(y_true)) > 2:
                if self.label_encoder is not None:
                    # Используем обученный label_encoder
                    try:
                        y_true_encoded = self.label_encoder.transform(y_true)
                        y_pred_encoded = self.label_encoder.transform(y_pred)
                    except ValueError:
                        # Если появились новые метки, создаем новый encoder
                        le = LabelEncoder()
                        y_true_encoded = le.fit_transform(y_true)
                        y_pred_encoded = le.transform(y_pred)
                else:
                    le = LabelEncoder()
                    y_true_encoded = le.fit_transform(y_true)
                    y_pred_encoded = le.transform(y_pred)
            else:
                y_true_encoded = y_true
                y_pred_encoded = y_pred
            
            # Получаем все уникальные метки для матрицы ошибок
            all_labels = np.unique(np.concatenate([y_true_encoded, y_pred_encoded]))
            
            # Вычисляем метрики
            try:
                accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
                precision = precision_score(y_true_encoded, y_pred_encoded, 
                                        average='weighted', zero_division=0)
                recall = recall_score(y_true_encoded, y_pred_encoded, 
                                    average='weighted', zero_division=0)
                f1 = f1_score(y_true_encoded, y_pred_encoded, 
                            average='weighted', zero_division=0)
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'is_multi_label': False,
                    'success': True
                }
                
                # Подробный отчет
                try:
                    metrics['report'] = classification_report(
                        y_true_encoded, y_pred_encoded, 
                        output_dict=True,
                        zero_division=0
                    )
                except:
                    metrics['report'] = {}
                
                # Матрица ошибок
                try:
                    cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=all_labels)
                    metrics['confusion_matrix'] = cm.tolist()
                    metrics['unique_labels'] = all_labels.tolist()
                except:
                    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
                    metrics['confusion_matrix'] = cm.tolist()
                
            except Exception as e:
                # Минимальные метрики в случае ошибки
                accuracy = np.mean(y_true_encoded == y_pred_encoded) if len(y_true_encoded) > 0 else 0
                metrics = {
                    'accuracy': accuracy,
                    'precision': accuracy,
                    'recall': accuracy,
                    'f1': accuracy,
                    'is_multi_label': False,
                    'success': True,
                    'error': str(e)[:100]
                }
        
        self.history['evaluation'] = metrics
        return metrics

    def _ensure_binary_matrix(self, y):
            """
            Преобразование меток в бинарную матрицу для multi-label классификации
            
            Args:
                y: метки (различные форматы)
                
            Returns:
                бинарная матрица (n_samples, n_classes)
            """
            if y is None or len(y) == 0:
                return np.zeros((0, 1), dtype=int)
            
            # Если уже бинарная матрица
            if len(y.shape) > 1 and y.shape[1] > 1:
                return y.astype(int)
            
            # Если одномерный массив
            if len(y.shape) == 1 or y.shape[1] == 1:
                y_flat = y.flatten() if hasattr(y, 'flatten') else y
                
                # Используем MultiLabelBinarizer если доступен
                if hasattr(self, 'mlb') and self.mlb is not None:
                    try:
                        # Предполагаем, что y_flat содержит списки меток
                        if isinstance(y_flat[0], (list, np.ndarray)):
                            return self.mlb.transform(y_flat)
                        else:
                            # Если не списки, преобразуем в списки
                            return self.mlb.transform([[label] for label in y_flat])
                    except Exception as e:
                        # Если преобразование не удалось, создаем простую матрицу
                        pass
                
                # Создаем простую бинарную матрицу
                unique_labels = np.unique(y_flat)
                n_classes = len(unique_labels)
                if n_classes == 0:
                    return np.zeros((len(y_flat), 1), dtype=int)
                
                binary_matrix = np.zeros((len(y_flat), n_classes), dtype=int)
                
                for i, label in enumerate(unique_labels):
                    binary_matrix[:, i] = (y_flat == label).astype(int)
                
                return binary_matrix
            
            return y.astype(int)
    
    def save(self, filepath: str):
        """Сохранение модели"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'label_encoder': self.label_encoder,
                'is_trained': self.is_trained,
                'history': self.history,
                'task_type': self.task_type,
                'is_multi_label': self.is_multi_label,
                'mlb': self.mlb,
                'n_classes': self.n_classes
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'ClassicalClassifier':
        """Загрузка модели"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls(model_type=data['model_type'])
        classifier.model = data['model']
        classifier.label_encoder = data['label_encoder']
        classifier.is_trained = data['is_trained']
        classifier.history = data['history']
        classifier.task_type = data.get('task_type', 'category')
        classifier.is_multi_label = data.get('is_multi_label', False)
        classifier.mlb = data.get('mlb')
        classifier.n_classes = data.get('n_classes')
        
        return classifier


# ============================================================
# КОМПАРАТОР МОДЕЛЕЙ
# ============================================================
class ModelComparator:
    """Класс для сравнения и выбора лучших моделей"""
    
    def __init__(self, models_config=None):
        self.models_config = models_config or []
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
    
    def add_model(self, model_name: str, model: ClassicalClassifier):
        """Добавление модели в компаратор"""
        self.models[model_name] = model
    
    def train_and_compare(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray, 
                         task_name: str = 'category'):
        """
        Обучение и сравнение всех моделей
        
        Args:
            X_train: признаки для обучения
            y_train: метки для обучения
            X_val: признаки для валидации
            y_val: метки для валидации
            task_name: название задачи (для определения типа)
            
        Returns:
            словарь с результатами
        """
        results = {}
        is_multi_label = task_name == 'categories'
        
        for model_config in self.models_config:
            model_name = model_config.get('name', 'Unknown')
            
            try:
                # Определяем параметры для multi-label
                if is_multi_label:
                    model_config['multi_label'] = True
                
                # Создаем классификатор
                classifier = ClassicalClassifier(**model_config)
                
                # Обучение
                classifier.fit(X_train, y_train, X_val, y_val)
                
                # Оценка на валидации
                val_metrics = classifier.evaluate(X_val, y_val)
                score = val_metrics.get('f1', 0)
                
                # Сохраняем результаты
                results[model_name] = {
                    'model': classifier,
                    'metrics': val_metrics,
                    'score': score,
                    'config': model_config
                }
                
                # Обновляем лучшую модель
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = classifier
                    self.best_model_name = model_name
                
                print(f"✅ {model_name}: F1 = {score:.4f}")
                
            except Exception as e:
                print(f"❌ Ошибка при обучении {model_name}: {e}")
                results[model_name] = {
                    'error': str(e),
                    'score': 0
                }
        
        self.results = results
        return results
    
    def get_best_model(self):
        """Получение лучшей модели"""
        return self.best_model
    
    def get_results_table(self):
        """Получение таблицы результатов"""
        data = []
        for model_name, result in self.results.items():
            if 'error' not in result:
                metrics = result.get('metrics', {})
                data.append({
                    'Model': model_name,
                    'F1-Score': result.get('score', 0),
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 'N/A'),
                    'Recall': metrics.get('recall', 'N/A'),
                    'Type': 'Multi-label' if result.get('config', {}).get('multi_label', False) else 'Single-label',
                    'Training Time (s)': result.get('model', {}).history.get('training_time', 0) if hasattr(result.get('model'), 'history') else 'N/A',
                    'Status': '✅ Успешно'
                })
            else:
                data.append({
                    'Model': model_name,
                    'F1-Score': 0,
                    'Accuracy': 0,
                    'Precision': 'N/A',
                    'Recall': 'N/A',
                    'Type': 'Error',
                    'Training Time (s)': 'N/A',
                    'Status': f'❌ {result.get("error", "Unknown error")}'
                })
        
        return pd.DataFrame(data).sort_values('F1-Score', ascending=False)


# ============================================================
# АНСАМБЛЕВЫЕ МОДЕЛИ
# ============================================================
class EnsembleClassifier:
    """Класс для создания ансамблей моделей"""
    
    def __init__(self, ensemble_type='voting', estimators=None):
        self.ensemble_type = ensemble_type
        self.estimators = estimators or []
        self.model = None
        self.is_trained = False
    
    def add_estimator(self, name: str, estimator):
        """Добавление оценщика в ансамбль"""
        self.estimators.append((name, estimator))
    
    def fit(self, X_train, y_train):
        """Обучение ансамбля"""
        if self.ensemble_type == 'voting':
            self.model = VotingClassifier(estimators=self.estimators, voting='soft')
        elif self.ensemble_type == 'stacking':
            self.model = StackingClassifier(estimators=self.estimators, final_estimator=LogisticRegression())
        elif self.ensemble_type == 'bagging':
            self.model = BaggingClassifier(estimator=self.estimators[0][1] if self.estimators else DecisionTreeClassifier())
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X):
        """Предсказание"""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Предсказание вероятностей"""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")
        return self.model.predict_proba(X)


# ============================================================
# УТИЛИТЫ
# ============================================================
def create_model_configs(task_type='category'):
    """
    Создание конфигураций моделей для разных задач
    
    Args:
        task_type: тип задачи ('category', 'sentiment', 'categories')
        
    Returns:
        список конфигураций моделей
    """
    is_multi_label = (task_type == 'categories')
    
    base_configs = [
        {
            'name': 'Logistic Regression',
            'model_type': 'logistic',
            'penalty': 'l2',
            'C': 1.0,
            'max_iter': 1000,
            'multi_label': is_multi_label
        },
        {
            'name': 'Random Forest',
            'model_type': 'random_forest',
            'n_estimators': 100,
            'max_depth': None,
            'multi_label': is_multi_label
        },
        {
            'name': 'SVM (linear)',
            'model_type': 'svm_linear',
            'C': 1.0,
            'max_iter': 1000,
            'multi_label': is_multi_label
        },
        {
            'name': 'SVM (RBF)',
            'model_type': 'svm_rbf',
            'C': 1.0,
            'gamma': 'scale',
            'multi_label': is_multi_label
        }
    ]
    
    if XGBOOST_AVAILABLE:
        base_configs.append({
            'name': 'XGBoost',
            'model_type': 'xgboost',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'multi_label': is_multi_label
        })
    
    if LIGHTGBM_AVAILABLE:
        base_configs.append({
            'name': 'LightGBM',
            'model_type': 'lightgbm',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'multi_label': is_multi_label
        })
    
    if CATBOOST_AVAILABLE:
        base_configs.append({
            'name': 'CatBoost',
            'model_type': 'catboost',
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'multi_label': is_multi_label
        })
    
    base_configs.extend([
        {
            'name': 'Naive Bayes',
            'model_type': 'naive_bayes',
            'multi_label': is_multi_label
        },
        {
            'name': 'K-Nearest Neighbors',
            'model_type': 'knn',
            'n_neighbors': 5,
            'multi_label': is_multi_label
        }
    ])
    
    return base_configs


def train_all_tasks(X_train, y_train_all, X_val, y_val_all, task_names=None):
    """
    Обучение моделей для всех типов задач
    
    Args:
        X_train: признаки для обучения
        y_train_all: словарь {task_name: y_train}
        X_val: признаки для валидации
        y_val_all: словарь {task_name: y_val}
        task_names: список названий задач
        
    Returns:
        словарь с результатами для каждой задачи
    """
    if task_names is None:
        task_names = list(y_train_all.keys())
    
    all_results = {}
    
    for task_name in task_names:
        if task_name not in y_train_all or task_name not in y_val_all:
            continue
        
        print(f"\n=== Обучение для задачи: {task_name} ===")
        
        # Определяем конфигурации моделей
        model_configs = create_model_configs(task_type=task_name)
        
        # Создаем компаратор
        comparator = ModelComparator(models_config=model_configs)
        
        # Обучение и сравнение
        results = comparator.train_and_compare(
            X_train, y_train_all[task_name],
            X_val, y_val_all[task_name],
            task_name=task_name
        )
        
        all_results[task_name] = {
            'comparator': comparator,
            'results': results,
            'best_model': comparator.get_best_model(),
            'best_score': comparator.best_score,
            'best_model_name': comparator.best_model_name
        }
    
    return all_results

# ============================================================
# ФУНКЦИИ ДЛЯ ИНТЕГРАЦИИ СО STREAMLIT
# ============================================================

def get_model_for_interactive(model, task_type):
    """Подготовка модели для интерактивного анализа (этап 8)"""
    class InteractiveModelWrapper:
        def __init__(self, model, task_type):
            self.model = model
            self.task_type = task_type
            self.is_multi_label = (task_type == 'categories')
            self.model_name = model.__class__.__name__
            
        def predict(self, X):
            return self.model.predict(X)
            
        def predict_proba(self, X):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                # Для моделей без predict_proba возвращаем фиктивные значения
                y_pred = self.model.predict(X)
                if self.is_multi_label:
                    return y_pred.astype(float)
                else:
                    n_classes = len(np.unique(y_pred))
                    proba = np.zeros((len(y_pred), n_classes))
                    for i, pred in enumerate(y_pred):
                        proba[i, pred] = 1.0
                    return proba
        
        def get_info(self):
            return {
                'type': 'classical',
                'task_type': self.task_type,
                'is_multi_label': self.is_multi_label,
                'model_name': self.model_name
            }
    
    return InteractiveModelWrapper(model, task_type)


# ============================================================
# ФУНКЦИИ ДЛЯ ОБУЧЕНИЯ ВСЕХ ЗАДАЧ (для Streamlit)
# ============================================================

def train_all_tasks_for_streamlit(X_train, y_train_all, X_val, y_val_all, task_names=None):
    """
    Обучение моделей для всех типов задач (специально для Streamlit)
    
    Args:
        X_train: признаки для обучения
        y_train_all: словарь {task_name: y_train}
        X_val: признаки для валидации
        y_val_all: словарь {task_name: y_val}
        task_names: список названий задач
        
    Returns:
        словарь с результатами для каждой задачи
    """
    if task_names is None:
        task_names = list(y_train_all.keys())
    
    all_results = {}
    
    for task_name in task_names:
        if task_name not in y_train_all:
            continue
        
        print(f"\n=== Обучение для задачи: {task_name} ===")
        
        # Определяем конфигурации моделей
        model_configs = create_model_configs(task_type=task_name)
        
        # Создаем компаратор
        comparator = ModelComparator(models_config=model_configs)
        
        # Получаем данные для этой задачи
        y_train_task = y_train_all[task_name]
        y_val_task = y_val_all.get(task_name) if y_val_all else None
        
        # Обучение и сравнение
        try:
            results = comparator.train_and_compare(
                X_train, y_train_task,
                X_val, y_val_task,
                task_name=task_name
            )
            
            all_results[task_name] = {
                'comparator': comparator,
                'results': results,
                'best_model': comparator.get_best_model(),
                'best_score': comparator.best_score,
                'best_model_name': comparator.best_model_name,
                'configs': model_configs
            }
            
            print(f"✅ Задача '{task_name}' завершена. Лучшая модель: {comparator.best_model_name} (F1: {comparator.best_score:.4f})")
            
        except Exception as e:
            print(f"❌ Ошибка при обучении задачи '{task_name}': {e}")
            all_results[task_name] = {
                'error': str(e),
                'best_score': 0
            }
    
    return all_results