"""
Модуль для нейросетевых классификаторов с поддержкой multi-label.
Этап 4: Нейросетевые и трансформерные модели.
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime
from collections import Counter, defaultdict
import traceback

warnings.filterwarnings('ignore')

# ============================================================================
# ИМПОРТ БИБЛИОТЕК
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, classification_report, confusion_matrix,
        roc_auc_score, hamming_loss, jaccard_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        AdamW, get_linear_schedule_with_warmup
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ
# ============================================================================

class TextDataset(Dataset):
    """Датасет для текстовых данных с поддержкой multi-label"""
    
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128, is_multi_label=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_multi_label = is_multi_label
        
        if labels is not None:
            if is_multi_label:
                # Для multi-label: y - это бинарная матрица
                if isinstance(labels[0], (list, np.ndarray)):
                    self.labels_tensor = torch.tensor(labels, dtype=torch.float32)
                else:
                    # Преобразуем в бинарную матрицу
                    if isinstance(labels[0], str):
                        # Предполагаем, что строки разделены запятыми
                        label_lists = [label.split(',') for label in labels]
                        mlb = MultiLabelBinarizer()
                        self.labels_tensor = torch.tensor(mlb.fit_transform(label_lists), dtype=torch.float32)
                    else:
                        self.labels_tensor = torch.tensor(labels, dtype=torch.float32)
            else:
                # Для single-label
                if isinstance(labels[0], str):
                    self.label_encoder = LabelEncoder()
                    self.labels_tensor = torch.tensor(
                        self.label_encoder.fit_transform(labels), 
                        dtype=torch.long
                    )
                    self.num_classes = len(self.label_encoder.classes_)
                else:
                    self.label_encoder = None
                    self.labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels_tensor = None
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            item = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }
            
            if self.labels_tensor is not None:
                item['labels'] = self.labels_tensor[idx]
            
            return item
        else:
            item = {'text': text}
            if self.labels_tensor is not None:
                item['labels'] = self.labels_tensor[idx]
            return item


class EarlyStopping:
    """Ранняя остановка"""
    
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def restore(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# ============================================================================
# БАЗОВЫЙ КЛАСС НЕЙРОСЕТЕВЫХ КЛАССИФИКАТОРОВ
# ============================================================================

class NeuralClassifier:
    """Базовый класс для нейросетевых классификаторов с поддержкой multi-label"""
    
    def __init__(self, model_name="neural_classifier", device=None, is_multi_label=False):
        self.model_name = model_name
        self.model = None
        self.device = self._get_device(device)
        self.is_trained = False
        self.is_multi_label = is_multi_label
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
        self.label_encoder = None
        self.mlb = None  # Для multi-label
        self.classes_ = None
        self.training_time = 0
        self.num_classes = None
        
        print(f"🧠 Инициализация '{model_name}' (multi_label: {is_multi_label})")
        print(f"   Устройство: {self.device}")
    
    def _get_device(self, device=None):
        if device is not None:
            return torch.device(device)
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
        return torch.device('cpu')
    
    def build_model(self):
        """Построение модели (должен быть реализован в подклассах)"""
        raise NotImplementedError
    
    def _create_dataset(self, X, y=None):
        if isinstance(X, np.ndarray) and len(X.shape) == 2:
            # Матрица признаков
            X_tensor = torch.tensor(X, dtype=torch.float32)
            if y is not None:
                if self.is_multi_label:
                    y_tensor = torch.tensor(y, dtype=torch.float32)
                else:
                    y_tensor = torch.tensor(y, dtype=torch.long)
                return TensorDataset(X_tensor, y_tensor)
            else:
                return TensorDataset(X_tensor)
        elif isinstance(X, list) and all(isinstance(x, str) for x in X):
            # Тексты
            return TextDataset(X, y, tokenizer=getattr(self, 'tokenizer', None), 
                              is_multi_label=self.is_multi_label)
        else:
            raise ValueError(f"Неподдерживаемый тип X: {type(X)}")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=10, batch_size=32, learning_rate=1e-3,
            verbose=True, patience=5, **kwargs):
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch не установлен")
        
        start_time = datetime.now()
        
        # Подготовка меток
        if self.is_multi_label:
            # Для multi-label: y_train уже должна быть бинарной матрицей
            if isinstance(y_train[0], (list, np.ndarray)):
                y_train_encoded = y_train
            else:
                # Пытаемся преобразовать
                if isinstance(y_train[0], str):
                    label_lists = [label.split(',') for label in y_train]
                    self.mlb = MultiLabelBinarizer()
                    y_train_encoded = self.mlb.fit_transform(label_lists)
                else:
                    y_train_encoded = y_train
            self.num_classes = y_train_encoded.shape[1] if len(y_train_encoded.shape) > 1 else 1
        else:
            # Для single-label
            if isinstance(y_train[0], str):
                self.label_encoder = LabelEncoder()
                y_train_encoded = self.label_encoder.fit_transform(y_train)
                self.classes_ = self.label_encoder.classes_
            else:
                y_train_encoded = y_train
                self.classes_ = np.unique(y_train)
            self.num_classes = len(self.classes_)
        
        # Построение модели если еще не построена
        if self.model is None:
            self.build_model()
        
        # Подготовка данных
        train_dataset = self._create_dataset(X_train, y_train_encoded)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            if self.is_multi_label:
                if self.mlb is not None and isinstance(y_val[0], str):
                    y_val_encoded = self.mlb.transform([label.split(',') for label in y_val])
                else:
                    y_val_encoded = y_val
            else:
                if self.label_encoder is not None and isinstance(y_val[0], str):
                    y_val_encoded = self.label_encoder.transform(y_val)
                else:
                    y_val_encoded = y_val
            
            val_dataset = self._create_dataset(X_val, y_val_encoded)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Оптимизатор и функция потерь
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        if self.is_multi_label:
            criterion = nn.BCEWithLogitsLoss()  # Бинарная кросс-энтропия для multi-label
        else:
            criterion = nn.CrossEntropyLoss()  # Для single-label
        
        # Ранняя остановка
        early_stopping = EarlyStopping(patience=patience)
        
        # Перемещение модели на устройство
        self.model.to(self.device)
        
        # Обучение
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            all_preds = []
            all_labels = []
            
            for batch_idx, batch in enumerate(train_loader):
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                elif isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Статистика
                train_loss += loss.item()
                
                # Для multi-label используем порог 0.5
                if self.is_multi_label:
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    train_correct += (preds == labels).sum().item() / self.num_classes
                    train_total += labels.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Расчет метрик
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
            
            if SKLEARN_AVAILABLE:
                if self.is_multi_label:
                    train_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
                else:
                    train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            else:
                train_f1 = 0.0
            
            # Валидация
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                val_all_preds = []
                val_all_labels = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        if isinstance(batch, (list, tuple)):
                            inputs, labels = batch
                            inputs = inputs.to(self.device)
                            labels = labels.to(self.device)
                        elif isinstance(batch, dict):
                            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                            labels = batch['labels'].to(self.device)
                        
                        if isinstance(inputs, dict):
                            outputs = self.model(**inputs)
                        else:
                            outputs = self.model(inputs)
                        
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        if self.is_multi_label:
                            preds = (torch.sigmoid(outputs) > 0.5).float()
                            val_correct += (preds == labels).sum().item() / self.num_classes
                            val_total += labels.size(0)
                            val_all_preds.extend(preds.cpu().numpy())
                            val_all_labels.extend(labels.cpu().numpy())
                        else:
                            _, predicted = torch.max(outputs.data, 1)
                            val_total += labels.size(0)
                            val_correct += (predicted == labels).sum().item()
                            val_all_preds.extend(predicted.cpu().numpy())
                            val_all_labels.extend(labels.cpu().numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
                
                if SKLEARN_AVAILABLE:
                    if self.is_multi_label:
                        val_f1 = f1_score(val_all_labels, val_all_preds, average='micro', zero_division=0)
                    else:
                        val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
                
                # Ранняя остановка
                if early_stopping(avg_val_loss, self.model):
                    if verbose:
                        print(f"Ранняя остановка на эпохе {epoch+1}")
                    break
            else:
                avg_val_loss = None
                val_acc = None
                val_f1 = None
            
            # Сохранение истории
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            
            if val_loader is not None:
                self.history['val_loss'].append(avg_val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_f1'].append(val_f1)
            
            # Вывод информации
            if verbose:
                log_msg = f"Эпоха {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}"
                if val_loader is not None:
                    log_msg += f", Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}"
                print(log_msg)
        
        # Восстановление лучших весов
        if val_loader is not None:
            early_stopping.restore(self.model)
        
        self.training_time = (datetime.now() - start_time).total_seconds()
        self.is_trained = True
        
        if verbose:
            print(f"✅ Обучение завершено за {self.training_time:.2f} секунд")
        
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Модель не обучена!")
        
        self.model.eval()
        dataset = self._create_dataset(X, None)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    outputs = self.model(inputs)
                elif isinstance(batch, dict):
                    if 'labels' in batch:
                        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    else:
                        inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**inputs)
                
                if self.is_multi_label:
                    # Для multi-label используем порог 0.5
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    all_predictions.extend(preds.cpu().numpy())
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
        
        predictions = np.array(all_predictions)
        
        # Декодирование меток
        if not self.is_multi_label and self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Модель не обучена!")
        
        self.model.eval()
        dataset = self._create_dataset(X, None)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    outputs = self.model(inputs)
                elif isinstance(batch, dict):
                    if 'labels' in batch:
                        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    else:
                        inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**inputs)
                
                if self.is_multi_label:
                    # Для multi-label: сигмоида для каждого класса
                    probs = torch.sigmoid(outputs)
                else:
                    # Для single-label: softmax по классам
                    probs = F.softmax(outputs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_probs)
    
    def evaluate(self, X, y_true):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn требуется для вычисления метрик")
        
        y_pred = self.predict(X)
        
        # Преобразуем y_true в правильный формат
        if self.is_multi_label:
            # Для multi-label
            if isinstance(y_true[0], (list, np.ndarray)):
                y_true_binary = y_true
            else:
                # Если метки строки, преобразуем
                if isinstance(y_true[0], str):
                    # Предполагаем, что строки разделены запятыми
                    label_lists = [label.split(',') for label in y_true]
                    if hasattr(self, 'mlb') and self.mlb is not None:
                        y_true_binary = self.mlb.transform(label_lists)
                    else:
                        # Создаем временный MultiLabelBinarizer
                        mlb = MultiLabelBinarizer()
                        y_true_binary = mlb.fit_transform(label_lists)
                else:
                    y_true_binary = y_true
            
            # Убедимся, что y_pred имеет правильную форму
            if len(y_pred.shape) == 1 or y_pred.shape[1] != y_true_binary.shape[1]:
                # Если y_pred - бинарная матрица, но формы не совпадают
                try:
                    y_pred_binary = y_pred.reshape(y_true_binary.shape)
                except:
                    # Если не получается, создаем простую матрицу
                    y_pred_binary = np.zeros_like(y_true_binary)
                    y_pred_binary[:, 0] = y_pred
            else:
                y_pred_binary = y_pred
            
            # Вычисление multi-label метрик
            try:
                accuracy = accuracy_score(y_true_binary, y_pred_binary)
                precision = precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'is_multi_label': True,
                    'success': True
                }
                
            except Exception as e:
                metrics = {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'is_multi_label': True,
                    'error': str(e),
                    'success': False
                }
        
        else:
            # Для single-label
            # Преобразуем y_true в числа если нужно
            if isinstance(y_true[0], str):
                if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                    try:
                        y_true_encoded = self.label_encoder.transform(y_true)
                    except:
                        # Если появились новые метки
                        le = LabelEncoder()
                        y_true_encoded = le.fit_transform(y_true)
                else:
                    le = LabelEncoder()
                    y_true_encoded = le.fit_transform(y_true)
            else:
                y_true_encoded = y_true
            
            # Преобразуем y_pred в числа если нужно
            if isinstance(y_pred[0], str):
                if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                    try:
                        y_pred_encoded = self.label_encoder.transform(y_pred)
                    except:
                        le = LabelEncoder()
                        y_pred_encoded = le.transform(y_pred)
                else:
                    le = LabelEncoder()
                    y_pred_encoded = le.transform(y_pred)
            else:
                y_pred_encoded = y_pred
            
            # Вычисляем метрики
            accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
            precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
            recall = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
            f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
            
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
                unique_labels = np.unique(np.concatenate([y_true_encoded, y_pred_encoded]))
                cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=unique_labels)
                metrics['confusion_matrix'] = cm.tolist()
            except:
                metrics['confusion_matrix'] = []
        
        self.history['evaluation'] = metrics
        return metrics


# ============================================================================
# МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН (MLP)
# ============================================================================

class SimpleNNClassifier(NeuralClassifier):
    """Многослойный персептрон"""
    
    def __init__(self, input_dim=None, hidden_dims=[256, 128], dropout=0.3,
                 activation='relu', use_batch_norm=True, **kwargs):
        super().__init__(model_name="MLP_Classifier", **kwargs)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.use_batch_norm = use_batch_norm
    
    def build_model(self):
        if self.input_dim is None or self.num_classes is None:
            raise ValueError("input_dim и num_classes должны быть заданы")
        
        layers = []
        prev_dim = self.input_dim
        
        # Создание скрытых слоев
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        # Выходной слой
        if self.is_multi_label:
            # Для multi-label: нет активации (будет использоваться BCEWithLogitsLoss)
            layers.append(nn.Linear(prev_dim, self.num_classes))
        else:
            # Для single-label: линейный слой (softmax будет в функции потерь)
            layers.append(nn.Linear(prev_dim, self.num_classes))
        
        self.model = nn.Sequential(*layers)
        
        print(f"✅ Построена MLP: вход {self.input_dim}, скрытые {self.hidden_dims}, выход {self.num_classes}")


# ============================================================================
# CNN ДЛЯ ТЕКСТА
# ============================================================================

class CNNClassifier(NeuralClassifier):
    """CNN для текста с 1D-свертками"""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=200,
                 num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5, **kwargs):
        super().__init__(model_name="CNN_Classifier", **kwargs)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        
        self.tokenizer = None
    
    def build_model(self):
        if self.vocab_size is None or self.num_classes is None:
            raise ValueError("vocab_size и num_classes должны быть заданы")
        
        class TextCNN(nn.Module):
            def __init__(self, vocab_size, embedding_dim, num_filters, 
                        filter_sizes, num_classes, dropout, is_multi_label):
                super().__init__()
                self.is_multi_label = is_multi_label
                
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.convs = nn.ModuleList([
                    nn.Conv1d(embedding_dim, num_filters, fs, padding=fs//2) 
                    for fs in filter_sizes
                ])
                self.dropout = nn.Dropout(dropout)
                total_filters = num_filters * len(filter_sizes)
                
                if is_multi_label:
                    self.fc = nn.Linear(total_filters, num_classes)
                else:
                    self.fc = nn.Linear(total_filters, num_classes)
                
                self._init_weights()
            
            def _init_weights(self):
                nn.init.xavier_uniform_(self.embedding.weight)
                for conv in self.convs:
                    nn.init.xavier_uniform_(conv.weight)
                nn.init.xavier_uniform_(self.fc.weight)
            
            def forward(self, x):
                if x.dtype != torch.long:
                    x = x.long()
                
                embedded = self.embedding(x)
                embedded = embedded.permute(0, 2, 1)
                
                pooled_outputs = []
                for conv in self.convs:
                    conv_out = conv(embedded)
                    conv_out = F.relu(conv_out)
                    pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                    pooled_outputs.append(pooled)
                
                cat = torch.cat(pooled_outputs, dim=1)
                cat = self.dropout(cat)
                output = self.fc(cat)
                
                return output
        
        self.model = TextCNN(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_filters=self.num_filters,
            filter_sizes=self.filter_sizes,
            num_classes=self.num_classes,
            dropout=self.dropout,
            is_multi_label=self.is_multi_label
        )
        
        print(f"✅ Построена CNN: {len(self.filter_sizes)} фильтров размеров {self.filter_sizes}")
    
    def create_tokenizer(self, texts):
        import re
        from collections import Counter
        
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.most_common(self.vocab_size-2)]
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        def tokenize(text):
            words = re.findall(r'\b\w+\b', text.lower())
            indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
            return indices[:self.max_length]
        
        self.tokenizer = tokenize
        self.vocab_size = len(vocab)
        return tokenize
    
    def prepare_texts(self, texts):
        if self.tokenizer is None:
            self.create_tokenizer(texts)
        
        tokenized = [self.tokenizer(text) for text in texts]
        
        padded = []
        for tokens in tokenized:
            if len(tokens) < self.max_length:
                padded_tokens = tokens + [0] * (self.max_length - len(tokens))
            else:
                padded_tokens = tokens[:self.max_length]
            padded.append(padded_tokens)
        
        return np.array(padded, dtype=np.int64)


# ============================================================================
# RNN (LSTM/GRU)
# ============================================================================

class RNNClassifier(NeuralClassifier):
    """Рекуррентные нейросети для текста"""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=200,
                 hidden_dim=128, num_layers=2, rnn_type='lstm',
                 bidirectional=True, dropout=0.3, attention=False, **kwargs):
        super().__init__(model_name=f"{rnn_type.upper()}_Classifier", **kwargs)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.attention = attention
        
        if self.rnn_type not in ['lstm', 'gru']:
            raise ValueError("rnn_type должен быть 'lstm' или 'gru'")
        
        self.tokenizer = None
    
    def build_model(self):
        if self.vocab_size is None or self.num_classes is None:
            raise ValueError("vocab_size и num_classes должны быть заданы")
        
        class TextRNN(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                        rnn_type, bidirectional, num_classes, dropout, attention, is_multi_label):
                super().__init__()
                self.rnn_type = rnn_type
                self.bidirectional = bidirectional
                self.num_layers = num_layers
                self.hidden_dim = hidden_dim
                self.attention = attention
                self.is_multi_label = is_multi_label
                
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                
                if rnn_type == 'lstm':
                    self.rnn = nn.LSTM(
                        embedding_dim, hidden_dim, num_layers,
                        bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0,
                        batch_first=True
                    )
                else:
                    self.rnn = nn.GRU(
                        embedding_dim, hidden_dim, num_layers,
                        bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0,
                        batch_first=True
                    )
                
                if attention:
                    self.attention_layer = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
                
                self.dropout = nn.Dropout(dropout)
                fc_input_dim = hidden_dim * (2 if bidirectional else 1)
                
                if is_multi_label:
                    self.fc = nn.Linear(fc_input_dim, num_classes)
                else:
                    self.fc = nn.Linear(fc_input_dim, num_classes)
                
                self._init_weights()
            
            def _init_weights(self):
                nn.init.xavier_uniform_(self.embedding.weight)
                nn.init.xavier_uniform_(self.fc.weight)
                nn.init.zeros_(self.fc.bias)
                if self.attention:
                    nn.init.xavier_uniform_(self.attention_layer.weight)
            
            def forward(self, x):
                if x.dtype != torch.long:
                    x = x.long()
                
                embedded = self.embedding(x)
                embedded = self.dropout(embedded)
                
                if self.rnn_type == 'lstm':
                    rnn_out, (hidden, cell) = self.rnn(embedded)
                else:
                    rnn_out, hidden = self.rnn(embedded)
                
                if self.attention:
                    attention_weights = torch.tanh(self.attention_layer(rnn_out))
                    attention_weights = F.softmax(attention_weights, dim=1)
                    context = torch.sum(rnn_out * attention_weights, dim=1)
                else:
                    if self.bidirectional:
                        hidden_forward = hidden[-2, :, :]
                        hidden_backward = hidden[-1, :, :]
                        context = torch.cat((hidden_forward, hidden_backward), dim=1)
                    else:
                        context = hidden[-1, :, :]
                
                context = self.dropout(context)
                output = self.fc(context)
                
                return output
        
        self.model = TextRNN(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            rnn_type=self.rnn_type,
            bidirectional=self.bidirectional,
            num_classes=self.num_classes,
            dropout=self.dropout,
            attention=self.attention,
            is_multi_label=self.is_multi_label
        )
        
        print(f"✅ Построена {self.rnn_type.upper()}: {self.num_layers} слоя, hidden_dim={self.hidden_dim}")
    
    def create_tokenizer(self, texts):
        import re
        from collections import Counter
        
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.most_common(self.vocab_size-2)]
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        def tokenize(text):
            words = re.findall(r'\b\w+\b', text.lower())
            indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
            return indices[:self.max_length]
        
        self.tokenizer = tokenize
        self.vocab_size = len(vocab)
        return tokenize
    
    def prepare_texts(self, texts):
        if self.tokenizer is None:
            self.create_tokenizer(texts)
        
        tokenized = [self.tokenizer(text) for text in texts]
        
        padded = []
        for tokens in tokenized:
            if len(tokens) < self.max_length:
                padded_tokens = tokens + [0] * (self.max_length - len(tokens))
            else:
                padded_tokens = tokens[:self.max_length]
            padded.append(padded_tokens)
        
        return np.array(padded, dtype=np.int64)


# ============================================================================
# ТРАНСФОРМЕРНЫЕ МОДЕЛИ
# ============================================================================

class TransformerClassifier(NeuralClassifier):
    """Fine-tuning трансформерных моделей"""
    
    def __init__(self, model_name="cointegrated/rubert-tiny", num_classes=None,
                 max_length=128, dropout=0.1, learning_rate=2e-5,
                 use_fp16=False, **kwargs):
        super().__init__(model_name=f"Transformer_{model_name.split('/')[-1]}", **kwargs)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers не установлен")
        
        self.original_model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.use_fp16 = use_fp16
        
        self.tokenizer = None
        self.config = None
    
    def _get_local_model_path(self):
        if os.path.exists(self.original_model_name):
            return self.original_model_name
        
        possible_paths = [
            f"./models/{self.original_model_name}",
            f"./models/{self.original_model_name.split('/')[-1]}",
            f"./models/rubert-tiny",
            self.original_model_name
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Найдена локальная модель: {path}")
                return path
        
        print(f"⚠️ Локальная модель не найдена: {self.original_model_name}")
        return self.original_model_name
    
    def build_model(self):
        if self.num_classes is None:
            raise ValueError("num_classes должен быть задан")
        
        print(f"🔄 Загрузка трансформера: {self.original_model_name}")
        
        model_path = self._get_local_model_path()
        
        try:
            # Загрузка токенизатора
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                print(f"✅ Токенизатор загружен")
            except:
                from transformers import BertTokenizer
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                print("🔄 Используем базовый токенизатор")
            
            # Загрузка конфигурации
            try:
                self.config = AutoConfig.from_pretrained(
                    model_path,
                    num_labels=self.num_classes,
                    hidden_dropout_prob=self.dropout,
                    attention_probs_dropout_prob=self.dropout
                )
            except:
                from transformers import BertConfig
                self.config = BertConfig(
                    num_labels=self.num_classes,
                    hidden_dropout_prob=self.dropout,
                    attention_probs_dropout_prob=self.dropout
                )
            
            # Загрузка модели
            try:
                base_model = AutoModel.from_pretrained(model_path, config=self.config)
                print(f"✅ Модель загружена")
            except Exception as e:
                print(f"❌ Ошибка загрузки модели: {e}")
                raise
            
            class TransformerForClassification(nn.Module):
                def __init__(self, base_model, config, num_classes, is_multi_label):
                    super().__init__()
                    self.base_model = base_model
                    self.config = config
                    self.num_classes = num_classes
                    self.is_multi_label = is_multi_label
                    
                    self.dropout = nn.Dropout(config.hidden_dropout_prob)
                    
                    if is_multi_label:
                        self.classifier = nn.Linear(config.hidden_size, num_classes)
                    else:
                        self.classifier = nn.Linear(config.hidden_size, num_classes)
                    
                    nn.init.xavier_uniform_(self.classifier.weight)
                    nn.init.zeros_(self.classifier.bias)
                
                def forward(self, input_ids, attention_mask=None):
                    outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    last_hidden_state = outputs.last_hidden_state
                    cls_output = last_hidden_state[:, 0, :]
                    cls_output = self.dropout(cls_output)
                    logits = self.classifier(cls_output)
                    
                    return logits
            
            self.model = TransformerForClassification(
                base_model, self.config, self.num_classes, self.is_multi_label
            )
            
            print(f"✅ Трансформер построен: {model_path}")
            print(f"   Размер скрытого состояния: {self.config.hidden_size}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            print("🔄 Создаем простую модель...")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        class SimpleFallbackMLP(nn.Module):
            def __init__(self, input_dim=768, hidden_dim=256, num_classes=None, is_multi_label=False):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.dropout = nn.Dropout(0.3)
                
                if is_multi_label:
                    self.fc2 = nn.Linear(hidden_dim, num_classes)
                else:
                    self.fc2 = nn.Linear(hidden_dim, num_classes)
                
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.xavier_uniform_(self.fc2.weight)
            
            def forward(self, input_ids, attention_mask=None):
                if input_ids.dtype != torch.float:
                    input_ids = input_ids.float()
                embedded = input_ids.mean(dim=1)
                hidden = F.relu(self.fc1(embedded))
                hidden = self.dropout(hidden)
                output = self.fc2(hidden)
                return output
        
        self.model = SimpleFallbackMLP(
            input_dim=768, 
            num_classes=self.num_classes,
            is_multi_label=self.is_multi_label
        )
        self.tokenizer = None
        print("✅ Создана простая MLP модель (fallback)")


# ============================================================================
# КОМПАРАТОР МОДЕЛЕЙ
# ============================================================================

class NeuralModelComparator:
    """Сравнение нейросетевых моделей"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_score = -1
    
    def add_model(self, model_name, model):
        self.models[model_name] = model
    
    def compare_models(self, test_data, test_labels, metrics=['accuracy', 'f1', 'precision', 'recall']):
        if not self.models:
            raise ValueError("Нет моделей для сравнения")
        
        results = []
        
        for model_name, model in self.models.items():
            print(f"🔍 Тестирование: {model_name}")
            
            if not hasattr(model, 'is_trained') or not model.is_trained:
                print(f"⚠️ Модель {model_name} не обучена, пропускаем")
                continue
            
            try:
                X_test = test_data.get(model_name)
                if X_test is None:
                    print(f"⚠️ Нет тестовых данных для {model_name}, пропускаем")
                    continue
                
                import time
                start_time = time.time()
                
                # Используем метод evaluate модели
                try:
                    eval_metrics = model.evaluate(X_test, test_labels)
                    
                    if eval_metrics.get('success', False):
                        inference_time = time.time() - start_time
                        
                        model_metrics = {
                            'model': model_name,
                            'inference_time': inference_time,
                            'accuracy': eval_metrics.get('accuracy', 0),
                            'f1': eval_metrics.get('f1', 0),
                            'precision': eval_metrics.get('precision', 0),
                            'recall': eval_metrics.get('recall', 0),
                            'is_multi_label': eval_metrics.get('is_multi_label', False)
                        }
                        
                        if eval_metrics['f1'] > self.best_score:
                            self.best_score = eval_metrics['f1']
                            self.best_model_name = model_name
                        
                        results.append(model_metrics)
                        print(f"   Метрики: accuracy={model_metrics['accuracy']:.4f}, f1={model_metrics['f1']:.4f}")
                    else:
                        print(f"❌ Ошибка при оценке модели: {eval_metrics.get('error', 'Unknown error')}")
                        continue
                        
                except Exception as e:
                    print(f"❌ Ошибка при оценке: {str(e)}")
                    continue
                
            except Exception as e:
                print(f"❌ Ошибка: {str(e)}")
                continue
        
        if results:
            self.results = pd.DataFrame(results)
            if 'f1' in self.results.columns:
                self.results = self.results.sort_values('f1', ascending=False)
            
            if self.best_model_name:
                print(f"\n🏆 Лучшая модель: {self.best_model_name} (F1: {self.best_score:.4f})")
        else:
            self.results = pd.DataFrame()
        
        return self.results
    
    def get_best_model(self):
        if self.best_model_name and self.best_model_name in self.models:
            return self.models[self.best_model_name]
        elif self.results is not None and not self.results.empty:
            best_model_name = self.results.iloc[0]['model']
            return self.models.get(best_model_name)
        else:
            return None


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ФУНКЦИИ
# ============================================================================

def create_neural_pipeline(model_type='cnn', **model_kwargs):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch не установлен")
    
    model_type = model_type.lower()
    
    if model_type == 'mlp':
        return SimpleNNClassifier(**model_kwargs)
    elif model_type == 'cnn':
        return CNNClassifier(**model_kwargs)
    elif model_type == 'rnn':
        return RNNClassifier(**model_kwargs)
    elif model_type == 'transformer':
        return TransformerClassifier(**model_kwargs)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")


def train_and_evaluate_neural_model(X_train, y_train, X_test, y_test, 
                                   model_type='cnn', model_kwargs=None,
                                   train_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}
    
    if train_kwargs is None:
        train_kwargs = {'epochs': 10, 'batch_size': 32, 'verbose': True}
    
    print(f"🚀 Запуск пайплайна: {model_type}")
    
    model = create_neural_pipeline(model_type, **model_kwargs)
    
    # Разделение на train/validation
    if SKLEARN_AVAILABLE:
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    else:
        split_idx = int(0.8 * len(X_train))
        X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
        y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
    
    print(f"📊 Размеры: Train={len(X_train_split)}, Val={len(X_val)}, Test={len(X_test)}")
    
    model.fit(X_train_split, y_train_split, X_val, y_val, **train_kwargs)
    
    print("🧪 Оценка на тестовых данных...")
    test_metrics = model.evaluate(X_test, y_test)
    
    return model, test_metrics


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    print("✅ Модуль neural_classifiers успешно загружен")
    print(f"Доступные библиотеки:")
    print(f"  PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
    print(f"  Transformers: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
    print(f"  Scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")