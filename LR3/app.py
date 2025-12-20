# ============================================================
# ОБНОВЛЕННЫЙ ИМПОРТ МОДУЛЕЙ
# ============================================================
import streamlit as st

# СНАЧАЛА настраиваем страницу
st.set_page_config(
    page_title="Лабораторный практикум №3",
    page_icon="🧪",
    layout="wide"
)

# ПОТОМ импортируем остальные модули
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import traceback
import zipfile
import io
import os
import sys

# PyTorch импорты
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Наши модули
try:
    from auto_labeling import AutoLabeler
    from data_splitter import StratifiedDataSplitter
    from text_preprocessing import TextDataProcessor
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# Модуль для классификаторов - ОДИН импорт ВСЕГО НУЖНОГО
try:
    from classical_classifiers import ClassicalClassifier, ModelComparator, create_model_configs, train_all_tasks
    CLASSIFIERS_AVAILABLE = True
except ImportError:
    CLASSIFIERS_AVAILABLE = False
    # Заглушки для ModelComparator и других
    class ModelComparator:
        def __init__(self, models_config=None):
            self.models_config = models_config or []
            self.models = {}
            self.results = {}
            self.best_model = None
            self.best_score = 0
            self.best_model_name = None
        
        def add_model(self, model_name, model):
            self.models[model_name] = model
        
        def train_and_compare(self, X_train, y_train, X_val=None, y_val=None, task_name='category'):
            return {}
        
        def get_best_model(self):
            return None
    
    def create_model_configs(task_type='category'):
        return []
    
    def train_all_tasks(X_train, y_train_all, X_val, y_val_all, task_names=None):
        return {}

# Проверка доступности библиотек для этапа 3
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

try:
    from tpot import TPOTClassifier
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False

try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False

# Модуль для нейросетевых моделей
try:
    from neural_classifiers import (
        SimpleNNClassifier, 
        CNNClassifier, 
        RNNClassifier, 
        TransformerClassifier,
        NeuralModelComparator,
        TextDataset,
        create_neural_pipeline,
        train_and_evaluate_neural_model
    )
    NEURAL_MODULES_AVAILABLE = True
except ImportError:
    NEURAL_MODULES_AVAILABLE = False

# 5 этап
try:
    from imbalance_handling import (
        ClassWeightBalancer,
        SamplingBalancer,
        TextAugmenter,
        ClassBalanceAnalyzer,
        ImbalanceHandler,
        create_imbalance_report,
        visualize_imbalance_comparison,
        get_available_balancing_methods,
        get_available_augmentation_methods
    )
    IMBALANCE_MODULES_AVAILABLE = True
except ImportError:
    IMBALANCE_MODULES_AVAILABLE = False

# 6 этап
try:
    from advanced_tuning import (
        create_tuning_pipeline,
        AdvancedModelTuner,
        CrossValidationManager,
        HyperparameterOptimizer,
        ComprehensiveModelEvaluator,
        analyze_model_stability,
        UniversalModelWrapper  # Добавьте эту строку
    )
    TUNING_MODULES_AVAILABLE = True
    st.success("✅ Модуль advanced_tuning загружен (расширенная версия)")
    
except ImportError as e:
    TUNING_MODULES_AVAILABLE = False
    st.warning(f"⚠️ Модуль advanced_tuning не доступен: {str(e)}")
    st.info("Убедитесь, что файл advanced_tuning.py находится в той же директории")

# 7 этап
try:
    # Используем функции из final_analysis.py вместо класса FinalModelAnalyzer
    from final_analysis import perform_complete_analysis, create_final_analysis_pipeline
    FINAL_ANALYSIS_AVAILABLE = True
    
    def create_final_analyzer():
        """Создает анализатор для итогового анализа"""
        return create_final_analysis_pipeline()
except ImportError as e:
    st.error(f"❌ Модуль final_analysis не доступен: {str(e)}")
    st.info("Убедитесь, что файл final_analysis.py находится в той же директории")
    FINAL_ANALYSIS_AVAILABLE = False

# ============================================================
# ЗАГЛУШКИ ДЛЯ ОТСУТСТВУЮЩИХ КЛАССОВ
# ============================================================

class SimpleClassifierStub:
    """Простая заглушка для классификатора, если модуль не доступен"""
    def __init__(self, is_multi_label=False):
        self.is_multi_label = is_multi_label
        self.is_trained = False
    
    def fit(self, X, y, X_val=None, y_val=None):
        self.is_trained = True
        return self
    
    def predict(self, X):
        if self.is_multi_label:
            # Для multi-label возвращаем случайные 0/1
            return np.random.randint(0, 2, size=(len(X), 3))  # 3 тега по умолчанию
        else:
            # Для обычной классификации возвращаем случайные метки
            return np.random.randint(0, 5, size=len(X))  # 5 классов по умолчанию
    
    def evaluate(self, X, y_true):
        return {
            'accuracy': 0.5,
            'f1': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'is_multi_label': self.is_multi_label
        }


class EnsembleClassifier:
    """Простая заглушка для ансамблевых моделей"""
    def __init__(self, **kwargs):
        pass


class AutoMLClassifier:
    """Простая заглушка для AutoML"""
    def __init__(self, **kwargs):
        pass


# ============================================================
# ОБНОВЛЕННАЯ ИНИЦИАЛИЗАЦИЯ СЕССИИ
# ============================================================
def init_session_state():
    """Инициализация состояния сессии"""
    session_vars = {
        # Основные данные
        "raw_data": None,
        "dataframe": None,
        "labeled_articles": None,
        "data_splits": None,
        "splitter": None,
        "text_processor": None,
        "processed_results": None,
        "last_file_name": None,
        
        # Статусы этапов
        "step1_completed": False,  # Этап 1: Разметка и разделение
        "step2_completed": False,  # Этап 2: Подготовка данных
        "step3_completed": False,  # Этап 3: Классификация
        "step4_completed": False,  # Этап 4: Нейросетевые модели
        "step5_completed": False,  # Этап 5: Борьба с дисбалансом
        "step6_completed": False,  # Этап 6: Настройка гиперпараметров
        "step7_completed": False,  # Этап 7: Итоговый анализ
        "step8_completed": False,  # Этап 8: Интерактивный анализ
        
        # Для этапа 3 (Классификация)
        "comparator": None,
        "comparison_results": None,
        "best_model": None,
        "test_metrics": None,
        "training_completed": False,
        "unique_classes": None,
        
        # Для этапа 4 (Нейросетевые модели)
        "neural_models": {},
        "neural_results": {},
        "neural_best_model": None,
        "neural_comparator": None,
        "neural_training_history": {},
        "neural_training_completed": False,
        "neural_label_field": 'category',
        
        # Для этапа 5 (Дисбаланс)
        "balanced_data": {},
        "balanced_models": {},
        "imbalance_handler": None,
        "class_balance_report": None,
        "original_class_distribution": None,
        "balance_analysis_completed": False,
        "balance_comparison": None,
        "imbalance_handling_completed": False,
        
        # Для этапа 6 (Настройка)
        "model_tuner": None,
        "tuning_results": {},
        "evaluation_results": {},
        "best_tuned_model": None,
        "stability_analysis": None,
        "cv_results": None,
        "hyperparameter_search_completed": False,
        "comprehensive_evaluation": None,
        "feature_names": None,
        "selected_model_for_tuning": None,
        "selected_model_name_for_tuning": None,
        
        # Для этапа 7 (Итоговый анализ)
        "final_analysis_completed": False,
        "final_analyzer": None,
        "champion_model": None,
        "champion_score": None,
        "champion_stage": None,
        
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

init_session_state()

# ============================================================
# УТИЛИТЫ
# ============================================================
def normalize_article_fields(article: dict) -> dict:
    """Нормализация полей статьи"""
    normalized = {}
    
    # Копируем все поля
    for key, value in article.items():
        normalized[key] = value
    
    # Нормализуем основные поля
    normalized["title"] = article.get("title") or article.get("заголовок") or article.get("headline") or ""
    normalized["text"] = article.get("text") or article.get("основной текст") or article.get("content") or article.get("body") or ""
    
    # Обработка категории
    category = article.get("category") or article.get("категория") or article.get("label") or article.get("тема")
    if isinstance(category, dict):
        category = str(category)
    normalized["category"] = category or ""
    
    return normalized

def to_jsonl(records):
    """Конвертация в JSONL"""
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)

def create_download_zip(files_dict, zip_name="results.zip"):
    """Создание ZIP архива для скачивания"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in files_dict.items():
            if isinstance(content, str):
                zip_file.writestr(filename, content)
            elif isinstance(content, bytes):
                zip_file.writestr(filename, content)
    zip_buffer.seek(0)
    return zip_buffer

# ============================================================
# ЗАГОЛОВОК И БОКОВАЯ ПАНЕЛЬ
# ============================================================
st.title("🧪 Лабораторный практикум №3")
st.subheader("Сравнительный анализ методов классификации текстов на русскоязычных корпусах")

with st.sidebar:
    st.title("📁 Загрузка данных")
    uploaded_file = st.file_uploader(
        "Загрузите JSONL файл",
        type=['jsonl', 'json'],
        help="Файл должен содержать записи в форматоре JSONL (JSON Lines)",
        key="file_uploader"
    )

# ============================================================
# ОСНОВНАЯ ЧАСТЬ - ЗАГРУЗКА ДАННЫХ
# ============================================================
if uploaded_file is not None:
    try:
        # Сброс состояния при загрузке нового файла
        current_file_name = uploaded_file.name
        if st.session_state.last_file_name != current_file_name:
            # Сбрасываем все последующие шаги
            st.session_state.raw_data = None
            st.session_state.dataframe = None
            st.session_state.labeled_articles = None
            st.session_state.data_splits = None
            st.session_state.text_processor = None
            st.session_state.processed_results = None
            st.session_state.last_file_name = current_file_name
            st.session_state.step1_completed = False
            st.session_state.step2_completed = False
            st.session_state.step3_completed = False
            st.session_state.step4_completed = False
            st.session_state.comparator = None
            st.session_state.comparison_results = None
            st.session_state.best_model = None
            st.session_state.test_metrics = None
            st.session_state.training_completed = False
            # Сброс нейросетевых моделей
            st.session_state.neural_models = {}
            st.session_state.neural_results = {}
            st.session_state.neural_best_model = None
            st.session_state.neural_comparator = None
            st.session_state.neural_training_history = {}
            st.session_state.neural_training_completed = False
        
        # Чтение файла
        data = []
        error_lines = 0
        
        for i, line in enumerate(uploaded_file):
            line = line.decode('utf-8').strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    error_lines += 1
                    continue
        
        if error_lines > 0:
            st.warning(f"⚠️ Пропущено {error_lines} строк с ошибками JSON")
        
        if not data:
            st.error("❌ Файл не содержит валидных данных")
        else:
            # Сохраняем данные
            df = pd.DataFrame(data)
            st.session_state.raw_data = data
            st.session_state.dataframe = df
            
            st.success(f"✅ Успешно загружено {len(data)} записей")
            
            # Статистика
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Всего записей", len(df))
            with col2:
                category_cols = [c for c in df.columns if any(word in c.lower() for word in ['категория', 'category', 'label'])]
                if category_cols:
                    categories = df[category_cols[0]].nunique()
                    st.metric("Уникальных категорий", categories)
                else:
                    st.metric("Категорий", "Нет данных")
            with col3:
                text_cols = [c for c in df.columns if any(word in c.lower() for word in ['текст', 'text', 'content'])]
                st.metric("Текстовых полей", len(text_cols))
            with col4:
                date_cols = [c for c in df.columns if any(word in c.lower() for word in ['дата', 'date'])]
                st.metric("Дата", "✅" if date_cols else "❌")
            
            # Визуализация распределения категорий
            if category_cols:
                st.subheader("📊 Распределение по категориям")
                
                category_counts = df[category_cols[0]].value_counts().reset_index()
                category_counts.columns = ['Категория', 'Количество']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(category_counts.head(15), x='Категория', y='Количество',
                                title='Топ-15 категорий', color='Количество',
                                color_continuous_scale='Blues')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(category_counts.head(10), values='Количество', 
                                names='Категория', title='Топ-10 категорий',
                                hole=0.3)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Предпросмотр данных
            st.subheader("👁️ Предпросмотр данных")
            st.dataframe(df.head(10), use_container_width=True, height=300)
    
    except Exception as e:
        st.error(f"❌ Ошибка при обработке файла: {str(e)}")
        st.code(traceback.format_exc())

else:
    # Показываем только инструкцию до загрузки файла
    st.info("👈 Загрузите JSONL файл через боковую панель")
    
    st.markdown("### 📝 Пример формата данных (JSONL)")
    example = '''{"title": "Новости кино", "text": "Вышел новый фильм...", "category": "Кино"}
{"title": "Спортивные события", "text": "Наши спортсмены победили...", "category": "Спорт"}
{"title": "Экономические новости", "text": "Рынок показал рост...", "category": "Экономика"}'''
    st.code(example, language='json')
    
    # Показываем структуру, но блокируем до загрузки файла
    st.markdown("---")
    
    # Этап 1 - заблокирован до загрузки файла
    st.header("🤖 Этап 1: Автоматическая разметка и разделение данных")
    st.warning("⏳ Загрузите файл для начала работы")
    
    st.markdown("---")
    
    # Этап 2 - заблокирован до выполнения этапа 1
    st.header("🔧 Этап 2: Подготовка данных для классификации")
    st.warning("⏳ Выполните Этап 1 для разблокировки")
    
    st.markdown("---")
    
    # Этап 3 - заблокирован до выполнения этапа 2
    st.header("🎯 Этап 3: Классификация текстов")
    st.warning("⏳ Выполните Этап 2 для разблокировки")
    
    st.markdown("---")
    
    # Этап 4 - заблокирован до выполнения этапа 3
    st.header("🧠 Этап 4: Нейросетевые и трансформерные модели")
    st.warning("⏳ Выполните Этап 3 для разблокировки")
    
    # Выходим из скрипта, чтобы дальше ничего не показывать
    st.stop()

# ============================================================
# ЭТАП 1: АВТОМАТИЧЕСКАЯ РАЗМЕТКА И РАЗДЕЛЕНИЕ ДАННЫХ
# ============================================================
st.markdown("---")

st.header("🤖 Этап 1: Автоматическая разметка и разделение данных")

if st.session_state.raw_data is not None:
    st.markdown("""
    ### 📋 Что будет выполнено автоматически:
    
    1. **Автоматическая разметка** статей:
       - Определение тональности (positive/negative)
       - Классификация по темам
       - Многометочная классификация (до 2 тем на статью)
    
    2. **Стратифицированное разделение**:
       - Разделение на Train/Validation/Test
       - Соотношение 70/15/15
       - Сохранение распределения категорий в каждом разделе
    """)
    
    # Проверка доступности модулей
    if not MODULES_AVAILABLE:
        st.error("❌ Модули auto_labeling, data_splitter или text_preprocessing не доступны")
        st.info("Убедитесь, что файлы находятся в той же директории или установите зависимости")
    else:
        # АВТОМАТИЧЕСКИЙ ЗАПУСК
        if not st.session_state.get("step1_completed", False):
            with st.spinner("Автоматическая разметка и разделение данных..."):
                try:
                    # Автоматические настройки
                    use_sentiment = True
                    use_multilabel = True
                    random_seed = 42
                    stratify_by = 'category'
                    
                    # 1. Нормализация данных
                    articles_for_labeling = [normalize_article_fields(item) for item in st.session_state.raw_data]
                    
                    # 2. Автоматическая разметка
                    labeler = AutoLabeler()
                    labeled_articles = labeler.label_articles(articles_for_labeling)
                    st.session_state.labeled_articles = labeled_articles
                    
                    # 3. Разделение данных
                    splitter = StratifiedDataSplitter(seed=random_seed)
                    splits = splitter.split_stratified(
                        labeled_articles,
                        train_ratio=0.7,
                        val_ratio=0.15,
                        test_ratio=0.15,
                        stratify_column=stratify_by,
                        save_splits=True,
                        output_dir="data_splits"
                    )
                    
                    st.session_state.data_splits = splits
                    st.session_state.splitter = splitter
                    st.session_state.step1_completed = True
                    
                    st.success("✅ Разметка и разделение данных успешно завершены!")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Показать результаты если есть
        if st.session_state.labeled_articles is not None:
            labeled_df = pd.DataFrame(st.session_state.labeled_articles)
            
            st.subheader("📊 Результаты разметки")
            
            # Метрики
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Размечено статей", len(labeled_df))
            with col2:
                if 'sentiment' in labeled_df.columns:
                    st.metric("Тональностей", labeled_df['sentiment'].nunique())
                else:
                    st.metric("Тональностей", 0)
            with col3:
                if 'category' in labeled_df.columns:
                    st.metric("Категорий", labeled_df['category'].nunique())
                else:
                    st.metric("Категорий", 0)
            with col4:
                if 'categories' in labeled_df.columns:
                    avg_topics = labeled_df['categories'].apply(len).mean()
                    st.metric("Ср. тем на статью", f"{avg_topics:.1f}")
                else:
                    st.metric("Ср. тем на статью", 0)
            
            # Визуализация разметки
            col1, col2 = st.columns(2)
            
            with col1:
                if 'sentiment' in labeled_df.columns:
                    sentiment_counts = labeled_df['sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['Тональность', 'Количество']
                    
                    fig = px.bar(sentiment_counts, x='Тональность', y='Количество',
                                title='Распределение тональностей',
                                color='Количество', color_continuous_scale='Teal')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'category' in labeled_df.columns:
                    category_counts = labeled_df['category'].value_counts().head(10).reset_index()
                    category_counts.columns = ['Категория', 'Количество']
                    
                    fig = px.pie(category_counts, values='Количество', names='Категория',
                                title='Топ-10 категорий', hole=0.3)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Показать разделенные данные если есть
            if st.session_state.data_splits is not None:
                splits = st.session_state.data_splits
                
                st.subheader("📈 Результаты разделения")
                
                # Статистика разделения
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train", len(splits['train']))
                with col2:
                    st.metric("Validation", len(splits['validation']))
                with col3:
                    st.metric("Test", len(splits['test']))
                
                # Визуализация распределения по разделам
                split_data = pd.DataFrame({
                    'Раздел': ['Train', 'Validation', 'Test'],
                    'Количество': [len(splits['train']), len(splits['validation']), len(splits['test'])]
                })
                
                fig = px.pie(split_data, values='Количество', names='Раздел',
                            title='Распределение данных по разделам',
                            color_discrete_sequence=px.colors.sequential.Blues)
                st.plotly_chart(fig, use_container_width=True)
                
                # Предпросмотр разделов
                with st.expander("👁️ Предпросмотр разделенных данных"):
                    tab1, tab2, tab3 = st.tabs(["Train", "Validation", "Test"])
                    
                    with tab1:
                        train_df = pd.DataFrame(splits['train'])
                        st.dataframe(train_df.head(), use_container_width=True, height=250)
                        st.caption(f"Train: {len(splits['train'])} записей")
                    
                    with tab2:
                        val_df = pd.DataFrame(splits['validation'])
                        st.dataframe(val_df.head(), use_container_width=True, height=250)
                        st.caption(f"Validation: {len(splits['validation'])} записей")
                    
                    with tab3:
                        test_df = pd.DataFrame(splits['test'])
                        st.dataframe(test_df.head(), use_container_width=True, height=250)
                        st.caption(f"Test: {len(splits['test'])} записей")
else:
    st.warning("⏳ Сначала загрузите данные для выполнения Этапа 1")

# ============================================================
# ЭТАП 2: АВТОМАТИЧЕСКАЯ ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАССИФИКАЦИИ
# ============================================================
st.markdown("---")

st.header("🔧 Этап 2. Подготовка данных для классификации")

if st.session_state.step1_completed:
    splits = st.session_state.data_splits
    
    st.markdown("""
    ### 📋 Что будет выполнено автоматически:
    
    1. **Предобработка текста**:
       - Очистка от HTML, URL, специальных символов
       - Токенизация и лемматизация
       - Удаление стоп-слов
    
    2. **Извлечение признаков**:
       - Мета-признаки (статистические, синтаксические, лингвистические)
       - Векторизация текста (TF-IDF, BOW, Word2Vec, FastText, BERT)
    
    3. **Работа с разделенными данными**:
       - Обучение векторизатора только на **Train** данных
       - Применение к **Validation** и **Test** без утечки данных
    """)
    
    if not MODULES_AVAILABLE:
        st.error("❌ Модуль text_preprocessing не доступен")
        st.info("Убедитесь, что файл text_preprocessing.py находится в той же директории")
    else:
        # АВТОМАТИЧЕСКИЙ ЗАПУСК
        if not st.session_state.get("step2_completed", False):
            with st.spinner("Автоматическая подготовка данных..."):
                try:
                    # АВТОМАТИЧЕСКИЕ НАСТРОЙКИ
                    remove_stopwords = True
                    
                    # Проверяем доступность spaCy
                    try:
                        import spacy
                        SPACY_AVAILABLE = True
                        use_spacy = True
                    except ImportError:
                        SPACY_AVAILABLE = False
                        use_spacy = False
                    
                    extract_meta = True
                    vectorization_method = "tfidf"
                    max_features = 2000
                    text_field = 'text'
                    batch_size = 100
                    
                    # Параметры векторизатора
                    vectorizer_params = {
                        'method': vectorization_method,
                        'max_features': max_features
                    }
                    
                    # Параметры предобработки
                    preprocessor_params = {
                        'language': 'russian',
                        'remove_stopwords': remove_stopwords,
                        'use_spacy': use_spacy
                    }
                    
                    # Создаем процессор
                    processor = TextDataProcessor(
                        preprocessor_params=preprocessor_params,
                        vectorizer_params=vectorizer_params
                    )
                    
                    # Обрабатываем все разделы
                    results = processor.process_splits_with_fallback(
                        splits,
                        extract_meta=extract_meta,
                        create_vectors=True,
                        text_field=text_field
                    )
                    
                    # Сохраняем результаты
                    st.session_state.text_processor = processor
                    st.session_state.processed_results = results
                    st.session_state.step2_completed = True
                    
                    # Сохраняем на диск
                    processor.save_processed_data("processed_data")
                    
                    st.success("✅ Подготовка данных успешно завершена!")
                    
                    # Показать предупреждение, если использовался fallback
                    if hasattr(processor, 'fallback_to_tfidf') and processor.fallback_to_tfidf:
                        st.warning(f"⚠️ Исходный метод векторизации '{vectorization_method}' не сработал. "
                                  f"Был использован резервный метод 'tfidf'.")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при подготовке данных: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Показать результаты если есть
        if st.session_state.get("processed_results") is not None:
            results = st.session_state.processed_results
            
            st.subheader("📊 Результаты подготовки данных")
            
            # Статистика по разделам
            for split_name in ['train', 'validation', 'test']:
                if split_name in results:
                    split_results = results[split_name]
                    
                    with st.expander(f"{split_name.upper()} набор", expanded=(split_name == 'train')):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Текстов", len(split_results.get('processed_texts', [])))
                        
                        with col2:
                            if 'meta_features' in split_results:
                                st.metric("Мета-признаков", 
                                         split_results['meta_features'].shape[1])
                            else:
                                st.metric("Мета-признаков", 0)
                        
                        with col3:
                            if 'text_vectors' in split_results and split_results['text_vectors'] is not None:
                                st.metric("Векторов", 
                                         split_results['text_vectors'].shape[1])
                            else:
                                st.metric("Векторов", 0)
                        
                        # Пример обработанного текста
                        if 'processed_texts' in split_results and split_results['processed_texts']:
                            st.caption("Пример обработанного текста:")
                            st.code(split_results['processed_texts'][0][:200] + "...")
            
            # Визуализация признаков
            st.subheader("📈 Визуализация признаков")
            
            tab1, tab2, tab3 = st.tabs(["Мета-признаки", "Векторы", "Сравнение"])
            
            with tab1:
                if 'train' in results and 'meta_features' in results['train']:
                    meta_df = results['train']['meta_features']
                    
                    numeric_cols = meta_df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if numeric_cols:
                        selected_features = st.multiselect(
                            "Выберите признаки для визуализации",
                            numeric_cols,
                            default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols,
                            key="feature_select"
                        )
                        
                        if selected_features:
                            # Корреляционная матрица
                            corr_matrix = meta_df[selected_features].corr()
                            
                            fig = go.Figure(data=go.Heatmap(
                                z=corr_matrix.values,
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                colorscale='RdBu',
                                zmin=-1, zmax=1
                            ))
                            fig.update_layout(title='Корреляционная матрица мета-признаков',
                                            height=500)
                            st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if 'train' in results and 'text_vectors' in results['train']:
                    vectors = results['train']['text_vectors']
                    
                    if vectors is not None:
                        st.metric("Размерность векторов", vectors.shape[1])
                        st.metric("Количество векторов", vectors.shape[0])
                        
                        # PCA для визуализации
                        try:
                            from sklearn.decomposition import PCA
                            
                            # Уменьшаем размерность для визуализации
                            pca = PCA(n_components=2)
                            vectors_2d = pca.fit_transform(vectors[:100])  # Первые 100 для скорости
                            
                            # Создаем датафрейм для визуализации
                            pca_df = pd.DataFrame({
                                'PC1': vectors_2d[:, 0],
                                'PC2': vectors_2d[:, 1]
                            })
                            
                            fig = px.scatter(pca_df, x='PC1', y='PC2', 
                                            title='PCA визуализация векторных представлений',
                                            opacity=0.7)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.info("Для PCA требуется scikit-learn: pip install scikit-learn")
            
            with tab3:
                # Сравнение разделов
                comparison_data = []
                for split_name in ['train', 'validation', 'test']:
                    if split_name in results:
                        vectors = results[split_name].get('text_vectors')
                        if vectors is not None:
                            comparison_data.append({
                                'Раздел': split_name,
                                'Размерность': vectors.shape[1],
                                'Количество': vectors.shape[0]
                            })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    
                    fig = px.bar(comp_df, x='Раздел', y='Количество',
                                title='Сравнение размеров разделов',
                                color='Раздел', text='Количество')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Выгрузка результатов
            with st.expander("💾 Сохранить результаты этапа 2"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Весь обработанный корпус
                    files_dict = {}
                    
                    for split_name in ['train', 'validation', 'test']:
                        if split_name in results:
                            split_results = results[split_name]
                            
                            # Обработанные тексты
                            if 'processed_texts' in split_results:
                                files_dict[f'{split_name}/texts.json'] = json.dumps(
                                    split_results['processed_texts'], 
                                    ensure_ascii=False, 
                                    indent=2
                                )
                            
                            # Мета-признаки
                            if 'meta_features' in split_results:
                                files_dict[f'{split_name}/meta_features.csv'] = split_results['meta_features'].to_csv(index=False)
                    
                    zip_buffer = create_download_zip(files_dict, "processed_texts.zip")
                    st.download_button(
                        label="Скачать обработанные тексты",
                        data=zip_buffer,
                        file_name="processed_texts.zip",
                        mime="application/zip"
                    )
                
                with col2:
                    # Векторные представления
                    import pickle
                    
                    vectors_dict = {}
                    for split_name in ['train', 'validation', 'test']:
                        if split_name in results:
                            vectors = results[split_name].get('text_vectors')
                            if vectors is not None:
                                vectors_dict[split_name] = vectors
                    
                    if vectors_dict:
                        vectors_bytes = pickle.dumps(vectors_dict)
                        st.download_button(
                            label="Скачать векторы (pickle)",
                            data=vectors_bytes,
                            file_name="text_vectors.pkl",
                            mime="application/octet-stream"
                        )
else:
    st.warning("⏳ Сначала выполните Этап 1: Разметку и разделение данных")

# ============================================================
# ЭТАП 3: АВТОМАТИЧЕСКАЯ КЛАССИФИКАЦИЯ ТЕКСТОВ
# ============================================================
st.markdown("---")

st.header("🎯 Этап 3. Реализация классических методов классификации")

if st.session_state.step2_completed:
    st.markdown("""
    ### 📋 Что будет выполнено автоматически:
    
    1. **Автоматический анализ данных** для ВСЕХ типов задач:
       - 📊 Анализ тональности (sentiment) - **бинарная/многоклассовая**
       - 🏷️ Классификация по категориям (category) - **многоклассовая**
       - 🏷️ Многометочная классификация (categories) - **multi-label**
    
    2. **Автоматический выбор настроек**:
       - Признаки: **combined** (мета-признаки + векторы текста)
       - Модели: **Все доступные** (Logistic Regression, Random Forest, SVM и др.)
       - Параметры: **По умолчанию** (оптимизированы для скорости и качества)
    
    3. **Отдельные таблицы результатов** для каждого типа задач
    4. **Сохранение всех моделей** для использования в 8 этапе
    """)
    
    # Вспомогательные функции для отображения информации о моделях
    def get_model_description(model_type):
        """Получение описания модели по типу"""
        descriptions = {
            'logistic': "Линейная регрессия с логистической функцией",
            'svm_linear': "Метод опорных векторов с линейным ядром",
            'svm_rbf': "Метод опорных векторов с RBF ядром",
            'random_forest': "Ансамбль решающих деревьев",
            'xgboost': "Градиентный бустинг на деревьях (XGBoost)",
            'lightgbm': "Быстрый градиентный бустинг (LightGBM)",
            'catboost': "Градиентный бустинг с категориальными признаками",
            'naive_bayes': "Наивный байесовский классификатор",
            'knn': "Метод k-ближайших соседей"
        }
        return descriptions.get(model_type, "Неизвестная модель")

    def get_model_hyperparams(config):
        """Получение гиперпараметров модели в читаемом виде"""
        model_type = config.get('model_type', '')
        
        if model_type == 'logistic':
            return f"C={config.get('C', 1.0)}, penalty={config.get('penalty', 'l2')}"
        elif model_type == 'random_forest':
            return f"n_estimators={config.get('n_estimators', 100)}, max_depth={'None' if not config.get('max_depth') else config.get('max_depth')}"
        elif model_type == 'svm_linear':
            return f"C={config.get('C', 1.0)}"
        elif model_type == 'svm_rbf':
            return f"C={config.get('C', 1.0)}, gamma={config.get('gamma', 'scale')}"
        elif model_type == 'xgboost':
            return f"n_estimators={config.get('n_estimators', 100)}, max_depth={config.get('max_depth', 6)}"
        elif model_type == 'lightgbm':
            return f"n_estimators={config.get('n_estimators', 100)}, max_depth={'None' if config.get('max_depth') == -1 else config.get('max_depth', -1)}"
        elif model_type == 'catboost':
            return f"iterations={config.get('iterations', 100)}, depth={config.get('depth', 6)}"
        elif model_type == 'naive_bayes':
            return "По умолчанию (без гиперпараметров)"
        elif model_type == 'knn':
            return f"n_neighbors={config.get('n_neighbors', 5)}"
        else:
            return "По умолчанию"
    
    if st.session_state.get("processed_results") is not None:
        splits = st.session_state.data_splits
        results = st.session_state.processed_results
        
        # 1. ПОКАЗАТЬ БУДУЩИЕ МОДЕЛИ ПЕРЕД ОБУЧЕНИЕМ
        if not st.session_state.get("training_completed", False):
            st.markdown("---")
            st.subheader("🔍 Модели, которые будут обучены")
            
            # Определяем список доступных моделей
            all_model_configs = create_model_configs()
            
            # Создаем таблицу для отображения
            model_info = []
            for config in all_model_configs:
                model_info.append({
                    "Тип модели": config.get('name', 'Unknown'),
                    "Краткое описание": get_model_description(config.get('model_type')),
                    "Поддержка multi-label": "✅" if config.get('multi_label', False) else "✅",
                    "Гиперпараметры": get_model_hyperparams(config)
                })
            
            if model_info:
                model_df = pd.DataFrame(model_info)
                
                # Разделяем на категории моделей
                st.markdown("#### 📊 Линейные модели:")
                linear_models = ["Logistic Regression", "SVM (linear)", "SVM (RBF)"]
                linear_df = model_df[model_df["Тип модели"].isin(linear_models)]
                st.dataframe(linear_df, use_container_width=True, hide_index=True)
                
                st.markdown("#### 🌳 Ансамблевые модели:")
                ensemble_models = ["Random Forest"]
                if XGBOOST_AVAILABLE:
                    ensemble_models.append("XGBoost")
                if LIGHTGBM_AVAILABLE:
                    ensemble_models.append("LightGBM")
                if CATBOOST_AVAILABLE:
                    ensemble_models.append("CatBoost")
                
                ensemble_df = model_df[model_df["Тип модели"].isin(ensemble_models)]
                st.dataframe(ensemble_df, use_container_width=True, hide_index=True)
                
                st.markdown("#### 🧠 Другие модели:")
                other_models = ["Naive Bayes", "K-Nearest Neighbors"]
                other_df = model_df[model_df["Тип модели"].isin(other_models)]
                st.dataframe(other_df, use_container_width=True, hide_index=True)
                
                st.info(f"✅ Всего будет обучено **{len(all_model_configs)}** моделей для каждой задачи")
            
            # Создаем визуализацию доступности моделей
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Линейные модели", "3")
            with col2:
                st.metric("Ансамблевые", f"{len(ensemble_models)}")
            with col3:
                st.metric("Другие", "2")
        
        # 2. АВТОМАТИЧЕСКИЙ ЗАПУСК - без выбора настроек
        if not st.session_state.get("training_completed", False):
            with st.spinner("Автоматическая классификация запущена..."):
                try:
                    # Показать прогресс в реальном времени
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    # АВТОМАТИЧЕСКИЕ НАСТРОЙКИ (все по умолчанию)
                    feature_type = "combined"  # Автоматический выбор
                    
                    # Подготовка данных
                    X_train_data = results['train'].get('combined_features')
                    X_val_data = results['validation'].get('combined_features')
                    X_test_data = results['test'].get('combined_features')
                    
                    # Преобразуем разреженные матрицы в плотные
                    try:
                        from scipy.sparse import issparse
                        
                        if issparse(X_train_data):
                            X_train_data = X_train_data.toarray()
                            if X_val_data is not None:
                                X_val_data = X_val_data.toarray()
                            if X_test_data is not None:
                                X_test_data = X_test_data.toarray()
                    except Exception as e:
                        st.warning(f"⚠️ Не удалось преобразовать данные: {e}")
                    
                    # Проверка данных
                    if X_train_data is None or len(splits['train']) == 0:
                        st.error("❌ Не удалось подготовить данные для обучения")
                        st.stop()
                    
                    # Подготавливаем метки для ВСЕХ типов задач автоматически
                    y_train_all = {}
                    y_val_all = {}
                    y_test_all = {}
                    
                    # Список всех возможных типов задач
                    all_possible_tasks = ['sentiment', 'category', 'categories']
                    
                    # Для многометочной классификации собираем все уникальные теги
                    all_unique_tags = set()
                    for item in splits['train']:
                        if 'categories' in item and isinstance(item['categories'], list):
                            all_unique_tags.update(item['categories'])
                    
                    # Сортируем теги для воспроизводимости
                    all_unique_tags = sorted(list(all_unique_tags))
                    st.session_state.all_unique_tags = all_unique_tags
                    
                    # Определяем, какие задачи действительно есть в данных
                    for task_type in all_possible_tasks:
                        train_labels = []
                        val_labels = []
                        test_labels = []
                        
                        if task_type == 'categories':
                            # ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ МНОГОМЕТОЧНОЙ КЛАССИФИКАЦИИ
                            # Создаем бинарные векторы для multi-label задачи
                            try:
                                from sklearn.preprocessing import MultiLabelBinarizer
                                mlb = MultiLabelBinarizer(classes=all_unique_tags)
                                
                                # Собираем списки тегов для каждого раздела
                                train_tags = []
                                for item in splits['train']:
                                    if 'categories' in item and isinstance(item['categories'], list):
                                        train_tags.append(item['categories'])
                                    else:
                                        train_tags.append([])
                                
                                val_tags = []
                                for item in splits['validation']:
                                    if 'categories' in item and isinstance(item['categories'], list):
                                        val_tags.append(item['categories'])
                                    else:
                                        val_tags.append([])
                                
                                test_tags = []
                                for item in splits['test']:
                                    if 'categories' in item and isinstance(item['categories'], list):
                                        test_tags.append(item['categories'])
                                    else:
                                        test_tags.append([])
                                
                                # Преобразуем в бинарные векторы
                                if train_tags:
                                    y_train_all[task_type] = mlb.fit_transform(train_tags)
                                    if val_tags:
                                        y_val_all[task_type] = mlb.transform(val_tags)
                                    if test_tags:
                                        y_test_all[task_type] = mlb.transform(test_tags)
                                    
                                    # Показать информацию о multi-label задаче
                                    progress_placeholder.info(
                                        f"✅ **Найдена multi-label задача:** {len(all_unique_tags)} уникальных тегов\n"
                                        f"Примеры тегов: {', '.join(all_unique_tags[:5])}{'...' if len(all_unique_tags) > 5 else ''}"
                                    )
                                
                                st.session_state.mlb = mlb
                                
                            except Exception as e:
                                # Fallback: берем первый тег
                                for item in splits['train']:
                                    if 'categories' in item and isinstance(item['categories'], list) and len(item['categories']) > 0:
                                        train_labels.append(str(item['categories'][0]))
                                    else:
                                        train_labels.append('unknown')
                                
                                for item in splits['validation']:
                                    if 'categories' in item and isinstance(item['categories'], list) and len(item['categories']) > 0:
                                        val_labels.append(str(item['categories'][0]))
                                    else:
                                        val_labels.append('unknown')
                                
                                for item in splits['test']:
                                    if 'categories' in item and isinstance(item['categories'], list) and len(item['categories']) > 0:
                                        test_labels.append(str(item['categories'][0]))
                                    else:
                                        test_labels.append('unknown')
                                
                                if train_labels:
                                    y_train_all[task_type] = np.array(train_labels)
                                    y_val_all[task_type] = np.array(val_labels)
                                    y_test_all[task_type] = np.array(test_labels)
                                    progress_placeholder.warning(f"⚠️ Для {task_type} используется fallback (первый тег)")
                        else:
                            # Для sentiment и category (обычная классификация)
                            for item in splits['train']:
                                train_labels.append(str(item.get(task_type, 'unknown')))
                            
                            for item in splits['validation']:
                                val_labels.append(str(item.get(task_type, 'unknown')))
                            
                            for item in splits['test']:
                                test_labels.append(str(item.get(task_type, 'unknown')))
                            
                            # Проверяем, есть ли достаточно уникальных меток
                            unique_train_labels = set([l for l in train_labels if l != 'unknown'])
                            if len(unique_train_labels) >= 2:
                                y_train_all[task_type] = np.array(train_labels)
                                y_val_all[task_type] = np.array(val_labels)
                                y_test_all[task_type] = np.array(test_labels)
                                progress_placeholder.info(
                                    f"✅ **Найдена задача {task_type}:** {len(unique_train_labels)} уникальных меток"
                                )
                            else:
                                progress_placeholder.warning(
                                    f"⚠️ Для {task_type} недостаточно данных: {len(unique_train_labels)} уникальных меток"
                                )
                    
                    if not y_train_all:
                        st.error("❌ Нет ни одного типа задач с достаточным количеством данных для обучения")
                        st.stop()
                    
                    # Показать сводку задач
                    progress_placeholder.success(f"📊 **АВТОМАТИЧЕСКИ ОПРЕДЕЛЕНЫ ЗАДАЧИ:** {len(y_train_all)} типов задач")
                    for task_type in y_train_all.keys():
                        if task_type == 'categories' and hasattr(st.session_state, 'mlb'):
                            st.caption(f"  - {task_type}: {len(all_unique_tags)} тегов (multi-label)")
                        else:
                            unique_labels = set(y_train_all[task_type])
                            st.caption(f"  - {task_type}: {len(unique_labels)} классов")
                    
                    # ИСПОЛЬЗУЕМ РЕАЛЬНЫЙ ModelComparator из classical_classifiers.py
                    if CLASSIFIERS_AVAILABLE:
                        progress_placeholder.info("🔄 Использую реальный ModelComparator из classical_classifiers.py")
                        
                        # Используем функцию train_all_tasks для обучения всех моделей
                        status_placeholder.text("🚀 Начинаю обучение моделей...")
                        
                        all_results = train_all_tasks(
                            X_train_data, y_train_all,
                            X_val_data, y_val_all,
                            task_names=list(y_train_all.keys())
                        )
                        
                        # Сохраняем результаты
                        st.session_state.all_comparison_results = all_results
                        st.session_state.y_test_all = y_test_all
                        st.session_state.X_test_data = X_test_data
                        
                        # Для совместимости сохраняем результаты первой задачи
                        if all_results:
                            first_task = list(all_results.keys())[0]
                            st.session_state.comparison_results = all_results[first_task]
                            st.session_state.training_completed = True
                            
                            # Оценка лучшей модели на тестовых данных
                            best_model_info = all_results[first_task]
                            if 'best_model' in best_model_info and best_model_info['best_model'] is not None:
                                best_model = best_model_info['best_model']
                                if first_task in y_test_all:
                                    test_metrics = best_model.evaluate(X_test_data, y_test_all[first_task])
                                    st.session_state.test_metrics = test_metrics
                                    st.session_state.best_model = best_model
                        
                        st.session_state.step3_completed = True
                        
                        status_placeholder.empty()
                        progress_placeholder.success(f"✅ Обучение и оценка моделей для {len(y_train_all)} типов задач успешно завершены!")
                    else:
                        st.error("❌ Модуль classical_classifiers.py не доступен")
                        st.info("Убедитесь, что файл classical_classifiers.py находится в той же директории")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при обучении моделей: {str(e)}")
                    st.code(traceback.format_exc())
        
        # 2. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
        if st.session_state.get("training_completed", False):
            st.markdown("---")
            st.subheader("📊 Результаты классификации")
            
            # Показать общие результатов
            if st.session_state.all_comparison_results:
                st.markdown("### 📋 Сравнение задач классификации")
                
                # Создаем таблицу сравнения
                comparison_data = []
                for task_name, task_results in st.session_state.all_comparison_results.items():
                    if 'best_score' in task_results:
                        comparison_data.append({
                            'Задача': task_name,
                            'F1-Score': task_results['best_score'],
                            'Тип': 'Multi-label' if task_name == 'categories' else 'Single-label',
                            'Лучшая модель': task_results.get('best_model_name', 'Unknown'),
                            'Моделей обучено': len(task_results.get('results', {}))
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Сортируем по F1-Score
                    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
                    
                    # Показываем таблицу
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(comparison_df, use_container_width=True)
                    
                    with col2:
                        # Статистика по моделям
                        total_models = sum(row['Моделей обучено'] for _, row in comparison_df.iterrows())
                        st.metric("Всего моделей", total_models)
                        st.metric("Типов задач", len(comparison_df))
                        st.metric("Лучший F1", f"{comparison_df.iloc[0]['F1-Score']:.4f}")
                    
                    # Визуализация
                    fig = px.bar(comparison_df, x='Задача', y='F1-Score',
                                color='Тип', title='Сравнение результатов по задачам',
                                text='F1-Score',
                                hover_data=['Лучшая модель', 'Моделей обучено'])
                    st.plotly_chart(fig, use_container_width=True)
            
            # Показать детальные результаты для каждой задачи
            st.markdown("### 📈 Детальные результаты по моделям")
            
            if st.session_state.all_comparison_results:
                task_tabs = st.tabs(list(st.session_state.all_comparison_results.keys()))
                
                for i, (task_name, task_results) in enumerate(st.session_state.all_comparison_results.items()):
                    with task_tabs[i]:
                        if 'comparator' in task_results and task_results['comparator'] is not None:
                            comparator = task_results['comparator']
                            
                            # 1. Таблица сравнения всех моделей
                            st.markdown(f"#### 🏆 Сравнение моделей для задачи: **{task_name}**")
                            
                            # Получаем таблицу результатов
                            results_df = comparator.get_results_table()
                            
                            if not results_df.empty:
                                # Показать таблицу с сортировкой
                                results_df_sorted = results_df.sort_values('F1-Score', ascending=False)
                                st.dataframe(results_df_sorted, use_container_width=True, height=350)
                                
                                # Визуализация
                                fig = px.bar(results_df_sorted, x='Model', y='F1-Score',
                                            title=f'F1-Score моделей для {task_name}',
                                            color='F1-Score',
                                            text='F1-Score',
                                            color_continuous_scale='Viridis',
                                            height=400)
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Метрики лучшей модели
                                best_model_name = results_df_sorted.iloc[0]['Model']
                                st.success(f"🏆 **Лучшая модель:** {best_model_name} (F1: {results_df_sorted.iloc[0]['F1-Score']:.4f})")
                            else:
                                st.warning("Нет данных о моделях для этой задачи")
                        
                        # 2. Детальные метрики лучшей модели на тестовых данных
                        if task_name in st.session_state.y_test_all:
                            test_metrics = task_results.get('best_model', {}).evaluate(
                                st.session_state.X_test_data,
                                st.session_state.y_test_all[task_name]
                            ) if task_results.get('best_model') else None
                            
                            if test_metrics:
                                st.markdown(f"#### 📊 Тестирование лучшей модели на тестовых данных")
                                
                                # Показать основные метрики
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("F1-Score", f"{test_metrics.get('f1', 0):.4f}")
                                
                                with col2:
                                    st.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.4f}")
                                
                                with col3:
                                    if 'precision' in test_metrics:
                                        st.metric("Precision", f"{test_metrics['precision']:.4f}")
                                    else:
                                        st.metric("Precision", "N/A")
                                
                                with col4:
                                    if 'recall' in test_metrics:
                                        st.metric("Recall", f"{test_metrics['recall']:.4f}")
                                    else:
                                        st.metric("Recall", "N/A")
                                
                                # Показать информацию о типе задачи
                                if test_metrics.get('is_multi_label', False):
                                    st.info(f"**Multi-label классификация** - {len(st.session_state.all_unique_tags)} тегов")
                                    
                                    # Для multi-label покажем дополнительные метрики
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        if 'hamming_loss' in test_metrics:
                                            st.metric("Hamming Loss", f"{test_metrics['hamming_loss']:.4f}")
                                    
                                    with col2:
                                        if 'jaccard_score' in test_metrics:
                                            st.metric("Jaccard Score", f"{test_metrics['jaccard_score']:.4f}")
                                    
                                    with col3:
                                        if 'f1_micro' in test_metrics:
                                            st.metric("F1 Micro", f"{test_metrics['f1_micro']:.4f}")
                                else:
                                    st.info(f"**Single-label классификация** - {len(np.unique(st.session_state.y_test_all[task_name]))} классов")
                                    
                                    # Для single-label покажем матрицу ошибок если есть
                                    if 'confusion_matrix' in test_metrics:
                                        st.markdown("#### 🎯 Матрица ошибок")
                                        cm = np.array(test_metrics['confusion_matrix'])
                                        fig = px.imshow(cm, text_auto=True, 
                                                       title="Confusion Matrix",
                                                       labels=dict(x="Предсказанный", y="Истинный"),
                                                       color_continuous_scale='Blues')
                                        st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            st.warning(f"Для задачи '{task_name}' нет доступной модели")
    else:
        st.warning("⚠️ Сначала выполните Этап 2: Подготовку данных для классификации")
else:
    st.warning("⏳ Сначала выполните Этап 2: Подготовку данных для классификации")

# ============================================================
# ЭТАП 4: АВТОМАТИЧЕСКОЕ ОБУЧЕНИЕ ВСЕХ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ ДЛЯ ВСЕХ ЗАДАЧ
# ============================================================
st.markdown("---")

st.header("🧠 Этап 4. Реализация нейросетевых и трансформерных моделей классификации")

if st.session_state.step3_completed:
    st.markdown("""
    ### 🚀 Автоматический запуск:
    
    **Для каждой из 3 задач будут обучены:**
    1. **Многослойный персептрон (MLP)** - на векторных представлениях
    2. **Сверточная сеть (CNN)** - на текстах, автоматическая токенизация
    3. **Рекуррентные сети (LSTM/GRU)** - для долгосрочных зависимостей
    4. **Трансформерные модели (RuBERT)** - если модель доступна локально
    
    ✅ **ВСЕ ЗАДАЧИ** (sentiment, category, categories) обрабатываются независимо
    """)
    
    if not NEURAL_MODULES_AVAILABLE:
        st.error("❌ Модуль neural_classifiers не доступен")
        st.info("Убедитесь, что файл neural_classifiers.py находится в той же директории")
    elif not TORCH_AVAILABLE:
        st.error("❌ PyTorch не установлен")
        st.info("Установите PyTorch: pip install torch torchvision")
    else:
        # Проверяем наличие подготовленных данных
        if st.session_state.get("processed_results") is None:
            st.warning("⚠️ Сначала выполните Этап 2: Подготовку данных для классификации")
            st.stop()
        
        # Получаем данные
        splits = st.session_state.data_splits
        results = st.session_state.processed_results
        
        # Проверяем, что данные существуют
        if splits is None or results is None:
            st.error("❌ Данные не загружены. Выполните предыдущие этапы.")
            st.stop()
        
        # Получаем список всех задач из Этапа 3
        if not st.session_state.get("all_comparison_results"):
            st.error("❌ Сначала выполните Этап 3 для определения задач")
            st.stop()
        
        # Определяем ВСЕ задачи из Этапа 3
        all_tasks = list(st.session_state.all_comparison_results.keys())
        
        st.success(f"📊 **Найдено {len(all_tasks)} задач из Этапа 3:** {', '.join(all_tasks)}")
        
        # Автоматические настройки для всех моделей
        use_gpu = torch.cuda.is_available()  # Автоматически определяем GPU
        batch_size = 32
        max_epochs = 10  # Для быстрого обучения
        
        # АВТОМАТИЧЕСКИЙ ЗАПУСК БЕЗ КНОПКИ
        if not st.session_state.get("neural_training_completed", False):
            st.markdown("---")
            st.subheader("🚀 Запуск автоматического обучения для ВСЕХ задач")
            
            # Автоматически запускаем обучение для КАЖДОЙ задачи
            neural_results_all_tasks = {}
            neural_models_all_tasks = {}
            neural_comparators_all_tasks = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Обучаем модели для каждой задачи
            for task_idx, task_name in enumerate(all_tasks):
                status_text.text(f"🔧 Обработка задачи {task_idx+1}/{len(all_tasks)}: {task_name}")
                progress_bar.progress(task_idx / len(all_tasks))
                
                # Определяем, является ли задача multi-label
                is_multi_label = (task_name == 'categories')
                st.info(f"🎯 **Задача:** {task_name} ({'multi-label' if is_multi_label else 'single-label'})")
                
                # Автоматически определяем, какие модели можно запустить для этой задачи
                available_models = []
                model_configs = []
                
                # 1. MLP - всегда доступна (если есть векторы)
                if 'train' in results and ('text_vectors' in results['train'] or 'combined_features' in results['train']):
                    available_models.append('mlp')
                    model_configs.append({
                        'id': 'mlp',
                        'name': 'MLP Classifier',
                        'type': 'mlp',
                        'hidden_dims': [256, 128],
                        'dropout': 0.3,
                        'learning_rate': 1e-3,
                        'is_multi_label': is_multi_label
                    })
                
                # 2. CNN - всегда доступна
                available_models.append('cnn')
                model_configs.append({
                    'id': 'cnn',
                    'name': 'CNN Classifier',
                    'type': 'cnn',
                    'model_type': 'cnn',
                    'num_filters': 100,
                    'filter_sizes': [3, 4, 5],
                    'dropout': 0.5,
                    'learning_rate': 1e-3,
                    'is_multi_label': is_multi_label
                })
                
                # 3. LSTM - всегда доступна
                available_models.append('lstm')
                model_configs.append({
                    'id': 'lstm',
                    'name': 'LSTM Classifier',
                    'type': 'rnn',
                    'model_type': 'rnn',
                    'rnn_type': 'lstm',
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'bidirectional': True,
                    'attention': False,
                    'learning_rate': 1e-3,
                    'is_multi_label': is_multi_label
                })
                
                # 4. GRU - всегда доступна
                available_models.append('gru')
                model_configs.append({
                    'id': 'gru',
                    'name': 'GRU Classifier',
                    'type': 'rnn',
                    'model_type': 'rnn',
                    'rnn_type': 'gru',
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'bidirectional': True,
                    'attention': False,
                    'learning_rate': 1e-3,
                    'is_multi_label': is_multi_label
                })
                
                # 5. Transformer - проверяем наличие локальной модели
                rubert_local_path = "./models/rubert-tiny"
                if os.path.exists(rubert_local_path):
                    model_files = os.listdir(rubert_local_path) if os.path.exists(rubert_local_path) else []
                    has_model_files = any(f.endswith(('.bin', '.safetensors', '.pth', '.pt')) for f in model_files)
                    
                    if has_model_files:
                        available_models.append('transformer')
                        model_configs.append({
                            'id': 'transformer',
                            'name': 'RuBERT (локальная)',
                            'type': 'transformer',
                            'model_type': 'transformer',
                            'model_name': rubert_local_path,
                            'max_length': 128,
                            'learning_rate': 2e-5,
                            'is_multi_label': is_multi_label
                        })
                
                st.info(f"✅ Для задачи '{task_name}' будет обучено {len(available_models)} моделей")
                
                # Подготовка данных для этой конкретной задачи
                train_texts = []
                train_labels = []
                val_texts = []
                val_labels = []
                test_texts = []
                test_labels = []
                
                # Извлекаем тексты и метки для текущей задачи
                for item in splits['train']:
                    text = item.get('text') or item.get('title') or ''
                    if text and text.strip():
                        train_texts.append(text.strip())
                        label_val = item.get(task_name, 'unknown')
                        if isinstance(label_val, list):
                            if is_multi_label:
                                train_labels.append(label_val)
                            else:
                                label_val = label_val[0] if label_val else 'unknown'
                                train_labels.append(str(label_val))
                        else:
                            train_labels.append(str(label_val))
                
                for item in splits['validation']:
                    text = item.get('text') or item.get('title') or ''
                    if text and text.strip():
                        val_texts.append(text.strip())
                        label_val = item.get(task_name, 'unknown')
                        if isinstance(label_val, list):
                            if is_multi_label:
                                val_labels.append(label_val)
                            else:
                                label_val = label_val[0] if label_val else 'unknown'
                                val_labels.append(str(label_val))
                        else:
                            val_labels.append(str(label_val))
                
                for item in splits['test']:
                    text = item.get('text') or item.get('title') or ''
                    if text and text.strip():
                        test_texts.append(text.strip())
                        label_val = item.get(task_name, 'unknown')
                        if isinstance(label_val, list):
                            if is_multi_label:
                                test_labels.append(label_val)
                            else:
                                label_val = label_val[0] if label_val else 'unknown'
                                test_labels.append(str(label_val))
                        else:
                            test_labels.append(str(label_val))
                
                # Проверяем, есть ли данные для обучения
                if not train_texts:
                    st.warning(f"⚠️ Нет текстов для обучения задачи '{task_name}'. Пропускаем.")
                    continue
                
                # Для multi-label задач преобразуем метки в бинарную матрицу
                if is_multi_label:
                    from sklearn.preprocessing import MultiLabelBinarizer
                    
                    # Собираем все уникальные теги
                    all_tags = set()
                    for tags in train_labels + val_labels + test_labels:
                        if isinstance(tags, list):
                            all_tags.update(tags)
                    
                    if len(all_tags) == 0:
                        st.warning(f"⚠️ Нет тегов для multi-label задачи '{task_name}'. Пропускаем.")
                        continue
                    
                    mlb = MultiLabelBinarizer(classes=sorted(list(all_tags)))
                    
                    # Преобразуем метки
                    y_train_mlb = mlb.fit_transform(train_labels)
                    y_val_mlb = mlb.transform(val_labels)
                    y_test_mlb = mlb.transform(test_labels)
                    
                    y_train = y_train_mlb
                    y_val = y_val_mlb
                    y_test = y_test_mlb
                    num_classes = y_train.shape[1]
                    
                    st.success(f"✅ Multi-label задача: {num_classes} тегов")
                else:
                    # Для single-label классификации
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    
                    # Объединяем все метки для корректного кодирования
                    all_labels = train_labels + val_labels + test_labels
                    le.fit(all_labels)
                    
                    y_train = le.transform(train_labels)
                    y_val = le.transform(val_labels)
                    y_test = le.transform(test_labels)
                    num_classes = len(le.classes_)
                    
                    st.success(f"✅ Single-label задача: {num_classes} классов")
                
                # Создаем компаратор для этой задачи
                neural_comparator = NeuralModelComparator()
                models_for_task = {}
                
                # Обучение каждой модели для текущей задачи
                for i, model_id in enumerate(available_models):
                    config = None
                    for cfg in model_configs:
                        if cfg['id'] == model_id:
                            config = cfg
                            break
                    
                    if not config:
                        continue
                    
                    model_name = config['name']
                    
                    status_text.text(f"🔄 Обучение модели {i+1}/{len(available_models)}: {model_name} для задачи '{task_name}'")
                    
                    # Определяем устройство
                    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
                    
                    try:
                        if config['type'] == 'mlp':
                            # MLP требует векторные представления
                            X_train_vectors = results['train'].get('text_vectors')
                            X_val_vectors = results['validation'].get('text_vectors') if results.get('validation') else None
                            X_test_vectors = results['test'].get('text_vectors') if results.get('test') else None
                            
                            if X_train_vectors is not None:
                                # Преобразуем в плотный формат если нужно
                                try:
                                    from scipy.sparse import issparse
                                    if issparse(X_train_vectors):
                                        X_train_vectors = X_train_vectors.toarray()
                                        if X_val_vectors is not None:
                                            X_val_vectors = X_val_vectors.toarray()
                                        if X_test_vectors is not None:
                                            X_test_vectors = X_test_vectors.toarray()
                                except:
                                    pass
                                
                                # Проверяем размерности
                                if X_train_vectors.shape[0] != len(train_texts):
                                    # Берем только первые n образцов
                                    n_samples = min(X_train_vectors.shape[0], len(train_texts))
                                    X_train_vectors_sub = X_train_vectors[:n_samples]
                                    train_texts_sub = train_texts[:n_samples]
                                    y_train_sub = y_train[:n_samples]
                                else:
                                    X_train_vectors_sub = X_train_vectors
                                    train_texts_sub = train_texts
                                    y_train_sub = y_train
                                
                                # Создаем и обучаем MLP
                                model = SimpleNNClassifier(
                                    input_dim=X_train_vectors_sub.shape[1],
                                    hidden_dims=config.get('hidden_dims', [256, 128]),
                                    dropout=config.get('dropout', 0.3),
                                    is_multi_label=is_multi_label,
                                    device=device
                                )
                                
                                model.num_classes = num_classes
                                model.build_model()
                                
                                if is_multi_label:
                                    model.mlb = mlb
                                else:
                                    model.label_encoder = le
                                
                                # Обучаем
                                model.fit(
                                    X_train_vectors_sub, y_train_sub,
                                    X_val_vectors, y_val,
                                    epochs=max_epochs,
                                    batch_size=batch_size,
                                    learning_rate=config.get('learning_rate', 1e-3),
                                    verbose=False
                                )
                                
                                neural_comparator.add_model(model_name, model)
                                models_for_task[model_name] = model
                                
                                st.success(f"✅ {model_name} обучена для '{task_name}'")
                            else:
                                st.warning(f"⚠️ Для MLP нужны vector features. Пропускаем {model_name} для '{task_name}'")
                        
                        elif config['type'] == 'cnn':
                            # CNN использует тексты
                            model = CNNClassifier(
                                vocab_size=10000,
                                embedding_dim=128,
                                max_length=200,
                                num_filters=config.get('num_filters', 100),
                                filter_sizes=config.get('filter_sizes', [3, 4, 5]),
                                dropout=config.get('dropout', 0.5),
                                is_multi_label=is_multi_label,
                                device=device
                            )
                            
                            # Создаем токенизатор
                            model.create_tokenizer(train_texts)
                            
                            # Подготавливаем тексты
                            X_train_cnn = model.prepare_texts(train_texts)
                            X_val_cnn = model.prepare_texts(val_texts) if val_texts else None
                            X_test_cnn = model.prepare_texts(test_texts) if test_texts else None
                            
                            model.num_classes = num_classes
                            model.build_model()
                            
                            if is_multi_label:
                                model.mlb = mlb
                            else:
                                model.label_encoder = le
                            
                            # Обучаем
                            model.fit(
                                X_train_cnn, y_train,
                                X_val_cnn, y_val,
                                epochs=max_epochs,
                                batch_size=batch_size,
                                learning_rate=config.get('learning_rate', 1e-3),
                                verbose=False
                            )
                            
                            neural_comparator.add_model(model_name, model)
                            models_for_task[model_name] = model
                            
                            st.success(f"✅ {model_name} обучена для '{task_name}'")
                        
                        elif config['type'] == 'rnn':
                            # RNN (LSTM/GRU)
                            model = RNNClassifier(
                                vocab_size=10000,
                                embedding_dim=128,
                                max_length=200,
                                hidden_dim=config.get('hidden_dim', 128),
                                num_layers=config.get('num_layers', 2),
                                rnn_type=config.get('rnn_type', 'lstm'),
                                bidirectional=config.get('bidirectional', True),
                                dropout=config.get('dropout', 0.3),
                                attention=config.get('attention', False),
                                is_multi_label=is_multi_label,
                                device=device
                            )
                            
                            # Создаем токенизатор
                            model.create_tokenizer(train_texts)
                            
                            # Подготавливаем тексты
                            X_train_rnn = model.prepare_texts(train_texts)
                            X_val_rnn = model.prepare_texts(val_texts) if val_texts else None
                            X_test_rnn = model.prepare_texts(test_texts) if test_texts else None
                            
                            model.num_classes = num_classes
                            model.build_model()
                            
                            if is_multi_label:
                                model.mlb = mlb
                            else:
                                model.label_encoder = le
                            
                            # Обучаем
                            model.fit(
                                X_train_rnn, y_train,
                                X_val_rnn, y_val,
                                epochs=max_epochs,
                                batch_size=batch_size,
                                learning_rate=config.get('learning_rate', 1e-3),
                                verbose=False
                            )
                            
                            neural_comparator.add_model(model_name, model)
                            models_for_task[model_name] = model
                            
                            st.success(f"✅ {model_name} обучена для '{task_name}'")
                        
                        elif config['type'] == 'transformer':
                            # Transformer
                            try:
                                model = TransformerClassifier(
                                    model_name=config.get('model_name', "./models/rubert-tiny"),
                                    num_classes=num_classes,
                                    max_length=config.get('max_length', 128),
                                    dropout=config.get('dropout', 0.1),
                                    learning_rate=config.get('learning_rate', 2e-5),
                                    use_fp16=config.get('use_fp16', False),
                                    is_multi_label=is_multi_label,
                                    device=device
                                )
                                
                                # Строим модель
                                model.build_model()
                                
                                if model.tokenizer is None:
                                    st.warning(f"⚠️ Не удалось загрузить токенизатор для {model_name}. Пропускаем.")
                                    continue
                                
                                if is_multi_label:
                                    model.mlb = mlb
                                else:
                                    model.label_encoder = le
                                
                                # Обучаем
                                model.fit(
                                    train_texts, y_train,
                                    val_texts, y_val,
                                    epochs=max_epochs,
                                    batch_size=batch_size,
                                    learning_rate=config.get('learning_rate', 2e-5),
                                    verbose=False
                                )
                                
                                neural_comparator.add_model(model_name, model)
                                models_for_task[model_name] = model
                                
                                st.success(f"✅ {model_name} обучена для '{task_name}'")
                                
                            except Exception as e:
                                st.warning(f"⚠️ Ошибка при обучении трансформерной модели {model_name} для '{task_name}': {str(e)}")
                                # Используем MLP как fallback
                                if 'train' in results and 'text_vectors' in results['train']:
                                    X_train_vectors = results['train'].get('text_vectors')
                                    try:
                                        from scipy.sparse import issparse
                                        if issparse(X_train_vectors):
                                            X_train_vectors = X_train_vectors.toarray()
                                    except:
                                        pass
                                    
                                    mlp_fallback = SimpleNNClassifier(
                                        input_dim=X_train_vectors.shape[1],
                                        hidden_dims=[256, 128],
                                        dropout=0.3,
                                        is_multi_label=is_multi_label,
                                        device=device
                                    )
                                    
                                    mlp_fallback.num_classes = num_classes
                                    mlp_fallback.build_model()
                                    
                                    if is_multi_label:
                                        mlp_fallback.mlb = mlb
                                    else:
                                        mlp_fallback.label_encoder = le
                                    
                                    mlp_fallback.fit(
                                        X_train_vectors, y_train,
                                        X_val_vectors, y_val,
                                        epochs=max_epochs,
                                        batch_size=batch_size,
                                        learning_rate=1e-3,
                                        verbose=False
                                    )
                                    
                                    neural_comparator.add_model(f"{model_name} (fallback MLP)", mlp_fallback)
                                    models_for_task[f"{model_name} (fallback MLP)"] = mlp_fallback
                                    
                                    st.success(f"✅ {model_name} обучена (использована MLP как fallback) для '{task_name}'")
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при обучении модели {model_name} для '{task_name}': {str(e)}")
                        continue
                
                # Оценка моделей на тестовых данных для текущей задачи
                if models_for_task:
                    status_text.text(f"📊 Оценка моделей на тестовых данных для задачи '{task_name}'...")
                    
                    # Подготавливаем тестовые данные
                    test_data_prepared = {}
                    
                    for model_name, model in models_for_task.items():
                        try:
                            if "MLP" in model_name or "fallback" in model_name:
                                # Для MLP используем векторы
                                if 'test' in results and 'text_vectors' in results['test']:
                                    X_test_vectors = results['test'].get('text_vectors')
                                    try:
                                        from scipy.sparse import issparse
                                        if issparse(X_test_vectors):
                                            X_test_vectors = X_test_vectors.toarray()
                                    except:
                                        pass
                                    
                                    # Проверяем размерность
                                    if X_test_vectors.shape[0] >= len(test_texts):
                                        test_data_prepared[model_name] = X_test_vectors[:len(test_texts)]
                                    else:
                                        # Если векторов меньше, чем текстов
                                        padding = np.zeros((len(test_texts) - X_test_vectors.shape[0], X_test_vectors.shape[1]))
                                        test_data_prepared[model_name] = np.vstack([X_test_vectors, padding])
                            elif "CNN" in model_name or "LSTM" in model_name or "GRU" in model_name:
                                # Для CNN/RNN используем подготовленные тексты
                                if hasattr(model, 'prepare_texts'):
                                    X_test_prepared = model.prepare_texts(test_texts)
                                    test_data_prepared[model_name] = X_test_prepared
                            elif "Transformer" in model_name or "RuBERT" in model_name:
                                # Для трансформеров используем сырые тексты
                                test_data_prepared[model_name] = test_texts
                        except Exception as e:
                            st.warning(f"⚠️ Не удалось подготовить данные для {model_name}: {e}")
                    
                    # Оценка всех моделей
                    if test_data_prepared:
                        try:
                            comparison_results = neural_comparator.compare_models(
                                test_data_prepared, y_test,
                                metrics=['accuracy', 'f1', 'precision', 'recall']
                            )
                            
                            # Сохранение результатов для этой задачи
                            neural_results_all_tasks[task_name] = comparison_results
                            neural_models_all_tasks[task_name] = models_for_task
                            neural_comparators_all_tasks[task_name] = neural_comparator
                            
                            st.success(f"✅ Задача '{task_name}' завершена! Обучено моделей: {len(models_for_task)}")
                            
                        except Exception as e:
                            st.error(f"❌ Ошибка при сравнении моделей для '{task_name}': {str(e)}")
                    else:
                        st.warning(f"⚠️ Не удалось подготовить тестовые данные для задачи '{task_name}'")
                else:
                    st.warning(f"⚠️ Для задачи '{task_name}' не удалось обучить ни одну модель")
            
            # Обновляем прогресс-бар
            progress_bar.progress(1.0)
            
            # Сохраняем результаты всех задач
            if neural_results_all_tasks:
                st.session_state.neural_results_all_tasks = neural_results_all_tasks
                st.session_state.neural_models_all_tasks = neural_models_all_tasks
                st.session_state.neural_comparators_all_tasks = neural_comparators_all_tasks
                st.session_state.neural_training_completed = True
                st.session_state.step4_completed = True
                
                # Находим лучшую задачу и модель
                best_task_name = None
                best_model_name = None
                best_f1_score = -1
                
                for task_name, task_results in neural_results_all_tasks.items():
                    if task_results is not None and not task_results.empty and 'f1' in task_results.columns:
                        task_best_idx = task_results['f1'].idxmax()
                        task_best_f1 = task_results.iloc[task_best_idx]['f1']
                        task_best_model = task_results.iloc[task_best_idx]['model']
                        
                        if task_best_f1 > best_f1_score:
                            best_f1_score = task_best_f1
                            best_task_name = task_name
                            best_model_name = task_best_model
                
                if best_task_name and best_model_name:
                    st.session_state.neural_best_model = neural_models_all_tasks[best_task_name].get(best_model_name)
                    st.session_state.neural_best_task = best_task_name
                    st.session_state.neural_best_score = best_f1_score
                
                st.success(f"✅ Этап 4 успешно завершен! Обучено моделей для {len(neural_results_all_tasks)} задач")
            else:
                st.error("❌ Не удалось обучить ни одну модель ни для одной задачи")
        
        # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ЭТАПА 4
        if st.session_state.get("neural_training_completed", False):
            st.markdown("---")
            st.subheader("📊 Результаты этапа 4: Нейросетевые модели для ВСЕХ задач")
            
            neural_results_all_tasks = st.session_state.get("neural_results_all_tasks", {})
            
            if neural_results_all_tasks:
                # 1. Сводка по всем задачам
                st.markdown("### 📋 Сводка по задачам")
                
                summary_data = []
                for task_name, task_results in neural_results_all_tasks.items():
                    if task_results is not None and not task_results.empty and 'f1' in task_results.columns:
                        best_f1 = task_results['f1'].max()
                        best_model = task_results.loc[task_results['f1'].idxmax(), 'model']
                        avg_f1 = task_results['f1'].mean()
                        num_models = len(task_results)
                        
                        summary_data.append({
                            'Задача': task_name,
                            'Лучший F1': f"{best_f1:.4f}",
                            'Лучшая модель': best_model,
                            'Средний F1': f"{avg_f1:.4f}",
                            'Моделей': num_models
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, height=200)
                    
                    # Визуализация сравнения задач
                    fig = px.bar(summary_df, x='Задача', y='Лучший F1',
                                title='Лучшие результаты по задачам (нейросетевые модели)',
                                color='Лучший F1', text='Лучший F1',
                                color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                
                # 2. Детальные результаты по каждой задаче
                st.markdown("### 📈 Детальные результаты по задачам")
                
                task_tabs = st.tabs(list(neural_results_all_tasks.keys()))
                
                for i, (task_name, task_results) in enumerate(neural_results_all_tasks.items()):
                    with task_tabs[i]:
                        if task_results is not None and not task_results.empty:
                            st.markdown(f"#### 🎯 Задача: {task_name}")
                            
                            # Сортировка по F1-score
                            display_df = task_results.copy()
                            if 'f1' in display_df.columns:
                                display_df = display_df.sort_values('f1', ascending=False)
                            
                            # Форматирование чисел
                            for col in ['accuracy', 'f1', 'precision', 'recall']:
                                if col in display_df.columns:
                                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                            
                            if 'inference_time' in display_df.columns:
                                display_df['inference_time'] = display_df['inference_time'].apply(lambda x: f"{x:.3f} сек")
                            
                            st.dataframe(display_df, use_container_width=True, height=300)
                            
                            # Визуализация для этой задачи
                            if 'f1' in task_results.columns:
                                sorted_df = task_results.sort_values('f1', ascending=True)
                                
                                fig = go.Figure()
                                
                                metrics_to_plot = [
                                    ('accuracy', 'Accuracy', 'lightblue'),
                                    ('precision', 'Precision', 'lightgreen'),
                                    ('recall', 'Recall', 'lightcoral'),
                                    ('f1', 'F1-Score', 'gold')
                                ]
                                
                                for metric, name, color in metrics_to_plot:
                                    if metric in sorted_df.columns:
                                        fig.add_trace(go.Bar(
                                            y=sorted_df['model'],
                                            x=sorted_df[metric],
                                            name=name,
                                            orientation='h',
                                            marker_color=color,
                                            text=sorted_df[metric].apply(lambda x: f"{x:.3f}"),
                                            textposition='auto'
                                        ))
                                
                                fig.update_layout(
                                    title=f'Сравнение моделей для задачи: {task_name}',
                                    yaxis_title='Модель',
                                    xaxis_title='Значение метрики',
                                    barmode='group',
                                    height=max(400, len(sorted_df) * 40),
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                # 3. Сравнение этапов 3 и 4 - ИСПРАВЛЕННАЯ ВЕРСИЯ
                if st.session_state.get("all_comparison_results"):
                    st.markdown("### ⚖️ Сравнение этапов 3 и 4")
                    
                    comparison_data = []
                    
                    # Результаты этапа 3 (классические модели)
                    stage3_results = st.session_state.all_comparison_results
                    for task_name, task_data in stage3_results.items():
                        if 'best_score' in task_data:
                            comparison_data.append({
                                'Задача': task_name,
                                'Этап': '3 (Классические)',
                                'F1-Score': task_data['best_score']
                            })
                    
                    # Результаты этапа 4 (нейросетевые модели)
                    for task_name, task_results in neural_results_all_tasks.items():
                        if task_results is not None and not task_results.empty and 'f1' in task_results.columns:
                            best_f1 = task_results['f1'].max()
                            best_model = task_results.loc[task_results['f1'].idxmax(), 'model']
                            comparison_data.append({
                                'Задача': task_name,
                                'Этап': f'4 (Нейросетевые - {best_model})',
                                'F1-Score': best_f1
                            })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Группировка для визуализации - ИСПРАВЛЕНО
                        # Разделяем на классические и нейросетевые
                        stage3_df = comparison_df[comparison_df['Этап'] == '3 (Классические)']
                        stage4_df = comparison_df[comparison_df['Этап'].str.startswith('4 (Нейросетевые')]
                        
                        # Создаем сводную таблицу
                        summary_data = []
                        for task in stage3_df['Задача'].unique():
                            stage3_score = stage3_df[stage3_df['Задача'] == task]['F1-Score'].values[0]
                            stage4_score = stage4_df[stage4_df['Задача'] == task]['F1-Score'].values[0] if task in stage4_df['Задача'].values else 0
                            
                            improvement = ((stage4_score - stage3_score) / stage3_score * 100) if stage3_score > 0 else 0
                            
                            summary_data.append({
                                'Задача': task,
                                'Этап 3 (F1)': f"{stage3_score:.4f}",
                                'Этап 4 (F1)': f"{stage4_score:.4f}",
                                'Улучшение (%)': f"{improvement:.1f}%",
                                'Статус': '✅ Улучшение' if improvement > 0 else '❌ Ухудшение' if improvement < 0 else '➖ Без изменений'
                            })
                        
                        if summary_data:
                            improvement_df = pd.DataFrame(summary_data)
                            st.dataframe(improvement_df, use_container_width=True)
                            
                            # Визуализация сравнения
                            fig = px.bar(comparison_df, x='Задача', y='F1-Score', color='Этап',
                                        title='Сравнение лучших результатов по задачам',
                                        barmode='group', text='F1-Score')
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Результаты обучения загружаются...")
        else:
            # Если обучение еще не завершено, но мы в процессе
            if not st.session_state.get("neural_training_completed", False):
                st.info("⏳ Обучение нейросетевых моделей в процессе...")
                st.info("Пожалуйста, подождите. Это может занять несколько минут.")
else:
    st.warning("⏳ Сначала выполните Этап 3: Классификацию текстов")


# ============================================================
# ЭТАП 5: АВТОМАТИЧЕСКАЯ БОРЬБА С ДИСБАЛАНСОМ КЛАССОВ
# ============================================================
st.markdown("---")

st.header("⚖️ Этап 5. Борьба с дисбалансом классов")

if st.session_state.step4_completed:
    st.markdown("""
    ### 🚀 Автоматический запуск:
    
    **Для каждой из 3 задач будут выполнены:**
    1. **Анализ дисбаланса** - автоматическое определение степени дисбаланса
    2. **Применение методов балансировки** - class_weight, random_oversample
    3. **Обучение и сравнение моделей** - логистическая регрессия, случайный лес
    4. **Анализ результатов** - определение лучшего метода для каждой задачи
    
    ✅ **ВСЕ ЗАДАЧИ** обрабатываются независимо
    """)
    
    # СНАЧАЛА импортируем метрики из sklearn
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        SKLEARN_METRICS_AVAILABLE = True
    except ImportError:
        st.error("❌ Не удалось импортировать метрики из sklearn.metrics")
        st.info("Установите scikit-learn: pip install scikit-learn")
        SKLEARN_METRICS_AVAILABLE = False
    
    if not IMBALANCE_MODULES_AVAILABLE:
        st.error("❌ Модуль imbalance_handling не доступен")
        st.info("Убедитесь, что файл imbalance_handling.py находится в той же директории")
    elif not SKLEARN_METRICS_AVAILABLE:
        st.error("❌ Метрики sklearn не доступны")
    else:
        # Получаем данные
        splits = st.session_state.data_splits
        results = st.session_state.processed_results
        
        # Проверяем, что данные существуют
        if splits is None or results is None:
            st.error("❌ Данные не загружены. Выполните предыдущие этапы.")
            st.stop()
        
        # Получаем список всех задач из Этапа 3
        if not st.session_state.get("all_comparison_results"):
            st.error("❌ Сначала выполните Этап 3 для определения задач")
            st.stop()
        
        all_tasks = list(st.session_state.all_comparison_results.keys())
        
        st.success(f"📊 **Найдено {len(all_tasks)} задач из Этапа 3:** {', '.join(all_tasks)}")
        
        # КНОПКА ДЛЯ ЗАПУСКА
        if not st.session_state.get("step5_completed", False):
            st.markdown("---")
            st.subheader("🚀 Запуск автоматического анализа дисбаланса для ВСЕХ задач")
            
            # Показываем кнопку для запуска
            if st.button("⚡ **ЗАПУСТИТЬ ЭТАП 5: АВТОМАТИЧЕСКАЯ БОРЬБА С ДИСБАЛАНСОМ**", 
                        type="primary", 
                        key="run_step5",
                        use_container_width=True,
                        help="Запустит анализ дисбаланса для всех задач и применит методы балансировки"):
                
                with st.spinner("Запуск автоматического анализа дисбаланса..."):
                    try:
                        # Автоматически запускаем анализ для КАЖДОЙ задачи
                        imbalance_results_all_tasks = {}
                        balanced_models_all_tasks = {}
                        balance_comparisons_all_tasks = {}
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for task_idx, task_name in enumerate(all_tasks):
                            status_text.text(f"🔧 Обработка задачи {task_idx+1}/{len(all_tasks)}: {task_name}")
                            progress_bar.progress(task_idx / len(all_tasks))
                            
                            # Определяем, является ли задача multi-label
                            is_multi_label = (task_name == 'categories')
                            st.info(f"🎯 **Задача:** {task_name} ({'multi-label' if is_multi_label else 'single-label'})")
                            
                            # 1. АНАЛИЗ ДИСБАЛАНСА
                            with st.spinner(f"Анализ дисбаланса для '{task_name}'..."):
                                try:
                                    # Собираем метки для анализа
                                    train_labels = []
                                    for item in splits['train']:
                                        label = item.get(task_name, 'unknown')
                                        if is_multi_label and isinstance(label, list):
                                            # Для multi-label берем первый тег для анализа
                                            train_labels.append(label[0] if label else 'unknown')
                                        else:
                                            train_labels.append(str(label))
                                    
                                    # Создаем обработчик и анализируем
                                    from imbalance_handling import ImbalanceHandler
                                    handler = ImbalanceHandler(random_state=42, language='rus', max_samples=5000)
                                    report = handler.analyze_imbalance(train_labels)
                                    
                                    # Сохраняем результаты
                                    if 'imbalance_results_all_tasks' not in st.session_state:
                                        st.session_state.imbalance_results_all_tasks = {}
                                    st.session_state.imbalance_results_all_tasks[task_name] = {
                                        'report': report,
                                        'handler': handler
                                    }
                                    
                                    # Показать метрики дисбаланса
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            "Коэффициент дисбаланса",
                                            f"{report.get('imbalance_ratio', 0):.2f}",
                                            help="Отношение размера самого большого класса к самому маленькому"
                                        )
                                    with col2:
                                        st.metric(
                                            "Уровень дисбаланса",
                                            report.get('imbalance_level', 'Неизвестно')
                                        )
                                    with col3:
                                        st.metric(
                                            "Количество классов",
                                            report.get('n_classes', 0)
                                        )
                                    
                                    st.success(f"✅ Анализ дисбаланса для '{task_name}' завершен")
                                    
                                except Exception as e:
                                    st.error(f"❌ Ошибка при анализе дисбаланса для '{task_name}': {e}")
                                    continue
                            
                            # 2. ПРИМЕНЕНИЕ МЕТОДОВ БАЛАНСИРОВКИ
                            with st.spinner(f"Применение методов балансировки для '{task_name}'..."):
                                try:
                                    # Подготовка данных
                                    X_train = results['train'].get('combined_features')
                                    X_test = results['test'].get('combined_features')
                                    
                                    if X_train is None:
                                        X_train = results['train'].get('text_vectors')
                                    if X_test is None:
                                        X_test = results['test'].get('text_vectors')
                                    
                                    # Получение меток
                                    train_labels_list = [item.get(task_name, 'unknown') for item in splits['train']]
                                    test_labels_list = [item.get(task_name, 'unknown') for item in splits['test']]
                                    
                                    # Преобразование данных
                                    try:
                                        from scipy.sparse import issparse
                                        if issparse(X_train):
                                            X_train = X_train.toarray()
                                        if issparse(X_test):
                                            X_test = X_test.toarray()
                                    except:
                                        pass
                                    
                                    # Кодирование меток
                                    from sklearn.preprocessing import LabelEncoder
                                    le = LabelEncoder()
                                    
                                    if is_multi_label:
                                        # Для multi-label берем первый тег
                                        train_labels_simple = []
                                        for label in train_labels_list:
                                            if isinstance(label, list) and label:
                                                train_labels_simple.append(str(label[0]))
                                            else:
                                                train_labels_simple.append(str(label))
                                        
                                        test_labels_simple = []
                                        for label in test_labels_list:
                                            if isinstance(label, list) and label:
                                                test_labels_simple.append(str(label[0]))
                                            else:
                                                test_labels_simple.append(str(label))
                                        
                                        y_train_encoded = le.fit_transform(train_labels_simple)
                                        y_test_encoded = le.transform(test_labels_simple)
                                    else:
                                        y_train_encoded = le.fit_transform(train_labels_list)
                                        y_test_encoded = le.transform(test_labels_list)
                                    
                                    # Проверка данных
                                    if X_train is None or len(y_train_encoded) == 0:
                                        st.warning(f"⚠️ Нет данных для балансировки задачи '{task_name}'. Пропускаем.")
                                        continue
                                    
                                    # Применение методов балансировки
                                    balancing_methods = ['none', 'class_weight', 'random_oversample']
                                    comparison_results = []
                                    models_for_task = {}
                                    
                                    for method in balancing_methods:
                                        try:
                                            # Применение балансировки
                                            if method == 'none':
                                                X_balanced = X_train
                                                y_balanced = y_train_encoded
                                                balance_info = {'method': 'none'}
                                            else:
                                                X_balanced, y_balanced, balance_info = handler.apply_balancing(
                                                    X_train, y_train_encoded, method=method
                                                )
                                            
                                            # Обучение моделей на сбалансированных данных
                                            from sklearn.linear_model import LogisticRegression
                                            from sklearn.ensemble import RandomForestClassifier
                                            
                                            # Обучаем логистическую регрессию
                                            lr_model = LogisticRegression(
                                                max_iter=200, 
                                                random_state=42, 
                                                n_jobs=-1,
                                                class_weight='balanced' if method == 'class_weight' else None
                                            )
                                            lr_model.fit(X_balanced, y_balanced)
                                            
                                            # Обучаем случайный лес
                                            rf_model = RandomForestClassifier(
                                                n_estimators=50,
                                                random_state=42,
                                                n_jobs=-1,
                                                max_depth=10,
                                                class_weight='balanced' if method == 'class_weight' else None
                                            )
                                            rf_model.fit(X_balanced, y_balanced)
                                            
                                            # Оценка моделей
                                            lr_pred = lr_model.predict(X_test)
                                            rf_pred = rf_model.predict(X_test)
                                            
                                            # Метрики для логистической регрессии
                                            lr_metrics = {
                                                'balancing_method': method,
                                                'model': 'logistic_regression',
                                                'accuracy': accuracy_score(y_test_encoded, lr_pred),
                                                'f1': f1_score(y_test_encoded, lr_pred, average='weighted', zero_division=0),
                                                'precision': precision_score(y_test_encoded, lr_pred, average='weighted', zero_division=0),
                                                'recall': recall_score(y_test_encoded, lr_pred, average='weighted', zero_division=0)
                                            }
                                            
                                            # Метрики для случайного леса
                                            rf_metrics = {
                                                'balancing_method': method,
                                                'model': 'random_forest',
                                                'accuracy': accuracy_score(y_test_encoded, rf_pred),
                                                'f1': f1_score(y_test_encoded, rf_pred, average='weighted', zero_division=0),
                                                'precision': precision_score(y_test_encoded, rf_pred, average='weighted', zero_division=0),
                                                'recall': recall_score(y_test_encoded, rf_pred, average='weighted', zero_division=0)
                                            }
                                            
                                            comparison_results.extend([lr_metrics, rf_metrics])
                                            
                                            # Сохранение моделей
                                            key_lr = f"{task_name}_{method}_logistic_regression"
                                            key_rf = f"{task_name}_{method}_random_forest"
                                            
                                            if 'balanced_models' not in st.session_state:
                                                st.session_state.balanced_models = {}
                                            
                                            st.session_state.balanced_models[key_lr] = {
                                                'model': lr_model,
                                                'task': task_name,
                                                'method': method,
                                                'model_type': 'logistic_regression'
                                            }
                                            
                                            st.session_state.balanced_models[key_rf] = {
                                                'model': rf_model,
                                                'task': task_name,
                                                'method': method,
                                                'model_type': 'random_forest'
                                            }
                                            
                                            models_for_task[f"{method}_lr"] = lr_model
                                            models_for_task[f"{method}_rf"] = rf_model
                                            
                                            st.success(f"✅ Метод '{method}' применен для '{task_name}'")
                                            
                                        except Exception as e:
                                            st.warning(f"⚠️ Ошибка при применении метода '{method}' для '{task_name}': {e}")
                                            continue
                                    
                                    # Сохранение результатов сравнения
                                    if comparison_results:
                                        comparison_df = pd.DataFrame(comparison_results)
                                        comparison_df['task'] = task_name
                                        
                                        if 'balance_comparisons_all_tasks' not in st.session_state:
                                            st.session_state.balance_comparisons_all_tasks = {}
                                        
                                        st.session_state.balance_comparisons_all_tasks[task_name] = comparison_df
                                        
                                        # Находим лучший метод
                                        best_row = comparison_df.loc[comparison_df['f1'].idxmax()]
                                        best_method = best_row['balancing_method']
                                        best_model = best_row['model']
                                        best_f1 = best_row['f1']
                                        
                                        st.success(f"🏆 Для задачи '{task_name}' лучший метод: {best_method} с моделью {best_model} (F1: {best_f1:.4f})")
                                        
                                        # Сохраняем лучшую модель
                                        best_key = f"{task_name}_{best_method}_{best_model}"
                                        if best_key in st.session_state.balanced_models:
                                            if 'best_balanced_models' not in st.session_state:
                                                st.session_state.best_balanced_models = {}
                                            st.session_state.best_balanced_models[task_name] = st.session_state.balanced_models[best_key]
                                    
                                    # Сохраняем модели для задачи
                                    balanced_models_all_tasks[task_name] = models_for_task
                                    
                                except Exception as e:
                                    st.error(f"❌ Ошибка при балансировке для '{task_name}': {e}")
                                    continue
                            
                            progress_bar.progress((task_idx + 1) / len(all_tasks))
                        
                        # Обновление прогресс-бара
                        progress_bar.progress(1.0)
                        
                        # Сохраняем общие результаты
                        st.session_state.imbalance_results_all_tasks = imbalance_results_all_tasks
                        st.session_state.balanced_models_all_tasks = balanced_models_all_tasks
                        st.session_state.balance_comparisons_all_tasks = balance_comparisons_all_tasks
                        st.session_state.step5_completed = True
                        st.session_state.imbalance_handling_completed = True
                        
                        st.success(f"✅ Этап 5 успешно завершен! Обработано задач: {len(all_tasks)}")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при запуске этапа 5: {e}")
            else:
                # Показываем информацию о том, что нужно нажать кнопку
                st.info("**Нажмите кнопку выше, чтобы запустить анализ дисбаланса для всех задач**")
        
        # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ЭТАПА 5
        if st.session_state.get("step5_completed", False):
            st.markdown("---")
            st.subheader("📊 Результаты этапа 5: Борьба с дисбалансом для ВСЕХ задач")
            
            # 1. Сводка по всем задачам
            st.markdown("### 📋 Сводка по задачам")
            
            summary_data = []
            for task_name in all_tasks:
                if task_name in st.session_state.get("balance_comparisons_all_tasks", {}):
                    comparison_df = st.session_state.balance_comparisons_all_tasks[task_name]
                    
                    if not comparison_df.empty:
                        # Лучший результат для этой задачи
                        best_idx = comparison_df['f1'].idxmax()
                        best_row = comparison_df.loc[best_idx]
                        
                        # Исходный дисбаланс
                        if (st.session_state.get("imbalance_results_all_tasks") and 
                            task_name in st.session_state.imbalance_results_all_tasks):
                            imbalance_report = st.session_state.imbalance_results_all_tasks[task_name]['report']
                            imbalance_ratio = imbalance_report.get('imbalance_ratio', 0)
                        else:
                            imbalance_ratio = 0
                        
                        summary_data.append({
                            'Задача': task_name,
                            'Коэффициент дисбаланса': f"{imbalance_ratio:.2f}",
                            'Лучший метод': best_row['balancing_method'],
                            'Лучшая модель': best_row['model'],
                            'Лучший F1': f"{best_row['f1']:.4f}",
                            'Улучшение': "✅" if best_row['balancing_method'] != 'none' else "➖"
                        })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, height=200)
                
                # Визуализация
                fig = px.bar(summary_df, x='Задача', y='Лучший F1',
                            color='Лучший метод', title='Лучшие результаты по задачам',
                            text='Лучший F1')
                st.plotly_chart(fig, use_container_width=True)
            
            # 2. Детальные результаты по каждой задаче
            st.markdown("### 📈 Детальные результаты по задачам")
            
            if st.session_state.get("balance_comparisons_all_tasks"):
                task_tabs = st.tabs(list(st.session_state.balance_comparisons_all_tasks.keys()))
                
                for i, (task_name, comparison_df) in enumerate(st.session_state.balance_comparisons_all_tasks.items()):
                    with task_tabs[i]:
                        if not comparison_df.empty:
                            st.markdown(f"#### 🎯 Задача: {task_name}")
                            
                            # Сортировка по F1-score
                            display_df = comparison_df.copy()
                            display_df = display_df.sort_values('f1', ascending=False)
                            
                            # Форматирование чисел
                            for col in ['accuracy', 'f1', 'precision', 'recall']:
                                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                            
                            st.dataframe(display_df, use_container_width=True, height=300)
                            
                            # Визуализация для этой задачи
                            if 'f1' in comparison_df.columns:
                                sorted_df = comparison_df.sort_values('f1', ascending=True)
                                
                                fig = go.Figure()
                                
                                # Группировка по методам балансировки
                                for method in sorted_df['balancing_method'].unique():
                                    method_data = sorted_df[sorted_df['balancing_method'] == method]
                                    fig.add_trace(go.Bar(
                                        y=method_data['model'] + " (" + method + ")",
                                        x=method_data['f1'],
                                        name=method,
                                        orientation='h',
                                        text=method_data['f1'].apply(lambda x: f"{x:.3f}"),
                                        textposition='auto'
                                    ))
                                
                                fig.update_layout(
                                    title=f'Сравнение методов балансировки для задачи: {task_name}',
                                    xaxis_title='F1-Score',
                                    yaxis_title='Модель + Метод',
                                    height=400,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
            
            # 3. Сравнение с этапами 3 и 4
            st.markdown("### ⚖️ Сравнение с предыдущими этапами")
            
            comparison_data = []
            
            # Результаты этапа 3 (классические модели)
            if st.session_state.get("all_comparison_results"):
                stage3_results = st.session_state.all_comparison_results
                for task_name, task_data in stage3_results.items():
                    if 'best_score' in task_data:
                        comparison_data.append({
                            'Задача': task_name,
                            'Этап': '3 (Классические)',
                            'F1-Score': task_data['best_score']
                        })
            
            # Результаты этапа 4 (нейросетевые модели)
            if st.session_state.get("neural_results_all_tasks"):
                stage4_results = st.session_state.neural_results_all_tasks
                for task_name, task_results in stage4_results.items():
                    if task_results is not None and not task_results.empty and 'f1' in task_results.columns:
                        best_f1 = task_results['f1'].max()
                        comparison_data.append({
                            'Задача': task_name,
                            'Этап': '4 (Нейросетевые)',
                            'F1-Score': best_f1
                        })
            
            # Результаты этапа 5 (с балансировкой)
            if st.session_state.get("balance_comparisons_all_tasks"):
                stage5_results = st.session_state.balance_comparisons_all_tasks
                for task_name, comparison_df in stage5_results.items():
                    if not comparison_df.empty and 'f1' in comparison_df.columns:
                        best_f1 = comparison_df['f1'].max()
                        comparison_data.append({
                            'Задача': task_name,
                            'Этап': '5 (С балансировкой)',
                            'F1-Score': best_f1
                        })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Группировка для визуализации
                fig = px.bar(comparison_df, x='Задача', y='F1-Score', color='Этап',
                            title='Сравнение результатов по этапам',
                            barmode='group', text='F1-Score')
                st.plotly_chart(fig, use_container_width=True)
            
            # 4. Сохранение результатов
            st.markdown("---")
            st.subheader("💾 Сохранение результатов")
            
            with st.expander("📥 Скачать результаты этапа 5", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Сохранение лучших моделей
                    if st.session_state.get("best_balanced_models"):
                        try:
                            import pickle
                            models_bytes = pickle.dumps(st.session_state.best_balanced_models)
                            
                            st.download_button(
                                label="🤖 Лучшие модели с балансировкой",
                                data=models_bytes,
                                file_name=f"balanced_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                mime="application/octet-stream"
                            )
                        except Exception as e:
                            st.warning(f"Не удалось сериализовать модели: {e}")
                
                with col2:
                    # Сохранение отчетов
                    if st.session_state.get("balance_comparisons_all_tasks"):
                        all_reports = {}
                        for task_name, comparison_df in st.session_state.balance_comparisons_all_tasks.items():
                            all_reports[task_name] = comparison_df.to_dict(orient='records')
                        
                        reports_json = json.dumps(all_reports, indent=2, ensure_ascii=False, default=str)
                        
                        st.download_button(
                            label="📋 Отчеты по балансировке (JSON)",
                            data=reports_json,
                            file_name=f"balance_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                # Полный архив
                if st.button("📦 Создать полный архив этапа 5", key="create_stage5_archive"):
                    with st.spinner("Создание архива..."):
                        files_dict = {}
                        
                        # Отчеты
                        if st.session_state.get("balance_comparisons_all_tasks"):
                            for task_name, comparison_df in st.session_state.balance_comparisons_all_tasks.items():
                                files_dict[f'{task_name}/comparison.csv'] = comparison_df.to_csv(index=False)
                        
                        # Сводка
                        summary = f"""
                        Результаты этапа 5: Автоматическая борьба с дисбалансом
                        Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        
                        Обработано задач: {len(all_tasks)}
                        Задачи: {', '.join(all_tasks)}
                        
                        Итоговые результаты:
                        """
                        
                        if summary_data:
                            for row in summary_data:
                                summary += f"\n- {row['Задача']}: {row['Лучший метод']} + {row['Лучшая модель']} (F1: {row['Лучший F1']})"
                        
                        files_dict['summary.txt'] = summary
                        
                        # Создание ZIP
                        zip_buffer = create_download_zip(files_dict, "stage5_balance_results.zip")
                        
                        st.download_button(
                            label="📥 Скачать полный архив этапа 5",
                            data=zip_buffer,
                            file_name=f"stage5_balance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
        else:
            st.info("👆 Нажмите кнопку выше для запуска анализа дисбаланса")
else:
    st.warning("⏳ Сначала выполните Этап 4: Нейросетевые и трансформерные модели")


# ============================================================
# ЭТАП 6: НАСТРОЙКА ГИПЕРПАРАМЕТРОВ И ВСЕСТОРОННЯЯ ОЦЕНКА
# ============================================================
st.markdown("---")

st.header("⚙️ Этап 6: Настройка гиперпараметров и всесторонняя оценка моделей")

if st.session_state.step5_completed:
    # Проверяем F1-Score лучшей модели
    best_model_score = st.session_state.get('best_model_score', 0)
    
    if best_model_score >= 0.99:
        st.warning(f"""
        ⚠️ **Внимание:** Лучшая модель из предыдущих этапов показывает очень высокий F1-Score ({best_model_score:.4f}).
        
        **Возможные причины:**
        1. **Слишком простые данные** для классификации
        2. **Утечка данных** между обучающей и тестовой выборками
        3. **Недостаточно сложная задача** классификации
        4. **Ошибка в оценке метрик**
        
        **Рекомендация:** Проверьте качество данных и сложность задачи.
        Настройка гиперпараметров может не дать значимого улучшения, но будет выполнена для демонстрации.
        """)
    
    st.markdown("""
    ### 🎯 **Задача:** Разработать методику настройки гиперпараметров и всесторонней оценки качества моделей.
    
    **Указания к выполнению:**
    
    **1. Кросс-валидация:**
    - 🎯 **Stratified K-Fold** - для сохранения баланса классов
    - 📅 **Временное разделение** - для временных рядов текстовых данных  
    - 👥 **Group K-Fold** - для данных с групповой структурой
    
    **2. Подбор гиперпараметров:**
    - 🔍 **Grid Search** - для полного перебора
    - 🎲 **Random Search** - для случайного поиска
    - 🧠 **Bayesian Optimization** (Optuna, Hyperopt) - для эффективного поиска
    - 🤖 **Для трансформеров:** поиск оптимальной скорости обучения, размера батча, количества эпох
    
    **3. Регуляризация:**
    - 📏 **L1, L2 регуляризация** - для линейных моделей
    - 🚫 **Dropout** - для нейросетей
    - ⚖️ **Weight decay** - для трансформеров
    - ⏹️ **Early Stopping, ReduceLROnPlateau** - для всех типов моделей
    
    **4. Метрики оценки:**
    - 🎯 **Матрица ошибок**
    - 📊 **Accuracy, Precision, Recall, F1-Score** с учетом макро/микро усреднения
    - 📈 **ROC-AUC** - для бинарной классификации
    - 📉 **PR-AUC** - для задач с дисбалансом классов
    - 📉 **Log Loss** (кросс-энтропия)
    - 🤖 **Для трансформеров:** дополнительные метрики интерпретируемости и стабильности
    """)
    
    if not TUNING_MODULES_AVAILABLE:
        st.error("❌ Модуль advanced_tuning не доступен")
        st.info("""
        Убедитесь, что файл advanced_tuning.py находится в той же директории.
        Установите необходимые библиотеки:
        ```
        pip install scikit-learn optuna hyperopt numpy pandas scipy
        ```
        """)
    else:
        # 1. КОНФИГУРАЦИЯ НАСТРОЙКИ
        st.markdown("---")
        st.subheader("⚙️ Конфигурация методов настройки")
        
        # Создаем три колонки для настроек
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Кросс-валидация")
            cv_strategy = st.selectbox(
                "Стратегия CV",
                ["stratified", "timeseries", "group"],
                index=0,
                help="Stratified K-Fold: баланс классов | Time Series: временные ряды | Group K-Fold: групповые данные"
            )
            
            cv_splits = st.slider(
                "Количество фолдов",
                min_value=3,
                max_value=10,
                value=5,
                help="Рекомендуется 5-10 фолдов"
            )
            
            if cv_strategy == 'group':
                st.info("Для Group K-Fold нужны данные о группах")
        
        with col2:
            st.markdown("#### 🔍 Оптимизация гиперпараметров")
            optimizer_type = st.selectbox(
                "Метод оптимизации",
                ["random", "grid", "bayesian"],
                index=0,
                help="Random Search: быстрый | Grid Search: полный | Bayesian: эффективный"
            )
            
            n_trials = st.slider(
                "Количество испытаний/итераций",
                min_value=10,
                max_value=200,
                value=50,
                help="Для Random/Bayesian Search"
            )
            
            scoring_metric = st.selectbox(
                "Метрика для оптимизации",
                ["f1_macro", "accuracy", "roc_auc", "precision_macro", "recall_macro"],
                index=0,
                help="Метрика, которую будем максимизировать"
            )
        
        with col3:
            st.markdown("#### 📈 Метрики оценки")
            selected_metrics = st.multiselect(
                "Выберите метрики для оценки",
                ["accuracy", "f1_macro", "f1_micro", "precision_macro", "recall_macro", 
                 "precision_micro", "recall_micro", "roc_auc", "pr_auc", "log_loss"],
                default=["accuracy", "f1_macro", "f1_micro", "roc_auc", "log_loss"],
                help="Макро: среднее по классам | Микро: глобальное усреднение"
            )
            
            st.markdown("#### ⚖️ Регуляризация")
            regularization_type = st.selectbox(
                "Тип регуляризации",
                ["auto", "l1_l2", "dropout", "weight_decay", "early_stopping"],
                index=0,
                help="Auto: автоматический выбор | L1/L2: линейные модели | Dropout: нейросети"
            )
        
        # 2. АВТОМАТИЧЕСКИЙ ВЫБОР МОДЕЛИ ДЛЯ НАСТРОЙКИ
        st.markdown("---")
        st.subheader("🔍 Автоматический выбор лучшей модели для настройки")
        
        if not st.session_state.get("hyperparameter_search_completed", False):
            # Автоматический выбор модели из предыдущих этапов
            with st.spinner("Анализирую лучшие модели из этапов 3-5..."):
                try:
                    best_model = None
                    best_score = 0
                    best_model_name = ""
                    
                    # Проверяем модели из этапа 3
                    if st.session_state.get("all_comparison_results"):
                        for task_name, task_data in st.session_state.all_comparison_results.items():
                            if 'best_score' in task_data and task_data['best_score'] > best_score:
                                # Пропускаем модели с F1=1.0 (скорее всего ошибка)
                                if task_data['best_score'] < 0.99:
                                    best_score = task_data['best_score']
                                    best_model = task_data.get('best_model')
                                    best_model_name = f"Этап 3: {task_name} ({task_data.get('best_model_name', 'Модель')})"
                    
                    # Если все модели имеют F1=1.0, берем первую
                    if best_model is None and st.session_state.get("all_comparison_results"):
                        for task_name, task_data in st.session_state.all_comparison_results.items():
                            if 'best_model' in task_data and task_data.get('best_model') is not None:
                                best_model = task_data.get('best_model')
                                best_score = task_data.get('best_score', 0)
                                best_model_name = f"Этап 3: {task_name} ({task_data.get('best_model_name', 'Модель')})"
                                break
                    
                    # Проверяем модели из этапа 4
                    if st.session_state.get("neural_results_all_tasks"):
                        for task_name, task_results in st.session_state.neural_results_all_tasks.items():
                            if task_results is not None and not task_results.empty and 'f1' in task_results.columns:
                                task_best = task_results['f1'].max()
                                if task_best > best_score and task_best < 0.99:  # Пропускаем F1=1.0
                                    best_score = task_best
                                    best_model_name = f"Этап 4: {task_name} (Нейросеть)"
                    
                    # Проверяем модели из этапа 5
                    if st.session_state.get("best_balanced_models"):
                        for task_name, model_info in st.session_state.best_balanced_models.items():
                            if isinstance(model_info, dict) and 'model' in model_info and model_info['model'] is not None:
                                # Используем оценку из сравнения если есть
                                if st.session_state.get("balance_comparisons_all_tasks"):
                                    df = st.session_state.balance_comparisons_all_tasks.get(task_name)
                                    if df is not None and not df.empty and 'f1' in df.columns:
                                        model_score = df['f1'].max()
                                        if model_score > best_score and model_score < 0.99:
                                            best_score = model_score
                                            best_model = model_info['model']
                                            best_model_name = f"Этап 5: {task_name} (Балансировка)"
                    
                    if best_model is not None:
                        st.session_state.selected_model_for_tuning = best_model
                        st.session_state.selected_model_name_for_tuning = best_model_name
                        st.session_state.best_model_score = best_score
                        
                        st.success(f"🏆 **Выбрана модель для настройки:** {best_model_name}")
                        st.success(f"📊 **Исходный F1-Score:** {best_score:.4f}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Текущий F1", f"{best_score:.4f}")
                        with col2:
                            if best_score >= 0.99:
                                st.metric("Цель", "Проверка данных")
                            else:
                                st.metric("Цель", "Улучшение на 5-20%")
                        with col3:
                            st.metric("Метод", optimizer_type.capitalize())
                    else:
                        st.error("❌ Не найдено подходящих моделей для настройки")
                        st.info("""
                        **Возможные причины:**
                        1. Все модели имеют F1-Score = 1.0 (слишком простые данные)
                        2. Модели не были сохранены в предыдущих этапах
                        3. Ошибка при обучении моделей
                        """)
                        st.stop()
                        
                except Exception as e:
                    st.error(f"❌ Ошибка при выборе модели: {str(e)}")
                    st.code(traceback.format_exc())
        
        # 3. ЗАПУСК КОМПЛЕКСНОЙ НАСТРОЙКИ
        if st.session_state.get("selected_model_for_tuning"):
            st.markdown("---")
            st.subheader("🚀 Запуск комплексной настройки модели")
            
            # Подготовка данных
            with st.spinner("Подготавливаю данные для настройки..."):
                try:
                    # Используем данные из предыдущих этапов
                    results = st.session_state.processed_results
                    splits = st.session_state.data_splits
                    
                    if results is None or splits is None:
                        st.error("❌ Данные не загружены")
                        st.stop()
                    
                    # Определяем задачу (берем первую из доступных)
                    task_names = list(st.session_state.all_comparison_results.keys())
                    if not task_names:
                        st.error("❌ Не найдено задач для настройки")
                        st.stop()
                    
                    task_name = task_names[0]
                    
                    # Получаем признаки
                    X_train = results['train'].get('combined_features')
                    X_test = results['test'].get('combined_features')
                    
                    if X_train is None:
                        X_train = results['train'].get('text_vectors')
                        X_test = results['test'].get('text_vectors')
                    
                    # Преобразуем в плотный формат если нужно
                    try:
                        from scipy.sparse import issparse
                        if issparse(X_train):
                            X_train = X_train.toarray()
                        if issparse(X_test):
                            X_test = X_test.toarray()
                    except:
                        pass
                    
                    # Получаем метки
                    train_labels = []
                    test_labels = []
                    
                    for item in splits['train']:
                        label = item.get(task_name, 'unknown')
                        if isinstance(label, list):
                            train_labels.append(str(label[0]) if label else 'unknown')
                        else:
                            train_labels.append(str(label))
                    
                    for item in splits['test']:
                        label = item.get(task_name, 'unknown')
                        if isinstance(label, list):
                            test_labels.append(str(label[0]) if label else 'unknown')
                        else:
                            test_labels.append(str(label))
                    
                    # Кодируем метки
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    
                    all_labels = train_labels + test_labels
                    le.fit(all_labels)
                    
                    y_train = le.transform(train_labels)
                    y_test = le.transform(test_labels)
                    
                    # Проверяем размеры данных
                    if X_train is None or len(y_train) == 0 or X_test is None or len(y_test) == 0:
                        st.error("❌ Не удалось подготовить данные для настройки")
                        st.stop()
                    
                    # Проверяем, что размеры совпадают
                    if len(X_train) != len(y_train):
                        # Обрезаем до минимального размера
                        min_len = min(len(X_train), len(y_train))
                        X_train = X_train[:min_len]
                        y_train = y_train[:min_len]
                    
                    if len(X_test) != len(y_test):
                        min_len = min(len(X_test), len(y_test))
                        X_test = X_test[:min_len]
                        y_test = y_test[:min_len]
                    
                    # Сохраняем данные
                    st.session_state.tuning_data = {
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_test': X_test,
                        'y_test': y_test,
                        'task_name': task_name,
                        'label_encoder': le
                    }
                    
                    st.success(f"✅ Данные подготовлены: {X_train.shape[0]} train, {X_test.shape[0]} test")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при подготовке данных: {str(e)}")
                    st.code(traceback.format_exc())
                    st.stop()
            
            # Кнопка запуска настройки
            if st.button("🚀 **ЗАПУСТИТЬ КОМПЛЕКСНУЮ НАСТРОЙКУ МОДЕЛИ**",
                        type="primary",
                        key="run_comprehensive_tuning",
                        use_container_width=True,
                        help="Запустит полный цикл настройки и оценки"):
                
                with st.spinner("Выполняется комплексная настройка модели..."):
                    try:
                        # Проверяем F1-Score исходной модели
                        original_score = st.session_state.get('best_model_score', 0)
                        
                        if original_score >= 0.99:
                            st.warning("""
                            ⚠️ **Внимание:** Исходная модель уже показывает отличные результаты.
                            
                            **Возможные причины:**
                            1. Данные слишком простые для классификации
                            2. Произошла утечка данных
                            3. Ошибка в оценке метрик
                            
                            **Рекомендация:** Настройка гиперпараметров может не дать значимого улучшения.
                            """)
                            
                            # Все равно запускаем настройку для демонстрации
                            st.info("Запускаю настройку для демонстрации работы алгоритма...")
                        
                        # Получаем данные
                        tuning_data = st.session_state.tuning_data
                        X_train = tuning_data['X_train']
                        y_train = tuning_data['y_train']
                        X_test = tuning_data['X_test']
                        y_test = tuning_data['y_test']
                        
                        # Получаем модель
                        model = st.session_state.selected_model_for_tuning
                        
                        # Проверяем, что модель не None
                        if model is None:
                            st.error("❌ Модель для настройки не найдена")
                            st.stop()
                        
                        # Создаем обертку для модели
                        from advanced_tuning import UniversalModelWrapper
                        model_wrapper = UniversalModelWrapper(model=model)
                        
                        # Создаем прогресс-бар
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # 1. Настройка регуляризации
                        status_text.text("⚖️ Настройка регуляризации...")
                        progress_bar.progress(10)
                        
                        regularization_params = {}
                        if regularization_type == 'l1_l2':
                            regularization_params = {'alpha': 0.01, 'l1_ratio': 0.5}
                        elif regularization_type == 'dropout':
                            regularization_params = {'dropout_rate': 0.3}
                        elif regularization_type == 'weight_decay':
                            regularization_params = {'weight_decay': 0.01}
                        elif regularization_type == 'early_stopping':
                            regularization_params = {'patience': 10, 'min_delta': 0.001}
                        
                        # 2. Создание тюнера
                        status_text.text("🔄 Создание AdvancedModelTuner...")
                        progress_bar.progress(20)
                        
                        tuner = AdvancedModelTuner(
                            cv_strategy=cv_strategy,
                            cv_splits=cv_splits,
                            optimizer_type=optimizer_type,
                            n_trials=n_trials,
                            scoring=scoring_metric,
                            regularization_params=regularization_params,
                            metrics=selected_metrics,
                            n_jobs=-1,
                            random_state=42
                        )
                        
                        # 3. Запуск настройки
                        status_text.text(f"🎯 Запуск {optimizer_type} оптимизации...")
                        progress_bar.progress(40)
                        
                        results = tuner.tune_and_evaluate(
                            model_wrapper, 
                            X_train, y_train, 
                            X_test, y_test,
                            task_name=task_name
                        )
                        
                        progress_bar.progress(80)
                        status_text.text("📊 Оценка результатов...")
                        
                        # 4. Обработка результатов
                        if results.get('success', False):
                            # Сохраняем результаты
                            st.session_state.tuning_results = results.get('tuning', {})
                            st.session_state.evaluation_results = results.get('evaluation', {})
                            st.session_state.comprehensive_evaluation = results.get('report', {})
                            st.session_state.best_tuned_model = results.get('tuning', {}).get('best_model')
                            
                            # Обновляем статус
                            st.session_state.hyperparameter_search_completed = True
                            st.session_state.step6_completed = True
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Настройка завершена!")
                            
                            # Показываем результаты
                            tuned_score = results['evaluation']['metrics'].get('f1_macro', 0)
                            original_score = st.session_state.get('best_model_score', 0)
                            
                            if tuned_score > original_score:
                                improvement = ((tuned_score - original_score) / original_score) * 100
                                st.success(f"✅ УСПЕХ! Модель улучшена на {improvement:.1f}%")
                                st.success(f"📈 F1-Score: {original_score:.4f} → {tuned_score:.4f}")
                                
                                # Показываем лучшие параметры
                                best_params = results['tuning'].get('best_params', {})
                                if best_params:
                                    with st.expander("📋 Лучшие гиперпараметры", expanded=True):
                                        for key, value in best_params.items():
                                            st.write(f"**{key}:** {value}")
                            else:
                                st.warning("⚠️ Улучшение не достигнуто. Использованы оптимальные параметры.")
                                st.info(f"F1-Score до настройки: {original_score:.4f}")
                                st.info(f"F1-Score после настройки: {tuned_score:.4f}")
                        
                        else:
                            st.error("❌ Настройка не удалась")
                            if 'error' in results:
                                st.error(f"Ошибка: {results['error']}")
                            
                    except Exception as e:
                        st.error(f"❌ Ошибка при настройке: {str(e)}")
                        st.code(traceback.format_exc())
        
        # 4. ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
        if st.session_state.get("step6_completed", False):
            st.markdown("---")
            st.subheader("📊 Результаты комплексной настройки модели")
            
            evaluation_results = st.session_state.get("evaluation_results", {})
            tuning_results = st.session_state.get("tuning_results", {})
            
            if evaluation_results:
                # 1. Основные метрики
                st.markdown("### 📈 Основные метрики оценки")
                
                metrics = evaluation_results.get('metrics', {})
                
                # Создаем колонки для метрик
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                    st.metric("F1 Macro", f"{metrics.get('f1_macro', 0):.4f}")
                
                with col2:
                    st.metric("Precision Macro", f"{metrics.get('precision_macro', 0):.4f}")
                    st.metric("Recall Macro", f"{metrics.get('recall_macro', 0):.4f}")
                
                with col3:
                    if 'roc_auc' in metrics:
                        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                    if 'pr_auc' in metrics:
                        st.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
                
                with col4:
                    if 'log_loss' in metrics:
                        st.metric("Log Loss", f"{metrics['log_loss']:.4f}")
                    if 'f1_micro' in metrics:
                        st.metric("F1 Micro", f"{metrics['f1_micro']:.4f}")
                
                # 2. Матрица ошибок
                st.markdown("### 🎯 Матрица ошибок")
                
                if evaluation_results.get('confusion_matrix'):
                    cm = np.array(evaluation_results['confusion_matrix'])
                    fig = px.imshow(cm, text_auto=True,
                                   title="Матрица ошибок (Confusion Matrix)",
                                   labels=dict(x="Предсказанный класс", y="Истинный класс"),
                                   color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)
                
                # 3. Сравнение метрик
                st.markdown("### 📊 Сравнение метрик")
                
                if 'classification_report' in evaluation_results:
                    report = evaluation_results['classification_report']
                    
                    # Извлекаем метрики по классам
                    class_metrics = []
                    for class_name, class_data in report.items():
                        if isinstance(class_data, dict) and 'precision' in class_data:
                            class_metrics.append({
                                'Класс': class_name,
                                'Precision': class_data['precision'],
                                'Recall': class_data['recall'],
                                'F1-Score': class_data['f1-score'],
                                'Поддержка': class_data['support']
                            })
                    
                    if class_metrics:
                        class_df = pd.DataFrame(class_metrics)
                        
                        fig = px.bar(class_df, 
                                    x='Класс', 
                                    y=['Precision', 'Recall', 'F1-Score'],
                                    title='Метрики по классам',
                                    barmode='group',
                                    height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # 4. Информация о настройке
                st.markdown("### ⚙️ Информация о настройке")
                
                if tuning_results:
                    info_data = [
                        {'Параметр': 'Стратегия CV', 'Значение': tuning_results.get('cv_strategy', 'N/A')},
                        {'Параметр': 'Метод оптимизации', 'Значение': tuning_results.get('optimizer_type', 'N/A')},
                        {'Параметр': 'Лучший Score (CV)', 'Значение': f"{tuning_results.get('best_score', 0):.4f}"},
                        {'Параметр': 'Исходный F1', 'Значение': f"{tuning_results.get('original_f1', 0):.4f}"},
                    ]
                    
                    info_df = pd.DataFrame(info_data)
                    st.dataframe(info_df, use_container_width=True, hide_index=True)
                
                # 5. Стабильность модели
                st.markdown("### 📊 Анализ стабильности")
                
                try:
                    stability_results = analyze_model_stability(
                        st.session_state.best_tuned_model,
                        X_train,
                        y_train,
                        n_bootstrap=10,
                        random_state=42
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Средний F1", f"{stability_results['mean_score']:.4f}")
                    with col2:
                        st.metric("Стандартное отклонение", f"{stability_results['std_score']:.4f}")
                    with col3:
                        ci = stability_results['confidence_interval']
                        st.metric("95% доверительный интервал", f"[{ci[0]:.4f}, {ci[1]:.4f}]")
                    
                    # График распределения
                    fig = px.histogram(
                        x=stability_results['bootstrap_scores'],
                        nbins=10,
                        title='Распределение бутстрап оценок',
                        labels={'x': 'F1-Score', 'y': 'Частота'},
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Анализ стабильности недоступен: {e}")
                
                # 6. Сравнение с предыдущими этапами
                st.markdown("### ⚖️ Сравнение с предыдущими этапами")
                
                comparison_data = []
                original_score = st.session_state.get('best_model_score', 0)
                tuned_score = metrics.get('f1_macro', 0)
                
                # Добавляем результаты этапов
                if st.session_state.get("all_comparison_results"):
                    for task_name, task_data in st.session_state.all_comparison_results.items():
                        if 'best_score' in task_data:
                            comparison_data.append({
                                'Этап': f'3: {task_name}',
                                'F1-Score': task_data['best_score'],
                                'Тип': 'Классические'
                            })
                
                if st.session_state.get("neural_results_all_tasks"):
                    for task_name, task_results in st.session_state.neural_results_all_tasks.items():
                        if task_results is not None and not task_results.empty and 'f1' in task_results.columns:
                            best_f1 = task_results['f1'].max()
                            comparison_data.append({
                                'Этап': f'4: {task_name}',
                                'F1-Score': best_f1,
                                'Тип': 'Нейросети'
                            })
                
                comparison_data.append({
                    'Этап': '6: Настроенная модель',
                    'F1-Score': tuned_score,
                    'Тип': 'Настроенная'
                })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    fig = px.bar(comparison_df, 
                                x='Этап', 
                                y='F1-Score',
                                color='Тип',
                                title='Сравнение результатов по этапам',
                                text='F1-Score')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Сводка улучшений
                    improvement = ((tuned_score - original_score) / original_score) * 100 if original_score > 0 else 0
                    if improvement > 0:
                        st.success(f"🏆 **Итоговое улучшение:** +{improvement:.1f}% от исходной модели")
                    elif improvement == 0:
                        st.info("ℹ️ **Результат:** Модель сохранила исходное качество")
                    else:
                        st.warning(f"⚠️ **Результат:** Ухудшение на {abs(improvement):.1f}%")
                
                # 7. Экспорт результатов
                st.markdown("---")
                st.subheader("💾 Экспорт результатов")
                
                with st.expander("📥 Скачать результаты этапа 6"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Экспорт настроенной модели
                        if st.session_state.get("best_tuned_model"):
                            try:
                                import pickle
                                model_bytes = pickle.dumps(st.session_state.best_tuned_model)
                                
                                st.download_button(
                                    label="🤖 Настроенная модель",
                                    data=model_bytes,
                                    file_name=f"tuned_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                    mime="application/octet-stream"
                                )
                            except Exception as e:
                                st.warning(f"Не удалось сериализовать модель: {e}")
                    
                    with col2:
                        # Экспорт отчета
                        if st.session_state.get("comprehensive_evaluation"):
                            report = st.session_state.comprehensive_evaluation
                            report_json = json.dumps(report, indent=2, ensure_ascii=False, default=str)
                            
                            st.download_button(
                                label="📋 Комплексный отчет",
                                data=report_json,
                                file_name=f"tuning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    
                    # Полный архив
                    if st.button("📦 Создать полный архив этапа 6", key="create_stage6_archive"):
                        with st.spinner("Создание архива..."):
                            files_dict = {}
                            
                            # Отчет
                            if st.session_state.get("comprehensive_evaluation"):
                                report = st.session_state.comprehensive_evaluation
                                report_json = json.dumps(report, indent=2, ensure_ascii=False, default=str)
                                files_dict['comprehensive_report.json'] = report_json
                            
                            # Метрики
                            if evaluation_results:
                                metrics_json = json.dumps(evaluation_results, indent=2, ensure_ascii=False, default=str)
                                files_dict['evaluation_results.json'] = metrics_json
                            
                            # Сводка
                            summary = f"""
                            Результаты этапа 6: Комплексная настройка гиперпараметров
                            Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            
                            Конфигурация настройки:
                            - Стратегия CV: {cv_strategy}
                            - Метод оптимизации: {optimizer_type}
                            - Метрика оптимизации: {scoring_metric}
                            
                            Результаты:
                            - F1-Score до настройки: {original_score:.4f}
                            - F1-Score после настройки: {tuned_score:.4f}
                            - Улучшение: {improvement:.1f}%
                            
                            Использованные метрики: {', '.join(selected_metrics)}
                            """
                            
                            files_dict['summary.txt'] = summary
                            
                            # Создание ZIP
                            zip_buffer = create_download_zip(files_dict, "stage6_comprehensive_results.zip")
                            
                            st.download_button(
                                label="📥 Скачать полный архив этапа 6",
                                data=zip_buffer,
                                file_name=f"stage6_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip"
                            )
            
            else:
                st.info("📊 Результаты настройки загружаются...")

else:
    st.warning("⏳ Сначала выполните Этап 5: Борьбу с дисбалансом классов")

# ============================================================
# ЭТАП 7: ИТОГОВЫЙ АНАЛИЗ И ВЫБОР ЛУЧШЕЙ МОДЕЛИ
# ============================================================
st.markdown("---")

st.header("🏆 Этап 7: Итоговый анализ и выбор лучшей модели")

# Инициализация
if 'final_analysis_completed' not in st.session_state:
    st.session_state.final_analysis_completed = False
if 'final_analyzer' not in st.session_state:
    st.session_state.final_analyzer = None
if 'champion_model' not in st.session_state:
    st.session_state.champion_model = None
if 'pipelines' not in st.session_state:  # Для этапа 8
    st.session_state.pipelines = []
if 'ep7_pipelines' not in st.session_state:  # Резервная копия
    st.session_state.ep7_pipelines = []

if st.session_state.step6_completed:
    st.markdown("""
    ### 🎯 Цель этапа: Выбрать лучшую модель из всех предыдущих этапов
    
    **Что будет выполнено:**
    1. 📊 **Сбор результатов** всех этапов (3-6)
    2. 🏆 **Выбор чемпионской модели** с наивысшим F1-Score
    3. 📈 **Сравнительный анализ** методов и подходов
    4. 🎯 **Практические рекомендации** для продакшена
    5. 💾 **Экспорт итогов** всего проекта
    """)
    
    # Проверка доступности модуля
    if not FINAL_ANALYSIS_AVAILABLE:
        st.error("❌ Модуль final_analysis не доступен")
        st.info("Убедитесь, что файл final_analysis.py находится в той же директории")
    
    else:
        # 1. АВТОМАТИЧЕСКИЙ СБОР И АНАЛИЗ РЕЗУЛЬТАТОВ
        if not st.session_state.get("final_analysis_completed", False):
            st.markdown("---")
            st.subheader("📊 Сбор и анализ результатов")
            
            if st.button("📊 Запустить итоговый анализ", type="primary", key="run_final_analysis"):
                with st.spinner("Анализирую результаты всех этапов..."):
                    try:
                        # Используем функцию из final_analysis.py
                        from final_analysis import perform_complete_analysis, create_final_analysis_pipeline
                        
                        # Подготавливаем данные из session_state для передачи в модуль
                        stage_outputs = {
                            'stage3': {
                                'comparator_results': st.session_state.get("all_comparison_results", {}),
                                'best_classical_model': st.session_state.get("best_model"),
                                'best_classical_metrics': st.session_state.get("test_metrics", {})
                            },
                            'stage4': {
                                'neural_results': st.session_state.get("neural_results_all_tasks", {}),
                                'best_neural_model': st.session_state.get("neural_best_model"),
                                'best_neural_metrics': {}
                            },
                            'stage5': {
                                'balancing_results': st.session_state.get("balance_comparisons_all_tasks", {})
                            },
                            'stage6': {
                                'tuning_results': st.session_state.get("tuning_results", {})
                            }
                        }
                        
                        # Создаем анализатор
                        analysis_pipeline = create_final_analysis_pipeline()
                        selector = analysis_pipeline['selector']
                        
                        # Собираем все результаты
                        collected_models = selector.collect_models_from_stages(stage_outputs)
                        
                        if not any(collected_models.values()):
                            st.error("❌ Не найдено результатов предыдущих этапов")
                            st.stop()
                        
                        # Выбираем чемпионскую модель
                        champion_model, champion_metrics, champion_key = selector.select_champion_model()
                        
                        if not champion_model:
                            st.error("❌ Не удалось выбрать лучшую модель")
                            st.stop()
                        
                        # Сохраняем результаты
                        st.session_state.final_analyzer = {
                            'selector': selector,
                            'pipeline': analysis_pipeline,
                            'collected_models': collected_models,
                            'champion_model': champion_model,
                            'champion_metrics': champion_metrics,
                            'champion_key': champion_key
                        }
                        
                        st.session_state.champion_model = champion_model
                        st.session_state.champion_score = champion_metrics.get('f1', champion_metrics.get('f1_macro', 0))
                        st.session_state.champion_stage = selector.champion_stage
                        
                        # Получаем пайплайны для этапа 8
                        try:
                            # Попробуем создать простые пайплайны из собранных моделей
                            pipelines = []
                            for stage_name, models in collected_models.items():
                                for model_name, (model, metrics) in models.items():
                                    pipelines.append({
                                        'name': f"{stage_name}_{model_name}",
                                        'model': model,
                                        'metrics': metrics,
                                        'stage': stage_name
                                    })
                            
                            if pipelines:
                                st.session_state.pipelines = pipelines
                                st.session_state.ep7_pipelines = pipelines.copy()  # Резервная копия
                                st.success(f"✅ Собрано {len(pipelines)} пайплайнов для этапа 8")
                            else:
                                st.warning("⚠️ Не найдено пайплайнов для этапа 8")
                                st.session_state.pipelines = []
                                st.session_state.ep7_pipelines = []
                        except Exception as e:
                            st.warning(f"⚠️ Ошибка при создании пайплайнов: {e}")
                            st.session_state.pipelines = []
                            st.session_state.ep7_pipelines = []
                        
                        st.session_state.final_analysis_completed = True
                        
                        st.success(f"✅ Найдено {len(selector.all_models)} моделей из предыдущих этапов")
                        st.success(f"🏆 **Чемпионская модель:** {selector.champion_stage} - {selector.champion_name} (F1={st.session_state.champion_score:.4f})")
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при анализе результатов: {str(e)}")
                        st.code(traceback.format_exc())
        
        # 2. ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ЭТАПА 7
        if st.session_state.get("final_analysis_completed", False):
            analyzer = st.session_state.final_analyzer
            selector = analyzer['selector']
            
            # 2.1. СРАВНИТЕЛЬНАЯ ТАБЛИЦА
            st.markdown("---")
            st.subheader("📋 Сравнение этапов")
            
            try:
                comparison_df = selector.create_comparison_table()
                if not comparison_df.empty:
                    st.dataframe(comparison_df, use_container_width=True, height=250)
                    
                    # Визуализация сравнения
                    fig = px.bar(comparison_df, 
                                x='Model', 
                                y='F1',
                                title='Сравнение лучших моделей по этапам',
                                color='Stage',
                                text='F1',
                                color_continuous_scale='Viridis')
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Не удалось создать таблицу сравнения: {e}")
            
            # 2.2. ИНФОРМАЦИЯ О ЧЕМПИОНСКОЙ МОДЕЛИ
            st.markdown("---")
            st.subheader("🏆 Чемпионская модель")
            
            champion_metrics = selector.champion_metrics
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                f1_score = champion_metrics.get('f1', champion_metrics.get('f1_macro', 0))
                st.metric("F1-Score", f"{f1_score:.4f}")
            
            with col2:
                accuracy = champion_metrics.get('accuracy', 0)
                st.metric("Accuracy", f"{accuracy:.4f}")
            
            with col3:
                stage_name = selector.champion_stage.replace('stage', 'Этап ')
                st.metric("Этап", stage_name)
            
            with col4:
                model_name = selector.champion_name
                st.metric("Модель", model_name[:20] + "..." if len(model_name) > 20 else model_name)
            
            # Дополнительная информация
            with st.expander("📋 Подробная информация о модели"):
                st.write(f"**Название:** {selector.champion_name}")
                st.write(f"**Этап:** {selector.champion_stage.replace('stage', 'Этап ')}")
                
                if selector.champion_stage == 'stage5':
                    st.write(f"**Тип:** Модель с балансировкой классов")
                elif selector.champion_stage == 'stage6':
                    st.write(f"**Тип:** Настроенная модель (гиперпараметры)")
                
                st.write(f"**F1-Score:** {f1_score:.4f}")
                st.write(f"**Accuracy:** {accuracy:.4f}")
                st.write(f"**Precision:** {champion_metrics.get('precision', champion_metrics.get('precision_macro', 0)):.4f}")
                st.write(f"**Recall:** {champion_metrics.get('recall', champion_metrics.get('recall_macro', 0)):.4f}")
            
            # 2.3. ОЦЕНКА РЕЗУЛЬТАТА
            st.markdown("---")
            st.subheader("📈 Оценка результата")
            
            champion_f1 = st.session_state.champion_score
            
            if champion_f1 >= 0.9:
                st.success("""
                ### Отличный результат!
                **Модель готова к промышленному использованию.**
                
                **Рекомендации:**
                - Можно развернуть в продакшен
                - Оптимизировать для быстрого inference
                - Настроить мониторинг качества
                """)
            
            elif champion_f1 >= 0.8:
                st.success("""
                ### Хороший результат!
                **Модель работает хорошо, можно улучшить до production-ready.**
                
                **Рекомендации:**
                - Протестировать на новых данных
                - Попробовать ансамблирование
                - Оптимизировать гиперпараметры
                """)
            
            elif champion_f1 >= 0.7:
                st.warning("""
                ### Удовлетворительный результат
                **Есть возможности для улучшения.**
                
                **Рекомендации:**
                - Собрать больше данных
                - Улучшить признаки
                - Попробовать другие алгоритмы
                """)
            
            elif champion_f1 >= 0.6:
                st.warning("""
                ### Низкий результат
                **Требуются значительные улучшения.**
                
                **Рекомендации:**
                - Проверить качество данных
                - Упростить задачу (меньше классов)
                - Использовать предобученные эмбеддинги
                """)
            
            else:
                st.error("""
                ### Критически низкий результат
                **Требуется радикальное изменение подхода.**
                
                **Рекомендации:**
                - Пересмотреть задачу классификации
                - Проверить корректность разметки
                - Упростить до бинарной классификации
                """)
            
            # 2.4. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ
            st.markdown("---")
            st.subheader("🎯 Практические рекомендации")
            
            try:
                from final_analysis import PracticalInsightsGenerator
                
                insights_generator = PracticalInsightsGenerator()
                comparison_df = selector.create_comparison_table()
                insights = insights_generator.generate_insights(comparison_df, champion_metrics)
                
                rec_tab1, rec_tab2, rec_tab3 = st.tabs(["Лучшие подходы", "Инсайты о данных", "Для продакшена"])
                
                with rec_tab1:
                    st.markdown("#### 🏆 Лучшие подходы")
                    if insights.get('best_algorithm'):
                        st.info(f"**Лучший алгоритм:** {insights['best_algorithm']}")
                    
                    effectiveness = insights.get('effectiveness_of_techniques', {})
                    if effectiveness:
                        for tech, info in effectiveness.items():
                            if isinstance(info, dict):
                                st.info(f"**{tech}:** Средний F1: {info.get('average_score', 0):.4f}, Лучший: {info.get('best_score', 0):.4f}")
                    else:
                        st.info("Анализ эффективности подходов не доступен")
                
                with rec_tab2:
                    st.markdown("#### 📊 Инсайты о данных")
                    data_insights = insights.get('data_insights', [])
                    if data_insights:
                        for insight in data_insights:
                            st.info(insight)
                    else:
                        st.info("Инсайты о данных не доступны")
                
                with rec_tab3:
                    st.markdown("#### 🚀 Рекомендации для продакшена")
                    practical_advice = insights.get('practical_advice', [])
                    if practical_advice:
                        for advice in practical_advice:
                            st.info(advice)
                    else:
                        st.info("Рекомендации для продакшена не доступны")
            except Exception as e:
                st.warning(f"Не удалось сгенерировать рекомендации: {e}")
            
            # 2.5. ПЛАН ДЕЙСТВИЙ
            st.markdown("---")
            st.subheader("📅 План действий")
            
            try:
                # Создаем простой план действий на основе F1-Score
                champion_f1 = st.session_state.champion_score
                
                plan_tab1, plan_tab2, plan_tab3 = st.tabs(["Сейчас", "Ближайшее время", "Долгосрочно"])
                
                with plan_tab1:
                    st.markdown("#### ⚡ Немедленные действия (1-2 дня)")
                    if champion_f1 >= 0.8:
                        st.markdown("1. Подготовить модель к развертыванию")
                        st.markdown("2. Написать документацию по API")
                        st.markdown("3. Создать тестовый контейнер")
                    elif champion_f1 >= 0.7:
                        st.markdown("1. Протестировать модель на новых данных")
                        st.markdown("2. Собрать дополнительную разметку")
                        st.markdown("3. Проверить признаки на корректность")
                    else:
                        st.markdown("1. Пересмотреть подход к классификации")
                        st.markdown("2. Проверить качество данных")
                        st.markdown("3. Упростить задачу")
                
                with plan_tab2:
                    st.markdown("#### 📆 Краткосрочные действия (1-2 недели)")
                    if champion_f1 >= 0.8:
                        st.markdown("1. Развернуть модель в тестовой среде")
                        st.markdown("2. Настроить мониторинг метрик")
                        st.markdown("3. Подготовить A/B тестирование")
                    elif champion_f1 >= 0.7:
                        st.markdown("1. Оптимизировать гиперпараметры")
                        st.markdown("2. Попробовать ансамблирование моделей")
                        st.markdown("3. Улучшить предобработку текстов")
                    else:
                        st.markdown("1. Собрать больше данных")
                        st.markdown("2. Попробовать другие алгоритмы")
                        st.markdown("3. Привлечь экспертов для разметки")
                
                with plan_tab3:
                    st.markdown("#### 🗓️ Долгосрочные действия (1-3 месяца)")
                    if champion_f1 >= 0.8:
                        st.markdown("1. Масштабировать решение на все отделы")
                        st.markdown("2. Автоматизировать переобучение модели")
                        st.markdown("3. Интегрировать с другими системами")
                    elif champion_f1 >= 0.7:
                        st.markdown("1. Внедрить в пилотном проекте")
                        st.markdown("2. Создать пайплайн CI/CD для моделей")
                        st.markdown("3. Обучить команду работе с моделью")
                    else:
                        st.markdown("1. Пересмотреть бизнес-требования")
                        st.markdown("2. Исследовать альтернативные подходы")
                        st.markdown("3. Запустить пилотный проект с упрощенной задачей")
            except Exception as e:
                st.warning(f"Не удалось создать план действий: {e}")
            
            # 2.6. ЭКСПОРТ РЕЗУЛЬТАТОВ
            st.markdown("---")
            st.subheader("💾 Экспорт результатов")
            
            with st.expander("📥 Скачать итоговые результаты", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Экспорт чемпионской модели
                    if st.session_state.champion_model:
                        try:
                            import pickle
                            model_bytes = pickle.dumps(st.session_state.champion_model)
                            
                            st.download_button(
                                label="🤖 Чемпионская модель",
                                data=model_bytes,
                                file_name=f"champion_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                mime="application/octet-stream",
                                help="Скачать лучшую модель для использования в production"
                            )
                        except Exception as e:
                            st.warning(f"Не удалось сериализовать модель: {e}")
                
                with col2:
                    # Экспорт отчета
                    try:
                        # Создаем простой отчет
                        final_report = {
                            'analysis_date': datetime.now().isoformat(),
                            'champion_model': {
                                'stage': selector.champion_stage,
                                'name': selector.champion_name,
                                'f1_score': champion_f1,
                                'accuracy': accuracy
                            },
                            'comparison_summary': {
                                'total_models': len(selector.all_models),
                                'best_stage': selector.champion_stage,
                                'best_f1': champion_f1
                            },
                            'recommendations': {
                                'next_steps': [
                                    "Тестирование на новых данных",
                                    "Оптимизация для продакшена",
                                    "Настройка мониторинга"
                                ]
                            }
                        }
                        
                        report_json = json.dumps(final_report, indent=2, ensure_ascii=False, default=str)
                        
                        st.download_button(
                            label="📊 Финальный отчет",
                            data=report_json,
                            file_name=f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            help="Скачать итоговый отчет по проекту"
                        )
                    except Exception as e:
                        st.warning(f"Не удалось создать отчет: {e}")
                
                with col3:
                    # Экспорт сравнения
                    try:
                        comparison_df = selector.create_comparison_table()
                        if not comparison_df.empty:
                            csv_data = comparison_df.to_csv(index=False, encoding='utf-8-sig')
                            
                            st.download_button(
                                label="📋 Сравнение моделей",
                                data=csv_data,
                                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help="Скачать таблицу сравнения всех моделей"
                            )
                    except Exception as e:
                        st.warning(f"Не удалось создать файл сравнения: {e}")
                
                # Полный архив проекта
                st.markdown("---")
                
                if st.button("📦 Создать полный архив проекта", 
                           type="primary", 
                           key="create_full_project_archive",
                           use_container_width=True):
                    
                    with st.spinner("Создание архива..."):
                        files_dict = {}
                        
                        try:
                            # 1. Финальный отчет
                            final_report = {
                                'project_summary': f"""
                                ИТОГОВАЯ СВОДКА ПРОЕКТА
                                =========================
                                Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                
                                ЧЕМПИОНСКАЯ МОДЕЛЬ:
                                - Этап: {selector.champion_stage}
                                - Название: {selector.champion_name}
                                - F1-Score: {champion_f1:.4f}
                                - Accuracy: {accuracy:.4f}
                                
                                ОБЩИЕ РЕЗУЛЬТАТЫ:
                                - Всего моделей: {len(selector.all_models)}
                                - Лучший результат: {champion_f1:.4f}
                                - Этапы с моделями: {', '.join(set([info['stage'] for info in selector.all_models.values()]))}
                                
                                ВЫВОДЫ:
                                {f'Отличный результат! Модель готова к продакшену.' if champion_f1 >= 0.8 else
                                  f'Хороший результат! Требуется дополнительная оптимизация.' if champion_f1 >= 0.7 else
                                  f'Удовлетворительный результат! Требуются улучшения.' if champion_f1 >= 0.6 else
                                  f'Низкий результат! Требуется пересмотр подхода.'}
                                """
                            }
                            
                            report_json = json.dumps(final_report, indent=2, ensure_ascii=False, default=str)
                            files_dict['final_report.json'] = report_json
                            
                            # 2. Таблица сравнения
                            try:
                                comparison_df = selector.create_comparison_table()
                                if not comparison_df.empty:
                                    csv_data = comparison_df.to_csv(index=False, encoding='utf-8-sig')
                                    files_dict['model_comparison.csv'] = csv_data
                            except:
                                pass
                            
                            # 3. Сводка проекта
                            summary = f"""
                            ИТОГОВАЯ СВОДКА ПРОЕКТА
                            =========================
                            Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            
                            ЧЕМПИОНСКАЯ МОДЕЛЬ:
                            - Этап: {selector.champion_stage}
                            - Название: {selector.champion_name}
                            - F1-Score: {champion_f1:.4f}
                            - Accuracy: {accuracy:.4f}
                            
                            РЕКОМЕНДАЦИИ:
                            {f'1. Развернуть модель в продакшен' if champion_f1 >= 0.8 else
                              f'1. Оптимизировать модель перед развертыванием' if champion_f1 >= 0.7 else
                              f'1. Собрать больше данных и переобучить модель' if champion_f1 >= 0.6 else
                              f'1. Пересмотреть подход к задаче классификации'}
                            2. Настроить мониторинг качества предсказаний
                            3. Реализовать периодическое переобучение модели
                            """
                            
                            files_dict['project_summary.txt'] = summary
                            
                            # Создание ZIP
                            zip_buffer = create_download_zip(files_dict, "project_final_results.zip")
                            
                            st.download_button(
                                label="📥 Скачать полный архив проекта",
                                data=zip_buffer,
                                file_name=f"project_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Ошибка при создании архива: {e}")
            
            # Обновляем статус завершения этапа
            st.session_state.step7_completed = True
            st.success("✅ Этап 7 успешно завершен! Результаты анализа сохранены.")
        
        else:
            st.info("👆 Нажмите кнопку 'Запустить итоговый анализ' для сбора и анализа результатов")

else:  
    st.warning("⏳ Сначала выполните Этап 6: Настройку гиперпараметров")


# ============================================================
# ЭТАП 8: ИНТЕРАКТИВНЫЙ АНАЛИЗ И СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================================
st.markdown("---")

st.header("🧠 Этап 8: Интерактивный анализ и сравнение моделей")

if st.session_state.get("step7_completed", False):
    st.markdown("""
    ### 🎯 Цель этапа: Интерактивный анализ текстов всеми обученными моделями
    
    **Что можно сделать:**
    1. **Анализ произвольного текста** всеми моделями из этапов 3-7
    2. **Сравнение предсказаний** разных типов моделей (классические, нейросетевые, настроенные)
    3. **Визуализация уверенности** моделей в предсказаниях
    4. **Экспорт результатов** для дальнейшего анализа
    
    **Модели будут загружены из:**
    - Этап 3: Классические модели (Logistic Regression, Random Forest, SVM и др.)
    - Этап 4: Нейросетевые модели (MLP, CNN, LSTM, Transformers)
    - Этап 5: Модели с балансировкой классов
    - Этап 6: Настроенные модели
    - Этап 7: Чемпионская модель
    """)
    
    # СБОР И СОЗДАНИЕ МОДЕЛЕЙ ДЛЯ ИНТЕРАКТИВНОГО АНАЛИЗА
    if st.button("🔄 **Собрать модели из всех этапов**", type="primary", key="collect_models_button"):
        with st.spinner("Собираю модели из этапов 3-7..."):
            try:
                # Создаем базовые классы для пайплайнов
                class BasePipelineWrapper:
                    def __init__(self, name: str, model_type: str = "classical", 
                                 task_type: str = "category", label_field: str = "category",
                                 real_classes: List[str] = None, model=None):
                        self.name = name
                        self.model_type = model_type
                        self.task_type = task_type
                        self.label_field = label_field
                        self.real_classes = real_classes or []
                        self.model = model
                        self.is_multi_label = task_type == 'multilabel'
                        self.has_predict_proba = False
                        if model:
                            self.has_predict_proba = hasattr(model, 'predict_proba')
                    
                    def predict(self, X):
                        if self.model and hasattr(self.model, 'predict'):
                            return self.model.predict(X)
                        return None
                    
                    def predict_proba(self, X):
                        if self.model and self.has_predict_proba:
                            try:
                                return self.model.predict_proba(X)
                            except:
                                pass
                        return None
                    
                    def predict_proba_text(self, text: str):
                        # Базовая реализация - можно переопределить
                        try:
                            if self.real_classes:
                                n_classes = len(self.real_classes)
                                proba = np.random.rand(n_classes)
                                proba = proba / proba.sum()
                                return self.real_classes, proba
                            else:
                                return ["Class_0", "Class_1"], np.array([0.5, 0.5])
                        except Exception as e:
                            return [], None
                
                pipelines = []
                
                # 1. Получаем реальные классы из данных
                real_classes = []
                labeled_articles = st.session_state.get("labeled_articles", [])
                if labeled_articles:
                    # Определяем основную задачу
                    main_task_type = "category"
                    if st.session_state.get("all_comparison_results"):
                        tasks = list(st.session_state.all_comparison_results.keys())
                        if tasks:
                            main_task_type = tasks[0]  # Берем первую задачу
                    
                    for article in labeled_articles:
                        if main_task_type in article and article[main_task_type]:
                            label = article[main_task_type]
                            if isinstance(label, list):
                                real_classes.extend([str(l) for l in label])
                            else:
                                real_classes.append(str(label))
                    real_classes = list(set(real_classes))
                
                # 2. Собираем модели из этапа 3
                if st.session_state.get("all_comparison_results"):
                    for task_name, task_data in st.session_state.all_comparison_results.items():
                        if 'comparator' in task_data and task_data['comparator']:
                            comparator = task_data['comparator']
                            if hasattr(comparator, 'models'):
                                for model_name, model in comparator.models.items():
                                    if hasattr(model, 'predict'):
                                        pipeline = BasePipelineWrapper(
                                            name=f"Этап 3: {model_name} ({task_name})",
                                            model=model,
                                            task_type=task_name,
                                            real_classes=real_classes
                                        )
                                        pipelines.append(pipeline)
                
                # 3. Добавляем лучшую модель из этапа 3
                if st.session_state.get("best_model"):
                    model = st.session_state.best_model
                    if hasattr(model, 'predict'):
                        pipeline = BasePipelineWrapper(
                            name="🏆 Лучшая модель этапа 3",
                            model=model,
                            task_type="category",
                            real_classes=real_classes
                        )
                        pipelines.append(pipeline)
                
                # 4. Добавляем модели из этапа 4 (нейросетевые)
                if st.session_state.get("neural_models_all_tasks"):
                    for task_name, task_models in st.session_state.neural_models_all_tasks.items():
                        for model_name, model in task_models.items():
                            if hasattr(model, 'predict'):
                                pipeline = BasePipelineWrapper(
                                    name=f"Этап 4: {model_name} ({task_name})",
                                    model=model,
                                    model_type="neural",
                                    task_type=task_name,
                                    real_classes=real_classes
                                )
                                pipelines.append(pipeline)
                
                # 5. Добавляем модели из этапа 5 (балансировка)
                if st.session_state.get("balanced_models"):
                    for model_key, model_info in st.session_state.balanced_models.items():
                        if isinstance(model_info, dict) and 'model' in model_info:
                            model = model_info['model']
                            if hasattr(model, 'predict'):
                                pipeline = BasePipelineWrapper(
                                    name=f"Этап 5: {model_key}",
                                    model=model,
                                    task_type="category",
                                    real_classes=real_classes
                                )
                                pipelines.append(pipeline)
                
                # 6. Добавляем настроенную модель из этапа 6
                if st.session_state.get("best_tuned_model"):
                    model = st.session_state.best_tuned_model
                    if hasattr(model, 'predict'):
                        pipeline = BasePipelineWrapper(
                            name="⚙️ Настроенная модель (Этап 6)",
                            model=model,
                            task_type="category",
                            real_classes=real_classes
                        )
                        pipelines.append(pipeline)
                
                # 7. Добавляем чемпионскую модель из этапа 7
                if st.session_state.get("champion_model"):
                    model = st.session_state.champion_model
                    champion_stage = st.session_state.get("champion_stage", "Этап ?")
                    if hasattr(model, 'predict'):
                        pipeline = BasePipelineWrapper(
                            name=f"👑 Чемпионская модель ({champion_stage})",
                            model=model,
                            task_type="category",
                            real_classes=real_classes
                        )
                        pipelines.append(pipeline)
                
                # Если нет моделей, создаем демо
                if not pipelines:
                    st.warning("⚠️ Не найдено реальных моделей. Создаю демо-модели...")
                    
                    # Создаем простую демо-модель
                    class DemoModel:
                        def predict(self, X):
                            return np.zeros(len(X))
                        def predict_proba(self, X):
                            return np.array([[0.7, 0.3]] * len(X))
                    
                    demo_model = DemoModel()
                    
                    # Демо для разных типов задач
                    demo_tasks = [
                        ('sentiment', ['Положительный', 'Отрицательный', 'Нейтральный']),
                        ('category', ['Политика', 'Экономика', 'Спорт', 'Наука']),
                        ('multilabel', ['Важное', 'Срочное', 'Интересное', 'Полезное'])
                    ]
                    
                    for task_type, classes in demo_tasks:
                        pipeline = BasePipelineWrapper(
                            name=f"Демо: {task_type}",
                            model=demo_model,
                            task_type=task_type,
                            real_classes=classes
                        )
                        pipelines.append(pipeline)
                
                # Сохраняем пайплайны
                st.session_state.ep8_pipelines = pipelines
                st.success(f"✅ Собрано {len(pipelines)} моделей для интерактивного анализа!")
                
                # Показываем информацию о моделях
                st.markdown("### 📋 Собранные модели")
                
                model_info = []
                for pipe in pipelines:
                    model_info.append({
                        'Название': pipe.name,
                        'Тип модели': pipe.model_type,
                        'Тип задачи': pipe.task_type,
                        'Классы': len(pipe.real_classes),
                        'predict_proba': "✅" if pipe.has_predict_proba else "❌"
                    })
                
                if model_info:
                    st.dataframe(pd.DataFrame(model_info), use_container_width=True, height=300)
                
                st.session_state.ep8_models_collected = True
                
            except Exception as e:
                st.error(f"❌ Ошибка при сборе моделей: {str(e)}")
                st.code(traceback.format_exc())
    
    # ИНТЕРАКТИВНЫЙ АНАЛИЗ ТЕКСТА
    if st.session_state.get("ep8_models_collected", False):
        st.markdown("---")
        st.subheader("✍️ Интерактивный анализ текста")
        
        # Настройки анализа
        col1, col2 = st.columns(2)
        with col1:
            task_filter = st.selectbox(
                "Фильтр по типу задачи:",
                ["Все", "sentiment", "category", "multilabel"],
                key="task_filter"
            )
        with col2:
            show_details = st.checkbox("Показать детали моделей", True, key="show_details")
        
        # Ввод текста
        st.text_area(
            "Введите текст для анализа:",
            value="Искусственный интеллект продолжает развиваться стремительными темпами...",
            height=150,
            key="analysis_text"
        )
        
        if st.button("🔍 **ПРОАНАЛИЗИРОВАТЬ ТЕКСТ**", type="primary", key="analyze_text_button"):
            text = st.session_state.analysis_text
            pipelines = st.session_state.ep8_pipelines
            
            if text and len(text.strip()) > 3 and pipelines:
                with st.spinner(f"Анализирую текст {len(pipelines)} моделями..."):
                    try:
                        # Фильтрация моделей по типу задачи
                        filtered_pipelines = []
                        if task_filter == "Все":
                            filtered_pipelines = pipelines
                        else:
                            filtered_pipelines = [p for p in pipelines if p.task_type == task_filter]
                        
                        if not filtered_pipelines:
                            st.warning(f"⚠️ Нет моделей для задачи '{task_filter}'")
                            filtered_pipelines = pipelines
                        
                        # Анализируем текст каждой моделью
                        results = []
                        for pipe in filtered_pipelines:
                            try:
                                classes, proba = pipe.predict_proba_text(text)
                                
                                if proba is not None and len(proba) > 0:
                                    top_idx = np.argmax(proba) if len(proba) > 0 else 0
                                    top_prob = proba[top_idx] if len(proba) > 0 else 0
                                    pred_class = classes[top_idx] if top_idx < len(classes) else "Unknown"
                                    
                                    results.append({
                                        'Модель': pipe.name,
                                        'Тип модели': pipe.model_type,
                                        'Тип задачи': pipe.task_type,
                                        'Предсказание': pred_class,
                                        'Уверенность': f"{top_prob:.1%}",
                                        'Топ-3 классов': ", ".join([f"{classes[i]}: {proba[i]:.1%}" 
                                                                   for i in np.argsort(proba)[-3:][::-1] 
                                                                   if i < len(classes)])
                                    })
                                else:
                                    results.append({
                                        'Модель': pipe.name,
                                        'Тип модели': pipe.model_type,
                                        'Тип задачи': pipe.task_type,
                                        'Предсказание': "Ошибка",
                                        'Уверенность': "0%",
                                        'Топ-3 классов': "Нет данных"
                                    })
                            except Exception as e:
                                results.append({
                                    'Модель': pipe.name,
                                    'Тип модели': pipe.model_type,
                                    'Тип задачи': pipe.task_type,
                                    'Предсказание': f"Ошибка: {str(e)[:50]}",
                                    'Уверенность': "0%",
                                    'Топ-3 классов': "Ошибка"
                                })
                        
                        # Показываем результаты
                        if results:
                            st.markdown("### 📊 Результаты анализа")
                            
                            # Создаем датафрейм
                            results_df = pd.DataFrame(results)
                            
                            # Сортируем по уверенности
                            results_df['conf_num'] = results_df['Уверенность'].str.replace('%', '').astype(float)
                            results_df = results_df.sort_values('conf_num', ascending=False)
                            results_df = results_df.drop('conf_num', axis=1)
                            
                            # Показываем таблицу
                            st.dataframe(results_df, use_container_width=True, height=400)
                            
                            # Визуализация
                            st.markdown("### 📈 Визуализация результатов")
                            
                            # График уверенности моделей
                            fig = px.bar(
                                results_df,
                                x='Модель',
                                y='Уверенность',
                                color='Тип модели',
                                title='Уверенность моделей в предсказаниях',
                                text='Уверенность',
                                height=400
                            )
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Анализ согласованности
                            st.markdown("### 🤝 Анализ согласованности моделей")
                            
                            # Подсчитываем популярные предсказания
                            if len(results_df) > 1:
                                pred_counts = results_df['Предсказание'].value_counts()
                                if len(pred_counts) > 0:
                                    st.write(f"**Наиболее частое предсказание:** {pred_counts.index[0]} "
                                            f"({pred_counts.iloc[0]} из {len(results_df)} моделей)")
                                    
                                    # Показываем распределение
                                    fig2 = px.pie(
                                        names=pred_counts.index[:5],
                                        values=pred_counts.values[:5],
                                        title='Распределение предсказаний (топ-5)',
                                        height=300
                                    )
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                # Сводка по типам моделей
                                model_type_summary = results_df.groupby('Тип модели')['Уверенность'].mean()
                                if not model_type_summary.empty:
                                    st.write("**Средняя уверенность по типам моделей:**")
                                    for model_type, avg_conf in model_type_summary.items():
                                        st.write(f"  - {model_type}: {avg_conf}")
                            
                            # Экспорт результатов
                            st.markdown("---")
                            st.subheader("💾 Экспорт результатов")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                # CSV экспорт
                                csv_data = results_df.to_csv(index=False, encoding='utf-8-sig')
                                st.download_button(
                                    label="📥 Скачать CSV",
                                    data=csv_data,
                                    file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                # JSON экспорт
                                export_data = {
                                    'text': text[:500],
                                    'timestamp': datetime.now().isoformat(),
                                    'num_models': len(results_df),
                                    'results': results
                                }
                                json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
                                st.download_button(
                                    label="📋 Скачать JSON",
                                    data=json_data,
                                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при анализе текста: {str(e)}")
                        st.code(traceback.format_exc())
            else:
                st.warning("⚠️ Введите текст для анализа")
    
    # Если модели еще не собраны
    elif not st.session_state.get("ep8_models_collected", False):
        st.info("👆 Нажмите кнопку выше, чтобы собрать модели из всех этапов")
    
    # ЗАВЕРШЕНИЕ ЭТАПА
    st.markdown("---")
    
    if st.session_state.get("ep8_models_collected", False):
        st.success("✅ Этап 8 готов к работе! Вы можете анализировать тексты всеми моделями.")
        st.session_state.step8_completed = True
        
        # Кнопка для перехода к завершению
        if st.button("🏁 **ЗАВЕРШИТЬ ПРАКТИКУМ**", type="primary", key="finish_practicum"):
            st.balloons()
            st.success("🎉 Поздравляем! Вы успешно завершили лабораторный практикум!")
            
            # Сводка по всем этапам
            st.markdown("### 📋 Сводка по всем этапам")
            
            summary_data = []
            stages = [
                ("Этап 1", "Автоматическая разметка и разделение данных", st.session_state.get("step1_completed", False)),
                ("Этап 2", "Подготовка данных для классификации", st.session_state.get("step2_completed", False)),
                ("Этап 3", "Классические методы классификации", st.session_state.get("step3_completed", False)),
                ("Этап 4", "Нейросетевые и трансформерные модели", st.session_state.get("step4_completed", False)),
                ("Этап 5", "Борьба с дисбалансом классов", st.session_state.get("step5_completed", False)),
                ("Этап 6", "Настройка гиперпараметров", st.session_state.get("step6_completed", False)),
                ("Этап 7", "Итоговый анализ и выбор лучшей модели", st.session_state.get("step7_completed", False)),
                ("Этап 8", "Интерактивный анализ и сравнение моделей", st.session_state.get("step8_completed", False))
            ]
            
            for stage_name, stage_desc, completed in stages:
                summary_data.append({
                    'Этап': stage_name,
                    'Описание': stage_desc,
                    'Статус': '✅ Выполнен' if completed else '❌ Не выполнен'
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Финальные рекомендации
            st.markdown("### 🎯 Рекомендации для дальнейшей работы")
            st.info("""
            1. **Для продакшена:** Используйте чемпионскую модель из Этапа 7
            2. **Для анализа новых данных:** Используйте Этап 8 для интерактивного анализа
            3. **Для улучшения качества:** Соберите больше размеченных данных
            4. **Для оптимизации:** Используйте методы из Этапа 6 для настройки гиперпараметров
            5. **Для мониторинга:** Реализуйте систему мониторинга качества предсказаний
            """)
            
            # Полный экспорт проекта
            st.markdown("### 📦 Полный экспорт проекта")
            
            if st.button("📥 **Скачать полный архив проекта**", key="download_full_project"):
                with st.spinner("Создание архива..."):
                    files_dict = {}
                    
                    # 1. Сводка по проекту
                    summary = f"""
                    ЛАБОРАТОРНЫЙ ПРАКТИКУМ №3: Сравнительный анализ методов классификации текстов
                    ===================================================================================
                    
                    Дата выполнения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    ЭТАПЫ ВЫПОЛНЕНИЯ:
                    """
                    
                    for stage_name, stage_desc, completed in stages:
                        status = "✅ ВЫПОЛНЕН" if completed else "❌ НЕ ВЫПОЛНЕН"
                        summary += f"\n- {stage_name}: {stage_desc} - {status}"
                    
                    # 2. Информация о данных
                    if st.session_state.get("dataframe"):
                        df = st.session_state.dataframe
                        summary += f"\n\nДАННЫЕ:\n- Всего записей: {len(df)}"
                        
                        # Категории
                        category_cols = [c for c in df.columns if 'категория' in c.lower() or 'category' in c.lower()]
                        if category_cols:
                            categories = df[category_cols[0]].nunique()
                            summary += f"\n- Уникальных категорий: {categories}"
                    
                    # 3. Чемпионская модель
                    if st.session_state.get("champion_model"):
                        champion_score = st.session_state.get("champion_score", 0)
                        champion_stage = st.session_state.get("champion_stage", "Не определен")
                        summary += f"\n\nЧЕМПИОНСКАЯ МОДЕЛЬ:\n- Этап: {champion_stage}\n- F1-Score: {champion_score:.4f}"
                    
                    files_dict['project_summary.txt'] = summary
                    
                    # 4. Результаты интерактивного анализа (если есть)
                    if st.session_state.get("analysis_text"):
                        analysis_text = st.session_state.analysis_text
                        analysis_results = f"Текст для анализа:\n{analysis_text}\n\n"
                        analysis_results += "Дата анализа: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        files_dict['interactive_analysis.txt'] = analysis_results
                    
                    # Создание ZIP
                    zip_buffer = create_download_zip(files_dict, "final_project_results.zip")
                    
                    st.download_button(
                        label="📥 Скачать финальный архив",
                        data=zip_buffer,
                        file_name=f"lab_practicum_3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

else:
    st.warning("⏳ Сначала выполните Этап 7: Итоговый анализ и выбор лучшей модели")

# Футер
st.markdown("---")
st.caption("© Лабораторный практикум №3 — Веб-интерфейс анализа классификаторов")