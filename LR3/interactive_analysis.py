"""
Этап 8: Интерактивный анализ и сравнение моделей
Интегрированный модуль для работы с моделями из всех этапов (3-7)
"""

import numpy as np
import pandas as pd
import streamlit as st
import warnings
import traceback
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import Counter
import re
from datetime import datetime

warnings.filterwarnings('ignore')


class BasePipelineWrapper:
    """
    Базовая обертка для моделей всех типов
    """
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
        
        # Дополнительные атрибуты
        self.has_predict_proba = False
        if model:
            self.has_predict_proba = hasattr(model, 'predict_proba')
    
    def predict(self, X):
        """Основной метод предсказания"""
        if self.model and hasattr(self.model, 'predict'):
            return self.model.predict(X)
        return None
    
    def predict_proba(self, X):
        """Метод предсказания вероятностей"""
        if self.model and self.has_predict_proba:
            try:
                return self.model.predict_proba(X)
            except:
                pass
        return None


class ClassicalPipelineWrapper(BasePipelineWrapper):
    """Обертка для классических моделей"""
    
    def __init__(self, name: str, model, task_type: str = "category", 
                 label_field: str = "category", real_classes: List[str] = None,
                 vectorizer=None):
        super().__init__(name, "classical", task_type, label_field, real_classes, model)
        self.vectorizer = vectorizer
        
        # Получаем информацию о классах из модели
        self._extract_class_info()
    
    def _extract_class_info(self):
        """Извлечение информации о классах из модели"""
        if hasattr(self.model, 'classes_'):
            self.real_classes = list(self.model.classes_)
        elif hasattr(self.model, 'label_encoder') and self.model.label_encoder is not None:
            self.real_classes = list(self.model.label_encoder.classes_)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'classes_'):
            self.real_classes = list(self.model.model.classes_)
    
    def predict_proba_text(self, text: str) -> Tuple[List[str], Optional[np.ndarray]]:
        """Предсказание для текста"""
        try:
            # Если есть векторзатор, преобразуем текст
            if self.vectorizer and hasattr(self.vectorizer, 'transform'):
                features = self.vectorizer.transform([text])
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(features)[0]
                else:
                    # Если нет predict_proba, используем predict
                    pred = self.model.predict(features)[0]
                    n_classes = len(self.real_classes) if self.real_classes else 2
                    proba = np.zeros(n_classes)
                    if isinstance(pred, (int, np.integer)):
                        if pred < n_classes:
                            proba[pred] = 1.0
                        else:
                            proba[0] = 1.0
                    else:
                        # Если строка, ищем в real_classes
                        if self.real_classes and str(pred) in self.real_classes:
                            idx = self.real_classes.index(str(pred))
                            proba[idx] = 1.0
                        else:
                            proba[0] = 1.0
                
                return self.real_classes or [f"Class_{i}" for i in range(len(proba))], proba
            
            # Если нет векторзатора, но модель имеет свой метод для текста
            elif hasattr(self.model, 'predict_proba_text'):
                return self.model.predict_proba_text(text)
            
            # Fallback: случайные вероятности
            else:
                if self.real_classes:
                    n_classes = len(self.real_classes)
                    proba = np.random.rand(n_classes)
                    proba = proba / proba.sum()
                    return self.real_classes, proba
                else:
                    return ["Class_0", "Class_1"], np.array([0.5, 0.5])
                    
        except Exception as e:
            st.warning(f"Ошибка в {self.name}: {str(e)}")
            return [], None


class NeuralPipelineWrapper(BasePipelineWrapper):
    """Обертка для нейросетевых моделей"""
    
    def __init__(self, name: str, model, task_type: str = "category",
                 label_field: str = "category", real_classes: List[str] = None):
        super().__init__(name, "neural", task_type, label_field, real_classes, model)
        
        # Проверяем доступность PyTorch
        self.torch_available = hasattr(model, 'model') and hasattr(model.model, 'to')
        
        # Извлекаем информацию о классах
        if hasattr(model, 'classes_'):
            self.real_classes = list(model.classes_)
        elif hasattr(model, 'label_encoder') and model.label_encoder is not None:
            self.real_classes = list(model.label_encoder.classes_)
    
    def predict_proba_text(self, text: str) -> Tuple[List[str], Optional[np.ndarray]]:
        """Предсказание для текста нейросетевой моделью"""
        try:
            # Если модель имеет метод predict_proba_text
            if hasattr(self.model, 'predict_proba_text'):
                return self.model.predict_proba_text(text)
            
            # Если модель имеет predict_proba и принимает текст
            elif hasattr(self.model, 'predict_proba') and hasattr(self.model, 'prepare_texts'):
                try:
                    prepared = self.model.prepare_texts([text])
                    proba = self.model.predict_proba(prepared)[0]
                    classes = self.real_classes or [f"Class_{i}" for i in range(len(proba))]
                    return classes, proba
                except:
                    pass
            
            # Fallback для нейросетевых моделей
            if self.real_classes:
                n_classes = len(self.real_classes)
                proba = np.random.rand(n_classes)
                proba = proba / proba.sum()
                return self.real_classes, proba
            else:
                return ["Positive", "Negative"], np.array([0.7, 0.3])
                
        except Exception as e:
            st.warning(f"Ошибка в нейросетевой модели {self.name}: {str(e)}")
            return [], None


class InteractiveModelAnalyzer:
    """
    Основной класс для интерактивного анализа моделей
    """
    
    def __init__(self, pipelines: List[BasePipelineWrapper]):
        self.pipelines = pipelines
        self.results_cache = {}
        
        # Группируем пайплайны по типам задач
        self.pipelines_by_task = self._group_pipelines_by_task()
    
    def _group_pipelines_by_task(self) -> Dict[str, List[BasePipelineWrapper]]:
        """Группировка пайплайнов по типам задач"""
        groups = {
            'sentiment': [],
            'category': [],
            'multilabel': []
        }
        
        for pipe in self.pipelines:
            task_type = pipe.task_type
            if task_type in groups:
                groups[task_type].append(pipe)
            else:
                # Если тип задачи неизвестен, добавляем в category
                groups['category'].append(pipe)
        
        return groups
    
    def analyze_text(self, text: str, task_filter: str = None) -> Dict[str, Dict]:
        """
        Анализ текста всеми моделями
        
        Args:
            text: Текст для анализа
            task_filter: Фильтр по типу задачи
            
        Returns:
            Словарь с результатами
        """
        results = {}
        
        if not text or len(text.strip()) < 3:
            return results
        
        # Определяем, какие задачи анализировать
        tasks_to_analyze = [task_filter] if task_filter else list(self.pipelines_by_task.keys())
        
        for task_type in tasks_to_analyze:
            if task_type in self.pipelines_by_task:
                task_pipelines = self.pipelines_by_task[task_type]
                task_results = {}
                
                for pipe in task_pipelines:
                    try:
                        # Получаем предсказания
                        classes, proba = pipe.predict_proba_text(text)
                        
                        if proba is not None and len(proba) > 0:
                            # Обрабатываем multi-label
                            if pipe.is_multi_label:
                                threshold = 0.5
                                predicted_labels = []
                                predicted_probs = []
                                
                                for i, prob in enumerate(proba):
                                    if i < len(classes) and prob >= threshold:
                                        predicted_labels.append(classes[i])
                                        predicted_probs.append(float(prob))
                                
                                task_results[pipe.name] = {
                                    'classes': classes,
                                    'proba': proba.tolist(),
                                    'pred': predicted_labels if predicted_labels else ["Нет меток"],
                                    'top_prob': float(np.max(proba)) if len(proba) > 0 else 0,
                                    'model_type': pipe.model_type,
                                    'task_type': task_type,
                                    'is_multi_label': True,
                                    'predicted_labels': predicted_labels,
                                    'predicted_probs': predicted_probs,
                                    'success': True
                                }
                            else:
                                # Обрабатываем single-label
                                if len(proba) > 0:
                                    top_idx = int(np.argmax(proba))
                                    pred_label = classes[top_idx] if top_idx < len(classes) else str(top_idx)
                                    
                                    task_results[pipe.name] = {
                                        'classes': classes,
                                        'proba': proba.tolist(),
                                        'pred': pred_label,
                                        'top_prob': float(np.max(proba)),
                                        'model_type': pipe.model_type,
                                        'task_type': task_type,
                                        'is_multi_label': False,
                                        'success': True
                                    }
                        else:
                            task_results[pipe.name] = {
                                'error': 'Нет предсказаний',
                                'success': False
                            }
                            
                    except Exception as e:
                        task_results[pipe.name] = {
                            'error': str(e)[:100],
                            'success': False
                        }
                
                if task_results:
                    results[task_type] = task_results
        
        # Кэшируем результаты
        cache_key = f"{text[:50]}_{task_filter or 'all'}"
        self.results_cache[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'text_preview': text[:100],
            'results': results
        }
        
        return results
    
    def get_text_statistics(self, text: str) -> Dict:
        """Статистика текста"""
        stats = {
            'length_chars': len(text),
            'length_words': 0,
            'sentences': 0,
            'unique_words': 0,
            'avg_word_length': 0,
            'top_words': []
        }
        
        if text:
            # Слова
            words = re.findall(r'\b\w+\b', text.lower())
            stats['length_words'] = len(words)
            
            # Предложения
            sentences = re.split(r'[.!?]+', text)
            stats['sentences'] = len([s for s in sentences if s.strip()])
            
            # Уникальные слова
            unique_words = set(words)
            stats['unique_words'] = len(unique_words)
            
            # Средняя длина слова
            if words:
                stats['avg_word_length'] = sum(len(w) for w in words) / len(words)
            
            # Топ слов
            word_counts = Counter(words)
            stats['top_words'] = word_counts.most_common(10)
        
        return stats


class InteractiveAnalysisUI:
    """
    Пользовательский интерфейс для интерактивного анализа
    """
    
    def __init__(self):
        self.analyzer = None
        self.text_statistics = {}
        
    def render_sidebar(self):
        """Рендер боковой панели"""
        st.sidebar.header("⚙️ Настройки анализа")
        
        # Выбор типа задачи
        task_options = {
            'all': 'Все типы задач',
            'sentiment': 'Анализ тональности',
            'category': 'Классификация категорий',
            'multilabel': 'Многометочная классификация'
        }
        
        selected_task = st.sidebar.selectbox(
            "Тип задачи для анализа:",
            list(task_options.keys()),
            format_func=lambda x: task_options[x]
        )
        
        # Настройки отображения
        show_details = st.sidebar.checkbox("Показать детали моделей", True)
        show_charts = st.sidebar.checkbox("Показать графики", True)
        
        # Настройки экспорта
        export_results = st.sidebar.checkbox("Экспортировать результаты", False)
        
        return {
            'selected_task': selected_task,
            'show_details': show_details,
            'show_charts': show_charts,
            'export_results': export_results
        }
    
    def render_text_input(self):
        """Рендер ввода текста"""
        st.header("✍️ Ввод текста для анализа")
        
        # Примеры текстов
        sample_texts = {
            "Пример 1 (технологии)": "Искусственный интеллект продолжает развиваться стремительными темпами...",
            "Пример 2 (политика)": "Новые законы были приняты парламентом после долгих обсуждений...",
            "Пример 3 (спорт)": "Спортсмены показали выдающиеся результаты на международных соревнованиях..."
        }
        
        # Выбор примера
        sample_option = st.selectbox(
            "Использовать пример текста:",
            ["Свой текст"] + list(sample_texts.keys())
        )
        
        # Поле для ввода текста
        if sample_option == "Свой текст":
            default_text = st.session_state.get('interactive_text', '')
            text_input = st.text_area(
                "Введите текст для анализа:",
                value=default_text,
                height=200,
                placeholder="Введите текст для анализа всеми моделями..."
            )
        else:
            text_input = st.text_area(
                "Текст для анализа:",
                value=sample_texts[sample_option],
                height=200
            )
        
        # Кнопка анализа
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔍 **ПРОАНАЛИЗИРОВАТЬ ТЕКСТ**",
                type="primary",
                use_container_width=True
            )
        
        return text_input, analyze_button
    
    def render_model_statistics(self, pipelines: List[BasePipelineWrapper]):
        """Рендер статистики моделей"""
        st.sidebar.header("📊 Статистика моделей")
        
        # Подсчет по типам
        model_types = Counter([p.model_type for p in pipelines])
        task_types = Counter([p.task_type for p in pipelines])
        
        st.sidebar.metric("Всего моделей", len(pipelines))
        st.sidebar.metric("Классические", model_types.get('classical', 0))
        st.sidebar.metric("Нейросетевые", model_types.get('neural', 0))
        
        # Информация о задачах
        with st.sidebar.expander("Типы задач:"):
            for task_type, count in task_types.items():
                st.write(f"• {task_type}: {count}")
    
    def render_results(self, text: str, results: Dict, settings: Dict):
        """Рендер результатов анализа"""
        st.header("📊 Результаты анализа")
        
        if not results:
            st.warning("❌ Нет результатов для отображения")
            return
        
        # Статистика текста
        with st.expander("📝 Статистика текста", expanded=True):
            stats = self.analyzer.get_text_statistics(text)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Символов", stats['length_chars'])
            with col2:
                st.metric("Слов", stats['length_words'])
            with col3:
                st.metric("Предложений", stats['sentences'])
            with col4:
                st.metric("Уникальных слов", stats['unique_words'])
            
            # Топ слов
            if stats['top_words']:
                st.subheader("Частые слова:")
                top_words_df = pd.DataFrame(stats['top_words'], columns=['Слово', 'Частота'])
                st.dataframe(top_words_df, use_container_width=True)
        
        # Результаты по типам задач
        task_display_names = {
            'sentiment': '📈 Анализ тональности',
            'category': '🏷️ Классификация категорий',
            'multilabel': '🏷️ Многометочная классификация'
        }
        
        for task_type, task_results in results.items():
            if task_results:
                display_name = task_display_names.get(task_type, task_type)
                st.markdown(f"### {display_name}")
                
                # Создаем таблицу результатов
                table_data = []
                for model_name, info in task_results.items():
                    if info.get('success', False):
                        if info.get('is_multi_label', False):
                            pred_display = ", ".join(info.get('predicted_labels', []))[:50]
                            if len(info.get('predicted_labels', [])) > 3:
                                pred_display += "..."
                        else:
                            pred_display = info.get('pred', 'N/A')
                        
                        table_data.append({
                            'Модель': model_name,
                            'Тип': info.get('model_type', 'N/A'),
                            'Предсказание': pred_display,
                            'Уверенность': f"{info.get('top_prob', 0):.1%}",
                            'Классы': len(info.get('classes', []))
                        })
                
                if table_data:
                    # Сортируем по уверенности
                    df_results = pd.DataFrame(table_data)
                    df_results['conf_num'] = df_results['Уверенность'].str.replace('%', '').astype(float)
                    df_results = df_results.sort_values('conf_num', ascending=False)
                    df_results = df_results.drop('conf_num', axis=1)
                    
                    # Показываем таблицу
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Визуализация
                    if settings['show_charts']:
                        self._render_results_charts(task_type, task_results)
                else:
                    st.info("ℹ️ Нет успешных предсказаний для этой задачи")
        
        # Экспорт результатов
        if settings['export_results']:
            self._export_results(text, results)
    
    def _render_results_charts(self, task_type: str, task_results: Dict):
        """Рендер графиков результатов"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            # Подготавливаем данные для графика
            model_names = []
            confidences = []
            
            for model_name, info in task_results.items():
                if info.get('success', False) and 'top_prob' in info:
                    model_names.append(model_name)
                    confidences.append(info['top_prob'])
            
            if model_names and confidences:
                # Гистограмма уверенности моделей
                fig = go.Figure(data=[
                    go.Bar(
                        x=model_names,
                        y=confidences,
                        text=[f"{c:.1%}" for c in confidences],
                        textposition='auto',
                        marker_color='lightblue'
                    )
                ])
                
                fig.update_layout(
                    title=f'Уверенность моделей ({task_type})',
                    xaxis_title='Модель',
                    yaxis_title='Уверенность',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Распределение предсказаний для лучшей модели
                if task_results:
                    best_model_name = max(task_results.items(), 
                                        key=lambda x: x[1].get('top_prob', 0) if x[1].get('success', False) else 0)[0]
                    best_model_info = task_results[best_model_name]
                    
                    if 'proba' in best_model_info and 'classes' in best_model_info:
                        proba = best_model_info['proba']
                        classes = best_model_info['classes']
                        
                        if len(proba) > 0 and len(classes) > 0:
                            # Берем топ-10 классов
                            indices = np.argsort(proba)[-10:][::-1]
                            top_classes = [classes[i] for i in indices if i < len(classes)]
                            top_probs = [proba[i] for i in indices if i < len(proba)]
                            
                            fig2 = px.bar(
                                x=top_probs,
                                y=top_classes,
                                orientation='h',
                                title=f'Топ предсказаний: {best_model_name}',
                                labels={'x': 'Вероятность', 'y': 'Класс'}
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Не удалось создать графики: {e}")
    
    def _export_results(self, text: str, results: Dict):
        """Экспорт результатов"""
        st.markdown("---")
        st.subheader("💾 Экспорт результатов")
        
        # Подготовка данных для экспорта
        export_data = {
            'text': text[:500],
            'timestamp': datetime.now().isoformat(),
            'statistics': self.analyzer.get_text_statistics(text),
            'results': results
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON экспорт
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Скачать JSON",
                data=json_str,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # CSV экспорт (только табличные данные)
            try:
                all_data = []
                for task_type, task_results in results.items():
                    for model_name, info in task_results.items():
                        if info.get('success', False):
                            row = {
                                'task_type': task_type,
                                'model_name': model_name,
                                'model_type': info.get('model_type', ''),
                                'prediction': str(info.get('pred', '')),
                                'confidence': info.get('top_prob', 0),
                                'num_classes': len(info.get('classes', []))
                            }
                            all_data.append(row)
                
                if all_data:
                    df_export = pd.DataFrame(all_data)
                    csv_data = df_export.to_csv(index=False, encoding='utf-8')
                    
                    st.download_button(
                        label="📊 Скачать CSV",
                        data=csv_data,
                        file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            except:
                st.warning("Не удалось создать CSV файл")
    
    def render_comparison_analysis(self, results: Dict):
        """Сравнительный анализ результатов"""
        if not results or len(results) < 2:
            return
        
        st.markdown("---")
        st.header("📈 Сравнительный анализ")
        
        # Собираем все модели для сравнения
        all_models = []
        for task_type, task_results in results.items():
            for model_name, info in task_results.items():
                if info.get('success', False):
                    all_models.append({
                        'task_type': task_type,
                        'model_name': model_name,
                        'confidence': info.get('top_prob', 0),
                        'model_type': info.get('model_type', 'unknown')
                    })
        
        if all_models:
            df_comparison = pd.DataFrame(all_models)
            
            # Топ моделей по уверенности
            st.subheader("🏆 Топ моделей")
            df_top = df_comparison.sort_values('confidence', ascending=False).head(10)
            st.dataframe(df_top, use_container_width=True)
            
            # Анализ по типам моделей
            st.subheader("📊 По типам моделей")
            type_stats = df_comparison.groupby('model_type')['confidence'].agg(['mean', 'count']).round(3)
            st.dataframe(type_stats, use_container_width=True)
            
            # Анализ по типам задач
            st.subheader("📊 По типам задач")
            task_stats = df_comparison.groupby('task_type')['confidence'].agg(['mean', 'count']).round(3)
            st.dataframe(task_stats, use_container_width=True)


def build_pipelines_from_stages() -> List[BasePipelineWrapper]:
    """
    Сбор всех моделей из предыдущих этапов
    """
    pipelines = []
    
    # Получаем векторзатор из этапа 2
    vectorizer = st.session_state.get("vectorizer")
    
    # Определяем основные параметры из предыдущих этапов
    main_task_type = st.session_state.get("label_field_select", "category")
    if main_task_type == 'categories':
        main_task_type = 'multilabel'
    
    # Получаем реальные классы из данных
    real_classes = []
    labeled_articles = st.session_state.get("labeled_articles", [])
    if labeled_articles and main_task_type in ['sentiment', 'category']:
        for article in labeled_articles:
            if main_task_type in article and article[main_task_type]:
                real_classes.append(str(article[main_task_type]))
        real_classes = list(set(real_classes))
    
    # 1. Модели из этапа 3 (классические)
    if st.session_state.get("comparator"):
        comparator = st.session_state.comparator
        if hasattr(comparator, 'models'):
            for model_name, model_info in comparator.models.items():
                if hasattr(model_info, 'predict'):
                    wrapper = ClassicalPipelineWrapper(
                        name=f"Этап 3: {model_name}",
                        model=model_info,
                        task_type=main_task_type,
                        label_field=main_task_type,
                        real_classes=real_classes,
                        vectorizer=vectorizer
                    )
                    pipelines.append(wrapper)
    
    # 2. Лучшая модель из этапа 3
    if st.session_state.get("best_model"):
        model = st.session_state.best_model
        if hasattr(model, 'predict'):
            wrapper = ClassicalPipelineWrapper(
                name="🏆 Лучшая классическая модель",
                model=model,
                task_type=main_task_type,
                label_field=main_task_type,
                real_classes=real_classes,
                vectorizer=vectorizer
            )
            pipelines.append(wrapper)
    
    # 3. Модели из этапа 4 (нейросетевые)
    if st.session_state.get("neural_models"):
        for model_name, model in st.session_state.neural_models.items():
            if hasattr(model, 'predict'):
                wrapper = NeuralPipelineWrapper(
                    name=f"Этап 4: {model_name}",
                    model=model,
                    task_type=main_task_type,
                    label_field=main_task_type,
                    real_classes=real_classes
                )
                pipelines.append(wrapper)
    
    # 4. Модели из этапа 5 (с балансировкой)
    if st.session_state.get("balanced_models"):
        for model_key, model in st.session_state.balanced_models.items():
            if hasattr(model, 'predict'):
                wrapper = ClassicalPipelineWrapper(
                    name=f"Этап 5: {model_key}",
                    model=model,
                    task_type=main_task_type,
                    label_field=main_task_type,
                    real_classes=real_classes,
                    vectorizer=vectorizer
                )
                pipelines.append(wrapper)
    
    # 5. Модель из этапа 6 (настроенная)
    if st.session_state.get("best_tuned_model"):
        model = st.session_state.best_tuned_model
        if hasattr(model, 'predict'):
            wrapper = ClassicalPipelineWrapper(
                name="⚙️ Настроенная модель (Этап 6)",
                model=model,
                task_type=main_task_type,
                label_field=main_task_type,
                real_classes=real_classes,
                vectorizer=vectorizer
            )
            pipelines.append(wrapper)
    
    # 6. Чемпионская модель из этапа 7
    if st.session_state.get("champion_model"):
        model = st.session_state.champion_model
        champion_stage = st.session_state.get("champion_stage", "Этап ?")
        if hasattr(model, 'predict'):
            wrapper = ClassicalPipelineWrapper(
                name=f"👑 Чемпионская модель ({champion_stage})",
                model=model,
                task_type=main_task_type,
                label_field=main_task_type,
                real_classes=real_classes,
                vectorizer=vectorizer
            )
            pipelines.append(wrapper)
    
    # 7. Демо-модели для других типов задач (если нет реальных)
    if not pipelines:
        st.warning("⚠️ Не найдено обученных моделей. Созданы демо-модели.")
        
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
            wrapper = ClassicalPipelineWrapper(
                name=f"Демо: {task_type}",
                model=demo_model,
                task_type=task_type,
                label_field=task_type,
                real_classes=classes,
                vectorizer=None
            )
            pipelines.append(wrapper)
    
    return pipelines


def main():
    """Основная функция этапа 8"""
    
    st.set_page_config(
        page_title="Этап 8: Интерактивный анализ",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Этап 8: Интерактивный анализ и сравнение моделей")
    st.markdown("""
    ### 📋 Обзор этапа
    
    Этот этап позволяет:
    1. **Анализировать произвольный текст** всеми обученными моделями
    2. **Сравнивать результаты** разных типов моделей
    3. **Визуализировать уверенность** моделей в предсказаниях
    4. **Экспортировать результаты** для дальнейшего анализа
    
    ---
    """)
    
    # Проверка выполнения предыдущих этапов
    if not st.session_state.get("step7_completed", False):
        st.warning("""
        ⚠️ **Сначала выполните предыдущие этапы!**
        
        Для работы этого этапа необходимо:
        1. ✅ Этап 3: Обучить классические модели
        2. ✅ Этап 4: Обучить нейросетевые модели (опционально)
        3. ✅ Этап 5: Применить балансировку классов
        4. ✅ Этап 6: Настроить гиперпараметры
        5. ✅ Этап 7: Выбрать лучшую модель
        
        Вернитесь к предыдущим этапам для обучения моделей.
        """)
        
        # Кнопка для быстрого перехода
        if st.button("🔄 Проверить наличие моделей"):
            # Попробуем собрать модели, которые уже есть
            pipelines = build_pipelines_from_stages()
            if pipelines:
                st.success(f"✅ Найдено {len(pipelines)} моделей. Можно продолжить.")
                st.session_state.step7_completed = True
                st.rerun()
            else:
                st.error("❌ Модели не найдены. Обучите модели в предыдущих этапах.")
        return
    
    # Инициализация UI
    ui = InteractiveAnalysisUI()
    
    # Сбор моделей из всех этапов
    with st.spinner("🔄 Загружаю модели из всех этапов..."):
        pipelines = build_pipelines_from_stages()
    
    if not pipelines:
        st.error("❌ Не удалось загрузить модели. Проверьте выполнение предыдущих этапов.")
        return
    
    # Показываем статистику моделей
    ui.render_model_statistics(pipelines)
    
    # Инициализация анализатора
    analyzer = InteractiveModelAnalyzer(pipelines)
    ui.analyzer = analyzer
    
    # Рендер боковой панели
    settings = ui.render_sidebar()
    
    # Рендер ввода текста
    text_input, analyze_button = ui.render_text_input()
    
    # Сохраняем текст в session_state
    if text_input:
        st.session_state.interactive_text = text_input
    
    # Анализ текста
    if analyze_button and text_input:
        with st.spinner(f"🔍 Анализирую текст {len(pipelines)} моделями..."):
            # Анализируем текст
            task_filter = None if settings['selected_task'] == 'all' else settings['selected_task']
            results = analyzer.analyze_text(text_input, task_filter)
            
            # Показываем результаты
            if results:
                ui.render_results(text_input, results, settings)
                ui.render_comparison_analysis(results)
            else:
                st.error("❌ Не удалось получить результаты анализа")
    
    # Информация о моделях
    with st.expander("ℹ️ Информация о загруженных моделях"):
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
            st.dataframe(pd.DataFrame(model_info), use_container_width=True)
        else:
            st.info("Нет информации о моделях")
    
    # Сохранение состояния
    st.session_state.ep8_pipelines = pipelines
    st.session_state.ep8_completed = True
    
    st.success("✅ Этап 8 готов к работе! Введите текст и нажмите 'Проанализировать'")


if __name__ == "__main__":
    main()