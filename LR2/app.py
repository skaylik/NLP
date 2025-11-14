# app.py
import streamlit as st
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import shutil
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import time
import tracemalloc
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import altair as alt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

matplotlib.use('Agg')

CPU_TOTAL = os.cpu_count() or 2
DEFAULT_WORKERS = min(max(1, CPU_TOTAL - 1), CPU_TOTAL)

# Импорт ваших модулей
from preparation_corpus import process_corpus, analyze_corpus, TextProcessor
from implementation_vectorization_methods import ClassicalVectorizers
from dimensionality_reduction_topic_modeling import DimensionalityReduction
from comparative_analysis_vectorization import VectorizationComparator
from training_models import DistributedRepresentations
from vector_arithmetic_semantic_operations import SemanticOperations, get_russian_test_sets

# Настройка страницы
st.set_page_config(
    page_title="Лабораторный практикум № 2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация session state
if 'corpus_processed' not in st.session_state:
    st.session_state.corpus_processed = False
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'vectorization_done' not in st.session_state:
    st.session_state.vectorization_done = False
if 'vectorizers' not in st.session_state:
    st.session_state.vectorizers = None
if 'dim_reduction_done' not in st.session_state:
    st.session_state.dim_reduction_done = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'distributed_models' not in st.session_state:
    st.session_state.distributed_models = None
if 'texts' not in st.session_state:
    st.session_state.texts = None
if 'categories' not in st.session_state:
    st.session_state.categories = None

# Заголовок приложения
st.title("Лабораторный практикум № 2")
st.subheader("Сравнительный анализ методов векторизации текста на материале русскоязычных новостных корпусов")
st.markdown("---")

# ============================================================================
# БОКОВАЯ ПАНЕЛЬ: ЗАГРУЗКА ДАННЫХ И СТАТУС
# ============================================================================
with st.sidebar:
    st.header("📁 Загрузка данных")
    uploaded_file = st.file_uploader(
        "Выберите JSONL файл",
        type=['jsonl', 'json'],
        help="Загрузите файл в формате JSONL"
    )

if uploaded_file is not None:
    # Сохраняем загруженный файл
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.jsonl') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        input_file_path = tmp_file.name

    st.session_state.input_file = input_file_path
    st.session_state.input_filename = uploaded_file.name

    # Анализ корпуса
    with st.spinner("Анализ..."):
        analysis = analyze_corpus(input_file_path)


# ============================================================================
# ОСНОВНОЙ КОНТЕНТ: ПОСЛЕДОВАТЕЛЬНЫЕ ЭТАПЫ
# ============================================================================

# Проверка наличия загруженного файла
if 'input_file' not in st.session_state:
    st.info("👆 Пожалуйста, загрузите JSONL файл в боковой панели для начала работы")
    st.stop()

# ============================================================================
# ЭТАП 1: ПОДГОТОВКА КОРПУСА
# ============================================================================
st.header("🔧 1. Подготовка экспериментального корпуса")
st.markdown("""
**Теория:** на этом шаге мы приводим сырые тексты к стандартному виду. Цель — очистить данные от шума, нормализовать слова и гарантировать, что все документы сопоставимы по структуре, прежде чем переходить к числовому представлению.
""")

st.markdown("### Настройки обработки")

# Фиксированный минимальный порог - 100000 слов
target_words = 100000

def render_preprocessing_results(summary):
    if not summary:
        return

    total_words = summary.get('total_words', 0)
    if total_words <= 0:
        return

    target = summary.get('target_words', target_words)
    processed_count = summary.get('processed_count', 0)
    stats = summary.get('stats', {})
    total_docs_in_file = summary.get('total_docs_in_file', 0)
    skipped_count = summary.get('skipped_count', 0)
    validation_failed = summary.get('validation_failed', 0)
    processing_stats = summary.get('processing_stats') or {}
    all_categories = summary.get('all_categories', [])
    total_categories_in_file = summary.get('total_categories_in_file', len(all_categories))

    achievement_ratio = total_words / target if target else 0

    if achievement_ratio >= 1.0:
        st.success(
            f"✅ **Обработка успешно завершена!** Собрано **{total_words:,} слов** - целевой показатель в {target:,} слов достигнут!"
        )
    else:
        st.warning(
            f"⚠️ **Обработка завершена.** Собрано **{total_words:,} слов** из целевых {target:,} ({achievement_ratio*100:.1f}%)"
        )

    st.subheader("📊 Результаты предобработки")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Обработано документов", f"{processed_count:,}")
    with col2:
        st.metric("Всего слов", f"{total_words:,}")
    with col3:
        if total_docs_in_file > 0:
            efficiency = (processed_count / total_docs_in_file) * 100
            st.metric("Эффективность обработки", f"{efficiency:.1f}%")
    with col4:
        cache_efficiency = processing_stats.get('cache_efficiency', 0)
        if cache_efficiency:
            st.metric("Эффективность кэша", f"{cache_efficiency:.1%}")

    st.markdown("#### 🎯 Прогресс сбора слов")
    progress_percent = min(achievement_ratio * 100, 100)
    st.progress(progress_percent / 100 if progress_percent else 0.0)
    st.caption(f"Собрано {total_words:,} из {target:,} слов ({progress_percent:.1f}%)")

    with st.expander("🔍 Детальная статистика обработки"):
        col1_exp, col2_exp = st.columns(2)

        with col1_exp:
            st.write("**📈 Метрики качества:**")
            if total_docs_in_file:
                st.write(f"- Исходных документов: {total_docs_in_file:,}")
            st.write(f"- Успешно обработано: {processed_count:,}")
            st.write(f"- Пропущено записей: {skipped_count}")
            st.write(f"- Не прошло валидацию: {validation_failed}")

            if processing_stats:
                st.write(f"- Документов обрезано: {processing_stats.get('documents_truncated', 0)}")
                st.write(f"- Низкое разнообразие: {processing_stats.get('low_diversity_documents', 0)}")
                st.write(f"- Ошибок лемматизации: {processing_stats.get('lemmatization_errors', 0)}")

        with col2_exp:
            st.write("**⚙️ Характеристики корпуса:**")
            if processed_count > 0:
                words_per_doc = total_words / processed_count
                st.write(f"- Средняя длина документа: {words_per_doc:.1f} слов")
            st.write(f"- Обработано категорий: {len(stats)}")
            st.write("**✅ Примененные методы:**")
            st.write("- Лемматизация (pymorphy3)")
            st.write("- Фильтрация стоп-слов")
            st.write("- Токенизация (NLTK)")
            st.write("- Очистка от шума")

    if stats:
        st.subheader("📋 Распределение слов по категориям")

        if total_categories_in_file:
            st.info(
                f"📋 Всего категорий в исходном файле: {total_categories_in_file}. "
                f"Обработано категорий: {len(stats)}"
            )

        stats_df = pd.DataFrame([
            {'Категория': cat, 'Количество слов': count}
            for cat, count in stats.items()
        ])
        stats_df = stats_df.sort_values('Количество слов', ascending=False)

        col1_stats, col2_stats = st.columns([2, 1])

        with col1_stats:
            st.write("**📊 Распределение слов по категориям**")
            st.bar_chart(stats_df.set_index('Категория'))

        with col2_stats:
            st.write("**📋 Детали по категориям**")
            st.dataframe(stats_df, use_container_width=True)

            top_categories = stats_df.head(3)
            st.write("**🏆 Топ-3 категории:**")
            for _, row in top_categories.iterrows():
                st.write(f"- {row['Категория']}: {row['Количество слов']:,} слов")

    processed_file = st.session_state.get('processed_file')
    if processed_file:
        try:
            with open(processed_file, 'r', encoding='utf-8') as f:
                processed_data = f.read()

            st.download_button(
                label="📥 Скачать обработанный корпус",
                data=processed_data,
                file_name=processed_file,
                mime="application/jsonl",
                help="Скачайте обработанные данные для использования в других инструментах"
            )
        except Exception as exc:
            st.error(f"Ошибка при подготовке файла для скачивания: {exc}")

# ИНФОРМАЦИОННЫЙ БЛОК - объединенный и улучшенный
st.info("""
**🔍 Процесс обработки текста включает:**

- **🧹 Очистка от шума** - Удаление HTML-тегов, замена URL/email/чисел на специальные токены
- **🔤 Нормализация** - Приведение к нижнему регистру, расширение сокращений ("т.е." → "то есть")
- **✂️ Токенизация** - Разбиение текста на слова с сохранением специальных токенов
- **🚫 Фильтрация** - Удаление стоп-слов и коротких слов (<3 символов)
- **📖 Лемматизация** - Приведение слов к нормальной форме с использованием pymorphy3
- **📊 Валидация** - Проверка минимальной длины текста (50 символов, 5 слов после обработки)

**🎯 Цель обработки:** Получить не менее **100,000 слов** качественно обработанного текста для последующего анализа
""")

enable_logging = st.checkbox(
    "Включить подробное логирование",
    value=False,
    help="Записывает детальную информацию о процессе обработки в файл text_processing.log"
)

if st.button("🚀 Начать обработку корпуса", type="primary", key="process_btn"):
    output_file = f"processed_{st.session_state.input_filename}"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(progress):
        progress_bar.progress(min(progress, 1.0))
    
    def status_callback(status):
        status_text.text(status)
    
    with st.spinner("Обработка корпуса..."):
        result = process_corpus(
            st.session_state.input_file,
            output_file,
            target_words=target_words,
            progress_callback=progress_callback,
            status_callback=status_callback,
            enable_logging=enable_logging
        )
    
    # Получаем результаты обработки
    total_words = result.get('total_words', 0)
    stats = result.get('category_stats', {})
    processed_count = result.get('processed_count', 0)
    total_lines = result.get('total_lines', 0)
    skipped_count = result.get('skipped_count', 0)
    validation_failed = result.get('validation_failed', 0)
    processing_stats = result.get('processing_stats', {})
    
    if total_words > 0:
        st.session_state.corpus_processed = True
        st.session_state.processed_file = output_file

        analysis = analyze_corpus(st.session_state.input_file)
        total_docs_in_file = analysis.get('total_documents', 0) if analysis else 0
        all_categories = analysis.get('categories_found', []) if analysis else []

        st.session_state.processing_summary = {
            "total_words": total_words,
            "processed_count": processed_count,
            "stats": stats,
            "total_lines": total_lines,
            "skipped_count": skipped_count,
            "validation_failed": validation_failed,
            "processing_stats": processing_stats,
            "target_words": target_words,
            "total_docs_in_file": total_docs_in_file,
            "all_categories": all_categories,
            "total_categories_in_file": len(all_categories)
        }
    else:
        st.error("❌ Обработка не удалась - не было собрано ни одного слова")
        st.info("""
        **💡 Возможные причины и решения:**
        
        - **Все записи не прошли валидацию** - проверьте наличие полей 'title' и 'text' в исходном файле
        - **Тексты слишком короткие** - минимальные требования: 50 символов и 5 слов после обработки
        - **Проблемы с кодировкой** - убедитесь, что файл в формате UTF-8
        - **Некорректный формат JSONL** - каждая строка должна быть валидным JSON объектом
        
        **Рекомендации:**
        - Проверьте файл text_processing.log для детального анализа ошибок
        - Убедитесь, что в исходных данных достаточно текстового контента
        - Попробуйте уменьшить строгость фильтрации (если доступны настройки)
        """)
        st.session_state.corpus_processed = False
        st.session_state.pop('processed_file', None)
        st.session_state.pop('processing_summary', None)

if st.session_state.get('processing_summary'):
    render_preprocessing_results(st.session_state.processing_summary)

st.markdown("---")

# ============================================================================
# ЭТАП 2: ВЕКТОРИЗАЦИЯ
# ============================================================================
st.header("🔢 2. Классические методы векторизации")
st.markdown("""
**Теория:** векторизация переводит текст в числовую матрицу «документ × признак». Мы используем классические модели (Bag of Words, TF‑IDF и др.), чтобы сохранить информацию о словах и их важности, подготовив данные для последующего анализа.
""")

def render_vectorization_results(summary):
    if not summary:
        return

    results = summary.get('results') or {}
    if not results:
        return

    text_count = summary.get('text_count', 0)
    methods_selected = summary.get('methods', [])
    max_features_val = summary.get('max_features')
    ngram_max_val = summary.get('ngram_max')

    if text_count:
        st.info(f"Загружено {text_count} документов")

    st.success("✅ Векторизация завершена!")

    caption_parts = []
    if methods_selected:
        caption_parts.append("методы: " + ", ".join(methods_selected))
    param_parts = []
    if max_features_val is not None:
        param_parts.append(f"максимум признаков = {max_features_val}")
    if ngram_max_val is not None:
        param_parts.append(f"максимальная n-грамма = {ngram_max_val}")
    if param_parts:
        caption_parts.append("параметры: " + ", ".join(param_parts))
    if caption_parts:
        st.caption("; ".join(caption_parts))

    st.subheader("📊 Результаты векторизации")

    results_df = pd.DataFrame(results).T
    st.markdown("#### 📋 Сравнительная таблица методов")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("📈 Анализ разреженности и плотности")

    sparsity_data = {}
    density_data = {}

    for method, stats in results.items():
        try:
            sparsity_str = stats['Разреженность (%)'].replace('%', '').replace(',', '')
            density_str = stats['Плотность (%)'].replace('%', '').replace(',', '')
            sparsity_data[method] = float(sparsity_str)
            density_data[method] = float(density_str)
        except Exception as exc:
            st.warning(f"Ошибка при обработке данных для метода {method}: {exc}")

    if sparsity_data and density_data:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 📊 Разреженность матриц")
            sparsity_df = pd.DataFrame({
                'Метод': list(sparsity_data.keys()),
                'Разреженность (%)': list(sparsity_data.values())
            }).sort_values('Разреженность (%)', ascending=False)
            st.bar_chart(sparsity_df.set_index('Метод'))

            with st.expander("🔍 Анализ разреженности"):
                min_sparsity_method = min(sparsity_data, key=sparsity_data.get)
                max_sparsity_method = max(sparsity_data, key=sparsity_data.get)
                st.write(f"**Наименьшая разреженность:** {min_sparsity_method} ({sparsity_data[min_sparsity_method]:.2f}%)")
                st.write(f"**Наибольшая разреженность:** {max_sparsity_method} ({sparsity_data[max_sparsity_method]:.2f}%)")
                st.write("""
                **Что означает разреженность:**
                - Высокая разреженность (>95%): матрица очень разрежена, много нулей
                - Средняя разреженность (80-95%): умеренная разреженность  
                - Низкая разреженность (<80%): матрица достаточно плотная
                """)

        with col2:
            st.markdown("##### 🎯 Плотность матриц")
            density_df = pd.DataFrame({
                'Метод': list(density_data.keys()),
                'Плотность (%)': list(density_data.values())
            }).sort_values('Плотность (%)', ascending=False)
            st.bar_chart(density_df.set_index('Метод'))

            with st.expander("🔍 Анализ плотности"):
                min_density_method = min(density_data, key=density_data.get)
                max_density_method = max(density_data, key=density_data.get)
                st.write(f"**Наименьшая плотность:** {min_density_method} ({density_data[min_density_method]:.2f}%)")
                st.write(f"**Наибольшая плотность:** {max_density_method} ({density_data[max_density_method]:.2f}%)")
                st.write("""
                **Что означает плотность:**
                - Высокая плотность (>20%): матрица содержит много информации
                - Средняя плотность (5-20%): умеренная информативность
                - Низкая плотность (<5%): матрица очень разрежена
                """)

        st.markdown("---")
        st.subheader("📊 Сравнительный анализ")

        comparison_df = pd.DataFrame([
            {
                'Метод': method,
                'Разреженность (%)': sparsity_data[method],
                'Плотность (%)': density_data[method],
                'Эффективность': 100 - sparsity_data[method]
            }
            for method in sparsity_data.keys()
        ])

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("##### 🏆 Лучшие методы")
            best_sparsity = comparison_df.loc[comparison_df['Разреженность (%)'].idxmin()]
            best_density = comparison_df.loc[comparison_df['Плотность (%)'].idxmax()]
            best_efficiency = comparison_df.loc[comparison_df['Эффективность'].idxmax()]

            st.metric("Лучшая разреженность", f"{best_sparsity['Метод']}", f"{best_sparsity['Разреженность (%)']:.1f}%")
            st.metric("Лучшая плотность", f"{best_density['Метод']}", f"{best_density['Плотность (%)']:.1f}%")
            st.metric("Общая эффективность", f"{best_efficiency['Метод']}", f"{best_efficiency['Эффективность']:.1f}%")

        with col4:
            st.markdown("##### 📋 Рекомендации")
            if best_sparsity['Разреженность (%)'] < 90:
                st.success("**✅ Хорошая ситуация:** Большинство методов имеют приемлемую разреженность")
            else:
                st.warning("**⚠️ Внимание:** Высокая разреженность может снизить эффективность моделей")

            if len(methods_selected) >= 2:
                st.info(f"**💡 Совет:** Для большинства задач рекомендуем **{best_efficiency['Метод']}**")

            st.write("""
            **Критерии выбора:**
            - Низкая разреженность = лучше для производительности
            - Высокая плотность = больше информации сохраняется
            - Баланс = оптимальная эффективность
            """)

        st.markdown("---")
        st.subheader("🔍 Детали по методам")

        for method, stats in results.items():
            with st.expander(f"{method}", expanded=False):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.write("**Основные характеристики:**")
                    st.write(f"- Размерность: {stats['Размерность']}")
                    st.write(f"- Всего элементов: {stats['Всего элементов']}")
                    st.write(f"- Ненулевые элементы: {stats['Ненулевые элементы']}")

                with col_b:
                    st.write("**Качественные показатели:**")
                    st.write(f"- Разреженность: {stats['Разреженность (%)']}")
                    st.write(f"- Плотность: {stats['Плотность (%)']}")

                    sparsity_val = sparsity_data.get(method, 0)
                    if sparsity_val < 80:
                        st.success("✅ Отличная плотность")
                    elif sparsity_val < 95:
                        st.info("ℹ️ Умеренная разреженность")
                    else:
                        st.warning("⚠️ Высокая разреженность")

    else:
        st.warning("Не удалось подготовить данные для визуализации")

if not st.session_state.corpus_processed:
    st.warning("⚠️ Сначала обработайте корпус на этапе 1")
else:
    st.markdown("### Настройки векторизации")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_features = st.number_input(
            "Максимум признаков",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            key="max_features_2"
        )
    
    with col2:
        ngram_max = st.selectbox(
            "Максимальная n-грамма",
            options=[1, 2, 3],
            index=1,
            key="ngram_max_2"
        )
    
    st.markdown("### Методы векторизации")
    
    methods = st.multiselect(
        "Выберите методы",
        options=[
            "One-Hot Encoding",
            "Bag of Words", 
            "TF-IDF",
            "Комбинированные n-граммы"
        ],
        default=["Bag of Words", "TF-IDF"],
        key="methods_2"
    )
    
    if st.button("🔄 Применить векторизацию", type="primary", key="vectorize_btn"):
        vectorizers = ClassicalVectorizers()
        
        with st.spinner("Загрузка корпуса..."):
            texts, categories = vectorizers.load_corpus(
                st.session_state.processed_file,
                text_field='text',
                category_field='category'
            )
        
        text_count = len(texts)
        
        st.session_state.texts = texts
        st.session_state.categories = categories
        st.session_state.vectorizers = vectorizers
        
        results = {}
        matrices = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_methods = len(methods) or 1
        
        for idx, method in enumerate(methods):
            status_text.text(f"Применение метода: {method}...")
            
            if method == "One-Hot Encoding":
                X = vectorizers.one_hot_encoding(
                    texts,
                    ngram_range=(1, ngram_max),
                    max_features=max_features
                )
                if X is not None:
                    results[method] = vectorizers.analyze_sparsity(X, method)
                    matrices[method] = X
            
            elif method == "Bag of Words":
                X = vectorizers.bag_of_words(
                    texts,
                    ngram_range=(1, ngram_max),
                    max_features=max_features,
                    binary=False
                )
                if X is not None:
                    results[method] = vectorizers.analyze_sparsity(X, method)
                    matrices[method] = X
            
            elif method == "TF-IDF":
                X = vectorizers.tfidf(
                    texts,
                    ngram_range=(1, ngram_max),
                    max_features=max_features
                )
                if X is not None:
                    results[method] = vectorizers.analyze_sparsity(X, method)
                    matrices[method] = X
                    st.session_state.last_tfidf_matrix = X
            
            elif method == "Комбинированные n-граммы":
                X = vectorizers.combined_ngrams(
                    texts,
                    max_ngram=ngram_max,
                    max_features=max_features
                )
                if X is not None:
                    results[method] = vectorizers.analyze_sparsity(X, method)
                    matrices[method] = X
            
            if total_methods:
                progress_bar.progress((idx + 1) / total_methods)
        
        status_text.text("Векторизация завершена!")
        
        if results:
            st.session_state.vectorization_done = True
            st.session_state.vectorization_matrices = matrices
            st.session_state.vectorization_summary = {
                "results": results,
                "text_count": text_count,
                "methods": methods,
                "max_features": max_features,
                "ngram_max": ngram_max
            }
        else:
            st.warning("Не удалось получить результаты векторизации")
            st.session_state.vectorization_done = False
            st.session_state.pop('vectorization_summary', None)

    if st.session_state.get('vectorization_summary'):
        render_vectorization_results(st.session_state.vectorization_summary)

st.markdown("---")

# ============================================================================
# ЭТАП 3: СНИЖЕНИЕ РАЗМЕРНОСТИ
# ============================================================================
st.header("📉 3. Снижение размерности и тематическое моделирование")
st.markdown("""
**Теория:** после векторизации пространство признаков очень велико. SVD и t‑SNE позволяют сжать его, выделить главные тематики и увидеть структуру корпуса, сохранив основную информацию при меньшем числе измерений.
""")

def render_dim_reduction_results(summary):
    if not summary:
        return

    st.success("✅ Снижение размерности завершено!")

    svd_shape = summary.get('svd_shape')
    if svd_shape:
        st.write(f"**SVD представление:** {svd_shape[0]} документов × {svd_shape[1]} компонент")

    component_stats = summary.get('component_stats') or {}
    if component_stats:
        col1, col2, col3 = st.columns(3)
        col1.metric("Исходная размерность", "×".join(map(str, component_stats.get('original_dimensions', []))))
        col2.metric("Сжатая размерность", "×".join(map(str, component_stats.get('reduced_dimensions', []))))
        col3.metric("Коэффициент сжатия", f"{component_stats.get('compression_ratio', 0):.2f}×")

    variance_info = summary.get('variance_info') or {}
    if variance_info:
        st.subheader("📈 Анализ кумулятивной дисперсии")
        col1, col2 = st.columns(2)
        col1.metric(
            "Оптимальное число компонент",
            variance_info.get('optimal_components', 0)
        )
        col2.metric(
            "Достигнутая доля дисперсии",
            f"{variance_info.get('achieved_variance', 0)*100:.1f}%"
        )

        variance_df = pd.DataFrame({
            'Компонента': variance_info.get('components_range', []),
            'Кумулятивная дисперсия': variance_info.get('cumulative_variance', [])
        })
        if not variance_df.empty:
            st.line_chart(variance_df.set_index('Компонента'))
            st.caption(
                f"Порог по дисперсии: {variance_info.get('variance_threshold', 0)*100:.0f}%"
            )

    component_keywords = summary.get('component_keywords') or []
    if component_keywords:
        st.subheader("🔎 Ключевые слова компонент")
        for component in component_keywords:
            idx = component.get('component')
            variance = component.get('explained_variance', 0)
            keywords = component.get('keywords', [])
            label = f"Компонента {idx + 1} • вклад {variance*100:.1f}%"
            with st.expander(label, expanded=False):
                if keywords:
                    keywords_df = pd.DataFrame(keywords, columns=['Слово', 'Вес'])
                    st.table(keywords_df)
                else:
                    st.info("Нет доступных ключевых слов для компоненты")

    tsne_points = summary.get('tsne_points')
    if tsne_points:
        st.subheader("🗺️ Визуализация t-SNE")
        df_vis = pd.DataFrame(tsne_points, columns=['x', 'y'])
        labels = summary.get('labels') or []
        if labels and len(labels) >= len(df_vis):
            df_vis['category'] = labels[:len(df_vis)]
            st.scatter_chart(df_vis, x='x', y='y', color='category', width='stretch')
        else:
            st.scatter_chart(df_vis, x='x', y='y', width='stretch')

if not st.session_state.vectorization_done:
    st.warning("⚠️ Сначала выполните векторизацию на этапе 2")
else:
    st.markdown("### Настройки снижения размерности")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_components = st.slider(
            "Число компонент SVD",
            min_value=10,
            max_value=200,
            value=50,
            step=5,
            key="n_components_3"
        )
    
    with col2:
        variance_threshold = st.slider(
            "Целевая доля дисперсии",
            min_value=0.70,
            max_value=0.99,
            value=0.90,
            step=0.01,
            key="variance_threshold_3"
        )
    
    tsne_perplexity = st.slider(
        "Perplexity для t-SNE",
        min_value=5,
        max_value=50,
        value=30,
        step=5,
        key="tsne_perplexity_3"
    )
    
    if st.button("🔄 Применить снижение размерности", type="primary", key="dim_red_btn"):
        dim_reduction = DimensionalityReduction()
        
        if st.session_state.vectorizers and st.session_state.texts:
            with st.spinner("Подготовка данных..."):
                X = st.session_state.vectorizers.tfidf(
                    st.session_state.texts,
                    ngram_range=(1, 2),
                    max_features=10000
                )
                dim_reduction.load_vectors(X)
            
            with st.spinner("Применяем SVD..."):
                svd_matrix = dim_reduction.apply_svd(n_components=n_components)
            
            component_stats = dim_reduction.get_component_statistics(svd_matrix)
            component_keywords = dim_reduction.interpret_svd_components(
                n_top_words=10,
                n_components=min(5, n_components)
            )
            
            max_components = min(max(n_components, 50), min(svd_matrix.shape[0], svd_matrix.shape[1]) - 1)
            variance_info = dim_reduction.find_optimal_components(
                max_components=max_components,
                variance_threshold=variance_threshold
            )
            
            with st.spinner("Выполняем t-SNE для визуализации..."):
                labels = st.session_state.categories or []
                tsne_points = dim_reduction.visualize_components(
                    svd_matrix,
                    labels=np.array(labels) if labels else None,
                    method='tsne',
                    perplexity=tsne_perplexity
                )
            
            st.session_state.dim_reduction_done = True
            st.session_state.dim_reduction_summary = {
                "svd_shape": tuple(svd_matrix.shape),
                "component_stats": {
                    "original_dimensions": component_stats.get('original_dimensions') if component_stats else None,
                    "reduced_dimensions": component_stats.get('reduced_dimensions') if component_stats else None,
                    "compression_ratio": component_stats.get('compression_ratio') if component_stats else None,
                },
                "component_keywords": [
                    {
                        "component": item.get('component', idx),
                        "explained_variance": float(item.get('explained_variance', 0)),
                        "keywords": [(word, float(weight)) for word, weight in item.get('keywords', [])]
                    }
                    for idx, item in enumerate(component_keywords or [])
                ],
                "variance_info": {
                    "optimal_components": variance_info.get('optimal_components'),
                    "achieved_variance": variance_info.get('achieved_variance'),
                    "variance_threshold": variance_info.get('variance_threshold'),
                    "components_range": list(variance_info.get('components_range', [])),
                    "cumulative_variance": list(variance_info.get('cumulative_variance', []))
                } if variance_info else {},
                "tsne_points": tsne_points.tolist() if tsne_points is not None else None,
                "labels": labels[:len(tsne_points)] if tsne_points is not None and labels else []
            }
        else:
            st.error("Не найдены данные для снижения размерности")
    
    if st.session_state.get('dim_reduction_summary'):
        render_dim_reduction_results(st.session_state.dim_reduction_summary)

st.markdown("---")

# ============================================================================
# ЭТАП 4: СРАВНИТЕЛЬНЫЙ АНАЛИЗ
# ============================================================================
st.header("📊 4. Сравнительный анализ методов векторизации")
st.markdown("""
**Теория:** разные способы векторизации дают разные признаки. Сравнивая их по плотности, размерности и семантической согласованности, мы выбираем наиболее подходящую модель признаков для дальнейших шагов.
""")

def render_comparative_analysis_results(summary):
    if not summary:
        return

    results_comp = summary.get('results') or {}
    if not results_comp:
        return

    comp_df = pd.DataFrame(results_comp).T

    if comp_df.empty:
        st.warning("Не удалось получить результаты сравнения")
        return

    st.success("✅ Сравнительный анализ завершен!")
    st.subheader("📈 Результаты сравнения")
    st.dataframe(comp_df, width='stretch')

    efficiency_cols = {}
    if 'Processing Time (s)' in comp_df.columns:
        efficiency_cols['time'] = pd.to_numeric(comp_df['Processing Time (s)'], errors='coerce')
    if 'Peak Memory (MB)' in comp_df.columns:
        efficiency_cols['memory'] = pd.to_numeric(comp_df['Peak Memory (MB)'], errors='coerce')

    if efficiency_cols:
        st.markdown("##### ⚙️ Вычислительная эффективность")
        col_time, col_mem = st.columns(2)

        if 'time' in efficiency_cols and not efficiency_cols['time'].isna().all():
            fastest_method = efficiency_cols['time'].idxmin()
            fastest_value = efficiency_cols['time'].min()
            col_time.metric(
                "Минимальное время обработки",
                f"{fastest_value:.2f} с",
                fastest_method
            )
        else:
            col_time.info("Нет данных по времени обработки")

        if 'memory' in efficiency_cols and not efficiency_cols['memory'].isna().all():
            best_memory_method = efficiency_cols['memory'].idxmin()
            best_memory_value = efficiency_cols['memory'].min()
            col_mem.metric(
                "Минимальное потребление памяти",
                f"{best_memory_value:.2f} МБ",
                best_memory_method
            )
        else:
            col_mem.info("Нет данных по памяти")

    col1, col2 = st.columns(2)

    with col1:
        if 'Semantic Coherence' in comp_df.columns:
            st.write("**Семантическая согласованность**")
            coherence_data = comp_df[['Semantic Coherence']].copy()
            st.bar_chart(coherence_data)

    with col2:
        if 'Sparsity (%)' in comp_df.columns:
            st.write("**Разреженность (%)**")
            sparsity_data = comp_df[['Sparsity (%)']].copy()
            st.bar_chart(sparsity_data)

    st.markdown("---")
    st.subheader("🔍 Детальный анализ")

    col3, col4 = st.columns(2)

    with col3:
        if 'Dimensions' in comp_df.columns:
            st.write("**Размерность векторов**")
            dim_data = comp_df[['Dimensions']].copy()
            st.bar_chart(dim_data)

    with col4:
        if 'Semantic Coherence' in comp_df.columns:
            best_method = comp_df['Semantic Coherence'].idxmax()
            best_score = comp_df['Semantic Coherence'].max()
            st.metric("Лучший метод по семантике", best_method, f"{best_score:.4f}")

    st.markdown("---")
    st.subheader("💡 Рекомендации")

    if 'Semantic Coherence' in comp_df.columns and 'Sparsity (%)' in comp_df.columns:
        best_semantic = comp_df['Semantic Coherence'].idxmax()
        best_sparsity = comp_df['Sparsity (%)'].idxmin()

        col5, col6 = st.columns(2)

        with col5:
            st.info(f"**Лучшая семантика:** {best_semantic}")
            st.write("Подходит для задач классификации и поиска")

        with col6:
            st.success(f"**Лучшая плотность:** {best_sparsity}")
            st.write("Эффективнее по памяти и вычислениям")

    st.markdown("""
    **Когда использовать Bag of Words:**
    - Простые задачи классификации
    - Когда важна интерпретируемость
    - Ограниченные вычислительные ресурсы

    **Когда использовать TF-IDF:**
    - Поисковые системы
    - Задачи с важностью редких слов
    - Когда нужна лучшая семантическая согласованность
    """)
 
if not st.session_state.vectorization_done:
    st.warning("⚠️ Сначала выполните векторизацию на этапе 2")
else:
    st.markdown("### Сравнение эффективности методов")

    if st.button("🔄 Провести сравнительный анализ", type="primary", key="compare_btn"):
        comparator = VectorizationComparator(st.session_state.vectorizers)
        comparator.texts = st.session_state.get('texts', [])
        comparator.categories = st.session_state.get('categories', [])
 
        with st.spinner("Проведение сравнительного анализа..."):
            results_comp = {}
 
            stored_matrices = st.session_state.get('vectorization_matrices', {}) or {}
            summary_methods = st.session_state.get('vectorization_summary', {}).get('methods', [])
            methods_in_order = [m for m in summary_methods if m in stored_matrices]
            for method_name in stored_matrices:
                if method_name not in methods_in_order:
                    methods_in_order.append(method_name)
 
            if not methods_in_order:
                st.warning("Нет сохранённых матриц для сравнения. Выполните векторизацию на этапе 2.")
            else:
                for method_name in methods_in_order:
                    matrix = stored_matrices.get(method_name)
                    if matrix is None:
                        continue
 
                    tracemalloc.start()
                    start_time = time.perf_counter()
 
                    evaluation = comparator.evaluate_method(matrix, comparator.categories, method_name)
 
                    elapsed = time.perf_counter() - start_time
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
 
                    if evaluation is None:
                        continue
 
                    evaluation['Processing Time (s)'] = round(elapsed, 2)
                    evaluation['Peak Memory (MB)'] = round(peak / (1024 * 1024), 2)
 
                    results_comp[method_name] = evaluation
 
        if results_comp:
            st.session_state.comparative_summary = {
                "results": results_comp
            }
        else:
            st.warning("Не удалось получить результаты сравнения")
            st.session_state.pop('comparative_summary', None)

    if st.session_state.get('comparative_summary'):
        render_comparative_analysis_results(st.session_state.comparative_summary)
 
# ============================================================================
# ЭТАП 5: ОБУЧЕНИЕ МОДЕЛЕЙ РАСПРЕДЕЛЁННЫХ ПРЕДСТАВЛЕНИЙ
# ============================================================================
st.header("🤖 5. Обучение моделей распределённых представлений")
st.markdown("""
**Теория:** распределённые модели (Word2Vec, FastText, Doc2Vec) учатся представлять слова и документы в виде плотных векторов, отражающих смысловые связи. Этот этап строит «семантическое пространство» корпуса.
""")

def render_training_results(summary):
    if not summary:
        return

    models_created = summary.get('models_created', 0)
    text_count = summary.get('text_count', 0)
    params = summary.get('params', {})
    evaluation = summary.get('evaluation', [])

    st.success(f"🎉 Обучение завершено! Успешно обучено {models_created} моделей")
    if text_count:
        st.caption(f"Использовано документов: {text_count}")

    if params:
        with st.expander("🔧 Параметры обучения", expanded=False):
            st.write(f"Модели: {', '.join(params.get('model_types', []))}")
            st.write(f"Размерность векторов: {params.get('vector_size')}")
            st.write(f"Окно контекста: {params.get('window')}")
            st.write(f"Минимальная частота: {params.get('min_count')}")
            st.write(f"Эпохи: {params.get('epochs')}")
            st.write(f"Максимум эпох после адаптации: {params.get('max_epochs')}")
            st.write(f"Негативных примеров: {params.get('negative')}")
            st.write(f"Потоки: {params.get('workers')}")
            st.write(f"Подсчёт loss: {'да' if params.get('compute_loss') else 'нет'}")

    if evaluation:
        st.subheader("🧪 Мгновенная проверка качества")
        for model_eval in evaluation:
            st.write(f"**{model_eval.get('model')}**:")

            for pair_info in model_eval.get('pairs', []):
                word1 = pair_info.get('word1')
                word2 = pair_info.get('word2')
                if 'missing_words' in pair_info:
                    missing = ", ".join(pair_info['missing_words'])
                    st.write(f"  {word1} - {word2}: отсутствуют {missing}")
                else:
                    similarity = pair_info.get('similarity', 0)
                    status = pair_info.get('status', '')
                    st.write(f"  {word1} - {word2}: {similarity:.3f} ({status})")

            neighbors = model_eval.get('neighbors', [])
            if neighbors:
                st.write("  Ближайшие к 'компьютер':")
                for neighbor in neighbors:
                    st.write(f"    - {neighbor['word']}: {neighbor['similarity']:.3f}")
            elif model_eval.get('neighbors_error'):
                st.write(f"  Ошибка при поиске соседей: {model_eval['neighbors_error']}")

            st.write("---")

    st.info("✅ Готово! Переходите к этапу 6 для детальных экспериментов")
 
if not st.session_state.corpus_processed:
    st.warning("⚠️ Сначала обработайте корпус на этапе 1")
else:
    # ИНФОРМАЦИЯ О КОРПУСЕ
    st.info("📊 **Характеристики вашего корпуса:**")
    if st.session_state.get('processing_summary'):
        proc_summary = st.session_state.processing_summary
        st.write(f"- **Документов:** {proc_summary.get('processed_count', '—')}")
        st.write(f"- **Всего слов:** {proc_summary.get('total_words', '—'):,}")
        total_docs_input = proc_summary.get('total_docs_in_file')
        if total_docs_input:
            st.write(f"- **Исходных документов:** {total_docs_input}")
        stats = proc_summary.get('stats') or {}
        st.write(f"- **Обработанных категорий:** {len(stats)}")
        all_categories = proc_summary.get('all_categories') or []
        if all_categories:
            st.write(f"- **Категорий всего:** {len(all_categories)}")
    else:
        st.write("- Загрузите и обработайте корпус, чтобы увидеть статистику.")
    
    # РАСШИРЕННЫЕ НАСТРОЙКИ
    st.markdown("### 🎛️ Настройка параметров обучения")
    
    cpu_total = CPU_TOTAL
    default_workers = DEFAULT_WORKERS

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 Архитектуры моделей")
        model_types = st.multiselect(
            "Выберите модели для обучения:",
            options=[
                "word2vec_skipgram", 
                "word2vec_cbow",
                "fasttext_skipgram", 
                "fasttext_cbow",
                "doc2vec"
            ],
            default=["word2vec_skipgram", "fasttext_skipgram"],
            help="Рекомендуется выбрать обе модели для сравнения"
        )
        
        vector_size = st.slider(
            "Размерность векторов:",
            min_value=50,
            max_value=300,
            value=100,
            step=25,
            help="Большая размерность = лучше качество, но требует больше данных. Для маленького корпуса 50-150 оптимально."
        )
        
        window = st.slider(
            "Размер окна контекста:",
            min_value=2,
            max_value=20,
            value=8,
            help="Большее окно = больше контекста, но менее точные связи. Для маленького корпуса 5-10 оптимально."
        )
    
    with col2:
        st.subheader("⚙️ Параметры обучения")
        
        min_count = st.slider(
            "Минимальная частота слова:",
            min_value=1,
            max_value=10,
            value=2,
            help="Слова, встречающиеся реже, будут проигнорированы. Для маленького корпуса 2-5 оптимально."
        )
        
        epochs = st.slider(
            "Количество эпох:",
            min_value=10,
            max_value=200,
            value=100,
            help="Больше эпох = лучше обучение, но дольше время. Для маленького корпуса 80-120 оптимально."
        )
        
        negative = st.slider(
            "Количество негативных примеров:",
            min_value=0,
            max_value=20,
            value=10,
            help="0 = Hierarchical Softmax (лучше для очень маленьких корпусов), >0 = Negative Sampling (обычно быстрее и лучше)"
        )

        workers_count = st.slider(
            "Количество потоков:",
            min_value=1,
            max_value=int(cpu_total),
            value=int(default_workers),
            help="Чем больше потоков, тем быстрее обучение (до количества доступных ядер)"
        )

    compute_loss_enabled = st.checkbox(
        "Подсчитывать функцию потерь во время обучения",
        value=False,
        help="Включите только при необходимости анализировать loss — это замедляет обучение на 30-40%"
    )

    max_epochs_cap = st.slider(
        "Максимум эпох после автоматической адаптации",
        min_value=int(max(20, epochs)),
        max_value=300,
        value=int(max(150, epochs)),
        step=10,
        help="Ограничивает автоматическое увеличение эпох для маленького корпуса"
    )
    
    # РЕКОМЕНДАЦИИ И ПРЕДУПРЕЖДЕНИЯ
    st.markdown("### 💡 Рекомендации по параметрам")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        if vector_size > 150:
            st.warning("⚠️ Большая размерность может привести к переобучению")
        else:
            st.success("✅ Размерность подходит для маленького корпуса")
            
        if window > 12:
            st.warning("⚠️ Слишком большое окно для маленького корпуса")
        else:
            st.success("✅ Размер окна оптимален")
    
    with rec_col2:
        if min_count == 1:
            st.warning("⚠️ Много шумных слов в словаре")
        else:
            st.success("✅ Хорошая фильтрация шума")
            
        if epochs < 50:
            st.warning("⚠️ Может быть недостаточно эпох")
        else:
            st.success("✅ Достаточно эпох для обучения")
        if max_epochs_cap > 200:
            st.warning("⚠️ Высокий лимит эпох может сильно увеличить время обучения")
        else:
            st.success("✅ Лимит эпох контролирует время обучения")
    
    with rec_col3:
        if negative == 0:
            st.info("ℹ️ Hierarchical Softmax - стабильно для маленьких данных")
        else:
            st.info("ℹ️ Negative Sampling - быстро и эффективно")
        if len(model_types) == 0:
            st.error("❌ Выберите хотя бы одну модель")
        else:
            st.success(f"✅ Выбрано {len(model_types)} моделей")
        if compute_loss_enabled:
            st.warning("⚠️ Подсчёт функции потерь замедлит обучение")
        if workers_count < default_workers:
            st.info(f"ℹ️ Можно увеличить количество потоков (сейчас {workers_count} из {cpu_total})")

    # КНОПКА ОБУЧЕНИЯ
    st.markdown("---")
    
    if st.button("🚀 Начать обучение с выбранными параметрами", type="primary", key="train_btn"):
        if not model_types:
            st.error("❌ Выберите хотя бы одну модель для обучения")
        else:
            distributed = DistributedRepresentations()
            
            # Загрузка текстов
            with st.spinner("Загрузка корпуса..."):
                texts = []
                categories = []
                
                with open(st.session_state.processed_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        text = data.get('text', '')
                        category = data.get('category', '')
                        
                        if text:
                            words = text.split()
                            texts.append(words)
                            categories.append(category)
            
            text_count = len(texts)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🎯 Начинаем обучение с выбранными параметрами...")
            progress_bar.progress(20)
            
            try:
                models_created = distributed.train_with_parameters(
                    texts=texts,
                    categories=categories if 'doc2vec' in model_types else None,
                    model_types=model_types,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    epochs=epochs,
                    negative=negative,
                    workers=workers_count,
                    compute_loss=compute_loss_enabled,
                    max_epochs=max_epochs_cap
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Обучение завершено!")
                
                if models_created > 0:
                    st.session_state.models_trained = True
                    st.session_state.distributed_models = distributed
                    
                    all_models = distributed.get_available_models()
                    test_pairs = [
                        ('компьютер', 'ноутбук'),
                        ('данные', 'информация'), 
                        ('программа', 'алгоритм'),
                        ('город', 'река')
                    ]
                    
                    evaluation_results = []
                    for model_name, model in all_models.items():
                        pair_results = []
                        for word1, word2 in test_pairs:
                            if word1 in model.wv and word2 in model.wv:
                                similarity = float(model.wv.similarity(word1, word2))
                                status_label = "✅ ХОРОШО" if similarity > 0.6 else "⚠️ СЛАБО" if similarity > 0.3 else "❌ ПЛОХО"
                                pair_results.append({
                                    "word1": word1,
                                    "word2": word2,
                                    "similarity": similarity,
                                    "status": status_label
                                })
                            else:
                                missing_words = [word for word in (word1, word2) if word not in model.wv]
                                pair_results.append({
                                    "word1": word1,
                                    "word2": word2,
                                    "missing_words": missing_words
                                })
                        
                        neighbors = []
                        neighbors_error = None
                        if 'компьютер' in model.wv:
                            try:
                                neighbors_raw = model.wv.most_similar('компьютер', topn=3)
                                neighbors = [
                                    {"word": neighbor_word, "similarity": float(sim)}
                                    for neighbor_word, sim in neighbors_raw
                                ]
                            except Exception as err:
                                neighbors_error = str(err)
                        else:
                            neighbors_error = "слово 'компьютер' отсутствует в словаре"
                        
                        evaluation_results.append({
                            "model": model_name,
                            "pairs": pair_results,
                            "neighbors": neighbors,
                            "neighbors_error": neighbors_error
                        })
                    
                    st.session_state.training_summary = {
                        "models_created": models_created,
                        "text_count": text_count,
                        "params": {
                            "model_types": model_types,
                            "vector_size": vector_size,
                            "window": window,
                            "min_count": min_count,
                            "epochs": epochs,
                        "negative": negative,
                        "workers": workers_count,
                        "compute_loss": compute_loss_enabled,
                        "max_epochs": max_epochs_cap
                        },
                        "evaluation": evaluation_results
                    }
                else:
                    st.error("❌ Не удалось обучить ни одну модель")
                    st.session_state.pop('training_summary', None)
            
            except Exception as e:
                st.error(f"❌ Ошибка при обучении: {str(e)}")
                st.info("💡 Попробуйте изменить параметры (уменьшить размерность или увеличить min_count)")
                st.session_state.pop('training_summary', None)

    if st.session_state.get('training_summary'):
        render_training_results(st.session_state.training_summary)

st.markdown("---")

# =========================================================================
# ЭТАП 6: ЭКСПЕРИМЕНТЫ С ВЕКТОРНЫМИ ПРОСТРАНСТВАМИ
# =========================================================================
st.header("🧮 6. Семантические эксперименты с векторными пространствами")
st.markdown("""
**Теория:** финальный этап проверяет качество полученных векторов. Мы анализируем расстояния, аналогии и ближайших соседей, чтобы оценить, насколько хорошо модели отражают смысловые отношения между словами и документами.
""")

if not st.session_state.get('models_trained') or not st.session_state.get('distributed_models'):
    st.warning("⚠️ Сначала обучите хотя бы одну модель на этапе 5, чтобы провести семантические эксперименты.")
else:
    available_models = {}
    try:
        available_models = st.session_state.distributed_models.get_available_models() or {}
    except Exception as exc:
        st.error(f"Не удалось получить список обученных моделей: {exc}")

    if not available_models:
        st.info("ℹ️ Обученные модели не найдены. Пожалуйста, выполните обучение на этапе 5.")
    else:
        semantic_ops = SemanticOperations(available_models)
        model_names = list(available_models.keys())

        if not st.session_state.get('semantic_styles_applied'):
            st.markdown(
                """
                <style>
                    .semantic-card {
                        background: linear-gradient(135deg, rgba(247,249,252,0.97), rgba(255,255,255,0.9));
                        border: 1px solid rgba(229,234,242,0.9);
                        border-radius: 18px;
                        padding: 1.1rem 1.25rem;
                        margin-bottom: 1.2rem;
                        box-shadow: 0 6px 16px rgba(15, 42, 98, 0.04);
                    }
                    .semantic-card h4 {
                        font-size: 1.05rem;
                        margin-bottom: 0.75rem;
                        font-weight: 600;
                    }
                    .semantic-chip {
                        display: inline-block;
                        padding: 0.25rem 0.7rem;
                        border-radius: 999px;
                        font-size: 0.8rem;
                        font-weight: 600;
                        margin-right: 0.4rem;
                        margin-bottom: 0.35rem;
                        color: #2e3a59;
                        background: rgba(67,97,238,0.08);
                        border: 1px solid rgba(67,97,238,0.18);
                    }
                    .semantic-metric-label {
                        font-size: 0.75rem;
                        letter-spacing: 0.05em;
                        text-transform: uppercase;
                        color: #6b7a99;
                    }
                    .semantic-metric-value {
                        font-size: 1.45rem;
                        font-weight: 600;
                        color: #2e3a59;
                    }
                    .semantic-hint {
                        font-size: 0.9rem;
                        color: #51607a;
                        margin-bottom: 0;
                    }
                    .semantic-badge {
                        display: inline-flex;
                        align-items: center;
                        font-size: 0.8rem;
                        padding: 0.3rem 0.6rem;
                        border-radius: 999px;
                        background: rgba(20, 184, 166, 0.12);
                        color: #0f766e;
                        border: 1px solid rgba(20, 184, 166, 0.25);
                        margin-right: 0.4rem;
                        margin-bottom: 0.35rem;
                    }
                    .semantic-badge span {
                        font-weight: 600;
                        margin-left: 0.35rem;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.session_state.semantic_styles_applied = True

        if not model_names:
            st.warning("Не удалось обнаружить доступные модели. Выполните обучение на этапе 5.")
            st.stop()

        if st.session_state.get('semantic_model_select') not in model_names:
            st.session_state.semantic_model_select = model_names[0]

        selected_model = st.session_state.get('semantic_model_select')

        tab_similarity, tab_analogies, tab_axes, tab_neighbors, tab_report = st.tabs([
            "6.1 Косинусное расстояние и сходство",
            "6.2 Векторные аналогии",
            "6.3 Семантические оси",
            "6.4 Ближайшие соседи",
            "6.5 Динамический отчёт"
        ])

        with tab_similarity:
            st.subheader("6.1. Косинусное расстояние и семантическое сходство")

            selected_model = st.selectbox(
                "Выберите обученную модель",
                options=model_names,
                index=0,
                key="semantic_model_select"
            )

            if selected_model:
                if st.session_state.get('semantic_active_model') != selected_model:
                    st.session_state.semantic_active_model = selected_model
                    st.session_state.pop('semantic_pair_report', None)
                    st.session_state.pop('semantic_matrix_result', None)
                    st.session_state.pop('semantic_group_report', None)
                    st.session_state.pop('semantic_manual_analogy_result', None)
                    st.session_state.pop('semantic_category_analogy_result', None)

                if 'semantic_pair_report' not in st.session_state:
                    st.session_state.semantic_pair_report = None
                if 'semantic_matrix_result' not in st.session_state:
                    st.session_state.semantic_matrix_result = None
                if 'semantic_group_report' not in st.session_state:
                    st.session_state.semantic_group_report = None

                model_stats = semantic_ops.get_model_statistics(selected_model)
                if model_stats:
                    with st.container():
                        st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                        st.markdown("#### 🔍 Профиль модели")
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        stat_col1.markdown(
                            f"""
                            <div class="semantic-metric-label">Размер словаря</div>
                            <div class="semantic-metric-value">{model_stats.get('vocabulary_size', 0):,}</div>
                            """,
                            unsafe_allow_html=True
                        )
                        stat_col2.markdown(
                            f"""
                            <div class="semantic-metric-label">Размерность векторов</div>
                            <div class="semantic-metric-value">{model_stats.get('vector_size', '—')}</div>
                            """,
                            unsafe_allow_html=True
                        )
                        stat_col3.markdown(
                            f"""
                            <div class="semantic-metric-label">Размер окна</div>
                            <div class="semantic-metric-value">{model_stats.get('window_size', '—')}</div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("### 🔍 Анализ косинусного сходства для выбранных пар слов")

                default_pairs = [
                    ("компьютер", "ноутбук"),
                    ("данные", "информация"),
                    ("город", "столица"),
                    ("женщина", "девушка"),
                    ("работа", "труд")
                ]
                default_pairs_text = "\n".join([", ".join(pair) for pair in default_pairs])

                with st.container():
                    st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                    st.markdown("#### ✏️ Настройка пар слов")
                    pairs_input = st.text_area(
                        "Укажите пары слов (по одной паре в строке, разделитель — запятая)",
                        value=default_pairs_text,
                        help="Пример строки: компьютер, ноутбук"
                    )
                    st.markdown(
                        '<p class="semantic-hint">Добавьте свои пары или используйте предложенные по умолчанию.</p>',
                        unsafe_allow_html=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                parsed_pairs = []
                for line in pairs_input.splitlines():
                    parts = [part.strip() for part in line.split(",") if part.strip()]
                    if len(parts) >= 2:
                        parsed_pairs.append((parts[0], parts[1]))

                with st.container():
                    st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                    compute_pairs_btn = st.button(
                        "🔁 Рассчитать косинусные сходства для пар",
                        key="compute_pairs_similarity"
                    )

                    if compute_pairs_btn:
                        if parsed_pairs:
                            with st.spinner("Вычисляем косинусные сходства для пар слов..."):
                                similarity_report = semantic_ops.cosine_similarity_analysis(selected_model, parsed_pairs)

                            st.session_state.semantic_pair_report = {
                                "pairwise_results": similarity_report.get('pairwise_analysis', []),
                                "distribution_stats": similarity_report.get('distribution_analysis', {}),
                                "input_pairs": list(parsed_pairs)
                            }
                        else:
                            st.warning("Добавьте хотя бы одну корректную пару слов для анализа.")
                            st.session_state.semantic_pair_report = None

                    pair_report = st.session_state.get('semantic_pair_report')
                    if pair_report:
                        if pair_report.get('input_pairs') != parsed_pairs:
                            st.info("Список пар изменён. Нажмите кнопку, чтобы пересчитать результаты.")

                        pairwise_results = pair_report.get('pairwise_results', [])
                        if pairwise_results:
                            st.markdown("##### 📋 Результаты по парам слов")
                            pairwise_df = pd.DataFrame(pairwise_results)
                            st.dataframe(pairwise_df, use_container_width=True)

                        distribution_stats = pair_report.get('distribution_stats') or {}
                        if distribution_stats:
                            st.markdown("##### 📈 Параметры распределения")
                            col_ds1, col_ds2, col_ds3, col_ds4 = st.columns(4)
                            col_ds1.metric("Среднее", f"{distribution_stats.get('mean_similarity', 0):.3f}")
                            col_ds2.metric("Ст. отклонение", f"{distribution_stats.get('std_similarity', 0):.3f}")
                            col_ds3.metric("Мин.", f"{distribution_stats.get('min_similarity', 0):.3f}")
                            col_ds4.metric("Макс.", f"{distribution_stats.get('max_similarity', 0):.3f}")
                    else:
                        st.info("Нажмите кнопку выше, чтобы выполнить расчёт сходства для пар.")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.session_state.setdefault('semantic_distance_cache', {})

                st.markdown("### 📊 Распределение косинусных расстояний и семантическая матрица")
                refresh_distance = st.button(
                    "🔄 Пересчитать распределение",
                    key=f"refresh_distance_{selected_model}"
                )

                cached_distance = st.session_state.semantic_distance_cache.get(selected_model)
                if refresh_distance or not cached_distance:
                    with st.spinner("Строим распределение расстояний..."):
                        distance_report = semantic_ops.analyze_distance_distribution(selected_model)
                    st.session_state.semantic_distance_cache[selected_model] = distance_report
                else:
                    distance_report = cached_distance

                if distance_report:
                    dist_info = distance_report.get('distance_distribution', {})
                    with st.container():
                        st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                        st.markdown("#### 🎯 Статистика распределения расстояний")
                        col_md1, col_md2, col_md3 = st.columns(3)
                        col_md1.metric("Среднее расстояние", f"{distance_report.get('mean_distance', 0):.3f}")
                        col_md2.metric("Ст. отклонение", f"{distance_report.get('std_distance', 0):.3f}")
                        col_md3.metric("Диапазон", f"{distance_report.get('min_distance', 0):.3f} – {distance_report.get('max_distance', 0):.3f}")

                    if dist_info:
                        hist = dist_info.get('histogram', [])
                        bin_centers = dist_info.get('bin_centers', [])
                        if hist and bin_centers:
                            fig_hist, ax_hist = plt.subplots()
                            bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.05
                            ax_hist.bar(bin_centers, hist, width=bin_width, color="#4C72B0", alpha=0.8)
                            ax_hist.set_xlabel("Косинусное расстояние")
                            ax_hist.set_ylabel("Частота")
                            ax_hist.set_title("Распределение косинусных расстояний (случайная выборка слов)")
                            st.pyplot(fig_hist)
                        st.markdown("</div>", unsafe_allow_html=True)

                    demo_word_pool = [
                        "компьютер", "программа", "данные", "информация", "алгоритм",
                        "система", "технология", "разработка", "сеть", "база"
                    ]
                    available_demo_words = [
                        word for word in demo_word_pool
                        if word in available_models[selected_model].wv
                    ]
                    selected_demo_words = available_demo_words[:10]

                    if len(selected_demo_words) >= 2:
                        demo_vectors = np.array([available_models[selected_model].wv[word] for word in selected_demo_words])
                        demo_norms = np.linalg.norm(demo_vectors, axis=1, keepdims=True)
                        demo_norms[demo_norms == 0] = 1.0
                        demo_vectors = demo_vectors / demo_norms
                        demo_similarity_matrix = np.dot(demo_vectors, demo_vectors.T)

                        with st.container():
                            st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                            st.markdown("#### 🟩 Демонстрационная матрица (10 слов)")
                            st.markdown(
                                "".join([f'<span class="semantic-chip">{word}</span>' for word in selected_demo_words]),
                                unsafe_allow_html=True
                            )
                            demo_df = pd.DataFrame(demo_similarity_matrix, index=selected_demo_words, columns=selected_demo_words)
                            st.dataframe(demo_df, use_container_width=True)

                            fig_heat, ax_heat = plt.subplots(figsize=(5, 4))
                            cax = ax_heat.imshow(demo_df.values, cmap="viridis", vmin=-1, vmax=1)
                            ax_heat.set_xticks(range(len(selected_demo_words)))
                            ax_heat.set_xticklabels(selected_demo_words, rotation=90)
                            ax_heat.set_yticks(range(len(selected_demo_words)))
                            ax_heat.set_yticklabels(selected_demo_words)
                            ax_heat.set_title("Тепловая карта семантической близости (10 слов)")
                            fig_heat.colorbar(cax, fraction=0.046, pad=0.04)
                            st.pyplot(fig_heat)
                            st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("### 🧩 Матрица семантической близости для выбранных слов")

                with st.container():
                    st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                    st.markdown("#### 🎯 Ваш набор слов")
                    default_test_words = "компьютер, ноутбук, данные, информация, система"
                    user_words_input = st.text_area(
                        "Введите слова через запятую (минимум два слова) для построения матрицы",
                        value=default_test_words,
                        key="semantic_user_words"
                    )
                    st.markdown(
                        '<p class="semantic-hint">Совет: комбинируйте разные тематические группы, чтобы увидеть семантические связи.</p>',
                        unsafe_allow_html=True
                    )

                matrix_result = st.session_state.get('semantic_matrix_result')
                if matrix_result and matrix_result.get('source_input') != user_words_input:
                    st.info("Список слов был изменён. Нажмите кнопку для обновления матрицы.")

                build_matrix_btn = st.button(
                    "Построить матрицу сходства",
                    key="build_user_similarity"
                )

                if build_matrix_btn:
                    test_words = [w.strip() for w in user_words_input.split(",") if w.strip()]

                    if len(test_words) < 2:
                        st.warning("Укажите минимум два слова для построения матрицы.")
                        st.session_state.semantic_matrix_result = None
                    else:
                        available_words = [word for word in test_words if word in available_models[selected_model].wv]
                        missing_words = [word for word in test_words if word and word not in available_models[selected_model].wv]

                        if missing_words:
                            st.warning(f"Следующие слова отсутствуют в словаре модели: {', '.join(missing_words)}")

                        if len(available_words) >= 2:
                            vectors = np.array([available_models[selected_model].wv[word] for word in available_words])
                            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                            norms[norms == 0] = 1.0
                            normalized_vectors = vectors / norms
                            similarity_matrix_test = np.dot(normalized_vectors, normalized_vectors.T)

                            st.session_state.semantic_matrix_result = {
                                "available_words": available_words,
                                "matrix": similarity_matrix_test.tolist(),
                                "missing_words": missing_words,
                                "source_input": user_words_input
                            }
                        else:
                            st.info("Недостаточно слов из словаря модели для построения матрицы.")
                            st.session_state.semantic_matrix_result = None

                matrix_result = st.session_state.get('semantic_matrix_result')
                if matrix_result:
                    available_words = matrix_result.get('available_words', [])
                    missing_words = matrix_result.get('missing_words', [])

                    with st.container():
                        st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                        st.markdown("#### 🧾 Матрица по выбранным словам")

                        if missing_words:
                            st.warning(f"Следующие слова отсутствуют в словаре модели: {', '.join(missing_words)}")

                        if len(available_words) >= 2:
                            similarity_df = pd.DataFrame(
                                matrix_result.get('matrix', []),
                                index=available_words,
                                columns=available_words
                            )

                            st.dataframe(similarity_df, use_container_width=True)

                            fig_test, ax_test = plt.subplots(figsize=(6, 5))
                            cax_test = ax_test.imshow(similarity_df.values, cmap="magma", vmin=-1, vmax=1)
                            ax_test.set_xticks(range(len(available_words)))
                            ax_test.set_xticklabels(available_words, rotation=90)
                            ax_test.set_yticks(range(len(available_words)))
                            ax_test.set_yticklabels(available_words)
                            ax_test.set_title("Матрица косинусного сходства для выбранных слов")
                            fig_test.colorbar(cax_test, fraction=0.046, pad=0.04)
                            st.pyplot(fig_test)
                        else:
                            st.info("Для построения матрицы необходимо минимум два слова из словаря модели.")
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("Матрица ещё не построена. Нажмите кнопку, чтобы выполнить расчёт.")

                st.markdown("### ⚖️ Оценка сходства для синонимов, антонимов и тематических пар")

                compute_groups_btn = st.button(
                    "Рассчитать показатели по семантическим группам",
                    key="compute_semantic_groups"
                )

                if compute_groups_btn:
                    test_sets = get_russian_test_sets()

                    semantic_groups = {
                        "Синонимы": [
                            ("компьютер", "ноутбук"),
                            ("программа", "приложение"),
                            ("данные", "информация"),
                            ("женщина", "девушка"),
                            ("работа", "труд")
                        ],
                        "Антонимы": [
                            ("хороший", "плохой"),
                            ("день", "ночь"),
                            ("высокий", "низкий"),
                            ("горячий", "холодный"),
                            ("мир", "война")
                        ],
                        "Тематические пары": test_sets.get('semantic_relationships', [])
                    }

                    aggregate_rows = []
                    detailed_results = {}

                    for group_name, group_pairs in semantic_groups.items():
                        valid_pairs = [(pair[0], pair[1]) for pair in group_pairs if len(pair) >= 2]
                        if not valid_pairs:
                            detailed_results[group_name] = []
                            continue

                        group_report = semantic_ops.cosine_similarity_analysis(selected_model, valid_pairs)
                        pair_results = group_report.get('pairwise_analysis', [])
                        detailed_results[group_name] = pair_results

                        similarities = [row['cosine_similarity'] for row in pair_results if row.get('cosine_similarity') is not None]
                        coverage = sum(1 for row in pair_results if row.get('cosine_similarity') is not None)

                        aggregate_rows.append({
                            "Группа": group_name,
                            "Среднее сходство": float(np.mean(similarities)) if similarities else None,
                            "Ст. отклонение": float(np.std(similarities)) if similarities else None,
                            "Покрытых пар": f"{coverage}/{len(valid_pairs)}"
                        })

                    st.session_state.semantic_group_report = {
                        "aggregate": aggregate_rows,
                        "details": detailed_results
                    }

                group_report = st.session_state.get('semantic_group_report')
                if group_report:
                    aggregate_rows = group_report.get('aggregate') or []
                    if aggregate_rows:
                        with st.container():
                            st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                            st.markdown("#### 📚 Итоги по группам")
                            aggregate_df = pd.DataFrame(aggregate_rows)
                            st.dataframe(aggregate_df, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("Недостаточно данных для расчёта агрегированных показателей.")

                    details = group_report.get('details') or {}
                    if details:
                        with st.expander("Детали по группам слов", expanded=False):
                            tabs_labels = list(details.keys())
                            if tabs_labels:
                                detail_tabs = st.tabs(tabs_labels)
                                for tab_widget, group_name in zip(detail_tabs, tabs_labels):
                                    with tab_widget:
                                        group_items = details.get(group_name, [])
                                        if group_items:
                                            st.markdown(
                                                "".join([f'<span class="semantic-chip">{pair["word_pair"]}</span>' for pair in group_items if pair.get("word_pair")]),
                                                unsafe_allow_html=True
                                            )
                                            st.dataframe(pd.DataFrame(group_items), use_container_width=True)
                                        else:
                                            st.info("Нет данных для отображения.")
                    else:
                        st.info("Подробные результаты отсутствуют.")
                else:
                    st.info("Нажмите кнопку, чтобы выполнить расчёт по семантическим группам.")

        with tab_analogies:
            st.subheader("6.2. Векторная арифметика и word analogies")

            selected_model_name = st.session_state.get('semantic_model_select')

            if not selected_model_name:
                st.warning("Сначала выберите модель на вкладке 6.1 и укажите данные для расчёта.")
            else:
                model_ref = available_models.get(selected_model_name)
                if model_ref is None:
                    st.error("Не удалось получить обученную модель. Повторите обучение на этапе 5.")
                else:
                    st.session_state.setdefault('manual_word_a', 'мужчина')
                    st.session_state.setdefault('manual_word_b', 'женщина')
                    st.session_state.setdefault('manual_word_c', 'король')
                    st.session_state.setdefault('manual_topn', 5)
                    st.session_state.setdefault('manual_preset_choice', "—")
                    st.session_state.setdefault('manual_pending_update', None)
                    st.session_state.setdefault('manual_reset_choice', False)
                    st.session_state.setdefault('semantic_manual_analogy_result', None)
                    st.session_state.setdefault('semantic_category_analogy_result', None)

                    if st.session_state.pop('manual_reset_choice', False):
                        st.session_state.manual_preset_choice = "—"

                    pending_update = st.session_state.pop('manual_pending_update', None)
                    if pending_update is not None:
                        a_val, b_val, c_val = pending_update
                        st.session_state.manual_word_a = a_val
                        st.session_state.manual_word_b = b_val
                        st.session_state.manual_word_c = c_val

                    preset_analogies = {
                        "👑 Мужчина − Женщина + Король": ("мужчина", "женщина", "король"),
                        "🌍 Москва − Россия + Франция": ("Москва", "Россия", "Франция"),
                        "⚖️ Хороший − Лучше + Плохой": ("хороший", "лучше", "плохой"),
                        "✍️ Делать − Сделал + Писать": ("делать", "сделал", "писать"),
                        "🏙️ Лондон − Англия + Германия": ("Лондон", "Англия", "Германия")
                    }

                    with st.container():
                        st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                        st.markdown("#### 🧮 Векторная арифметика (A − B + C)")
                        st.markdown(
                            '<p class="semantic-hint">Введите три слова и получите список слов, которые модель считает подходящими к выражению A − B + C.</p>',
                            unsafe_allow_html=True
                        )

                        preset_options_list = ["—"] + list(preset_analogies.keys())
                        current_preset = st.session_state.get('manual_preset_choice', "—")
                        if current_preset not in preset_options_list:
                            current_preset = "—"
                            st.session_state.manual_preset_choice = "—"

                        preset_choice = st.selectbox(
                            "Быстрый пример",
                            options=preset_options_list,
                            index=preset_options_list.index(current_preset),
                            key="manual_preset_choice"
                        )

                        preset_cols = st.columns([1, 1, 2])
                        with preset_cols[0]:
                            apply_preset = st.button("Подставить пример", key="apply_manual_preset")
                        with preset_cols[1]:
                            clear_inputs = st.button("Очистить ввод", key="clear_manual_inputs")

                        if apply_preset:
                            choice = st.session_state.get('manual_preset_choice', "—")
                            if choice != "—":
                                st.session_state.manual_pending_update = preset_analogies.get(choice, ("", "", ""))
                                st.session_state.semantic_manual_analogy_result = None

                        if clear_inputs:
                            st.session_state.manual_pending_update = ("", "", "")
                            st.session_state.manual_reset_choice = True
                            st.session_state.semantic_manual_analogy_result = None

                        col_a, col_b, col_c = st.columns(3)
                        word_a = col_a.text_input(
                            "Слово A (исходное)",
                            value=st.session_state.get('manual_word_a', 'мужчина'),
                            key="manual_word_a"
                        )
                        word_b = col_b.text_input(
                            "Слово B (вычесть)",
                            value=st.session_state.get('manual_word_b', 'женщина'),
                            key="manual_word_b"
                        )
                        word_c = col_c.text_input(
                            "Слово C (добавить)",
                            value=st.session_state.get('manual_word_c', 'король'),
                            key="manual_word_c"
                        )

                        topn = st.slider(
                            "Количество результатов (Top-N)",
                            min_value=1,
                            max_value=10,
                            value=int(st.session_state.get('manual_topn', 5)),
                            key="manual_topn"
                        )

                        run_manual_btn = st.button("🔎 Выполнить аналогию", key="run_manual_analogy")
                        st.markdown("</div>", unsafe_allow_html=True)

                    if run_manual_btn:
                        words = [word_a.strip(), word_b.strip(), word_c.strip()]
                        if any(not w for w in words):
                            st.warning("Заполните все три слова, чтобы выполнить векторную арифметику.")
                            st.session_state.semantic_manual_analogy_result = None
                        else:
                            missing_words = [w for w in words if w not in model_ref.wv]
                            if missing_words:
                                st.warning(f"Следующие слова отсутствуют в словаре модели: {', '.join(missing_words)}")
                                st.session_state.semantic_manual_analogy_result = {
                                    "error": f"Слова отсутствуют в словаре: {', '.join(missing_words)}",
                                    "words": tuple(words)
                                }
                            else:
                                try:
                                    results = model_ref.wv.most_similar(
                                        positive=[word_a, word_c],
                                        negative=[word_b],
                                        topn=int(topn)
                                    )
                                    st.session_state.semantic_manual_analogy_result = {
                                        "words": tuple(words),
                                        "topn": int(topn),
                                        "results": [(candidate, float(score)) for candidate, score in results]
                                    }
                                except Exception as err:
                                    st.error(f"Не удалось выполнить аналогию: {err}")
                                    st.session_state.semantic_manual_analogy_result = {
                                        "error": f"Ошибка выполнения: {err}",
                                        "words": tuple(words)
                                    }

                    manual_result = st.session_state.get('semantic_manual_analogy_result')
                    current_words_tuple = (word_a.strip(), word_b.strip(), word_c.strip())
                    if manual_result and not manual_result.get('error') and manual_result.get('words'):
                        if tuple(w.strip() for w in manual_result.get('words')) != current_words_tuple:
                            st.info("Вы изменили входные слова. Нажмите «Выполнить аналогию», чтобы пересчитать результаты.")

                    if manual_result:
                        with st.container():
                            st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                            st.markdown("#### 📊 Результаты аналогии")

                            error_text = manual_result.get('error')
                            if error_text:
                                st.warning(error_text)
                            else:
                                words_tuple = manual_result.get('words', ("", "", ""))
                                expression = f"{words_tuple[0]} − {words_tuple[1]} + {words_tuple[2]}"
                                st.markdown(f"**Выражение:** `{expression}`")

                                results_list = manual_result.get('results', [])
                                if results_list:
                                    top_word, top_score = results_list[0]
                                    badge_html = (
                                        f'<span class="semantic-badge">Ответ<span>{top_word}</span></span>'
                                        f'<span class="semantic-badge">Сходство<span>{top_score:.3f}</span></span>'
                                    )
                                    st.markdown(badge_html, unsafe_allow_html=True)

                                    manual_df = pd.DataFrame(results_list, columns=["Слово", "Сходство"])
                                    st.dataframe(manual_df, use_container_width=True)
                                else:
                                    st.info("Модель не вернула кандидатов для указанного выражения.")
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("Введите слова и нажмите кнопку, чтобы получить результат выражения A − B + C.")

                    with st.container():
                        st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                        st.markdown("#### 📚 Категории аналогий")
                        st.markdown(
                            '<p class="semantic-hint">Проверяем три группы: столицы стран, степени сравнения прилагательных и глагольные формы прошедшего времени.</p>',
                            unsafe_allow_html=True
                        )
                        run_categories_btn = st.button("🚀 Запустить оценку категорий", key="run_category_analogies")
                        st.markdown("</div>", unsafe_allow_html=True)

                    if run_categories_btn:
                        with st.spinner("Вычисляем точность аналогий по категориям..."):
                            category_eval = semantic_ops.categorical_analogy_evaluation(selected_model_name)
                        st.session_state.semantic_category_analogy_result = category_eval

                    category_result = st.session_state.get('semantic_category_analogy_result')
                    if category_result:
                        label_map = {
                            "semantic_capitals": "Столицы стран",
                            "semantic_gender": "Родовые пары",
                            "syntactic_comparative": "Степени сравнения",
                            "morphological_verbs": "Глаголы прошедшего времени"
                        }

                        with st.container():
                            st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                            st.markdown("#### 📈 Сводка по аналогиям")
                            overall_accuracy = category_result.get('overall_accuracy', 0.0)
                            total_tests = category_result.get('total_tests', 0)
                            total_correct = category_result.get('total_correct', 0)

                            col_acc1, col_acc2, col_acc3 = st.columns(3)
                            col_acc1.metric("Общая точность", f"{overall_accuracy * 100:.1f}%")
                            col_acc2.metric("Количество тестов", total_tests)
                            col_acc3.metric("Верных ответов", total_correct)

                            summary_rows = []
                            for key, value in category_result.items():
                                if key in ("overall_accuracy", "total_tests", "total_correct"):
                                    continue
                                total = value.get('total', 0)
                                if total == 0:
                                    accuracy_text = "—"
                                else:
                                    accuracy_text = f"{value.get('accuracy', 0.0) * 100:.1f}%"
                                summary_rows.append({
                                    "Категория": label_map.get(key, key),
                                    "Тестов": total,
                                    "Верно": value.get('correct', 0),
                                    "Точность": accuracy_text
                                })

                            if summary_rows:
                                summary_df = pd.DataFrame(summary_rows)
                                st.dataframe(summary_df, use_container_width=True)
                            else:
                                st.info("Для выбранной модели не удалось собрать тесты аналогий.")
                            st.markdown("</div>", unsafe_allow_html=True)

                        with st.expander("Детали по категориям", expanded=False):
                            for key, value in category_result.items():
                                if key in ("overall_accuracy", "total_tests", "total_correct"):
                                    continue
                                friendly_name = label_map.get(key, key)
                                details = value.get('details', [])
                                total = value.get('total', 0)
                                correct = value.get('correct', 0)
                                accuracy_pct = value.get('accuracy', 0.0) * 100 if total else 0.0

                                st.markdown(
                                    f"**{friendly_name}** — точность {accuracy_pct:.1f}% ({correct}/{total})"
                                )
                                if details:
                                    detail_df = pd.DataFrame([
                                        {
                                            "Аналогия": item.get('analogy'),
                                            "Ожидаемый ответ": item.get('expected'),
                                            "Предсказание": item.get('predicted'),
                                            "Топ-1": "✅" if item.get('is_correct') else "❌",
                                            "Топ-5": ", ".join(item.get('top_5', []))
                                        }
                                        for item in details
                                    ])
                                    st.dataframe(detail_df, use_container_width=True)
                                else:
                                    st.info("Недостаточно данных: слова отсутствуют в словаре модели.")
                    else:
                        st.info("Нажмите кнопку, чтобы выполнить проверку аналогий по категориям.")

        with tab_axes:
            st.subheader("6.3. Анализ семантических осей")

            selected_model_name = st.session_state.get('semantic_model_select')
            if not selected_model_name:
                st.warning("Выберите модель на вкладке 6.1, чтобы выполнить анализ осей.")
            else:
                st.session_state.setdefault('semantic_axes_cache', {})
                st.session_state.setdefault('semantic_axes_custom_words', "мужчина, женщина, программист, учитель, хороший, плохой, успех, провал")
                st.session_state.setdefault('semantic_axes_topn', 8)

                with st.container():
                    st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                    st.markdown("#### 🧭 Описание")
                    st.markdown(
                        """
                        Анализируем встроенные оси (гендерную, профессиональную, оценочную и временную), 
                        измеряем смещение и смотрим, какие слова оказываются на полюсах. Вы также можете 
                        проецировать собственный набор слов на каждую ось.
                        """
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                axes_topn = st.slider(
                    "Сколько слов показывать на каждом полюсе",
                    min_value=3,
                    max_value=15,
                    value=int(st.session_state.get('semantic_axes_topn', 8)),
                    key="semantic_axes_topn"
                )

                custom_words_input = st.text_area(
                    "Слова для проекции (через запятую)",
                    key="semantic_axes_custom_words"
                )

                compute_axes_btn = st.button("🔍 Выполнить анализ осей", key="compute_semantic_axes")

                if compute_axes_btn:
                    with st.spinner("Вычисляем семантические оси..."):
                        axes_result = semantic_ops.comprehensive_axes_analysis(selected_model_name)
                    st.session_state.semantic_axes_cache[selected_model_name] = axes_result

                axes_result = st.session_state.semantic_axes_cache.get(selected_model_name)

                if not axes_result:
                    st.info("Нажмите кнопку выше, чтобы выполнить анализ семантических осей для выбранной модели.")
                else:
                    axis_labels = {
                        "gender_axis": "Гендерная ось",
                        "profession_axis": "Профессиональная ось",
                        "evaluation_axis": "Оценочная ось",
                        "temporal_axis": "Временная ось"
                    }

                    summary_rows = []
                    for axis_key, axis_data in axes_result.items():
                        summary_rows.append({
                            "Ось": axis_labels.get(axis_key, axis_key),
                            "Сила оси": float(axis_data.get('axis_strength', 0.0)),
                            "Смещение": float(axis_data.get('bias_metric', 0.0))
                        })

                    with st.container():
                        st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                        st.markdown("#### 📈 Сводные показатели")
                        if summary_rows:
                            summary_df = pd.DataFrame(summary_rows)
                            summary_df["Сила оси"] = summary_df["Сила оси"].map(lambda x: f"{x:.3f}")
                            summary_df["Смещение"] = summary_df["Смещение"].map(lambda x: f"{x:.3f}")
                            st.dataframe(summary_df, use_container_width=True)
                        else:
                            st.info("Не удалось вычислить показатели для выбранной модели.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    custom_words = [w.strip() for w in custom_words_input.split(',') if w.strip()]

                    combined_custom_df = []
                    missing_overall = set()

                    for axis_key, axis_data in axes_result.items():
                        axis_name = axis_labels.get(axis_key, axis_key)
                        positive_df = pd.DataFrame(axis_data.get('positive_end') or [], columns=["Слово", "Проекция"])
                        negative_df = pd.DataFrame(axis_data.get('negative_end') or [], columns=["Слово", "Проекция"])
                        full_df = pd.DataFrame(axis_data.get('all_projections') or [], columns=["Слово", "Проекция"])

                        display_df = full_df if not full_df.empty else pd.concat([positive_df, negative_df], ignore_index=True)

                        if not display_df.empty:
                            positive_display = display_df.sort_values("Проекция", ascending=False).head(int(axes_topn))
                            negative_display = display_df.sort_values("Проекция", ascending=True).head(int(axes_topn))
                        else:
                            positive_display = positive_df.sort_values("Проекция", ascending=False).head(int(axes_topn))
                            negative_display = negative_df.sort_values("Проекция", ascending=True).head(int(axes_topn))

                        with st.container():
                            st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                            st.markdown(f"#### 🧭 {axis_name}")

                            metric_cols = st.columns(2)
                            metric_cols[0].metric("Сила оси", f"{axis_data.get('axis_strength', 0.0):.3f}")
                            metric_cols[1].metric("Смещение", f"{axis_data.get('bias_metric', 0.0):.3f}")

                            axis_combined = pd.concat([
                                positive_display.assign(Полюс="Положительный"),
                                negative_display.assign(Полюс="Отрицательный")
                            ], ignore_index=True)

                            if not axis_combined.empty:
                                axis_combined["Проекция"] = axis_combined["Проекция"].astype(float)
                                axis_combined.sort_values("Проекция", ascending=True, inplace=True)
                                max_abs_axis = float(max(abs(axis_combined["Проекция"].min()), abs(axis_combined["Проекция"].max()), 1e-6))
                                chart_height_axis = max(160, 28 * len(axis_combined))

                                axis_chart = alt.Chart(axis_combined).mark_bar().encode(
                                    y=alt.Y('Слово:N', sort=None, title=''),
                                    x=alt.X(
                                        'Проекция:Q',
                                        title='Проекция вдоль оси',
                                        scale=alt.Scale(domain=[-max_abs_axis, max_abs_axis], zero=True, nice=False)
                                    ),
                                    color=alt.Color('Полюс:N', scale=alt.Scale(range=['#ff6b6b', '#4dabf7'])),
                                    tooltip=[
                                        alt.Tooltip('Слово:N', title='Слово'),
                                        alt.Tooltip('Полюс:N', title='Полюс'),
                                        alt.Tooltip('Проекция:Q', format='.3f', title='Проекция')
                                    ]
                                ).properties(height=chart_height_axis, width='container')

                                axis_zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='#6b7a99', strokeDash=[4, 4]).encode(x='x:Q')
                                st.altair_chart(axis_chart + axis_zero_line, use_container_width=True)
                            else:
                                st.info("Недостаточно данных для визуализации оси.")

                        if custom_words:
                            projections, missing_words = semantic_ops.project_words_on_axis(
                                selected_model_name,
                                axis_data.get('axis_direction'),
                                custom_words
                            )
                            if projections:
                                custom_df = pd.DataFrame(projections, columns=["Слово", "Проекция"])
                                custom_df["Проекция"] = custom_df["Проекция"].astype(float)
                                
                                # Сохраняем исходный порядок слов для правильного отображения
                                custom_df_sorted = custom_df.copy()
                                custom_df_sorted.sort_values("Проекция", ascending=True, inplace=True)
                                custom_df_sorted["Полюс"] = np.where(custom_df_sorted["Проекция"] >= 0, "Положительный", "Отрицательный")
                                
                                # Создаем список слов в порядке отображения (от отрицательных к положительным)
                                word_order = custom_df_sorted["Слово"].tolist()
                                
                                max_abs_custom = float(max(abs(custom_df_sorted["Проекция"].min()), abs(custom_df_sorted["Проекция"].max()), 1e-6))
                                # Увеличиваем высоту графика, чтобы все слова были видны
                                chart_height_custom = max(200, 35 * len(custom_df_sorted))
                                
                                st.markdown(f"**Ваши слова** (отображено {len(custom_df_sorted)} из {len(custom_words)} слов)")

                                custom_chart = alt.Chart(custom_df_sorted).mark_bar().encode(
                                    y=alt.Y('Слово:N', sort=word_order, title=''),
                                    x=alt.X(
                                        'Проекция:Q',
                                        title='Проекция вдоль оси',
                                        scale=alt.Scale(domain=[-max_abs_custom, max_abs_custom], zero=True, nice=False)
                                    ),
                                    color=alt.Color('Полюс:N', scale=alt.Scale(range=['#ff6b6b', '#4dabf7'])),
                                    tooltip=[
                                        alt.Tooltip('Слово:N', title='Слово'),
                                        alt.Tooltip('Проекция:Q', format='.3f', title='Проекция')
                                    ]
                                ).properties(
                                    height=min(chart_height_custom, 800),  # Ограничиваем максимальную высоту для производительности
                                    width='container'
                                )

                                custom_zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='#6b7a99', strokeDash=[4, 4]).encode(x='x:Q')
                                st.altair_chart(custom_chart + custom_zero_line, use_container_width=True)
                                
                                # Также показываем таблицу для удобства просмотра всех слов
                                with st.expander(f"📋 Таблица всех слов для оси {axis_name}", expanded=False):
                                    display_df = custom_df_sorted[["Слово", "Проекция", "Полюс"]].copy()
                                    st.dataframe(display_df, use_container_width=True)

                                custom_df_sorted["Ось"] = axis_name
                                combined_custom_df.append(custom_df_sorted)
                            elif missing_words and len(missing_words) == len(custom_words):
                                st.info(f"Все введенные слова отсутствуют в словаре модели для оси {axis_name}.")
                            if missing_words:
                                missing_overall.update(missing_words)

                        st.markdown("</div>", unsafe_allow_html=True)

                    if combined_custom_df:
                        merged_df = pd.concat(combined_custom_df, ignore_index=True)
                        with st.container():
                            st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                            st.markdown("#### 🗺️ Проекции ваших слов")
                            st.dataframe(merged_df, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                    if missing_overall:
                        st.warning(
                            "Следующие слова отсутствуют в словаре модели: " + ", ".join(sorted(missing_overall))
                        )

        with tab_neighbors:
            st.subheader("6.4. Качественный анализ ближайших соседей")

            selected_model_name = st.session_state.get('semantic_model_select')
            if not selected_model_name:
                st.warning("Выберите модель на вкладке 6.1, чтобы выполнить анализ соседей.")
            else:
                st.session_state.setdefault('semantic_neighbors_cache', {})
                st.session_state.setdefault('semantic_neighbors_words', "компьютер, программа, данные, город, хороший, работа, время, женщина, мужчина, технология, система")
                st.session_state.setdefault('semantic_neighbors_topk', 10)

                with st.container():
                    st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                    st.markdown("#### 🔍 Описание")
                    st.markdown(
                        """
                        Находим топ-10 ближайших соседей для выбранных слов, оцениваем их семантическую 
                        согласованность и фиксируем случаи смешения семантики и синтаксиса. 
                        """
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                topk_value = st.slider(
                    "Количество соседей (Top-N)",
                    min_value=5,
                    max_value=20,
                    value=int(st.session_state.get('semantic_neighbors_topk', 10)),
                    key="semantic_neighbors_topk"
                )

                neighbors_words_input = st.text_area(
                    "Слова для анализа (через запятую)",
                    key="semantic_neighbors_words"
                )

                analyze_neighbors_btn = st.button("🔎 Выполнить анализ соседей", key="compute_neighbors_analysis")

                cache_key = (selected_model_name, topk_value, tuple(sorted([w.strip() for w in neighbors_words_input.split(',') if w.strip()])))

                if analyze_neighbors_btn or cache_key not in st.session_state.semantic_neighbors_cache:
                    test_words = [w.strip() for w in neighbors_words_input.split(',') if w.strip()]
                    if not test_words:
                        st.warning("Укажите хотя бы одно слово для анализа соседей.")
                        neighbors_result = None
                    else:
                        with st.spinner("Собираем ближайших соседей..."):
                            neighbors_result = semantic_ops.nearest_neighbors_analysis(
                                selected_model_name,
                                test_words,
                                top_k=int(topk_value)
                            )
                        st.session_state.semantic_neighbors_cache[cache_key] = neighbors_result
                else:
                    neighbors_result = st.session_state.semantic_neighbors_cache.get(cache_key)

                if not neighbors_result:
                    st.info("Нажмите кнопку выше, чтобы выполнить анализ ближайших соседей.")
                else:
                    overall_analysis = neighbors_result.get('overall_analysis', {})

                    with st.container():
                        st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                        st.markdown("#### 📈 Сводные показатели")
                        col_n1, col_n2, col_n3 = st.columns(3)
                        col_n1.metric("Средняя согласованность", f"{overall_analysis.get('mean_semantic_coherence', 0):.3f}")
                        col_n2.metric("Ст. отклонение", f"{overall_analysis.get('semantic_coherence_std', 0):.3f}")
                        col_n3.metric("Слов проанализировано", overall_analysis.get('total_words_analyzed', 0))

                        neighbor_category_analysis = overall_analysis.get('neighbor_category_analysis', {})
                        if neighbor_category_analysis:
                            cat_rows = []
                            for category, values in neighbor_category_analysis.items():
                                cat_rows.append({
                                    "Категория": category,
                                    "Среднее количество": f"{values.get('mean_count', 0):.2f}",
                                    "Всего": values.get('total_occurrences', 0)
                                })
                            cat_df = pd.DataFrame(cat_rows)
                            st.dataframe(cat_df, use_container_width=True)
                        else:
                            st.info("Дополнительные категории соседей не выявлены.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    def classify_neighbor(target_word: str, neighbor_word: str, similarity: float) -> str:
                        if (neighbor_word in target_word or target_word in neighbor_word or
                                len(set(neighbor_word) & set(target_word)) > 3):
                            return "Морфологические"
                        if semantic_ops._check_syntactic_relation(target_word, neighbor_word):
                            return "Синтаксические"
                        if similarity > 0.6:
                            return "Семантические"
                        if semantic_ops._check_thematic_relation(target_word, neighbor_word):
                            return "Тематические"
                        return "Прочие"

                    color_scale_neighbor = alt.Scale(domain=["Семантические", "Морфологические", "Синтаксические", "Тематические", "Прочие"],
                                                     range=['#4dabf7', '#ffa94d', '#845ef7', '#51cf66', '#adb5bd'])

                    word_results = neighbors_result.get('word_analysis', {})
                    input_order = [w.strip() for w in neighbors_words_input.split(',') if w.strip()]

                    for word in input_order:
                        word_data = word_results.get(word)
                        if not word_data:
                            continue

                        with st.container():
                            st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                            st.markdown(f"#### 🔠 {word}")

                            status = word_data.get('status', 'success')
                            if status != 'success':
                                st.warning(f"{word_data.get('status', 'Недоступно')}")
                            else:
                                neighbors = word_data.get('neighbors', []) or []
                                semantic_coherence = word_data.get('semantic_coherence', 0)
                                avg_similarity = word_data.get('average_similarity', 0)
                                neighbor_types_counts = word_data.get('neighbor_types', {})

                                metric_cols = st.columns(2)
                                metric_cols[0].metric("Семантическая согласованность", f"{semantic_coherence:.3f}")
                                metric_cols[1].metric("Средняя похожесть", f"{avg_similarity:.3f}")

                                neighbor_rows = []
                                for neighbor_word, similarity in neighbors:
                                    category = classify_neighbor(word, neighbor_word, similarity)
                                    neighbor_rows.append({
                                        "Сосед": neighbor_word,
                                        "Сходство": float(similarity),
                                        "Категория": category
                                    })
                                neighbors_df = pd.DataFrame(neighbor_rows)

                                if not neighbors_df.empty:
                                    neighbors_df.sort_values("Сходство", ascending=True, inplace=True)
                                    neighbor_order = neighbors_df["Сосед"].tolist()
                                    chart_height_neighbors = max(280, 34 * len(neighbors_df))

                                    neighbors_chart = alt.Chart(neighbors_df).mark_bar().encode(
                                        y=alt.Y('Сосед:N', sort=neighbor_order, title=''),
                                        x=alt.X('Сходство:Q', title='Косинусное сходство', scale=alt.Scale(domain=[0, 1])),
                                        color=alt.Color('Категория:N', scale=color_scale_neighbor),
                                        tooltip=[
                                            alt.Tooltip('Сосед:N', title='Сосед'),
                                            alt.Tooltip('Категория:N', title='Категория'),
                                            alt.Tooltip('Сходство:Q', format='.3f', title='Сходство')
                                        ]
                                    ).properties(height=chart_height_neighbors, width='container')

                                    st.altair_chart(neighbors_chart, use_container_width=True)
                                else:
                                    st.info("Соседей не найдено.")

                                if neighbor_types_counts:
                                    type_rows = [
                                        {"Категория": label.capitalize(), "Количество": count}
                                        for label, count in neighbor_types_counts.items()
                                    ]
                                    type_df = pd.DataFrame(type_rows)
                                    st.dataframe(type_df, use_container_width=True)

                                    semantic_mix = neighbor_types_counts.get('semantic_synonyms', 0)
                                    syntactic_mix = neighbor_types_counts.get('syntactic_related', 0)
                                    morph_mix = neighbor_types_counts.get('morphological_variants', 0)
                                    thematic_mix = neighbor_types_counts.get('thematic_related', 0)
                                    total_neighbors = max(len(neighbors), 1)

                                    notes = []
                                    if semantic_mix < total_neighbors * 0.4 and (syntactic_mix or morph_mix):
                                        notes.append("Есть смешение семантических и синтаксических соседей")
                                    if thematic_mix > 0 and semantic_mix < total_neighbors * 0.5:
                                        notes.append("Преобладают тематические ассоциации")

                                    if notes:
                                        st.warning("; ".join(notes))
                                else:
                                    st.info("Категории соседей определить не удалось.")

                            st.markdown("</div>", unsafe_allow_html=True)

        with tab_report:
            st.subheader("6.5. Динамический отчёт")

            selected_model_name = st.session_state.get('semantic_model_select')
            if not selected_model_name:
                st.warning("Выберите модель на вкладке 6.1, чтобы сформировать отчёт.")
            else:
                model_ref = available_models.get(selected_model_name)
                if model_ref is None:
                    st.error("Не удалось получить обученную модель. Повторите обучение на этапе 5.")
                else:
                    st.markdown(
                        """
                        Отчёт объединяет ключевые результаты предыдущих подпунктов: 
                        векторную арифметику, аналоги, тепловые карты близостей и 
                        визуализацию семантических проекций.
                        """
                    )

                    # ---------------- Vector arithmetic summary ----------------
                    manual_result = st.session_state.get('semantic_manual_analogy_result')
                    with st.container():
                        st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                        st.markdown("### 🔢 Векторная арифметика")

                        if manual_result and not manual_result.get('error'):
                            words_tuple = manual_result.get('words', ("", "", ""))
                            expression = f"{words_tuple[0]} − {words_tuple[1]} + {words_tuple[2]}"
                            st.markdown(f"**Выражение:** `{expression}`")

                            results_list = manual_result.get('results', [])
                            if results_list:
                                arithmetic_df = pd.DataFrame(results_list, columns=["Слово", "Сходство"])
                                arithmetic_df.sort_values("Сходство", ascending=True, inplace=True)
                                chart_height_arith = max(180, 26 * len(arithmetic_df))
                                arith_chart = alt.Chart(arithmetic_df).mark_bar(color='#4dabf7').encode(
                                    y=alt.Y('Слово:N', sort=None, title=''),
                                    x=alt.X('Сходство:Q', title='Косинусное сходство', scale=alt.Scale(domain=[0, 1])),
                                    tooltip=[
                                        alt.Tooltip('Слово:N', title='Слово'),
                                        alt.Tooltip('Сходство:Q', format='.3f', title='Сходство')
                                    ]
                                ).properties(height=chart_height_arith, width='container')
                                st.altair_chart(arith_chart, use_container_width=True)
                            else:
                                st.info("Результатов не найдено.")
                        elif manual_result and manual_result.get('error'):
                            st.warning(manual_result.get('error'))
                        else:
                            st.info("Нет результатов векторной арифметики — выполните расчёт на вкладке 6.2.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # ---------------- Analogy statistics ----------------
                    category_result = st.session_state.get('semantic_category_analogy_result')
                    with st.container():
                        st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                        st.markdown("### 📊 Статистика по аналогиям")

                        if category_result:
                            label_map = {
                                "semantic_capitals": "Столицы стран",
                                "semantic_gender": "Родовые пары",
                                "syntactic_comparative": "Степени сравнения",
                                "morphological_verbs": "Глаголы прошедшего времени"
                            }
                            summary_rows = []
                            for key, value in category_result.items():
                                if key in ("overall_accuracy", "total_tests", "total_correct"):
                                    continue
                                total = value.get('total', 0)
                                accuracy = value.get('accuracy', 0.0)
                                summary_rows.append({
                                    "Категория": label_map.get(key, key),
                                    "Точность": accuracy * 100,
                                    "Тестов": total
                                })

                            if summary_rows:
                                analogy_df = pd.DataFrame(summary_rows)
                                analogy_chart = alt.Chart(analogy_df).mark_bar().encode(
                                    x=alt.X('Точность:Q', title='Точность (%)', scale=alt.Scale(domain=[0, 100])),
                                    y=alt.Y('Категория:N', sort='-x', title=''),
                                    color=alt.Color('Категория:N', legend=None),
                                    tooltip=[
                                        alt.Tooltip('Категория:N', title='Категория'),
                                        alt.Tooltip('Точность:Q', format='.1f', title='Точность (%)'),
                                        alt.Tooltip('Тестов:Q', title='Кол-во тестов')
                                    ]
                                ).properties(height=200, width='container')
                                st.altair_chart(analogy_chart, use_container_width=True)
                            else:
                                st.info("Нет данных по аналогиям — выполните сравнительный анализ на вкладке 6.2.")
                        else:
                            st.info("Запустите оценку аналогий на вкладке 6.2, чтобы увидеть статистику.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # ---------------- Heatmap of semantic similarities ----------------
                    st.session_state.setdefault('semantic_distance_cache', {})
                    distance_report = st.session_state.semantic_distance_cache.get(selected_model_name)
                    if distance_report is None:
                        with st.spinner("Готовим матрицу сходств..."):
                            distance_report = semantic_ops.analyze_distance_distribution(selected_model_name)
                        st.session_state.semantic_distance_cache[selected_model_name] = distance_report

                    with st.container():
                        st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                        st.markdown("### 🔥 Тепловая карта сходств")

                        heatmap_words_count = st.slider(
                            "Сколько слов отображать в тепловой карте",
                            min_value=5,
                            max_value=40,
                            value=20,
                            key="semantic_heatmap_wordcount"
                        )

                        if distance_report and distance_report.get('similarity_matrix'):
                            sample_words = distance_report.get('sample_words', [])
                            similarity_matrix = np.array(distance_report.get('similarity_matrix'))

                            if len(sample_words) >= heatmap_words_count:
                                selected_indices = np.arange(heatmap_words_count)
                                selected_words = [sample_words[i] for i in selected_indices]
                                matrix_subset = similarity_matrix[np.ix_(selected_indices, selected_indices)]

                                heatmap_df = pd.DataFrame(matrix_subset, index=selected_words, columns=selected_words)
                                heatmap_long = heatmap_df.reset_index().melt(id_vars='index', var_name='Слово2', value_name='Сходство')
                                heatmap_long.rename(columns={'index': 'Слово1'}, inplace=True)

                                heatmap_chart = alt.Chart(heatmap_long).mark_rect().encode(
                                    x=alt.X('Слово2:N', title='', sort=selected_words),
                                    y=alt.Y('Слово1:N', title='', sort=selected_words),
                                    color=alt.Color('Сходство:Q', scale=alt.Scale(scheme='blues'), title='Сходство'),
                                    tooltip=[
                                        alt.Tooltip('Слово1:N', title='Слово 1'),
                                        alt.Tooltip('Слово2:N', title='Слово 2'),
                                        alt.Tooltip('Сходство:Q', format='.3f', title='Сходство')
                                    ]
                                ).properties(width='container', height=400)
                                st.altair_chart(heatmap_chart, use_container_width=True)
                            else:
                                st.info("Недостаточно слов для построения тепловой карты — попробуйте пересчитать матрицу на вкладке 6.1.")
                        else:
                            st.info("Матрицу сходств не удалось сформировать.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # ---------------- 2D/3D projections ----------------
                    with st.container():
                        st.markdown('<div class="semantic-card">', unsafe_allow_html=True)
                        st.markdown("### 🗺️ Проекция в семантическом пространстве")

                        projection_mode = st.radio(
                            "Режим проекции",
                            options=["2D", "3D"],
                            horizontal=True,
                            key="semantic_projection_mode"
                        )
                        projection_sample = st.slider(
                            "Количество слов для проекции",
                            min_value=30,
                            max_value=200,
                            value=80,
                            key="semantic_projection_sample"
                        )
                        cluster_count = st.slider(
                            "Количество кластеров (0 = без кластеризации)",
                            min_value=0,
                            max_value=10,
                            value=4,
                            key="semantic_projection_clusters"
                        )

                        all_words = list(model_ref.wv.key_to_index.keys())
                        if len(all_words) < 10:
                            st.warning("В словаре модели недостаточно слов для проекции.")
                        else:
                            rng = np.random.default_rng(42)
                            sample_size = min(projection_sample, len(all_words))
                            sampled_words = rng.choice(all_words, size=sample_size, replace=False)
                            vectors = model_ref.wv[sampled_words]

                            if projection_mode == "2D":
                                reducer = PCA(n_components=2)
                                coords = reducer.fit_transform(vectors)
                                coord_df = pd.DataFrame(coords, columns=['x', 'y'])
                            else:
                                reducer = PCA(n_components=3)
                                coords = reducer.fit_transform(vectors)
                                coord_df = pd.DataFrame(coords, columns=['x', 'y', 'z'])

                            coord_df['Слово'] = sampled_words

                            if cluster_count and cluster_count > 1:
                                kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
                                clusters = kmeans.fit_predict(coords)
                                coord_df['Кластер'] = clusters.astype(str)
                            else:
                                coord_df['Кластер'] = 'Все'

                            if projection_mode == "2D":
                                proj_chart = alt.Chart(coord_df).mark_circle(size=80).encode(
                                    x=alt.X('x:Q', title='Компонента 1'),
                                    y=alt.Y('y:Q', title='Компонента 2'),
                                    color=alt.Color('Кластер:N', legend=alt.Legend(title='Кластеры')),
                                    tooltip=[
                                        alt.Tooltip('Слово:N', title='Слово'),
                                        alt.Tooltip('Кластер:N', title='Кластер'),
                                        alt.Tooltip('x:Q', format='.3f', title='Компонента 1'),
                                        alt.Tooltip('y:Q', format='.3f', title='Компонента 2')
                                    ]
                                ).properties(height=500, width='container')
                                st.altair_chart(proj_chart, use_container_width=True)
                            else:
                                fig_3d = px.scatter_3d(
                                    coord_df,
                                    x='x', y='y', z='z',
                                    color='Кластер',
                                    hover_name='Слово',
                                    title="3D проекция семантического пространства"
                                )
                                fig_3d.update_traces(marker=dict(size=5))
                                st.plotly_chart(fig_3d, use_container_width=True)

                        st.markdown("</div>", unsafe_allow_html=True)

