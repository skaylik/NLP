import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, regexp_tokenize
from nltk.stem import SnowballStemmer
import spacy
import razdel
import pymorphy3
import re
import string

# Загрузка ресурсов NLTK
nltk.download('punkt_tab')

class TextProcessor:
    def __init__(self):
        self._nlp_spacy = None
        self._morph = None
        self._snowball_stemmer = None
    
    @property
    def nlp_spacy(self):
        if self._nlp_spacy is None:
            try:
                self._nlp_spacy = spacy.load("ru_core_news_sm")
            except:
                # Убрано предупреждение для пользователя
                self._nlp_spacy = None
        return self._nlp_spacy
    
    @property
    def morph(self):
        if self._morph is None:
            self._morph = pymorphy3.MorphAnalyzer()
        return self._morph
    
    @property
    def snowball_stemmer(self):
        if self._snowball_stemmer is None:
            self._snowball_stemmer = SnowballStemmer("russian")
        return self._snowball_stemmer
    
    def tokenize_naive(self, text):
        tokens = text.split()
        return self._filter_tokens(tokens)
    
    def tokenize_regex(self, text):
        # Только слова и числа, без знаков препинания
        tokens = regexp_tokenize(text, r'\b\w+\b|\d+')
        return self._filter_tokens(tokens)
    
    def tokenize_nltk(self, text):
        tokens = word_tokenize(text, language='russian')
        # Фильтруем знаки препинания
        tokens = [token for token in tokens if token not in string.punctuation and not all(c in string.punctuation for c in token)]
        return self._filter_tokens(tokens)
    
    def tokenize_spacy(self, text):
        if self.nlp_spacy:
            doc = self.nlp_spacy(text)
            # Берем только токены, которые не являются знаками препинания
            tokens = [token.text for token in doc if not token.is_punct]
            return self._filter_tokens(tokens)
        return []
    
    def tokenize_razdel(self, text):
        tokens = [token.text for token in razdel.tokenize(text)]
        # Фильтруем знаки препинания
        tokens = [token for token in tokens if token not in string.punctuation and not all(c in string.punctuation for c in token)]
        return self._filter_tokens(tokens)
    
    def _filter_tokens(self, tokens):
        """Фильтрация токенов - убирает пустые, знаки препинания и слишком короткие"""
        filtered = []
        for token in tokens:
            clean_token = token.strip()
            if clean_token and len(clean_token) > 0:
                # Убираем знаки препинания и одиночные символы (кроме букв и цифр)
                if (clean_token not in string.punctuation and 
                    not all(c in string.punctuation for c in clean_token) and
                    (clean_token.isalnum() or len(clean_token) > 1)):
                    filtered.append(clean_token)
        return filtered
    
    def _should_normalize(self, token):
        """Определяет, нужно ли нормализовать токен"""
        # Не нормализуем: числа, короткие предлоги/союзы
        if (token.isdigit() or
            token in ['a', 'и', 'в', 'с', 'у', 'о', 'к', 'на', 'по', 'за', 'из']):
            return False
        return True
    
    def stem_snowball(self, tokens):
        """Улучшенный стемминг с проверкой"""
        stemmed = []
        for token in tokens:
            if not self._should_normalize(token):
                stemmed.append(token)
            else:
                try:
                    stemmed_token = self.snowball_stemmer.stem(token)
                    # Проверяем, что стемминг не испортил слово
                    if len(stemmed_token) >= 2:
                        stemmed.append(stemmed_token)
                    else:
                        stemmed.append(token)
                except:
                    stemmed.append(token)
        return stemmed
    
    def lemmatize_pymorphy(self, tokens):
        """Улучшенная лемматизация с контекстной обработкой"""
        lemmatized = []
        for token in tokens:
            if not self._should_normalize(token):
                lemmatized.append(token)
            else:
                try:
                    parsed = self.morph.parse(token)
                    if parsed:
                        # Выбираем наиболее вероятную лемму с учетом контекста
                        best_parse = parsed[0]
                        lemma = best_parse.normal_form
                        
                        # Проверяем, что лемма не короче 2 символов
                        if len(lemma) >= 2:
                            lemmatized.append(lemma)
                        else:
                            lemmatized.append(token)
                    else:
                        lemmatized.append(token)
                except:
                    lemmatized.append(token)
        return lemmatized
    
    def lemmatize_spacy(self, tokens):
        if self.nlp_spacy:
            text = ' '.join(tokens)
            doc = self.nlp_spacy(text)
            lemmatized = []
            for token in doc:
                if not self._should_normalize(token.text):
                    lemmatized.append(token.text)
                else:
                    lemma = token.lemma_
                    if len(lemma) >= 2:
                        lemmatized.append(lemma)
                    else:
                        lemmatized.append(token.text)
            return lemmatized
        return tokens

def validate_texts(texts):
    """Валидация загруженных текстов"""
    if not texts:
        raise ValueError("Нет текстов для обработки")
    
    valid_texts = [text for text in texts if text and isinstance(text, str) and len(text.strip()) > 0]
    
    if len(valid_texts) == 0:
        raise ValueError("Все тексты пустые или некорректного формата")
    
    return valid_texts

def load_texts_from_jsonl(uploaded_file):
    """Загрузка текстов из JSONL файла"""
    texts = []
    success_count = 0
    error_count = 0
    
    uploaded_file.seek(0)
    
    for i, line in enumerate(uploaded_file):
        try:
            line_str = line.decode('utf-8').strip()
            
            if not line_str:
                continue
                
            article = json.loads(line_str)
            
            text_content = article.get('text') or article.get('content') or article.get('body') or article.get('title')
            if text_content and isinstance(text_content, str) and text_content.strip():
                texts.append(text_content.strip())
                success_count += 1
            else:
                error_count += 1
                
        except (json.JSONDecodeError, KeyError, AttributeError, UnicodeDecodeError) as e:
            error_count += 1
            continue
    
    # Убрано предупреждение об ошибках
    return texts

def create_token_length_distribution(tokens_list):
    """Создание графика распределения длин токенов"""
    token_lengths = [len(token) for tokens in tokens_list for token in tokens if token.strip()]
    
    if not token_lengths:
        return create_empty_plot("Нет данных для отображения")
    
    fig = px.histogram(
        x=token_lengths,
        nbins=30,
        title='Распределение длин токенов',
        labels={'x': 'Длина токена', 'y': 'Количество'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(showlegend=False)
    return fig

def create_token_frequency_chart(tokens_list, top_n=20):
    """Создание графика частотности токенов"""
    all_tokens = [token.lower() for tokens in tokens_list for token in tokens if token.strip()]
    
    if not all_tokens:
        return create_empty_plot("Нет данных для отображения")
    
    token_counter = Counter(all_tokens)
    top_tokens = token_counter.most_common(top_n)
    
    tokens, freqs = zip(*top_tokens)
    
    fig = px.bar(
        x=tokens,
        y=freqs,
        title=f'Топ-{top_n} самых частых токенов',
        labels={'x': 'Токен', 'y': 'Частота'},
        color=freqs,
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        xaxis_tickangle=-45, 
        showlegend=False,
        height=400
    )
    return fig

def create_empty_plot(message):
    """Создание пустого графика с сообщением"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400
    )
    return fig

def calculate_oov_rate(processed_texts, reference_vocab):
    """Расчет доли OOV слов"""
    all_tokens = [token for tokens in processed_texts for token in tokens if token.strip()]
    
    if not reference_vocab or not all_tokens:
        return 0
    
    oov_count = sum(1 for token in all_tokens if token not in reference_vocab)
    return (oov_count / len(all_tokens)) * 100

def create_oov_gauge(oov_rate):
    """Создание индикатора OOV"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = oov_rate,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Доля OOV (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 20], 'color': "yellow"},
                {'range': [20, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(
        margin=dict(t=50, b=10, l=10, r=10)
    )
    return fig

def generate_report(processed_texts, method_name, processing_time):
    """Генерация отчета с метриками"""
    all_tokens = [token for tokens in processed_texts for token in tokens if token.strip()]
    
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens)) if all_tokens else 0
    avg_token_length = sum(len(token) for token in all_tokens) / len(all_tokens) if all_tokens else 0
    
    metrics_df = pd.DataFrame({
        'Метрика': [
            'Метод обработки',
            'Общее количество токенов', 
            'Количество уникальных токенов',
            'Средняя длина токена',
            'Время обработки (сек)'
        ],
        'Значение': [
            method_name,
            f"{total_tokens:,}",
            f"{unique_tokens:,}",
            f"{avg_token_length:.2f}",
            f"{processing_time:.2f}"
        ]
    })
    
    return metrics_df

def preprocess_text(text):
    """Предобработка текста для улучшения токенизации"""
    # Заменяем множественные пробелы на одинарные
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_with_progress(processor, texts, tokenize_func, normalize_func):
    """Обработка с улучшенным прогресс-баром"""
    processed_texts = []
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, text in enumerate(texts):
        progress_text.text(f"Обработка текста {i+1}/{len(texts)}")
        
        try:
            # Предобработка текста
            cleaned_text = preprocess_text(text)
            tokens = tokenize_func(cleaned_text)
            
            # Дополнительная фильтрация
            tokens = [token for token in tokens if token.strip()]
            
            # Убираем последовательные дубликаты одиночных символов
            filtered_tokens = []
            for j, token in enumerate(tokens):
                if (j == 0 or 
                    token != tokens[j-1] or 
                    len(token) > 1 or 
                    token.isalnum()):
                    filtered_tokens.append(token)
            
            # Применяем нормализацию
            normalized_tokens = normalize_func(filtered_tokens)
            
            # Финальная фильтрация
            final_tokens = [token for token in normalized_tokens if token.strip()]
            
            processed_texts.append(final_tokens)
        except Exception:
            # Убрано сообщение об ошибке
            processed_texts.append([])
        
        progress_bar.progress((i + 1) / len(texts))
    
    progress_text.empty()
    progress_bar.empty()
    
    return processed_texts

def main():
    st.set_page_config(
        page_title="NLP Text Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 NLP Анализатор текста")
    st.markdown("Интерактивный инструмент для анализа и обработки текстовых данных")
    st.markdown("---")
    
    if 'processor' not in st.session_state:
        st.session_state.processor = TextProcessor()
    
    if 'loaded_texts' not in st.session_state:
        st.session_state.loaded_texts = []
    
    # Боковая панель для настроек
    with st.sidebar:
        st.header("⚙️ Настройки обработки")
        
        st.subheader("📁 Загрузка данных")
        uploaded_file = st.file_uploader(
            "Выберите JSONL файл", 
            type=['jsonl'],
            help="Файл должен содержать тексты. Каждая строка - отдельный JSON объект."
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Загрузить тексты", use_container_width=True):
                with st.spinner("Загружаю файл..."):
                    texts = load_texts_from_jsonl(uploaded_file)
                    if texts:
                        st.session_state.loaded_texts = texts
                        st.success(f"✅ Загружено {len(texts)} текстов")
                    else:
                        st.error("❌ Не удалось загрузить тексты из файла")
        
        st.subheader("🔤 Метод токенизации")
        tokenization_method = st.selectbox(
            "Как разделить текст на токены:",
            [
                "Наивная (по пробелам)",
                "Регулярные выражения (только слова)", 
                "NLTK (интеллектуальная)",
                "spaCy (с учетом контекста)",
                "razdel (специализирован для русского)"
            ],
            index=4
        )
        
        st.subheader("🔄 Метод нормализации")
        normalization_method = st.selectbox(
            "Как нормализовать слова:",
            [
                "Без нормализации (исходные слова)",
                "Snowball стемминг (основа слова)",
                "pymorphy3 лемматизация (словарная форма)", 
                "spaCy лемматизация (контекстная)"
            ],
            index=2
        )
        
        st.subheader("📈 Настройки анализа")
        top_n_tokens = st.slider(
            "Количество токенов для частотного анализа:",
            min_value=10,
            max_value=50,
            value=20
        )
        
        sample_size = st.slider(
            "Количество текстов для обработки:",
            min_value=1,
            max_value=500,
            value=50
        )
        
        process_button = st.button("🚀 Запустить обработку", type="primary", use_container_width=True)
    
    # Основная область
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Данные для обработки")
        
        texts_to_process = []
        if st.session_state.loaded_texts:
            texts_to_process = st.session_state.loaded_texts
            
            if sample_size > 0 and len(texts_to_process) > sample_size:
                texts_to_process = texts_to_process[:sample_size]
            
            st.subheader("📝 Примеры текстов:")
            for i, text in enumerate(texts_to_process[:3]):
                with st.expander(f"Текст {i+1} ({len(text)} символов)"):
                    st.text(text[:500] + "..." if len(text) > 500 else text)
        else:
            st.info("👆 Загрузите JSONL файл и нажмите кнопку 'Загрузить тексты' для начала работы")
    
    with col2:
        st.header("📊 Статистика данных")
        if texts_to_process:
            total_chars = sum(len(text) for text in texts_to_process)
            total_words = sum(len(text.split()) for text in texts_to_process)
            avg_words = total_words / len(texts_to_process) if texts_to_process else 0
            avg_chars = total_chars / len(texts_to_process) if texts_to_process else 0
            
            st.metric("Количество текстов", len(texts_to_process))
            st.metric("Общее количество символов", f"{total_chars:,}")
            st.metric("Общее количество слов", f"{total_words:,}")
            st.metric("Средняя длина текста (слов)", f"{avg_words:.1f}")
            st.metric("Средняя длина текста (символов)", f"{avg_chars:.1f}")
        else:
            st.info("Данные не загружены")
    
    # Обработка и визуализация
    if process_button and texts_to_process:
        try:
            validated_texts = validate_texts(texts_to_process)
        except ValueError as e:
            # Убрано сообщение об ошибке
            return
            
        st.markdown("---")
        st.header("📈 Результаты обработки")
        
        with st.spinner("🔄 Обрабатываю тексты..."):
            token_funcs = {
                "Наивная (по пробелам)": st.session_state.processor.tokenize_naive,
                "Регулярные выражения (только слова)": st.session_state.processor.tokenize_regex,
                "NLTK (интеллектуальная)": st.session_state.processor.tokenize_nltk,
                "spaCy (с учетом контекста)": st.session_state.processor.tokenize_spacy,
                "razdel (специализирован для русского)": st.session_state.processor.tokenize_razdel
            }
            
            norm_funcs = {
                "Без нормализации (исходные слова)": lambda x: x,
                "Snowball стемминг (основа слова)": st.session_state.processor.stem_snowball,
                "pymorphy3 лемматизация (словарная форма)": st.session_state.processor.lemmatize_pymorphy,
                "spaCy лемматизация (контекстная)": st.session_state.processor.lemmatize_spacy
            }
            
            tokenize_func = token_funcs[tokenization_method]
            normalize_func = norm_funcs[normalization_method]
            
            start_time = time.time()
            processed_texts = process_with_progress(
                st.session_state.processor, 
                validated_texts, 
                tokenize_func, 
                normalize_func
            )
            processing_time = time.time() - start_time
            
            # Создание референсного словаря для OOV
            split_index = int(len(processed_texts) * 0.8)
            reference_vocab = set()
            for tokens in processed_texts[:split_index]:
                reference_vocab.update(tokens)
            
            test_texts = processed_texts[split_index:] if split_index < len(processed_texts) else processed_texts
            oov_rate = calculate_oov_rate(test_texts, reference_vocab)
            
            method_name = f"{tokenization_method} + {normalization_method}"
            report_df = generate_report(processed_texts, method_name, processing_time)
        
        st.success(f"✅ Обработка завершена за {processing_time:.2f} секунд")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📋 Отчет обработки")
            st.dataframe(report_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("⚠️ Анализ OOV")
            st.plotly_chart(create_oov_gauge(oov_rate), use_container_width=True)
            st.caption("OOV показывает процент токенов, которых нет в обучающем словаре")
        
        with col3:
            st.subheader("🔍 Пример обработки")
            if processed_texts and processed_texts[0]:
                sample_tokens = processed_texts[0][:15]
                st.write("Первые 15 токенов из первого текста:")
                for i, token in enumerate(sample_tokens, 1):
                    st.write(f"{i}. {token}")
        
        # Визуализации
        st.markdown("---")
        st.header("📊 Визуализации")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📏 Распределение длин токенов")
            fig_length = create_token_length_distribution(processed_texts)
            st.plotly_chart(fig_length, use_container_width=True)
        
        with col2:
            st.subheader("📊 Частотность токенов")
            fig_freq = create_token_frequency_chart(processed_texts, top_n_tokens)
            st.plotly_chart(fig_freq, use_container_width=True)
        
        # Детальная статистика
        st.subheader("📈 Детальная статистика токенов")
        
        all_tokens = [token for tokens in processed_texts for token in tokens if token.strip()]
        if all_tokens:
            token_counter = Counter(all_tokens)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Всего токенов", f"{len(all_tokens):,}")
            with col2:
                st.metric("Уникальных токенов", f"{len(token_counter):,}")
            with col3:
                unique_once = sum(1 for count in token_counter.values() if count == 1)
                st.metric("Токены с частотой 1", f"{unique_once:,}")
            with col4:
                if token_counter:
                    most_common_token, most_common_freq = token_counter.most_common(1)[0]
                    st.metric("Самый частый токен", f"'{most_common_token}': {most_common_freq:,}")
            
            # Таблица с топ токенами
            st.subheader("🏆 Топ токены")
            top_tokens_df = pd.DataFrame(
                token_counter.most_common(20),
                columns=['Токен', 'Частота']
            )
            st.dataframe(top_tokens_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()