"""
Модуль для предобработки текстовых данных и извлечения признаков.
Этап 2: Подготовка данных для классификации с учетом train/validation/test разделения
"""

import re
import string
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
from collections import Counter
import warnings
import pickle
import json
import os
warnings.filterwarnings('ignore')

# Для NLP обработки
try:
    import spacy
    from spacy.lang.ru.stop_words import STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️ spaCy не установлен. Установите: pip install spacy")

try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK не установлен. Установите: pip install nltk")

# Для векторизации
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn не установлен. Установите: pip install scikit-learn")

# Для эмбеддингов
try:
    import gensim
    from gensim.models import Word2Vec, FastText
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠️ Gensim не установлен. Установите: pip install gensim")

# Для трансформеров
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers не установлен. Установите: pip install transformers")


class TextPreprocessor:
    """Класс для текстовой предобработки"""
    
    def __init__(self, language: str = 'russian', 
                 remove_stopwords: bool = True,
                 use_spacy: bool = True):
        """
        Инициализация препроцессора
        
        Args:
            language: язык текста ('russian' или 'english')
            remove_stopwords: удалять ли стоп-слова
            use_spacy: использовать spaCy (True) или NLTK (False)
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.use_spacy = use_spacy
        
        # Инициализация NLP моделей
        self.nlp = None
        self.stop_words = set()
        
        if use_spacy and SPACY_AVAILABLE:
            try:
                # Загружаем модель spaCy для русского языка
                self.nlp = spacy.load("ru_core_news_sm" if language == 'russian' else "en_core_web_sm")
                self.stop_words = STOP_WORDS
                print(f"✅ Загружена spaCy модель для языка: {language}")
            except OSError:
                print(f"⚠️ spaCy модель не найдена. Использую простую обработку.")
                self.nlp = None
        elif NLTK_AVAILABLE:
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                if language == 'russian':
                    # Для русского языка в NLTK
                    self.stop_words = set(stopwords.words('russian'))
                else:
                    self.stop_words = set(stopwords.words('english'))
                print(f"✅ Используем NLTK для языка: {language}")
            except:
                print("⚠️ NLTK не может загрузить стоп-слова")
        
        # Паттерны для очистки
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'[\+]?[78]\s?[\(]?\d{3}[\)]?\s?\d{3}[\-]?\d{2}[\-]?\d{2}')
        
        # Словарь для эмодзи
        self.emoji_dict = {
            '😀': 'смайлик_радость', '😂': 'смех', '😊': 'улыбка', '😍': 'любовь',
            '😭': 'плач', '😡': 'злость', '😱': 'ужас', '👍': 'лайк',
            '👎': 'дизлайк', '❤️': 'сердце', '🙏': 'спасибо', '😔': 'грусть',
            '🤔': 'задумчивость', '😎': 'круто', '🤗': 'объятия', '😴': 'сон',
            '🤮': 'тошнота', '🤯': 'взрыв_мозга', '🥰': 'влюбленность',
            '😤': 'разочарование', '😨': 'страх', '😩': 'усталость'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Очистка текста от HTML, URL, специальных символов
        
        Args:
            text: исходный текст
            
        Returns:
            очищенный текст
        """
        if not isinstance(text, str):
            return ""
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление URL
        text = self.url_pattern.sub('', text)
        
        # Удаление HTML тегов
        text = self.html_pattern.sub('', text)
        
        # Удаление email
        text = self.email_pattern.sub('', text)
        
        # Удаление телефонов
        text = self.phone_pattern.sub('', text)
        
        # Замена эмодзи на текст
        for emoji, desc in self.emoji_dict.items():
            text = text.replace(emoji, f' {desc} ')
        
        # Удаление специальных символов, но сохранение кириллицы и пунктуации
        text = re.sub(r'[^а-яёa-z0-9\s.,!?;:()\-"\']', ' ', text)
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_with_spacy(self, text: str) -> List[str]:
        """
        Токенизация и лемматизация с использованием spaCy
        
        Args:
            text: очищенный текст
            
        Returns:
            список лемматизированных токенов
        """
        if not self.nlp:
            return self.tokenize_simple(text)
        
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            # Пропускаем пунктуацию и пробелы
            if token.is_punct or token.is_space:
                continue
            
            # Удаляем стоп-слова если нужно
            if self.remove_stopwords and token.text in self.stop_words:
                continue
            
            # Используем лемму или текст
            lemma = token.lemma_ if token.lemma_ != '-PRON-' else token.text
            
            # Добавляем если лемма не пустая
            if lemma.strip():
                tokens.append(lemma)
        
        return tokens
    
    def tokenize_simple(self, text: str) -> List[str]:
        """
        Простая токенизация без spaCy
        
        Args:
            text: очищенный текст
            
        Returns:
            список токенов
        """
        # Простая токенизация по пробелам и знакам препинания
        tokens = re.findall(r'\b[а-яёa-z]+\b', text)
        
        # Удаление стоп-слов если нужно
        if self.remove_stopwords and self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def preprocess(self, text: str, return_string: bool = False) -> Union[str, List[str]]:
        """
        Полный пайплайн предобработки
        
        Args:
            text: исходный текст
            return_string: вернуть строку (True) или список токенов (False)
            
        Returns:
            обработанный текст
        """
        # Очистка
        cleaned_text = self.clean_text(text)
        
        # Токенизация
        if self.use_spacy and self.nlp:
            tokens = self.tokenize_with_spacy(cleaned_text)
        else:
            tokens = self.tokenize_simple(cleaned_text)
        
        if return_string:
            return ' '.join(tokens)
        else:
            return tokens
    
    def preprocess_batch(self, texts: List[str], return_string: bool = False) -> List[Union[str, List[str]]]:
        """
        Предобработка списка текстов
        
        Args:
            texts: список текстов
            return_string: вернуть строку (True) или список токенов (False)
            
        Returns:
            список обработанных текстов
        """
        return [self.preprocess(text, return_string) for text in texts]


class FeatureExtractor:
    """Класс для извлечения мета-признаков из текста"""
    
    def __init__(self):
        """Инициализация экстрактора признаков"""
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
    
    def extract_statistical_features(self, text: str) -> Dict:
        """
        Извлечение статистических признаков
        
        Args:
            text: текст
            
        Returns:
            словарь статистических признаков
        """
        if not text:
            return {}
        
        # Разделяем на слова
        words = text.split()
        chars = text.replace(' ', '')
        
        # Подсчеты
        num_words = len(words)
        num_chars = len(chars)
        num_sentences = len(re.split(r'[.!?]+', text))
        
        # Вычисление средних
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
        
        # Уникальность
        unique_words = len(set(words))
        lexical_diversity = unique_words / num_words if num_words > 0 else 0
        
        # Длина в символах
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        
        # Подсчет цифр
        digit_count = sum(c.isdigit() for c in text)
        digit_ratio = digit_count / len(text) if len(text) > 0 else 0
        
        # Подсчет заглавных букв
        uppercase_count = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_count / len(text) if len(text) > 0 else 0
        
        # Подсчет знаков препинации
        punctuation_chars = set(string.punctuation + '«»—–')
        punctuation_count = sum(1 for c in text if c in punctuation_chars)
        punctuation_ratio = punctuation_count / len(text) if len(text) > 0 else 0
        
        # Сложность текста (адаптированная формула)
        syllables_count = self._count_syllables_russian(text)
        flesch_kincaid = 206.835 - 1.3 * (num_words / num_sentences) - 60.1 * (syllables_count / num_words) if num_words > 0 and num_sentences > 0 else 0
        
        features = {
            'word_count': num_words,
            'char_count': char_count,
            'char_count_no_spaces': char_count_no_spaces,
            'sentence_count': num_sentences,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'unique_word_count': unique_words,
            'lexical_diversity': lexical_diversity,
            'digit_count': digit_count,
            'digit_ratio': digit_ratio,
            'uppercase_count': uppercase_count,
            'uppercase_ratio': uppercase_ratio,
            'punctuation_count': punctuation_count,
            'punctuation_ratio': punctuation_ratio,
            'syllable_count': syllables_count,
            'flesch_kincaid_score': flesch_kincaid,
            'is_short_text': 1 if num_words < 10 else 0,
            'is_long_text': 1 if num_words > 100 else 0
        }
        
        return features
    
    def _count_syllables_russian(self, text: str) -> int:
        """
        Подсчет слогов в русском тексте
        
        Args:
            text: текст
            
        Returns:
            количество слогов
        """
        vowels = 'аеёиоуыэюя'
        text = text.lower()
        
        syllables = 0
        for char in text:
            if char in vowels:
                syllables += 1
        
        return syllables if syllables > 0 else 1
    
    def extract_batch_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Извлечение признаков для списка текстов
        
        Args:
            texts: список текстов
            
        Returns:
            DataFrame с признаками
        """
        features_list = []
        for text in texts:
            features = self.extract_statistical_features(text)
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def normalize_features(self, features_df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Нормализация мета-признаков
        
        Args:
            features_df: DataFrame с признаками
            fit: обучить скейлер (True) или использовать обученный (False)
            
        Returns:
            нормализованные признаки
        """
        if self.scaler is None:
            return features_df.values
        
        if fit:
            return self.scaler.fit_transform(features_df.values)
        else:
            return self.scaler.transform(features_df.values)


class SplitAwareVectorizer:
    """
    Векторизатор, который учитывает разделение на train/validation/test
    Обучается только на train данных!
    """
    
    def __init__(self, method: str = 'tfidf', **kwargs):
        """
        Инициализация векторизатора
        
        Args:
            method: метод векторизации ('bow', 'tfidf', 'word2vec', 'fasttext', 'bert')
            **kwargs: параметры для конкретного метода
        """
        self.method = method
        self.vectorizer = None
        self.embedding_model = None
        self.tokenizer = None
        self.model = None
        self.is_fitted = False
        self.vector_size = kwargs.get('vector_size', 100)  # Размерность векторов по умолчанию
        
        # Для статистических методов
        if method in ['bow', 'tfidf']:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn не установлен")
            
            if method == 'bow':
                self.vectorizer = CountVectorizer(
                    max_features=kwargs.get('max_features', 5000),
                    ngram_range=kwargs.get('ngram_range', (1, 2)),
                    min_df=kwargs.get('min_df', 2),
                    max_df=kwargs.get('max_df', 0.95)
                )
            else:  # tfidf
                self.vectorizer = TfidfVectorizer(
                    max_features=kwargs.get('max_features', 5000),
                    ngram_range=kwargs.get('ngram_range', (1, 2)),
                    min_df=kwargs.get('min_df', 2),
                    max_df=kwargs.get('max_df', 0.95)
                )
        
        # Для Word2Vec - обучение на месте, если модель не предоставлена
        elif method == 'word2vec':
            if not GENSIM_AVAILABLE:
                raise ImportError("gensim не установлен")
            
            model_path = kwargs.get('model_path')
            if model_path and os.path.exists(model_path):
                try:
                    self.embedding_model = Word2Vec.load(model_path)
                    self.vector_size = self.embedding_model.vector_size
                    self.is_fitted = True
                    print(f"✅ Загружена модель Word2Vec из {model_path}")
                except:
                    print(f"⚠️ Не удалось загрузить модель Word2Vec из {model_path}")
                    self.embedding_model = None
            else:
                # Если модель не указана, будем обучать на данных
                print("ℹ️ Модель Word2Vec не указана, будет обучена на данных")
                self.embedding_model = None
        
        # Для FastText - аналогично
        elif method == 'fasttext':
            if not GENSIM_AVAILABLE:
                raise ImportError("gensim не установлен")
            
            model_path = kwargs.get('model_path')
            if model_path and os.path.exists(model_path):
                try:
                    self.embedding_model = FastText.load(model_path)
                    self.vector_size = self.embedding_model.vector_size
                    self.is_fitted = True
                    print(f"✅ Загружена модель FastText из {model_path}")
                except:
                    print(f"⚠️ Не удалось загрузить модель FastText из {model_path}")
                    self.embedding_model = None
            else:
                print("ℹ️ Модель FastText не указана, будет обучена на данных")
                self.embedding_model = None
        
        # Для BERT
        elif method in ['bert', 'rubert']:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers не установлен")
            
            model_name = 'cointegrated/rubert-tiny' if method == 'rubert' else 'bert-base-multilingual-cased'
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
                self.vector_size = self.model.config.hidden_size
                self.is_fitted = True
                print(f"✅ Загружена модель: {model_name}")
            except Exception as e:
                print(f"⚠️ Не удалось загрузить модель {model_name}: {e}")
                self.model = None
    
    def fit(self, texts: List[str]):
        """
        Обучение векторизатора на train данных
        
        Args:
            texts: список текстов (только train!)
        """
        if self.vectorizer and not self.is_fitted:
            self.vectorizer.fit(texts)
            self.is_fitted = True
            print(f"✅ Обучен векторизатор {self.method}")
        
        # Обучение Word2Vec или FastText на данных, если модель не была предоставлена
        elif self.method == 'word2vec' and not self.is_fitted:
            print("🔄 Обучение Word2Vec на предоставленных данных...")
            try:
                # Токенизация текстов
                tokenized_texts = [text.split() for text in texts]
                
                self.embedding_model = Word2Vec(
                    sentences=tokenized_texts,
                    vector_size=self.vector_size,
                    window=5,
                    min_count=2,
                    workers=4,
                    epochs=10,
                    seed=42
                )
                self.is_fitted = True
                print(f"✅ Обучена модель Word2Vec на {len(texts)} текстах")
            except Exception as e:
                print(f"❌ Ошибка при обучении Word2Vec: {e}")
                self.embedding_model = None
        
        elif self.method == 'fasttext' and not self.is_fitted:
            print("🔄 Обучение FastText на предоставленных данных...")
            try:
                # Токенизация текстов
                tokenized_texts = [text.split() for text in texts]
                
                self.embedding_model = FastText(
                    sentences=tokenized_texts,
                    vector_size=self.vector_size,
                    window=5,
                    min_count=2,
                    workers=4,
                    epochs=10,
                    seed=42
                )
                self.is_fitted = True
                print(f"✅ Обучена модель FastText на {len(texts)} текстах")
            except Exception as e:
                print(f"❌ Ошибка при обучении FastText: {e}")
                self.embedding_model = None
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Преобразование текстов в векторы
        
        Args:
            texts: список текстов
            
        Returns:
            матрица векторов
        """
        if self.method in ['bow', 'tfidf']:
            if not self.is_fitted:
                raise ValueError("Векторизатор не обучен! Сначала вызовите fit() на train данных.")
            return self.vectorizer.transform(texts)
        
        elif self.method == 'word2vec':
            return self._get_word2vec_vectors(texts)
        
        elif self.method == 'fasttext':
            return self._get_fasttext_vectors(texts)
        
        elif self.method in ['bert', 'rubert']:
            return self._get_bert_vectors(texts)
        
        else:
            raise ValueError(f"Метод {self.method} не поддерживается")
    
    def _get_word2vec_vectors(self, texts: List[str]) -> np.ndarray:
        """Получение векторов Word2Vec"""
        if not self.embedding_model:
            # Если модель не обучена, возвращаем нулевые векторы
            print("⚠️ Модель Word2Vec не обучена, возвращаю нулевые векторы")
            return np.zeros((len(texts), self.vector_size))
        
        vectors = []
        for text in texts:
            words = text.split()
            word_vectors = []
            
            for word in words:
                try:
                    vec = self.embedding_model.wv[word]
                    word_vectors.append(vec)
                except KeyError:
                    continue
            
            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                doc_vector = np.zeros(self.vector_size)
            
            vectors.append(doc_vector)
        
        return np.array(vectors)
    
    def _get_fasttext_vectors(self, texts: List[str]) -> np.ndarray:
        """Получение векторов FastText"""
        if not self.embedding_model:
            # Если модель не обучена, возвращаем нулевые векторы
            print("⚠️ Модель FastText не обучена, возвращаю нулевые векторы")
            return np.zeros((len(texts), self.vector_size))
        
        vectors = []
        for text in texts:
            words = text.split()
            word_vectors = []
            
            for word in words:
                try:
                    vec = self.embedding_model.wv[word]
                    word_vectors.append(vec)
                except KeyError:
                    try:
                        vec = self.embedding_model.wv.get_vector(word)
                        word_vectors.append(vec)
                    except:
                        continue
            
            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                doc_vector = np.zeros(self.vector_size)
            
            vectors.append(doc_vector)
        
        return np.array(vectors)
    
    def _get_bert_vectors(self, texts: List[str]) -> np.ndarray:
        """Получение векторов BERT"""
        if not self.model or not self.tokenizer:
            # Если модель не загружена, возвращаем нулевые векторы
            print("⚠️ Модель BERT не загружена, возвращаю нулевые векторы")
            return np.zeros((len(texts), self.vector_size))
        
        vectors = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', 
                                   truncation=True, max_length=512,
                                   padding='max_length')
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
                vectors.append(cls_embedding)
        
        return np.array(vectors)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Обучение и преобразование (только для train!)
        
        Args:
            texts: список текстов
            
        Returns:
            матрица векторов
        """
        if self.vectorizer:
            self.fit(texts)
        return self.transform(texts)
    
    def save(self, filepath: str):
        """Сохранение векторизатора"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'SplitAwareVectorizer':
        """Загрузка векторизатора"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class TextDataProcessor:
    """
    Основной класс для обработки текстовых данных
    с учетом разделения на train/validation/test
    """
    
    def __init__(self, 
                 preprocessor_params: Dict = None,
                 vectorizer_params: Dict = None):
        """
        Инициализация процессора
        
        Args:
            preprocessor_params: параметры препроцессора
            vectorizer_params: параметры векторизатора
        """
        # Параметры по умолчанию
        if preprocessor_params is None:
            preprocessor_params = {
                'language': 'russian',
                'remove_stopwords': True,
                'use_spacy': True
            }
        
        if vectorizer_params is None:
            vectorizer_params = {
                'method': 'tfidf',
                'max_features': 5000
            }
        
        self.preprocessor = TextPreprocessor(**preprocessor_params)
        self.feature_extractor = FeatureExtractor()
        self.vectorizer = None
        self.vectorizer_params = vectorizer_params
        
        # Хранилище для результатов
        self.processed_data = {
            'train': {},
            'validation': {},
            'test': {}
        }
        self.vectorization_successful = True
        self.fallback_to_tfidf = False
    
    def process_splits(self, 
                      splits: Dict[str, List[Dict]],
                      extract_meta: bool = True,
                      create_vectors: bool = True,
                      text_field: str = 'text') -> Dict[str, Dict]:
        """
        Обработка всех разделов данных (train/validation/test)
        
        Args:
            splits: словарь с разбитыми данными {'train': [...], 'validation': [...], 'test': [...]}
            extract_meta: извлекать мета-признаки
            create_vectors: создавать векторные представления
            text_field: поле с текстом в данных
            
        Returns:
            словарь с результатами для каждого раздела
        """
        print("🔧 Запуск обработки всех разделов данных...")
        
        results = {}
        
        try:
            # 1. Обработка TRAIN данных (обучаем на них)
            print("\n1️⃣ Обработка TRAIN данных (обучение)...")
            train_texts = self._extract_texts(splits['train'], text_field)
            train_processed = self._process_split(
                'train', train_texts, 
                extract_meta=extract_meta, 
                create_vectors=create_vectors,
                fit_vectorizer=True
            )
            results['train'] = train_processed
            
            # 2. Обработка VALIDATION данных (используем обученный векторизатор)
            print("\n2️⃣ Обработка VALIDATION данных (преобразование)...")
            val_texts = self._extract_texts(splits['validation'], text_field)
            val_processed = self._process_split(
                'validation', val_texts,
                extract_meta=extract_meta,
                create_vectors=create_vectors,
                fit_vectorizer=False
            )
            results['validation'] = val_processed
            
            # 3. Обработка TEST данных (используем обученный векторизатор)
            print("\n3️⃣ Обработка TEST данных (преобразование)...")
            test_texts = self._extract_texts(splits['test'], text_field)
            test_processed = self._process_split(
                'test', test_texts,
                extract_meta=extract_meta,
                create_vectors=create_vectors,
                fit_vectorizer=False
            )
            results['test'] = test_processed
            
            print("\n✅ Обработка всех разделов завершена!")
            
        except Exception as e:
            print(f"❌ Ошибка при обработке данных методом {self.vectorizer_params.get('method')}: {e}")
            
            # Пробуем использовать TF-IDF как резервный вариант
            if self.vectorizer_params.get('method') != 'tfidf':
                print("🔄 Пробуем переключиться на TF-IDF как резервный метод...")
                self.fallback_to_tfidf = True
                self.vectorizer_params['method'] = 'tfidf'
                self.vectorizer = None
                
                # Повторяем попытку с TF-IDF
                return self.process_splits(splits, extract_meta, create_vectors, text_field)
            else:
                # Если уже TF-IDF и все равно ошибка
                print("❌ Критическая ошибка при обработке данных")
                raise
        
        return results
    
    def process_splits_with_fallback(self, 
                                   splits: Dict[str, List[Dict]],
                                   extract_meta: bool = True,
                                   create_vectors: bool = True,
                                   text_field: str = 'text') -> Dict[str, Dict]:
        """
        Обработка всех разделов данных с резервным вариантом
        
        Если выбранный метод векторизации не работает, 
        автоматически переключается на TF-IDF
        """
        print("🔧 Запуск обработки всех разделов данных с резервным вариантом...")
        
        try:
            # Пытаемся выполнить с выбранным методом
            return self.process_splits(splits, extract_meta, create_vectors, text_field)
        except Exception as e:
            print(f"⚠️ Ошибка при обработке методом {self.vectorizer_params.get('method')}: {e}")
            print("🔄 Переключение на метод TF-IDF...")
            
            # Меняем метод на TF-IDF
            self.vectorizer_params['method'] = 'tfidf'
            self.vectorizer = None  # Сбрасываем векторизатор
            
            # Пробуем снова
            return self.process_splits(splits, extract_meta, create_vectors, text_field)
    
    def _extract_texts(self, data: List[Dict], text_field: str) -> List[str]:
        """Извлечение текстов из данных"""
        texts = []
        for item in data:
            # Пробуем разные возможные поля с текстом
            text = (item.get(text_field) or 
                   item.get('text') or 
                   item.get('основной текст') or 
                   item.get('content') or 
                   '')
            
            # Добавляем заголовок если есть
            title = (item.get('title') or 
                    item.get('заголовок') or 
                    item.get('headline') or 
                    '')
            
            if title and text:
                combined = f"{title}. {text}"
            elif title:
                combined = title
            else:
                combined = text
            
            texts.append(combined)
        
        return texts
    
    def _process_split(self, 
                      split_name: str,
                      texts: List[str],
                      extract_meta: bool = True,
                      create_vectors: bool = True,
                      fit_vectorizer: bool = True) -> Dict:
        """
        Обработка одного раздела данных
        
        Args:
            split_name: имя раздела ('train', 'validation', 'test')
            texts: список текстов
            extract_meta: извлекать мета-признаки
            create_vectors: создавать векторные представления
            fit_vectorizer: обучать векторизатор (True) или использовать обученный (False)
            
        Returns:
            словарь с результатами
        """
        print(f"  📊 Обработка {split_name}: {len(texts)} текстов")
        
        result = {}
        
        # 1. Предобработка текстов
        print(f"    1️⃣ Предобработка...")
        processed_texts = self.preprocessor.preprocess_batch(texts, return_string=True)
        result['processed_texts'] = processed_texts
        
        # 2. Извлечение мета-признаков
        if extract_meta:
            print(f"    2️⃣ Извлечение мета-признаков...")
            meta_features = self.feature_extractor.extract_batch_features(processed_texts)
            # Нормализуем признаки
            if split_name == 'train' or not hasattr(self.feature_extractor.scaler, 'mean_'):
                meta_array = self.feature_extractor.normalize_features(meta_features, fit=(split_name == 'train'))
            else:
                meta_array = self.feature_extractor.normalize_features(meta_features, fit=False)
            
            result['meta_features'] = meta_features
            result['meta_features_array'] = meta_array
            print(f"       Извлечено {meta_features.shape[1]} мета-признаков")
        
        # 3. Создание векторных представлений
        if create_vectors:
            print(f"    3️⃣ Создание векторных представлений...")
            try:
                # Создаем векторизатор если его нет
                if self.vectorizer is None:
                    self.vectorizer = SplitAwareVectorizer(**self.vectorizer_params)
                
                if fit_vectorizer:
                    # Обучение векторизатора (только для train)
                    text_vectors = self.vectorizer.fit_transform(processed_texts)
                    print(f"      Векторизатор обучен на {split_name} данных")
                else:
                    # Использование обученного векторизатора
                    text_vectors = self.vectorizer.transform(processed_texts)
                    print(f"      Использован обученный векторизатор")
                
                # Преобразуем разреженные матрицы в плотные для совместимости
                if hasattr(text_vectors, 'toarray'):
                    text_vectors = text_vectors.toarray()
                
                result['text_vectors'] = text_vectors
                print(f"       Создано векторов: {text_vectors.shape}")
                
            except Exception as e:
                print(f"      ⚠️ Ошибка при создании векторов: {e}")
                result['text_vectors'] = None
        
        # 4. Комбинированные признаки (если оба типа извлечены)
        if extract_meta and create_vectors and result.get('meta_features_array') is not None and result.get('text_vectors') is not None:
            print(f"    4️⃣ Комбинирование признаков...")
            combined = np.hstack([result['meta_features_array'], result['text_vectors']])
            result['combined_features'] = combined
            print(f"       Комбинированные признаки: {combined.shape}")
        
        # Сохраняем в общее хранилище
        self.processed_data[split_name] = result
        
        return result
    
    def get_dense_features(self, splits_results=None):
        """
        Преобразование разреженных признаков в плотные
        
        Args:
            splits_results: результаты обработки разделов
            
        Returns:
            Словарь с плотными признаками
        """
        dense_results = {}
        
        if splits_results is None and hasattr(self, 'results'):
            splits_results = self.results
        
        if not splits_results:
            return dense_results
        
        for split_name, split_data in splits_results.items():
            dense_results[split_name] = {}
            
            # Копируем все данные
            for key, value in split_data.items():
                if key != 'text_vectors':
                    dense_results[split_name][key] = value
            
            # Преобразуем векторы
            if 'text_vectors' in split_data and split_data['text_vectors'] is not None:
                vectors = split_data['text_vectors']
                
                # Преобразуем разреженные матрицы в плотные
                if hasattr(vectors, 'toarray'):
                    dense_vectors = vectors.toarray()
                else:
                    dense_vectors = vectors
                
                dense_results[split_name]['text_vectors'] = dense_vectors
                dense_results[split_name]['text_vectors_dense'] = True
            
            # Также обрабатываем combined_features
            if 'combined_features' in split_data and split_data['combined_features'] is not None:
                combined = split_data['combined_features']
                
                if hasattr(combined, 'toarray'):
                    dense_combined = combined.toarray()
                else:
                    dense_combined = combined
                
                dense_results[split_name]['combined_features'] = dense_combined
        
        return dense_results
    
    def get_processed_texts(self, split_name: str) -> List[str]:
        """Получение обработанных текстов для раздела"""
        return self.processed_data.get(split_name, {}).get('processed_texts', [])
    
    def get_features(self, split_name: str, feature_type: str = 'combined') -> Optional[np.ndarray]:
        """
        Получение признаков для раздела
        
        Args:
            split_name: имя раздела
            feature_type: тип признаков ('meta', 'text', 'combined')
            
        Returns:
            матрица признаков или None
        """
        data = self.processed_data.get(split_name, {})
        
        if feature_type == 'meta':
            return data.get('meta_features_array')
        elif feature_type == 'text':
            return data.get('text_vectors')
        elif feature_type == 'combined':
            return data.get('combined_features')
        else:
            return None
    
    def save_processed_data(self, output_dir: str = "processed_data"):
        """Сохранение обработанных данных на диск"""
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, data in self.processed_data.items():
            if data:
                split_dir = os.path.join(output_dir, split_name)
                os.makedirs(split_dir, exist_ok=True)
                
                # Сохраняем обработанные тексты
                if 'processed_texts' in data:
                    with open(os.path.join(split_dir, 'texts.json'), 'w', encoding='utf-8') as f:
                        json.dump(data['processed_texts'], f, ensure_ascii=False, indent=2)
                
                # Сохраняем мета-признаки
                if 'meta_features' in data:
                    data['meta_features'].to_csv(os.path.join(split_dir, 'meta_features.csv'), index=False)
                
                # Сохраняем векторы
                if 'text_vectors' in data and data['text_vectors'] is not None:
                    np.save(os.path.join(split_dir, 'text_vectors.npy'), data['text_vectors'])
                
                # Сохраняем комбинированные признаки
                if 'combined_features' in data and data['combined_features'] is not None:
                    np.save(os.path.join(split_dir, 'combined_features.npy'), data['combined_features'])
                
                print(f"💾 Сохранены данные для {split_name} в {split_dir}")
        
        # Сохраняем векторизатор
        if self.vectorizer:
            self.vectorizer.save(os.path.join(output_dir, 'vectorizer.pkl'))
            print(f"💾 Сохранен векторизатор")
    
    def load_processed_data(self, input_dir: str = "processed_data"):
        """Загрузка обработанных данных с диска"""
        for split_name in ['train', 'validation', 'test']:
            split_dir = os.path.join(input_dir, split_name)
            if os.path.exists(split_dir):
                data = {}
                
                # Загружаем тексты
                texts_file = os.path.join(split_dir, 'texts.json')
                if os.path.exists(texts_file):
                    with open(texts_file, 'r', encoding='utf-8') as f:
                        data['processed_texts'] = json.load(f)
                
                # Загружаем мета-признаки
                meta_file = os.path.join(split_dir, 'meta_features.csv')
                if os.path.exists(meta_file):
                    data['meta_features'] = pd.read_csv(meta_file)
                
                # Загружаем векторы
                vectors_file = os.path.join(split_dir, 'text_vectors.npy')
                if os.path.exists(vectors_file):
                    data['text_vectors'] = np.load(vectors_file)
                
                # Загружаем комбинированные признаки
                combined_file = os.path.join(split_dir, 'combined_features.npy')
                if os.path.exists(combined_file):
                    data['combined_features'] = np.load(combined_file)
                
                self.processed_data[split_name] = data
                print(f"📂 Загружены данные для {split_name}")
        
        # Загружаем векторизатор
        vectorizer_file = os.path.join(input_dir, 'vectorizer.pkl')
        if os.path.exists(vectorizer_file):
            self.vectorizer = SplitAwareVectorizer.load(vectorizer_file)
            print(f"📂 Загружен векторизатор")


# Пример использования
def example_usage_with_splits():
    """Пример использования с разделенными данными"""
    
    # Создаем тестовые разделенные данные
    splits = {
        'train': [
            {'text': 'Это тренировочный текст номер один. Он будет использован для обучения.', 'category': 'технологии'},
            {'text': 'Еще один тренировочный текст для машинного обучения.', 'category': 'наука'},
        ],
        'validation': [
            {'text': 'Валидационный текст для настройки параметров модели.', 'category': 'технологии'},
        ],
        'test': [
            {'text': 'Тестовый текст для финальной оценки модели.', 'category': 'наука'},
        ]
    }
    
    print("🧪 Пример обработки разделенных данных")
    print("=" * 60)
    
    # Создаем процессор
    processor = TextDataProcessor(
        preprocessor_params={
            'language': 'russian',
            'remove_stopwords': True,
            'use_spacy': False
        },
        vectorizer_params={
            'method': 'tfidf',
            'max_features': 100,
            'ngram_range': (1, 1)
        }
    )
    
    # Обрабатываем все разделы
    results = processor.process_splits(
        splits,
        extract_meta=True,
        create_vectors=True
    )
    
    # Показываем результаты
    for split_name, result in results.items():
        print(f"\n📊 {split_name.upper()}:")
        print(f"  Обработано текстов: {len(result.get('processed_texts', []))}")
        
        if 'meta_features' in result:
            print(f"  Мета-признаков: {result['meta_features'].shape[1]}")
        
        if 'text_vectors' in result and result['text_vectors'] is not None:
            print(f"  Размерность векторов: {result['text_vectors'].shape[1]}")
        
        if 'combined_features' in result and result['combined_features'] is not None:
            print(f"  Комбинированных признаков: {result['combined_features'].shape[1]}")
    
    # Сохраняем результаты
    processor.save_processed_data("example_processed")


if __name__ == "__main__":
    example_usage_with_splits()