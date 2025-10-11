import pandas as pd
import time
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from razdel import tokenize
import pymorphy3
import spacy
from nltk.stem import SnowballStemmer

class TextProcessingExperiment:
    def __init__(self, texts):
        self.texts = texts
        self.results = []
        
        # Инициализация инструментов один раз
        self.morph = pymorphy3.MorphAnalyzer()
        self.nlp_spacy = spacy.load("ru_core_news_sm")
        self.snowball_stemmer = SnowballStemmer("russian")
        
        # Создаем эталонный словарь из исходных текстов
        self.reference_vocab = self._create_reference_vocab()

    def _create_reference_vocab(self):
        """Создает эталонный словарь из исходных текстов с помощью razdel"""
        vocab = set()
        for text in self.texts[:100]:  # Используем подмножество для словаря
            tokens = [token.text for token in tokenize(text)]
            vocab.update(tokens)
        return vocab

    # Методы токенизации
    def tokenize_naive(self, text):
        return text.split()

    def tokenize_regex(self, text):
        import re
        return re.findall(r'\b\w+\b', text)

    def tokenize_nltk(self, text):
        from nltk.tokenize import word_tokenize
        return word_tokenize(text, language='russian')

    def tokenize_spacy(self, text):
        doc = self.nlp_spacy(text)
        return [token.text for token in doc]

    def tokenize_razdel(self, text):
        return [token.text for token in tokenize(text)]

    # Методы нормализации
    def stem_snowball(self, tokens):
        return [self.snowball_stemmer.stem(token) for token in tokens]

    def lemmatize_pymorphy(self, tokens):
        return [self.morph.parse(token)[0].normal_form for token in tokens]

    def lemmatize_spacy(self, tokens):
        # Более эффективный подход для spaCy
        text = ' '.join(tokens)
        doc = self.nlp_spacy(text)
        return [token.lemma_ for token in doc]

    # Метрики оценки (упрощенные и улучшенные)
    def calculate_vocabulary_size(self, processed_texts):
        unique_tokens = set()
        for tokens in processed_texts:
            unique_tokens.update(tokens)
        return len(unique_tokens)

    def calculate_oov_rate(self, processed_texts):
        """Доля слов, отсутствующих в эталонном словаре"""
        if not self.reference_vocab:
            return 0
            
        total_tokens = 0
        oov_count = 0
        
        for tokens in processed_texts:
            total_tokens += len(tokens)
            oov_count += sum(1 for token in tokens if token not in self.reference_vocab)
        
        return (oov_count / total_tokens * 100) if total_tokens > 0 else 0

    def calculate_processing_speed(self, processing_times, num_texts):
        """Скорость обработки в статьях в секунду"""
        total_time = sum(processing_times)
        return num_texts / total_time if total_time > 0 else 0

    def run_experiment(self):
        """Упрощенный и более четкий эксперимент"""
        
        # Тестируем комбинации токенизации + нормализации
        configurations = [
            # Только токенизация
            ('Наивная токенизация', 'tokenize_naive', None),
            ('Regex токенизация', 'tokenize_regex', None),
            ('NLTK токенизация', 'tokenize_nltk', None),
            ('spaCy токенизация', 'tokenize_spacy', None),
            ('Razdel токенизация', 'tokenize_razdel', None),
            
            # Razdel + нормализация (честное сравнение)
            ('Razdel + Snowball', 'tokenize_razdel', 'stem_snowball'),
            ('Razdel + pymorphy3', 'tokenize_razdel', 'lemmatize_pymorphy'),
            ('Razdel + spaCy lemma', 'tokenize_razdel', 'lemmatize_spacy'),
        ]

        for name, tokenizer_name, normalizer_name in configurations:
            print(f"Тестируем: {name}")
            
            tokenizer = getattr(self, tokenizer_name)
            normalizer = getattr(self, normalizer_name) if normalizer_name else None
            
            processing_times = []
            all_processed_tokens = []
            
            start_total = time.time()
            
            for text in self.texts:
                start_time = time.time()
                
                # Применяем токенизацию
                tokens = tokenizer(text)
                
                # Применяем нормализацию если есть
                if normalizer:
                    tokens = normalizer(tokens)
                
                processing_times.append(time.time() - start_time)
                all_processed_tokens.append(tokens)
            
            total_time = time.time() - start_total
            
            # Вычисляем метрики
            vocab_size = self.calculate_vocabulary_size(all_processed_tokens)
            oov_rate = self.calculate_oov_rate(all_processed_tokens)
            speed = self.calculate_processing_speed(processing_times, len(self.texts))
            
            self.results.append({
                'Метод': name,
                'Тип': 'нормализация' if normalizer else 'токенизация',
                'Объем словаря': vocab_size,
                'Доля OOV (%)': round(oov_rate, 2),
                'Скорость (ст/сек)': round(speed, 2),
                'Время обработки (сек)': round(total_time, 2)
            })

    def save_results(self, filename='tokenization_metrics.csv'):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        return df

# Загрузка данных и запуск
def load_texts_from_jsonl(file_path, max_texts=500):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_texts:
                break
            if line.strip():
                article = json.loads(line)
                texts.append(article['text'])
    return texts

if __name__ == "__main__":
    texts = load_texts_from_jsonl('indicator_ru_corpus_advanced_cleaned.jsonl', max_texts=500)
    
    experiment = TextProcessingExperiment(texts)
    experiment.run_experiment()
    results_df = experiment.save_results()
    
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА:")
    print(results_df.to_string(index=False))