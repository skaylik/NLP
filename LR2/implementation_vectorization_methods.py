# implementation_vectorization_methods.py
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
import json

class ClassicalVectorizers:
    def __init__(self):
        self.vectorizers = {}
        self.vocabularies = {}
        
    def load_corpus(self, jsonl_file, text_field='text', category_field='category', limit=None):
        """Загрузка корпуса из JSONL файла"""
        texts = []
        categories = []
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                try:
                    data = json.loads(line.strip())
                    if text_field in data and data[text_field]:
                        texts.append(data[text_field])
                        # Безопасное получение категории
                        category = data.get(category_field, '')
                        categories.append(category if category else 'другое')
                except json.JSONDecodeError:
                    continue
        
        return texts, categories
    
    def one_hot_encoding(self, texts, ngram_range=(1, 1), max_features=None):
        """One-Hot Encoding для слов и n-грамм"""
        try:
            vectorizer = CountVectorizer(
                binary=True,
                ngram_range=ngram_range,
                max_features=max_features,
                token_pattern=r'(?u)\b\w+\b|<\w+>'  # Поддержка специальных токенов
            )
            
            X = vectorizer.fit_transform(texts)
            key = f'one_hot_{ngram_range[0]}_{ngram_range[1]}'
            self.vectorizers[key] = vectorizer
            self.vocabularies[key] = vectorizer.get_feature_names_out()
            
            return X
        except Exception as e:
            print(f"Ошибка One-Hot Encoding: {e}")
            return None
    
    def bag_of_words(self, texts, ngram_range=(1, 1), max_features=None, binary=False):
        """Bag of Words с различными схемами взвешивания"""
        try:
            vectorizer = CountVectorizer(
                binary=binary,
                ngram_range=ngram_range,
                max_features=max_features,
                token_pattern=r'(?u)\b\w+\b|<\w+>'  # Поддержка специальных токенов
            )
            
            X = vectorizer.fit_transform(texts)
            key = f'bow_{"binary" if binary else "freq"}_{ngram_range[0]}_{ngram_range[1]}'
            self.vectorizers[key] = vectorizer
            self.vocabularies[key] = vectorizer.get_feature_names_out()
            
            return X
        except Exception as e:
            print(f"Ошибка Bag of Words: {e}")
            return None
    
    def tfidf(self, texts, ngram_range=(1, 1), max_features=None,
             smooth_idf=True, sublinear_tf=False, norm='l2'):
        """TF-IDF с настройкой параметров"""
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                smooth_idf=smooth_idf,
                sublinear_tf=sublinear_tf,
                norm=norm,
                token_pattern=r'(?u)\b\w+\b|<\w+>'  # Поддержка специальных токенов
            )
            
            X = vectorizer.fit_transform(texts)
            key = f'tfidf_{ngram_range[0]}_{ngram_range[1]}_smooth{int(smooth_idf)}_sublinear{int(sublinear_tf)}'
            self.vectorizers[key] = vectorizer
            self.vocabularies[key] = vectorizer.get_feature_names_out()
            
            return X
        except Exception as e:
            print(f"Ошибка TF-IDF: {e}")
            return None
    
    def combined_ngrams(self, texts, max_ngram=2, max_features=None):
        """Комбинированные n-граммы с ограничением по максимальной n-грамме"""
        try:
            ngram_ranges = [(1, n) for n in range(1, max_ngram + 1)]
            combined_features = []
            feature_names = []
            vectorizers = []
            
            for ngram_range in ngram_ranges:
                vectorizer = CountVectorizer(
                    ngram_range=ngram_range,
                    max_features=max_features,
                    token_pattern=r'(?u)\b\w+\b|<\w+>'  # Поддержка специальных токенов
                )
                X = vectorizer.fit_transform(texts)
                combined_features.append(X)
                feature_names.extend(vectorizer.get_feature_names_out())
                vectorizers.append(vectorizer)
            
            X_combined = hstack(combined_features) if len(combined_features) > 1 else combined_features[0]
            
            key = f'combined_ngrams_1_to_{max_ngram}'
            self.vectorizers[key] = vectorizers
            self.vocabularies[key] = feature_names
            
            return X_combined
        except Exception as e:
            print(f"Ошибка комбинации n-грамм: {e}")
            return None
    
    def analyze_sparsity(self, matrix, method_name):
        """Анализ разреженности матрицы с оптимизацией для sparse матриц"""
        try:
            if hasattr(matrix, 'toarray'):
                # Для sparse матриц
                total_elements = matrix.shape[0] * matrix.shape[1]
                non_zero_elements = matrix.nnz
                zero_elements = total_elements - non_zero_elements
                sparsity = zero_elements / total_elements
                density = non_zero_elements / total_elements
            else:
                # Для dense матриц
                total_elements = matrix.size
                non_zero_elements = np.count_nonzero(matrix)
                zero_elements = total_elements - non_zero_elements
                sparsity = zero_elements / total_elements
                density = non_zero_elements / total_elements
            
            return {
                'Метод': method_name,
                'Размерность': f"{matrix.shape[0]}×{matrix.shape[1]}",
                'Всего элементов': f"{total_elements:,}",
                'Ненулевые элементы': f"{non_zero_elements:,}",
                'Нулевые элементы': f"{zero_elements:,}",
                'Разреженность (%)': f"{sparsity*100:.2f}%",
                'Плотность (%)': f"{density*100:.2f}%"
            }
        except Exception as e:
            print(f"Ошибка анализа разреженности: {e}")
            return None
    
    def compare_all_methods(self, texts, ngram_ranges=[(1, 1), (1, 2), (1, 3)]):
        """Сравнительный анализ всех методов векторизации"""
        results = {}
        
        # Тестируем различные n-gram диапазоны
        for ngram_range in ngram_ranges:
            methods = [
                (f'One-Hot {ngram_range}', lambda t: self.one_hot_encoding(t, ngram_range=ngram_range)),
                (f'BoW {ngram_range}', lambda t: self.bag_of_words(t, ngram_range=ngram_range, binary=False)),
                (f'Binary BoW {ngram_range}', lambda t: self.bag_of_words(t, ngram_range=ngram_range, binary=True)),
                (f'TF-IDF {ngram_range}', lambda t: self.tfidf(t, ngram_range=ngram_range)),
            ]
            
            for name, method in methods:
                X = method(texts)
                if X is not None:
                    results[name] = self.analyze_sparsity(X, name)
        
        # Тестируем комбинированные n-граммы
        X_combined = self.combined_ngrams(texts, max_ngram=3)
        if X_combined is not None:
            results['Combined_Ngrams'] = self.analyze_sparsity(X_combined, 'Combined_Ngrams')
        
        return results
    
    def get_vocabulary_size(self, method_name):
        """Получение размера словаря для метода"""
        if method_name in self.vocabularies:
            return len(self.vocabularies[method_name])
        
        # Поиск по паттерну
        for key in self.vocabularies:
            if method_name in key:
                return len(self.vocabularies[key])
        return 0
    
    def get_feature_names(self, method_name, top_n=None):
        """Получение имен признаков"""
        features = []
        
        if method_name in self.vocabularies:
            features = self.vocabularies[method_name]
        else:
            # Поиск по паттерну
            for key in self.vocabularies:
                if method_name in key:
                    features = self.vocabularies[key]
                    break
        
        if top_n and features:
            return features[:top_n]
        return features

    # Алиасы для обратной совместимости
    def tfidf_vectorizer(self, texts, ngram_range=(1, 1), max_features=None,
                        smooth_idf=True, sublinear_tf=False, norm='l2'):
        """Алиас для tfidf метода"""
        return self.tfidf(texts, ngram_range, max_features, smooth_idf, sublinear_tf, norm)
    
    def combine_ngrams(self, texts, ngram_ranges=[(1, 1), (1, 2), (1, 3)]):
        """Алиас для combined_ngrams метода"""
        max_ngram = max(r[1] for r in ngram_ranges)
        return self.combined_ngrams(texts, max_ngram)