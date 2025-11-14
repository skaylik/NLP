# vectorization_comparison.py
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import psutil
import os
import json

class VectorizationComparator:
    def __init__(self, vectorizers=None):
        self.vectorizers = vectorizers
        self.results = []
        self.texts = []
        self.categories = []
        
    def load_corpus(self, jsonl_file, text_field='text', max_docs=5000):
        """Загрузка корпуса для сравнения"""
        texts = []
        categories = []
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_docs:
                    break
                try:
                    data = json.loads(line.strip())
                    texts.append(data[text_field])
                    categories.append(data.get('category', ''))
                except (json.JSONDecodeError, KeyError):
                    continue
        
        self.texts = texts
        self.categories = categories
        self.category_encoder = LabelEncoder()
        if categories and len(set(categories)) > 1:
            self.category_ids = self.category_encoder.fit_transform(categories)
        else:
            self.category_ids = np.array([])
        
        return len(texts)
    
    def get_memory_usage(self):
        """Получение текущего использования памяти"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def calculate_semantic_coherence(self, vector_matrix, sample_size=1000):
        """Вычисление семантической согласованности через косинусное сходство"""
        if len(self.categories) == 0 or len(set(self.categories)) < 2:
            return 0.0
            
        if len(self.texts) > sample_size:
            indices = np.random.choice(len(self.texts), sample_size, replace=False)
            if hasattr(vector_matrix, 'shape'):
                sample_matrix = vector_matrix[indices]
            else:
                sample_matrix = [vector_matrix[i] for i in indices]
            sample_categories = [self.categories[i] for i in indices]
        else:
            sample_matrix = vector_matrix
            sample_categories = self.categories
        
        similarities = []
        unique_categories = set(sample_categories)
        
        for category in unique_categories:
            category_indices = [i for i, cat in enumerate(sample_categories) if cat == category]
            if len(category_indices) > 1:
                if hasattr(sample_matrix, 'shape'):
                    category_vectors = sample_matrix[category_indices]
                else:
                    category_vectors = [sample_matrix[i] for i in category_indices]
                
                if hasattr(category_vectors, 'toarray'):
                    category_vectors = category_vectors.toarray()
                
                try:
                    category_similarity = cosine_similarity(category_vectors)
                    np.fill_diagonal(category_similarity, 0)
                    
                    if len(category_indices) > 1:
                        avg_similarity = category_similarity.sum() / (len(category_indices) * (len(category_indices) - 1))
                        similarities.append(avg_similarity)
                except ValueError:
                    continue
        
        return np.mean(similarities) if similarities else 0.0
    
    def evaluate_method(self, vector_matrix, categories, method_name="Custom"):
        """Упрощенная оценка метода векторизации"""
        if vector_matrix is None:
            return None
            
        try:
            # Анализ размерности
            dimensions = vector_matrix.shape[1] if hasattr(vector_matrix, 'shape') else len(vector_matrix[0])
            
            # Анализ разреженности
            if hasattr(vector_matrix, 'nnz'):
                sparsity = 1 - (vector_matrix.nnz / (vector_matrix.shape[0] * vector_matrix.shape[1]))
            else:
                dense_matrix = vector_matrix.toarray() if hasattr(vector_matrix, 'toarray') else np.array(vector_matrix)
                sparsity = np.count_nonzero(dense_matrix == 0) / dense_matrix.size
            
            # Семантическая согласованность
            semantic_coherence = self.calculate_semantic_coherence(vector_matrix)
            
            result = {
                'Method': method_name,
                'Dimensions': dimensions,
                'Sparsity (%)': round(sparsity * 100, 2),
                'Semantic Coherence': round(semantic_coherence, 4),
                'Accuracy': 0.0,  # Заглушка для совместимости
                'F1-Score': 0.0   # Заглушка для совместимости
            }
            
            return result
            
        except Exception as e:
            print(f"Ошибка при оценке метода {method_name}: {e}")
            return None
    
    def compare_methods(self):
        """Сравнение всех методов векторизации"""
        if self.vectorizers is None:
            return []
            
        methods_config = [
            ('One-Hot (1-gram)', {'method': 'one_hot_encoding', 'ngram_range': (1, 1)}),
            ('One-Hot (1-2-gram)', {'method': 'one_hot_encoding', 'ngram_range': (1, 2)}),
            ('BoW (1-gram)', {'method': 'bag_of_words', 'ngram_range': (1, 1), 'binary': False}),
            ('BoW (1-2-gram)', {'method': 'bag_of_words', 'ngram_range': (1, 2), 'binary': False}),
            ('TF-IDF (1-gram)', {'method': 'tfidf', 'ngram_range': (1, 1)}),
            ('TF-IDF (1-2-gram)', {'method': 'tfidf', 'ngram_range': (1, 2)}),
        ]
        
        for method_name, config in methods_config:
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            try:
                method_func = getattr(self.vectorizers, config['method'])
                kwargs = {k: v for k, v in config.items() if k != 'method'}
                vector_matrix = method_func(self.texts, **kwargs)
                
                processing_time = time.time() - start_time
                end_memory = self.get_memory_usage()
                memory_usage = end_memory - start_memory
                
                result = self.evaluate_vectorization_method(
                    method_name, vector_matrix, processing_time, memory_usage
                )
                
                if result:
                    self.results.append(result)
                
            except Exception as e:
                print(f"Ошибка при применении метода {method_name}: {e}")
                continue
        
        return self.results
    
    def evaluate_vectorization_method(self, method_name, vector_matrix, processing_time, memory_usage):
        """Оценка одного метода векторизации"""
        if vector_matrix is None:
            return None
            
        dimensions = vector_matrix.shape[1] if hasattr(vector_matrix, 'shape') else len(vector_matrix[0])
        
        if hasattr(vector_matrix, 'nnz'):
            sparsity = 1 - (vector_matrix.nnz / (vector_matrix.shape[0] * vector_matrix.shape[1]))
        else:
            dense_matrix = vector_matrix.toarray() if hasattr(vector_matrix, 'toarray') else np.array(vector_matrix)
            sparsity = np.count_nonzero(dense_matrix == 0) / dense_matrix.size
        
        semantic_coherence = self.calculate_semantic_coherence(vector_matrix)
        
        result = {
            'method': method_name,
            'dimensions': dimensions,
            'sparsity': round(sparsity, 4),
            'semantic_coherence': round(semantic_coherence, 4),
            'processing_time_sec': round(processing_time, 2),
            'memory_usage_mb': round(memory_usage, 2)
        }
        
        return result
    
    def save_results(self, output_file='vectorization_metrics.csv'):
        """Сохранение результатов в CSV"""
        if not self.results:
            return None
            
        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        return output_file
    
    def get_summary_statistics(self):
        """Получение сводной статистики по методам"""
        if not self.results:
            return {}
            
        df = pd.DataFrame(self.results)
        
        best_semantic = df.loc[df['semantic_coherence'].idxmax()] if len(df) > 0 else None
        best_processing = df.loc[df['processing_time_sec'].idxmin()] if len(df) > 0 else None
        best_memory = df.loc[df['memory_usage_mb'].idxmin()] if len(df) > 0 else None
        
        summary = {
            'best_semantic_coherence': best_semantic['method'] if best_semantic is not None else 'N/A',
            'best_semantic_score': best_semantic['semantic_coherence'] if best_semantic is not None else 0,
            'best_processing_time': best_processing['method'] if best_processing is not None else 'N/A',
            'best_processing_score': best_processing['processing_time_sec'] if best_processing is not None else 0,
            'lowest_memory_usage': best_memory['method'] if best_memory is not None else 'N/A',
            'lowest_memory_score': best_memory['memory_usage_mb'] if best_memory is not None else 0,
            'avg_dimensions': round(df['dimensions'].mean(), 2),
            'avg_processing_time': round(df['processing_time_sec'].mean(), 2),
            'avg_memory_usage': round(df['memory_usage_mb'].mean(), 2),
            'total_methods_tested': len(df)
        }
        
        return summary