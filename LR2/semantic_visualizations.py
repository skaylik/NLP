# semantic_visualizations.py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import re
from collections import defaultdict

class SemanticVisualizations:
    """Класс для визуализации семантических операций и анализа"""
    
    def __init__(self, models):
        self.models = models
    
    def _get_word_vector(self, word: str, model) -> tuple:
        """
        Получение вектора слова с поддержкой OOV для FastText
        Возвращает (вектор, статус) где статус: 'found', 'oov_fasttext', 'not_found'
        """
        # Сначала проверяем точное совпадение
        if word in model.wv:
            return (model.wv[word].copy(), 'found')
        
        # Для FastText пробуем получить вектор через OOV
        # Проверяем, что это FastText модель (имеет метод get_vector для OOV)
        is_fasttext = False
        try:
            from gensim.models import FastText as FastTextModel
            is_fasttext = isinstance(model, FastTextModel)
        except:
            # Альтернативная проверка: наличие метода get_vector и название модели
            is_fasttext = (hasattr(model, 'wv') and 
                          hasattr(model.wv, 'get_vector') and 
                          'fasttext' in str(type(model)).lower())
        
        if is_fasttext:
            try:
                # FastText может обработать OOV слова через subword information
                vector = model.wv.get_vector(word, norm=True)
                return (vector, 'oov_fasttext')
            except KeyError:
                pass
        
        # Пробуем варианты написания (первая буква заглавная/строчная)
        variants = [word.lower(), word.capitalize(), word.title()]
        for variant in variants:
            if variant != word and variant in model.wv:
                return (model.wv[variant].copy(), 'found')
        
        # Для FastText пробуем варианты через OOV
        if is_fasttext:
            for variant in variants:
                if variant != word:
                    try:
                        vector = model.wv.get_vector(variant, norm=True)
                        return (vector, 'oov_fasttext')
                    except KeyError:
                        continue
        
        return (None, 'not_found')
    
    def parse_vector_expression(self, expression: str, model) -> Dict[str, Any]:
        """
        Парсинг выражения типа "король - мужчина + женщина"
        Возвращает промежуточные векторы и результат
        Поддерживает OOV слова для FastText моделей
        """
        try:
            # Нормализуем выражение
            expression = expression.strip()
            
            # Определяем тип модели
            is_fasttext = False
            try:
                from gensim.models import FastText as FastTextModel
                is_fasttext = isinstance(model, FastTextModel)
            except:
                is_fasttext = (hasattr(model, 'wv') and 
                              hasattr(model.wv, 'get_vector') and 
                              'fasttext' in str(type(model)).lower())
            
            model_type = 'FastText' if is_fasttext else 'Word2Vec'
            
            # Разбиваем на части: слова и операции
            parts = re.split(r'([+\-])', expression)
            parts = [p.strip() for p in parts if p.strip()]
            
            positive_words = []
            negative_words = []
            operation_steps = []
            oov_words = []  # Слова, обработанные через OOV
            missing_words = []  # Слова, которые не удалось найти
            
            current_sign = '+'
            
            for part in parts:
                if part in ['+', '-']:
                    current_sign = part
                else:
                    word = part.strip()
                    vector, status = self._get_word_vector(word, model)
                    
                    if status == 'not_found':
                        missing_words.append(word)
                    else:
                        if current_sign == '+':
                            positive_words.append(word)
                        else:
                            negative_words.append(word)
                        
                        if status == 'oov_fasttext':
                            oov_words.append(word)
                        
                        operation_steps.append({
                            'word': word,
                            'operation': current_sign,
                            'vector': vector.copy(),
                            'status': status
                        })
            
            # Если есть отсутствующие слова, возвращаем ошибку с информацией
            if missing_words:
                error_msg = f'Слова отсутствуют в модели: {", ".join(missing_words)}'
                if is_fasttext:
                    error_msg += f'\n\n💡 Для FastText можно попробовать другие формы слова (например, "{missing_words[0].lower()}" или "{missing_words[0].capitalize()}")'
                else:
                    error_msg += f'\n\n💡 Word2Vec не поддерживает OOV слова. Используйте FastText для работы с неизвестными словами.'
                return {'error': error_msg}
            
            if not positive_words and not negative_words:
                return {'error': 'Не найдено ни одного слова в модели'}
            
            # Вычисляем промежуточные векторы
            intermediate_vectors = []
            
            # Шаг 1: Начальный вектор (первое положительное слово)
            if positive_words:
                current_vector = model.wv[positive_words[0]].copy()
                intermediate_vectors.append({
                    'step': 1,
                    'description': f'Начальный вектор: {positive_words[0]}',
                    'vector': current_vector.copy(),
                    'norm': np.linalg.norm(current_vector)
                })
                
                # Добавляем остальные положительные
                for word in positive_words[1:]:
                    current_vector += model.wv[word]
                    intermediate_vectors.append({
                        'step': len(intermediate_vectors) + 1,
                        'description': f'Добавление: +{word}',
                        'vector': current_vector.copy(),
                        'norm': np.linalg.norm(current_vector)
                    })
            
            # Вычитаем отрицательные
            for word in negative_words:
                current_vector -= model.wv[word]
                intermediate_vectors.append({
                    'step': len(intermediate_vectors) + 1,
                    'description': f'Вычитание: -{word}',
                    'vector': current_vector.copy(),
                    'norm': np.linalg.norm(current_vector)
                })
            
            # Финальный результат
            final_vector = current_vector
            
            result = {
                'expression': expression,
                'positive_words': positive_words,
                'negative_words': negative_words,
                'intermediate_vectors': intermediate_vectors,
                'final_vector': final_vector,
                'final_norm': np.linalg.norm(final_vector),
                'operation_steps': operation_steps,
                'model_type': model_type
            }
            
            # Добавляем информацию об OOV словах
            if oov_words:
                result['oov_words'] = oov_words
                result['info'] = f'ℹ️ Некоторые слова ({", ".join(oov_words)}) обработаны через OOV (FastText)'
            
            return result
            
        except Exception as e:
            return {'error': f'Ошибка парсинга: {str(e)}'}
    
    def visualize_vector_arithmetic(self, expression_result: Dict[str, Any], model, top_n: int = 10) -> Dict[str, Any]:
        """Визуализация промежуточных векторов и результата"""
        if 'error' in expression_result:
            return expression_result
        
        # Получаем ближайших соседей для каждого промежуточного шага
        neighbors_for_steps = []
        
        for step_data in expression_result['intermediate_vectors']:
            try:
                # Создаем временный ключ для вектора
                neighbors = model.wv.similar_by_vector(
                    step_data['vector'], 
                    topn=top_n
                )
                neighbors_for_steps.append({
                    'step': step_data['step'],
                    'description': step_data['description'],
                    'neighbors': neighbors,
                    'vector_norm': step_data['norm']
                })
            except:
                neighbors_for_steps.append({
                    'step': step_data['step'],
                    'description': step_data['description'],
                    'neighbors': [],
                    'vector_norm': step_data['norm']
                })
        
        # Ближайшие соседи для финального результата
        try:
            final_neighbors = model.wv.similar_by_vector(
                expression_result['final_vector'],
                topn=top_n
            )
        except:
            final_neighbors = []
        
        return {
            **expression_result,
            'step_neighbors': neighbors_for_steps,
            'final_neighbors': final_neighbors
        }
    
    def calculate_cosine_distance(self, word1: str, word2: str, model) -> Optional[float]:
        """Вычисление косинусного расстояния между двумя словами"""
        try:
            if word1 not in model.wv or word2 not in model.wv:
                return None
            
            vec1 = model.wv[word1]
            vec2 = model.wv[word2]
            
            # Косинусное расстояние (1 - косинусное сходство)
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            distance = 1 - similarity
            
            return {
                'word1': word1,
                'word2': word2,
                'cosine_similarity': float(similarity),
                'cosine_distance': float(distance),
                'euclidean_distance': float(np.linalg.norm(vec1 - vec2))
            }
        except Exception as e:
            return None
    
    def build_semantic_graph(self, words: List[str], model, threshold: float = 0.3) -> Dict[str, Any]:
        """Построение графа семантических связей"""
        try:
            G = nx.Graph()
            
            # Проверяем наличие слов в модели
            valid_words = [w for w in words if w in model.wv]
            
            if len(valid_words) < 2:
                return {'error': 'Недостаточно слов в модели для построения графа'}
            
            # Добавляем узлы
            for word in valid_words:
                G.add_node(word)
            
            # Вычисляем попарные сходства и добавляем рёбра
            edges_data = []
            for i, word1 in enumerate(valid_words):
                for word2 in valid_words[i+1:]:
                    similarity = cosine_similarity(
                        [model.wv[word1]], 
                        [model.wv[word2]]
                    )[0][0]
                    
                    if similarity >= threshold:
                        G.add_edge(word1, word2, weight=similarity)
                        edges_data.append({
                            'source': word1,
                            'target': word2,
                            'similarity': float(similarity)
                        })
            
            # Вычисляем метрики графа
            metrics = {
                'nodes': len(G.nodes()),
                'edges': len(G.edges()),
                'density': nx.density(G),
                'average_clustering': nx.average_clustering(G) if len(G.nodes()) > 1 else 0,
                'connected_components': nx.number_connected_components(G)
            }
            
            # Позиции для визуализации (используем eigenvectors)
            if len(G.nodes()) > 0:
                try:
                    pos = nx.spring_layout(G, k=1, iterations=50)
                except:
                    pos = nx.circular_layout(G)
            else:
                pos = {}
            
            return {
                'graph': G,
                'positions': pos,
                'edges_data': edges_data,
                'metrics': metrics,
                'words': valid_words
            }
            
        except Exception as e:
            return {'error': f'Ошибка построения графа: {str(e)}'}
    
    def visualize_semantic_axis_interactive(self, axis_data: Dict[str, Any], test_words: List[str], model) -> Dict[str, Any]:
        """Интерактивная визуализация проекций слов на семантическую ось"""
        try:
            positive_words = axis_data.get('positive', [])
            negative_words = axis_data.get('negative', [])
            
            # Вычисляем направление оси
            positive_vectors = [model.wv[w] for w in positive_words if w in model.wv]
            negative_vectors = [model.wv[w] for w in negative_words if w in model.wv]
            
            if not positive_vectors or not negative_vectors:
                return {'error': 'Недостаточно слов для определения оси'}
            
            axis_direction = np.mean(positive_vectors, axis=0) - np.mean(negative_vectors, axis=0)
            axis_direction = axis_direction / np.linalg.norm(axis_direction)
            
            # Проецируем тестовые слова
            projections = []
            for word in test_words:
                if word in model.wv:
                    vector = model.wv[word]
                    projection = np.dot(vector, axis_direction)
                    projections.append({
                        'word': word,
                        'projection': float(projection),
                        'vector': vector
                    })
            
            # Сортируем по проекции
            projections.sort(key=lambda x: x['projection'])
            
            return {
                'axis_direction': axis_direction,
                'projections': projections,
                'positive_words': positive_words,
                'negative_words': negative_words
            }
            
        except Exception as e:
            return {'error': f'Ошибка визуализации оси: {str(e)}'}
    
    def generate_comprehensive_report(self, model_name: str, semantic_ops, test_words: List[str] = None) -> Dict[str, Any]:
        """Генерация комплексного отчета"""
        try:
            if model_name not in self.models:
                return {'error': 'Модель не найдена'}
            
            model = self.models[model_name]
            
            if test_words is None:
                test_words = ['компьютер', 'программа', 'данные', 'город', 'хороший']
            
            report = {
                'model_name': model_name,
                'vocabulary_size': len(model.wv.key_to_index),
                'vector_size': model.vector_size if hasattr(model, 'vector_size') else 0
            }
            
            # 1. Анализ распределения расстояний
            distance_analysis = semantic_ops.analyze_distance_distribution(model_name)
            report['distance_analysis'] = distance_analysis
            
            # 2. Оценка аналогий
            analogy_analysis = semantic_ops.categorical_analogy_evaluation(model_name)
            report['analogy_analysis'] = analogy_analysis
            
            # 3. Анализ ближайших соседей
            neighbors_analysis = semantic_ops.comprehensive_neighbors_analysis(model_name)
            report['neighbors_analysis'] = neighbors_analysis
            
            # 4. Построение графа для тестовых слов
            graph_data = self.build_semantic_graph(test_words, model, threshold=0.3)
            report['semantic_graph'] = graph_data
            
            # 5. Матрица сходства для тестовых слов
            similarity_matrix = np.eye(len(test_words))
            word_to_index = {word: idx for idx, word in enumerate(test_words)}
            
            for i, word1 in enumerate(test_words):
                for j, word2 in enumerate(test_words):
                    if word1 in model.wv and word2 in model.wv:
                        sim = cosine_similarity(
                            [model.wv[word1]], 
                            [model.wv[word2]]
                        )[0][0]
                        similarity_matrix[i][j] = sim
            
            report['similarity_matrix'] = {
                'matrix': similarity_matrix.tolist(),
                'words': test_words
            }
            
            return report
            
        except Exception as e:
            return {'error': f'Ошибка генерации отчета: {str(e)}'}
    
    def project_to_2d_3d(self, words: List[str], model, method: str = 'tsne', dim: int = 2) -> Dict[str, Any]:
        """Проекция слов в 2D/3D пространство"""
        try:
            valid_words = [w for w in words if w in model.wv]
            
            if len(valid_words) < 2:
                return {'error': 'Недостаточно слов для проекции'}
            
            vectors = np.array([model.wv[w] for w in valid_words])
            
            if method == 'tsne':
                reducer = TSNE(n_components=dim, random_state=42, perplexity=min(30, len(valid_words)-1))
            elif method == 'pca':
                reducer = PCA(n_components=dim)
            else:
                return {'error': 'Неизвестный метод проекции'}
            
            projected = reducer.fit_transform(vectors)
            
            return {
                'words': valid_words,
                'projections': projected.tolist(),
                'method': method,
                'dimensions': dim,
                'explained_variance': getattr(reducer, 'explained_variance_ratio_', None)
            }
            
        except Exception as e:
            return {'error': f'Ошибка проекции: {str(e)}'}


