# vector_arithmetic_semantic_operations.py (ИСПРАВЛЕННАЯ ВЕРСИЯ)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional  # ДОБАВЛЕН ИМПОРТ
import logging
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticOperations:
    def __init__(self, models):
        self.models = models
        self.operation_history = []
        
    def _validate_model_and_words(self, model_name: str, words: List[str]) -> bool:
        """Валидация модели и проверка наличия слов"""
        try:
            if model_name not in self.models:
                logger.error(f"Модель {model_name} не найдена")
                return False
            
            model = self.models[model_name]
            
            if not hasattr(model, 'wv'):
                logger.error(f"Модель {model_name} не поддерживает векторные операции")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка валидации: {e}")
            return False

    def comprehensive_semantic_evaluation(self, model_name: str) -> Dict[str, Any]:
        """
        КОМПЛЕКСНАЯ ОЦЕНКА СЕМАНТИЧЕСКИХ СВОЙСТВ ПО ЗАДАНИЮ 3.6
        """
        evaluation_results = {}
        
        # 6.1 Анализ распределения расстояний
        evaluation_results['distance_analysis'] = self.analyze_distance_distribution(model_name)
        
        # 6.2 Векторные аналогии по категориям
        evaluation_results['analogy_analysis'] = self.categorical_analogy_evaluation(model_name)
        
        # 6.3 Семантические оси
        evaluation_results['semantic_axes'] = self.comprehensive_axes_analysis(model_name)
        
        # 6.4 Анализ ближайших соседей
        evaluation_results['neighbors_analysis'] = self.comprehensive_neighbors_analysis(model_name)
        
        # Общая оценка качества
        evaluation_results['quality_score'] = self._calculate_overall_quality(evaluation_results)
        
        return evaluation_results

    def analyze_distance_distribution(self, model_name: str) -> Dict[str, Any]:
        """
        6.1 Анализ распределения косинусных расстояний в пространстве
        """
        try:
            if model_name not in self.models:
                return {}
            
            model = self.models[model_name]
            
            # Берем случайную выборку векторов для анализа
            all_words = list(model.wv.key_to_index.keys())
            sample_size = min(300, len(all_words))
            if sample_size < 2:
                return {}

            sample_words = np.random.choice(all_words, sample_size, replace=False)

            # Получаем векторы и нормализуем их
            vectors = model.wv[sample_words]
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized_vectors = vectors / norms

            # Вычисляем матрицу косинусных сходств через матричное умножение
            similarity_matrix = np.matmul(normalized_vectors, normalized_vectors.T)
            similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

            # Получаем верхний треугольник без диагонали для распределения расстояний
            triu_indices = np.triu_indices_from(similarity_matrix, k=1)
            similarities_flat = similarity_matrix[triu_indices]
            distances = 1.0 - similarities_flat
            
            # Анализ распределения
            if distances:
                hist, bins = np.histogram(distances, bins=20)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                return {
                    'mean_distance': np.mean(distances),
                    'std_distance': np.std(distances),
                    'min_distance': np.min(distances),
                    'max_distance': np.max(distances),
                    'distance_distribution': {
                        'histogram': hist.tolist(),
                        'bin_centers': bin_centers.tolist(),
                        'bins': bins.tolist()
                    },
                    'similarity_matrix': similarity_matrix.tolist(),
                    'sample_words': sample_words.tolist()
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Ошибка в analyze_distance_distribution: {e}")
            return {}

    def cosine_similarity_analysis(self, model_name: str, word_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        6.1 Анализ косинусного сходства между векторами
        """
        try:
            if model_name not in self.models:
                return {}
            
            model = self.models[model_name]
            results = []
            
            for word1, word2 in word_pairs:
                try:
                    if word1 in model.wv and word2 in model.wv:
                        vec1 = model.wv[word1]
                        vec2 = model.wv[word2]
                        
                        # Косинусное сходство
                        cos_sim = cosine_similarity([vec1], [vec2])[0][0]
                        
                        # Евклидово расстояние
                        euclidean_dist = np.linalg.norm(vec1 - vec2)
                        
                        results.append({
                            'word_pair': f"{word1} - {word2}",
                            'cosine_similarity': cos_sim,
                            'euclidean_distance': euclidean_dist,
                            'relationship_type': self._classify_relationship(cos_sim)
                        })
                    else:
                        results.append({
                            'word_pair': f"{word1} - {word2}",
                            'cosine_similarity': None,
                            'euclidean_distance': None,
                            'relationship_type': "Слова отсутствуют в модели"
                        })
                        
                except Exception as e:
                    logger.warning(f"Ошибка для пары {word1}-{word2}: {e}")
                    continue
            
            # Анализ распределения расстояний
            similarities = [r['cosine_similarity'] for r in results if r['cosine_similarity'] is not None]
            
            if similarities:
                distribution_analysis = {
                    'mean_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities),
                    'min_similarity': np.min(similarities),
                    'max_similarity': np.max(similarities),
                }
            else:
                distribution_analysis = {}
            
            return {
                'pairwise_analysis': results,
                'distribution_analysis': distribution_analysis
            }
            
        except Exception as e:
            logger.error(f"Ошибка в cosine_similarity_analysis: {e}")
            return {}

    def categorical_analogy_evaluation(self, model_name: str) -> Dict[str, Any]:
        """
        6.2 Оценка векторных аналогий по категориям
        """
        analogy_categories = {
            'semantic_capitals': [
                (['Москва', 'Россия'], ['Париж'], 'Франция'),
                (['Берлин', 'Германия'], ['Рим'], 'Италия'),
                (['Лондон', 'Англия'], ['Мадрид'], 'Испания')
            ],
            'semantic_gender': [
                (['король', 'королева'], ['принц'], 'принцесса'),
                (['актер', 'актриса'], ['певец'], 'певица'),
                (['учитель', 'учительница'], ['доктор'], 'докторша')
            ],
            'syntactic_comparative': [
                (['хороший', 'лучше'], ['плохой'], 'хуже'),
                (['большой', 'больше'], ['маленький'], 'меньше'),
                (['сильный', 'сильнее'], ['слабый'], 'слабее')
            ],
            'morphological_verbs': [
                (['делать', 'сделал'], ['писать'], 'написал'),
                (['читать', 'прочитал'], ['смотреть'], 'посмотрел'),
                (['говорить', 'сказал'], ['думать'], 'подумал')
            ]
        }
        
        results = {}
        overall_correct = 0
        overall_total = 0
        
        for category_name, analogies in analogy_categories.items():
            correct = 0
            total = 0
            category_results = []
            
            for positive, negative, expected in analogies:
                try:
                    if model_name in self.models:
                        model = self.models[model_name]
                        all_words = positive + negative + [expected]
                        if all(word in model.wv for word in all_words):
                            results_list = model.wv.most_similar(
                                positive=positive, negative=negative, topn=5
                            )
                            top_words = [word for word, score in results_list]
                            
                            is_correct = expected in top_words
                            if is_correct:
                                correct += 1
                            
                            category_results.append({
                                'analogy': f"{positive[0]} − {positive[1]} + {negative[0]} = ?",
                                'expected': expected,
                                'predicted': top_words[0] if top_words else 'N/A',
                                'top_5': top_words,
                                'is_correct': is_correct
                            })
                            total += 1
                except Exception as e:
                    logger.warning(f"Ошибка аналогии {category_name}: {e}")
                    continue
            
            accuracy = correct / total if total > 0 else 0
            results[category_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'details': category_results
            }
            
            overall_correct += correct
            overall_total += total
        
        results['overall_accuracy'] = overall_correct / overall_total if overall_total > 0 else 0
        results['total_tests'] = overall_total
        results['total_correct'] = overall_correct
        
        return results

    def word_analogy(self, model_name: str, word1: str, word2: str, word3: str, topn: int = 5) -> List[Tuple[str, float]]:
        """
        Векторная аналогия: word1 : word2 = word3 : ?
        """
        try:
            if not self._validate_model_and_words(model_name, [word1, word2, word3]):
                return []
            
            model = self.models[model_name]
            results = model.wv.most_similar(positive=[word2, word3], negative=[word1], topn=topn)
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в векторной аналогии: {e}")
            return []

    def comprehensive_axes_analysis(self, model_name: str) -> Dict[str, Any]:
        """
        6.3 Расширенный анализ семантических осей и смещений
        """
        axes_definitions = {
            'gender_axis': {
                'positive': ['мужчина', 'отец', 'брат', 'дед', 'дядя'],
                'negative': ['женщина', 'мать', 'сестра', 'бабушка', 'тётя']
            },
            'profession_axis': {
                'positive': ['врач', 'учитель', 'инженер', 'программист', 'ученый'],
                'negative': ['пациент', 'ученик', 'клиент', 'пользователь', 'студент']
            },
            'evaluation_axis': {
                'positive': ['хороший', 'отличный', 'прекрасный', 'замечательный', 'великолепный'],
                'negative': ['плохой', 'ужасный', 'отвратительный', 'скверный', 'мерзкий']
            },
            'temporal_axis': {
                'positive': ['утро', 'день', 'рассвет', 'весна', 'лето'],
                'negative': ['ночь', 'вечер', 'закат', 'осень', 'зима']
            }
        }
        
        return self.semantic_axes_analysis(model_name, axes_definitions)

    def semantic_axes_analysis(self, model_name: str, axis_definitions: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """
        6.3 Анализ семантических осей и смещений
        """
        try:
            results = {}
            
            for axis_name, definition in axis_definitions.items():
                positive_words = definition.get('positive', [])
                negative_words = definition.get('negative', [])
                
                # Проверяем наличие слов
                all_words = positive_words + negative_words
                if not self._validate_model_and_words(model_name, all_words):
                    continue
                
                model = self.models[model_name]
                
                # Вычисляем направление оси
                positive_vectors = []
                negative_vectors = []
                
                for word in positive_words:
                    if word in model.wv:
                        positive_vectors.append(model.wv[word])
                
                for word in negative_words:
                    if word in model.wv:
                        negative_vectors.append(model.wv[word])
                
                if len(positive_vectors) == 0 or len(negative_vectors) == 0:
                    continue
                
                # Направление оси: среднее положительных - среднее отрицательных
                axis_direction = np.mean(positive_vectors, axis=0) - np.mean(negative_vectors, axis=0)
                axis_direction = axis_direction / np.linalg.norm(axis_direction)  # Нормализуем
                
                # Проецируем тестовые слова на ось
                test_words = list(model.wv.key_to_index.keys())[:200]  # Ограничиваем для производительности
                projections = {}
                
                for word in test_words:
                    if word not in all_words:  # Исключаем слова, определяющие ось
                        word_vector = model.wv[word]
                        projection = np.dot(word_vector, axis_direction)
                        projections[word] = projection
                
                # Сортируем по проекции
                sorted_projections = sorted(projections.items(), key=lambda x: x[1])
                
                results[axis_name] = {
                    'axis_direction': axis_direction,
                    'positive_end': sorted_projections[-10:],  # 10 слов на положительном конце
                    'negative_end': sorted_projections[:10],   # 10 слов на отрицательном конце
                    'bias_metric': self._calculate_bias_metric(sorted_projections),
                    'axis_strength': np.linalg.norm(axis_direction),
                    'all_projections': sorted_projections
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в semantic_axes_analysis: {e}")
            return {}

    def comprehensive_neighbors_analysis(self, model_name: str) -> Dict[str, Any]:
        """
        6.4 Расширенный анализ ближайших соседей
        """
        test_words = [
            'компьютер', 'программа', 'данные', 'город', 'хороший',
            'работа', 'время', 'человек', 'женщина', 'мужчина',
            'технология', 'информация', 'разработка', 'система'
        ]
        
        return self.nearest_neighbors_analysis(model_name, test_words, top_k=10)

    def nearest_neighbors_analysis(self, model_name: str, test_words: List[str], top_k: int = 10) -> Dict[str, Any]:
        """
        6.4 Качественный анализ ближайших соседей
        """
        try:
            if not self._validate_model_and_words(model_name, test_words):
                return {}
            
            model = self.models[model_name]
            results = {}
            semantic_coherence_scores = []
            neighbor_categories = defaultdict(list)
            
            for word in test_words:
                try:
                    if word in model.wv:
                        neighbors = model.wv.most_similar(word, topn=top_k)
                        
                        # Анализ семантической согласованности
                        coherence_score = self._calculate_semantic_coherence(model, word, neighbors)
                        semantic_coherence_scores.append(coherence_score)
                        
                        # Анализ типов соседей
                        neighbor_types = self._analyze_neighbor_types(word, neighbors)
                        
                        results[word] = {
                            'neighbors': neighbors,
                            'semantic_coherence': coherence_score,
                            'average_similarity': np.mean([sim for _, sim in neighbors]),
                            'neighbor_types': neighbor_types,
                            'status': 'success'
                        }
                        
                        # Собираем статистику по категориям
                        for neighbor_type, count in neighbor_types.items():
                            neighbor_categories[neighbor_type].append(count)
                    else:
                        results[word] = {
                            'neighbors': [],
                            'semantic_coherence': 0.0,
                            'average_similarity': 0.0,
                            'neighbor_types': {},
                            'status': 'Слово отсутствует в модели'
                        }
                        
                except Exception as e:
                    logger.warning(f"Ошибка анализа соседей для {word}: {e}")
                    continue
            
            # Анализ категорий соседей
            category_analysis = {}
            for category, counts in neighbor_categories.items():
                category_analysis[category] = {
                    'mean_count': np.mean(counts),
                    'total_occurrences': sum(counts)
                }
            
            overall_analysis = {
                'mean_semantic_coherence': np.mean(semantic_coherence_scores) if semantic_coherence_scores else 0,
                'semantic_coherence_std': np.std(semantic_coherence_scores) if semantic_coherence_scores else 0,
                'total_words_analyzed': len([w for w in test_words if w in model.wv]),
                'neighbor_category_analysis': category_analysis
            }
            
            return {
                'word_analysis': results,
                'overall_analysis': overall_analysis
            }
            
        except Exception as e:
            logger.error(f"Ошибка в nearest_neighbors_analysis: {e}")
            return {}

    def _analyze_neighbor_types(self, target_word: str, neighbors: List[Tuple[str, float]]) -> Dict[str, int]:
        """Анализ типов семантических отношений у соседей"""
        neighbor_types = {
            'semantic_synonyms': 0,      # Семантические синонимы
            'morphological_variants': 0, # Морфологические варианты
            'thematic_related': 0,       # Тематически связанные
            'syntactic_related': 0,      # Синтаксически связанные
            'other': 0                   # Другие
        }
        
        for neighbor_word, similarity in neighbors:
            # Проверяем морфологическую связь
            if (neighbor_word in target_word or target_word in neighbor_word or
                len(set(neighbor_word) & set(target_word)) > 3):
                neighbor_types['morphological_variants'] += 1
            # Проверяем синтаксические отношения (например, степени сравнения)
            elif self._check_syntactic_relation(target_word, neighbor_word):
                neighbor_types['syntactic_related'] += 1
            # Проверяем семантическую близость (высокое сходство)
            elif similarity > 0.6:
                neighbor_types['semantic_synonyms'] += 1
            # Проверяем тематическую связь
            elif self._check_thematic_relation(target_word, neighbor_word):
                neighbor_types['thematic_related'] += 1
            else:
                neighbor_types['other'] += 1
        
        return neighbor_types

    def _check_thematic_relation(self, word1: str, word2: str) -> bool:
        """Проверка тематической связи между словами"""
        # Простые эвристики для русского языка
        thematic_groups = {
            'tech': ['компьютер', 'программа', 'данные', 'система', 'информация', 'технология'],
            'city': ['город', 'улица', 'дом', 'площадь', 'здание', 'парк'],
            'emotion': ['хороший', 'плохой', 'радость', 'грусть', 'любовь', 'ненависть'],
            'profession': ['врач', 'учитель', 'инженер', 'программист', 'ученый'],
            'family': ['мужчина', 'женщина', 'отец', 'мать', 'брат', 'сестра']
        }
        
        for group, words in thematic_groups.items():
            if word1 in words and word2 in words:
                return True
        return False

    def _check_syntactic_relation(self, word1: str, word2: str) -> bool:
        """Эвристическая проверка синтаксических отношений (степени сравнения и др.)"""
        syntactic_map = {
            'хороший': ['лучше', 'лучший', 'наилучший'],
            'плохой': ['хуже', 'худший', 'наихудший'],
            'большой': ['больше', 'больший', 'наибольший'],
            'маленький': ['меньше', 'меньший', 'наименьший'],
            'быстрый': ['быстрее', 'быстрей', 'скорее'],
            'медленный': ['медленнее', 'медленней', 'помедленнее'],
            'сильный': ['сильнее', 'посильнее'],
            'умный': ['умнее', 'поумнее'],
            'дорогой': ['дороже', 'подороже'],
            'дешёвый': ['дешевле', 'подешевле']
        }

        if word2 in syntactic_map.get(word1, []) or word1 in syntactic_map.get(word2, []):
            return True

        base_suffixes = ('ый', 'ий', 'ой', 'ая', 'ое')
        comparative_suffixes = ('ее', 'ей', 'айший', 'ейший', 'шая', 'шее')

        def _normalize(w: str) -> str:
            for suf in base_suffixes:
                if w.endswith(suf) and len(w) > len(suf) + 1:
                    return w[:-len(suf)]
            return w

        def _is_comparative_form(base: str, candidate: str) -> bool:
            base_root = _normalize(base)
            for suf in comparative_suffixes:
                if candidate.endswith(suf):
                    return candidate.startswith(base_root[:max(1, len(base_root) - 1)])
            return False

        if _is_comparative_form(word1, word2) or _is_comparative_form(word2, word1):
            return True

        return False

    def project_words_on_axis(self, model_name: str, axis_direction, words: List[str]) -> Tuple[List[Tuple[str, float]], List[str]]:
        """Проецирование произвольных слов на выбранную ось"""
        projections = []
        missing = []

        if model_name not in self.models or axis_direction is None or words is None:
            return projections, words or []

        model = self.models[model_name]
        axis_vector = np.array(axis_direction)
        axis_norm = np.linalg.norm(axis_vector)
        if axis_norm == 0:
            return projections, words
        axis_vector = axis_vector / axis_norm

        for word in words:
            if word in model.wv:
                value = float(np.dot(model.wv[word], axis_vector))
                projections.append((word, value))
            else:
                missing.append(word)

        return projections, missing

    def _calculate_semantic_coherence(self, model, target_word: str, neighbors: List[Tuple[str, float]]) -> float:
        """Вычисление семантической согласованности соседей"""
        try:
            if len(neighbors) < 2:
                return 0.0
            
            # Вычисляем попарное сходство между соседями
            neighbor_words = [neighbor[0] for neighbor in neighbors]
            neighbor_vectors = []
            
            for word in neighbor_words:
                if word in model.wv:
                    neighbor_vectors.append(model.wv[word])
            
            if len(neighbor_vectors) < 2:
                return 0.0
                
            neighbor_vectors = np.array(neighbor_vectors)
            similarity_matrix = cosine_similarity(neighbor_vectors)
            np.fill_diagonal(similarity_matrix, 0)  # Исключаем диагональ
            
            # Среднее попарное сходство
            n = len(neighbor_vectors)
            total_pairs = n * (n - 1)
            if total_pairs > 0:
                coherence = np.sum(similarity_matrix) / total_pairs
                return coherence
            return 0.0
            
        except Exception as e:
            logger.warning(f"Ошибка вычисления согласованности: {e}")
            return 0.0

    def _calculate_bias_metric(self, projections: List[Tuple[str, float]]) -> float:
        """Вычисление метрики смещения"""
        if not projections:
            return 0.0
        
        values = [proj[1] for proj in projections]
        # Мера асимметрии распределения
        skewness = np.mean((values - np.mean(values)) ** 3) / (np.std(values) ** 3)
        return abs(skewness)

    def _classify_relationship(self, similarity: float) -> str:
        """Классификация типа семантического отношения"""
        if similarity > 0.7:
            return "Сильные синонимы/Очень близкие"
        elif similarity > 0.4:
            return "Тематически близкие"
        elif similarity > 0.1:
            return "Слабая связь"
        elif similarity > -0.1:
            return "Нейтральные"
        else:
            return "Антонимы/Противоположные"

    def _calculate_overall_quality(self, evaluation_results: Dict[str, Any]) -> float:
        """Вычисление общей оценки качества"""
        weights = {
            'distance_analysis': 0.2,
            'analogy_analysis': 0.3,
            'semantic_axes': 0.25,
            'neighbors_analysis': 0.25
        }
        
        total_score = 0
        total_weight = 0
        
        # Оценка распределения расстояний
        if evaluation_results.get('distance_analysis'):
            dist_analysis = evaluation_results['distance_analysis']
            if dist_analysis.get('mean_distance', 0) > 0:
                # Идеальное среднее расстояние ~0.5-0.7
                mean_dist = dist_analysis['mean_distance']
                distance_score = 1 - abs(mean_dist - 0.6) / 0.6
                total_score += distance_score * weights['distance_analysis']
                total_weight += weights['distance_analysis']
        
        # Оценка аналогий
        if evaluation_results.get('analogy_analysis'):
            analogy_acc = evaluation_results['analogy_analysis'].get('overall_accuracy', 0)
            total_score += analogy_acc * weights['analogy_analysis']
            total_weight += weights['analogy_analysis']
        
        # Оценка семантических осей
        if evaluation_results.get('semantic_axes'):
            axes_count = len(evaluation_results['semantic_axes'])
            axes_score = min(axes_count / 4, 1.0)  # Максимум 4 оси
            total_score += axes_score * weights['semantic_axes']
            total_weight += weights['semantic_axes']
        
        # Оценка соседей
        if evaluation_results.get('neighbors_analysis'):
            neighbors_analysis = evaluation_results['neighbors_analysis']
            coherence = neighbors_analysis.get('overall_analysis', {}).get('mean_semantic_coherence', 0)
            total_score += coherence * weights['neighbors_analysis']
            total_weight += weights['neighbors_analysis']
        
        return total_score / total_weight if total_weight > 0 else 0.0

    def get_model_statistics(self, model_name: str) -> Dict[str, Any]:
        """Получение статистики модели"""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        stats = {
            'model_type': 'word2vec' if 'word2vec' in model_name else 'fasttext' if 'fasttext' in model_name else 'doc2vec',
            'vocabulary_size': len(model.wv.key_to_index) if hasattr(model, 'wv') else 0,
            'vector_size': model.vector_size if hasattr(model, 'vector_size') else 0,
        }
        
        if hasattr(model, 'window'):
            stats['window_size'] = model.window
        
        return stats

    def get_vocabulary_coverage(self, model_name: str, test_words: List[str]) -> Dict[str, Any]:
        """Проверка покрытия словаря модели"""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        found_words = [word for word in test_words if word in model.wv]
        
        return {
            'total_test_words': len(test_words),
            'found_words': len(found_words),
            'coverage_percentage': (len(found_words) / len(test_words)) * 100,
            'missing_words': [word for word in test_words if word not in model.wv]
        }


def get_russian_test_sets() -> Dict[str, Any]:
    """
    Стандартные тестовые наборы для русского языка
    
    Returns:
        Словарь с тестовыми наборами
    """
    return {
        'semantic_analogies': [
            ('Москва', 'Россия', 'Париж', 'Франция'),
            ('король', 'королева', 'мужчина', 'женщина'),
            ('собака', 'щенок', 'кошка', 'котенок'),
            ('быстро', 'быстрее', 'медленно', 'медленнее'),
            ('солнце', 'день', 'луна', 'ночь')
        ],
        'common_words': [
            'компьютер', 'программа', 'данные', 'система', 'информация',
            'технология', 'разработка', 'алгоритм', 'сеть', 'база'
        ],
        'semantic_relationships': [
            ('город', 'столица'),
            ('учеба', 'образование'),
            ('работа', 'профессия'),
            ('здоровье', 'медицина'),
            ('деньги', 'финансы')
        ],
        'semantic_axes': {
            'gender': ['мужчина', 'женщина', 'отец', 'мать'],
            'profession': ['врач', 'учитель', 'инженер', 'рабочий'],
            'evaluation': ['хороший', 'плохой', 'отличный', 'ужасный']
        }
    }


def analyze_model_quality(models: Dict[str, Any], test_words: List[str] = None) -> pd.DataFrame:
    """
    Анализ качества нескольких моделей
    
    Args:
        models: словарь моделей
        test_words: тестовые слова для проверки покрытия
        
    Returns:
        DataFrame с результатами анализа
    """
    if test_words is None:
        test_words = get_russian_test_sets()['common_words']
    
    semantic_ops = SemanticOperations(models)
    results = []
    
    for model_name in models:
        try:
            # Комплексная оценка
            evaluation = semantic_ops.comprehensive_semantic_evaluation(model_name)
            stats = semantic_ops.get_model_statistics(model_name)
            coverage = semantic_ops.get_vocabulary_coverage(model_name, test_words)
            
            if stats and coverage:
                results.append({
                    'Модель': model_name,
                    'Тип модели': stats.get('model_type', 'N/A'),
                    'Размер словаря': stats.get('vocabulary_size', 0),
                    'Размерность': stats.get('vector_size', 0),
                    'Покрытие теста': f"{coverage.get('coverage_percentage', 0):.1f}%",
                    'Точность аналогий': f"{evaluation.get('analogy_analysis', {}).get('overall_accuracy', 0)*100:.1f}%",
                    'Сем. согласованность': f"{evaluation.get('neighbors_analysis', {}).get('overall_analysis', {}).get('mean_semantic_coherence', 0):.3f}",
                    'Общая оценка': f"{evaluation.get('quality_score', 0)*100:.1f}%"
                })
        except Exception as e:
            print(f"Ошибка при анализе модели {model_name}: {e}")
            continue
    
    return pd.DataFrame(results)