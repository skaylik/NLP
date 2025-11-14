# training_models.py (УПРОЩЕННАЯ ВЕРСИЯ ТОЛЬКО С РАСШИРЕННЫМИ НАСТРОЙКАМИ)
import time
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import psutil
import gc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class DistributedRepresentations:
    def __init__(self):
        self.word_models = {}
        self.doc_models = {}
        self.evaluation_results = {}
        self.training_history = []
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('model_training')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def train_with_parameters(self, texts: List[List[str]], categories: List[str] = None, 
                            model_types: List[str] = None, vector_size: int = 100,
                            window: int = 8, min_count: int = 2, epochs: int = 100,
                            sg: int = 1, workers: Optional[int] = None, hs: int = 0, 
                            negative: int = 10, sample: float = 1e-5,
                            compute_loss: bool = False, max_epochs: Optional[int] = 150):
        """
        ОБУЧЕНИЕ С ВЫБОРОМ ПАРАМЕТРОВ ПОЛЬЗОВАТЕЛЕМ
        Автоматическая адаптация параметров для маленького корпуса
        """
        valid_texts, total_words, unique_words = self.validate_corpus(texts)

        if workers is None:
            workers = max(1, multiprocessing.cpu_count() - 1)
            self.logger.info(f"🧵 Автовыбор количества потоков: {workers}")
        elif workers < 1:
            self.logger.warning(f"⚠️ Некорректное значение workers={workers}, используем 1")
            workers = 1
        else:
            cpu_total = multiprocessing.cpu_count()
            if workers > cpu_total:
                self.logger.info(f"🧵 Ограничение workers: {workers} → {cpu_total} (максимум доступных ядер)")
                workers = cpu_total

        if max_epochs is not None:
            max_epochs = max(max_epochs, epochs)
        
        # АВТОМАТИЧЕСКАЯ АДАПТАЦИЯ ПАРАМЕТРОВ ДЛЯ МАЛЕНЬКОГО КОРПУСА
        corpus_size = len(valid_texts)
        is_small_corpus = corpus_size < 1000 or total_words < 50000
        
        if is_small_corpus:
            self.logger.warning(f"⚠️ Обнаружен маленький корпус ({corpus_size} документов, {total_words} слов)")
            self.logger.info("🔧 Автоматическая адаптация параметров для маленького корпуса...")
            
            # Для маленького корпуса:
            # - min_count должен быть 1, чтобы не терять редкие слова
            if min_count > 1:
                original_min_count = min_count
                min_count = 1
                self.logger.info(f"  → min_count: {original_min_count} → {min_count} (сохраняем редкие слова)")
            
            # - window должен быть меньше (3-5 вместо 8)
            # Для очень маленького корпуса используем еще меньшее окно
            if corpus_size < 600:
                target_window = 3
            else:
                target_window = 5
            
            if window > target_window:
                original_window = window
                window = target_window
                self.logger.info(f"  → window: {original_window} → {window} (меньше окно для маленького корпуса)")
            
            # - sample должен быть оптимальным для downsampling частых слов
            # Для очень маленького корпуса используем умеренный sample
            if corpus_size < 600:
                target_sample = 1e-3  # 0.001 для очень маленького корпуса (меньше даунсемплинг)
            else:
                target_sample = 1e-4  # 0.0001 для обычного маленького корпуса
            
            if sample < target_sample or sample > 1e-2:
                original_sample = sample
                sample = target_sample
                self.logger.info(f"  → sample: {original_sample} → {sample} (оптимальный downsampling для маленького корпуса)")
            
            # - epochs можно увеличить для лучшего обучения
            # Для очень маленького корпуса нужно еще больше эпох
            if corpus_size < 600:
                target_epochs = 300
            else:
                target_epochs = 200
            
            if epochs < target_epochs:
                adjusted_epochs = max(target_epochs, epochs * 2)
            else:
                adjusted_epochs = epochs

            if max_epochs is not None and adjusted_epochs > max_epochs:
                self.logger.info(f"  → epochs: {adjusted_epochs} → {max_epochs} (ограничение максимума эпох)")
                adjusted_epochs = max_epochs

            if epochs != adjusted_epochs:
                self.logger.info(f"  → epochs: {epochs} → {adjusted_epochs} (обновлено после адаптации)")
            epochs = adjusted_epochs
            
            # - Оптимизируем negative sampling для маленького корпуса
            # Для маленького корпуса лучше использовать меньше negative samples (5-10 вместо 10-25)
            if corpus_size < 600 and hs == 0:
                if negative > 10:
                    original_negative = negative
                    negative = 10  # Оптимальное значение для маленького корпуса
                    self.logger.info(f"  → negative: {original_negative} → {negative} (оптимизировано для маленького корпуса)")
                elif negative < 5:
                    original_negative = negative
                    negative = 5  # Минимум для стабильности
                    self.logger.info(f"  → negative: {original_negative} → {negative} (установлен минимум)")
            
            # - Для очень маленького корпуса можно использовать hierarchical softmax
            # Но negative sampling тоже хорошо работает, поэтому оставляем выбор пользователю
            # Автоматически переключаем только если корпус очень маленький (<300 документов)
            if corpus_size < 300 and hs == 0 and negative > 5:
                original_hs = hs
                original_negative = negative
                hs = 1
                negative = 0  # hs=1 несовместим с negative sampling
                self.logger.info(f"  → hs: {original_hs} → {hs}, negative: {original_negative} → {negative} (hierarchical softmax для очень маленького корпуса)")
            
            # - Оптимизируем размерность векторов для маленького корпуса
            # Не изменяем vector_size если он был явно указан как 50 (популярный выбор)
            # Увеличиваем только если размерность слишком маленькая (<50)
            if corpus_size < 600:
                if vector_size < 50:
                    # Если размерность слишком маленькая (<50), увеличиваем до минимума 50
                    original_vector_size = vector_size
                    vector_size = 50
                    self.logger.info(f"  → vector_size: {original_vector_size} → {vector_size} (увеличена до минимума)")
                # Для vector_size >= 50 оставляем как есть (пользователь явно указал)
            elif vector_size > 200:
                # Для большего корпуса ограничиваем максимальную размерность
                original_vector_size = vector_size
                vector_size = 200
                self.logger.info(f"  → vector_size: {original_vector_size} → {vector_size} (ограничена размерность)")
        
        self.logger.info("🚀 ЗАПУСК ОБУЧЕНИЯ С ВЫБРАННЫМИ ПАРАМЕТРАМИ...")
        self.logger.info(f"📊 Корпус: {len(valid_texts)} документов, {total_words} слов, {unique_words} уникальных слов")
        self.logger.info(
            f"⚙️ Параметры: size={vector_size}, window={window}, min_count={min_count}, "
            f"epochs={epochs}, sample={sample:.2e}, hs={hs}, negative={negative}, "
            f"workers={workers}, compute_loss={compute_loss}"
        )
        
        if model_types is None:
            model_types = ['word2vec_skipgram', 'fasttext_skipgram']
        
        models_created = 0
        start_time = time.time()
        
        # WORD2VEC МОДЕЛИ
        if 'word2vec_skipgram' in model_types or 'word2vec_cbow' in model_types:
            self.logger.info("🎯 Обучение Word2Vec моделей...")
            
            word2vec_configs = []
            if 'word2vec_skipgram' in model_types:
                word2vec_configs.append({'sg': 1, 'name': 'word2vec_skipgram'})
            if 'word2vec_cbow' in model_types:
                word2vec_configs.append({'sg': 0, 'name': 'word2vec_cbow'})
            
            for config in word2vec_configs:
                try:
                    model_name = f"{config['name']}_vs{vector_size}_w{window}_mc{min_count}_e{epochs}"
                    self.logger.info(f"🔧 Обучение {model_name}...")
                    
                    model_start = time.time()
                    model = Word2Vec(
                        sentences=valid_texts,
                        vector_size=vector_size,
                        window=window,
                        min_count=min_count,
                        sg=config['sg'],
                        workers=workers,
                        epochs=epochs,
                        hs=hs,
                        negative=negative,
                        sample=sample,
                        seed=42,
                        compute_loss=compute_loss
                    )
                    
                    training_time = time.time() - model_start
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    self.word_models[model_name] = model
                    models_created += 1
                    
                    self.training_history.append({
                        'model_name': model_name,
                        'training_time': training_time,
                        'memory_usage': memory_usage,
                        'vocab_size': len(model.wv.key_to_index),
                        'vector_size': vector_size,
                        'window': window,
                        'min_count': min_count,
                        'epochs': epochs,
                        'architecture': 'Skip-gram' if config['sg'] == 1 else 'CBOW',
                        'compute_loss': compute_loss,
                        'workers': workers
                    })
                    
                    vocab_size = len(model.wv.key_to_index)
                    self.logger.info(f"✅ {model_name} обучен за {training_time:.1f}с! Словарь: {vocab_size} слов")
                    
                    # Проверка наличия важных слов в словаре модели
                    important_words = ['компьютер', 'ноутбук', 'данные', 'информация', 'программа', 
                                      'алгоритм', 'город', 'река', 'система', 'технология']
                    missing_in_model = [w for w in important_words if w not in model.wv]
                    if missing_in_model:
                        self.logger.warning(f"  ⚠️ Отсутствуют в словаре модели: {', '.join(missing_in_model)}")
                    else:
                        self.logger.info(f"  ✅ Все важные слова присутствуют в словаре")
                    
                except Exception as e:
                    self.logger.error(f"❌ Ошибка {model_name}: {e}")
        
        # FASTTEXT МОДЕЛИ
        if 'fasttext_skipgram' in model_types or 'fasttext_cbow' in model_types:
            self.logger.info("🎯 Обучение FastText моделей...")
            
            fasttext_configs = []
            if 'fasttext_skipgram' in model_types:
                fasttext_configs.append({'sg': 1, 'name': 'fasttext_skipgram'})
            if 'fasttext_cbow' in model_types:
                fasttext_configs.append({'sg': 0, 'name': 'fasttext_cbow'})
            
            for config in fasttext_configs:
                try:
                    model_name = f"{config['name']}_vs{vector_size}_w{window}_mc{min_count}_e{epochs}"
                    self.logger.info(f"🔧 Обучение {model_name}...")
                    
                    model_start = time.time()
                    model = FastText(
                        sentences=valid_texts,
                        vector_size=vector_size,
                        window=window,
                        min_count=min_count,
                        sg=config['sg'],
                        workers=workers,
                        epochs=epochs,
                        hs=hs,
                        negative=negative,
                        sample=sample,  # Добавляем параметр sample для FastText
                        seed=42
                    )
                    
                    training_time = time.time() - model_start
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    self.word_models[model_name] = model
                    models_created += 1
                    
                    self.training_history.append({
                        'model_name': model_name,
                        'training_time': training_time,
                        'memory_usage': memory_usage,
                        'vocab_size': len(model.wv.key_to_index),
                        'vector_size': vector_size,
                        'window': window,
                        'min_count': min_count,
                        'epochs': epochs,
                        'architecture': 'FastText Skip-gram' if config['sg'] == 1 else 'FastText CBOW',
                        'workers': workers,
                        'compute_loss': False
                    })
                    
                    vocab_size = len(model.wv.key_to_index)
                    self.logger.info(f"✅ {model_name} обучен за {training_time:.1f}с! Словарь: {vocab_size} слов")
                    
                    # Проверка наличия важных слов в словаре модели
                    important_words = ['компьютер', 'ноутбук', 'данные', 'информация', 'программа', 
                                      'алгоритм', 'город', 'река', 'система', 'технология']
                    missing_in_model = [w for w in important_words if w not in model.wv]
                    if missing_in_model:
                        self.logger.warning(f"  ⚠️ Отсутствуют в словаре модели: {', '.join(missing_in_model)}")
                    else:
                        self.logger.info(f"  ✅ Все важные слова присутствуют в словаре")
                    
                except Exception as e:
                    self.logger.error(f"❌ Ошибка {model_name}: {e}")
        
        # DOC2VEC МОДЕЛИ
        if 'doc2vec' in model_types and categories and len(categories) == len(valid_texts):
            try:
                self.logger.info("🎯 Обучение Doc2Vec моделей...")
                tagged_documents = [
                    TaggedDocument(words=text, tags=[str(i), categories[i]]) 
                    for i, text in enumerate(valid_texts)
                ]
                
                doc2vec_configs = [
                    {'dm': 1, 'name': 'doc2vec_pv-dm'},
                    {'dm': 0, 'name': 'doc2vec_pv-dbow'}
                ]
                
                for config in doc2vec_configs:
                    try:
                        model_name = f"{config['name']}_vs{vector_size}_w{window}_mc{min_count}"
                        self.logger.info(f"🔧 Обучение {model_name}...")
                        
                        model_start = time.time()
                        model = Doc2Vec(
                            documents=tagged_documents,
                            vector_size=vector_size,
                            window=window,
                            min_count=min_count,
                            dm=config['dm'],
                            workers=workers,
                            epochs=epochs,
                            seed=42
                        )
                        
                        training_time = time.time() - model_start
                        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                        
                        self.doc_models[model_name] = model
                        models_created += 1
                        
                        self.training_history.append({
                            'model_name': model_name,
                            'training_time': training_time,
                            'memory_usage': memory_usage,
                            'vector_size': vector_size,
                            'window': window,
                            'architecture': 'PV-DM' if config['dm'] == 1 else 'PV-DBOW',
                            'epochs': epochs,
                            'workers': workers
                        })
                        
                        self.logger.info(f"✅ {model_name} обучен за {training_time:.1f}с!")
                        
                    except Exception as e:
                        self.logger.error(f"❌ Ошибка {model_name}: {e}")
            except Exception as e:
                self.logger.error(f"❌ Ошибка Doc2Vec: {e}")
        
        total_time = time.time() - start_time
        self.logger.info(f"🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО! Создано {models_created} моделей за {total_time:.1f} секунд")
        
        return models_created

    def evaluate_models_comprehensive(self, test_words: List[str] = None):
        """
        КОМПЛЕКСНАЯ ОЦЕНКА МОДЕЛЕЙ
        """
        if test_words is None:
            test_words = [
                'компьютер', 'программа', 'данные', 'система', 'информация',
                'технология', 'разработка', 'алгоритм', 'сеть', 'база'
            ]
        
        evaluation_results = {}
        
        for model_name, model in self.word_models.items():
            try:
                metrics = {}
                
                # 1. Размер словаря и покрытие
                vocab_size = len(model.wv.key_to_index)
                coverage = self._calculate_vocabulary_coverage(model, test_words)
                
                metrics['vocabulary_size'] = vocab_size
                metrics['test_coverage'] = coverage['coverage_percentage']
                metrics['oov_rate'] = 100 - coverage['coverage_percentage']
                
                # 2. Word analogy accuracy
                analogy_accuracy = self._evaluate_analogies(model)
                metrics['analogy_accuracy'] = analogy_accuracy
                
                # 3. Семантическое сходство
                similarity_score = self._evaluate_semantic_similarity(model)
                metrics['semantic_similarity_score'] = similarity_score
                
                # 4. Морфологическая устойчивость (только для FastText)
                if 'fasttext' in model_name:
                    morphology_score = self._evaluate_morphological_robustness(model)
                    metrics['morphology_score'] = morphology_score
                
                # 5. Время и память
                training_info = next((item for item in self.training_history if item['model_name'] == model_name), {})
                metrics['training_time'] = training_info.get('training_time', 0)
                metrics['memory_usage'] = training_info.get('memory_usage', 0)
                
                evaluation_results[model_name] = metrics
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка оценки {model_name}: {e}")
                continue
        
        self.evaluation_results = evaluation_results
        return evaluation_results

    def _evaluate_analogies(self, model) -> float:
        """Оценка точности аналогий для русского языка"""
        analogy_tests = [
            (['Москва', 'Россия'], ['Париж'], 'Франция'),
            (['король', 'королева'], ['мужчина'], 'женщина'),
            (['собака', 'щенок'], ['кошка'], 'котенок'),
            (['хороший', 'лучше'], ['плохой'], 'хуже'),
            (['большой', 'больше'], ['маленький'], 'меньше'),
            (['делать', 'сделал'], ['писать'], 'написал'),
        ]
        
        correct = 0
        total = 0
        
        for positive, negative, expected in analogy_tests:
            try:
                all_words = positive + negative + [expected]
                if all(word in model.wv for word in all_words):
                    results = model.wv.most_similar(positive=positive, negative=negative, topn=3)
                    top_words = [word for word, score in results]
                    if expected in top_words:
                        correct += 1
                    total += 1
            except:
                continue
        
        return correct / total if total > 0 else 0.0

    def _evaluate_morphological_robustness(self, model) -> float:
        """Оценка морфологической устойчивости FastText"""
        test_words = ['компьютер', 'программа', 'данные']
        variations_score = 0
        
        for word in test_words:
            try:
                variations = [
                    word + 'ы',
                    word[:-1] if len(word) > 3 else word,
                    word + 'ный'
                ]
                
                if word in model.wv:
                    neighbors = [w for w, s in model.wv.most_similar(word, topn=10)]
                    found_variations = sum(1 for var in variations if var in neighbors)
                    variations_score += found_variations / len(variations)
                    
            except:
                continue
        
        return variations_score / len(test_words) if test_words else 0.0

    def _evaluate_semantic_similarity(self, model) -> float:
        """Оценка семантического сходства"""
        test_pairs = [
            ('компьютер', 'ноутбук'),
            ('данные', 'информация'),
            ('программа', 'алгоритм'),
            ('город', 'река')
        ]
        
        similarities = []
        for word1, word2 in test_pairs:
            if word1 in model.wv and word2 in model.wv:
                try:
                    similarity = model.wv.similarity(word1, word2)
                    similarities.append(similarity)
                except:
                    continue
        
        return np.mean(similarities) if similarities else 0.0

    def _calculate_vocabulary_coverage(self, model, test_words: List[str]) -> Dict[str, Any]:
        """Расчет покрытия словаря"""
        found_words = [word for word in test_words if word in model.wv]
        
        return {
            'total_test_words': len(test_words),
            'found_words': len(found_words),
            'coverage_percentage': (len(found_words) / len(test_words)) * 100,
            'missing_words': [word for word in test_words if word not in model.wv]
        }

    def validate_corpus(self, texts: List[List[str]]) -> Tuple[List[List[str]], int, int]:
        """Валидация корпуса"""
        if not texts or len(texts) == 0:
            raise ValueError("Пустой корпус для обучения")
        
        valid_texts = [text for text in texts if text and len(text) > 0]
        if len(valid_texts) == 0:
            raise ValueError("Все тексты пустые после фильтрации")
        
        total_words = sum(len(text) for text in valid_texts)
        all_words = [word for text in valid_texts for word in text]
        unique_words = len(set(all_words))
        
        # Подсчет частоты слов
        from collections import Counter
        word_freq = Counter(all_words)
        
        # Важные слова для проверки
        important_words = ['компьютер', 'ноутбук', 'данные', 'информация', 'программа', 
                          'алгоритм', 'город', 'река', 'система', 'технология']
        
        self.logger.info(f"📊 Анализ корпуса:")
        self.logger.info(f"- Документов: {len(valid_texts)}")
        self.logger.info(f"- Всего слов: {total_words}")
        self.logger.info(f"- Уникальных слов: {unique_words}")
        
        # Проверка наличия важных слов
        self.logger.info(f"🔍 Проверка важных слов в корпусе:")
        for word in important_words:
            freq = word_freq.get(word, 0)
            if freq > 0:
                self.logger.info(f"  ✅ '{word}': {freq} раз")
            else:
                self.logger.warning(f"  ❌ '{word}': отсутствует в корпусе")
        
        # Статистика по частотам
        words_with_freq_1 = sum(1 for freq in word_freq.values() if freq == 1)
        words_with_freq_2 = sum(1 for freq in word_freq.values() if freq == 2)
        
        self.logger.info(f"📈 Статистика частот:")
        self.logger.info(f"  - Слов с частотой 1: {words_with_freq_1} ({words_with_freq_1/unique_words*100:.1f}%)")
        self.logger.info(f"  - Слов с частотой 2: {words_with_freq_2} ({words_with_freq_2/unique_words*100:.1f}%)")
        self.logger.info(f"  ⚠️ С min_count=2 будет потеряно {words_with_freq_1} слов ({words_with_freq_1/unique_words*100:.1f}%)")
        
        return valid_texts, total_words, unique_words

    def get_available_models(self) -> Dict[str, Any]:
        """Получение всех доступных моделей"""
        all_models = {}
        all_models.update(self.word_models)
        all_models.update(self.doc_models)
        return all_models

    def get_training_history_df(self) -> pd.DataFrame:
        """Получение истории обучения в виде DataFrame"""
        return pd.DataFrame(self.training_history)

    def get_evaluation_results_df(self) -> pd.DataFrame:
        """Получение результатов оценки в виде DataFrame"""
        return pd.DataFrame.from_dict(self.evaluation_results, orient='index')