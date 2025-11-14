# preparation_corpus.py
import json
import re
import logging
from collections import defaultdict
import os
from datetime import datetime
import hashlib

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    import pymorphy3
except ImportError:
    pymorphy3 = None

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_processing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ProcessingConfig:
    """Конфигурация параметров обработки"""
    MIN_TEXT_LENGTH = 50
    MIN_WORDS_AFTER_PROCESSING = 5
    MAX_DOCUMENT_WORDS = 5000
    MAX_DOCUMENT_CHARS = 100000
    LEXICAL_DIVERSITY_THRESHOLD = 0.3
    TARGET_WORDS = 100000

class TextProcessor:
    def __init__(self, enable_logging=False):
        """
        Инициализация процессора текста
        
        Args:
            enable_logging: включить логирование в консоль
        """
        if enable_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
            
        self._download_nltk_resources()
        
        # Инициализация морфологического анализатора
        if pymorphy3:
            try:
                self.morph = pymorphy3.MorphAnalyzer()
                self.lemmatization_enabled = True
                logger.info("pymorphy3 успешно инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации pymorphy3: {e}")
                self.morph = None
                self.lemmatization_enabled = False
        else:
            self.morph = None
            self.lemmatization_enabled = False
            logger.warning("pymorphy3 не доступен, будет использована только токенизация")
            
        # Кэш для лемматизации
        self.lemmatize_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
            
        # Стоп-слова
        try:
            nltk_stopwords = set(stopwords.words('russian'))
        except LookupError:
            nltk_stopwords = set()
            
        self.stop_words = {
            'и', 'в', 'не', 'на', 'с', 'по', 'к', 'у', 'о', 'от', 'до', 'для', 
            'за', 'из', 'или', 'же', 'но', 'бы', 'ли', 'то', 'это', 'как', 'так',
            'вот', 'что', 'чтоб', 'чтобы', 'ну', 'вы', 'бы', 'что', 'кто', 'он',
            'она', 'они', 'оно', 'мы', 'вы', 'меня', 'тебя', 'его', 'ее', 'их',
            'мне', 'тебе', 'ему', 'ей', 'нам', 'вам', 'ими', 'мой', 'твой', 'свой',
            'наш', 'ваш', 'её', 'его', 'их', 'под', 'при', 'об', 'со', 'из-за',
            'по-над', 'без', 'между', 'ради', 'сквозь', 'среди', 'благодаря',
            'ведь', 'либо', 'нибудь', 'какой', 'какая', 'какое', 'какие',
            'чей', 'чья', 'чьё', 'чьи', 'сколько', 'который', 'котора', 'которое',
            'этот', 'эта', 'это', 'эти', 'тот', 'та', 'то', 'те', 'сам', 'сама',
            'само', 'сами', 'весь', 'вся', 'всё', 'все', 'самый', 'самая', 'самое'
        }
        self.stop_words.update(nltk_stopwords)

        # Регулярные выражения для фильтрации
        self.re_html_tags = re.compile(r'<[^>]+>')
        self.re_urls = re.compile(r'(?i)\b((?:https?://|www\.)\S+)')
        self.re_emails = re.compile(r'(?i)\b[\w\.-]+@[\w\.-]+\.[a-z]{2,}\b')
        self.re_numbers = re.compile(r'\b\d+(?:[\d\.,:/-]\d+)*\b')
        self.re_multispace = re.compile(r'\s+')
        self.re_nonword = re.compile(r'[^\w\s<>]')
        
        # Паттерны для фильтрации
        self.re_russian_word = re.compile(r'^[а-яё\-]+$')
        self.re_english_word = re.compile(r'^[a-z\-]+$')
        self.re_repeating_chars = re.compile(r'(.)\1{2,}')
        
        # Паттерны для токенизации
        self.word_pattern = re.compile(r'''
            (?:<URL>|<EMAIL>|<NUM>|<DATE>|<TIME>) |
            [а-яё]+(?:-[а-яё]+)* |
            [a-z]+(?:-[a-z]+)*
        ''', re.VERBOSE | re.IGNORECASE)

        # Сокращения
        self.abbreviations = {
            'т.е.': 'то есть', 'т. е.': 'то есть', 'т.к.': 'так как', 'т. к.': 'так как',
            'т.д.': 'так далее', 'т. д.': 'так далее', 'т.п.': 'тому подобное', 'т. п.': 'тому подобное',
            'др.': 'другие', 'г.': 'год', 'ул.': 'улица', 'рис.': 'рисунок',
            'см.': 'смотри', 'д.': 'дом', 'корп.': 'корпус', 'стр.': 'строение',
            'проф.': 'профессор', 'акад.': 'академик', 'доц.': 'доцент'
        }
        
        self.abbr_patterns = [
            (re.compile(rf'(?i)(?<![а-яё]){re.escape(k)}(?![а-яё])'), v) for k, v in self.abbreviations.items()
        ]

        # Статистика качества
        self.stats = {
            'total_documents': 0,
            'documents_truncated': 0,
            'low_diversity_documents': 0,
            'lemmatization_errors': 0,
            'cache_efficiency': 0.0,
            'garbage_words_filtered': 0,
            'mixed_words_filtered': 0
        }

        logger.info("TextProcessor инициализирован успешно")

    def _download_nltk_resources(self):
        """Загрузка необходимых ресурсов NLTK"""
        resources = ['punkt', 'stopwords', 'punkt_tab']
        
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
            except LookupError:
                try:
                    nltk.download(resource.split('_')[0], quiet=True)
                except Exception as e:
                    logger.error(f"Не удалось загрузить ресурс {resource}: {e}")

    def _is_garbage_word(self, word):
        """
        Проверка на мусорные слова
        
        Args:
            word: слово для проверки
            
        Returns:
            True если слово считается мусором
        """
        if word in {'<URL>', '<EMAIL>', '<NUM>', '<DATE>', '<TIME>'}:
            return False
            
        word_lower = word.lower()
        
        if len(word_lower) <= 2:
            return True
            
        if any(char.isdigit() for char in word_lower):
            return True
            
        if self.re_repeating_chars.search(word_lower):
            return True
            
        has_cyrillic = bool(re.search(r'[а-яё]', word_lower))
        has_latin = bool(re.search(r'[a-z]', word_lower))
        if has_cyrillic and has_latin:
            return True
            
        if not re.match(r'^[а-яёa-z\-]+$', word_lower):
            return True
            
        return False

    def _lemmatize_word(self, word):
        """
        Лемматизация слова с кэшированием (улучшенная версия)
        
        Args:
            word: слово для лемматизации
            
        Returns:
            лемматизированная форма слова или исходное слово при ошибке
        """
        if not self.lemmatization_enabled:
            return word
            
        if word in self.lemmatize_cache:
            self.cache_hits += 1
            return self.lemmatize_cache[word]
            
        self.cache_misses += 1
        
        try:
            # Получаем все возможные разборы
            parses = self.morph.parse(word)
            
            if not parses:
                return word
            
            # Берем первый разбор с достаточно высоким score
            best_parse = None
            for parse in parses:
                if parse.score >= 0.3:  # Снижаем порог для маленького корпуса
                    best_parse = parse
                    break
            
            if best_parse is None:
                best_parse = parses[0]  # Берем первый даже если score низкий
            
            # Проверяем, что нормальная форма валидна
            if (len(best_parse.normal_form) > 1 and
                'PNCT' not in best_parse.tag and
                'NUMB' not in best_parse.tag and
                'LATN' not in best_parse.tag and
                'UNKN' not in best_parse.tag and
                self.re_russian_word.match(best_parse.normal_form)):
                
                lemma = best_parse.normal_form
                
                # Сохраняем в кэш только если лемма отличается от исходного слова
                if lemma != word:
                    self.lemmatize_cache[word] = lemma
                    return lemma
            
            # Если лемматизация не дала результат или результат сомнителен, возвращаем исходное слово
            return word
            
        except Exception as e:
            self.stats['lemmatization_errors'] += 1
            logger.debug(f"Ошибка лемматизации слова '{word}': {e}")
            return word

    def calculate_lexical_diversity(self, text):
        """
        Вычисление лексического разнообразия текста
        
        Args:
            text: обработанный текст
            
        Returns:
            коэффициент лексического разнообразия (0-1)
        """
        words = text.split()
        if not words:
            return 0.0
        unique_words = len(set(words))
        return unique_words / len(words)

    def get_text_hash(self, text):
        """
        Вычисление хеша текста для дедупликации
        
        Args:
            text: текст для хеширования
            
        Returns:
            MD5 хеш текста
        """
        return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

    def improved_tokenize(self, text):
        """
        Токенизация с сохранением специальных токенов и фильтрацией мусора
        
        Args:
            text: текст для токенизации
            
        Returns:
            список токенов
        """
        try:
            special_tokens = re.findall(r'<(?:URL|EMAIL|NUM|DATE|TIME)>', text)
            placeholder = " §SPECIAL§ "
            text_with_placeholders = re.sub(r'<(?:URL|EMAIL|NUM|DATE|TIME)>', placeholder, text)
            
            tokens = word_tokenize(text_with_placeholders, language='russian')
            
            result = []
            special_idx = 0
            
            for token in tokens:
                if token == '§SPECIAL§' and special_idx < len(special_tokens):
                    result.append(special_tokens[special_idx])
                    special_idx += 1
                else:
                    if not self._is_garbage_word(token):
                        result.append(token)
                    else:
                        self.stats['garbage_words_filtered'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            return []

    def validate_record(self, data):
        """Проверка валидности записи"""
        required_fields = ['title', 'text']
        
        for field in required_fields:
            if field not in data or not data[field] or not isinstance(data[field], str):
                return False
        
        if len(data['text'].strip()) < ProcessingConfig.MIN_TEXT_LENGTH:
            return False
            
        if len(data['text']) > ProcessingConfig.MAX_DOCUMENT_CHARS:
            logger.warning(f"Текст превышает максимальную длину {ProcessingConfig.MAX_DOCUMENT_CHARS} символов")
            
        return True

    def clean_text(self, text):
        """Очистка и предобработка текста"""
        if not isinstance(text, str):
            return ""
        
        original = text

        # 1) Удаление HTML-разметки
        text = self.re_html_tags.sub(' ', original)

        # 2) Замены URL и email
        text = self.re_urls.sub('<URL>', text)
        text = self.re_emails.sub('<EMAIL>', text)

        # 3) Расширение сокращений
        text = text.replace('<URL>', '§URL§').replace('<EMAIL>', '§EMAIL§')
        text = text.lower()
        for pat, repl in self.abbr_patterns:
            text = pat.sub(repl, text)
        text = text.replace('§url§', '<URL>').replace('§email§', '<EMAIL>')

        # 4) Замена числительных и дат
        def replace_special_patterns(match):
            num = match.group()
            if re.match(r'\b\d{1,2}[:.]\d{2}\b', num):
                return '<TIME>'
            elif re.match(r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b', num):
                return '<DATE>'
            elif re.match(r'\b\d+[,.]?\d*%\b', num):
                return '<NUM>'
            elif re.match(r'\b\d{1,3}(?:[.,]\d{3})+\b', num):
                return '<NUM>'
            else:
                return '<NUM>'
        
        text = self.re_numbers.sub(replace_special_patterns, text)

        # 5) Очистка от шума
        text = self.re_nonword.sub(' ', text)

        # 6) Стандартизация пробелов
        text = self.re_multispace.sub(' ', text).strip()

        # 7) Токенизация с фильтрацией
        words = []
        try:
            tokens = self.improved_tokenize(text)
            token_count = 0
            
            for token in tokens:
                if token_count >= ProcessingConfig.MAX_DOCUMENT_WORDS:
                    self.stats['documents_truncated'] += 1
                    logger.warning(f"Документ обрезан до {ProcessingConfig.MAX_DOCUMENT_WORDS} токенов")
                    break
                    
                if token.upper() in {'<URL>', '<EMAIL>', '<NUM>', '<DATE>', '<TIME>'}:
                    words.append(token.upper())
                    token_count += 1
                    continue

                if token.lower() in self.stop_words:
                    continue
                    
                if len(token) <= 2:
                    continue

                processed = token.lower()
                
                if (self.lemmatization_enabled and 
                    len(processed) > 2 and 
                    self.re_russian_word.match(processed)):
                    processed = self._lemmatize_word(processed)

                words.append(processed)
                token_count += 1
                
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            return ""

        result = ' '.join(words)
        
        diversity = self.calculate_lexical_diversity(result)
        if diversity < ProcessingConfig.LEXICAL_DIVERSITY_THRESHOLD and len(words) > 10:
            self.stats['low_diversity_documents'] += 1
            logger.warning(f"Низкое лексическое разнообразие: {diversity:.2f}")
        
        if len(words) > 50:
            cache_efficiency = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            self.stats['cache_efficiency'] = cache_efficiency
            
        return result

    def get_cache_stats(self):
        """Возвращает статистику кэша лемматизации"""
        total = self.cache_hits + self.cache_misses
        efficiency = self.cache_hits / total if total > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_efficiency': efficiency,
            'cache_size': len(self.lemmatize_cache)
        }

    def get_processing_stats(self):
        """Возвращает статистику качества обработки"""
        return self.stats.copy()

def process_corpus(input_file, output_file, target_words=None, 
                  progress_callback=None, status_callback=None, enable_logging=False):
    """
    Обработка корпуса текстов
    
    Args:
        input_file: путь к входному файлу
        output_file: путь к выходному файлу
        target_words: целевое количество слов (None - обработать все данные)
        progress_callback: функция для обновления прогресса (принимает float 0-1)
        status_callback: функция для обновления статуса (принимает строку)
        enable_logging: включить логирование в консоль
    """
    processor = TextProcessor(enable_logging=enable_logging)
    total_words = 0
    category_stats = defaultdict(int)
    processed_count = 0
    skipped_count = 0
    validation_failed = 0
    duplicate_count = 0

    seen_text_hashes = set()

    if not os.path.exists(input_file):
        error_msg = f"Входной файл не найден: {input_file}"
        logger.error(error_msg)
        if status_callback:
            status_callback(f"Ошибка: {error_msg}")
        return {
            'total_words': 0,
            'processed_count': 0,
            'total_lines': 0,
            'category_stats': {},
            'skipped_count': 0,
            'validation_failed': 0,
            'duplicate_count': 0,
            'processing_stats': processor.get_processing_stats()
        }

    logger.info(f"Начата обработка файла: {input_file}")
    if status_callback:
        status_callback(f"Начата обработка файла: {input_file}")

    total_lines_in_file = 0
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines_in_file = sum(1 for _ in f)
    except:
        total_lines_in_file = 0
    
    if status_callback and total_lines_in_file > 0:
        status_callback(f"Найдено {total_lines_in_file} строк в файле. Начало обработки...")

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    skipped_count += 1
                    continue
                    
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    skipped_count += 1
                    continue
                
                if not processor.validate_record(data):
                    validation_failed += 1
                    continue
                
                original_category = data.get('category', '').strip() if data.get('category') else ''
                
                if '/' in original_category:
                    category = original_category.split('/')[0].strip()
                else:
                    category = original_category
                
                data['category'] = category if category else ''

                original_text = data['text']
                processed_text = processor.clean_text(original_text)
                
                word_count = len(processed_text.split())
                if word_count < ProcessingConfig.MIN_WORDS_AFTER_PROCESSING:
                    skipped_count += 1
                    continue
                
                text_hash = processor.get_text_hash(processed_text)
                if text_hash in seen_text_hashes:
                    duplicate_count += 1
                    logger.debug(f"Найден дубликат текста, пропускаем")
                    continue
                seen_text_hashes.add(text_hash)
                
                lexical_diversity = processor.calculate_lexical_diversity(processed_text)
                
                out_record = {
                    'title': data.get('title', ''),
                    'text': processed_text,
                    'date': data.get('date', ''),
                    'url': data.get('url', ''),
                    'category': data.get('category', 'другое'),
                    'word_count': word_count,
                    'original_category': original_category,
                    'processing_timestamp': datetime.now().isoformat(),
                    'lexical_diversity': round(lexical_diversity, 3),
                    'text_hash': text_hash,
                    'processing_quality': {
                        'lemmatization_enabled': processor.lemmatization_enabled,
                        'cache_efficiency': processor.get_cache_stats()['cache_efficiency'],
                        'garbage_words_filtered': processor.stats['garbage_words_filtered']
                    }
                }
                
                optional_fields = ['timestamp', 'char_count', 'source', 'author']
                for field in optional_fields:
                    if field in data:
                        out_record[field] = data[field]

                total_words += word_count
                category_stats[out_record['category']] += word_count
                processed_count += 1

                outfile.write(json.dumps(out_record, ensure_ascii=False) + '\n')

                if line_num % 100 == 0 or line_num == 1:
                    if total_lines_in_file > 0:
                        progress = min(line_num / total_lines_in_file, 1.0)
                    else:
                        progress = min(line_num / 10000, 1.0)
                    
                    if progress_callback:
                        progress_callback(progress)
                    
                    status_message = f"Обработано строк: {line_num}/{total_lines_in_file if total_lines_in_file > 0 else '?'}, документов: {processed_count}, слов: {total_words}"
                    if target_words:
                        status_message += f" (цель: {target_words})"
                    
                    if status_callback:
                        status_callback(status_message)

            logger.info(f"Обработка всех данных завершена. Всего строк: {line_num}, документов: {processed_count}, слов: {total_words}")

    except Exception as e:
        error_msg = f"Ошибка при обработке файла: {e}"
        logger.error(error_msg)
        if status_callback:
            status_callback(f"Ошибка: {error_msg}")
        return {
            'total_words': 0,
            'processed_count': 0,
            'total_lines': 0,
            'category_stats': {},
            'skipped_count': 0,
            'validation_failed': 0,
            'duplicate_count': 0,
            'processing_stats': processor.get_processing_stats()
        }

    if progress_callback:
        progress_callback(1.0)
    
    processing_stats = processor.get_processing_stats()
    
    logger.info("=== СТАТИСТИКА КАЧЕСТВА ОБРАБОТКИ ===")
    logger.info(f"Обработано документов: {processed_count}")
    logger.info(f"Всего слов: {total_words}")
    logger.info(f"Пропущено (дубликаты): {duplicate_count}")
    logger.info(f"Пропущено (валидация): {validation_failed}")
    logger.info(f"Пропущено (другие причины): {skipped_count}")
    logger.info(f"Документов обрезано: {processing_stats['documents_truncated']}")
    logger.info(f"Документов с низким разнообразием: {processing_stats['low_diversity_documents']}")
    logger.info(f"Ошибок лемматизации: {processing_stats['lemmatization_errors']}")
    logger.info(f"Эффективность кэша: {processing_stats['cache_efficiency']:.2%}")
    logger.info(f"Отфильтровано мусорных слов: {processing_stats['garbage_words_filtered']}")
    
    if target_words and total_words < target_words:
        warning_msg = f"Внимание: получено {total_words} слов, что меньше указанного порога {target_words}"
        logger.warning(warning_msg)
        if status_callback:
            status_callback(f"⚠️ {warning_msg}")
    else:
        success_msg = f"Успешно: получено {total_words} слов"
        if target_words:
            success_msg += f" (превышен порог: {target_words})"
        logger.info(success_msg)
        if status_callback:
            status_callback(success_msg)

    cache_stats = processor.get_cache_stats()
    logger.info(f"Статистика кэша лемматизации: {cache_stats}")

    logger.info(f"Обработка завершена. Слов: {total_words}, документов: {processed_count}")
    logger.info(f"Статистика категорий: {dict(category_stats)}")
    
    if status_callback:
        if total_words == 0 and processed_count == 0:
            status_msg = f"⚠️ Обработка завершена, но не обработано ни одного документа."
            if skipped_count > 0 or validation_failed > 0 or duplicate_count > 0:
                status_msg += f" Пропущено: {skipped_count}, невалидных: {validation_failed}, дубликатов: {duplicate_count}."
            status_callback(status_msg)
        else:
            status_msg = f"✅ Обработка завершена. Слов: {total_words}, документов: {processed_count}"
            if skipped_count > 0 or validation_failed > 0 or duplicate_count > 0:
                status_msg += f" (пропущено: {skipped_count}, невалидных: {validation_failed}, дубликатов: {duplicate_count})"
            status_callback(status_msg)
    
    return {
        'total_words': total_words,
        'processed_count': processed_count,
        'total_lines': total_lines_in_file,
        'category_stats': dict(category_stats),
        'skipped_count': skipped_count,
        'validation_failed': validation_failed,
        'duplicate_count': duplicate_count,
        'processing_stats': processing_stats
    }

def analyze_corpus(file_path):
    """Быстрый анализ корпуса без полной обработки"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        total_lines = len(lines)
        sample_data = []
        categories = set()
        field_stats = defaultdict(int)
        
        for line in lines:
            try:
                data = json.loads(line.strip())
                if len(sample_data) < 20:
                    sample_data.append(data)
                
                if 'category' in data and data.get('category'):
                    category = str(data['category']).strip()
                    if '/' in category:
                        category = category.split('/')[0].strip()
                    if category:
                        categories.add(category)
                
                for field in data.keys():
                    field_stats[field] += 1
                    
            except:
                continue
                
        analysis_result = {
            'total_documents': total_lines,
            'sample_size': len(sample_data),
            'categories_found': sorted(list(categories)),
            'fields_present': dict(field_stats),
            'sample_data': sample_data[:3]
        }
        
        logger.info(f"Анализ корпуса: {total_lines} документов, {len(categories)} категорий")
        return analysis_result
        
    except Exception as e:
        logger.error(f"Ошибка анализа корпуса: {e}")
        return {}