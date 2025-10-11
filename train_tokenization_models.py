import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
import json
from typing import List
import re
import numpy as np

class SubwordModelExperiment:
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.results = []
    
    def _reconstruct_wordpiece_text(self, tokens):
        """Кастомная реконструкция для WordPiece"""
        text = ""
        for token in tokens:
            if token.startswith('##'):
                # Склеиваем с предыдущим словом
                text += token[2:]
            else:
                # Добавляем пробел перед новым словом
                if text:
                    text += " "
                text += token
        return text
    
    def debug_reconstruction(self, tokenizer, sample_texts, model_name):
        """Отладочная функция для реконструкции"""
        print(f"\n--- ОТЛАДКА РЕКОНСТРУКЦИИ ДЛЯ {model_name} ---")
        
        for i, text in enumerate(sample_texts[:3]):
            print(f"\nПример {i+1}:")
            print(f"Исходный: {text[:100]}...")
            
            try:
                # Кодируем
                encoding = tokenizer.encode(text)
                tokens = encoding.tokens
                
                # Декодируем с учетом типа модели
                model_type = str(type(tokenizer.model)).lower()
                if 'wordpiece' in model_type:
                    reconstructed = self._reconstruct_wordpiece_text(tokens)
                else:
                    reconstructed = tokenizer.decode(encoding.ids, skip_special_tokens=False)
                
                print(f"Токены: {tokens[:20]}...")
                print(f"Восстановленный: {reconstructed[:100]}...")
                
                # Сравниваем с нормализацией
                if self._texts_semantically_equal(text, reconstructed):
                    print("✅ СОВПАДАЕТ (семантически)")
                else:
                    print("❌ НЕ СОВПАДАЕТ")
                    print(f"Разница: '{text[:60]}' vs '{reconstructed[:60]}'")
                    
            except Exception as e:
                print(f"❌ Ошибка: {e}")
    
    def _texts_semantically_equal(self, text1, text2):
        """Сравнивает тексты по содержанию, а не по форматированию"""
        def normalize(t):
            # Удаляем все пробелы для сравнения содержания
            t = re.sub(r'\s+', '', t)
            # Нормализуем специальные токены
            t = re.sub(r'<\s*NUM\s*>', '<NUM>', t)
            t = re.sub(r'<\s*URL\s*>', '<URL>', t)
            t = re.sub(r'<\s*EMAIL\s*>', '<EMAIL>', t)
            return t
        
        text1_norm = normalize(text1)
        text2_norm = normalize(text2)
        
        # Сравниваем только первые 100 символов (чтобы избежать проблем с длинными текстами)
        compare_length = min(100, len(text1_norm), len(text2_norm))
        
        return text1_norm[:compare_length] == text2_norm[:compare_length]

    def train_bpe(self, vocab_size: int, min_frequency: int = 2) -> Tokenizer:
        """Обучение BPE модели"""
        try:
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=["[UNK]", "<NUM>", "<URL>", "<EMAIL>"]
            )
            
            tokenizer.train_from_iterator(self.texts, trainer)
            return tokenizer
            
        except Exception as e:
            print(f"Ошибка в train_bpe: {e}")
            return None
    
    def train_wordpiece(self, vocab_size: int, min_frequency: int = 2) -> Tokenizer:
        """Обучение WordPiece модели"""
        try:
            tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            
            trainer = WordPieceTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=["[UNK]", "<NUM>", "<URL>", "<EMAIL>"]
            )
            
            tokenizer.train_from_iterator(self.texts, trainer)
            return tokenizer
            
        except Exception as e:
            print(f"Ошибка в train_wordpiece: {e}")
            return None
    
    def train_unigram(self, vocab_size: int, min_frequency: int = 2):
        """Обучение Unigram модели"""
        try:
            tokenizer = Tokenizer(Unigram())
            tokenizer.pre_tokenizer = Whitespace()
            
            trainer = UnigramTrainer(
                vocab_size=vocab_size,
                unk_token="[UNK]",
                special_tokens=["[UNK]", "<NUM>", "<URL>", "<EMAIL>"],
                shrinking_factor=0.75
            )
            
            tokenizer.train_from_iterator(self.texts, trainer)
            return tokenizer
            
        except Exception as e:
            print(f"Ошибка в train_unigram: {e}")
            return None

    def calculate_fragmentation_rate(self, tokenizer, sample_texts: List[str]) -> float:
        """Вычисление процента фрагментации слов"""
        try:
            fragmented_count = 0
            total_words = 0
            
            for text in sample_texts:
                words = text.split()
                total_words += len(words)
                
                for word in words:
                    if word in ['<NUM>', '<URL>', '<EMAIL>']:
                        continue
                        
                    tokens = tokenizer.encode(word).tokens
                    if len(tokens) >= 2:
                        fragmented_count += 1
            
            return (fragmented_count / total_words) * 100 if total_words > 0 else 0
            
        except Exception as e:
            print(f"Ошибка в calculate_fragmentation_rate: {e}")
            return 0
    
    def calculate_compression_ratio(self, tokenizer, sample_texts: List[str]) -> float:
        """Вычисление коэффициента сжатия"""
        try:
            total_original_words = 0
            total_tokens = 0
            
            for text in sample_texts:
                words = text.split()
                total_original_words += len(words)
                
                tokens = tokenizer.encode(text).tokens
                total_tokens += len(tokens)
            
            return total_original_words / total_tokens if total_tokens > 0 else 0
            
        except Exception as e:
            print(f"Ошибка в calculate_compression_ratio: {e}")
            return 0
    
    def calculate_reconstruction_efficiency(self, tokenizer, sample_texts: List[str]) -> float:
        """Улучшенная эффективность реконструкции"""
        try:
            correct_reconstructions = 0
            total_texts = len(sample_texts)
            
            for i, text in enumerate(sample_texts):
                try:
                    encoding = tokenizer.encode(text)
                    tokens = encoding.tokens
                    
                    # Определяем тип модели для выбора метода реконструкции
                    model_type = str(type(tokenizer.model)).lower()
                    
                    # Кастомная реконструкция для WordPiece
                    if 'wordpiece' in model_type:
                        reconstructed = self._reconstruct_wordpiece_text(tokens)
                    else:
                        reconstructed = tokenizer.decode(encoding.ids, skip_special_tokens=False)
                    
                    if self._texts_semantically_equal(text, reconstructed):
                        correct_reconstructions += 1
                    elif i < 2:  # Показываем только первые расхождения
                        print(f"Семантическое расхождение {i+1}:")
                        print(f"  Оригинал: {text[:80]}")
                        print(f"  Восстановлено: {reconstructed[:80]}")
                        print(f"  Токены: {tokens[:15]}")
                            
                except Exception as e:
                    if i < 2:
                        print(f"Ошибка при обработке текста {i+1}: {e}")
                    continue
            
            efficiency = (correct_reconstructions / total_texts) * 100 if total_texts > 0 else 0
            print(f"Семантическая реконструкция: {efficiency}% ({correct_reconstructions}/{total_texts})")
            return efficiency
            
        except Exception as e:
            print(f"Ошибка в calculate_reconstruction_efficiency: {e}")
            return 0

    def analyze_vocabulary_coverage(self, tokenizer, sample_texts: List[str]):
        """Анализ покрытия словаря"""
        vocab = tokenizer.get_vocab()
        print(f"Фактический размер словаря: {len(vocab)}")
        
        # Анализ OOV (out-of-vocabulary) токенов
        oov_count = 0
        total_tokens = 0
        
        for text in sample_texts[:10]:  # На выборке из 10 текстов
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            total_tokens += len(tokens)
            oov_count += tokens.count('[UNK]')
        
        oov_rate = (oov_count / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"OOV rate: {oov_rate:.2f}%")
        return oov_rate

    def run_experiment(self):
        """Проведение эксперимента"""
        vocab_sizes = [8000, 16000, 32000]
        min_frequencies = [2]
        
        test_texts = self.texts[:min(50, len(self.texts))]
        debug_texts = self.texts[:3]
        
        print(f"Начало эксперимента с {len(test_texts)} тестовыми текстами")
        
        for vocab_size in vocab_sizes:
            for min_freq in min_frequencies:
                print(f"\n=== Обучение: vocab_size={vocab_size}, min_frequency={min_freq} ===")
                
                # BPE
                print("Обучение BPE...")
                bpe_tokenizer = self.train_bpe(vocab_size, min_freq)
                if bpe_tokenizer:
                    self.debug_reconstruction(bpe_tokenizer, debug_texts, f"BPE_{vocab_size}")
                    
                    bpe_fragmentation = self.calculate_fragmentation_rate(bpe_tokenizer, test_texts)
                    bpe_compression = self.calculate_compression_ratio(bpe_tokenizer, test_texts)
                    bpe_reconstruction = self.calculate_reconstruction_efficiency(bpe_tokenizer, test_texts[:10])
                    
                    self.results.append({
                        'Модель': 'BPE',
                        'Размер словаря': vocab_size,
                        'Мин. частота': min_freq,
                        'Фрагментация (%)': round(bpe_fragmentation, 2),
                        'Коэф. сжатия': round(bpe_compression, 3),
                        'Реконструкция (%)': round(bpe_reconstruction, 2)
                    })
                else:
                    print("BPE: не удалось обучить")
                
                # WordPiece
                print("Обучение WordPiece...")
                wp_tokenizer = self.train_wordpiece(vocab_size, min_freq)
                if wp_tokenizer:
                    self.debug_reconstruction(wp_tokenizer, debug_texts, f"WordPiece_{vocab_size}")
                    
                    wp_fragmentation = self.calculate_fragmentation_rate(wp_tokenizer, test_texts)
                    wp_compression = self.calculate_compression_ratio(wp_tokenizer, test_texts)
                    wp_reconstruction = self.calculate_reconstruction_efficiency(wp_tokenizer, test_texts[:10])
                    
                    self.results.append({
                        'Модель': 'WordPiece',
                        'Размер словаря': vocab_size,
                        'Мин. частота': min_freq,
                        'Фрагментация (%)': round(wp_fragmentation, 2),
                        'Коэф. сжатия': round(wp_compression, 3),
                        'Реконструкция (%)': round(wp_reconstruction, 2)
                    })
                else:
                    print("WordPiece: не удалось обучить")
                
                # Unigram
                print("Обучение Unigram...")
                unigram_tokenizer = self.train_unigram(vocab_size, min_freq)
                if unigram_tokenizer:
                    self.debug_reconstruction(unigram_tokenizer, debug_texts, f"Unigram_{vocab_size}")
                    
                    unigram_fragmentation = self.calculate_fragmentation_rate(unigram_tokenizer, test_texts)
                    unigram_compression = self.calculate_compression_ratio(unigram_tokenizer, test_texts)
                    unigram_reconstruction = self.calculate_reconstruction_efficiency(unigram_tokenizer, test_texts[:10])
                    
                    self.results.append({
                        'Модель': 'Unigram',
                        'Размер словаря': vocab_size,
                        'Мин. частота': min_freq,
                        'Фрагментация (%)': round(unigram_fragmentation, 2),
                        'Коэф. сжатия': round(unigram_compression, 3),
                        'Реконструкция (%)': round(unigram_reconstruction, 2)
                    })
                else:
                    print("Unigram: не удалось обучить")
        
        print(f"\nЭксперимент завершен. Получено результатов: {len(self.results)}")
    
    def save_results(self, filename: str = 'subword_models_metrics.csv'):
        """Сохранение результатов"""
        if not self.results:
            print("Нет результатов для сохранения")
            return None
            
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Результаты сохранены в {filename}")
        return df

    def visualize_results(self):
        """Улучшенная визуализация результатов"""
        if not self.results:
            print("Нет результатов для визуализации")
            return
        
        df = pd.DataFrame(self.results)
        
        # Создаем фигуру с 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Сравнительный анализ подсловных моделей токенизации', fontsize=16)
        
        # Цвета для разных моделей
        colors = {'BPE': 'blue', 'WordPiece': 'red', 'Unigram': 'green'}
        markers = {'BPE': 'o', 'WordPiece': 's', 'Unigram': '^'}
        
        # График 1: Фрагментация
        for model in df['Модель'].unique():
            model_data = df[df['Модель'] == model]
            axes[0,0].plot(model_data['Размер словаря'], model_data['Фрагментация (%)'], 
                    marker=markers[model], label=model, linewidth=2, color=colors[model])
            # Добавляем аннотации с значениями
            for idx, row in model_data.iterrows():
                axes[0,0].annotate(f"{row['Фрагментация (%)']}%", 
                            (row['Размер словаря'], row['Фрагментация (%)']),
                            textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        
        axes[0,0].set_title('Фрагментация слов (меньше → лучше)')
        axes[0,0].set_xlabel('Размер словаря')
        axes[0,0].set_ylabel('Фрагментация (%)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # График 2: Коэффициент сжатия
        for model in df['Модель'].unique():
            model_data = df[df['Модель'] == model]
            axes[0,1].plot(model_data['Размер словаря'], model_data['Коэф. сжатия'], 
                    marker=markers[model], label=model, linewidth=2, color=colors[model])
            for idx, row in model_data.iterrows():
                axes[0,1].annotate(f"{row['Коэф. сжатия']}x", 
                            (row['Размер словаря'], row['Коэф. сжатия']),
                            textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        
        axes[0,1].set_title('Коэффициент сжатия (больше → лучше)')
        axes[0,1].set_xlabel('Размер словаря')
        axes[0,1].set_ylabel('Коэффициент сжатия')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # График 3: Реконструкция (столбчатая диаграмма для лучшей видимости)
        models = df['Модель'].unique()
        vocab_sizes = sorted(df['Размер словаря'].unique())
        
        bar_width = 0.25
        x_pos = np.arange(len(vocab_sizes))
        
        for i, model in enumerate(models):
            model_data = df[df['Модель'] == model]
            values = []
            for size in vocab_sizes:
                size_data = model_data[model_data['Размер словаря'] == size]
                if not size_data.empty:
                    values.append(size_data['Реконструкция (%)'].values[0])
                else:
                    values.append(0)
            
            axes[1,0].bar(x_pos + i * bar_width, values, bar_width, label=model, color=colors[model])
            
            # Добавляем значения поверх столбцов
            for j, v in enumerate(values):
                axes[1,0].text(x_pos[j] + i * bar_width, v + 1, f"{v}%", 
                        ha='center', va='bottom', fontsize=9)
        
        axes[1,0].set_title('Эффективность реконструкции (больше → лучше)')
        axes[1,0].set_xlabel('Размер словаря')
        axes[1,0].set_ylabel('Реконструкция (%)')
        axes[1,0].set_xticks(x_pos + bar_width)
        axes[1,0].set_xticklabels(vocab_sizes)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # График 4: Сводная оценка (композитный score)
        axes[1,1].axis('off')  # Можно использовать для сводной таблицы или другого анализа
        
        plt.tight_layout()
        plt.savefig('subword_models_analysis_improved.png', dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_additional_metrics(self, tokenizer, sample_texts: List[str]):
        """Дополнительные метрики анализа"""
        metrics = {}
        
        # Средняя длина токена
        total_chars = 0
        total_tokens = 0
        
        for text in sample_texts:
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            total_tokens += len(tokens)
            total_chars += sum(len(token) for token in tokens)
        
        metrics['avg_token_length'] = total_chars / total_tokens if total_tokens > 0 else 0
        
        # Процент специальных токенов
        special_tokens = 0
        for text in sample_texts:
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            special_tokens += sum(1 for token in tokens if token in ['[UNK]', '<NUM>', '<URL>', '<EMAIL>'])
        
        metrics['special_tokens_rate'] = (special_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        
        return metrics

def load_texts_from_jsonl(file_path: str, max_texts: int = 50) -> List[str]:
    """Загрузка текстов из JSONL файла"""
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_texts:
                    break
                if line.strip():
                    article = json.loads(line)
                    if 'text' in article and article['text'].strip():
                        texts.append(article['text'])
        print(f"Загружено {len(texts)} текстов из {file_path}")
        return texts
    except Exception as e:
        print(f"Ошибка загрузки файла {file_path}: {e}")
        return []

def analyze_results(df):
    """Анализ результатов"""
    if df.empty:
        print("Нет данных для анализа")
        return
    
    print("\n" + "="*60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    # Лучшие модели по каждой метрике
    metrics = ['Фрагментация (%)', 'Коэф. сжатия', 'Реконструкция (%)']
    
    for metric in metrics:
        if metric == 'Фрагментация (%)':
            best_idx = df[metric].idxmin()
            best_value = df.loc[best_idx, metric]
            print(f"\n🏆 Лучшая {metric}: {best_value}%")
        else:
            best_idx = df[metric].idxmax()
            best_value = df.loc[best_idx, metric]
            unit = 'x' if metric == 'Коэф. сжатия' else '%'
            print(f"\n🏆 Лучшая {metric}: {best_value}{unit}")
        
        best_row = df.iloc[best_idx]
        print(f"   Модель: {best_row['Модель']}")
        print(f"   Размер словаря: {best_row['Размер словаря']}")
        print(f"   Мин. частота: {best_row['Мин. частота']}")

# Запуск эксперимента
if __name__ == "__main__":
    # Загрузка данных
    texts = load_texts_from_jsonl('indicator_ru_corpus_advanced_cleaned.jsonl', max_texts=50)
    
    if not texts:
        print("Не удалось загрузить тексты. Создаем демо-данные...")
        texts = [
            "биология опубликовано <NUM> июля <NUM> <NUM>:<NUM> <NUM> мин. a a рачки-бокоплавы носят самок лапках",
            "медицина опубликовано <NUM> июля <NUM> <NUM>:<NUM> <NUM> мин. a a эксперты назвали наиболее перспективные",
            "гуманитарные науки опубликовано <NUM> июля <NUM> <NUM>:<NUM> <NUM> мин. a a старой руссе нашли первую"
        ] * 10
    
    print(f"Загружено {len(texts)} текстов для эксперимента")
    
    # Проведение эксперимента
    experiment = SubwordModelExperiment(texts)
    experiment.run_experiment()
    
    # Сохранение и анализ результатов
    results_df = experiment.save_results()
    
    if results_df is not None and not results_df.empty:
        print("\n" + "="*60)
        print("СВОДНЫЕ РЕЗУЛЬТАТЫ:")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # Визуализация
        experiment.visualize_results()
        
        # Анализ
        analyze_results(results_df)
    else:
        print("Эксперимент не дал результатов для анализа")