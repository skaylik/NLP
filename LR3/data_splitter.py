"""
Модуль для стратифицированного разделения данных на train/validation/test
с сохранением распределения категорий и обработкой ошибок
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import json
from typing import Dict, List, Tuple, Any
import traceback
import os


class StratifiedDataSplitter:
    """Класс для стратифицированного разделения данных"""
    
    def __init__(self, seed: int = 42):
        """
        Инициализация сплиттера
        
        Args:
            seed: сид для воспроизводимости
        """
        self.seed = seed
        np.random.seed(seed)
        self.splits = None
        print(f"🎲 Инициализирован StratifiedDataSplitter с seed={seed}")
    
    def split_stratified(self, 
                        data: List[Dict], 
                        train_ratio: float = 0.7, 
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15,
                        stratify_column: str = 'category',
                        save_splits: bool = True,
                        output_dir: str = "data_splits") -> Dict[str, List[Dict]]:
        """
        Стратифицированное разделение данных по категориям
        
        Args:
            data: список словарей с данными
            train_ratio: доля тренировочных данных
            val_ratio: доля валидационных данных
            test_ratio: доля тестовых данных
            stratify_column: колонка для стратификации
            save_splits: сохранить разделы на диск
            output_dir: директория для сохранения
            
        Returns:
            словарь с разбитыми данными: {'train': [...], 'validation': [...], 'test': [...]}
        """
        print(f"🔍 Начинаем стратифицированное разделение {len(data)} записей")
        print(f"📊 Соотношения: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        print(f"🏷️ Стратификация по колонке: {stratify_column}")
        
        # Проверка соотношений
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            print(f"⚠️ Сумма соотношений {total_ratio} != 1.0, корректируем...")
            train_ratio = train_ratio / total_ratio
            val_ratio = val_ratio / total_ratio
            test_ratio = test_ratio / total_ratio
        
        try:
            # Преобразуем в DataFrame
            df = pd.DataFrame(data)
            print(f"✅ Создан DataFrame: {df.shape[0]} строк, {df.shape[1]} колонок")
            
            if df.empty:
                print("⚠️ DataFrame пуст, возвращаем пустые разделы")
                return {'train': [], 'validation': [], 'test': []}
            
            # Если колонки для стратификации нет, используем случайное разделение
            if stratify_column not in df.columns:
                print(f"⚠️ Колонка '{stratify_column}' не найдена. Использую случайное разделение.")
                return self._split_random(data, train_ratio, val_ratio, test_ratio)
            
            # Преобразуем значения для стратификации
            df[stratify_column] = df[stratify_column].astype(str)
            
            # Получаем уникальные категории
            categories = df[stratify_column].unique()
            print(f"📊 Категорий для стратификации: {len(categories)}")
            
            if len(categories) == 0:
                print("⚠️ Нет категорий для стратификации, использую случайное разделение")
                return self._split_random(data, train_ratio, val_ratio, test_ratio)
            
            # Разделяем каждую категорию отдельно
            train_data = []
            val_data = []
            test_data = []
            
            category_stats = {}
            
            for category in categories:
                # Получаем данные для текущей категории
                category_df = df[df[stratify_column] == category].copy()
                category_records = category_df.to_dict('records')
                category_count = len(category_records)
                
                if category_count < 3:
                    # Если в категории мало данных, просто добавляем в train
                    print(f"  ⚠️ Категория '{category}': слишком мало данных ({category_count}), добавляем в train")
                    train_data.extend(category_records)
                    category_stats[category] = {'train': category_count, 'validation': 0, 'test': 0}
                    continue
                
                # Рассчитываем размеры разделов
                train_size = max(1, int(category_count * train_ratio))
                test_size = max(1, int(category_count * test_ratio))
                val_size = category_count - train_size - test_size
                
                # Корректировка
                if val_size < 0:
                    val_size = 0
                    train_size = category_count - test_size
                
                # Разделение
                try:
                    np.random.shuffle(category_records)
                    
                    train_data.extend(category_records[:train_size])
                    val_data.extend(category_records[train_size:train_size + val_size])
                    test_data.extend(category_records[train_size + val_size:])
                    
                    category_stats[category] = {
                        'train': train_size,
                        'validation': val_size,
                        'test': test_size
                    }
                    
                except Exception as e:
                    print(f"  ⚠️ Ошибка при разделении категории '{category}': {e}")
                    # Простое случайное разделение
                    train_data.extend(category_records[:train_size])
                    val_data.extend(category_records[train_size:train_size + val_size])
                    test_data.extend(category_records[train_size + val_size:])
            
            # Перемешиваем финальные наборы
            np.random.shuffle(train_data)
            np.random.shuffle(val_data)
            np.random.shuffle(test_data)
            
            self.splits = {
                'train': train_data,
                'validation': val_data,
                'test': test_data
            }
            
            # Выводим статистику
            self._print_statistics(category_stats, train_data, val_data, test_data)
            
            # Сохраняем на диск если нужно
            if save_splits:
                self._save_splits(output_dir)
            
            return self.splits
            
        except Exception as e:
            print(f"❌ Ошибка при стратифицированном разделении: {e}")
            print(traceback.format_exc())
            return self._split_random(data, train_ratio, val_ratio, test_ratio)
    
    def _split_random(self, data: List[Dict], train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, List[Dict]]:
        """Простое случайное разделение"""
        print("🎲 Использую случайное разделение...")
        
        np.random.shuffle(data)
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        self.splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        print(f"📊 Случайное разделение:")
        print(f"  Train: {len(train_data)} записей ({len(train_data)/n*100:.1f}%)")
        print(f"  Validation: {len(val_data)} записей ({len(val_data)/n*100:.1f}%)")
        print(f"  Test: {len(test_data)} записей ({len(test_data)/n*100:.1f}%)")
        
        return self.splits
    
    def _save_splits(self, output_dir: str = "data_splits"):
        """Сохранение разделов на диск"""
        if not self.splits:
            print("⚠️ Нет данных для сохранения")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, data in self.splits.items():
            filename = os.path.join(output_dir, f"{split_name}.jsonl")
            with open(filename, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"💾 Сохранен {split_name} набор: {filename} ({len(data)} записей)")
        
        # Сохраняем метаданные
        metadata = {
            'splits': {k: len(v) for k, v in self.splits.items()},
            'seed': self.seed,
            'total_records': sum(len(v) for v in self.splits.values()),
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = os.path.join(output_dir, "split_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Сохранена метаданная: {metadata_file}")
    
    def load_splits(self, input_dir: str = "data_splits") -> Dict[str, List[Dict]]:
        """Загрузка разделов с диска"""
        splits = {}
        
        for split_name in ['train', 'validation', 'test']:
            filename = os.path.join(input_dir, f"{split_name}.jsonl")
            if os.path.exists(filename):
                data = []
                with open(filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                splits[split_name] = data
                print(f"📂 Загружен {split_name} набор: {filename} ({len(data)} записей)")
            else:
                print(f"⚠️ Файл {filename} не найден")
                splits[split_name] = []
        
        self.splits = splits
        return splits
    
    def _print_statistics(self, category_stats: Dict, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """Печать статистики разделения"""
        
        print(f"\n{'='*60}")
        print("📊 СТАТИСТИКА РАЗДЕЛЕНИЯ ДАННЫХ")
        print('='*60)
        
        # Общая статистика
        total_train = len(train_data)
        total_val = len(val_data)
        total_test = len(test_data)
        total = total_train + total_val + total_test
        
        print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
        print(f"  Всего записей: {total}")
        print(f"  Train: {total_train} ({total_train/total*100:.1f}%)")
        print(f"  Validation: {total_val} ({total_val/total*100:.1f}%)")
        print(f"  Test: {total_test} ({total_test/total*100:.1f}%)")
        
        print(f"\n{'='*60}")
        print("✅ РАЗДЕЛЕНИЕ ЗАВЕРШЕНО")
        print('='*60)
    
    def get_split_statistics_df(self) -> pd.DataFrame:
        """Получение статистики в виде DataFrame"""
        if not self.splits:
            return pd.DataFrame()
        
        stats = []
        for split_name, data in self.splits.items():
            if data:
                df = pd.DataFrame(data)
                if 'category' in df.columns:
                    category_counts = df['category'].value_counts()
                    for category, count in category_counts.items():
                        stats.append({
                            'split': split_name,
                            'category': category,
                            'count': count
                        })
        
        return pd.DataFrame(stats)