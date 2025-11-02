"""
Скрипт для подготовки данных FLEURS для обучения и оценки.
"""

import pandas as pd
import re
import tempfile
import soundfile as sf


def load_fleurs_data(split='train'):
    """
    Загружает данные FLEURS для указанного split (train/validation/test)
    используя библиотеку datasets от HuggingFace
    
    Args:
        split: один из ['train', 'dev', 'test']
               'dev' автоматически преобразуется в 'validation'
    
    Returns:
        DataFrame с данными
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Ошибка: библиотека datasets не установлена")
        print("Установите её: pip install datasets")
        return pd.DataFrame()
    
    # Преобразуем 'dev' в 'validation' для совместимости с datasets
    dataset_split = 'validation' if split == 'dev' else split
    
    print(f"Загрузка FLEURS (ru_ru, {dataset_split}) из HuggingFace...")
    
    try:
        dataset = load_dataset("google/fleurs", "ru_ru", split=dataset_split)
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        return pd.DataFrame()
    
    # Преобразование в DataFrame
    data_list = []
    for idx, item in enumerate(dataset):
        data_list.append({
            'id': item['id'],
            'audio_array': item['audio']['array'],
            'sampling_rate': item['audio']['sampling_rate'],
            'raw_text': item['raw_transcription'],
            'transcription': item['transcription'],
            'num_samples': item['num_samples'],
            'gender': item['gender']
        })
    
    data = pd.DataFrame(data_list)
    print(f"✓ Загружено {len(data)} образцов")
    
    return data


def normalize_text(text):
    """
    Нормализация текста согласно требованиям задания:
    - приведение к нижнему регистру
    - удаление знаков препинания
    - сохранение цифр и латиницы
    
    Args:
        text: исходный текст
    
    Returns:
        нормализованный текст
    """
    # Проверка входных данных
    if not isinstance(text, str):
        return ""
    
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление знаков препинания, но сохранение букв, цифр и пробелов
    # Сохраняем кириллицу, латиницу и цифры
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    
    # Удаление лишних пробелов
    text = ' '.join(text.split())
    
    return text


# Пример использования
if __name__ == "__main__":
    print("Проверка и подготовка данных FLEURS...")
    print("=" * 60)
    
    print("Загрузка данных...")
    
    # Загрузка данных
    train_data = load_fleurs_data('train')
    dev_data = load_fleurs_data('dev')
    test_data = load_fleurs_data('test')
    
    print(f"\nСтатистика датасета:")
    print(f"Train samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Total: {len(train_data) + len(dev_data) + len(test_data)}")
    
    # Пример нормализации
    if len(train_data) > 0:
        print("\n" + "=" * 60)
        print("Пример нормализации текста:")
        sample_text = train_data.iloc[0]['raw_text']
        normalized = normalize_text(sample_text)
        print(f"\nОригинал: {sample_text}")
        print(f"Нормализованный: {normalized}")
    
    print("\n" + "=" * 60)
    print("Подготовка данных завершена!")
