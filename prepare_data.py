"""
Скрипт для подготовки данных FLEURS для обучения и оценки.
"""

import os
import pandas as pd
import torch
from pathlib import Path
import re


def load_fleurs_data(split='train'):
    """
    Загружает данные FLEURS для указанного split (train/dev/test)
    
    Args:
        split: один из ['train', 'dev', 'test']
    
    Returns:
        DataFrame с данными
    """
    base_path = Path('fleurs/data/ru_ru')
    tsv_file = base_path / f'{split}.tsv'
    audio_dir = base_path / 'audio' / split
    
    # Чтение TSV файла
    data = pd.read_csv(tsv_file, sep='\t', header=None, 
                       names=['id', 'filename', 'raw_text', 'normalized_text', 
                              'phonemes', 'num_samples', 'gender'])
    
    # Добавление полных путей к аудио файлам
    data['audio_path'] = data['filename'].apply(lambda x: str(audio_dir / x))
    
    # Проверка существования файлов
    data['exists'] = data['audio_path'].apply(os.path.exists)
    missing = (~data['exists']).sum()
    
    if missing > 0:
        print(f"Предупреждение: {missing} аудио файлов не найдено в {split}")
        print(f"Убедитесь, что архивы распакованы в директории {audio_dir}")
    
    data = data[data['exists']]
    
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


def extract_audio_files():
    """
    Распаковывает аудио файлы из архивов
    """
    import tarfile
    
    base_path = Path('fleurs/data/ru_ru/audio')
    
    for split in ['train', 'dev', 'test']:
        archive_path = base_path / f'{split}.tar.gz'
        extract_path = base_path / split
        
        if not extract_path.exists():
            print(f"Распаковка {archive_path}...")
            try:
                with tarfile.open(archive_path, 'r:gz') as tar:
                    # Безопасная распаковка: проверяем пути файлов
                    members = tar.getmembers()
                    safe_members = []
                    for member in members:
                        # Проверка на directory traversal
                        member_path = Path(member.name)
                        if member_path.is_absolute() or '..' in member_path.parts:
                            print(f"Пропуск небезопасного пути: {member.name}")
                            continue
                        safe_members.append(member)
                    
                    tar.extractall(path=base_path, members=safe_members)
                print(f"✓ {split} распакован")
            except Exception as e:
                print(f"✗ Ошибка при распаковке {split}: {e}")
        else:
            print(f"✓ {split} уже распакован")


# Пример использования
if __name__ == "__main__":
    print("Проверка и подготовка данных FLEURS...")
    print("=" * 60)
    
    # Распаковка архивов если необходимо
    extract_audio_files()
    
    print("\n" + "=" * 60)
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
    print("\n" + "=" * 60)
    print("Пример нормализации текста:")
    sample_text = train_data.iloc[0]['raw_text']
    normalized = normalize_text(sample_text)
    print(f"\nОригинал: {sample_text}")
    print(f"Нормализованный: {normalized}")
    
    print("\n" + "=" * 60)
    print("Подготовка данных завершена!")
