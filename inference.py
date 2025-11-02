"""
Скрипт для запуска инференса модели GigaAM на датасете FLEURS.
"""

from prepare_data import load_fleurs_data, normalize_text
from tqdm import tqdm
import pandas as pd
import argparse


def run_inference(model, data_df, output_file='predictions.csv'):
    """
    Запуск инференса на датасете
    
    Args:
        model: модель GigaAM
        data_df: DataFrame с данными (содержит audio_array)
        output_file: путь для сохранения результатов
    
    Returns:
        predictions: список предсказаний
        references: список эталонных транскрипций
    """
    import numpy as np
    import tempfile
    import soundfile as sf
    import os
    
    predictions = []
    references = []
    audio_ids = []
    
    print(f"Запуск инференса на {len(data_df)} образцах...")
    
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Running inference"):
        audio_array = row['audio_array']
        sampling_rate = row['sampling_rate']
        audio_id = row.get('id', idx)
        reference_text = normalize_text(row['raw_text'])
        
        try:
            # Создаем временный wav файл из массива
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio_array, sampling_rate)
            
            # Транскрибация аудио
            prediction = model.transcribe(tmp_path)
            
            # Удаляем временный файл
            os.unlink(tmp_path)
            
            # Нормализация предсказания
            prediction = normalize_text(prediction)
            
            predictions.append(prediction)
            references.append(reference_text)
            audio_ids.append(audio_id)
            
        except Exception as e:
            print(f"\nОшибка при обработке образца {audio_id}: {e}")
            predictions.append("")
            references.append(reference_text)
            audio_ids.append(audio_id)
            # Очистка временного файла в случае ошибки
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # Сохранение результатов
    results_df = pd.DataFrame({
        'audio_id': audio_ids,
        'reference': references,
        'prediction': predictions
    })
    
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Предсказания сохранены в {output_file}")
    
    return predictions, references


def main():
    parser = argparse.ArgumentParser(description='Запуск инференса GigaAM на FLEURS')
    parser.add_argument('--split', type=str, default='dev', 
                       choices=['train', 'dev', 'test'],
                       help='Какой split использовать (по умолчанию: dev)')
    parser.add_argument('--model', type=str, default='ctc',
                       choices=['ctc', 'v2_ctc', 'v1_ctc', 'rnnt', 'v2_rnnt', 'v1_rnnt'],
                       help='Версия модели GigaAM (по умолчанию: ctc)')
    parser.add_argument('--output', type=str, default=None,
                       help='Имя выходного файла (по умолчанию: {split}_predictions.csv)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Ограничить количество образцов для обработки')
    
    args = parser.parse_args()
    
    # Определение имени выходного файла
    if args.output is None:
        args.output = f'{args.split}_predictions.csv'
    
    # Загрузка модели с обработкой ошибок
    print(f"Загрузка модели GigaAM ({args.model})...")
    try:
        import gigaam
        model = gigaam.load_model(args.model)
        print("✓ Модель загружена успешно!")
    except ImportError:
        print("✗ Ошибка: модуль gigaam не установлен")
        print("Установите GigaAM: cd GigaAM && pip install -e . && cd ..")
        return
    except Exception as e:
        print(f"✗ Ошибка при загрузке модели: {e}")
        return
    
    # Загрузка данных
    print(f"\nЗагрузка данных ({args.split})...")
    try:
        data = load_fleurs_data(args.split)
        print(f"✓ Загружено {len(data)} образцов")
    except Exception as e:
        print(f"✗ Ошибка при загрузке данных: {e}")
        return
    
    # Ограничение количества образцов, если указано
    if args.limit is not None:
        data = data.head(args.limit)
        print(f"Ограничено до {len(data)} образцов")
    
    # Запуск инференса
    print()
    predictions, references = run_inference(model, data, args.output)
    
    # Статистика
    valid_predictions = sum(1 for p in predictions if p)
    print(f"\n{'='*60}")
    print(f"Статистика:")
    print(f"Всего образцов: {len(predictions)}")
    print(f"Успешных предсказаний: {valid_predictions}")
    print(f"Неудачных предсказаний: {len(predictions) - valid_predictions}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
