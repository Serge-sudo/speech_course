"""
Скрипт для запуска инференса модели GigaAM на датасете FLEURS.
"""

import gigaam
from prepare_data import load_fleurs_data, normalize_text
from tqdm import tqdm
import pandas as pd
import argparse


def run_inference(model, data_df, output_file='predictions.csv'):
    """
    Запуск инференса на датасете
    
    Args:
        model: модель GigaAM
        data_df: DataFrame с данными
        output_file: путь для сохранения результатов
    
    Returns:
        predictions: список предсказаний
        references: список эталонных транскрипций
    """
    predictions = []
    references = []
    audio_paths = []
    
    print(f"Запуск инференса на {len(data_df)} образцах...")
    
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Running inference"):
        audio_path = row['audio_path']
        reference_text = normalize_text(row['raw_text'])
        
        try:
            # Транскрибация аудио
            prediction = model.transcribe(audio_path)
            
            # Нормализация предсказания
            prediction = normalize_text(prediction)
            
            predictions.append(prediction)
            references.append(reference_text)
            audio_paths.append(audio_path)
            
        except Exception as e:
            print(f"\nОшибка при обработке {audio_path}: {e}")
            predictions.append("")
            references.append(reference_text)
            audio_paths.append(audio_path)
    
    # Сохранение результатов
    results_df = pd.DataFrame({
        'audio_path': audio_paths,
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
    
    # Загрузка модели
    print(f"Загрузка модели GigaAM ({args.model})...")
    try:
        model = gigaam.load_model(args.model)
        print("✓ Модель загружена успешно!")
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
