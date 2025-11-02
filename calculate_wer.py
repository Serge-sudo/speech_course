"""
Скрипт для расчета метрик WER и CER на основе предсказаний модели.
"""

import pandas as pd
from jiwer import wer, cer
from prepare_data import normalize_text
import argparse


def calculate_metrics(predictions_file='dev_predictions.csv', verbose=True):
    """
    Расчет метрик WER и CER
    
    Args:
        predictions_file: путь к CSV файлу с предсказаниями
        verbose: выводить ли детальную информацию
    
    Returns:
        word_error_rate: значение WER (0-1)
        character_error_rate: значение CER (0-1)
    """
    # Загрузка предсказаний
    try:
        df = pd.read_csv(predictions_file)
    except Exception as e:
        print(f"Ошибка при загрузке файла {predictions_file}: {e}")
        return None, None
    
    references = df['reference'].tolist()
    predictions = df['prediction'].tolist()
    
    # Фильтрация пустых предсказаний
    valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) 
                   if pred and ref and isinstance(pred, str) and isinstance(ref, str)]
    
    if not valid_pairs:
        print("Не найдено валидных предсказаний!")
        return None, None
    
    references_valid, predictions_valid = zip(*valid_pairs)
    
    # Расчет метрик
    word_error_rate = wer(references_valid, predictions_valid)
    character_error_rate = cer(references_valid, predictions_valid)
    
    if verbose:
        print(f"{'='*60}")
        print(f"Результаты оценки")
        print(f"{'='*60}")
        print(f"Файл с предсказаниями: {predictions_file}")
        print(f"Всего образцов: {len(df)}")
        print(f"Валидных образцов: {len(valid_pairs)}")
        print(f"Невалидных образцов: {len(df) - len(valid_pairs)}")
        print(f"\nМетрики:")
        print(f"  Word Error Rate (WER):      {word_error_rate*100:.2f}%")
        print(f"  Character Error Rate (CER): {character_error_rate*100:.2f}%")
        print(f"{'='*60}")
        
        # Проверка достижения целевого WER
        if word_error_rate < 0.08:
            print(f"\n✓ УСПЕХ! Целевой WER < 8% достигнут!")
            print(f"  Текущий WER: {word_error_rate*100:.2f}%")
        else:
            print(f"\n✗ Целевой WER не достигнут")
            print(f"  Текущий WER: {word_error_rate*100:.2f}%")
            print(f"  Цель: < 8.00%")
            print(f"  Разница: +{(word_error_rate - 0.08)*100:.2f}%")
        print(f"{'='*60}")
    
    return word_error_rate, character_error_rate


def show_examples(predictions_file='dev_predictions.csv', num_examples=5):
    """
    Показать примеры предсказаний
    
    Args:
        predictions_file: путь к CSV файлу с предсказаниями
        num_examples: количество примеров для показа
    """
    df = pd.read_csv(predictions_file)
    
    print(f"\nПримеры предсказаний:")
    print(f"{'='*60}")
    
    # Показать примеры с правильными предсказаниями
    correct_examples = df[df['reference'] == df['prediction']].head(num_examples // 2)
    if len(correct_examples) > 0:
        print("\nПравильные предсказания:")
        for i, row in correct_examples.iterrows():
            print(f"\n[Пример {i+1}]")
            print(f"Reference:  {row['reference']}")
            print(f"Prediction: {row['prediction']}")
            print(f"✓ СОВПАДЕНИЕ")
    
    # Показать примеры с ошибками
    incorrect_examples = df[df['reference'] != df['prediction']].head(num_examples - len(correct_examples))
    if len(incorrect_examples) > 0:
        print("\nПредсказания с ошибками:")
        for i, row in incorrect_examples.iterrows():
            print(f"\n[Пример {i+1}]")
            print(f"Reference:  {row['reference']}")
            print(f"Prediction: {row['prediction']}")
            
            # Подсчет количества различий в словах
            ref_words = set(row['reference'].split())
            pred_words = set(row['prediction'].split())
            missing_words = ref_words - pred_words
            extra_words = pred_words - ref_words
            
            if missing_words:
                print(f"Пропущены: {', '.join(missing_words)}")
            if extra_words:
                print(f"Лишние: {', '.join(extra_words)}")


def main():
    parser = argparse.ArgumentParser(description='Расчет метрик WER и CER')
    parser.add_argument('--input', type=str, default='dev_predictions.csv',
                       help='Путь к CSV файлу с предсказаниями')
    parser.add_argument('--examples', type=int, default=5,
                       help='Количество примеров для показа')
    parser.add_argument('--no-verbose', action='store_true',
                       help='Не выводить детальную информацию')
    
    args = parser.parse_args()
    
    # Расчет метрик
    wer_score, cer_score = calculate_metrics(
        args.input, 
        verbose=not args.no_verbose
    )
    
    if wer_score is None:
        return
    
    # Показать примеры
    if args.examples > 0:
        show_examples(args.input, args.examples)


if __name__ == "__main__":
    main()
