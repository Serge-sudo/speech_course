# Быстрый старт

Краткое руководство по запуску решения.

## Установка

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Установка GigaAM
cd GigaAM
pip install -e .
cd ..
```

## Подготовка данных

```bash
# Распаковка аудио файлов и проверка данных
python prepare_data.py
```

## Быстрый запуск

### Вариант 1: Jupyter Notebook (рекомендуется)

Откройте и запустите `solution.ipynb` - ноутбук содержит полное решение с комментариями.

### Вариант 2: Скрипты Python

```bash
# 1. Подготовка данных (распаковка архивов)
python prepare_data.py

# 2. Запуск инференса на валидационном наборе
python inference.py --split dev --model ctc --output dev_predictions.csv

# 3. Расчет метрик WER и CER
python calculate_wer.py --input dev_predictions.csv --examples 10
```

## Параметры

### inference.py

- `--split`: выбор набора данных (`train`, `dev`, `test`) - по умолчанию `dev`
- `--model`: версия модели GigaAM (`ctc`, `v2_ctc`, `rnnt`, `v2_rnnt`) - по умолчанию `ctc`
- `--output`: имя выходного файла - по умолчанию `{split}_predictions.csv`
- `--limit`: ограничить количество образцов для обработки

### calculate_wer.py

- `--input`: путь к CSV файлу с предсказаниями - по умолчанию `dev_predictions.csv`
- `--examples`: количество примеров для показа - по умолчанию `5`

## Примеры использования

```bash
# Инференс на 100 образцах из dev набора
python inference.py --split dev --limit 100

# Инференс с моделью RNNT (более точная)
python inference.py --model rnnt --output dev_rnnt_predictions.csv

# Расчет метрик с показом 10 примеров
python calculate_wer.py --input dev_predictions.csv --examples 10

# Инференс на тестовом наборе
python inference.py --split test --output test_predictions.csv
```

## Ожидаемые результаты

При использовании предобученной модели GigaAM-CTC v2:
- **WER на валидации**: ожидается < 8%
- Модель RNNT обычно показывает лучшие результаты (на ~1-2% ниже WER)

## Структура выходных файлов

После запуска инференса создается CSV файл со следующими колонками:
- `audio_path`: путь к аудио файлу
- `reference`: нормализованная эталонная транскрипция
- `prediction`: нормализованное предсказание модели

## Решение проблем

### Ошибка "audio files not found"
```bash
# Убедитесь, что архивы распакованы
python prepare_data.py
```

### Ошибка при установке GigaAM
```bash
# Убедитесь, что ffmpeg установлен
sudo apt-get install ffmpeg
```

### Низкое качество (WER > 8%)
- Попробуйте модель `v2_rnnt` (более точная)
- Проверьте правильность нормализации текста
- Убедитесь, что используете последнюю версию GigaAM

## Дополнительная информация

Полное описание решения см. в `SOLUTION.md`
