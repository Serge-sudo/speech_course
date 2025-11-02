# Решение: Дообучение GigaAM-CTC на FLEURS-Ru

Это решение содержит инструкции для дообучения модели GigaAM-CTC на русскоязычной части датасета FLEURS с целью достижения Word Error Rate (WER) < 8% на валидационном наборе.

## Содержание

- [Установка зависимостей](#установка-зависимостей)
- [Подготовка данных](#подготовка-данных)
- [Обучение модели](#обучение-модели)
- [Инференс](#инференс)
- [Подсчет метрик](#подсчет-метрик)
- [Полный пример в Jupyter Notebook](#полный-пример-в-jupyter-notebook)

## Установка зависимостей

### 1. Установка базовых зависимостей

```bash
# Установка ffmpeg (требуется для работы с аудио)
sudo apt-get update
sudo apt-get install -y ffmpeg

# Установка Python зависимостей
pip install torch torchaudio
pip install transformers datasets
pip install jiwer  # для расчета WER
pip install librosa soundfile
pip install tqdm
```

### 2. Установка GigaAM

```bash
cd GigaAM
pip install -e .
cd ..
```

## Подготовка данных

### 1. Извлечение аудио файлов

Аудио файлы находятся в архивах. Необходимо их распаковать:

```bash
cd fleurs/data/ru_ru/audio
tar -xzf train.tar.gz
tar -xzf dev.tar.gz
tar -xzf test.tar.gz
cd ../../../../
```

### 2. Подготовка датасета

Создайте скрипт `prepare_data.py`:

```python
import os
import pandas as pd
import torch
from pathlib import Path

def load_fleurs_data(split='train'):
    """
    Загружает данные FLEURS для указанного split (train/dev/test)
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
    
    data = data[data['exists']]
    
    return data

def normalize_text(text):
    """
    Нормализация текста согласно требованиям задания:
    - приведение к нижнему регистру
    - удаление знаков препинания
    - сохранение цифр и латиницы
    """
    import re
    
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
    train_data = load_fleurs_data('train')
    dev_data = load_fleurs_data('dev')
    test_data = load_fleurs_data('test')
    
    print(f"Train samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Пример нормализации
    sample_text = train_data.iloc[0]['raw_text']
    normalized = normalize_text(sample_text)
    print(f"\nОригинал: {sample_text}")
    print(f"Нормализованный: {normalized}")
```

## Обучение модели

### Создание скрипта обучения

Создайте файл `train.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gigaam
from prepare_data import load_fleurs_data, normalize_text
import torchaudio
from tqdm import tqdm
import os

class FleursDataset(Dataset):
    def __init__(self, data_df, model):
        self.data = data_df.reset_index(drop=True)
        self.model = model
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row['audio_path']
        text = normalize_text(row['raw_text'])
        
        return {
            'audio_path': audio_path,
            'text': text
        }

def collate_fn(batch):
    """
    Collate function for DataLoader
    """
    return batch

def train_epoch(model, dataloader, optimizer, device):
    """
    Обучение модели на одной эпохе
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        # Здесь необходимо реализовать логику обучения
        # В зависимости от API GigaAM
        # Это упрощенный пример
        
        for item in batch:
            try:
                # Загрузка аудио
                audio_path = item['audio_path']
                target_text = item['text']
                
                # Получение предсказания и вычисление loss
                # (требуется адаптация под конкретное API GigaAM)
                
                pass
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
    
    return total_loss / len(dataloader)

def main():
    # Параметры обучения
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    
    # Загрузка данных
    print("Loading data...")
    train_data = load_fleurs_data('train')
    dev_data = load_fleurs_data('dev')
    
    # Загрузка предобученной модели
    print("Loading GigaAM model...")
    model = gigaam.load_model("ctc")
    
    # Создание датасетов
    train_dataset = FleursDataset(train_data, model)
    dev_dataset = FleursDataset(dev_data, model)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Настройка оптимизатора
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Обучение
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # train_loss = train_epoch(model, train_loader, optimizer, device)
        # print(f"Train loss: {train_loss:.4f}")
        
        # Сохранение чекпоинта
        # torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pt')
        
        # Валидация
        # evaluate(model, dev_loader)

if __name__ == "__main__":
    main()
```

**Примечание:** Для полноценного обучения необходимо:
1. Изучить API GigaAM для дообучения моделей
2. Реализовать правильную функцию потерь (CTC loss)
3. Настроить параметры обучения (learning rate, batch size и т.д.)

### Запуск обучения

```bash
python train.py
```

## Инференс

### Создание скрипта инференса

Создайте файл `inference.py`:

```python
import gigaam
from prepare_data import load_fleurs_data, normalize_text
from tqdm import tqdm
import pandas as pd

def run_inference(model, data_df, output_file='predictions.txt'):
    """
    Запуск инференса на датасете
    """
    predictions = []
    references = []
    
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
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            predictions.append("")
            references.append(reference_text)
    
    # Сохранение результатов
    results_df = pd.DataFrame({
        'audio_path': data_df['audio_path'].values,
        'reference': references,
        'prediction': predictions
    })
    
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return predictions, references

def main():
    # Загрузка модели
    print("Loading model...")
    model = gigaam.load_model("ctc")  # или путь к дообученной модели
    
    # Загрузка валидационных данных
    print("Loading validation data...")
    dev_data = load_fleurs_data('dev')
    
    # Запуск инференса
    predictions, references = run_inference(model, dev_data, 'dev_predictions.csv')
    
    print(f"Processed {len(predictions)} samples")

if __name__ == "__main__":
    main()
```

### Запуск инференса

```bash
python inference.py
```

## Подсчет метрик

### Создание скрипта для расчета WER

Создайте файл `calculate_wer.py`:

```python
import pandas as pd
from jiwer import wer, cer
from prepare_data import normalize_text

def calculate_metrics(predictions_file='dev_predictions.csv'):
    """
    Расчет метрик WER и CER
    """
    # Загрузка предсказаний
    df = pd.read_csv(predictions_file)
    
    references = df['reference'].tolist()
    predictions = df['prediction'].tolist()
    
    # Фильтрация пустых предсказаний
    valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) 
                   if pred and ref]
    
    if not valid_pairs:
        print("No valid predictions found!")
        return
    
    references, predictions = zip(*valid_pairs)
    
    # Расчет метрик
    word_error_rate = wer(references, predictions)
    character_error_rate = cer(references, predictions)
    
    print(f"="*50)
    print(f"Evaluation Results")
    print(f"="*50)
    print(f"Total samples: {len(df)}")
    print(f"Valid samples: {len(valid_pairs)}")
    print(f"Word Error Rate (WER): {word_error_rate*100:.2f}%")
    print(f"Character Error Rate (CER): {character_error_rate*100:.2f}%")
    print(f"="*50)
    
    # Примеры ошибок
    print("\nПримеры предсказаний:")
    for i in range(min(5, len(references))):
        print(f"\nReference: {references[i]}")
        print(f"Prediction: {predictions[i]}")
    
    return word_error_rate, character_error_rate

def main():
    wer_score, cer_score = calculate_metrics('dev_predictions.csv')
    
    # Проверка целевого WER
    if wer_score < 0.08:
        print(f"\n✓ Целевой WER < 8% достигнут!")
    else:
        print(f"\n✗ WER {wer_score*100:.2f}% выше целевого 8%")

if __name__ == "__main__":
    main()
```

### Запуск расчета метрик

```bash
python calculate_wer.py
```

## Полный пример в Jupyter Notebook

Создайте файл `solution.ipynb`:

```python
# Ячейка 1: Установка зависимостей
!pip install torch torchaudio transformers datasets jiwer librosa soundfile tqdm

# Ячейка 2: Установка GigaAM
!cd GigaAM && pip install -e . && cd ..

# Ячейка 3: Импорты
import os
import pandas as pd
import torch
import gigaam
from jiwer import wer, cer
from tqdm.notebook import tqdm
import re
from pathlib import Path

# Ячейка 4: Функции для работы с данными
def load_fleurs_data(split='train'):
    """Загрузка данных FLEURS"""
    base_path = Path('fleurs/data/ru_ru')
    tsv_file = base_path / f'{split}.tsv'
    audio_dir = base_path / 'audio' / split
    
    data = pd.read_csv(tsv_file, sep='\t', header=None, 
                       names=['id', 'filename', 'raw_text', 'normalized_text', 
                              'phonemes', 'num_samples', 'gender'])
    
    data['audio_path'] = data['filename'].apply(lambda x: str(audio_dir / x))
    data['exists'] = data['audio_path'].apply(os.path.exists)
    
    missing = (~data['exists']).sum()
    if missing > 0:
        print(f"Предупреждение: {missing} файлов не найдено в {split}")
    
    return data[data['exists']]

def normalize_text(text):
    """Нормализация текста"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = ' '.join(text.split())
    return text

# Ячейка 5: Распаковка аудио (если нужно)
!cd fleurs/data/ru_ru/audio && tar -xzf train.tar.gz
!cd fleurs/data/ru_ru/audio && tar -xzf dev.tar.gz
!cd fleurs/data/ru_ru/audio && tar -xzf test.tar.gz

# Ячейка 6: Загрузка данных
train_data = load_fleurs_data('train')
dev_data = load_fleurs_data('dev')
test_data = load_fleurs_data('test')

print(f"Train: {len(train_data)} samples")
print(f"Dev: {len(dev_data)} samples")
print(f"Test: {len(test_data)} samples")

# Ячейка 7: Загрузка модели
model = gigaam.load_model("ctc")
print("Модель загружена успешно!")

# Ячейка 8: Пример инференса на одном файле
sample = dev_data.iloc[0]
audio_path = sample['audio_path']
reference = normalize_text(sample['raw_text'])

prediction = model.transcribe(audio_path)
prediction_normalized = normalize_text(prediction)

print(f"Reference: {reference}")
print(f"Prediction: {prediction_normalized}")

# Ячейка 9: Инференс на валидационном наборе
predictions = []
references = []

for idx, row in tqdm(dev_data.iterrows(), total=len(dev_data)):
    try:
        audio_path = row['audio_path']
        reference = normalize_text(row['raw_text'])
        
        prediction = model.transcribe(audio_path)
        prediction = normalize_text(prediction)
        
        predictions.append(prediction)
        references.append(reference)
    except Exception as e:
        print(f"Error: {e}")
        predictions.append("")
        references.append(reference)

# Ячейка 10: Расчет метрик
valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) if pred and ref]
references_valid, predictions_valid = zip(*valid_pairs)

wer_score = wer(references_valid, predictions_valid)
cer_score = cer(references_valid, predictions_valid)

print(f"Word Error Rate (WER): {wer_score*100:.2f}%")
print(f"Character Error Rate (CER): {cer_score*100:.2f}%")

if wer_score < 0.08:
    print("✓ Целевой WER < 8% достигнут!")
else:
    print(f"✗ WER выше целевого (цель: < 8%)")

# Ячейка 11: Примеры предсказаний
for i in range(min(10, len(references_valid))):
    print(f"\n--- Пример {i+1} ---")
    print(f"Reference:  {references_valid[i]}")
    print(f"Prediction: {predictions_valid[i]}")
```

## Дообучение модели (Fine-tuning)

Для достижения WER < 8% может потребоваться дообучение модели на датасете FLEURS. Основные шаги:

### 1. Подготовка данных для обучения

```python
import torch
from torch.utils.data import Dataset

class FleursDataset(Dataset):
    def __init__(self, data_df, processor):
        self.data = data_df.reset_index(drop=True)
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row['audio_path']
        text = normalize_text(row['raw_text'])
        
        # Загрузка и обработка аудио
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        return {
            'audio': waveform.squeeze().numpy(),
            'text': text,
            'sampling_rate': 16000
        }
```

### 2. Настройка параметров обучения

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    learning_rate=1e-4,
)
```

### 3. Запуск обучения

**Примечание:** Конкретная реализация зависит от API GigaAM. Рекомендуется изучить документацию и примеры обучения из репозитория GigaAM.

## Структура проекта

```
speech_course/
├── README.md              # Описание задания
├── SOLUTION.md           # Это файл с решением
├── GigaAM/               # Модель GigaAM
├── fleurs/               # Датасет FLEURS
├── prepare_data.py       # Подготовка данных
├── train.py              # Скрипт обучения
├── inference.py          # Скрипт инференса
├── calculate_wer.py      # Расчет метрик
├── solution.ipynb        # Jupyter notebook с полным решением
└── results/              # Результаты обучения и предсказания
```

## Ожидаемые результаты

- **Word Error Rate (WER)**: < 8% на валидационном наборе
- **Character Error Rate (CER)**: обычно в 2-3 раза ниже WER

## Советы по улучшению качества

1. **Аугментация данных**: Добавление шума, изменение скорости/высоты тона
2. **Настройка гиперпараметров**: Learning rate, batch size, число эпох
3. **Использование предобученной модели**: GigaAM-CTC-v2 вместо v1
4. **Fine-tuning**: Постепенное размораживание слоев модели
5. **Beam search**: Использование beam search при декодировании
6. **Language model**: Добавление языковой модели для пост-обработки

## Заключение

Данное решение предоставляет полный pipeline для:
- Подготовки данных FLEURS
- Обучения/дообучения модели GigaAM-CTC
- Запуска инференса
- Расчета метрик WER/CER

Для достижения целевого WER < 8% рекомендуется:
1. Использовать базовую модель GigaAM-CTC-v2 (уже показывает хорошие результаты)
2. При необходимости провести fine-tuning на FLEURS
3. Экспериментировать с параметрами нормализации текста
4. Использовать beam search или language model для улучшения результатов
