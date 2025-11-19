# Домашнее задание #3: Обучение Speaker Encoder с Angular Margin Loss

- в файле [speaker_recognition.py](./speaker_recognition.py) реализован бейзлайн пайплайна обучения
- вам необходимо:
    - скачать и распаковать [train](https://disk.yandex.ru/d/HBEZ7MbCepdgzw) и [dev](https://disk.yandex.ru/d/LKDitPbiU5XS2A) части датасета
    - реализовать класс `AngularMarginSoftmax`
    - реализовать метод `evaluate` для подсчета `Equal Error Rate`
    - обучить модель с AngularMarginSoftmax, добиться сходимости `train top1 accuracy > 0.99`, `dev EER < 4.5%`
    - подготовить отчет о проделанной работе (с графиками обучения, подобранными параметрами, кодом)

**Дедлайн:** 19 ноября 2025, 23:59 (МСК)  
После указанного срока итоговый балл умножается на **0.7**.

# Запуск
```
python train_angular_margin.py
```

**Мониторинг**
```bash
tensorboard --logdir=./logs
```

ОТЧЕТ

1. Достигнут результат 
    train top1 accuracy > 0.99 [папка train_tensorboard]
    dev EER < 4.5% [файл visualize.ipynb]
2. Графики обучения в папке train_tensorboard
3. Файл visualize.ipynb заполнен
4. Параметры обучения находится в train_angular_margin.py
5. Код находится в файле speaker_recognition.py

По результатам работы используя angular margin на данных получилось получить ERR **4.4**.


