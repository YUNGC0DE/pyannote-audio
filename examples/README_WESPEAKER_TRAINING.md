# Обучение модели эмбеддингов WeSpeaker для ускорения диаризации

В этом руководстве описано, как обучить собственную модель эмбеддингов WeSpeaker и использовать её с ONNX для ускорения диаризации в pyannote-audio.

## Обзор

Процесс состоит из трех основных шагов:
1. Подготовка данных для обучения
2. Обучение модели WeSpeaker
3. Экспорт модели в ONNX и использование с pyannote-audio

## Требования

Установите необходимые зависимости:

```bash
pip install torch torchaudio soundfile pandas numpy matplotlib tqdm scikit-learn
pip install wespeaker  # Библиотека для обучения моделей эмбеддингов
pip install onnx onnxruntime  # Для экспорта и использования ONNX-моделей
```

## 1. Подготовка данных

Для обучения модели эмбеддингов необходимы аудиозаписи с речью разных дикторов. Вы можете использовать:

- Датасет VoxCeleb
- Собственный датасет
- Результаты диаризации

### Использование скрипта подготовки данных

Скрипт `prepare_data_for_wespeaker.py` поможет подготовить данные в нужном формате:

```bash
# Для датасета VoxCeleb
python examples/prepare_data_for_wespeaker.py \
    --dataset_type voxceleb \
    --input_dir /path/to/voxceleb \
    --output_dir ./wespeaker_data \
    --min_duration 3.0 \
    --max_duration 10.0 \
    --train_ratio 0.9 \
    --num_speakers 100  # Ограничить количество дикторов (опционально)

# Для собственного датасета
python examples/prepare_data_for_wespeaker.py \
    --dataset_type custom \
    --input_dir /path/to/custom_dataset \
    --output_dir ./wespeaker_data \
    --min_duration 3.0 \
    --max_duration 10.0

# Для результатов диаризации
python examples/prepare_data_for_wespeaker.py \
    --dataset_type diarization \
    --input_dir /path/to/diarization_results \
    --output_dir ./wespeaker_data \
    --min_duration 3.0 \
    --max_duration 10.0
```

Скрипт создаст два CSV-файла:
- `train.csv` - данные для обучения
- `val.csv` - данные для валидации

Каждый файл содержит две колонки:
- `wav_path` - путь к аудиофайлу
- `speaker_id` - идентификатор диктора

## 2. Обучение модели

Для обучения модели используйте скрипт `train_wespeaker_embeddings.py`:

```bash
python examples/train_wespeaker_embeddings.py \
    --train_data ./wespeaker_data/train.csv \
    --val_data ./wespeaker_data/val.csv \
    --model_type resnet34 \  # или ecapa_tdnn
    --embedding_size 192 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 0.001 \
    --augment \  # Включить аугментацию данных
    --checkpoint_dir ./wespeaker_checkpoints \
    --onnx_output ./wespeaker_model.onnx
```

### Параметры обучения

- `--model_type`: Тип модели (`resnet34` или `ecapa_tdnn`)
- `--embedding_size`: Размерность эмбеддинга (по умолчанию 192)
- `--batch_size`: Размер батча (по умолчанию 32)
- `--num_epochs`: Количество эпох обучения (по умолчанию 50)
- `--lr`: Скорость обучения (по умолчанию 0.001)
- `--weight_decay`: Регуляризация L2 (по умолчанию 1e-5)
- `--augment`: Включить аугментацию данных
- `--checkpoint_dir`: Директория для сохранения чекпоинтов
- `--onnx_output`: Путь для сохранения ONNX-модели

### Мониторинг обучения

В процессе обучения выводится информация о прогрессе:
- Потери на обучающей и валидационной выборках
- Точность классификации
- Equal Error Rate (EER)

После обучения создаются графики в директории с чекпоинтами:
- `training_curves.png` - графики потерь и точности

## 3. Использование модели с pyannote-audio

После обучения модель экспортируется в формат ONNX и может быть использована с pyannote-audio для ускорения диаризации.

### Использование ONNX-модели

```python
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

# Инициализация с вашей ONNX-моделью
pipeline = SpeakerDiarization(
    embedding="path/to/wespeaker_model.onnx",  # Путь к вашей ONNX-модели
    embedding_batch_size=32,
    segmentation_batch_size=32,
    use_onnx=True
)

# Параметры ONNX для оптимизации
onnx_params = {
    "inter_op_num_threads": 4,
    "intra_op_num_threads": 4,
    "execution_mode": 1,  # параллельный режим
    "graph_optimization_level": 99,  # все оптимизации
}

# Применяем параметры
for key, value in onnx_params.items():
    setattr(pipeline._embedding.session_.get_session_options(), key, value)

# Запуск диаризации
diarization = pipeline("path/to/audio.wav")
```

### Сравнение производительности

Для сравнения производительности стандартной и вашей ONNX-модели используйте скрипт `onnx_diarization.py`:

```bash
# Измените путь к вашей модели в скрипте
python examples/onnx_diarization.py
```

## Советы по обучению

1. **Данные**: Чем больше разнообразных дикторов в обучающих данных, тем лучше будет работать модель.

2. **Аугментация**: Включите аугментацию данных для улучшения обобщающей способности модели.

3. **Размер модели**: Модель `resnet34` обеспечивает хороший баланс между качеством и скоростью.

4. **Размерность эмбеддинга**: Значение 192 обычно обеспечивает хороший баланс между качеством и скоростью.

5. **Длительность аудио**: Оптимальная длительность аудиофрагментов для обучения - от 3 до 10 секунд.

6. **Количество эпох**: Обычно достаточно 30-50 эпох для достижения хороших результатов.

## Дополнительные ресурсы

- [WeSpeaker GitHub](https://github.com/wenet-e2e/wespeaker)
- [ONNX Runtime документация](https://onnxruntime.ai/)
- [pyannote-audio документация](https://github.com/pyannote/pyannote-audio)
- [VoxCeleb датасет](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) 