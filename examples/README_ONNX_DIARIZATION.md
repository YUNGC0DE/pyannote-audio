# Ускорение диаризации с помощью ONNX

В этом руководстве описано, как ускорить процесс диаризации в pyannote-audio с помощью ONNX Runtime.

## Проблема

Стандартная диаризация в pyannote-audio может работать медленно, особенно на этапе извлечения эмбеддингов. Это происходит из-за:

1. Последовательной обработки каждого сегмента речи
2. Неоптимизированного извлечения признаков
3. Отсутствия батчевой обработки

## Решение

Мы оптимизировали процесс диаризации следующими способами:

1. Использование ONNX Runtime для ускорения вывода модели
2. Увеличение размера батча для параллельной обработки
3. Оптимизация процесса извлечения эмбеддингов
4. Настройка параметров ONNX для максимальной производительности

## Установка зависимостей

```bash
pip install onnxruntime  # для CPU
# или
pip install onnxruntime-gpu  # для GPU
```

## Использование

### Базовый пример

```python
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

# Инициализация с ONNX-моделью
pipeline = SpeakerDiarization(
    embedding="hbredin/wespeaker-voxceleb-resnet34-LM",  # ONNX-модель
    embedding_batch_size=32,  # Увеличенный размер батча
    segmentation_batch_size=32,
    use_onnx=True  # Включаем оптимизации ONNX
)

# Запуск диаризации
diarization = pipeline("path/to/audio.wav")

# Вывод результатов
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{turn.start:.1f}s → {turn.end:.1f}s] {speaker}")
```

### Расширенный пример с настройкой параметров ONNX

```python
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
import torch

# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры ONNX для оптимизации
onnx_params = {
    "inter_op_num_threads": 4,  # Количество потоков для параллелизации между узлами
    "intra_op_num_threads": 4,  # Количество потоков для параллелизации внутри узлов
    "execution_mode": 1,  # 0: последовательный, 1: параллельный
    "graph_optimization_level": 99,  # 0: отключено, 1: базовый, 2: расширенный, 99: все
}

# Инициализация с ONNX-моделью и параметрами
pipeline = SpeakerDiarization(
    embedding="hbredin/wespeaker-voxceleb-resnet34-LM",
    embedding_batch_size=32,
    segmentation_batch_size=32,
    use_onnx=True,
    **onnx_params
)

# Перемещаем модель на нужное устройство
pipeline.to(device)

# Запуск диаризации
diarization = pipeline("path/to/audio.wav")
```

## Сравнение производительности

Для сравнения производительности стандартной и оптимизированной диаризации можно использовать скрипт `onnx_diarization.py`:

```bash
python examples/onnx_diarization.py
```

Типичные результаты ускорения:
- CPU: 2-3x ускорение
- GPU: 3-5x ускорение

## Советы по оптимизации

1. **Размер батча**: Увеличьте `embedding_batch_size` и `segmentation_batch_size` для лучшей параллелизации. Оптимальные значения зависят от доступной памяти.

2. **Количество потоков**: Настройте `inter_op_num_threads` и `intra_op_num_threads` в соответствии с количеством ядер CPU.

3. **GPU**: Если доступен GPU, используйте его для максимальной производительности.

4. **Модель**: Модель `hbredin/wespeaker-voxceleb-resnet34-LM` обеспечивает хороший баланс между скоростью и качеством.

## Ограничения

1. Не все модели доступны в формате ONNX. В настоящее время поддерживается только модель `hbredin/wespeaker-voxceleb-resnet34-LM`.

2. Качество диаризации может незначительно отличаться от стандартной модели.

## Дополнительные ресурсы

- [ONNX Runtime документация](https://onnxruntime.ai/)
- [pyannote-audio документация](https://github.com/pyannote/pyannote-audio)
- [WeSpeaker модели](https://github.com/wenet-e2e/wespeaker) 