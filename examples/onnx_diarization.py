#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Пример использования ускоренной диаризации с ONNX
"""

import time
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

# Путь к аудиофайлу для диаризации
AUDIO_FILE = "path/to/your/audio.wav"

def benchmark_diarization(use_onnx=False, batch_size=32):
    """Сравнение производительности стандартной и ONNX-оптимизированной диаризации
    
    Parameters
    ----------
    use_onnx : bool, optional
        Использовать ли ONNX для ускорения, по умолчанию False
    batch_size : int, optional
        Размер батча для обработки, по умолчанию 32
    
    Returns
    -------
    tuple
        (diarization, elapsed_time)
    """
    
    # Определяем устройство для вычислений
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Параметры ONNX для оптимизации
    onnx_params = {
        "inter_op_num_threads": 4,
        "intra_op_num_threads": 4,
        "execution_mode": 1,  # параллельный режим
        "graph_optimization_level": 99,  # все оптимизации
    }
    
    # Инициализируем пайплайн диаризации
    if use_onnx:
        print("Инициализация диаризации с ONNX...")
        pipeline = SpeakerDiarization(
            embedding="hbredin/wespeaker-voxceleb-resnet34-LM",  # ONNX-модель
            embedding_batch_size=batch_size,
            segmentation_batch_size=batch_size,
            use_onnx=True,
            **onnx_params
        )
    else:
        print("Инициализация стандартной диаризации...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="YOUR_HF_TOKEN"  # Замените на ваш токен
        )
        # Увеличиваем размер батча для стандартной модели
        if isinstance(pipeline, SpeakerDiarization):
            pipeline.embedding_batch_size = batch_size
            pipeline.segmentation_batch_size = batch_size
    
    # Перемещаем модель на нужное устройство
    pipeline.to(device)
    
    # Запускаем диаризацию и замеряем время
    start_time = time.time()
    diarization = pipeline(AUDIO_FILE)
    elapsed_time = time.time() - start_time
    
    print(f"Время выполнения: {elapsed_time:.2f} секунд")
    
    return diarization, elapsed_time

def main():
    """Основная функция для сравнения производительности"""
    
    print("\n=== Стандартная диаризация ===")
    _, standard_time = benchmark_diarization(use_onnx=False)
    
    print("\n=== ONNX-оптимизированная диаризация ===")
    diarization, onnx_time = benchmark_diarization(use_onnx=True)
    
    # Выводим результаты сравнения
    speedup = standard_time / onnx_time if onnx_time > 0 else float('inf')
    print(f"\n=== Результаты сравнения ===")
    print(f"Стандартная диаризация: {standard_time:.2f} секунд")
    print(f"ONNX-диаризация: {onnx_time:.2f} секунд")
    print(f"Ускорение: {speedup:.2f}x")
    
    # Выводим результаты диаризации
    print("\n=== Результаты диаризации ===")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"[{turn.start:.1f}s → {turn.end:.1f}s] {speaker}")

if __name__ == "__main__":
    main() 