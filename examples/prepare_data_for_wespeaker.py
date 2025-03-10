#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для подготовки данных для обучения модели WeSpeaker
"""

import os
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import random
import shutil
from typing import List, Dict, Tuple, Optional
import multiprocessing
from functools import partial

def process_voxceleb(
    voxceleb_dir: str,
    output_dir: str,
    min_duration: float = 3.0,
    max_duration: float = 10.0,
    train_ratio: float = 0.9,
    num_speakers: Optional[int] = None,
    num_samples_per_speaker: Optional[int] = None,
    seed: int = 42,
):
    """Обрабатывает датасет VoxCeleb для обучения модели WeSpeaker
    
    Args:
        voxceleb_dir: директория с данными VoxCeleb
        output_dir: директория для сохранения обработанных данных
        min_duration: минимальная длительность аудио в секундах
        max_duration: максимальная длительность аудио в секундах
        train_ratio: доля данных для обучения
        num_speakers: количество дикторов для использования (None = все)
        num_samples_per_speaker: количество образцов на диктора (None = все)
        seed: seed для воспроизводимости
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Создаем выходные директории
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Создаем директории для обучающих и валидационных данных
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Находим всех дикторов
    voxceleb_dir = Path(voxceleb_dir)
    speaker_dirs = sorted([d for d in voxceleb_dir.glob("*") if d.is_dir()])
    
    if num_speakers is not None:
        speaker_dirs = speaker_dirs[:num_speakers]
    
    print(f"Найдено {len(speaker_dirs)} дикторов")
    
    # Собираем все аудиофайлы
    all_files = []
    for speaker_dir in tqdm(speaker_dirs, desc="Сканирование дикторов"):
        speaker_id = speaker_dir.name
        wav_files = []
        
        # Рекурсивно ищем все wav файлы
        for wav_file in speaker_dir.glob("**/*.wav"):
            try:
                info = sf.info(wav_file)
                duration = info.duration
                
                if min_duration <= duration <= max_duration:
                    wav_files.append({
                        "speaker_id": speaker_id,
                        "wav_path": str(wav_file),
                        "duration": duration,
                    })
            except Exception as e:
                print(f"Ошибка при обработке {wav_file}: {e}")
        
        # Ограничиваем количество образцов на диктора
        if num_samples_per_speaker is not None and len(wav_files) > num_samples_per_speaker:
            wav_files = random.sample(wav_files, num_samples_per_speaker)
        
        all_files.extend(wav_files)
    
    print(f"Всего найдено {len(all_files)} аудиофайлов")
    
    # Разделяем на обучающую и валидационную выборки
    random.shuffle(all_files)
    train_size = int(len(all_files) * train_ratio)
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]
    
    print(f"Обучающая выборка: {len(train_files)} файлов")
    print(f"Валидационная выборка: {len(val_files)} файлов")
    
    # Создаем CSV файлы
    train_df = pd.DataFrame(train_files)
    val_df = pd.DataFrame(val_files)
    
    train_csv_path = output_dir / "train.csv"
    val_csv_path = output_dir / "val.csv"
    
    train_df[["wav_path", "speaker_id"]].to_csv(train_csv_path, index=False)
    val_df[["wav_path", "speaker_id"]].to_csv(val_csv_path, index=False)
    
    print(f"Данные сохранены в {output_dir}")
    print(f"Обучающие данные: {train_csv_path}")
    print(f"Валидационные данные: {val_csv_path}")

def process_custom_dataset(
    audio_dir: str,
    output_dir: str,
    min_duration: float = 3.0,
    max_duration: float = 10.0,
    train_ratio: float = 0.9,
    num_speakers: Optional[int] = None,
    num_samples_per_speaker: Optional[int] = None,
    seed: int = 42,
):
    """Обрабатывает пользовательский датасет для обучения модели WeSpeaker
    
    Предполагается, что аудиофайлы организованы в структуре:
    audio_dir/
        speaker1/
            file1.wav
            file2.wav
            ...
        speaker2/
            file1.wav
            ...
    
    Args:
        audio_dir: директория с аудиоданными
        output_dir: директория для сохранения обработанных данных
        min_duration: минимальная длительность аудио в секундах
        max_duration: максимальная длительность аудио в секундах
        train_ratio: доля данных для обучения
        num_speakers: количество дикторов для использования (None = все)
        num_samples_per_speaker: количество образцов на диктора (None = все)
        seed: seed для воспроизводимости
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Создаем выходные директории
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Находим всех дикторов
    audio_dir = Path(audio_dir)
    speaker_dirs = sorted([d for d in audio_dir.glob("*") if d.is_dir()])
    
    if num_speakers is not None:
        speaker_dirs = speaker_dirs[:num_speakers]
    
    print(f"Найдено {len(speaker_dirs)} дикторов")
    
    # Собираем все аудиофайлы
    all_files = []
    for speaker_dir in tqdm(speaker_dirs, desc="Сканирование дикторов"):
        speaker_id = speaker_dir.name
        wav_files = []
        
        # Ищем все аудиофайлы
        for ext in ["*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg"]:
            for audio_file in speaker_dir.glob(ext):
                try:
                    info = sf.info(audio_file)
                    duration = info.duration
                    
                    if min_duration <= duration <= max_duration:
                        wav_files.append({
                            "speaker_id": speaker_id,
                            "wav_path": str(audio_file),
                            "duration": duration,
                        })
                except Exception as e:
                    print(f"Ошибка при обработке {audio_file}: {e}")
        
        # Ограничиваем количество образцов на диктора
        if num_samples_per_speaker is not None and len(wav_files) > num_samples_per_speaker:
            wav_files = random.sample(wav_files, num_samples_per_speaker)
        
        all_files.extend(wav_files)
    
    print(f"Всего найдено {len(all_files)} аудиофайлов")
    
    # Разделяем на обучающую и валидационную выборки
    random.shuffle(all_files)
    train_size = int(len(all_files) * train_ratio)
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]
    
    print(f"Обучающая выборка: {len(train_files)} файлов")
    print(f"Валидационная выборка: {len(val_files)} файлов")
    
    # Создаем CSV файлы
    train_df = pd.DataFrame(train_files)
    val_df = pd.DataFrame(val_files)
    
    train_csv_path = output_dir / "train.csv"
    val_csv_path = output_dir / "val.csv"
    
    train_df[["wav_path", "speaker_id"]].to_csv(train_csv_path, index=False)
    val_df[["wav_path", "speaker_id"]].to_csv(val_csv_path, index=False)
    
    print(f"Данные сохранены в {output_dir}")
    print(f"Обучающие данные: {train_csv_path}")
    print(f"Валидационные данные: {val_csv_path}")

def process_diarization_output(
    diarization_dir: str,
    output_dir: str,
    min_duration: float = 3.0,
    max_duration: float = 10.0,
    train_ratio: float = 0.9,
    num_speakers: Optional[int] = None,
    num_samples_per_speaker: Optional[int] = None,
    seed: int = 42,
):
    """Обрабатывает результаты диаризации для обучения модели WeSpeaker
    
    Предполагается, что результаты диаризации организованы в структуре:
    diarization_dir/
        file1/
            speaker1.wav
            speaker2.wav
            ...
        file2/
            speaker1.wav
            ...
    
    Args:
        diarization_dir: директория с результатами диаризации
        output_dir: директория для сохранения обработанных данных
        min_duration: минимальная длительность аудио в секундах
        max_duration: максимальная длительность аудио в секундах
        train_ratio: доля данных для обучения
        num_speakers: количество дикторов для использования (None = все)
        num_samples_per_speaker: количество образцов на диктора (None = все)
        seed: seed для воспроизводимости
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Создаем выходные директории
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Создаем временную директорию для организации данных по дикторам
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Находим все аудиофайлы и группируем их по дикторам
    diarization_dir = Path(diarization_dir)
    speaker_files = {}
    
    for file_dir in tqdm(list(diarization_dir.glob("*")), desc="Сканирование файлов"):
        if not file_dir.is_dir():
            continue
        
        for audio_file in file_dir.glob("*.wav"):
            try:
                # Извлекаем ID диктора из имени файла
                speaker_id = audio_file.stem
                
                info = sf.info(audio_file)
                duration = info.duration
                
                if min_duration <= duration <= max_duration:
                    if speaker_id not in speaker_files:
                        speaker_files[speaker_id] = []
                    
                    speaker_files[speaker_id].append({
                        "speaker_id": speaker_id,
                        "wav_path": str(audio_file),
                        "duration": duration,
                    })
            except Exception as e:
                print(f"Ошибка при обработке {audio_file}: {e}")
    
    # Ограничиваем количество дикторов
    if num_speakers is not None and len(speaker_files) > num_speakers:
        selected_speakers = random.sample(list(speaker_files.keys()), num_speakers)
        speaker_files = {k: speaker_files[k] for k in selected_speakers}
    
    print(f"Найдено {len(speaker_files)} дикторов")
    
    # Собираем все файлы
    all_files = []
    for speaker_id, files in speaker_files.items():
        # Ограничиваем количество образцов на диктора
        if num_samples_per_speaker is not None and len(files) > num_samples_per_speaker:
            files = random.sample(files, num_samples_per_speaker)
        
        all_files.extend(files)
    
    print(f"Всего найдено {len(all_files)} аудиофайлов")
    
    # Разделяем на обучающую и валидационную выборки
    random.shuffle(all_files)
    train_size = int(len(all_files) * train_ratio)
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]
    
    print(f"Обучающая выборка: {len(train_files)} файлов")
    print(f"Валидационная выборка: {len(val_files)} файлов")
    
    # Создаем CSV файлы
    train_df = pd.DataFrame(train_files)
    val_df = pd.DataFrame(val_files)
    
    train_csv_path = output_dir / "train.csv"
    val_csv_path = output_dir / "val.csv"
    
    train_df[["wav_path", "speaker_id"]].to_csv(train_csv_path, index=False)
    val_df[["wav_path", "speaker_id"]].to_csv(val_csv_path, index=False)
    
    print(f"Данные сохранены в {output_dir}")
    print(f"Обучающие данные: {train_csv_path}")
    print(f"Валидационные данные: {val_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Подготовка данных для обучения модели WeSpeaker")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["voxceleb", "custom", "diarization"],
                        help="Тип датасета (voxceleb, custom, diarization)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Директория с исходными данными")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Директория для сохранения обработанных данных")
    parser.add_argument("--min_duration", type=float, default=3.0,
                        help="Минимальная длительность аудио в секундах")
    parser.add_argument("--max_duration", type=float, default=10.0,
                        help="Максимальная длительность аудио в секундах")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Доля данных для обучения")
    parser.add_argument("--num_speakers", type=int, default=None,
                        help="Количество дикторов для использования (None = все)")
    parser.add_argument("--num_samples_per_speaker", type=int, default=None,
                        help="Количество образцов на диктора (None = все)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed для воспроизводимости")
    
    args = parser.parse_args()
    
    if args.dataset_type == "voxceleb":
        process_voxceleb(
            voxceleb_dir=args.input_dir,
            output_dir=args.output_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            train_ratio=args.train_ratio,
            num_speakers=args.num_speakers,
            num_samples_per_speaker=args.num_samples_per_speaker,
            seed=args.seed,
        )
    elif args.dataset_type == "custom":
        process_custom_dataset(
            audio_dir=args.input_dir,
            output_dir=args.output_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            train_ratio=args.train_ratio,
            num_speakers=args.num_speakers,
            num_samples_per_speaker=args.num_samples_per_speaker,
            seed=args.seed,
        )
    elif args.dataset_type == "diarization":
        process_diarization_output(
            diarization_dir=args.input_dir,
            output_dir=args.output_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            train_ratio=args.train_ratio,
            num_speakers=args.num_speakers,
            num_samples_per_speaker=args.num_samples_per_speaker,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Неизвестный тип датасета: {args.dataset_type}")

if __name__ == "__main__":
    main() 