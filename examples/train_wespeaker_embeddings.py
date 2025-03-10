#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для обучения модели эмбеддингов WeSpeaker
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import soundfile as sf
import argparse
import random
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple, Optional, Union

# Импортируем необходимые компоненты WeSpeaker
try:
    import wespeaker
    from wespeaker.models.resnet import ResNet
    from wespeaker.models.ecapa_tdnn import ECAPA_TDNN
    from wespeaker.utils.compute_eval_metrics import compute_eer
except ImportError:
    print("WeSpeaker не установлен. Устанавливаем...")
    import subprocess
    subprocess.check_call(["pip", "install", "wespeaker"])
    import wespeaker
    from wespeaker.models.resnet import ResNet
    from wespeaker.models.ecapa_tdnn import ECAPA_TDNN
    from wespeaker.utils.compute_eval_metrics import compute_eer

# Функция для извлечения признаков Fbank
def compute_fbank(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    frame_length: int = 25,
    frame_shift: int = 10,
):
    """Вычисляет признаки Fbank из аудиосигнала
    
    Args:
        waveform: аудиосигнал (одноканальный)
        sample_rate: частота дискретизации
        n_mels: количество мел-фильтров
        frame_length: длина окна в мс
        frame_shift: сдвиг окна в мс
        
    Returns:
        fbank: признаки Fbank размера (num_frames, n_mels)
    """
    try:
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
        
        waveform_tensor = torch.from_numpy(waveform).float()
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)
            
        fbank = kaldi.fbank(
            waveform_tensor,
            num_mel_bins=n_mels,
            frame_length=frame_length,
            frame_shift=frame_shift,
            sample_frequency=sample_rate,
            dither=0.0,
        )
        return fbank.numpy()
    except ImportError:
        print("torchaudio не установлен. Устанавливаем...")
        import subprocess
        subprocess.check_call(["pip", "install", "torchaudio"])
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
        
        waveform_tensor = torch.from_numpy(waveform).float()
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)
            
        fbank = kaldi.fbank(
            waveform_tensor,
            num_mel_bins=n_mels,
            frame_length=frame_length,
            frame_shift=frame_shift,
            sample_frequency=sample_rate,
            dither=0.0,
        )
        return fbank.numpy()

# Класс датасета для обучения
class SpeakerDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        max_frames: int = 300,
        min_frames: int = 200,
        sample_rate: int = 16000,
        n_mels: int = 80,
        augment: bool = False,
    ):
        """Датасет для обучения модели эмбеддингов
        
        Args:
            data_file: путь к файлу с данными (формат: wav_path,speaker_id)
            max_frames: максимальное количество фреймов
            min_frames: минимальное количество фреймов
            sample_rate: частота дискретизации
            n_mels: количество мел-фильтров
            augment: использовать аугментацию данных
        """
        self.data = pd.read_csv(data_file)
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.augment = augment
        
        # Создаем маппинг speaker_id -> индекс
        unique_speakers = sorted(self.data['speaker_id'].unique())
        self.speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
        self.num_speakers = len(unique_speakers)
        
        print(f"Загружено {len(self.data)} записей, {self.num_speakers} уникальных дикторов")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        wav_path = row['wav_path']
        speaker_id = row['speaker_id']
        
        # Загружаем аудио
        waveform, sr = sf.read(wav_path)
        if sr != self.sample_rate:
            # Ресемплируем, если частота дискретизации не совпадает
            try:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
            except ImportError:
                print("librosa не установлен. Устанавливаем...")
                import subprocess
                subprocess.check_call(["pip", "install", "librosa"])
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
        
        # Применяем аугментацию (если включена)
        if self.augment:
            # Добавляем шум
            if random.random() < 0.5:
                noise_level = random.uniform(0.0, 0.1)
                noise = np.random.randn(len(waveform)) * noise_level
                waveform = waveform + noise
            
            # Изменяем скорость
            if random.random() < 0.3:
                speed_factor = random.uniform(0.9, 1.1)
                try:
                    import librosa
                    waveform = librosa.effects.time_stretch(waveform, rate=speed_factor)
                except ImportError:
                    pass
        
        # Вычисляем признаки Fbank
        fbank = compute_fbank(
            waveform, 
            sample_rate=self.sample_rate, 
            n_mels=self.n_mels
        )
        
        # Обрезаем или дополняем до нужной длины
        num_frames = fbank.shape[0]
        if num_frames >= self.max_frames:
            # Случайно выбираем отрезок нужной длины
            start = random.randint(0, num_frames - self.max_frames)
            fbank = fbank[start:start + self.max_frames]
        elif num_frames < self.min_frames:
            # Если слишком короткий, дублируем фреймы
            repeat_factor = self.min_frames // num_frames + 1
            fbank = np.tile(fbank, (repeat_factor, 1))[:self.min_frames]
        else:
            # Дополняем нулями
            pad_length = self.max_frames - num_frames
            fbank = np.pad(fbank, ((0, pad_length), (0, 0)), mode='constant')
        
        # Нормализация
        fbank = (fbank - fbank.mean(axis=0)) / (fbank.std(axis=0) + 1e-6)
        
        # Преобразуем в тензор
        fbank_tensor = torch.from_numpy(fbank).float()
        label = self.speaker_to_idx[speaker_id]
        
        return fbank_tensor, label

# Функция для создания модели
def create_model(model_type: str, num_classes: int, embedding_size: int = 192):
    """Создает модель WeSpeaker
    
    Args:
        model_type: тип модели ('resnet34', 'ecapa_tdnn')
        num_classes: количество классов (дикторов)
        embedding_size: размерность эмбеддинга
        
    Returns:
        model: модель WeSpeaker
    """
    if model_type == 'resnet34':
        model = ResNet(
            feat_dim=80,  # размерность входных признаков (n_mels)
            embedding_size=embedding_size,
            num_classes=num_classes,
            resnet_type='34'
        )
    elif model_type == 'ecapa_tdnn':
        model = ECAPA_TDNN(
            feat_dim=80,
            embedding_size=embedding_size,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
    
    return model

# Функция для обучения модели
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    checkpoint_dir: str = './checkpoints',
    log_interval: int = 10,
):
    """Обучает модель WeSpeaker
    
    Args:
        model: модель WeSpeaker
        train_loader: загрузчик обучающих данных
        val_loader: загрузчик валидационных данных
        device: устройство для обучения
        num_epochs: количество эпох
        lr: скорость обучения
        weight_decay: регуляризация L2
        checkpoint_dir: директория для сохранения чекпоинтов
        log_interval: интервал для логирования
    """
    # Создаем директорию для чекпоинтов
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Определяем оптимизатор и функцию потерь
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    
    # Логи для отслеживания прогресса
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs} [Обучение]")
        for batch_idx, (features, labels) in enumerate(pbar):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, embeddings = model(features, return_embedding=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % log_interval == 0:
                pbar.set_postfix({
                    'loss': train_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Эпоха {epoch+1}/{num_epochs} [Валидация]")
            for batch_idx, (features, labels) in enumerate(pbar):
                features, labels = features.to(device), labels.to(device)
                
                outputs, embeddings = model(features, return_embedding=True)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Сохраняем эмбеддинги для вычисления EER
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': val_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Вычисляем EER на валидационном наборе
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels)
        
        # Создаем пары для вычисления EER
        num_samples = min(5000, len(all_embeddings))  # Ограничиваем количество пар для скорости
        indices = np.random.choice(len(all_embeddings), num_samples, replace=False)
        
        embeddings_subset = all_embeddings[indices]
        labels_subset = all_labels[indices]
        
        scores = []
        labels = []
        
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                # Косинусное сходство
                score = np.dot(embeddings_subset[i], embeddings_subset[j]) / (
                    np.linalg.norm(embeddings_subset[i]) * np.linalg.norm(embeddings_subset[j])
                )
                scores.append(score)
                # 1 если тот же диктор, 0 если разные
                label = 1 if labels_subset[i] == labels_subset[j] else 0
                labels.append(label)
        
        # Вычисляем EER
        eer, _ = compute_eer(np.array(scores), np.array(labels))
        
        print(f"Эпоха {epoch+1}/{num_epochs}:")
        print(f"  Обучение: Потери = {train_loss:.4f}, Точность = {train_acc:.2f}%")
        print(f"  Валидация: Потери = {val_loss:.4f}, Точность = {val_acc:.2f}%, EER = {eer*100:.2f}%")
        
        # Обновляем планировщик скорости обучения
        scheduler.step(val_loss)
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'eer': eer,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  Сохранена лучшая модель с потерями {val_loss:.4f}")
        
        # Сохраняем чекпоинт каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'eer': eer,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Строим графики
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Обучение')
    plt.plot(val_losses, label='Валидация')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.title('Потери')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    plt.title('Точность на валидации')
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_curves.png'))
    plt.close()
    
    return model

# Функция для экспорта модели в ONNX
def export_to_onnx(
    model: nn.Module,
    output_path: str,
    embedding_size: int = 192,
    input_shape: Tuple[int, int, int] = (1, 300, 80),
):
    """Экспортирует модель в формат ONNX
    
    Args:
        model: обученная модель WeSpeaker
        output_path: путь для сохранения ONNX-модели
        embedding_size: размерность эмбеддинга
        input_shape: форма входных данных (batch_size, num_frames, n_mels)
    """
    model.eval()
    
    # Создаем тестовый вход
    dummy_input = torch.randn(input_shape)
    
    # Создаем директорию, если не существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Экспортируем модель
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['feats'],
        output_names=['embs'],
        dynamic_axes={
            'feats': {0: 'batch_size', 1: 'num_frames'},
            'embs': {0: 'batch_size'}
        }
    )
    
    print(f"Модель успешно экспортирована в {output_path}")
    
    # Проверяем модель
    try:
        import onnx
        import onnxruntime as ort
        
        # Загружаем и проверяем модель
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Создаем сессию для инференса
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 4
        sess_options.intra_op_num_threads = 4
        sess = ort.InferenceSession(
            output_path, 
            sess_options=sess_options, 
            providers=['CPUExecutionProvider']
        )
        
        # Проверяем инференс
        ort_inputs = {sess.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = sess.run(None, ort_inputs)
        
        print(f"Проверка ONNX модели успешна. Размерность выхода: {ort_outputs[0].shape}")
    except ImportError:
        print("onnx или onnxruntime не установлены. Пропускаем проверку.")

def main():
    parser = argparse.ArgumentParser(description="Обучение модели эмбеддингов WeSpeaker")
    parser.add_argument("--train_data", type=str, required=True, help="Путь к CSV-файлу с обучающими данными")
    parser.add_argument("--val_data", type=str, required=True, help="Путь к CSV-файлу с валидационными данными")
    parser.add_argument("--model_type", type=str, default="resnet34", choices=["resnet34", "ecapa_tdnn"], help="Тип модели")
    parser.add_argument("--embedding_size", type=int, default=192, help="Размерность эмбеддинга")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча")
    parser.add_argument("--num_epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=0.001, help="Скорость обучения")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Регуляризация L2")
    parser.add_argument("--max_frames", type=int, default=300, help="Максимальное количество фреймов")
    parser.add_argument("--min_frames", type=int, default=200, help="Минимальное количество фреймов")
    parser.add_argument("--n_mels", type=int, default=80, help="Количество мел-фильтров")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Частота дискретизации")
    parser.add_argument("--augment", action="store_true", help="Использовать аугментацию данных")
    parser.add_argument("--checkpoint_dir", type=str, default="./wespeaker_checkpoints", help="Директория для сохранения чекпоинтов")
    parser.add_argument("--num_workers", type=int, default=4, help="Количество рабочих процессов для загрузки данных")
    parser.add_argument("--onnx_output", type=str, default="./wespeaker_model.onnx", help="Путь для сохранения ONNX-модели")
    
    args = parser.parse_args()
    
    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Создаем датасеты
    train_dataset = SpeakerDataset(
        data_file=args.train_data,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        augment=args.augment,
    )
    
    val_dataset = SpeakerDataset(
        data_file=args.val_data,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        augment=False,  # Без аугментации для валидации
    )
    
    # Создаем загрузчики данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Создаем модель
    model = create_model(
        model_type=args.model_type,
        num_classes=train_dataset.num_speakers,
        embedding_size=args.embedding_size,
    )
    model = model.to(device)
    
    # Выводим информацию о модели
    print(f"Модель: {args.model_type}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Обучаем модель
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Загружаем лучшую модель
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Экспортируем модель в ONNX
    export_to_onnx(
        model=model,
        output_path=args.onnx_output,
        embedding_size=args.embedding_size,
        input_shape=(1, args.max_frames, args.n_mels),
    )
    
    print("Обучение и экспорт модели завершены!")

if __name__ == "__main__":
    main() 