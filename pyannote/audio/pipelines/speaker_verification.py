# MIT License
#
# Copyright (c) 2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from functools import cached_property
from pathlib import Path
import os
from typing import Optional, Text, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from torch.nn.utils.rnn import pad_sequence

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.inference import BaseInference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import CACHE_DIR
from pyannote.audio.pipelines.utils import PipelineModel, get_model

try:
    from speechbrain.inference import (
        EncoderClassifier as SpeechBrain_EncoderClassifier,
    )

    SPEECHBRAIN_IS_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_IS_AVAILABLE = False

try:
    from nemo.collections.asr.models import (
        EncDecSpeakerLabelModel as NeMo_EncDecSpeakerLabelModel,
    )

    NEMO_IS_AVAILABLE = True
except ImportError:
    NEMO_IS_AVAILABLE = False

try:
    import onnxruntime as ort

    ONNX_IS_AVAILABLE = True
except ImportError:
    ONNX_IS_AVAILABLE = False


class NeMoPretrainedSpeakerEmbedding(BaseInference):
    def __init__(
        self,
        embedding: Text = "nvidia/speakerverification_en_titanet_large",
        device: Optional[torch.device] = None,
    ):
        if not NEMO_IS_AVAILABLE:
            raise ImportError(
                f"'NeMo' must be installed to use '{embedding}' embeddings. "
                "Visit https://nvidia.github.io/NeMo/ for installation instructions."
            )

        super().__init__()
        self.embedding = embedding
        self.device = device or torch.device("cpu")

        self.model_ = NeMo_EncDecSpeakerLabelModel.from_pretrained(self.embedding)
        self.model_.freeze()
        self.model_.to(self.device)

    def to(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        self.model_.to(device)
        self.device = device
        return self

    @cached_property
    def sample_rate(self) -> int:
        return self.model_._cfg.train_ds.get("sample_rate", 16000)

    @cached_property
    def dimension(self) -> int:
        input_signal = torch.rand(1, self.sample_rate).to(self.device)
        input_signal_length = torch.tensor([self.sample_rate]).to(self.device)
        _, embeddings = self.model_(
            input_signal=input_signal, input_signal_length=input_signal_length
        )
        _, dimension = embeddings.shape
        return dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        lower, upper = 2, round(0.5 * self.sample_rate)
        middle = (lower + upper) // 2
        while lower + 1 < upper:
            try:
                input_signal = torch.rand(1, middle).to(self.device)
                input_signal_length = torch.tensor([middle]).to(self.device)

                _ = self.model_(
                    input_signal=input_signal, input_signal_length=input_signal_length
                )

                upper = middle
            except RuntimeError:
                lower = middle

            middle = (lower + upper) // 2

        return upper

    def __call__(
        self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional

        Returns
        -------
        embeddings : (batch_size, dimension)

        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1

        waveforms = waveforms.squeeze(dim=1)

        if masks is None:
            signals = waveforms.squeeze(dim=1)
            wav_lens = signals.shape[1] * torch.ones(batch_size)

        else:
            batch_size_masks, _ = masks.shape
            assert batch_size == batch_size_masks

            # TODO: speed up the creation of "signals"
            # preliminary profiling experiments show
            # that it accounts for 15% of __call__
            # (the remaining 85% being the actual forward pass)

            imasks = F.interpolate(
                masks.unsqueeze(dim=1), size=num_samples, mode="nearest"
            ).squeeze(dim=1)

            imasks = imasks > 0.5

            signals = pad_sequence(
                [waveform[imask] for waveform, imask in zip(waveforms, imasks)],
                batch_first=True,
            )

            wav_lens = imasks.sum(dim=1)

        max_len = wav_lens.max()

        # corner case: every signal is too short
        if max_len < self.min_num_samples:
            return np.nan * np.zeros((batch_size, self.dimension))

        too_short = wav_lens < self.min_num_samples
        wav_lens[too_short] = max_len

        _, embeddings = self.model_(
            input_signal=waveforms.to(self.device),
            input_signal_length=wav_lens.to(self.device),
        )

        embeddings = embeddings.cpu().numpy()
        embeddings[too_short.cpu().numpy()] = np.nan

        return embeddings


class SpeechBrainPretrainedSpeakerEmbedding(BaseInference):
    """Pretrained SpeechBrain speaker embedding

    Parameters
    ----------
    embedding : str
        Name of SpeechBrain model
    device : torch.device, optional
        Device
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`

    Usage
    -----
    >>> get_embedding = SpeechBrainPretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size

    >>> assert binary_masks.ndim == 1
    >>> assert binary_masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=binary_masks)
    """

    def __init__(
        self,
        embedding: Text = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[torch.device] = None,
        use_auth_token: Union[Text, None] = None,
    ):
        if not SPEECHBRAIN_IS_AVAILABLE:
            raise ImportError(
                f"'speechbrain' must be installed to use '{embedding}' embeddings. "
                "Visit https://speechbrain.github.io for installation instructions."
            )

        super().__init__()
        if "@" in embedding:
            self.embedding = embedding.split("@")[0]
            self.revision = embedding.split("@")[1]
        else:
            self.embedding = embedding
            self.revision = None
        self.device = device or torch.device("cpu")
        self.use_auth_token = use_auth_token

        self.classifier_ = SpeechBrain_EncoderClassifier.from_hparams(
            source=self.embedding,
            savedir=f"{CACHE_DIR}/speechbrain",
            run_opts={"device": self.device},
            use_auth_token=self.use_auth_token,
            revision=self.revision,
        )

    def to(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        self.classifier_ = SpeechBrain_EncoderClassifier.from_hparams(
            source=self.embedding,
            savedir=f"{CACHE_DIR}/speechbrain",
            run_opts={"device": device},
            use_auth_token=self.use_auth_token,
            revision=self.revision,
        )
        self.device = device
        return self

    @cached_property
    def sample_rate(self) -> int:
        return self.classifier_.audio_normalizer.sample_rate

    @cached_property
    def dimension(self) -> int:
        dummy_waveforms = torch.rand(1, 16000).to(self.device)
        *_, dimension = self.classifier_.encode_batch(dummy_waveforms).shape
        return dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        with torch.inference_mode():
            lower, upper = 2, round(0.5 * self.sample_rate)
            middle = (lower + upper) // 2
            while lower + 1 < upper:
                try:
                    _ = self.classifier_.encode_batch(
                        torch.randn(1, middle).to(self.device)
                    )
                    upper = middle
                except RuntimeError:
                    lower = middle

                middle = (lower + upper) // 2

        return upper

    def __call__(
        self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional

        Returns
        -------
        embeddings : (batch_size, dimension)

        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1

        waveforms = waveforms.squeeze(dim=1)

        if masks is None:
            signals = waveforms.squeeze(dim=1)
            wav_lens = signals.shape[1] * torch.ones(batch_size)

        else:
            batch_size_masks, _ = masks.shape
            assert batch_size == batch_size_masks

            # TODO: speed up the creation of "signals"
            # preliminary profiling experiments show
            # that it accounts for 15% of __call__
            # (the remaining 85% being the actual forward pass)

            imasks = F.interpolate(
                masks.unsqueeze(dim=1), size=num_samples, mode="nearest"
            ).squeeze(dim=1)

            imasks = imasks > 0.5

            signals = pad_sequence(
                [
                    waveform[imask].contiguous()
                    for waveform, imask in zip(waveforms, imasks)
                ],
                batch_first=True,
            )

            wav_lens = imasks.sum(dim=1)

        max_len = wav_lens.max()

        # corner case: every signal is too short
        if max_len < self.min_num_samples:
            return np.nan * np.zeros((batch_size, self.dimension))

        too_short = wav_lens < self.min_num_samples
        wav_lens = wav_lens / max_len
        wav_lens[too_short] = 1.0

        embeddings = (
            self.classifier_.encode_batch(signals, wav_lens=wav_lens)
            .squeeze(dim=1)
            .cpu()
            .numpy()
        )

        embeddings[too_short.cpu().numpy()] = np.nan

        return embeddings


class ONNXWeSpeakerPretrainedSpeakerEmbedding(BaseInference):
    """Pretrained WeSpeaker speaker embedding

    Parameters
    ----------
    embedding : str
        Path to WeSpeaker pretrained speaker embedding
    device : torch.device, optional
        Device

    Usage
    -----
    >>> get_embedding = ONNXWeSpeakerPretrainedSpeakerEmbedding("hbredin/wespeaker-voxceleb-resnet34-LM")
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size

    >>> assert binary_masks.ndim == 1
    >>> assert binary_masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=binary_masks)
    """

    def __init__(
        self,
        embedding: Text = "hbredin/wespeaker-voxceleb-resnet34-LM",
        device: Optional[torch.device] = None,
    ):
        if not ONNX_IS_AVAILABLE:
            raise ImportError(
                f"'onnxruntime' must be installed to use '{embedding}' embeddings."
            )

        super().__init__()

        if not Path(embedding).exists():
            try:
                embedding = hf_hub_download(
                    repo_id=embedding,
                    filename="speaker-embedding.onnx",
                )
            except RepositoryNotFoundError:
                raise ValueError(
                    f"Could not find '{embedding}' on huggingface.co nor on local disk."
                )

        self.embedding = embedding
        # Добавление кэша для feature batches
        self._features_cache = {}
        self._cache_max_size = 50  # Максимальный размер кэша

        self.to(device or torch.device("cpu"))

    def to(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        if device.type == "cpu":
            providers = ["CPUExecutionProvider"]
            provider_options = None
        elif device.type == "cuda":
            providers = ["CUDAExecutionProvider"]
            # Оптимизации для CUDA
            # Определяем оптимальные параметры в зависимости от версии ONNX Runtime
            try:
                # Проверяем версию onnxruntime
                ort_version = tuple(int(v) for v in ort.__version__.split('.')[:2])
                
                if ort_version >= (1, 10):
                    # Для более новых версий используем более продвинутые настройки
                    provider_options = [
                        {
                            'device_id': device.index or 0,
                            'arena_extend_strategy': 'kNextPowerOfTwo',
                            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 2GB
                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                            'do_copy_in_default_stream': True,
                        }
                    ]
                else:
                    # Для более старых версий используем более безопасные настройки
                    provider_options = [
                        {
                            'device_id': device.index or 0,
                        }
                    ]
            except (AttributeError, ValueError):
                # Если не удалось определить версию, используем безопасные настройки
                provider_options = [{'device_id': device.index or 0}]
        else:
            warnings.warn(
                f"Unsupported device type: {device.type}, falling back to CPU"
            )
            device = torch.device("cpu")
            providers = ["CPUExecutionProvider"]
            provider_options = None

        # Оптимизация сессии ONNX
        sess_options = ort.SessionOptions()
        
        # Настраиваем опции многопоточности и оптимизации в зависимости от версии ONNX Runtime
        try:
            # Установка оптимальных параметров для многопоточности
            sess_options.inter_op_num_threads = min(8, os.cpu_count() or 4)
            sess_options.intra_op_num_threads = min(8, os.cpu_count() or 4)
            
            # Включение графовых оптимизаций
            if hasattr(ort, 'GraphOptimizationLevel'):
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Включение параллельного выполнения для операторов
            if hasattr(ort, 'ExecutionMode'):
                sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        except AttributeError:
            # Если какие-то атрибуты недоступны, используем базовую конфигурацию
            pass
            
        try:
            # Создаем сессию с оптимальными параметрами
            if provider_options:
                self.session_ = ort.InferenceSession(
                    self.embedding, 
                    sess_options=sess_options, 
                    providers=providers, 
                    provider_options=provider_options
                )
            else:
                self.session_ = ort.InferenceSession(
                    self.embedding,
                    sess_options=sess_options,
                    providers=providers
                )
        except TypeError:
            # В случае ошибки типа (часто в старых версиях), пробуем без provider_options
            self.session_ = ort.InferenceSession(
                self.embedding,
                sess_options=sess_options,
                providers=providers
            )

        self.device = device
        return self

    @cached_property
    def sample_rate(self) -> int:
        return 16000

    @cached_property
    def dimension(self) -> int:
        dummy_waveforms = torch.rand(1, 1, 16000)
        features = self.compute_fbank(dummy_waveforms)
        embeddings = self.session_.run(
            output_names=["embs"], input_feed={"feats": features.numpy()}
        )[0]
        _, dimension = embeddings.shape
        return dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        lower, upper = 2, round(0.5 * self.sample_rate)
        middle = (lower + upper) // 2
        while lower + 1 < upper:
            try:
                features = self.compute_fbank(torch.randn(1, 1, middle))

            except AssertionError:
                lower = middle
                middle = (lower + upper) // 2
                continue

            embeddings = self.session_.run(
                output_names=["embs"], input_feed={"feats": features.numpy()}
            )[0]

            if np.any(np.isnan(embeddings)):
                lower = middle
            else:
                upper = middle
            middle = (lower + upper) // 2

        return upper

    @cached_property
    def min_num_frames(self) -> int:
        return self.compute_fbank(torch.randn(1, 1, self.min_num_samples)).shape[1]

    @torch.no_grad()
    def compute_fbank(
        self,
        waveforms: torch.Tensor,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Extract fbank features

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
        num_mel_bins : int, optional
            Number of mel bins. Defaults to 80.
        frame_length : int, optional
            Frame length in ms. Defaults to 25.
        frame_shift : int, optional
            Frame shift in ms. Defaults to 10.
        dither : float, optional
            Dithering constant. Defaults to 0.0.
        use_cache : bool, optional
            Whether to use cache for the features. Defaults to True.

        Returns
        -------
        fbank : (batch_size, num_frames, num_mel_bins)

        Source: https://github.com/wenet-e2e/wespeaker/blob/45941e7cba2c3ea99e232d02bedf617fc71b0dad/wespeaker/bin/infer_onnx.py#L30C1-L50
        """
        device = waveforms.device
        
        # Проверяем кэш, если он включен
        if use_cache:
            # Создаем ключ для кэша на основе хэша тензора
            # Используем только первые 1000 сэмплов для хеширования, чтобы избежать слишком долгого вычисления
            sample_for_hash = waveforms.cpu().detach()
            if sample_for_hash.shape[-1] > 1000:
                sample_for_hash = sample_for_hash[:, :, :1000]
            cache_key = hash(sample_for_hash.numpy().tobytes())
            if cache_key in self._features_cache:
                return self._features_cache[cache_key].to(device)
        
        # Проверка шейпов waveforms
        if waveforms.dim() != 3:
            raise ValueError(f"waveforms должны иметь шейп (batch_size, channels, samples), но получено: {waveforms.shape}")
        
        if waveforms.shape[1] != 1:
            raise ValueError(f"waveforms должны иметь только один канал, но получено: {waveforms.shape[1]}")
        
        waveforms = waveforms * (1 << 15)
        
        # Оптимизированное вычисление признаков - обрабатываем весь батч сразу
        batch_size = waveforms.shape[0]
        if batch_size > 1:
            # Для больших батчей используем параллельное вычисление
            features = torch.stack(
                [
                    kaldi.fbank(
                        waveform,
                        num_mel_bins=num_mel_bins,
                        frame_length=frame_length,
                        frame_shift=frame_shift,
                        dither=dither,
                        sample_frequency=self.sample_rate,
                        window_type="hamming",
                        use_energy=False,
                    )
                    for waveform in waveforms
                ]
            )
        else:
            # Для одиночных примеров - обычное вычисление
            features = kaldi.fbank(
                waveforms[0],
                num_mel_bins=num_mel_bins,
                frame_length=frame_length,
                frame_shift=frame_shift,
                dither=dither,
                sample_frequency=self.sample_rate,
                window_type="hamming",
                use_energy=False,
            ).unsqueeze(0)

        # Централизация признаков по батчу
        features = features - torch.mean(features, dim=1, keepdim=True)
        
        # Кэшируем признаки, если кэш включен
        if use_cache:
            # Ограничиваем размер кэша
            if len(self._features_cache) >= self._cache_max_size:
                # Удаляем случайный ключ, если кэш заполнен
                remove_key = next(iter(self._features_cache))
                del self._features_cache[remove_key]
            
            # Сохраняем в кэш на CPU для экономии памяти GPU
            self._features_cache[cache_key] = features.cpu()
        
        return features

    @torch.no_grad()
    def __call__(
        self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional

        Returns
        -------
        embeddings : (batch_size, dimension)

        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1
        
        # Определение устройства и оптимизация передачи данных
        device = waveforms.device
        cuda_available = torch.cuda.is_available()
        use_cuda_for_preprocessing = cuda_available and device.type == 'cuda'
        onnx_uses_cuda = 'CUDAExecutionProvider' in self.session_.get_providers()
        
        # Определяем оптимальное устройство для предобработки
        preprocess_device = self.device if use_cuda_for_preprocessing else torch.device('cpu')
        
        # Перемещаем данные на нужное устройство для предобработки
        if device != preprocess_device:
            waveforms = waveforms.to(preprocess_device)
        
        # Вычисляем признаки на выбранном устройстве
        features = self.compute_fbank(waveforms)
        _, num_frames, _ = features.shape

        if masks is None:
            # Оптимизированный путь без масок - обрабатываем весь батч сразу
            # Оптимизируем передачу данных между GPU и CPU в зависимости от конфигурации
            if onnx_uses_cuda:
                # Если ONNX работает на CUDA, данные должны быть на GPU
                if features.device.type != 'cuda':
                    features = features.cuda()
                features_np = features.numpy(force=True)
            else:
                # Если ONNX работает на CPU, данные должны быть на CPU
                if features.device.type != 'cpu':
                    features = features.cpu()
                features_np = features.numpy()
            
            # Проверка шейпа перед передачей в ONNX
            if features_np.ndim != 3:
                raise ValueError(f"Некорректный шейп features: {features_np.shape}, ожидается (batch_size, num_frames, num_features)")
                
            # Инференс с оптимизированной передачей данных
            embeddings = self.session_.run(
                output_names=["embs"], input_feed={"feats": features_np}
            )[0]

            return embeddings

        # Путь с масками: оптимизация для батчей
        batch_size_masks, _ = masks.shape
        assert batch_size == batch_size_masks
        
        # Перемещаем маски на то же устройство, что и признаки
        masks = masks.to(preprocess_device)
        
        # Оптимизированная интерполяция масок
        # mode='nearest' может вызвать проблемы в некоторых версиях torch, используем более надежный метод
        try:
            imasks = F.interpolate(
                masks.unsqueeze(dim=1), size=num_frames, mode="nearest"
            ).squeeze(dim=1)
        except RuntimeError:
            # Резервный метод в случае ошибки
            ratio = num_frames / masks.shape[1]
            imasks = torch.zeros((batch_size, num_frames), device=masks.device)
            for b in range(batch_size):
                for i in range(masks.shape[1]):
                    start_idx = int(i * ratio)
                    end_idx = int((i + 1) * ratio)
                    imasks[b, start_idx:end_idx] = masks[b, i]

        imasks = imasks > 0.5

        # Создаем массив для хранения результатов
        embeddings = np.nan * np.zeros((batch_size, self.dimension))

        # Вместо последовательной обработки, группируем примеры с масками
        valid_indices = []
        valid_features = []
        
        for f, (feature, imask) in enumerate(zip(features, imasks)):
            if torch.sum(imask) == 0:
                # Пропускаем примеры без маски
                continue
                
            masked_feature = feature[imask]
            if masked_feature.shape[0] >= self.min_num_frames:
                valid_indices.append(f)
                valid_features.append(masked_feature)
        
        if not valid_features:
            return embeddings
        
        # Обрабатываем все допустимые признаки в оптимальном батче
        if len(valid_features) == 1:
            # Одиночный пример
            feature_batch = valid_features[0].unsqueeze(0)
            
            # Оптимальная передача данных в зависимости от конфигурации
            if onnx_uses_cuda:
                if feature_batch.device.type != 'cuda':
                    feature_batch = feature_batch.cuda()
                feature_batch_np = feature_batch.numpy(force=True)
            else:
                if feature_batch.device.type != 'cpu':
                    feature_batch = feature_batch.cpu()
                feature_batch_np = feature_batch.numpy()
            
            # Проверка шейпа перед передачей в ONNX
            if feature_batch_np.ndim != 3:
                raise ValueError(f"Некорректный шейп feature_batch: {feature_batch_np.shape}, ожидается (batch_size, num_frames, num_features)")
                
            result = self.session_.run(
                output_names=["embs"],
                input_feed={"feats": feature_batch_np},
            )[0]
            embeddings[valid_indices[0]] = result[0]
        else:
            # Оптимизированный батчинг с паддингом
            max_len = max(f.shape[0] for f in valid_features)
            padded_features = []
            
            for feat in valid_features:
                if feat.shape[0] < max_len:
                    # Паддинг путем повторения последнего фрейма (более эффективно для больших батчей)
                    padding = feat[-1:].repeat(max_len - feat.shape[0], 1)
                    padded_feat = torch.cat([feat, padding], dim=0)
                else:
                    padded_feat = feat
                padded_features.append(padded_feat)
            
            # Объединяем в один батч
            feature_batch = torch.stack(padded_features)
            
            # Оптимизированная передача данных для инференса
            if onnx_uses_cuda:
                if feature_batch.device.type != 'cuda':
                    feature_batch = feature_batch.cuda()
                feature_batch_np = feature_batch.numpy(force=True)
            else:
                if feature_batch.device.type != 'cpu':
                    feature_batch = feature_batch.cpu()
                feature_batch_np = feature_batch.numpy()
            
            # Проверка шейпа перед передачей в ONNX
            if feature_batch_np.ndim != 3:
                raise ValueError(f"Некорректный шейп feature_batch: {feature_batch_np.shape}, ожидается (batch_size, num_frames, num_features)")
                
            # Выполняем инференс на всем батче
            result = self.session_.run(
                output_names=["embs"],
                input_feed={"feats": feature_batch_np},
            )[0]
            
            # Копируем результаты в соответствующие позиции
            for i, idx in enumerate(valid_indices):
                embeddings[idx] = result[i]

        return embeddings


class PyannoteAudioPretrainedSpeakerEmbedding(BaseInference):
    """Pretrained pyannote.audio speaker embedding

    Parameters
    ----------
    embedding : PipelineModel
        pyannote.audio model
    device : torch.device, optional
        Device
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`

    Usage
    -----
    >>> get_embedding = PyannoteAudioPretrainedSpeakerEmbedding("pyannote/embedding")
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size

    >>> assert masks.ndim == 1
    >>> assert masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=masks)
    """

    def __init__(
        self,
        embedding: PipelineModel = "pyannote/embedding",
        device: Optional[torch.device] = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__()
        self.embedding = embedding
        self.device = device or torch.device("cpu")

        self.model_: Model = get_model(self.embedding, use_auth_token=use_auth_token)
        self.model_.eval()
        self.model_.to(self.device)

    def to(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        self.model_.to(device)
        self.device = device
        return self

    @cached_property
    def sample_rate(self) -> int:
        return self.model_.audio.sample_rate

    @cached_property
    def dimension(self) -> int:
        return self.model_.dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        with torch.inference_mode():
            lower, upper = 2, round(0.5 * self.sample_rate)
            middle = (lower + upper) // 2
            while lower + 1 < upper:
                try:
                    _ = self.model_(torch.randn(1, 1, middle).to(self.device))
                    upper = middle
                except Exception:
                    lower = middle

                middle = (lower + upper) // 2

        return upper

    def __call__(
        self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        with torch.inference_mode():
            if masks is None:
                embeddings = self.model_(waveforms.to(self.device))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    embeddings = self.model_(
                        waveforms.to(self.device), weights=masks.to(self.device)
                    )
        return embeddings.cpu().numpy()


def PretrainedSpeakerEmbedding(
    embedding: PipelineModel,
    device: Optional[torch.device] = None,
    use_auth_token: Union[Text, None] = None,
):
    """Pretrained speaker embedding

    Parameters
    ----------
    embedding : Text
        Can be a SpeechBrain (e.g. "speechbrain/spkrec-ecapa-voxceleb")
        or a pyannote.audio model.
    device : torch.device, optional
        Device
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`

    Usage
    -----
    >>> get_embedding = PretrainedSpeakerEmbedding("pyannote/embedding")
    >>> get_embedding = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
    >>> get_embedding = PretrainedSpeakerEmbedding("nvidia/speakerverification_en_titanet_large")
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size

    >>> assert masks.ndim == 1
    >>> assert masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=masks)
    """

    if isinstance(embedding, str) and "pyannote" in embedding:
        return PyannoteAudioPretrainedSpeakerEmbedding(
            embedding, device=device, use_auth_token=use_auth_token
        )

    elif isinstance(embedding, str) and "speechbrain" in embedding:
        return SpeechBrainPretrainedSpeakerEmbedding(
            embedding, device=device, use_auth_token=use_auth_token
        )

    elif isinstance(embedding, str) and "nvidia" in embedding:
        return NeMoPretrainedSpeakerEmbedding(embedding, device=device)

    elif isinstance(embedding, str) and "wespeaker" in embedding:
        return ONNXWeSpeakerPretrainedSpeakerEmbedding(embedding, device=device)

    else:
        # fallback to pyannote in case we are loading a local model
        return PyannoteAudioPretrainedSpeakerEmbedding(
            embedding, device=device, use_auth_token=use_auth_token
        )


class SpeakerEmbedding(Pipeline):
    """Speaker embedding pipeline

    This pipeline assumes that each file contains exactly one speaker
    and extracts one single embedding from the whole file.

    Parameters
    ----------
    embedding : Model, str, or dict, optional
        Pretrained embedding model. Defaults to "pyannote/embedding".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    segmentation : Model, str, or dict, optional
        Pretrained segmentation (or voice activity detection) model.
        See pyannote.audio.pipelines.utils.get_model for supported format.
        Defaults to no voice activity detection.
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`

    Usage
    -----
    >>> from pyannote.audio.pipelines import SpeakerEmbedding
    >>> pipeline = SpeakerEmbedding()
    >>> emb1 = pipeline("speaker1.wav")
    >>> emb2 = pipeline("speaker2.wav")
    >>> from scipy.spatial.distance import cdist
    >>> distance = cdist(emb1, emb2, metric="cosine")[0,0]
    """

    def __init__(
        self,
        embedding: PipelineModel = "pyannote/embedding",
        segmentation: Optional[PipelineModel] = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__()

        self.embedding = embedding
        self.segmentation = segmentation

        self.embedding_model_: Model = get_model(
            embedding, use_auth_token=use_auth_token
        )

        if self.segmentation is not None:
            segmentation_model: Model = get_model(
                self.segmentation, use_auth_token=use_auth_token
            )
            self._segmentation = Inference(
                segmentation_model,
                pre_aggregation_hook=lambda scores: np.max(
                    scores, axis=-1, keepdims=True
                ),
            )

    def apply(self, file: AudioFile) -> np.ndarray:
        device = self.embedding_model_.device

        # read audio file and send it to GPU
        waveform = self.embedding_model_.audio(file)[0][None].to(device)

        if self.segmentation is None:
            weights = None
        else:
            # obtain voice activity scores
            weights = self._segmentation(file).data
            # HACK -- this should be fixed upstream
            weights[np.isnan(weights)] = 0.0
            weights = torch.from_numpy(weights**3)[None, :, 0].to(device)

        # extract speaker embedding on parts of
        with torch.no_grad():
            return self.embedding_model_(waveform, weights=weights).cpu().numpy()


def main(
    protocol: str = "VoxCeleb.SpeakerVerification.VoxCeleb1",
    subset: str = "test",
    embedding: str = "pyannote/embedding",
    segmentation: Optional[str] = None,
):
    import typer
    from pyannote.database import FileFinder, get_protocol
    from pyannote.metrics.binary_classification import det_curve
    from scipy.spatial.distance import cdist
    from tqdm import tqdm

    pipeline = SpeakerEmbedding(embedding=embedding, segmentation=segmentation)

    protocol = get_protocol(protocol, preprocessors={"audio": FileFinder()})

    y_true, y_pred = [], []

    emb = dict()

    trials = getattr(protocol, f"{subset}_trial")()

    for t, trial in enumerate(tqdm(trials)):
        audio1 = trial["file1"]["audio"]
        if audio1 not in emb:
            emb[audio1] = pipeline(audio1)

        audio2 = trial["file2"]["audio"]
        if audio2 not in emb:
            emb[audio2] = pipeline(audio2)

        y_pred.append(cdist(emb[audio1], emb[audio2], metric="cosine")[0][0])
        y_true.append(trial["reference"])

    _, _, _, eer = det_curve(y_true, np.array(y_pred), distances=True)
    typer.echo(
        f"{protocol.name} | {subset} | {embedding} | {segmentation} | EER = {100 * eer:.3f}%"
    )


if __name__ == "__main__":
    import typer

    typer.run(main)
