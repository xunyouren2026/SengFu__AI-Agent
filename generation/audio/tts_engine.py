"""
TTS语音合成引擎 - 纯Python实现
包含音素转换、声学模型、声码器、声音克隆、音乐生成和音效生成功能。
仅使用标准库，不依赖外部库。
"""

import math
import struct
import random
import hashlib
import cmath
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


# ---------------------------------------------------------------------------
# 公共数据结构
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """统一生成结果"""
    data: Any = None
    format: str = ""
    sample_rate: int = 22050
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeakerProfile:
    """说话人配置"""
    speaker_id: int = 0
    pitch_mean: float = 220.0
    pitch_std: float = 30.0
    speed: float = 1.0
    energy: float = 0.8
    timbre_features: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Phonemizer - 文本到音素转换
# ---------------------------------------------------------------------------

class Phonemizer:
    """文本到音素转换器"""

    def __init__(self):
        self._vowels = {
            'a', 'e', 'i', 'o', 'u',
            'AA', 'AE', 'AH', 'AO', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'UH', 'UW',
        }
        self._consonants = {
            'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm',
            'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z',
            'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L',
            'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V',
            'W', 'Y', 'Z', 'ZH',
        }
        self._graphemes: Dict[str, List[str]] = {
            'a': ['AH'], 'e': ['EH'], 'i': ['IH'], 'o': ['AO'], 'u': ['UH'],
            'b': ['B'], 'c': ['K'], 'd': ['D'], 'f': ['F'], 'g': ['G'],
            'h': ['HH'], 'j': ['JH'], 'k': ['K'], 'l': ['L'], 'm': ['M'],
            'n': ['N'], 'p': ['P'], 'q': ['K'], 'r': ['R'], 's': ['S'],
            't': ['T'], 'v': ['V'], 'w': ['W'], 'x': ['K', 'S'], 'y': ['Y'],
            'z': ['Z'],
            'th': ['TH'], 'sh': ['SH'], 'ch': ['CH'], 'ph': ['F'],
            'wh': ['W', 'HH'], 'ng': ['NG'], 'ck': ['K'],
            'ee': ['IY'], 'oo': ['UW'], 'ea': ['IY'], 'ou': ['OW'],
            'ow': ['OW'], 'ai': ['EY'], 'ay': ['EY'], 'oi': ['OY'],
            'oy': ['OY'], 'au': ['AO'], 'aw': ['AO'], 'er': ['ER'],
            'ir': ['ER'], 'ur': ['ER'], 'ar': ['AA', 'R'], 'or': ['AO', 'R'],
            'ie': ['IH'], 'igh': ['AY'], 'ough': ['AO', 'F'],
            'tion': ['SH', 'AH', 'N'], 'sion': ['ZH', 'AH', 'N'],
            'ed': ['T'], 'es': ['EH', 'Z'], 'ly': ['L', 'IH'],
        }
        self._abbreviations: Dict[str, str] = {
            "mr": "mister", "mrs": "misses", "dr": "doctor", "prof": "professor",
            "sr": "senior", "jr": "junior", "st": "saint", "ave": "avenue",
            "blvd": "boulevard", "dept": "department", "etc": "et cetera",
            "vs": "versus", "approx": "approximately", "min": "minute",
            "max": "maximum", "avg": "average", "fig": "figure",
            "eg": "for example", "ie": "that is", "approx": "approximately",
        }

    def phonemize(self, text: str, language: str = "en") -> List[str]:
        """文本转音素"""
        text = text.lower().strip()
        if not text:
            return []
        text = self._handle_numbers(text)
        text = self._handle_abbreviations(text)
        words = self._split_words(text)
        phonemes: List[str] = []
        for word in words:
            word_phonemes = self._word_to_phonemes(word)
            phonemes.extend(word_phonemes)
            phonemes.append(' ')  # 词间停顿
        return [p for p in phonemes if p]

    def _split_words(self, text: str) -> List[str]:
        """分词"""
        import re
        words = re.split(r'[\s,.;:!?()\[\]{}"\']+', text)
        return [w for w in words if w]

    def _word_to_phonemes(self, word: str) -> List[str]:
        """单词转音素"""
        phonemes = self._apply_rules(word)
        if not phonemes:
            phonemes = list(word)
        return phonemes

    def _apply_rules(self, word: str) -> List[str]:
        """应用发音规则"""
        phonemes: List[str] = []
        i = 0
        n = len(word)
        while i < n:
            matched = False
            for length in range(min(4, n - i), 1, -1):
                substr = word[i:i + length]
                if substr in self._graphemes:
                    phonemes.extend(self._graphemes[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                ch = word[i]
                if ch in self._graphemes:
                    phonemes.extend(self._graphemes[ch])
                else:
                    phonemes.append(ch)
                i += 1
        return phonemes

    def _handle_numbers(self, text: str) -> str:
        """数字转文字"""
        import re
        _ones = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        }
        _teens = {
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
            '18': 'eighteen', '19': 'nineteen',
        }
        _tens = {
            '2': 'twenty', '3': 'thirty', '4': 'forty', '5': 'fifty',
            '6': 'sixty', '7': 'seventy', '8': 'eighty', '9': 'ninety',
        }

        def _convert_number(num_str: str) -> str:
            num = int(num_str)
            if num < 10:
                return _ones[num_str]
            if num < 20:
                return _teens[num_str]
            if num < 100:
                t = num_str[0]
                r = num_str[1]
                if r == '0':
                    return _tens[t]
                return _tens[t] + ' ' + _ones[r]
            if num < 1000:
                h = num_str[0]
                rest = str(num % 100)
                if rest == '0':
                    return _ones[h] + ' hundred'
                return _ones[h] + ' hundred ' + _convert_number(rest)
            return num_str

        def _replace(match):
            return _convert_number(match.group(0))

        text = re.sub(r'\b\d{1,3}\b', _replace, text)
        return text

    def _handle_abbreviations(self, text: str) -> str:
        """缩写展开"""
        import re
        for abbr, full in self._abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'[.\s]', full + ' ', text)
            text = re.sub(r'\b' + re.escape(abbr) + r'$', full, text)
        return text


# ---------------------------------------------------------------------------
# 2. AcousticModel - 声学模型
# ---------------------------------------------------------------------------

class AcousticModel:
    """声学模型 - 生成梅尔频谱"""

    def __init__(self):
        self._sample_rate: int = 22050
        self._hop_length: int = 256
        self._n_fft: int = 1024
        self._mel_bands: int = 80
        self._phoneme_durations: Dict[str, float] = {
            ' ': 0.05, 'B': 0.08, 'CH': 0.10, 'D': 0.07, 'DH': 0.07,
            'F': 0.09, 'G': 0.08, 'HH': 0.06, 'JH': 0.10, 'K': 0.08,
            'L': 0.09, 'M': 0.09, 'N': 0.08, 'NG': 0.10, 'P': 0.07,
            'R': 0.08, 'S': 0.09, 'SH': 0.10, 'T': 0.06, 'TH': 0.09,
            'V': 0.08, 'W': 0.07, 'Y': 0.06, 'Z': 0.08, 'ZH': 0.10,
            'AA': 0.12, 'AE': 0.11, 'AH': 0.10, 'AO': 0.12, 'EH': 0.10,
            'ER': 0.11, 'EY': 0.13, 'IH': 0.08, 'IY': 0.11, 'OW': 0.12,
            'UH': 0.09, 'UW': 0.11,
        }

    def generate_mel_spectrogram(self, phonemes: List[str], speaker_id: int = 0) -> List[List[float]]:
        """生成梅尔频谱"""
        durations = [self._phoneme_to_duration(p) for p in phonemes]
        total_duration = sum(durations)
        n_frames = max(2, int(total_duration * self._sample_rate / self._hop_length))
        f0 = self._f0_generation(phonemes)
        energy = self._energy_generation(phonemes)

        mel_spec: List[List[float]] = []
        random.seed(speaker_id * 137 + 42)
        for t in range(n_frames):
            frame: List[float] = []
            progress = t / max(1, n_frames - 1)
            for band in range(self._mel_bands):
                freq_norm = band / max(1, self._mel_bands - 1)
                base_energy = energy[min(t, len(energy) - 1)] if energy else 0.5
                f0_val = f0[min(t, len(f0) - 1)] if f0 else 220.0
                f0_norm = (f0_val - 80) / 400.0
                f0_norm = max(0.0, min(1.0, f0_norm))
                val = base_energy * math.exp(-2.0 * (freq_norm - f0_norm) ** 2)
                val += random.gauss(0, 0.02)
                val += 0.3 * math.sin(2 * math.pi * progress * 3 + band * 0.1)
                val = max(0.0, min(1.0, val))
                frame.append(val)
            mel_spec.append(frame)
        return mel_spec

    def _phoneme_to_duration(self, phoneme: str) -> float:
        """音素持续时间"""
        return self._phoneme_durations.get(phoneme, 0.08)

    def _f0_generation(self, phonemes: List[str]) -> List[float]:
        """基频(F0)生成"""
        f0_list: List[float] = []
        for p in phonemes:
            dur = self._phoneme_to_duration(p)
            n_samples = max(1, int(dur * 100))
            if p in (' ',):
                f0_list.extend([0.0] * n_samples)
            elif p in ('AA', 'AE', 'AH', 'AO', 'ER', 'EY', 'OW', 'UW'):
                base = random.uniform(180, 260)
                for i in range(n_samples):
                    f0_list.append(base + random.gauss(0, 5))
            else:
                base = random.uniform(120, 180)
                for i in range(n_samples):
                    f0_list.append(base + random.gauss(0, 3))
        return f0_list

    def _energy_generation(self, phonemes: List[str]) -> List[float]:
        """能量生成"""
        energy_list: List[float] = []
        for p in phonemes:
            dur = self._phoneme_to_duration(p)
            n_samples = max(1, int(dur * 100))
            if p == ' ':
                energy_list.extend([0.01] * n_samples)
            elif p in ('B', 'D', 'G', 'K', 'P', 'T'):
                base = random.uniform(0.6, 0.9)
                for i in range(n_samples):
                    progress = i / max(1, n_samples - 1)
                    env = math.exp(-3 * progress)
                    energy_list.append(base * env + random.gauss(0, 0.02))
            else:
                base = random.uniform(0.3, 0.7)
                for i in range(n_samples):
                    progress = i / max(1, n_samples - 1)
                    env = math.sin(math.pi * progress)
                    energy_list.append(base * env + random.gauss(0, 0.01))
        return energy_list

    def _mel_to_linear(self, mel_spec: List[List[float]]) -> List[List[float]]:
        """梅尔到线性频谱"""
        n_mel = len(mel_spec[0]) if mel_spec else 0
        n_linear = self._n_fft // 2 + 1
        mel_matrix = self._create_mel_matrix(n_mel, n_linear)
        linear_spec: List[List[float]] = []
        for frame in mel_spec:
            linear_frame = [0.0] * n_linear
            for i in range(n_linear):
                val = 0.0
                for j in range(n_mel):
                    val += mel_matrix[j][i] * frame[j]
                linear_frame[i] = max(0.0, val)
            linear_spec.append(linear_frame)
        return linear_spec

    def _create_mel_matrix(self, n_mel: int, n_freq: int) -> List[List[float]]:
        """创建梅尔滤波器组矩阵"""
        f_min = 0.0
        f_max = self._sample_rate / 2.0
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mel_points = [mel_min + (mel_max - mel_min) * i / (n_mel + 1) for i in range(n_mel + 2)]
        hz_points = [self._mel_to_hz(m) for m in mel_points]
        bin_points = [int((f / self._sample_rate) * self._n_fft) for f in hz_points]
        bin_points = [min(b, n_freq - 1) for b in bin_points]
        matrix: List[List[float]] = []
        for m in range(n_mel):
            row = [0.0] * n_freq
            f_left = bin_points[m]
            f_center = bin_points[m + 1]
            f_right = bin_points[m + 2]
            for k in range(f_left, f_center):
                if f_center != f_left:
                    row[k] = (k - f_left) / max(1, f_center - f_left)
            for k in range(f_center, f_right):
                if f_right != f_center:
                    row[k] = (f_right - k) / max(1, f_right - f_center)
            matrix.append(row)
        return matrix

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _griffin_lim(self, spec: List[List[float]], n_iter: int = 60) -> List[float]:
        """Griffin-Lim相位重建"""
        n_fft = self._n_fft
        hop = self._hop_length
        n_frames = len(spec)
        output_len = (n_frames - 1) * hop + n_fft
        phase = [0.0] * (n_fft // 2 + 1)
        for _ in range(n_iter):
            stft_matrix: List[List[complex]] = []
            for t, mag_frame in enumerate(spec):
                freqs = []
                for k in range(n_fft // 2 + 1):
                    angle = phase[k] + random.gauss(0, 0.1)
                    freqs.append(cmath.rect(mag_frame[k] if k < len(mag_frame) else 0.0, angle))
                stft_matrix.append(freqs)
            waveform = self._istft_simple(stft_matrix, hop, n_fft)
            new_stft = self._stft_simple(waveform, n_fft, hop)
            for t in range(min(n_frames, len(new_stft))):
                for k in range(min(len(phase), len(new_stft[t]))):
                    phase[k] = cmath.phase(new_stft[t][k])
        final_stft: List[List[complex]] = []
        for t, mag_frame in enumerate(spec):
            freqs = []
            for k in range(n_fft // 2 + 1):
                freqs.append(cmath.rect(mag_frame[k] if k < len(mag_frame) else 0.0, phase[k]))
            final_stft.append(freqs)
        return self._istft_simple(final_stft, hop, n_fft)[:output_len]

    def _istft_simple(self, stft_matrix: List[List[complex]], hop: int, n_fft: int) -> List[float]:
        """简化逆STFT"""
        n_frames = len(stft_matrix)
        output_len = (n_frames - 1) * hop + n_fft
        output = [0.0] * output_len
        window_sum = [0.0] * output_len
        window = self._apply_window(list(range(n_fft)), "hann")
        for t, frame in enumerate(stft_matrix):
            full_frame = [0.0] * n_fft
            for k in range(len(frame)):
                full_frame[k] = frame[k].real
            for k in range(len(frame), n_fft // 2 + 1):
                if k < n_fft:
                    conj_idx = n_fft - k
                    if conj_idx < n_fft and conj_idx > 0:
                        full_frame[k] = frame[conj_idx - 1].real if conj_idx - 1 < len(frame) else 0.0
            for i in range(n_fft):
                idx = t * hop + i
                if idx < output_len:
                    output[idx] += full_frame[i] * window[i]
                    window_sum[idx] += window[i] * window[i]
        for i in range(output_len):
            if window_sum[i] > 1e-8:
                output[i] /= window_sum[i]
        return output

    def _stft_simple(self, waveform: List[float], n_fft: int, hop: int) -> List[List[complex]]:
        """简化STFT"""
        n_frames = max(1, (len(waveform) - n_fft) // hop + 1)
        window = self._apply_window(list(range(n_fft)), "hann")
        result: List[List[complex]] = []
        for t in range(n_frames):
            start = t * hop
            frame = [waveform[start + i] * window[i] if start + i < len(waveform) else 0.0
                     for i in range(n_fft)]
            spectrum = []
            for k in range(n_fft // 2 + 1):
                re = 0.0
                im = 0.0
                for n in range(n_fft):
                    angle = 2.0 * math.pi * k * n / n_fft
                    re += frame[n] * math.cos(angle)
                    im -= frame[n] * math.sin(angle)
                spectrum.append(complex(re / n_fft, im / n_fft))
            result.append(spectrum)
        return result

    def _pre_emphasis(self, waveform: List[float], coeff: float = 0.97) -> List[float]:
        """预加重"""
        if not waveform:
            return []
        result = [waveform[0]]
        for i in range(1, len(waveform)):
            result.append(waveform[i] - coeff * waveform[i - 1])
        return result

    def _apply_window(self, frame: List[float], window_type: str = "hann") -> List[float]:
        """加窗"""
        n = len(frame)
        if window_type == "hann":
            window = [0.5 * (1 - math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]
        elif window_type == "hamming":
            window = [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]
        elif window_type == "blackman":
            window = [0.42 - 0.5 * math.cos(2 * math.pi * i / (n - 1))
                      + 0.08 * math.cos(4 * math.pi * i / (n - 1)) for i in range(n)]
        else:
            window = [1.0] * n
        return [f * w for f, w in zip(frame, window)]


# ---------------------------------------------------------------------------
# 3. Vocoder - 声码器
# ---------------------------------------------------------------------------

class Vocoder:
    """声码器 - 频谱到波形"""

    def __init__(self):
        self._sample_rate: int = 22050

    def synthesize(self, mel_spec: List[List[float]]) -> List[float]:
        """频谱到波形"""
        if not mel_spec:
            return []
        acoustic = AcousticModel()
        linear_spec = acoustic._mel_to_linear(mel_spec)
        waveform = self._griffin_lim(linear_spec, n_iter=100)
        waveform = acoustic._pre_emphasis(waveform, 0.97)
        max_val = max(abs(v) for v in waveform) if waveform else 1.0
        if max_val > 0:
            waveform = [v / max_val * 0.9 for v in waveform]
        return waveform

    def _griffin_lim(self, spec: List[List[float]], n_iter: int = 100) -> List[float]:
        """Griffin-Lim算法"""
        n_fft = 1024
        hop = 256
        n_frames = len(spec)
        n_bins = len(spec[0]) if spec else 0
        output_len = (n_frames - 1) * hop + n_fft
        phase = [random.uniform(0, 2 * math.pi) for _ in range(n_bins)]
        for _ in range(n_iter):
            est_stft: List[List[complex]] = []
            for t in range(n_frames):
                frame = []
                for k in range(n_bins):
                    mag = spec[t][k] if k < len(spec[t]) else 0.0
                    frame.append(cmath.rect(mag, phase[k]))
                est_stft.append(frame)
            waveform = self._istft(est_stft, hop, n_fft)
            new_stft = self._stft(waveform, n_fft, hop)
            for t in range(min(n_frames, len(new_stft))):
                for k in range(min(n_bins, len(new_stft[t]))):
                    phase[k] = cmath.phase(new_stft[t][k])
        final_stft: List[List[complex]] = []
        for t in range(n_frames):
            frame = []
            for k in range(n_bins):
                mag = spec[t][k] if k < len(spec[t]) else 0.0
                frame.append(cmath.rect(mag, phase[k]))
            final_stft.append(frame)
        waveform = self._istft(final_stft, hop, n_fft)
        return waveform[:output_len]

    def _istft(self, stft_matrix: List[List[complex]], hop_length: int, win_length: int) -> List[float]:
        """逆短时傅里叶变换(纯Python)"""
        n_frames = len(stft_matrix)
        output_len = (n_frames - 1) * hop_length + win_length
        output = [0.0] * output_len
        window = self._hann_window(win_length)
        win_sum = [0.0] * output_len
        for t in range(n_frames):
            frame = stft_matrix[t]
            full = [0.0] * win_length
            for k in range(min(len(frame), win_length // 2 + 1)):
                full[k] = frame[k].real
                conj_k = win_length - k
                if conj_k < win_length and conj_k > k:
                    full[conj_k] = frame[k].real
            for i in range(win_length):
                idx = t * hop_length + i
                if idx < output_len:
                    output[idx] += full[i] * window[i]
                    win_sum[idx] += window[i] ** 2
        for i in range(output_len):
            if win_sum[i] > 1e-8:
                output[i] /= win_sum[i]
        return output

    def _stft(self, waveform: List[float], n_fft: int, hop_length: int) -> List[List[complex]]:
        """短时傅里叶变换(纯Python)"""
        n_frames = max(1, (len(waveform) - n_fft) // hop_length + 1)
        window = self._hann_window(n_fft)
        result: List[List[complex]] = []
        for t in range(n_frames):
            start = t * hop_length
            frame = [waveform[start + i] * window[i] if start + i < len(waveform) else 0.0
                     for i in range(n_fft)]
            spectrum = []
            for k in range(n_fft // 2 + 1):
                re_sum = 0.0
                im_sum = 0.0
                for n in range(n_fft):
                    angle = -2.0 * math.pi * k * n / n_fft
                    re_sum += frame[n] * math.cos(angle)
                    im_sum += frame[n] * math.sin(angle)
                spectrum.append(complex(re_sum, im_sum))
            result.append(spectrum)
        return result

    def _hann_window(self, length: int) -> List[float]:
        """汉宁窗"""
        if length <= 1:
            return [1.0]
        return [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (length - 1))) for i in range(length)]

    def _overlap_add(self, frames: List[List[float]], hop_length: int) -> List[float]:
        """重叠相加"""
        if not frames:
            return []
        frame_len = len(frames[0])
        output_len = (len(frames) - 1) * hop_length + frame_len
        output = [0.0] * output_len
        for t, frame in enumerate(frames):
            for i in range(len(frame)):
                idx = t * hop_length + i
                if idx < output_len:
                    output[idx] += frame[i]
        return output


# ---------------------------------------------------------------------------
# 4. VoiceCloner - 声音克隆
# ---------------------------------------------------------------------------

class VoiceCloner:
    """声音克隆"""

    def __init__(self):
        self._reference_embeddings: Dict[int, List[float]] = {}
        self._speaker_profiles: Dict[int, SpeakerProfile] = {}
        self._sample_rate: int = 22050
        self._n_fft: int = 512

    def register_speaker(self, speaker_id: int, audio_samples: List[float]):
        """注册说话人"""
        features = self._extract_voice_features(audio_samples)
        self._reference_embeddings[speaker_id] = features
        profile = self._build_profile(audio_samples)
        self._speaker_profiles[speaker_id] = profile

    def _build_profile(self, audio: List[float]) -> SpeakerProfile:
        """构建说话人配置"""
        if not audio:
            return SpeakerProfile()
        energy_vals = [a ** 2 for a in audio]
        mean_energy = sum(energy_vals) / len(energy_vals)
        rms = math.sqrt(mean_energy)
        zero_crossings = sum(1 for i in range(1, len(audio)) if audio[i] * audio[i - 1] < 0)
        duration = len(audio) / self._sample_rate
        zcr = zero_crossings / max(0.001, duration)
        estimated_pitch = zcr * 0.5 * 60
        estimated_pitch = max(80, min(400, estimated_pitch))
        return SpeakerProfile(
            pitch_mean=estimated_pitch,
            pitch_std=estimated_pitch * 0.1,
            speed=1.0,
            energy=min(1.0, rms * 5),
            timbre_features=self._extract_voice_features(audio),
        )

    def _extract_voice_features(self, audio: List[float]) -> List[float]:
        """提取声音特征"""
        if not audio:
            return [0.0] * 13
        mfcc = self._compute_mfcc(audio, n_mfcc=13)
        features: List[float] = []
        for i in range(min(13, len(mfcc))):
            coeffs = mfcc[i]
            features.append(sum(coeffs) / max(1, len(coeffs)))
        while len(features) < 13:
            features.append(0.0)
        return features

    def _compute_mfcc(self, audio: List[float], n_mfcc: int = 13) -> List[List[float]]:
        """MFCC特征(纯Python)"""
        n_fft = self._n_fft
        hop = 128
        n_mels = 26
        if len(audio) < n_fft:
            audio = audio + [0.0] * (n_fft - len(audio))
        n_frames = max(1, (len(audio) - n_fft) // hop + 1)
        window = [0.5 * (1 - math.cos(2 * math.pi * i / (n_fft - 1))) for i in range(n_fft)]
        mel_matrix = self._make_mel_filterbank(n_mels, n_fft // 2 + 1, self._sample_rate)
        mfccs: List[List[float]] = [[] for _ in range(n_mfcc)]
        for t in range(n_frames):
            start = t * hop
            frame = [audio[start + i] * window[i] if start + i < len(audio) else 0.0
                     for i in range(n_fft)]
            power = self._power_spectrum(frame)
            mel_energies = [0.0] * n_mels
            for m in range(n_mels):
                for k in range(len(power)):
                    mel_energies[m] += power[k] * mel_matrix[m][k]
            log_mel = [math.log(max(1e-10, e)) for e in mel_energies]
            dct = self._dct_ii(log_mel, n_mfcc)
            for c in range(n_mfcc):
                mfccs[c].append(dct[c])
        return mfccs

    def _power_spectrum(self, frame: List[float]) -> List[float]:
        """功率谱"""
        n = len(frame)
        power = []
        for k in range(n // 2 + 1):
            re = 0.0
            im = 0.0
            for i in range(n):
                angle = -2.0 * math.pi * k * i / n
                re += frame[i] * math.cos(angle)
                im += frame[i] * math.sin(angle)
            power.append((re * re + im * im) / (n * n))
        return power

    def _make_mel_filterbank(self, n_mels: int, n_freq: int, sr: int) -> List[List[float]]:
        """梅尔滤波器组"""
        def hz2mel(hz):
            return 2595 * math.log10(1 + hz / 700.0)

        def mel2hz(mel):
            return 700 * (10 ** (mel / 2595.0) - 1)

        mel_low = hz2mel(0)
        mel_high = hz2mel(sr / 2)
        mels = [mel_low + (mel_high - mel_low) * i / (n_mels + 1) for i in range(n_mels + 2)]
        hzs = [mel2hz(m) for m in mels]
        bins = [int((f / sr) * (2 * (n_freq - 1))) for f in hzs]
        bins = [max(0, min(b, n_freq - 1)) for b in bins]
        fb: List[List[float]] = []
        for m in range(n_mels):
            row = [0.0] * n_freq
            for k in range(bins[m], bins[m + 1]):
                if bins[m + 1] != bins[m]:
                    row[k] = (k - bins[m]) / (bins[m + 1] - bins[m])
            for k in range(bins[m + 1], bins[m + 2]):
                if bins[m + 2] != bins[m + 1]:
                    row[k] = (bins[m + 2] - k) / (bins[m + 2] - bins[m + 1])
            fb.append(row)
        return fb

    @staticmethod
    def _dct_ii(x: List[float], n_out: int) -> List[float]:
        """DCT-II"""
        n = len(x)
        result: List[float] = []
        for k in range(n_out):
            val = 0.0
            for i in range(n):
                val += x[i] * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
            result.append(val)
        return result

    def _compute_spectral_features(self, audio: List[float]) -> dict:
        """频谱特征"""
        if not audio:
            return {"centroid": 0.0, "bandwidth": 0.0, "rolloff": 0.0, "flatness": 0.0}
        n_fft = self._n_fft
        hop = 128
        n_frames = max(1, (len(audio) - n_fft) // hop + 1)
        window = [0.5 * (1 - math.cos(2 * math.pi * i / (n_fft - 1))) for i in range(n_fft)]
        centroids = []
        bandwidths = []
        rolloffs = []
        flatnesses = []
        for t in range(n_frames):
            start = t * hop
            frame = [audio[start + i] * window[i] if start + i < len(audio) else 0.0
                     for i in range(n_fft)]
            power = self._power_spectrum(frame)
            total = sum(power)
            if total < 1e-10:
                centroids.append(0.0)
                bandwidths.append(0.0)
                rolloffs.append(0.0)
                flatnesses.append(0.0)
                continue
            centroid = sum(k * power[k] for k in range(len(power))) / total
            variance = sum(power[k] * (k - centroid) ** 2 for k in range(len(power))) / total
            bw = math.sqrt(variance)
            cumsum = 0.0
            rolloff = 0.0
            for k in range(len(power)):
                cumsum += power[k]
                if cumsum >= 0.85 * total:
                    rolloff = float(k)
                    break
            log_power = [math.log(max(1e-10, p)) for p in power]
            geo_mean = math.exp(sum(log_power) / len(log_power))
            ari_mean = total / len(power)
            flatness = geo_mean / max(1e-10, ari_mean)
            centroids.append(centroid)
            bandwidths.append(bw)
            rolloffs.append(rolloff)
            flatnesses.append(flatness)
        return {
            "centroid": sum(centroids) / len(centroids),
            "bandwidth": sum(bandwidths) / len(bandwidths),
            "rolloff": sum(rolloffs) / len(rolloffs),
            "flatness": sum(flatnesses) / len(flatnesses),
        }

    def clone_voice(self, text: str, target_speaker_id: int) -> List[float]:
        """克隆声音"""
        if target_speaker_id not in self._speaker_profiles:
            return []
        profile = self._speaker_profiles[target_speaker_id]
        phonemizer = Phonemizer()
        phonemes = phonemizer.phonemize(text)
        acoustic = AcousticModel()
        mel_spec = acoustic.generate_mel_spectrogram(phonemes, target_speaker_id)
        adapted = self._adapt_acoustic_model(
            self._reference_embeddings.get(0, [0.0] * 13),
            self._reference_embeddings.get(target_speaker_id, [0.0] * 13),
        )
        for t in range(len(mel_spec)):
            for b in range(len(mel_spec[t])):
                mel_spec[t][b] *= (1.0 + adapted[min(b, len(adapted) - 1)] * 0.1)
                mel_spec[t][b] = max(0.0, min(1.0, mel_spec[t][b]))
        vocoder = Vocoder()
        waveform = vocoder.synthesize(mel_spec)
        speed_factor = profile.speed
        if speed_factor != 1.0:
            new_len = int(len(waveform) / speed_factor)
            waveform = self._resample(waveform, new_len)
        return waveform

    def _adapt_acoustic_model(self, source_features: List[float], target_features: List[float]) -> List[float]:
        """适配声学模型"""
        n = max(len(source_features), len(target_features))
        diff: List[float] = []
        for i in range(n):
            s = source_features[i] if i < len(source_features) else 0.0
            t = target_features[i] if i < len(target_features) else 0.0
            diff.append(t - s)
        return diff

    @staticmethod
    def _resample(audio: List[float], new_len: int) -> List[float]:
        """重采样"""
        if not audio or new_len <= 0:
            return []
        result: List[float] = []
        for i in range(new_len):
            pos = i * (len(audio) - 1) / max(1, new_len - 1)
            idx = int(pos)
            frac = pos - idx
            if idx + 1 < len(audio):
                result.append(audio[idx] * (1 - frac) + audio[idx + 1] * frac)
            else:
                result.append(audio[idx] if idx < len(audio) else 0.0)
        return result


# ---------------------------------------------------------------------------
# 5. TTSEngine - TTS引擎主类
# ---------------------------------------------------------------------------

class TTSEngine:
    """TTS引擎主类"""

    def __init__(self):
        self._phonemizer = Phonemizer()
        self._acoustic_model = AcousticModel()
        self._vocoder = Vocoder()
        self._voice_cloner = VoiceCloner()
        self._supported_languages = ["en", "zh", "ja", "ko", "fr", "de", "es"]

    def synthesize(self, text: str, speaker_id: int = 0, speed: float = 1.0,
                   pitch: float = 1.0, language: str = "en") -> GenerationResult:
        """文本转语音"""
        phonemes = self._phonemizer.phonemize(text, language)
        mel_spec = self._acoustic_model.generate_mel_spectrogram(phonemes, speaker_id)
        mel_spec = self._apply_prosody(mel_spec, pitch, speed)
        waveform = self._vocoder.synthesize(mel_spec)
        if speed != 1.0:
            new_len = int(len(waveform) / speed)
            waveform = VoiceCloner._resample(waveform, new_len)
        return GenerationResult(
            data=waveform,
            format="wav",
            sample_rate=self._acoustic_model._sample_rate,
            metadata={"phonemes": phonemes, "speaker_id": speaker_id, "speed": speed, "pitch": pitch},
        )

    def synthesize_with_emotion(self, text: str, emotion: str = "neutral") -> GenerationResult:
        """带情感的语音合成"""
        phonemes = self._phonemizer.phonemize(text)
        mel_spec = self._acoustic_model.generate_mel_spectrogram(phonemes, 0)
        mel_spec = self._add_emotion(mel_spec, emotion)
        waveform = self._vocoder.synthesize(mel_spec)
        return GenerationResult(
            data=waveform,
            format="wav",
            sample_rate=self._acoustic_model._sample_rate,
            metadata={"phonemes": phonemes, "emotion": emotion},
        )

    def _apply_prosody(self, mel_spec: List[List[float]], pitch: float, speed: float) -> List[List[float]]:
        """韵律调整"""
        if pitch == 1.0 and speed == 1.0:
            return mel_spec
        result: List[List[float]] = []
        for t, frame in enumerate(mel_spec):
            new_frame: List[float] = []
            for b, val in enumerate(frame):
                if pitch != 1.0:
                    shift = int((pitch - 1.0) * 10)
                    src_b = max(0, min(len(frame) - 1, b - shift))
                    val = frame[src_b]
                new_frame.append(val)
            result.append(new_frame)
        if speed != 1.0:
            new_len = max(1, int(len(result) / speed))
            resampled: List[List[float]] = []
            for i in range(new_len):
                pos = i * (len(result) - 1) / max(1, new_len - 1)
                idx = int(pos)
                frac = pos - idx
                if idx + 1 < len(result):
                    row = [result[idx][b] * (1 - frac) + result[idx + 1][b] * frac
                           for b in range(len(result[idx]))]
                else:
                    row = list(result[idx]) if idx < len(result) else [0.0] * self._acoustic_model._mel_bands
                resampled.append(row)
            result = resampled
        return result

    def _add_emotion(self, mel_spec: List[List[float]], emotion: str) -> List[List[float]]:
        """添加情感"""
        emotion_params: Dict[str, Dict[str, float]] = {
            "happy": {"pitch_shift": 0.15, "energy_boost": 0.2, "tempo": 1.1},
            "sad": {"pitch_shift": -0.1, "energy_boost": -0.2, "tempo": 0.85},
            "angry": {"pitch_shift": 0.1, "energy_boost": 0.3, "tempo": 1.15},
            "surprised": {"pitch_shift": 0.2, "energy_boost": 0.25, "tempo": 1.2},
            "fearful": {"pitch_shift": 0.05, "energy_boost": -0.1, "tempo": 0.9},
            "calm": {"pitch_shift": -0.05, "energy_boost": -0.15, "tempo": 0.9},
            "neutral": {"pitch_shift": 0.0, "energy_boost": 0.0, "tempo": 1.0},
        }
        params = emotion_params.get(emotion.lower(), emotion_params["neutral"])
        pitch_shift = params["pitch_shift"]
        energy_boost = params["energy_boost"]
        result: List[List[float]] = []
        for frame in mel_spec:
            new_frame: List[float] = []
            for b, val in enumerate(frame):
                freq_norm = b / max(1, len(frame) - 1)
                shifted = freq_norm + pitch_shift * 0.3
                shifted = max(0.0, min(1.0, shifted))
                src_b = int(shifted * (len(frame) - 1))
                src_b = max(0, min(len(frame) - 1, src_b))
                new_val = frame[src_b] * (1.0 + energy_boost)
                new_val = max(0.0, min(1.0, new_val))
                new_frame.append(new_val)
            result.append(new_frame)
        return result

    def _save_wav(self, audio: List[float], path: str):
        """保存WAV文件(纯Python写WAV头)"""
        sample_rate = self._acoustic_model._sample_rate
        bits_per_sample = 16
        num_channels = 1
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(audio) * num_channels * bits_per_sample // 8
        file_size = 36 + data_size
        with open(path, 'wb') as f:
            f.write(b'RIFF')
            f.write(struct.pack('<I', file_size))
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))
            f.write(struct.pack('<H', 1))
            f.write(struct.pack('<H', num_channels))
            f.write(struct.pack('<I', sample_rate))
            f.write(struct.pack('<I', byte_rate))
            f.write(struct.pack('<H', block_align))
            f.write(struct.pack('<H', bits_per_sample))
            f.write(b'data')
            f.write(struct.pack('<I', data_size))
            for sample in audio:
                int_val = int(max(-1.0, min(1.0, sample)) * 32767)
                f.write(struct.pack('<h', int_val))

    def get_available_speakers(self) -> List[int]:
        """可用说话人"""
        return list(self._voice_cloner._speaker_profiles.keys())

    def get_supported_languages(self) -> List[str]:
        """支持语言"""
        return list(self._supported_languages)


# ---------------------------------------------------------------------------
# 6. MusicGenerator - 音乐生成
# ---------------------------------------------------------------------------

class MusicGenerator:
    """音乐生成器"""

    def __init__(self):
        self._sample_rate: int = 32000
        self._genre_chords: Dict[str, List[List[str]]] = {
            "pop": [["C", "G", "Am", "F"], ["C", "F", "G", "C"]],
            "rock": [["E", "A", "B", "E"], ["A", "D", "E", "A"]],
            "jazz": [["Dm7", "G7", "Cmaj7", "Am7"], ["Cmaj7", "Em7", "Am7", "Dm7"]],
            "classical": [["C", "Am", "F", "G"], ["G", "C", "Am", "Dm"]],
            "blues": [["A7", "D7", "A7", "E7"], ["A7", "A7", "D7", "D7"]],
            "electronic": [["Am", "F", "C", "G"], ["Em", "Am", "Dm", "G"]],
        }
        self._note_freqs: Dict[str, float] = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88,
        }

    def generate(self, prompt: str, duration: float = 10.0, genre: str = "auto") -> GenerationResult:
        """生成音乐"""
        params = self._parse_music_prompt(prompt)
        if genre == "auto":
            genre = params.get("genre", "pop")
        bpm = params.get("bpm", 120)
        bars = max(1, int(duration * bpm / 240))
        chords = self._generate_chord_progression(genre, bars)
        melody = self._generate_melody(chords, genre)
        rhythm = self._generate_rhythm(genre, bpm)
        audio = self._synthesize_instruments(chords, melody, rhythm)
        target_len = int(duration * self._sample_rate)
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = audio + [0.0] * (target_len - len(audio))
        return GenerationResult(
            data=audio,
            format="wav",
            sample_rate=self._sample_rate,
            metadata={"genre": genre, "bpm": bpm, "duration": duration, "bars": bars},
        )

    def _parse_music_prompt(self, prompt: str) -> dict:
        """解析音乐提示词"""
        prompt_lower = prompt.lower()
        params: dict = {"genre": "pop", "bpm": 120, "mood": "neutral"}
        genre_keywords = {
            "pop": ["pop", "popular"], "rock": ["rock", "guitar"],
            "jazz": ["jazz", "swing"], "classical": ["classical", "orchestra", "piano"],
            "blues": ["blues", "blue"], "electronic": ["electronic", "edm", "techno", "synth"],
        }
        for genre, keywords in genre_keywords.items():
            for kw in keywords:
                if kw in prompt_lower:
                    params["genre"] = genre
                    break
        import re
        bpm_match = re.search(r'(\d+)\s*bpm', prompt_lower)
        if bpm_match:
            params["bpm"] = int(bpm_match.group(1))
        if "fast" in prompt_lower or "upbeat" in prompt_lower:
            params["bpm"] = 140
        elif "slow" in prompt_lower or "gentle" in prompt_lower:
            params["bpm"] = 80
        return params

    def _generate_chord_progression(self, genre: str, bars: int) -> List[List[str]]:
        """和弦进行生成"""
        templates = self._genre_chords.get(genre, self._genre_chords["pop"])
        progression: List[List[str]] = []
        for i in range(bars):
            template = templates[i % len(templates)]
            progression.append(list(template))
        return progression

    def _generate_melody(self, chords: List[List[str]], genre: str) -> List[Tuple[float, float]]:
        """旋律生成 - 返回 (频率, 起始时间) 列表"""
        melody: List[Tuple[float, float]] = []
        scale_intervals = [0, 2, 4, 5, 7, 9, 11, 12]
        random.seed(hash(genre) % 2 ** 31)
        beat_duration = 60.0 / 120.0
        for bar_idx, bar_chords in enumerate(chords):
            root = bar_chords[0][0] if bar_chords else 'C'
            root_freq = self._note_freqs.get(root, 261.63)
            for beat in range(4):
                t = (bar_idx * 4 + beat) * beat_duration
                interval = random.choice(scale_intervals)
                octave = random.choice([0.5, 1.0, 1.0, 2.0])
                freq = root_freq * octave * (2 ** (interval / 12.0))
                melody.append((freq, t))
                if random.random() < 0.3:
                    t2 = t + beat_duration * 0.5
                    interval2 = random.choice(scale_intervals)
                    freq2 = root_freq * octave * (2 ** (interval2 / 12.0))
                    melody.append((freq2, t2))
        return melody

    def _generate_rhythm(self, genre: str, bpm: float) -> List[float]:
        """节奏生成 - 返回打击时间点列表"""
        beat_dur = 60.0 / bpm
        rhythm: List[float] = []
        if genre in ("rock", "pop"):
            for bar in range(16):
                for beat in range(4):
                    rhythm.append((bar * 4 + beat) * beat_dur)
                    if beat in (1, 3):
                        rhythm.append((bar * 4 + beat) * beat_dur + beat_dur * 0.5)
        elif genre == "jazz":
            for bar in range(16):
                for i in range(8):
                    t = bar * 4 * beat_dur + i * beat_dur * 0.5
                    if random.random() < 0.7:
                        rhythm.append(t)
        elif genre == "electronic":
            for bar in range(16):
                for i in range(16):
                    t = bar * 4 * beat_dur + i * beat_dur * 0.25
                    if i % 4 == 0 or random.random() < 0.3:
                        rhythm.append(t)
        else:
            for bar in range(16):
                for beat in range(4):
                    rhythm.append((bar * 4 + beat) * beat_dur)
        return sorted(rhythm)

    def _synthesize_instruments(self, chords: List[List[str]], melody: List[Tuple[float, float]],
                                rhythm: List[float]) -> List[float]:
        """合成乐器"""
        total_bars = len(chords)
        beat_dur = 60.0 / 120.0
        total_dur = total_bars * 4 * beat_dur + 2.0
        total_samples = int(total_dur * self._sample_rate)
        audio = [0.0] * total_samples
        for bar_idx, bar_chords in enumerate(chords):
            for chord_name in bar_chords:
                root = chord_name[0] if chord_name else 'C'
                freq = self._note_freqs.get(root, 261.63)
                start_sample = int(bar_idx * 4 * beat_dur * self._sample_rate)
                dur_samples = int(4 * beat_dur * self._sample_rate)
                for i in range(dur_samples):
                    idx = start_sample + i
                    if idx >= total_samples:
                        break
                    t = i / self._sample_rate
                    env = math.exp(-t * 1.5)
                    val = env * math.sin(2 * math.pi * freq * t) * 0.15
                    fifth = freq * 1.5
                    val += env * math.sin(2 * math.pi * fifth * t) * 0.08
                    third = freq * 1.25
                    val += env * math.sin(2 * math.pi * third * t) * 0.06
                    audio[idx] += val
        for freq, start_time in melody:
            start_sample = int(start_time * self._sample_rate)
            note_dur = 0.3
            dur_samples = int(note_dur * self._sample_rate)
            for i in range(dur_samples):
                idx = start_sample + i
                if idx >= total_samples:
                    break
                t = i / self._sample_rate
                env = math.exp(-t * 4) * (1 - math.exp(-t * 50))
                val = env * math.sin(2 * math.pi * freq * t) * 0.2
                val += env * 0.3 * math.sin(2 * math.pi * freq * 2 * t)
                audio[idx] += val
        for t in rhythm:
            sample = int(t * self._sample_rate)
            dur = int(0.05 * self._sample_rate)
            for i in range(dur):
                idx = sample + i
                if idx >= total_samples:
                    break
                ti = i / self._sample_rate
                env = math.exp(-ti * 40)
                val = env * (random.random() * 2 - 1) * 0.15
                audio[idx] += val
        max_val = max(abs(v) for v in audio) if audio else 1.0
        if max_val > 0:
            audio = [v / max_val * 0.9 for v in audio]
        return audio

    def _generate_drum_pattern(self, genre: str, bpm: float, bars: int) -> List[List[float]]:
        """鼓点模式"""
        beat_dur = 60.0 / bpm
        total_samples = int(bars * 4 * beat_dur * self._sample_rate)
        kick = [0.0] * total_samples
        snare = [0.0] * total_samples
        hihat = [0.0] * total_samples
        for bar in range(bars):
            for beat in range(4):
                t = (bar * 4 + beat) * beat_dur
                sample = int(t * self._sample_rate)
                dur = int(0.1 * self._sample_rate)
                for i in range(dur):
                    idx = sample + i
                    if idx >= total_samples:
                        break
                    ti = i / self._sample_rate
                    env = math.exp(-ti * 20)
                    kick[idx] += env * math.sin(2 * math.pi * 60 * ti * math.exp(-ti * 10)) * 0.3
                if beat in (1, 3):
                    dur_s = int(0.08 * self._sample_rate)
                    for i in range(dur_s):
                        idx = sample + i
                        if idx >= total_samples:
                            break
                        ti = i / self._sample_rate
                        env = math.exp(-ti * 30)
                        snare[idx] += env * (random.random() * 2 - 1) * 0.2
                for sub in range(2):
                    t_h = t + sub * beat_dur * 0.5
                    s_h = int(t_h * self._sample_rate)
                    dur_h = int(0.03 * self._sample_rate)
                    for i in range(dur_h):
                        idx = s_h + i
                        if idx >= total_samples:
                            break
                        ti = i / self._sample_rate
                        env = math.exp(-ti * 60)
                        hihat[idx] += env * (random.random() * 2 - 1) * 0.1
        return [kick, snare, hihat]

    def _apply_reverb(self, audio: List[float], decay: float = 0.5) -> List[float]:
        """混响效果"""
        if not audio:
            return []
        delay_samples = int(0.05 * self._sample_rate)
        result = list(audio)
        n_delays = 6
        for d in range(n_delays):
            delay = delay_samples * (d + 1)
            gain = decay ** (d + 1)
            for i in range(delay, len(audio)):
                result[i] += audio[i - delay] * gain
        max_val = max(abs(v) for v in result) if result else 1.0
        if max_val > 0:
            result = [v / max_val for v in result]
        return result

    def _apply_eq(self, audio: List[float], gains: Dict[str, float]) -> List[float]:
        """均衡器"""
        if not audio:
            return []
        n = len(audio)
        result = list(audio)
        sr = self._sample_rate
        for band, gain_db in gains.items():
            if band == "low":
                cutoff = 300.0
                alpha = 2 * math.pi * cutoff / sr
                b0 = alpha / (1 + alpha)
                a1 = (1 - alpha) / (1 + alpha)
                gain = 10 ** (gain_db / 20.0)
                prev = 0.0
                for i in range(n):
                    curr = b0 * (audio[i] + prev) + a1 * result[i]
                    prev = audio[i]
                    result[i] = result[i] * (1 - b0) + curr * gain * b0
            elif band == "mid":
                cutoff = 1000.0
                alpha = 2 * math.pi * cutoff / sr
                b0 = alpha / (1 + alpha)
                a1 = (1 - alpha) / (1 + alpha)
                gain = 10 ** (gain_db / 20.0)
                prev = 0.0
                for i in range(n):
                    curr = b0 * (audio[i] + prev) + a1 * result[i]
                    prev = audio[i]
                    result[i] = result[i] * (1 - b0) + curr * gain * b0
            elif band == "high":
                cutoff = 4000.0
                alpha = 2 * math.pi * cutoff / sr
                b0 = alpha / (1 + alpha)
                a1 = (1 - alpha) / (1 + alpha)
                gain = 10 ** (gain_db / 20.0)
                prev = 0.0
                for i in range(n):
                    curr = b0 * (audio[i] + prev) + a1 * result[i]
                    prev = audio[i]
                    result[i] = result[i] * (1 - b0) + curr * gain * b0
        max_val = max(abs(v) for v in result) if result else 1.0
        if max_val > 0:
            result = [v / max_val for v in result]
        return result


# ---------------------------------------------------------------------------
# 7. SoundEffectGenerator - 音效生成
# ---------------------------------------------------------------------------

class SoundEffectGenerator:
    """音效生成器"""

    def __init__(self):
        self._sample_rate: int = 22050

    def generate(self, description: str, duration: float = 2.0) -> GenerationResult:
        """生成音效"""
        template = self._match_sound_template(description)
        n_samples = int(duration * self._sample_rate)
        if template == "footstep":
            audio = self._generate_footstep("hard")
        elif template == "rain":
            audio = self._generate_rain(0.5)
        elif template == "wind":
            audio = self._generate_wind(0.5)
        elif template == "explosion":
            audio = self._generate_explosion(10.0)
        elif template == "beep":
            audio = self._generate_tone(880, duration)
        elif template == "siren":
            audio = self._generate_siren(duration)
        else:
            audio = self._white_noise(duration, 0.3)
        if len(audio) > n_samples:
            audio = audio[:n_samples]
        elif len(audio) < n_samples:
            audio = audio + [0.0] * (n_samples - len(audio))
        return GenerationResult(
            data=audio,
            format="wav",
            sample_rate=self._sample_rate,
            metadata={"description": description, "template": template, "duration": duration},
        )

    def _match_sound_template(self, description: str) -> str:
        """匹配音效模板"""
        desc = description.lower()
        templates = {
            "footstep": ["footstep", "foot", "step", "walk", "stomp"],
            "rain": ["rain", "raindrop", "drizzle", "shower"],
            "wind": ["wind", "breeze", "gust", "blow"],
            "explosion": ["explosion", "boom", "blast", "bang", "crash"],
            "beep": ["beep", "tone", "ping", "ding", "alert"],
            "siren": ["siren", "alarm", "warning", "horn"],
        }
        for name, keywords in templates.items():
            for kw in keywords:
                if kw in desc:
                    return name
        return "noise"

    def _generate_footstep(self, surface: str) -> List[float]:
        """脚步声"""
        sr = self._sample_rate
        audio: List[float] = []
        for step in range(4):
            step_audio: List[float] = []
            dur = 0.15
            n = int(dur * sr)
            random.seed(step * 17)
            for i in range(n):
                t = i / sr
                env = math.exp(-t * 30)
                if surface == "hard":
                    val = env * (random.random() * 2 - 1) * 0.5
                    val += env * math.sin(2 * math.pi * 200 * t) * 0.3
                else:
                    val = env * (random.random() * 2 - 1) * 0.3
                    val += env * math.sin(2 * math.pi * 100 * t) * 0.2
                step_audio.append(val)
            gap = [0.0] * int(0.3 * sr)
            audio.extend(step_audio)
            audio.extend(gap)
        return audio

    def _generate_rain(self, intensity: float) -> List[float]:
        """雨声"""
        sr = self._sample_rate
        duration = 2.0
        n = int(duration * sr)
        audio: List[float] = []
        random.seed(42)
        for i in range(n):
            t = i / sr
            val = 0.0
            for drop in range(int(intensity * 20)):
                freq = random.uniform(2000, 8000)
                phase = random.uniform(0, 2 * math.pi)
                amp = random.uniform(0.01, 0.05) * intensity
                val += amp * math.sin(2 * math.pi * freq * t + phase)
            val += (random.random() * 2 - 1) * 0.02 * intensity
            audio.append(val)
        return audio

    def _generate_wind(self, speed: float) -> List[float]:
        """风声"""
        sr = self._sample_rate
        duration = 2.0
        n = int(duration * sr)
        audio: List[float] = []
        random.seed(123)
        for i in range(n):
            t = i / sr
            mod = 0.5 + 0.5 * math.sin(2 * math.pi * 0.2 * t)
            mod += 0.3 * math.sin(2 * math.pi * 0.07 * t + 1.0)
            val = (random.random() * 2 - 1) * 0.3 * speed * mod
            low = math.sin(2 * math.pi * 80 * t + math.sin(2 * math.pi * 0.3 * t) * 3) * 0.1 * speed
            audio.append(val + low)
        return audio

    def _generate_explosion(self, distance: float) -> List[float]:
        """爆炸声"""
        sr = self._sample_rate
        duration = 2.0
        n = int(duration * sr)
        audio: List[float] = []
        random.seed(999)
        decay_rate = max(0.5, distance * 0.3)
        for i in range(n):
            t = i / sr
            env = math.exp(-t * decay_rate)
            noise = (random.random() * 2 - 1) * env * 0.8
            low_freq = math.sin(2 * math.pi * 50 * t * math.exp(-t * 2)) * env * 0.5
            mid_freq = math.sin(2 * math.pi * 200 * t * math.exp(-t * 5)) * env * 0.3
            shock = 0.0
            if t < 0.05:
                shock = math.sin(2 * math.pi * 30 * t) * (1 - t / 0.05) * 0.6
            audio.append(noise + low_freq + mid_freq + shock)
        return audio

    def _generate_tone(self, freq: float, duration: float) -> List[float]:
        """生成纯音"""
        sr = self._sample_rate
        n = int(duration * sr)
        audio: List[float] = []
        for i in range(n):
            t = i / sr
            env = min(1.0, t * 20) * math.exp(-t * 2)
            val = env * math.sin(2 * math.pi * freq * t) * 0.5
            audio.append(val)
        return audio

    def _generate_siren(self, duration: float) -> List[float]:
        """生成警报声"""
        sr = self._sample_rate
        n = int(duration * sr)
        audio: List[float] = []
        for i in range(n):
            t = i / sr
            freq = 600 + 400 * math.sin(2 * math.pi * 2 * t)
            val = math.sin(2 * math.pi * freq * t) * 0.4
            audio.append(val)
        return audio

    def _white_noise(self, duration: float, amplitude: float) -> List[float]:
        """白噪声"""
        n = int(duration * self._sample_rate)
        random.seed(0)
        return [(random.random() * 2 - 1) * amplitude for _ in range(n)]

    def _pink_noise(self, duration: float) -> List[float]:
        """粉红噪声"""
        n = int(duration * self._sample_rate)
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
        audio: List[float] = []
        random.seed(0)
        for _ in range(n):
            white = random.random() * 2 - 1
            b0 = 0.99886 * b0 + white * 0.0555179
            b1 = 0.99332 * b1 + white * 0.0750759
            b2 = 0.96900 * b2 + white * 0.1538520
            b3 = 0.86650 * b3 + white * 0.3104856
            b4 = 0.55000 * b4 + white * 0.5329522
            b5 = -0.7616 * b5 - white * 0.0168980
            pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362
            b6 = white * 0.115926
            audio.append(pink * 0.11)
        return audio

    def _apply_envelope(self, audio: List[float], attack: float, decay: float,
                        sustain: float, release: float) -> List[float]:
        """ADSR包络"""
        if not audio:
            return []
        sr = self._sample_rate
        n = len(audio)
        attack_samples = int(attack * sr)
        decay_samples = int(decay * sr)
        release_samples = int(release * sr)
        sustain_samples = max(0, n - attack_samples - decay_samples - release_samples)
        result: List[float] = []
        idx = 0
        for i in range(attack_samples):
            if idx < n:
                env = i / max(1, attack_samples)
                result.append(audio[idx] * env)
                idx += 1
        for i in range(decay_samples):
            if idx < n:
                env = 1.0 - (1.0 - sustain) * (i / max(1, decay_samples))
                result.append(audio[idx] * env)
                idx += 1
        for i in range(sustain_samples):
            if idx < n:
                result.append(audio[idx] * sustain)
                idx += 1
        for i in range(release_samples):
            if idx < n:
                env = sustain * (1.0 - i / max(1, release_samples))
                result.append(audio[idx] * env)
                idx += 1
        while idx < n:
            result.append(0.0)
            idx += 1
        return result

    def _lowpass_filter(self, audio: List[float], cutoff: float) -> List[float]:
        """低通滤波器(纯Python) - 一阶IIR"""
        if not audio:
            return []
        sr = self._sample_rate
        rc = 1.0 / (2.0 * math.pi * cutoff)
        dt = 1.0 / sr
        alpha = dt / (rc + dt)
        result: List[float] = [audio[0] * alpha]
        for i in range(1, len(audio)):
            result.append(alpha * audio[i] + (1 - alpha) * result[i - 1])
        return result

    def _highpass_filter(self, audio: List[float], cutoff: float) -> List[float]:
        """高通滤波器(纯Python) - 一阶IIR"""
        if not audio:
            return []
        sr = self._sample_rate
        rc = 1.0 / (2.0 * math.pi * cutoff)
        dt = 1.0 / sr
        alpha = rc / (rc + dt)
        result: List[float] = [0.0]
        for i in range(1, len(audio)):
            result.append(alpha * result[i - 1] + alpha * (audio[i] - audio[i - 1]))
        return result
