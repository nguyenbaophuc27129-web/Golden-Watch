"""
MODULE 2: SPEECH ANALYSIS - PSCS v8.0 (WITH ML)
Phân tích giọng nói để phát hiện đột quỵ (Dysarthria) với ML Model

Tác giả: PSCS Team
Ngày: 28/08/2026
"""

import os
import sys
import time
import numpy as np
import librosa
import torch
import torch.nn as nn
import joblib
from vosk import Model, KaldiRecognizer
import json
from collections import deque

# Optional dependencies
PYAUDIO_AVAILABLE = False
SOUNDDEVICE_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    print("[SPEECH] sounddevice not installed. Install: pip install sounddevice")
    print("[SPEECH] Audio recording will be unavailable")


class DysarthriaClassifier(nn.Module):
    """ML Classifier cho dysarthria detection"""

    def __init__(self, input_dim=48, hidden_dims=[256, 128, 64], num_classes=2, dropout=0.5):
        super(DysarthriaClassifier, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SpeechAnalysisModule:
    """Module phân tích giọng nói phát hiện đột quỵ với ML Model"""

    def __init__(self, vosk_model_path=None, ml_model_path=None, scaler_path=None, sample_rate=16000):
        """
        Khởi tạo Speech Analysis Module

        Args:
            vosk_model_path: Đường dẫn đến Vosk model (Tiếng Việt)
            ml_model_path: Đường dẫn đến ML model (.pth)
            scaler_path: Đường dẫn đến scaler (.pkl)
            sample_rate: Tần số mẫu (16kHz cho Vosk)
        """
        self.sample_rate = sample_rate
        self.vosk_model_path = vosk_model_path
        self.ml_model_path = ml_model_path
        self.scaler_path = scaler_path

        # Vosk components
        self.vosk_model = None
        self.recognizer = None

        # ML components
        self.ml_model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Personal baseline (WPM calibration) - None = dùng ngưỡng phổ quát
        self.baseline = None

        # Load models
        self._load_models()

    def _load_models(self):
        """Load cả Vosk và ML models"""
        # Load Vosk
        if self.vosk_model_path and os.path.exists(self.vosk_model_path):
            try:
                self.vosk_model = Model(self.vosk_model_path)
                self.recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
                self.recognizer.SetWords(True)
                print(f"[SPEECH] Vosk model loaded from: {self.vosk_model_path}")
            except Exception as e:
                print(f"[SPEECH] Error loading Vosk model: {e}")

        # Load ML model
        if self.ml_model_path and os.path.exists(self.ml_model_path):
            try:
                # Check PyTorch version for weights_only parameter
                torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
                if torch_version >= (2, 6):
                    state_dict = torch.load(self.ml_model_path, weights_only=True, map_location=self.device)
                else:
                    state_dict = torch.load(self.ml_model_path, map_location=self.device)

                # Auto-detect architecture from checkpoint
                # Layer pattern: Linear(4i), BatchNorm(4i+1), ReLU(4i+2), Dropout(4i+3)
                hidden_dims = []
                idx = 0
                while f'network.{idx}.weight' in state_dict:
                    hidden_dims.append(state_dict[f'network.{idx}.weight'].shape[0])
                    idx += 4
                num_classes = hidden_dims.pop()
                input_dim = state_dict['network.0.weight'].shape[1]

                self.ml_model = DysarthriaClassifier(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    num_classes=num_classes
                )
                self.ml_model.load_state_dict(state_dict)
                self.ml_model.to(self.device)
                self.ml_model.eval()
                print(f"[SPEECH] ML model loaded from: {self.ml_model_path}")
                print(f"[SPEECH] Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {num_classes}")
            except Exception as e:
                self.ml_model = None  # Fail-safe: fallback to rule-based
                print(f"[SPEECH] Error loading ML model: {e}")
                print("[SPEECH] Falling back to rule-based scoring")

        # Load scaler
        if self.scaler_path and os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                print(f"[SPEECH] Scaler loaded from: {self.scaler_path}")
            except Exception as e:
                print(f"[SPEECH] Error loading scaler: {e}")

    def record_audio(self, duration_seconds=5):
        """
        Thu âm thanh từ microphone

        Args:
            duration_seconds: Thời gian ghi âm (giây)

        Returns:
            audio_data: Mảng numpy chứa audio
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice is not installed. Install with: pip install sounddevice")

        try:
            import sounddevice as sd

            # Record audio
            frames = int(duration_seconds * self.sample_rate)
            recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype='int16')
            sd.wait()  # Wait until recording is finished

            # Convert to numpy array and normalize
            audio_data = recording.flatten()
            audio_float = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            return audio_float

        except Exception as e:
            print(f"[SPEECH] Error recording audio: {e}")
            return None

    def detect_voice_activity(self, audio, frame_ms=30, energy_factor=0.1):
        """
        VAD (Voice Activity Detection) - lọc nhiễu nền bằng năng lượng khung.

        Dùng để: (1) tính speech_ratio (tỉ lệ khung có giọng nói),
                 (2) phát hiện NO_SPEECH, (3) đo nhiễu nền.
        Lưu ý: KHÔNG cắt frame để đưa vào ML model (model được train trên
        audio nguyên bản), VAD chỉ dùng cho metrics.

        Returns:
            dict: speech_ratio (0-1), noise_floor, has_speech (bool)
        """
        frame_len = int(self.sample_rate * frame_ms / 1000)
        if len(audio) < frame_len:
            return {'speech_ratio': 0.0, 'noise_floor': 0.0, 'has_speech': False}

        n_frames = len(audio) // frame_len
        frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
        energy = np.sum(frames ** 2, axis=1)

        max_e = float(energy.max())
        if max_e <= 0:
            return {'speech_ratio': 0.0, 'noise_floor': 0.0, 'has_speech': False}

        # Ngưỡng = 10% năng lượng khung cao nhất (robust với nhiễu nền ổn định)
        threshold = max_e * energy_factor
        voiced = energy > threshold

        # Giảm rung: một khung im lặng kẹp giữa 2 khung có tiếng vẫn giữ
        smoothed = voiced.copy()
        smoothed[1:-1] = voiced[1:-1] | voiced[:-2] | voiced[2:]

        speech_ratio = float(voiced.sum()) / n_frames
        noise_floor = float(np.percentile(energy, 10))

        return {
            'speech_ratio': round(speech_ratio, 3),
            'noise_floor': noise_floor,
            'has_speech': bool(voiced.sum() >= max(1, n_frames // 10))
        }

    def select_best_window(self, audio, duration_seconds=5, hop_seconds=0.5,
                           frame_ms=30, energy_factor=0.1):
        """
        M2-10: VAD-guided window selection.

        Quét các cửa sổ dài duration_seconds (bước hop_seconds), chọn cửa sổ
        có speech_ratio cao nhất dựa trên năng lượng khung (cùng thuật toán
        detect_voice_activity — ngưỡng 10% năng lượng max).

        Lý do: training dùng 5s đầu file (đều là câu đọc có tiếng), nên lúc
        test/production phải chọn cửa sổ CÓ TIẾNG để khớp phân phối train.
        Chọn 5s đầu file WAV của session headMic (thường im lặng) làm
        features lệch về "im lặng" → dysarthria thật bị miss.

        Returns:
            (start_sample, speech_ratio): vị trí cửa sổ tốt nhất + tỉ lệ tiếng
        """
        win_len = int(self.sample_rate * duration_seconds)
        if len(audio) == 0:
            return 0, 0.0
        # Audio ngắn hơn/hơn bằng 1 cửa sổ → dùng nguyên bản
        if len(audio) <= win_len:
            ratio = self.detect_voice_activity(audio, frame_ms, energy_factor)['speech_ratio']
            return 0, ratio

        frame_len = int(self.sample_rate * frame_ms / 1000)
        n_frames = len(audio) // frame_len
        if n_frames == 0:
            return 0, 0.0

        # Năng lượng từng khung trên TOÀN bộ audio (vectorized)
        frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
        energy = np.sum(frames ** 2, axis=1)
        max_e = float(energy.max())
        if max_e <= 0:
            return 0, 0.0

        voiced = (energy > max_e * energy_factor).astype(np.int64)

        # Sliding window qua cumulative sum (O(1) mỗi cửa sổ)
        frames_per_win = win_len // frame_len
        frames_per_hop = max(1, int(self.sample_rate * hop_seconds) // frame_len)
        cum = np.concatenate(([0], np.cumsum(voiced)))

        best_start_frame, best_ratio = 0, -1.0
        last_start = n_frames - frames_per_win
        for start_f in range(0, last_start + 1, frames_per_hop):
            end_f = min(start_f + frames_per_win, n_frames)
            ratio = (cum[end_f] - cum[start_f]) / (end_f - start_f)
            if ratio > best_ratio:
                best_ratio, best_start_frame = ratio, start_f

        return best_start_frame * frame_len, round(float(best_ratio), 3)

    def extract_window(self, audio, duration_seconds=5):
        """
        Trích cửa sổ có nhiều tiếng nói nhất từ audio (chuẩn bị cho ML).

        Returns:
            (window, info): window đúng duration giây (pad nếu thiếu);
                info = {'window_start_s', 'window_speech_ratio'}
        """
        start, ratio = self.select_best_window(audio, duration_seconds)
        win_len = int(self.sample_rate * duration_seconds)
        window = audio[start:start + win_len]
        if len(window) < win_len:
            window = np.pad(window, (0, win_len - len(window)))
        info = {
            'window_start_s': round(start / self.sample_rate, 2),
            'window_speech_ratio': ratio,
        }
        return window, info

    def set_baseline(self, wpm=None, pitch_mean=None, session_name='user_default'):
        """
        Đặt baseline cá nhân (từ calibration hoặc load từ file)
        """
        self.baseline = {
            'session_name': session_name,
            'wpm': float(wpm) if wpm else None,
            'pitch_mean': float(pitch_mean) if pitch_mean else None,
            'updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    def save_baseline(self, file_path):
        """Lưu baseline ra file JSON"""
        if self.baseline is None:
            print("[SPEECH] No baseline to save")
            return False
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.baseline, f, indent=2)
            print(f"[SPEECH] Baseline saved to: {file_path}")
            return True
        except Exception as e:
            print(f"[SPEECH] Error saving baseline: {e}")
            return False

    def load_baseline(self, file_path):
        """Load baseline cá nhân từ file JSON (im lặng nếu file chưa tồn tại)"""
        if not file_path or not os.path.exists(file_path):
            return False
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.baseline = json.load(f)
            print(f"[SPEECH] Baseline loaded from: {file_path} (wpm={self.baseline.get('wpm')})")
            return True
        except Exception as e:
            print(f"[SPEECH] No baseline loaded ({e}) - using universal thresholds")
            return False

    def calibrate_baseline(self, duration_seconds=30, save_path=None):
        """
        Calibration nhanh (~30 giây): user đọc 1 đoạn văn tự nhiên.
        Output: baseline WPM + pitch_mean cá nhân (dùng cho Layer 1 Defense sau này).
        """
        audio = self.record_audio(duration_seconds=duration_seconds)
        if audio is None:
            print("[SPEECH] Calibration recording failed")
            return None

        text, word_count = self.transcribe_speech(audio)
        wpm = self.calculate_wpm(word_count, duration_seconds)
        _, features_dict = self.extract_features(audio)
        pitch_mean = float(features_dict.get('pitch_mean', [0])[0])

        self.set_baseline(wpm=wpm, pitch_mean=pitch_mean)
        if save_path:
            self.save_baseline(save_path)

        print(f"[SPEECH] Calibration done: wpm={wpm:.1f}, pitch_mean={pitch_mean:.1f}Hz, words={word_count}")
        return self.baseline

    def extract_features(self, audio):
        """
        Trích xuất features từ audio (48 features total)

        Args:
            audio: Mảng numpy audio

        Returns:
            features: Mảng numpy 48 features
        """
        try:
            features_dict = {}

            # 1. MFCC features (13+13+13 = 39 features)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            features_dict['mfcc_mean'] = np.mean(mfcc, axis=1)  # 13
            features_dict['mfcc_std'] = np.std(mfcc, axis=1)   # 13
            features_dict['mfcc_delta'] = np.mean(np.diff(mfcc, axis=1), axis=1)  # 13

            # 2. Pitch features (4 features)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                pitch = pitches[idx, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if len(pitch_values) > 0:
                features_dict['pitch_mean'] = [np.mean(pitch_values)]
                features_dict['pitch_std'] = [np.std(pitch_values)]
                features_dict['pitch_min'] = [np.min(pitch_values)]
                features_dict['pitch_max'] = [np.max(pitch_values)]
            else:
                features_dict['pitch_mean'] = [0]
                features_dict['pitch_std'] = [0]
                features_dict['pitch_min'] = [0]
                features_dict['pitch_max'] = [0]

            # 3. Energy features (3 features)
            frame_length = int(self.sample_rate * 0.02)
            energy = []
            for i in range(0, len(audio) - frame_length, frame_length):
                frame = audio[i:i + frame_length]
                energy.append(np.sum(frame ** 2))

            if len(energy) > 0:
                features_dict['energy_mean'] = [np.mean(energy)]
                features_dict['energy_std'] = [np.std(energy)]
                features_dict['energy_range'] = [np.max(energy) - np.min(energy)]
            else:
                features_dict['energy_mean'] = [0]
                features_dict['energy_std'] = [0]
                features_dict['energy_range'] = [0]

            # 4. ZCR features (2 features)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features_dict['zcr_mean'] = [np.mean(zcr)]
            features_dict['zcr_std'] = [np.std(zcr)]

            # Concatenate all features (13+13+13+4+3+2 = 48)
            features = np.concatenate([
                features_dict['mfcc_mean'],
                features_dict['mfcc_std'],
                features_dict['mfcc_delta'],
                features_dict['pitch_mean'],
                features_dict['pitch_std'],
                features_dict['pitch_min'],
                features_dict['pitch_max'],
                features_dict['energy_mean'],
                features_dict['energy_std'],
                features_dict['energy_range'],
                features_dict['zcr_mean'],
                features_dict['zcr_std']
            ])

            return features, features_dict

        except Exception as e:
            print(f"[SPEECH] Feature extraction error: {e}")
            return np.zeros(48), {}

    def transcribe_speech(self, audio):
        """
        Speech-to-Text sử dụng Vosk

        Args:
            audio: Mảng numpy audio

        Returns:
            text: Văn bản được transcribe
            word_count: Số từ được nhận diện
        """
        if self.vosk_model is None or self.recognizer is None:
            return "", 0

        try:
            # Convert audio to int16 for Vosk
            audio_int16 = (audio * 32768).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            # Reset recognizer
            self.recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
            self.recognizer.SetWords(True)

            # Feed to recognizer
            if self.recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.recognizer.Result())
                text = result.get('text', '')
                word_count = len(text.split())
                return text, word_count
            else:
                # Partial result
                partial = json.loads(self.recognizer.PartialResult())
                partial_text = partial.get('partial', '')
                return partial_text, len(partial_text.split())

        except Exception as e:
            print(f"[SPEECH] Transcription error: {e}")
            return "", 0

    def calculate_wpm(self, word_count, duration_seconds):
        """Tính WPM (Words Per Minute)"""
        if duration_seconds <= 0:
            return 0.0
        return (word_count / duration_seconds) * 60

    def predict_dysarthria(self, audio, duration_seconds):
        """
        Dự đoán dysarthria sử dụng ML model

        Args:
            audio: Mảng numpy audio
            duration_seconds: Thời gian thu âm (giây)

        Returns:
            results: Dictionary chứa tất cả metrics và prediction
        """
        results = {
            'status': 'NO_SPEECH',
            'speech_prob': 0.0,
            'nihss_score': 0,
            'metrics': {}
        }

        try:
            # 0. VAD - lọc nhiễu nền, phát hiện im lặng
            vad = self.detect_voice_activity(audio)
            results['metrics']['speech_ratio'] = vad['speech_ratio']
            results['metrics']['noise_floor'] = round(vad['noise_floor'], 6)

            # 1. Extract features
            features, features_dict = self.extract_features(audio)

            # 2. Transcribe speech
            text, word_count = self.transcribe_speech(audio)
            results['metrics']['transcript'] = text
            results['metrics']['word_count'] = word_count

            # 3. Calculate WPM (so với baseline cá nhân nếu có)
            wpm = self.calculate_wpm(word_count, duration_seconds)
            results['metrics']['wpm'] = wpm
            if self.baseline and self.baseline.get('wpm'):
                results['metrics']['wpm_baseline'] = self.baseline['wpm']
                results['metrics']['wpm_delta_percent'] = round(
                    (wpm - self.baseline['wpm']) / self.baseline['wpm'] * 100, 1
                )

            # 4. Add detailed features to results
            results['metrics']['pitch_mean'] = float(features_dict.get('pitch_mean', [0])[0])
            results['metrics']['pitch_std'] = float(features_dict.get('pitch_std', [0])[0])
            results['metrics']['energy_mean'] = float(features_dict.get('energy_mean', [0])[0])
            results['metrics']['zcr_mean'] = float(features_dict.get('zcr_mean', [0])[0])

            # 5. ML prediction
            if self.ml_model is not None and self.scaler is not None:
                # Scale features
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                features_tensor = torch.FloatTensor(features_scaled).to(self.device)

                # Predict
                with torch.no_grad():
                    outputs = self.ml_model(features_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    dysarthria_prob = probs[0][1].item() * 100  # Probability of dysarthria

                results['speech_prob'] = dysarthria_prob
            else:
                # Fallback to rule-based
                results['speech_prob'] = self._rule_based_score(wpm, features_dict)

            # 6. Determine status
            # NO_SPEECH: gần như im lặng hoàn toàn + không nhận dạng được từ nào
            if vad['speech_ratio'] < 0.03 and word_count == 0:
                results['status'] = 'NO_SPEECH'
                # Giữ prob thô trong metrics để debug, reset prob chính
                # (NO_SPEECH không được phép báo động ở tầng fusion)
                results['metrics']['raw_prob'] = results['speech_prob']
                results['speech_prob'] = 0.0
                return results
            if results['speech_prob'] < 30:
                results['status'] = 'NORMAL'
            elif results['speech_prob'] < 60:
                results['status'] = 'WARNING'
            else:
                results['status'] = 'DANGER'

            # 7. Map to NIHSS Item 10 (Dysarthria)
            results['nihss_score'] = self._map_to_nihss(results['speech_prob'])

            return results

        except Exception as e:
            print(f"[SPEECH] Prediction error: {e}")
            return results

    def _rule_based_score(self, wpm, features_dict):
        """Fallback rule-based scoring (dùng baseline cá nhân nếu có)"""
        score = 0.0

        # WPM deviation (baseline cá nhân ưu tiên, mặc định 150)
        if self.baseline and self.baseline.get('wpm'):
            reference_wpm = float(self.baseline['wpm'])
            # Lệch quá 40% so với chính mình → bất thường
            wpm_deviation = abs(wpm - reference_wpm) / reference_wpm * 100
            if wpm_deviation > 40:
                score += 30
            elif wpm_deviation > 25:
                score += 20
        else:
            wpm_deviation = abs(wpm - 150)
            if wpm_deviation > 50:
                score += 30
            elif wpm_deviation > 30:
                score += 20

        # Pitch variability
        pitch_std = features_dict.get('pitch_std', [0])[0] if isinstance(features_dict.get('pitch_std'), list) else 0
        if pitch_std > 50:
            score += 20

        # Energy variability
        energy_std = features_dict.get('energy_std', [0])[0] if isinstance(features_dict.get('energy_std'), list) else 0
        if energy_std > 0.0003:
            score += 10

        return min(score, 100.0)

    def _map_to_nihss(self, speech_score):
        """Mapping speech score sang NIHSS Item 10 (Dysarthria)"""
        if speech_score < 30:
            return 0  # No articulatory disturbance
        elif speech_score < 50:
            return 1  # Mild to moderate dysarthria
        elif speech_score < 70:
            return 2  # Severe dysarthria
        else:
            return 3  # Severe dysarthria or mute


# Test function
def test_speech_module():
    """Test Speech Module với microphone"""
    # Get parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))

    vosk_model_path = os.path.join(project_dir, "models/vosk-model-vn-0.4")
    ml_model_path = os.path.join(project_dir, "models/speech_classifier_20260828_175927.pth")
    scaler_path = os.path.join(project_dir, "models/speech_classifier_20260828_175927_scaler.pkl")

    # Initialize module
    speech = SpeechAnalysisModule(
        vosk_model_path=vosk_model_path,
        ml_model_path=ml_model_path,
        scaler_path=scaler_path
    )

    print("="*70)
    print("MODULE 2 SPEECH TEST (WITH ML MODEL)")
    print("="*70)
    print()
    print("Instructions:")
    print("  - Speak clearly in Vietnamese")
    print("  - Count from 1 to 10")
    print()
    print("Press ENTER to start recording (5 seconds)...")
    input()

    audio = speech.record_audio(duration_seconds=5)
    if audio is not None:
        results = speech.predict_dysarthria(audio, 5)
        print(f"\nResults: {results['status']}, Score: {results['speech_prob']:.1f}%, NIHSS: {results['nihss_score']}")
        print(f"Transcript: {results['metrics'].get('transcript', 'N/A')}")
        print(f"WPM: {results['metrics'].get('wpm', 0):.1f}")

    print()
    print("Test complete!")


if __name__ == "__main__":
    test_speech_module()
