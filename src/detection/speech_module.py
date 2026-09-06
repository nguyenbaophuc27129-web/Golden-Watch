"""
MODULE 2: SPEECH ANALYSIS - PSCS v8.0
Phân tích giọng nói để phát hiện đột quỵ (Dysarthria)

Tác giả: PSCS Team
Ngày: 28/08/2026
"""

import os
import sys
import numpy as np
import librosa
from vosk import Model, KaldiRecognizer
import json
import wave
import threading
import queue
from collections import deque

# Optional dependencies
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[SPEECH] PyAudio not installed. Install: pip install pyaudio - speech_module.py:26")
    print("[SPEECH] Audio recording will be unavailable - speech_module.py:27")


class SpeechAnalysisModule:
    """Module phân tích giọng nói phát hiện đột quỵ"""

    def __init__(self, model_path=None, sample_rate=16000):
        """
        Khởi tạo Speech Analysis Module

        Args:
            model_path: Đường dẫn đến Vosk model (Tiếng Việt)
            sample_rate: Tần số mẫu (16kHz cho Vosk)
        """
        self.sample_rate = sample_rate
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self.is_recording = False
        self.audio_queue = queue.Queue()

        # Baseline metrics (cần calibrate cho từng người)
        self.baseline_wpm = 150  # Words per minute (người bình thường)
        self.baseline_pitch = 150  # Hz (trung bình)

        # Thresholds (từ bảng cơ sở khoa học)
        self.thresholds = {
            'jitter_max': 3.0,      # % - UA-Speech dataset, 2021
            'shimmer_max': 6.0,     # % - Dysarthria study, 2023
            'wpm_min': 100,         # - TORGO dataset
            'wpm_max': 180,         # - TORGO dataset
            'pitch_var_max': 50,    # Hz - Biến thiên cao
            'mfcc_dist_max': 15.0   # - Khoảng cách MFCC
        }

        # History để tính toán
        self.word_history = deque(maxlen=100)  # Lưu 100 từ gần nhất
        self.pitch_history = deque(maxlen=50)  # Lưu 50 pitch values

        # Load model nếu có path
        if model_path and os.path.exists(model_path):
            self._load_vosk_model()

    def _load_vosk_model(self):
        """Load Vosk model cho Speech-to-Text"""
        try:
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            print(f"[SPEECH] Vosk model loaded from: {self.model_path} - speech_module.py:76")
            return True
        except Exception as e:
            print(f"[SPEECH] Error loading Vosk model: {e} - speech_module.py:79")
            return False

    def record_audio(self, duration_seconds=5):
        """
        Thu âm thanh từ microphone

        Args:
            duration_seconds: Thời gian ghi âm (giây)

        Returns:
            audio_data: Mảng numpy chứa audio
        """
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio is not installed. Install with: pip install pyaudio")

        try:
            p = pyaudio.PyAudio()

            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )

            print(f"[SPEECH] Recording for {duration_seconds} seconds... - speech_module.py:106")
            frames = []

            for _ in range(int(self.sample_rate / 1024 * duration_seconds)):
                data = stream.read(1024)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            return audio_float

        except Exception as e:
            print(f"[SPEECH] Error recording audio: {e} - speech_module.py:124")
            return None

    def detect_voice_activity(self, audio, threshold=0.02):
        """
        Voice Activity Detection (VAD) - phát hiện có người nói không

        Args:
            audio: Mảng numpy audio
            threshold: Ngưỡng RMS energy

        Returns:
            is_speech: True nếu phát hiện tiếng nói
            speech_ratio: Tỷ lệ frames có tiếng nói
        """
        try:
            # Chia nhỏ thành frames 20ms
            frame_length = int(self.sample_rate * 0.02)
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length)

            # Tính RMS energy cho mỗi frame
            rms = np.sqrt(np.mean(frames ** 2, axis=0))

            # VAD: frames có energy > threshold
            speech_frames = np.sum(rms > threshold)
            speech_ratio = speech_frames / len(rms)

            return speech_ratio > 0.3, speech_ratio

        except Exception as e:
            print(f"[SPEECH] VAD error: {e} - speech_module.py:154")
            return False, 0.0

    def extract_mfcc_features(self, audio):
        """
        Trích xuất MFCC (Mel-Frequency Cepstral Coefficients)

        Args:
            audio: Mảng numpy audio

        Returns:
            mfcc_mean: MFCC trung bình
            mfcc_delta: Delta MFCC (biến đổi)
        """
        try:
            # Trích xuất 13 MFCC coefficients
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)

            # Mean và delta
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_delta = np.mean(np.diff(mfcc, axis=1), axis=1)

            return mfcc_mean, mfcc_delta

        except Exception as e:
            print(f"[SPEECH] MFCC error: {e} - speech_module.py:179")
            return np.zeros(13), np.zeros(13)

    def calculate_pitch(self, audio):
        """
        Tính pitch (cao độ giọng)

        Args:
            audio: Mảng numpy audio

        Returns:
            pitch_mean: Pitch trung bình (Hz)
            pitch_std: Độ lệch chuẩn pitch
        """
        try:
            # Sử dụng PYIN algorithm để extract pitch
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)

            # Lấy pitch values có magnitude > threshold
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if len(pitch_values) > 0:
                pitch_mean = np.mean(pitch_values)
                pitch_std = np.std(pitch_values)
                return pitch_mean, pitch_std
            else:
                return 0.0, 0.0

        except Exception as e:
            print(f"[SPEECH] Pitch calculation error: {e} - speech_module.py:213")
            return 0.0, 0.0

    def calculate_jitter_shimmer(self, audio):
        """
        Tính Jitter và Shimmer (chỉ số chất lượng giọng)

        Args:
            audio: Mảng numpy audio

        Returns:
            jitter: Jitter (%) - biến đổi pitch
            shimmer: Shimmer (%) - biến đổi amplitude
        """
        try:
            # Tính pitch first
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)

            # Extract pitch sequence
            pitch_sequence = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_sequence.append(pitch)

            if len(pitch_sequence) < 2:
                return 0.0, 0.0

            # Jitter: biến đổi pitch giữa các kỳ liên tiếp
            pitch_diffs = np.abs(np.diff(pitch_sequence))
            jitter = np.mean(pitch_diffs) / np.mean(pitch_sequence) * 100 if np.mean(pitch_sequence) > 0 else 0.0

            # Shimmer: biến đổi amplitude (sử dụng energy)
            frame_length = int(self.sample_rate * 0.01)  # 10ms frames
            energy = []
            for i in range(0, len(audio) - frame_length, frame_length):
                frame = audio[i:i + frame_length]
                energy.append(np.sum(frame ** 2))

            if len(energy) < 2:
                shimmer = 0.0
            else:
                energy_diffs = np.abs(np.diff(energy))
                shimmer = np.mean(energy_diffs) / np.mean(energy) * 100 if np.mean(energy) > 0 else 0.0

            return jitter, shimmer

        except Exception as e:
            print(f"[SPEECH] Jitter/Shimmer error: {e} - speech_module.py:262")
            return 0.0, 0.0

    def transcribe_speech(self, audio):
        """
        Speech-to-Text sử dụng Vosk

        Args:
            audio: Mảng numpy audio

        Returns:
            text: Văn bản được transcribe
            word_count: Số từ được nhận diện
        """
        if self.model is None or self.recognizer is None:
            print("[SPEECH] Vosk model not loaded - speech_module.py:277")
            return "", 0

        try:
            # Convert audio to int16 for Vosk
            audio_int16 = (audio * 32768).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

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
            print(f"[SPEECH] Transcription error: {e} - speech_module.py:298")
            return "", 0

    def calculate_wpm(self, word_count, duration_seconds):
        """
        Tính WPM (Words Per Minute)

        Args:
            word_count: Số từ
            duration_seconds: Thời gian (giây)

        Returns:
            wpm: Words per minute
        """
        if duration_seconds <= 0:
            return 0.0

        wpm = (word_count / duration_seconds) * 60
        return wpm

    def analyze_speech(self, audio, duration_seconds):
        """
        Phân tích giọng nói hoàn chỉnh

        Args:
            audio: Mảng numpy audio
            duration_seconds: Thời gian thu âm (giây)

        Returns:
            results: Dictionary chứa tất cả metrics
        """
        results = {
            'status': 'NO_SPEECH',
            'speech_prob': 0.0,
            'nihss_score': 0,
            'metrics': {}
        }

        try:
            # 1. Voice Activity Detection
            has_speech, speech_ratio = self.detect_voice_activity(audio)

            if not has_speech:
                print("[SPEECH] No speech detected - speech_module.py:341")
                return results

            results['metrics']['speech_ratio'] = speech_ratio

            # 2. Transcribe speech
            text, word_count = self.transcribe_speech(audio)
            results['metrics']['transcript'] = text
            results['metrics']['word_count'] = word_count

            # 3. Calculate WPM
            wpm = self.calculate_wpm(word_count, duration_seconds)
            results['metrics']['wpm'] = wpm

            # 4. Extract MFCC
            mfcc_mean, mfcc_delta = self.extract_mfcc_features(audio)
            results['metrics']['mfcc_mean'] = mfcc_mean.tolist()
            results['metrics']['mfcc_delta'] = mfcc_delta.tolist()

            # 5. Calculate Pitch
            pitch_mean, pitch_std = self.calculate_pitch(audio)
            results['metrics']['pitch_mean'] = pitch_mean
            results['metrics']['pitch_std'] = pitch_std

            # 6. Calculate Jitter & Shimmer
            jitter, shimmer = self.calculate_jitter_shimmer(audio)
            results['metrics']['jitter'] = jitter
            results['metrics']['shimmer'] = shimmer

            # 7. Calculate Speech Score (0-100)
            speech_score = self._calculate_speech_score(
                wpm, jitter, shimmer, pitch_std, mfcc_delta
            )
            results['speech_prob'] = speech_score

            # 8. Determine status
            if speech_score < 30:
                results['status'] = 'NORMAL'
            elif speech_score < 60:
                results['status'] = 'WARNING'
            else:
                results['status'] = 'DANGER'

            # 9. Map to NIHSS Item 10 (Dysarthria)
            results['nihss_score'] = self._map_to_nihss(speech_score)

            print(f"[SPEECH] Score: {speech_score:.1f}%, Status: {results['status']}, NIHSS: {results['nihss_score']} - speech_module.py:387")
            return results

        except Exception as e:
            print(f"[SPEECH] Analysis error: {e} - speech_module.py:391")
            return results

    def _calculate_speech_score(self, wpm, jitter, shimmer, pitch_std, mfcc_delta):
        """
        Tính Speech Score (0-100) dựa trên các metrics

        Args:
            wpm: Words per minute
            jitter: Jitter (%)
            shimmer: Shimmer (%)
            pitch_std: Pitch standard deviation
            mfcc_delta: MFCC delta

        Returns:
            score: Speech probability (0-100)
        """
        score = 0.0

        # 1. WPM deviation (30 points)
        wpm_deviation = abs(wpm - self.baseline_wpm)
        if wpm_deviation > 50:
            score += 30
        elif wpm_deviation > 30:
            score += 20
        elif wpm_deviation > 10:
            score += 10

        # 2. Jitter (25 points)
        if jitter > self.thresholds['jitter_max'] * 2:
            score += 25
        elif jitter > self.thresholds['jitter_max']:
            score += 15
        elif jitter > self.thresholds['jitter_max'] * 0.5:
            score += 5

        # 3. Shimmer (25 points)
        if shimmer > self.thresholds['shimmer_max'] * 2:
            score += 25
        elif shimmer > self.thresholds['shimmer_max']:
            score += 15
        elif shimmer > self.thresholds['shimmer_max'] * 0.5:
            score += 5

        # 4. Pitch variability (10 points)
        if pitch_std > self.thresholds['pitch_var_max']:
            score += 10
        elif pitch_std > self.thresholds['pitch_var_max'] * 0.5:
            score += 5

        # 5. MFCC deviation (10 points)
        mfcc_dist = np.linalg.norm(mfcc_delta)
        if mfcc_dist > self.thresholds['mfcc_dist_max']:
            score += 10
        elif mfcc_dist > self.thresholds['mfcc_dist_max'] * 0.5:
            score += 5

        return min(score, 100.0)

    def _map_to_nihss(self, speech_score):
        """
        Mapping speech score sang NIHSS Item 10 (Dysarthria)

        Args:
            speech_score: Speech score (0-100)

        Returns:
            nihss_score: NIHSS score (0-4)
        """
        if speech_score < 30:
            return 0  # No articulatory disturbance
        elif speech_score < 50:
            return 1  # Mild to moderate dysarthria
        elif speech_score < 70:
            return 2  # Severe dysarthria
        else:
            return 3  # Severe dysarthria or mute

    def calibrate_baseline(self, audio_samples):
        """
        Calibrate baseline cho từng người (optional)

        Args:
            audio_samples: List of audio samples from normal speech
        """
        wpm_list = []
        pitch_list = []

        for audio in audio_samples:
            # Calculate WPM
            text, word_count = self.transcribe_speech(audio)
            duration = len(audio) / self.sample_rate
            wpm = self.calculate_wpm(word_count, duration)
            wpm_list.append(wpm)

            # Calculate pitch
            pitch_mean, _ = self.calculate_pitch(audio)
            if pitch_mean > 0:
                pitch_list.append(pitch_mean)

        if wpm_list:
            self.baseline_wpm = np.mean(wpm_list)
        if pitch_list:
            self.baseline_pitch = np.mean(pitch_list)

        print(f"[SPEECH] Baseline calibrated: WPM={self.baseline_wpm:.1f}, Pitch={self.baseline_pitch:.1f} Hz - speech_module.py:496")


# Test function
def test_speech_module():
    """Test Speech Module với microphone"""
    # Get parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))
    model_path = os.path.join(project_dir, "models/vosk-model-vn-0.4")

    # Initialize module
    speech = SpeechAnalysisModule(model_path=model_path)

    if speech.model is None:
        print("[ERROR] Cannot load Vosk model - speech_module.py:511")
        return

    print("= - speech_module.py:514"*70)
    print("MODULE 2 SPEECH TEST - speech_module.py:515")
    print("= - speech_module.py:516"*70)
    print()
    print("Instructions: - speech_module.py:518")
    print("Speak clearly in Vietnamese - speech_module.py:519")
    print("Count from 1 to 10 - speech_module.py:520")
    print("Describe what you did today - speech_module.py:521")
    print()

    # Test 1: Normal speech
    print("Test 1: Normal speech - speech_module.py:525")
    print("Press ENTER to start recording (5 seconds)... - speech_module.py:526")
    input()

    audio = speech.record_audio(duration_seconds=5)
    if audio is not None:
        results = speech.analyze_speech(audio, 5)
        print(f"Results: {results['status']}, Score: {results['speech_prob']:.1f}%, NIHSS: {results['nihss_score']} - speech_module.py:532")

    print()
    print("Test complete! - speech_module.py:535")


if __name__ == "__main__":
    test_speech_module()
