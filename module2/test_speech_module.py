# -*- coding: utf-8 -*-
"""
TEST SPEECH MODULE - Module 2
Test Speech Analysis Module với microphone real-time
"""

import sys
import os

# Get parent directory (fga_project/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

from detection.speech_module import SpeechAnalysisModule
import time

# Configuration
VOSK_MODEL = os.path.join(parent_dir, "models/vosk-model-vn-0.4")


def test_speech_basic():
    """Test cơ bản Speech Module"""
    print("="*70)
    print("MODULE 2 SPEECH TEST - BASIC")
    print("="*70)
    print()

    # Initialize
    speech = SpeechAnalysisModule(model_path=VOSK_MODEL)

    if speech.model is None:
        print("[ERROR] Cannot load Vosk model")
        print(f"Model path: {VOSK_MODEL}")
        return

    print("[OK] Vosk model loaded successfully")
    print()
    print("Instructions:")
    print("  Test 1: Đọc to các số từ 1 đến 10")
    print("  Test 2: Mô tả hoạt động hôm nay của bạn")
    print("  Test 3: Mô phỏng giọng líu lưỡi (nếu có thể)")
    print()

    # Test 1: Normal counting
    print("="*70)
    print("TEST 1: Normal counting (1-10)")
    print("="*70)
    print("Press ENTER when ready to record (5 seconds)...")
    input()

    audio = speech.record_audio(duration_seconds=5)
    if audio is not None:
        results = speech.analyze_speech(audio, 5)
        print(f"\nResults:")
        print(f"  Status: {results['status']}")
        print(f"  Speech Prob: {results['speech_prob']:.1f}%")
        print(f"  NIHSS Score: {results['nihss_score']}")
        print(f"  WPM: {results['metrics'].get('wpm', 0):.1f}")
        print(f"  Jitter: {results['metrics'].get('jitter', 0):.2f}%")
        print(f"  Shimmer: {results['metrics'].get('shimmer', 0):.2f}%")
        print(f"  Pitch: {results['metrics'].get('pitch_mean', 0):.1f} Hz")
        print(f"  Transcript: {results['metrics'].get('transcript', 'N/A')}")

    print()
    time.sleep(1)

    # Test 2: Describe today
    print("="*70)
    print("TEST 2: Describe your day")
    print("="*70)
    print("Press ENTER when ready to record (5 seconds)...")
    input()

    audio = speech.record_audio(duration_seconds=5)
    if audio is not None:
        results = speech.analyze_speech(audio, 5)
        print(f"\nResults:")
        print(f"  Status: {results['status']}")
        print(f"  Speech Prob: {results['speech_prob']:.1f}%")
        print(f"  NIHSS Score: {results['nihss_score']}")
        print(f"  WPM: {results['metrics'].get('wpm', 0):.1f}")
        print(f"  Transcript: {results['metrics'].get('transcript', 'N/A')}")

    print()
    print("="*70)
    print("TEST COMPLETE")
    print("="*70)


def test_speech_indicators():
    """Test các indicators chi tiết"""
    print("="*70)
    print("MODULE 2 SPEECH TEST - INDICATORS")
    print("="*70)
    print()

    speech = SpeechAnalysisModule(model_path=VOSK_MODEL)

    if speech.model is None:
        print("[ERROR] Cannot load Vosk model")
        return

    print("Test: Recording 5 seconds of speech...")
    print("Press ENTER when ready...")
    input()

    audio = speech.record_audio(duration_seconds=5)
    if audio is None:
        print("[ERROR] Recording failed")
        return

    print("\n" + "="*70)
    print("DETAILED METRICS")
    print("="*70)

    results = speech.analyze_speech(audio, 5)

    print(f"\n1. TRANSCRIPTION:")
    print(f"   Text: {results['metrics'].get('transcript', 'N/A')}")
    print(f"   Word Count: {results['metrics'].get('word_count', 0)}")

    print(f"\n2. SPEECH RATE:")
    wpm = results['metrics'].get('wpm', 0)
    print(f"   WPM: {wpm:.1f} words/min")
    print(f"   Normal range: 100-180 WPM")
    if wpm < 100:
        print(f"   [WARNING] Speech too slow")
    elif wpm > 180:
        print(f"   [WARNING] Speech too fast")
    else:
        print(f"   [OK] Normal speech rate")

    print(f"\n3. VOICE QUALITY:")
    jitter = results['metrics'].get('jitter', 0)
    shimmer = results['metrics'].get('shimmer', 0)
    print(f"   Jitter: {jitter:.2f}% (max: 3%)")
    print(f"   Shimmer: {shimmer:.2f}% (max: 6%)")
    if jitter > 3:
        print(f"   [WARNING] High jitter (pitch instability)")
    if shimmer > 6:
        print(f"   [WARNING] High shimmer (amplitude instability)")

    print(f"\n4. PITCH ANALYSIS:")
    pitch_mean = results['metrics'].get('pitch_mean', 0)
    pitch_std = results['metrics'].get('pitch_std', 0)
    print(f"   Mean Pitch: {pitch_mean:.1f} Hz")
    print(f"   Pitch Std: {pitch_std:.1f} Hz")
    if pitch_std > 50:
        print(f"   [WARNING] High pitch variability")

    print(f"\n5. MFCC FEATURES:")
    mfcc = results['metrics'].get('mfcc_mean', [])
    if len(mfcc) > 0:
        print(f"   MFCC shape: {len(mfcc)} coefficients")
        print(f"   First 5 coeffs: {mfcc[:5]}")

    print(f"\n6. FINAL SCORE:")
    print(f"   Status: {results['status']}")
    print(f"   Speech Probability: {results['speech_prob']:.1f}%")
    print(f"   NIHSS Item 10 (Dysarthria): {results['nihss_score']}/4")

    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Speech Module')
    parser.add_argument('--mode', type=str, default='basic',
                        choices=['basic', 'indicators'],
                        help='Test mode: basic or indicators')

    args = parser.parse_args()

    if args.mode == 'basic':
        test_speech_basic()
    elif args.mode == 'indicators':
        test_speech_indicators()
