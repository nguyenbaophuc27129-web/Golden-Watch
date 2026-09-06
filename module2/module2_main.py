# -*- coding: utf-8 -*-
"""
MODULE 2 MAIN - SPEECH ANALYSIS (PSCS v8.0)
Phân tích giọng nói phát hiện đột quỵ (Dysarthria)

Usage:
    venv/Scripts/python.exe module2/module2_main.py --mode quick

Author: PSCS Team
Date: 28/08/2026
"""

import sys
import os
import time
import numpy as np

# Dam bao in tieng Viet dung tren Windows console (cp1252)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Get parent directory (fga_project/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

from detection.speech_module_v2 import SpeechAnalysisModule

# Configuration
VOSK_MODEL = os.path.join(parent_dir, "models/vosk-model-vn-0.4")
ML_MODEL = os.path.join(parent_dir, "models/speech_torgo_20260828_211130.pth")
SCALER = os.path.join(parent_dir, "models/speech_torgo_20260828_211130_scaler.pkl")
BASELINE = os.path.join(parent_dir, "data/baselines/user_default.json")

# Color codes for terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(title):
    """In header đẹp"""
    print()
    print("="*70)
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(70)}{Colors.ENDC}")
    print("="*70)
    print()


def print_status(status, score, nihss):
    """In status với màu"""
    if status == "NORMAL":
        color = Colors.GREEN
        symbol = "✓"
    elif status == "WARNING":
        color = Colors.YELLOW
        symbol = "!"
    else:
        color = Colors.RED
        symbol = "⚠"

    print(f"{color}{Colors.BOLD}Status: {status}{Colors.ENDC}")
    print(f"{color}Speech Probability: {score:.1f}%{Colors.ENDC}")
    print(f"{color}NIHSS Item 10 (Dysarthria): {nihss}/4{Colors.ENDC}")
    print(symbol + "-"*65)


def print_metrics(metrics):
    """In metrics chi tiết"""
    print(f"{Colors.BOLD}DETAILED METRICS:{Colors.ENDC}")
    print()

    # Speech rate
    wpm = metrics.get('wpm', 0)
    print("Speech Rate:")
    print(f"WPM: {wpm:.1f} words/min (Normal: 100-180)")
    if wpm < 100:
        print(f"{Colors.YELLOW}[SLOW]{Colors.ENDC} Speech rate below normal")
    elif wpm > 180:
        print(f"{Colors.YELLOW}[FAST]{Colors.ENDC} Speech rate above normal")
    else:
        print(f"{Colors.GREEN}[OK]{Colors.ENDC} Normal speech rate")
    print()

    # Voice quality
    jitter = metrics.get('jitter', 0)
    shimmer = metrics.get('shimmer', 0)
    print("Voice Quality:")
    print(f"Jitter: {jitter:.2f}% (Threshold: <3%)")
    print(f"Shimmer: {shimmer:.2f}% (Threshold: <6%)")
    if jitter > 3 or shimmer > 6:
        print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} Voice quality issues detected")
    else:
        print(f"{Colors.GREEN}[OK]{Colors.ENDC} Normal voice quality")
    print()

    # Pitch
    pitch_mean = metrics.get('pitch_mean', 0)
    pitch_std = metrics.get('pitch_std', 0)
    print("Pitch Analysis:")
    print(f"Mean Pitch: {pitch_mean:.1f} Hz")
    print(f"Pitch Std Dev: {pitch_std:.1f} Hz")
    if pitch_std > 50:
        print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} High pitch variability")
    else:
        print(f"{Colors.GREEN}[OK]{Colors.ENDC} Normal pitch variability")
    print()

    # Transcription
    transcript = metrics.get('transcript', '')
    word_count = metrics.get('word_count', 0)
    print("Transcription:")
    print(f"Words detected: {word_count}")
    print(f"Text: {transcript if transcript else '(No speech detected)'}")
    print()


def create_module():
    """Khởi tạo SpeechAnalysisModule với Vosk + ML model + baseline cá nhân"""
    m = SpeechAnalysisModule(
        vosk_model_path=VOSK_MODEL,
        ml_model_path=ML_MODEL,
        scaler_path=SCALER
    )
    m.load_baseline(BASELINE)  # Tự động load nếu đã từng calibration
    return m


def interactive_test():
    """Test tương tác với người dùng"""
    print_header("MODULE 2: SPEECH ANALYSIS")

    # Initialize module
    speech = create_module()

    if speech.vosk_model is None:
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot load Vosk model")
        print(f"Model path: {VOSK_MODEL}")
        return

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} Vosk model loaded successfully")
    if speech.ml_model is not None:
        print(f"{Colors.GREEN}[OK]{Colors.ENDC} ML model loaded successfully")
    else:
        print(f"{Colors.YELLOW}[WARN]{Colors.ENDC} ML model not loaded - using rule-based fallback")
    print()

    print(f"{Colors.BOLD}INSTRUCTIONS:{Colors.ENDC}")
    print("Module 2 sẽ phân tích giọng nói của bạn để phát hiện đột quỵ.")
    print("Bạn sẽ được yêu cầu đọc hoặc nói các câu cụ thể.")
    print()
    print(f"{Colors.CYAN}Press ENTER to continue...{Colors.ENDC}")
    input()

    # Test cases
    test_cases = [
        {
            'name': 'Test 1: Normal Counting',
            'instruction': 'Đọc to các số từ 1 đến 10',
            'duration': 5,
            'expected': 'NORMAL'
        },
        {
            'name': 'Test 2: Normal Speech',
            'instruction': 'Mô tả hoạt động hôm nay của bạn (vòng 30 giây)',
            'duration': 5,
            'expected': 'NORMAL'
        },
        {
            'name': 'Test 3: Rapid Speech',
            'instruction': 'Đọc nhanh một đoạn văn bất kỳ',
            'duration': 5,
            'expected': 'NORMAL hoặc WARNING'
        },
        {
            'name': 'Test 4: Slow Speech',
            'instruction': 'Đọc chậm các số từ 1 đến 10 (mô phỏng)',
            'duration': 5,
            'expected': 'WARNING hoặc DANGER'
        }
    ]

    all_results = []

    for idx, test in enumerate(test_cases):
        print_header(test['name'])

        print(f"{Colors.BOLD}Instruction:{Colors.ENDC} {test['instruction']}")
        print(f"{Colors.CYAN}Duration: {test['duration']} seconds{Colors.ENDC}")
        print()
        print(f"{Colors.YELLOW}Press ENTER when ready to record...{Colors.ENDC}")
        input()

        print(f"{Colors.CYAN}Recording...{Colors.ENDC}")

        audio = speech.record_audio(duration_seconds=test['duration'])

        if audio is None:
            print(f"{Colors.RED}[ERROR]{Colors.ENDC} Recording failed")
            continue

        print(f"{Colors.GREEN}[OK]{Colors.ENDC} Recording complete")
        print()
        print(f"{Colors.CYAN}Analyzing...{Colors.ENDC}")

        results = speech.predict_dysarthria(audio, test['duration'])

        all_results.append({
            'name': test['name'],
            'results': results
        })

        print()
        print_status(results['status'], results['speech_prob'], results['nihss_score'])
        print()

        print_metrics(results['metrics'])

        print()
        print(f"{Colors.CYAN}Press ENTER for next test...{Colors.ENDC}")
        input()

    # Final summary
    print_header("FINAL SUMMARY")

    print(f"{Colors.BOLD}Test Results Summary:{Colors.ENDC}")
    print()

    normal_count = sum(1 for r in all_results if r['results']['status'] == 'NORMAL')
    warning_count = sum(1 for r in all_results if r['results']['status'] == 'WARNING')
    danger_count = sum(1 for r in all_results if r['results']['status'] == 'DANGER')

    print(f"Total tests: {len(all_results)}")
    print(f"{Colors.GREEN}NORMAL:{Colors.ENDC} {normal_count}")
    print(f"{Colors.YELLOW}WARNING:{Colors.ENDC} {warning_count}")
    print(f"{Colors.RED}DANGER:{Colors.ENDC} {danger_count}")
    print()

    if len(all_results) > 0:
        avg_score = np.mean([r['results']['speech_prob'] for r in all_results])
        print(f"Average Speech Probability: {avg_score:.1f}%")
        print()

    if danger_count > 0:
        print(f"{Colors.RED}[NOTE]{Colors.ENDC} Some tests showed DANGER level.")
        print("This could indicate dysarthria or speech abnormality.")
    elif warning_count > 0:
        print(f"{Colors.YELLOW}[NOTE]{Colors.ENDC} Some tests showed WARNING level.")
        print("Consider retesting or consulting a specialist.")
    else:
        print(f"{Colors.GREEN}[OK]{Colors.ENDC} All tests within NORMAL range.")

    print()
    print("="*70)


def calibrate():
    """Calibration 30 giây - tạo baseline cá nhân (WPM + pitch)"""
    print_header("MODULE 2: CALIBRATION (30 giây)")

    print(f"{Colors.BOLD}Mục đích:{Colors.ENDC} Ghi nhận giọng nói bình thường của bạn")
    print("để phát hiện bất thường chính xác hơn (so với chính mình).")
    print()
    print("Instruction: Nói tự nhiên về hoạt động hôm nay của bạn")
    print(f"{Colors.CYAN}Duration: 30 seconds{Colors.ENDC}")
    print()
    print(f"{Colors.YELLOW}Press ENTER when ready to record...{Colors.ENDC}")
    input()

    speech = create_module()
    baseline = speech.calibrate_baseline(duration_seconds=30, save_path=BASELINE)

    if baseline:
        print()
        print(f"{Colors.GREEN}[OK]{Colors.ENDC} Baseline đã lưu: {BASELINE}")
        print(f"  WPM: {baseline['wpm']:.1f} | Pitch: {baseline['pitch_mean']:.1f} Hz")
    print()
    print("="*70)


def quick_test():
    """Test nhanh một lần"""
    print_header("MODULE 2 QUICK TEST")

    speech = create_module()

    if speech.vosk_model is None:
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot load Vosk model")
        return

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} Vosk model loaded")
    if speech.ml_model is not None:
        print(f"{Colors.GREEN}[OK]{Colors.ENDC} ML model loaded")
    else:
        print(f"{Colors.YELLOW}[WARN]{Colors.ENDC} ML model not loaded - using rule-based fallback")
    print()
    print("Instruction: Đọc to các số từ 1 đến 10")
    print()
    print(f"{Colors.YELLOW}Press ENTER when ready...{Colors.ENDC}")
    input()

    audio = speech.record_audio(duration_seconds=5)
    if audio is None:
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Recording failed")
        return

    results = speech.predict_dysarthria(audio, 5)

    print()
    print_status(results['status'], results['speech_prob'], results['nihss_score'])
    print()
    print_metrics(results['metrics'])

    print()
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Module 2: Speech Analysis')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'quick', 'calibrate'],
                        help='interactive (full test) | quick (single test) | calibrate (30s baseline)')

    args = parser.parse_args()

    try:
        if args.mode == 'interactive':
            interactive_test()
        elif args.mode == 'quick':
            quick_test()
        elif args.mode == 'calibrate':
            calibrate()
    except KeyboardInterrupt:
        print()
        print(f"{Colors.YELLOW}[INTERRUPTED]{Colors.ENDC} Test stopped by user")
        print()
    except Exception as e:
        print()
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} {e}")
        print()
