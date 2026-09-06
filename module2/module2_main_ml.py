# -*- coding: utf-8 -*-
"""
MODULE 2 MAIN - SPEECH ANALYSIS (PSCS v8.0 WITH ML)
Phân tích giọng nói phát hiện đột quỵ (Dysarthria) với ML Model

Usage:
    py -3.11 module2/module2_main_ml.py

Author: PSCS Team
Date: 28/08/2026
"""

import sys
import os
import time
import numpy as np

# Get parent directory (fga_project/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

from detection.speech_module_v2 import SpeechAnalysisModule

# Configuration
VOSK_MODEL = os.path.join(parent_dir, "models/vosk-model-vn-0.4")
# TORGO model (trained with 17,633 real samples, 83% accuracy)
ML_MODEL = os.path.join(parent_dir, "models/speech_torgo_20260828_211130.pth")
SCALER = os.path.join(parent_dir, "models/speech_torgo_20260828_211130_scaler.pkl")

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
        symbol = "[OK]"
    elif status == "WARNING":
        color = Colors.YELLOW
        symbol = "[WARNING]"
    else:
        color = Colors.RED
        symbol = "[DANGER]"

    print(f"{color}{symbol} Status: {status}{Colors.ENDC}")
    print(f"{color}    Dysarthria Probability: {score:.1f}%{Colors.ENDC}")
    print(f"{color}    NIHSS Item 10 (Dysarthria): {nihss}/4{Colors.ENDC}")


def print_metrics(metrics):
    """In metrics chi tiết"""
    print()
    print(f"{Colors.BOLD}DETAILED METRICS:{Colors.ENDC}")

    # Transcription
    transcript = metrics.get('transcript', '')
    word_count = metrics.get('word_count', 0)
    print(f"\n  Transcription:")
    print(f"    Words detected: {word_count}")
    if transcript:
        print(f"    Text: {transcript}")
    else:
        print(f"    Text: (No speech detected)")

    # Speech rate
    wpm = metrics.get('wpm', 0)
    print(f"\n  Speech Rate:")
    print(f"    WPM: {wpm:.1f} words/min (Normal: 100-180)")
    if wpm < 100:
        print(f"    {Colors.YELLOW}[SLOW]{Colors.ENDC} Speech rate below normal")
    elif wpm > 180:
        print(f"    {Colors.YELLOW}[FAST]{Colors.ENDC} Speech rate above normal")
    else:
        print(f"    {Colors.GREEN}[OK]{Colors.ENDC} Normal speech rate")

    # Voice quality
    pitch_mean = metrics.get('pitch_mean', 0)
    pitch_std = metrics.get('pitch_std', 0)
    print(f"\n  Voice Quality:")
    print(f"    Mean Pitch: {pitch_mean:.1f} Hz")
    print(f"    Pitch Std Dev: {pitch_std:.1f} Hz")
    if pitch_std > 50:
        print(f"    {Colors.YELLOW}[NOTE]{Colors.ENDC} Higher pitch variability")


def interactive_test():
    """Test tương tác với người dùng"""
    print_header("MODULE 2: SPEECH ANALYSIS (WITH ML MODEL)")

    # Initialize module
    speech = SpeechAnalysisModule(
        vosk_model_path=VOSK_MODEL,
        ml_model_path=ML_MODEL,
        scaler_path=SCALER
    )

    if speech.ml_model is None:
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot load ML model")
        print(f"Model path: {ML_MODEL}")
        return

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} ML Model loaded successfully")
    print(f"{Colors.GREEN}[OK]{Colors.ENDC} Vosk model loaded successfully")
    print()

    print(f"{Colors.BOLD}INSTRUCTIONS:{Colors.ENDC}")
    print("  Module 2 sẽ phân tích giọng nói của bạn để phát hiện đột quỵ.")
    print("  Bạn sẽ được yêu cầu đọc hoặc nói các câu cụ thể.")
    print()
    print(f"{Colors.CYAN}Press ENTER to continue...{Colors.ENDC}")
    input()

    # Test cases
    test_cases = [
        {
            'name': 'Test 1: Normal Counting',
            'instruction': 'Doc to cac so tu 1 den 10',
            'duration': 5
        },
        {
            'name': 'Test 2: Normal Speech',
            'instruction': 'Mo ta hoat dong hom nay cua ban',
            'duration': 5
        },
        {
            'name': 'Test 3: Slow Speech (Mimic)',
            'instruction': 'Doc cham cac so tu 1 den 10 (mo phong)',
            'duration': 5
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
        print(f"{Colors.CYAN}Analyzing with ML model...{Colors.ENDC}")

        results = speech.predict_dysarthria(audio, test['duration'])

        all_results.append({
            'name': test['name'],
            'results': results
        })

        print()
        print_status(results['status'], results['speech_prob'], results['nihss_score'])
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

    print(f"  Total tests: {len(all_results)}")
    print(f"  {Colors.GREEN}NORMAL:{Colors.ENDC} {normal_count}")
    print(f"  {Colors.YELLOW}WARNING:{Colors.ENDC} {warning_count}")
    print(f"  {Colors.RED}DANGER:{Colors.ENDC} {danger_count}")
    print()

    if len(all_results) > 0:
        avg_score = np.mean([r['results']['speech_prob'] for r in all_results])
        print(f"  Average Dysarthria Probability: {avg_score:.1f}%")
        print()

        if danger_count > 0:
            print(f"{Colors.RED}[NOTE]{Colors.ENDC} Some tests showed DANGER level.")
        elif warning_count > 0:
            print(f"{Colors.YELLOW}[NOTE]{Colors.ENDC} Some tests showed WARNING level.")
        else:
            print(f"{Colors.GREEN}[OK]{Colors.ENDC} All tests within NORMAL range.")

    print()
    print("="*70)


def quick_test():
    """Test nhanh một lần"""
    print_header("MODULE 2 QUICK TEST (WITH ML)")

    speech = SpeechAnalysisModule(
        vosk_model_path=VOSK_MODEL,
        ml_model_path=ML_MODEL,
        scaler_path=SCALER
    )

    if speech.ml_model is None:
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot load ML model")
        return

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} ML Model loaded")
    print()
    print("Instruction: Doc to cac so tu 1 den 10")
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
    print_metrics(results['metrics'])

    print()
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Module 2: Speech Analysis with ML')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'quick'],
                        help='Test mode: interactive (full test) or quick (single test)')

    args = parser.parse_args()

    try:
        if args.mode == 'interactive':
            interactive_test()
        elif args.mode == 'quick':
            quick_test()
    except KeyboardInterrupt:
        print()
        print(f"{Colors.YELLOW}[INTERRUPTED]{Colors.ENDC} Test stopped by user")
        print()
    except Exception as e:
        print()
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} {e}")
        import traceback
        traceback.print_exc()
        print()
