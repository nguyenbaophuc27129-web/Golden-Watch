# -*- coding: utf-8 -*-
"""
MODULE 4 MAIN - GAIT ABNORMALITY DETECTION (PSCS v8.0)
Phát hiện bất thường dáng đi do đột quỵ sử dụng time series analysis

Usage:
    py -3.11 module4/module4_main.py --mode file --file "path/to/gait/file.txt"
    py -3.11 module4/module4_main.py --mode webcam

Author: PSCS Team
Date: 29/08/2026
"""

import sys
import os
import argparse
import numpy as np
import cv2
import time

# Get parent directory (fga_project/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

from detection.gait_module import GaitAbnormalityDetector, GaitPoseDetector

# Configuration
ML_MODEL = os.path.join(parent_dir, "models/gait_classifier_20260829_120925.pth")
SCALER = os.path.join(parent_dir, "models/gait_classifier_20260829_120925_scaler.pkl")
YOLO_MODEL = os.path.join(parent_dir, "models/yolov8n-pose.pt")

# Color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


def print_header(title):
    print()
    print("="*70)
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(70)}{Colors.ENDC}")
    print("="*70)
    print()


def test_with_file(file_path):
    """Test Module 4 với gait time series file"""
    print_header("MODULE 4: GAIT ABNORMALITY DETECTION (FILE MODE)")

    if not os.path.exists(file_path):
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} File not found: {file_path}")
        return

    # Initialize detector with ML model
    detector = GaitAbnormalityDetector(
        model_path=ML_MODEL,
        scaler_path=SCALER
    )

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} ML model loaded")
    print()

    # Load gait data
    print(f"{Colors.BOLD}Loading file:{Colors.ENDC} {file_path}")
    data = detector.load_gait_data(file_path)

    if data is None:
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot load file")
        return

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} Data loaded: {len(data)} samples")
    print()

    # Detect gait abnormality
    result = detector.detect_gait_abnormality(data)

    # Display results
    print_header("DETECTION RESULTS")

    print(f"{Colors.BOLD}Status:{Colors.ENDC} {result['status']}")
    print(f"{Colors.BOLD}Gait Abnormality Prob:{Colors.ENDC} {result['gait_prob']:.2f}%")
    print(f"{Colors.BOLD}NIHSS Item 6 (Motor Leg):{Colors.ENDC} {result['nihss_score']}/4")
    print()

    if result['metrics']:
        print(f"{Colors.BOLD}Gait Metrics:{Colors.ENDC}")
        for key, value in result['metrics'].items():
            print(f"  {key}: {value:.4f}")
        print()

    # Status interpretation
    if result['status'] == 'NORMAL':
        print(f"{Colors.GREEN}[OK]{Colors.ENDC} Normal gait pattern detected")
    elif result['status'] == 'WARNING':
        print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} Possible gait abnormality")
    else:
        print(f"{Colors.RED}[DANGER]{Colors.ENDC} Gait abnormality detected")

    print()
    print("="*70)


def test_with_webcam(duration=10):
    """Test Module 4 với webcam real-time"""
    print_header("MODULE 4: GAIT ABNORMALITY DETECTION (WEBCAM MODE)")

    # Initialize pose detector
    detector = GaitPoseDetector(model_path=YOLO_MODEL)

    if detector.model is None:
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot load YOLO model")
        return

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} YOLOv8n-pose loaded")
    print()

    print(f"{Colors.BOLD}INSTRUCTIONS:{Colors.ENDC}")
    print("  Module 4 sẽ phát hiện bất thường dáng đi từ camera.")
    print("  Hãy:")
    print("  1. Đứng trước camera, toàn thân hiển thị")
    print("  2. Đi bộ tại chỗ hoặc đi ngang qua camera")
    print("  3. Thực hiện trong 10 giây")
    print()
    print(f"{Colors.CYAN}Press ENTER to start...{Colors.ENDC}")
    input()

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot open camera")
        return

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} Camera opened")
    print()
    print(f"{Colors.BOLD}Testing for {duration} seconds...{Colors.ENDC}")
    print(f"{Colors.CYAN}Press 'q' to quit early{Colors.ENDC}")
    print()

    # Test for specified duration
    start_time = time.time()
    frame_count = 0
    all_results = []

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect gait abnormality
        result = detector.detect_gait_from_frame(frame_rgb)

        if result['status'] != 'NO_PERSON':
            all_results.append(result)

            # Display on frame
            h, w = frame.shape[:2]
            frame_display = cv2.resize(frame, (1280, int(1280 * h / w)))

            # Draw status box
            if result['status'] == 'NORMAL':
                color = (0, 255, 0)
            elif result['status'] == 'WARNING':
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame_display, (10, 10), (450, 120), (0, 0, 0), -1)
            cv2.rectangle(frame_display, (10, 10), (450, 120), color, 2)
            cv2.putText(frame_display, f"Module 4: Gait Analysis", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_display, f"Status: {result['status']}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame_display, f"Gait Prob: {result['gait_prob']:.1f}%", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_display, f"NIHSS Leg: {result['nihss_score']}/4", (20, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show metrics if available
            y_offset = 145
            for key, value in result['metrics'].items():
                if y_offset < 200:
                    cv2.putText(frame_display, f"{key}: {value:.2f}", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y_offset += 18

            cv2.imshow('Module 4 - Gait Abnormality Detection', frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print()
    print_header("TEST SUMMARY")

    if len(all_results) > 0:
        avg_prob = sum(r['gait_prob'] for r in all_results) / len(all_results)
        final_status = all_results[-1]['status']
        final_nihss = all_results[-1]['nihss_score']

        print(f"{Colors.BOLD}Frames analyzed:{Colors.ENDC} {len(all_results)}")
        print(f"{Colors.BOLD}Average Gait Prob:{Colors.ENDC} {avg_prob:.1f}%")
        print(f"{Colors.BOLD}Final Status:{Colors.ENDC} {final_status}")
        print(f"{Colors.BOLD}NIHSS Item 6 (Motor Leg):{Colors.ENDC} {final_nihss}/4")
        print()

        if final_status == 'NORMAL':
            print(f"{Colors.GREEN}[OK]{Colors.ENDC} No gait abnormality detected")
        elif final_status == 'WARNING':
            print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} Possible gait abnormality")
        else:
            print(f"{Colors.RED}[DANGER]{Colors.ENDC} Gait abnormality detected")
    else:
        print(f"{Colors.YELLOW}[NOTE]{Colors.ENDC} No person detected in frames")

    print()
    print("="*70)


def test_with_dataset_samples():
    """Test with actual dataset samples"""
    print_header("MODULE 4: TEST WITH DATASET SAMPLES")

    detector = GaitAbnormalityDetector(
        model_path=ML_MODEL,
        scaler_path=SCALER
    )

    if detector.model is None:
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot load ML model")
        return

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} ML model loaded")
    print()

    # Dataset path
    dataset_path = os.path.join(
        parent_dir,
        "data/datasets/gait/gait-in-aging-and-disease-database-1.0.0/gait-in-aging-and-disease-database-1.0.0"
    )

    if not os.path.exists(dataset_path):
        print(f"{Colors.YELLOW}[NOTE]{Colors.ENDC} Dataset not found")
        return

    # Test files
    test_files = [
        ("o1-76-si.txt", "Normal (Old)"),
        ("y1-23-si.txt", "Normal (Young)"),
        ("pd1-si.txt", "Abnormal (Parkinson's)")
    ]

    print(f"{Colors.BOLD}Testing {len(test_files)} samples...{Colors.ENDC}")
    print()

    for filename, label in test_files:
        file_path = os.path.join(dataset_path, filename)

        if not os.path.exists(file_path):
            continue

        data = detector.load_gait_data(file_path)
        if data is None:
            continue

        result = detector.detect_gait_abnormality(data)

        status_color = Colors.GREEN if result['status'] == 'NORMAL' else Colors.RED

        print(f"{Colors.BOLD}File:{Colors.ENDC} {filename}")
        print(f"  Expected: {label}")
        print(f"  Detected: {status_color}{result['status']}{Colors.ENDC}")
        print(f"  Gait Prob: {result['gait_prob']:.2f}%")
        print(f"  NIHSS: {result['nihss_score']}/4")
        print()

    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Module 4: Gait Abnormality Detection')
    parser.add_argument('--mode', type=str, default='samples',
                        choices=['file', 'webcam', 'samples'],
                        help='Test mode: file (single file), webcam (real-time), samples (dataset samples)')
    parser.add_argument('--file', type=str, help='Path to gait data file (.txt)')
    parser.add_argument('--duration', type=int, default=10, help='Webcam test duration (seconds)')

    args = parser.parse_args()

    try:
        if args.mode == 'file':
            if not args.file:
                print(f"{Colors.RED}[ERROR]{Colors.ENDC} Please specify --file")
            else:
                test_with_file(args.file)
        elif args.mode == 'webcam':
            test_with_webcam(args.duration)
        elif args.mode == 'samples':
            test_with_dataset_samples()
    except KeyboardInterrupt:
        print()
        print(f"{Colors.YELLOW}[INTERRUPTED]{Colors.ENDC} Test stopped by user")
    except Exception as e:
        print()
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} {e}")
        import traceback
        traceback.print_exc()
