# -*- coding: utf-8 -*-
"""
MODULE 3 MAIN - ARM WEAKNESS DETECTION (PSCS v8.0)
Phát hiện yếu tay do đột quỵ sử dụng YOLOv8n-Pose

Usage:
    py -3.11 module3/module3_main.py

Author: PSCS Team
Date: 29/08/2026
"""

import sys
import os

# Get parent directory (fga_project/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

from detection.arm_module import ArmWeaknessDetector
import cv2
import time

# Configuration
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


def test_with_webcam():
    """Test Module 3 với webcam real-time"""
    print_header("MODULE 3: ARM WEAKNESS DETECTION")

    # Initialize detector
    detector = ArmWeaknessDetector(model_path=YOLO_MODEL)

    if detector.model is None:
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot load YOLO model")
        return

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} YOLOv8n-pose loaded")
    print()

    print(f"{Colors.BOLD}INSTRUCTIONS:{Colors.ENDC}")
    print("  Module 3 sẽ phát hiện yếu tay (Arm Weakness) từ camera.")
    print("  Hãy:")
    print("  1. Ngồi trước camera, toàn thân hiển thị")
    print("  2. Giơ cả 2 tay lên trước ngực")
    print("  3. Giữ nguyên trong 5 giây")
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
    print(f"{Colors.BOLD}Testing for 10 seconds...{Colors.ENDC}")
    print(f"{Colors.CYAN}Press 'q' to quit early{Colors.ENDC}")
    print()

    # Test for 10 seconds
    start_time = time.time()
    frame_count = 0
    all_results = []

    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect arm weakness
        result = detector.detect_arm_weakness(frame_rgb)

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

            cv2.rectangle(frame_display, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.rectangle(frame_display, (10, 10), (400, 100), color, 2)
            cv2.putText(frame_display, f"Module 3: Arm Weakness", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_display, f"Status: {result['status']}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame_display, f"Arm Prob: {result['arm_prob']:.1f}%", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Module 3 - Arm Weakness Detection', frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print()
    print_header("TEST SUMMARY")

    if len(all_results) > 0:
        avg_prob = sum(r['arm_prob'] for r in all_results) / len(all_results)
        final_status = all_results[-1]['status']
        final_nihss = all_results[-1]['nihss_score']

        print(f"{Colors.BOLD}Frames analyzed:{Colors.ENDC} {len(all_results)}")
        print(f"{Colors.BOLD}Average Arm Prob:{Colors.ENDC} {avg_prob:.1f}%")
        print(f"{Colors.BOLD}Final Status:{Colors.ENDC} {final_status}")
        print(f"{Colors.BOLD}NIHSS Item 5:{Colors.ENDC} {final_nihss}/4")
        print()

        if final_status == 'NORMAL':
            print(f"{Colors.GREEN}[OK]{Colors.ENDC} No arm weakness detected")
        elif final_status == 'WARNING':
            print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} Possible arm weakness")
        else:
            print(f"{Colors.RED}[DANGER]{Colors.ENDC} Arm weakness detected")
    else:
        print(f"{Colors.YELLOW}[NOTE]{Colors.ENDC} No person detected in frames")

    print()
    print("="*70)


def quick_test():
    """Test nhanh một frame"""
    print_header("MODULE 3 QUICK TEST")

    detector = ArmWeaknessDetector(model_path=YOLO_MODEL)

    if detector.model is None:
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot load YOLO model")
        return

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} YOLOv8n-pose loaded")
    print()
    print("Instruction: Giơ cả 2 tay lên trước ngực")
    print()
    print(f"{Colors.CYAN}Press ENTER to capture one frame...{Colors.ENDC}")
    input()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot open camera")
        return

    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect_arm_weakness(frame_rgb)

        print()
        print(f"Status: {result['status']}")
        print(f"Arm Weakness Prob: {result['arm_prob']:.1f}%")
        print(f"NIHSS Item 5 (Motor Arm): {result['nihss_score']}/4")
        print()

        if 'weak_arm' in result['metrics']:
            weak_arm = result['metrics']['weak_arm']
            print(f"Weak Arm: {weak_arm.upper()}")
        print()
    else:
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot capture frame")

    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Module 3: Arm Weakness Detection')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'quick'],
                        help='Test mode: full (10 seconds) or quick (one frame)')

    args = parser.parse_args()

    try:
        if args.mode == 'full':
            test_with_webcam()
        elif args.mode == 'quick':
            quick_test()
    except KeyboardInterrupt:
        print()
        print(f"{Colors.YELLOW}[INTERRUPTED]{Colors.ENDC} Test stopped by user")
    except Exception as e:
        print()
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} {e}")
        import traceback
        traceback.print_exc()
