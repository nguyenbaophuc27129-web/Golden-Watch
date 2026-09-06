# -*- coding: utf-8 -*-
"""
MODULE 5 MAIN - VISUAL FIELD / EYE MOVEMENT DETECTION (PSCS v8.0)
Phát hiện khiếm thị trường và bất thường vận chuyển mắt do đột quỵ

Usage:
    py -3.11 module5/module5_main.py

Author: PSCS Team
Date: 29/08/2026
"""

import sys
import os
import cv2
import time

# Get parent directory (fga_project/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

from detection.visual_module import VisualFieldDetector

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


def test_with_webcam(duration=15):
    """Test Module 5 với webcam real-time"""
    print_header("MODULE 5: VISUAL FIELD / EYE MOVEMENT DETECTION")

    # Initialize detector
    detector = VisualFieldDetector()

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} Visual Field Detector initialized")
    print()

    print(f"{Colors.BOLD}INSTRUCTIONS:{Colors.ENDC}")
    print("  Module 5 will detect visual field deficits and eye movement abnormalities.")
    print("  Please:")
    print("  1. Sit in front of camera, face clearly visible")
    print("  2. Look straight at the camera")
    print("  3. Follow the on-screen directions:")
    print("     - Look LEFT (3 seconds)")
    print("     - Look RIGHT (3 seconds)")
    print("     - Look UP (3 seconds)")
    print("     - Look DOWN (3 seconds)")
    print("     - Look CENTER (3 seconds)")
    print()
    print(f"{Colors.CYAN}Press ENTER to start...{Colors.ENDC}")
    try:
        input()
    except EOFError:
        pass  # Continue if no input available

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot open camera")
        return

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} Camera opened")
    print()
    print(f"{Colors.BOLD}Starting visual field test...{Colors.ENDC}")
    print(f"{Colors.CYAN}Press 'q' to quit early{Colors.ENDC}")
    print()

    # Test directions
    directions = ['Look LEFT', 'Look RIGHT', 'Look UP', 'Look DOWN', 'Look CENTER']
    direction_duration = 3  # seconds per direction
    current_dir_idx = 0
    dir_start_time = time.time()

    all_results = []
    direction_scores = {d: [] for d in directions}

    start_time = time.time()
    test_duration = duration

    while time.time() - start_time < test_duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect visual field abnormality
        result = detector.detect_visual_field_abnormality(frame_rgb)

        if result['status'] != 'NO_FACE':
            all_results.append(result)

            # Store score for current direction
            current_direction = directions[current_dir_idx]
            direction_scores[current_direction].append(result['visual_prob'])

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

            # Background box
            cv2.rectangle(frame_display, (10, 10), (500, 180), (0, 0, 0), -1)
            cv2.rectangle(frame_display, (10, 10), (500, 180), color, 2)

            # Module title
            cv2.putText(frame_display, f"Module 5: Visual Field", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Current direction
            cv2.putText(frame_display, f"Direction: {current_direction}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Status
            cv2.putText(frame_display, f"Status: {result['status']}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Visual prob
            cv2.putText(frame_display, f"Visual Prob: {result['visual_prob']:.1f}%", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # NIHSS scores
            cv2.putText(frame_display, f"NIHSS Gaze: {result['nihss_gaze_score']}/2", (20, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame_display, f"NIHSS Visual: {result['nihss_visual_score']}/3", (20, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Show metrics
            y_offset = 195
            for key, value in result['metrics'].items():
                if y_offset < 280:
                    if isinstance(value, float):
                        text = f"{key}: {value:.2f}"
                    else:
                        text = f"{key}: {value}"
                    cv2.putText(frame_display, text, (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                    y_offset += 18

            cv2.imshow('Module 5 - Visual Field Detection', frame_display)

        # Switch direction every 3 seconds
        if time.time() - dir_start_time > direction_duration:
            current_dir_idx = (current_dir_idx + 1) % len(directions)
            dir_start_time = time.time()
            print(f"  {Colors.CYAN}->{Colors.ENDC} {directions[current_dir_idx]}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print()
    print_header("TEST SUMMARY")

    if len(all_results) > 0:
        avg_prob = sum(r['visual_prob'] for r in all_results) / len(all_results)
        final_status = all_results[-1]['status']
        final_nihss_gaze = all_results[-1]['nihss_gaze_score']
        final_nihss_visual = all_results[-1]['nihss_visual_score']

        print(f"{Colors.BOLD}Total frames analyzed:{Colors.ENDC} {len(all_results)}")
        print(f"{Colors.BOLD}Average Visual Prob:{Colors.ENDC} {avg_prob:.1f}%")
        print(f"{Colors.BOLD}Final Status:{Colors.ENDC} {final_status}")
        print()

        # Direction-wise summary
        print(f"{Colors.BOLD}Direction-wise Results:{Colors.ENDC}")
        for direction, scores in direction_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                status_color = Colors.GREEN if avg_score < 30 else Colors.YELLOW if avg_score < 60 else Colors.RED
                print(f"  {direction}: {status_color}{avg_score:.1f}%{Colors.ENDC} ({len(scores)} frames)")
        print()

        # NIHSS scores
        print(f"{Colors.BOLD}NIHSS Scores:{Colors.ENDC}")
        print(f"  Item 4 (Best Gaze): {final_nihss_gaze}/2")
        print(f"  Item 5 (Visual Fields): {final_nihss_visual}/3")
        print()

        if final_status == 'NORMAL':
            print(f"{Colors.GREEN}[OK]{Colors.ENDC} No visual field abnormality detected")
        elif final_status == 'WARNING':
            print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} Possible visual field deficit")
        else:
            print(f"{Colors.RED}[DANGER]{Colors.ENDC} Visual field abnormality detected")
    else:
        print(f"{Colors.YELLOW}[NOTE]{Colors.ENDC} No face detected in frames")

    print()
    print("="*70)


def quick_test():
    """Test nhanh một frame"""
    print_header("MODULE 5 QUICK TEST")

    detector = VisualFieldDetector()

    print(f"{Colors.GREEN}[OK]{Colors.ENDC} Visual Field Detector initialized")
    print()
    print("Instruction: Look straight at the camera")
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
        result = detector.detect_visual_field_abnormality(frame_rgb)

        print()
        print(f"Status: {result['status']}")
        print(f"Visual Field Prob: {result['visual_prob']:.1f}%")
        print(f"NIHSS Item 4 (Best Gaze): {result['nihss_gaze_score']}/2")
        print(f"NIHSS Item 5 (Visual Fields): {result['nihss_visual_score']}/3")
        print()

        print("Metrics:")
        for key, value in result['metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()
    else:
        print(f"{Colors.RED}[ERROR]{Colors.ENDC} Cannot capture frame")

    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Module 5: Visual Field Detection')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'quick'],
                        help='Test mode: full (15 seconds) or quick (one frame)')
    parser.add_argument('--duration', type=int, default=15,
                        help='Test duration in seconds')

    args = parser.parse_args()

    try:
        if args.mode == 'full':
            test_with_webcam(args.duration)
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
