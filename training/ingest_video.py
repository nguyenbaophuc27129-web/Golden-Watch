# -*- coding: utf-8 -*-
"""NK-30: Trích khung hình + audio 16kHz từ video tự thu (bộ kit 3 ngày).

Dùng cho:
- Video NIHSS tổng hợp  -> khung jpg (chấm tay 2 rater) + wav (module nói)
- Video YouTube tải về  -> khung jpg + wav để chạy pipeline

Cách chạy:
    python training/ingest_video.py video.mp4
    python training/ingest_video.py video.mp4 --out data/thu_tu_lieu/V01 --frames 20
    python training/ingest_video.py thu_muc_video/ --frames 15
    python training/ingest_video.py video.mp4 --no-wav

Đầu ra mỗi video X.mp4 (trong --out, mặc định trùng thư mục video):
    X_f01.jpg ... X_fNN.jpg   (khung hình chia đều theo thời gian)
    X.wav                     (mono 16kHz — đúng chuẩn predict_dysarthria)
Cuối cùng in ra bảng gợi ý dán vào CSV_DANH_MUC_DU_LIEU.csv.
"""
import argparse
import os
import subprocess
import sys

VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv')


def find_videos(path):
    """Trả về list đường dẫn video từ 1 file hoặc 1 thư mục."""
    if os.path.isfile(path):
        return [path]
    vids = []
    for name in sorted(os.listdir(path)):
        if name.lower().endswith(VIDEO_EXT):
            vids.append(os.path.join(path, name))
    return vids


def extract_frames(video_path, out_dir, stem, n_frames):
    """Trích n khung chia đều theo thời gian -> jpg. Trả về list file."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[LOI] Khong mo duoc video: {video_path}')
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        print(f'[LOI] Khong dem duoc so khung: {video_path}')
        return []
    idxs = sorted({int(round(total * (i + 0.5) / n_frames))
                   for i in range(n_frames)})
    files = []
    for k, fi in enumerate(idxs, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        fp = os.path.join(out_dir, f'{stem}_f{k:02d}.jpg')
        cv2.imwrite(fp, frame)
        files.append(fp)
    cap.release()
    dur = total / fps
    return files, dur


def extract_wav(video_path, out_dir, stem):
    """Trích audio mono 16kHz bằng ffmpeg của imageio_ffmpeg (đã có sẵn)."""
    from imageio_ffmpeg import get_ffmpeg_exe
    ffmpeg = get_ffmpeg_exe()
    out = os.path.join(out_dir, f'{stem}.wav')
    cmd = [ffmpeg, '-y', '-i', video_path, '-vn', '-ac', '1', '-ar', '16000',
           '-loglevel', 'error', out]
    try:
        subprocess.run(cmd, check=True)
        return out
    except subprocess.CalledProcessError as e:
        print(f'[CANH BAO] Khong trich duoc audio ({video_path}): {e}')
        return None


def main():
    ap = argparse.ArgumentParser(description='Trích khung + audio 16kHz')
    ap.add_argument('input', help='file video hoặc thư mục chứa video')
    ap.add_argument('--out', default=None, help='thư mục xuất (mặc định: cùng chỗ video)')
    ap.add_argument('--frames', type=int, default=15, help='số khung/video (mặc định 15)')
    ap.add_argument('--no-wav', action='store_true', help='không trích audio')
    args = ap.parse_args()

    vids = find_videos(args.input)
    if not vids:
        print(f'[LOI] Khong tim thay video nao trong: {args.input}')
        sys.exit(1)

    rows = []
    for vp in vids:
        stem = os.path.splitext(os.path.basename(vp))[0]
        out_dir = args.out or os.path.dirname(os.path.abspath(vp))
        os.makedirs(out_dir, exist_ok=True)

        res = extract_frames(vp, out_dir, stem, args.frames)
        if not res:
            continue
        files, dur = res
        wav = None if args.no_wav else extract_wav(vp, out_dir, stem)
        print(f'[OK] {os.path.basename(vp)}: {len(files)} khung, '
              f'{dur:.1f}s, wav={"co" if wav else "khong"}')
        loai = 'video_nihss' if 'nihss' in stem.lower() else 'video'
        rows.append(f'{loai},{os.path.basename(vp)},,video,,'
                    f'{dur:.0f},,trich {len(files)} khung + wav')

    print('\n---- Gợi ý dán vào CSV_DANH_MUC_DU_LIEU.csv ----')
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
