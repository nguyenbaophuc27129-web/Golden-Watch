# -*- coding: utf-8 -*-
"""
TẠO BẢNG CƠ SỞ KHOA HỌC (SYS-06) — BANG_CO_SO_KHOA_HOC.xlsx
5 sheets theo lịch Ngày 2. Trạng thái nguồn được đánh dấu TRUNG THỰC:
  VERIFIED    : đã có DOI/PMID trong THANG_DO_CHUAN_QUOC_TE.md (verify trước 06/09)
  CANONICAL   : trích dẫn kinh điển, đã đối chiếu kiến thức, KHÔNG verify
                online được hôm nay (hạn mức search) → cần check PubMed trước
                khi in poster
  SELF-DESIGN : ngưỡng tự thiết kế (sau khi ROC optimize trên dữ liệu thật)
                → điểm cần validate lâm sàng, ghi rõ trong báo cáo

Chạy: venv/Scripts/python.exe scripts/generate_science_table.py
"""

import os
import sys
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_ROOT, 'BANG_CO_SO_KHOA_HOC.xlsx')

HEADER_FILL = PatternFill('solid', fgColor='1F4E79')
STATUS_FILL = {'VERIFIED': 'C6EFCE', 'CANONICAL': 'FFEB9C',
               'SELF-DESIGN': 'FFC7CE'}
HDR_FONT = Font(color='FFFFFF', bold=True)


def sheet(wb, name, headers, rows, widths):
    ws = wb.create_sheet(name)
    for c, h in enumerate(headers, 1):
        cell = ws.cell(1, c, h)
        cell.fill, cell.font = HEADER_FILL, HDR_FONT
        cell.alignment = Alignment(horizontal='center')
    for r, row in enumerate(rows, 2):
        for c, v in enumerate(row, 1):
            cell = ws.cell(r, c, v)
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            if c == len(headers):
                st = str(v).split()[0]
                if st in STATUS_FILL:
                    cell.fill = PatternFill('solid', fgColor=STATUS_FILL[st])
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    ws.freeze_panes = 'A2'


def main():
    wb = Workbook()
    wb.remove(wb.active)

    H = ['Thang đo / Ngưỡng', 'Giá trị hệ thống dùng', 'Nguồn (tác giả, năm, tạp chí)',
         'DOI / PMID', 'Trạng thái']
    sheet(wb, '1_NIHSS_FAST', H, [
        ['NIHSS tổng thể (15 items, 0-42)', 'Khung chấm điểm 4 items',
         'Brott T. et al., 1989, Stroke', 'PMID 2757041', 'VERIFIED'],
        ['NIHSS Item 4 — Facial Palsy', '0-3 điểm', 'Brott 1989 (NIHSS gốc)',
         'PMID 2757041', 'VERIFIED'],
        ['NIHSS Item 5 — Motor Arm', '0-4 điểm', 'Brott 1989 (NIHSS gốc)',
         'PMID 2757041', 'VERIFIED'],
        ['NIHSS Item 6 — Motor Leg', '0-4 điểm', 'Brott 1989 (NIHSS gốc)',
         'PMID 2757041', 'VERIFIED'],
        ['NIHSS Item 10 — Dysarthria', '0-2 điểm (max 2, không phải 3)',
         'Brott 1989 (NIHSS gốc)', 'PMID 2757041', 'VERIFIED'],
        ['FAST protocol (Face-Arm-Speech-Time)', '4 bước sàng lọc',
         'Kothari RU. et al., 1999, Acad Emerg Med', 'PMID 10449154', 'VERIFIED'],
        ['FAST độ nhạy/độ đặc hiệu', 'Sens 85-95%, Spec 90-95%',
         'Harbison J. et al., 2006 (xác nhận trong dự án)', 'cần verify PMID',
         'CANONICAL'],
        ['Time is Brain — 1.9 triệu neuron/phút',
         'Thông điệp thời gian vàng < 4.5h',
         'Saver JL., 2006, Stroke 37(2):563-564', 'Stroke.2006;37:563-564',
         'CANONICAL'],
        ['Cửa sổ thrombolysis (tPA)', '4.5 giờ từ khởi phát',
         'Hacke W. et al., 2008, NEJM (ECASS III)', 'NEJM 2008;359:1317-29',
         'CANONICAL'],
    ], [34, 30, 42, 24, 13])

    sheet(wb, '2_Face_M1', H, [
        ['Mouth asymmetry (nụ cười lệch)', '25% ngưỡng báo động',
         'Tự thiết kế — chưa có nguồn trực tiếp', '—', 'SELF-DESIGN'],
        ['Eye deviation / eyelid droop', '30° ngưỡng',
         'Tự thiết kế — chưa có nguồn trực tiếp', '—', 'SELF-DESIGN'],
        ['Face tilt', '10° ngưỡng',
         'Tự thiết kế — chưa có nguồn trực tiếp', '—', 'SELF-DESIGN'],
        ['House-Brackmann grading (tham chiếu lâm sàng)',
         'Thang chấm liệt mặt 5 độ', 'House JW, Brackmann DE., 1985, '
         'Otolaryngol Head Neck Surg', 'PMID 3821900 (cần verify)', 'CANONICAL'],
        ['Model ML M1', '93.75% (n=2783 frames tự thu)',
         'Dữ liệu nội bộ — validate lâm sàng đang chờ', '—', 'SELF-DESIGN'],
    ], [34, 30, 42, 24, 13])

    sheet(wb, '3_Speech_M2', H, [
        ['Jitter (độ rung cơ thanh quản)', '<3% bình thường, >5% khó nói',
         'Maryn Y. et al., 1996', 'PMID 8784714', 'VERIFIED'],
        ['Shimmer (biên độ rung)', '<6% bình thường, >10% khó nói',
         'Ramig LA., 1988', 'PMID 3195366', 'VERIFIED'],
        ['WPM — nói chậm/nói khó', 'So baseline cá nhân; lệch >40% bất thường',
         'Baseline cá nhân (Layer 1 Defense) — tự thiết kế', '—',
         'SELF-DESIGN'],
        ['TORGO dataset (dysarthria, tiếng Anh)',
         'Train ML M2: 48→256→128→64→2, 83.07%',
         'Rudzicz F. et al., 2012, TORGO', 'doi:10.1017/S1351324912000036 '
         '(cần verify)', 'CANONICAL'],
        ['VAD-guided window + median consensus (M2-10)',
         'TPR 96.3%, FPR 0% @th56 (Youden J=1.00, n=55)',
         'Phương pháp nội bộ 06/09 trên TORGO', '—', 'SELF-DESIGN'],
    ], [34, 30, 42, 24, 13])

    sheet(wb, '4_Arm_Gait_Radar', H, [
        ['Arm drop (rơi tay)', '100 px / ~5cm trong 10s',
         'Tự thiết kế — đo pixel có giới hạn khoảng cách', '—', 'SELF-DESIGN'],
        ['Stride length', '0.6-0.8m bình thường, <0.5m bất thường',
         'Hausdorff JM., 2005, J Gerontol A', 'doi:10.1093/gerona/60.4.476',
         'VERIFIED'],
        ['Cadence', '100-130 bước/phút', 'Menz HB. et al., 2003',
         'PMID 12948460', 'VERIFIED'],
        ['Stride variability', '<0.05s bình thường, >0.07s bất thường',
         'Hausdorff JM., 2007', 'PMC1853168', 'VERIFIED'],
        ['Step width', '0.1-0.15m bình thường, >0.2m đột quỵ',
         'Kong PW., 2010, J Neurol Sci', 'doi:10.1016/j.jns.2010.09.007',
         'VERIFIED'],
        ['Gait symmetry index', '<15% (chưa có nguồn)',
         'Tự thiết kế — 1/4 threshold M4 chưa có nguồn', '—', 'SELF-DESIGN'],
        ['Gait velocity phân loại sau đột quỵ',
         '0.4/0.8 m/s ranh giới (nhà/cộng đồng)',
         'Perry J. et al., 1995, Stroke', 'Stroke.1995;26:982-989 (cần verify)',
         'CANONICAL'],
        ['Radar ngã: biến động vị trí', '>1m trong <2s',
         'Tự thiết kế (LD2450 không có z-axis → proxy)', '—', 'SELF-DESIGN'],
        ['Radar ngã: bất hoạt', '>45s sau biến động',
         'Tự thiết kế — cần đối chiếu mmWave fall detection', '—',
         'SELF-DESIGN'],
    ], [34, 30, 42, 24, 13])

    sheet(wb, '5_Metrics_ML', H, [
        ['TPR/TPF (độ nhạy)', '>90% mục tiêu', 'Chuẩn ML y tế',
         'Powers D., 2011 (F1/P/R)', 'VERIFIED'],
        ['FPR (tỉ lệ báo nhầm)', '<5% mục tiêu',
         'Chuẩn hệ thống cảnh báo y tế', 'Fawcett T., 2006 (ROC)',
         'VERIFIED'],
        ['ROC / Youden J (chọn ngưỡng)', 'J = TPR - FPR tối đa',
         'Youden WJ., 1950; Fawcett 2006', 'Fawcett 2006 (đã verify trước)',
         'VERIFIED'],
        ['Youden J áp thực tế M2', 'th=56%: TPR 96.3% / FPR 0% (n=55)',
         'Nội bộ 06/09', '—', 'SELF-DESIGN'],
        ['NIHSS CI (Monte Carlo ± noise)', 'total ± 1.96·SD',
         'Phương pháp bootstrap/MC chuẩn thống kê', 'Efron 1979 (cần verify)',
         'CANONICAL'],
    ], [34, 30, 42, 24, 13])

    wb.save(OUT)
    n_rows = sum(wb[s].max_row - 1 for s in wb.sheetnames)
    print(f"Saved: {OUT}")
    print(f"Sheets: {wb.sheetnames}")
    print(f"Tổng dòng: {n_rows}")


if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
