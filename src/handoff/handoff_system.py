# -*- coding: utf-8 -*-
"""
HANDOFF SYSTEM - PSCS v8.0 (Golden-Watch)
Gói bàn giao y khoa: TIMELINE + REPORT + QR + PDF.
Mục tiêu (TONG_QUAN v6.0): rút handoff time 15 phút → 5 phút.

3 THÀNH PHẦN:
  1. Timeline : T0 bật hệ thống → T1 dấu hiệu đầu → T2 tiến triển →
                T3 cảnh báo → T4 chuyển viện (thời gian vàng)
  2. Report   : JSON đầy đủ (bệnh nhân + NIHSS + triage + timeline + alerts)
  3. Export   : QR code (nội dung tóm tắt cho nhân viên y tế quét nhanh)
                + PDF (reportlab, font Arial để có dấu tiếng Việt)

Tác giả: PSCS Team
Ngày: 06/09/2026
"""

import os
import sys
import json
import time
import base64


class HandoffSystem:
    """Timeline + report + QR/PDF export cho một ca giám sát."""

    def __init__(self, patient_info=None, output_dir=None):
        """
        Args:
            patient_info: {'name', 'age', 'gender', 'phone', 'address', ...}
            output_dir: nơi lưu báo cáo (mặc định exports/handoff/)
        """
        self.patient_info = patient_info or {}
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        self.output_dir = output_dir or os.path.join(
            project_root, 'exports', 'handoff')
        os.makedirs(self.output_dir, exist_ok=True)

        self.t0 = time.time()
        self.events = []          # timeline
        self.add_event('T0', 'He thong bat dau giam sat')

    # ------------------------------------------------------------------
    # 1. TIMELINE
    # ------------------------------------------------------------------
    def add_event(self, tag, description, data=None):
        """Thêm mốc T0-T4 (hoặc tag tuỳ ý) vào timeline."""
        ev = {
            'tag': tag,
            'description': description,
            'elapsed_min': round((time.time() - self.t0) / 60, 1),
            'timestamp': time.strftime('%H:%M:%S'),
            'data': data or {},
        }
        self.events.append(ev)
        return ev

    # ------------------------------------------------------------------
    # 2. REPORT
    # ------------------------------------------------------------------
    def build_report(self, fusion_result, nihss_result, triage_result=None,
                     alerts=None, defense_stats=None):
        """Ghép mọi dữ liệu của ca thành 1 report dict (JSON-serializable)."""
        return {
            'meta': {
                'product': 'PSCS Golden-Watch v8.0',
                'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'handoff_min': round((time.time() - self.t0) / 60, 1),
                'disclaimer': ('He thong HO TRO phat hien som, khong phan '
                               'chan doan. Goi 115 ngay khi co dau hieu.'),
            },
            'patient': self.patient_info,
            'fusion': fusion_result or {},
            'nihss': nihss_result or {},
            'triage': triage_result or {},
            'alerts': alerts or [],
            'defense': defense_stats or {},
            'timeline': self.events,
        }

    def save_report(self, report):
        """Lưu JSON, trả về đường dẫn."""
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.output_dir, f'handoff_{ts}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path

    # ------------------------------------------------------------------
    # 3a. QR CODE — tóm tắt cho nhân viên y tế quét nhanh
    # ------------------------------------------------------------------
    def generate_qr(self, report):
        """QR chứa: bệnh nhân + NIHSS + subtype + thời gian (tóm tắt, không
        phải full data — QR phải quét được nhanh). Trả về (png_path, data)."""
        import qrcode

        nihss = (report.get('nihss') or {})
        tri = (report.get('triage') or {})
        sub = (tri.get('subtype') or {})
        hosp = tri.get('hospital') or {}
        p = report.get('patient') or {}
        lines = [
            f"PSCS HANDOFF {time.strftime('%d/%m %H:%M')}",
            f"BN: {p.get('name','?')} | {p.get('age','?')}T "
            f"{p.get('gender','?')}",
            f"NIHSS(uoc): {nihss.get('total','?')}±{nihss.get('margin',0)}/13",
            f"Nghi: {sub.get('subtype','?').upper()} "
            f"({sub.get('confidence','?')}%)",
            f"Muc do: {tri.get('severity','?')}",
            f"BV de xuat: {hosp.get('name','GOI 115')}",
            f"LKE: {', '.join(str(e['tag'])+' '+str(e['elapsed_min'])+'p' for e in report.get('timeline', []))}",
        ]
        data = '\n'.join(lines)

        qr = qrcode.QRCode(box_size=8, border=2)
        qr.add_data(data)
        qr.make(fit=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.output_dir, f'handoff_qr_{ts}.png')
        qr.make_image(fill_color='black', back_color='white').save(path)
        return path, data

    # ------------------------------------------------------------------
    # 3b. PDF — báo cáo đầy đủ (font Arial: dấu tiếng Việt)
    # ------------------------------------------------------------------
    def generate_pdf(self, report):
        """Xuất PDF A4. Trả về đường dẫn."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas

        ts = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.output_dir, f'handoff_{ts}.pdf')
        c = canvas.Canvas(path, pagesize=A4)
        w, h = A4

        # Font tiếng Việt: Arial + Arial Bold Windows (fallback Helvetica)
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            pdfmetrics.registerFont(TTFont('VN', 'C:/Windows/Fonts/arial.ttf'))
            pdfmetrics.registerFont(TTFont(
                'VN-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
            font = 'VN'
        except Exception:
            font = 'Helvetica'

        def text(x, y, s, size=10, bold=False):
            c.setFont(font + ('-Bold' if bold and font == 'VN' else ''), size)
            c.drawString(x, y, s)

        y = h - 1.5 * cm
        text(2 * cm, y, 'PSCS GOLDEN-WATCH — BÁO CÁO BÀN GIAO Y KHOA',
             14, bold=True)
        y -= 0.6 * cm
        text(2 * cm, y, report['meta'].get('disclaimer', ''), 8)

        y -= 1.0 * cm
        p = report.get('patient') or {}
        text(2 * cm, y, 'BỆNH NHÂN', 11, bold=True); y -= 0.55 * cm
        for k, label in (('name', 'Họ tên'), ('age', 'Tuổi'),
                         ('gender', 'Giới tính'), ('phone', 'Điện thoại'),
                         ('address', 'Địa chỉ')):
            if p.get(k):
                text(2 * cm, y, f'{label}: {p[k]}', 10); y -= 0.5 * cm

        y -= 0.5 * cm
        nih = report.get('nihss') or {}
        text(2 * cm, y, 'ƯỚC TÍNH NIHSS (4 items, tối đa 13)', 11, bold=True)
        y -= 0.55 * cm
        vn_items = {'item4_facial_palsy': 'Item 4 — Méo mặt',
                    'item5_motor_arm': 'Item 5 — Tay yếu',
                    'item6_motor_leg': 'Item 6 — Chân yếu',
                    'item10_dysarthria': 'Item 10 — Nói khó'}
        for it, label in vn_items.items():
            if it in (nih.get('items') or {}):
                text(2 * cm, y, f'{label}: {nih["items"][it]}', 10)
                y -= 0.5 * cm
        text(2 * cm, y,
             f"TỔNG: {nih.get('total','?')} ± {nih.get('margin',0)} "
             f"(CI 95%: {nih.get('ci_low','?')}–{nih.get('ci_high','?')})",
             10, bold=True)
        y -= 0.8 * cm

        tri = report.get('triage') or {}
        sub = tri.get('subtype') or {}
        text(2 * cm, y, 'SƠ BỘ (KHÔNG PHẢN CHẨN ĐOÁN)', 11, bold=True)
        y -= 0.55 * cm
        text(2 * cm, y,
             f"Nghi {sub.get('subtype','?')} ({sub.get('confidence','?')}%) "
             f"— Mức độ: {tri.get('severity','?')}", 10)
        y -= 0.5 * cm
        hosp = tri.get('hospital') or {}
        if hosp:
            text(2 * cm, y,
                 f"BV đề xuất: {hosp.get('name','')} — {hosp.get('phone','')}",
                 10)
            y -= 0.5 * cm

        y -= 0.5 * cm
        text(2 * cm, y, 'TIMELINE (THỜI GIAN VÀNG)', 11, bold=True)
        y -= 0.55 * cm
        for ev in report.get('timeline', []):
            text(2 * cm, y,
                 f"{ev['tag']} +{ev['elapsed_min']}p ({ev['timestamp']}) — "
                 f"{ev['description']}", 9)
            y -= 0.45 * cm
            if y < 4 * cm:
                c.showPage()
                y = h - 2 * cm

        # QR ở góc cuối
        qr_path = report.get('qr_path')
        if qr_path and os.path.exists(qr_path):
            from reportlab.lib.utils import ImageReader
            c.drawImage(ImageReader(qr_path), w - 5.5 * cm, 2 * cm,
                        4 * cm, 4 * cm)
            text(w - 5.5 * cm, 1.5 * cm, 'Quét xem tóm tắt', 8)

        c.save()
        return path

    # ------------------------------------------------------------------
    # TIỆN ÍCH
    # ------------------------------------------------------------------
    @staticmethod
    def qr_data_url(png_path):
        """QR → data URL (nhúng thẳng vào Streamlit không cần file)."""
        with open(png_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}"


# ======================================================================
# TEST
# ======================================================================
def test_handoff_system():
    print("=" * 70)
    print("HANDOFF SYSTEM TEST")
    print("=" * 70)

    hs = HandoffSystem(
        patient_info={'name': 'NGUYEN VAN TEST', 'age': 67,
                      'gender': 'Nam', 'phone': '0900000000'})
    hs.add_event('T1', 'Dau hieu dau: meo mat ben phai')
    hs.add_event('T3', 'CANH BAO EMERGENCY — fusion 85, NIHSS 9')
    hs.add_event('T4', 'Goi 115')

    mods = {'face': {'score': 70, 'status': 'WARNING'},
            'arm': {'arm_prob': 88, 'status': 'DANGER'},
            'gait': {'gait_prob': 40, 'status': 'MONITOR'},
            'speech': {'speech_prob': 65, 'status': 'DANGER'}}
    from fusion.nihss_estimator import estimate_nihss, calculate_nihss_ci
    est = estimate_nihss(mods)
    ci = calculate_nihss_ci(mods, n_iter=300)
    nih = {**est, **ci}

    from fusion.triage_engine import TriageEngine
    tri = TriageEngine().triage({}, nih['total'],
                                symptoms={'gradual_progression': True},
                                module_results=mods)

    report = hs.build_report({}, nih, tri, alerts=[{'level': 'EMERGENCY',
                                                    'score': 85}])
    jpath = hs.save_report(report)
    qr_path, qr_data = hs.generate_qr(report)
    pdf_path = hs.generate_pdf(report)
    report['qr_path'] = qr_path
    pdf2 = hs.generate_pdf(report)  # lần 2 có QR

    print(f"1. Report JSON: {jpath} ({os.path.getsize(jpath)} bytes)")
    print(f"2. QR: {qr_path}")
    print(qr_data[:120])
    print(f"3. PDF: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
    print(f"   PDF có QR: {os.path.getsize(pdf2)} bytes")
    assert os.path.exists(jpath) and os.path.exists(qr_path)
    assert os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000
    print("\nALL HANDOFF TESTS PASS")


if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_handoff_system()
