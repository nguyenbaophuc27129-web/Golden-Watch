# -*- coding: utf-8 -*-
"""
ALERT SYSTEM v2 - PSCS v8.0 (Golden-Watch)
Hệ thống báo động khi Fusion Engine phát hiện nguy cơ đột quỵ.

v2 (06/09): BỎ Zalo → kênh APP IN-FEED (người thân mở app Streamlit thấy
tin nhắn báo động ngay + lịch sử), đúng yêu cầu sản phẩm gia đình.

Logic:
  1. INPUT: điểm nguy cơ (0-100) hoặc output của FusionEngine
  2. QUYẾT ĐỊNH báo động:
     - score >= 80 hoặc risk EMERGENCY  -> EMERGENCY (gọi 115 — thời gian vàng)
     - score >= 50 hoặc risk WARNING    -> WARNING
     - còn lại                          -> chỉ ghi log, không báo
  3. 3 KÊNH báo động:
     - Buzzer : còi local (winsound trên Windows, beep pattern theo mức)
     - Log    : console + file JSONL (logs/alerts/) để truy vết sau
     - App    : TIN NHẮN BÁO ĐỘNG cho người thân (feed trong app Streamlit,
                get_recent_alerts() đọc lại được) kèm khuyến nghị thời gian vàng
  4. COOLDOWN: chống báo động lặp (spam) - cùng mức trong X giây chỉ báo 1 lần

Tác giả: PSCS Team
Ngày: 06/09/2026 (v2)
"""

import os
import sys
import json
import time
import uuid
from collections import deque

# winsound chỉ có trên Windows (buzzer local)
WINSOUND_AVAILABLE = False
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    pass


class AlertSystem:
    """
    AlertSystem: nhận điểm nguy cơ -> quyết định -> phát báo động đa kênh.

    Usage:
        alerter = AlertSystem()
        handled = alerter.process_fusion_result(fusion_result)   # auto
        alerter.send_alert(85, "Fused score vuot nguong")        # manual
    """

    # Ngưỡng điểm cho từng mức báo động (đồng bộ FusionEngine RISK_THRESHOLDS)
    LEVEL_SCORES = {
        'EMERGENCY': 80,   # lịch Ngày 5: trigger khi score > 80
        'WARNING': 50,
    }

    # Pattern còi theo mức (tần số Hz, thời lượng ms)
    BUZZER_PATTERNS = {
        'WARNING': [(880, 300), (880, 300)],
        'EMERGENCY': [(1200, 250), (900, 250), (1200, 250), (900, 250)],
    }

    def __init__(self, alert_score_threshold=80, log_dir=None,
                 cooldown_seconds=60, buzzer_enabled=True,
                 history_size=200, session_id=None):
        """
        Args:
            alert_score_threshold: điểm EMERGENCY (mặc định 80 theo lịch)
            log_dir: thư mục log JSONL (None = logs/alerts so với project root)
            cooldown_seconds: khoảng cách tối thiểu 2 alert cùng mức
            buzzer_enabled: bật/tắt còi (tắt khi test không cần tiếng)
        """
        self.alert_threshold = alert_score_threshold
        self.cooldown_seconds = cooldown_seconds
        self.buzzer_enabled = buzzer_enabled
        self.session_id = session_id or uuid.uuid4().hex[:8]

        # Log dir mặc định: <project_root>/logs/alerts/
        if log_dir is None:
            project_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
            log_dir = os.path.join(project_root, 'logs', 'alerts')
        self.log_dir = log_dir
        self.log_file = os.path.join(
            log_dir, f"alerts_{time.strftime('%Y%m%d')}.jsonl")

        # Chống spam: thời điểm alert cuối theo mức
        self._last_alert_time = {}
        self.suppressed_count = 0

        # Lịch sử alert trong bộ nhớ (cho Dashboard/get_stats)
        self.history = deque(maxlen=history_size)

        # v2: APP FEED — tin nhắn cho người thân (Streamlit đọc lại được)
        self.app_feed = deque(maxlen=50)

    # ------------------------------------------------------------------
    # 1. INPUT CHÍNH
    # ------------------------------------------------------------------
    def process_fusion_result(self, fusion_result):
        """
        Tự động xử lý output của FusionEngine.

        Returns:
            dict: {'triggered': bool, 'level': str|None, 'reason': str}
        """
        if not fusion_result:
            return {'triggered': False, 'level': None, 'reason': 'NO_DATA'}

        score = fusion_result.get('fused_score', 0)
        risk = fusion_result.get('risk_level', 'NORMAL')
        nihss = fusion_result.get('nihss_total', 0)

        if risk == 'EMERGENCY' or score >= self.alert_threshold:
            msg = (f"NGUY CO DOT QUY CA! score={score}, NIHSS~{nihss}. "
                   f"Quy tac: {fusion_result.get('triggered_rules', [])}")
            return self.send_alert(score, msg, level='EMERGENCY',
                                   source='fusion', extra={'risk': risk,
                                                           'nihss': nihss})
        if risk == 'WARNING' or score >= self.LEVEL_SCORES['WARNING']:
            msg = (f"Canh bao: score={score}, NIHSS~{nihss}. "
                   f"{fusion_result.get('recommendation', '')}")
            return self.send_alert(score, msg, level='WARNING',
                                   source='fusion', extra={'risk': risk,
                                                           'nihss': nihss})
        # NORMAL / MONITOR: chỉ ghi log (không báo động)
        self._log_event('INFO', score,
                        f"risk={risk} score={score} - khong bao dong",
                        source='fusion')
        return {'triggered': False, 'level': None,
                'reason': f'risk={risk} duoi nguong'}

    def send_alert(self, score, message, level=None, source='manual',
                   extra=None):
        """
        API chính theo lịch Ngày 5: send_alert(score, message)

        Args:
            score: điểm nguy cơ 0-100
            message: nội dung alert
            level: 'EMERGENCY'|'WARNING' (None = tự suy từ score)
            source: 'fusion'|'manual'|'radar'|...
            extra: dict dữ liệu kèm (risk, nihss...)

        Returns:
            dict: {'triggered': bool, 'level': str|None, 'reason': str}
        """
        if level is None:
            level = ('EMERGENCY' if score >= self.alert_threshold
                     else 'WARNING' if score >= self.LEVEL_SCORES['WARNING']
                     else 'INFO')

        # ----- COOLDOWN: chống spam cùng mức -----
        now = time.time()
        last = self._last_alert_time.get(level, 0)
        if now - last < self.cooldown_seconds:
            self.suppressed_count += 1
            return {'triggered': False, 'level': level,
                    'reason': f'cooldown ({self.cooldown_seconds}s)'}

        self._last_alert_time[level] = now

        # ----- PHÁT THEO 3 KÊNH -----
        event = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': self.session_id,
            'level': level,
            'score': score,
            'message': message,
            'source': source,
            'extra': extra or {},
        }
        self._buzzer(level)
        self._log_event(level, score, message, source, extra)
        event['app_message'] = self._app(level, score, message, extra)

        self.history.append(event)
        return {'triggered': True, 'level': level, 'reason': 'ALERT_SENT',
                'event': event}

    # ------------------------------------------------------------------
    # 2. CÁC KÊNH
    # ------------------------------------------------------------------
    def _buzzer(self, level):
        """Kênh 1: còi local - beep pattern theo mức (chỉ Windows)"""
        pattern = self.BUZZER_PATTERNS.get(level)
        if not pattern or not self.buzzer_enabled:
            return
        if WINSOUND_AVAILABLE:
            try:
                for freq, dur_ms in pattern:
                    winsound.Beep(freq, dur_ms)
            except Exception as e:
                print(f"[ALERT] Buzzer error: {e}")
        else:
            print(f"[ALERT] Buzzer (khong co winsound): {level} x{len(pattern)} beep")

    def _log_event(self, level, score, message, source, extra=None):
        """Kênh 2: console + file JSONL (1 dòng 1 event - dễ phân tích sau)"""
        tag = {'INFO': '[ALERT-INFO]', 'WARNING': '[ALERT-WARNING]',
               'EMERGENCY': '[ALERT-EMERGENCY]'}.get(level, '[ALERT]')
        print(f"{tag} score={score} | {message}")

        try:
            os.makedirs(self.log_dir, exist_ok=True)
            record = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'session_id': self.session_id,
                'level': level,
                'score': score,
                'message': message,
                'source': source,
                'extra': extra or {},
            }
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"[ALERT] Log file error: {e}")

    def configure_zalo(self, access_token, user_id):
        """ĐÃ LOẠI v2 — giữ tên hàm để tương thích, không làm gì."""
        print("[ALERT] Zalo da loai bo v2 — dung kenh App in-feed")

    def _app(self, level, score, message, extra=None):
        """
        Kênh 3 (v2): TIN NHẮN BÁO ĐỘNG trong app cho người thân.
        Nội dung nêu rõ: nguy cơ + việc cần làm NGAY (thời gian vàng < 4.5h
        theo Saver 2006 — 'Time is Brain').
        """
        if level == 'EMERGENCY':
            family_msg = (
                f"CANH BAO DO — nguoi than co nguyen co DOT QUY cao "
                f"(diem {score:.0f}/100). "
                f"GOI 115 NGAY — thoi gianvang < 4.5 gio, moi phut tre mo "
                f"~1.9 trieu neuron. Khuyen nghi: {extra.get('recommendation', 'goi 115, khong de nhan benh an/uoong nuoc') if extra else 'goi 115'}")
        elif level == 'WARNING':
            family_msg = (
                f"Canh bao VANG — co dau hieu bat thuong (diem {score:.0f}/100). "
                f"Nguoi than kiem tra ngay: noi chuyen, nang 2 tay, cuoi thu. "
                f"Neu 1 trong 3 kho → GOI 115.")
        else:
            family_msg = f"Ghi nhan (diem {score:.0f}/100)."
        feed_item = {
            'timestamp': time.strftime('%d/%m %H:%M:%S'),
            'level': level,
            'score': score,
            'text': family_msg,
        }
        self.app_feed.appendleft(feed_item)
        return family_msg

    def get_recent_alerts(self, n=10):
        """App Streamlit đọc feed tin nhắn báo động (mới nhất trước)."""
        return list(self.app_feed)[:n]

    # ------------------------------------------------------------------
    # 3. TIỆN ÍCH
    # ------------------------------------------------------------------
    def get_stats(self):
        """Thống kê phiên (cho Dashboard)"""
        levels = [e['level'] for e in self.history]
        return {
            'session_id': self.session_id,
            'total_alerts': len(self.history),
            'emergency': levels.count('EMERGENCY'),
            'warning': levels.count('WARNING'),
            'suppressed_by_cooldown': self.suppressed_count,
            'log_file': self.log_file,
        }


# ======================================================================
# TEST - kịch bản theo lịch Ngày 5
# ======================================================================
def test_alert_system():
    print("=" * 70)
    print("ALERT SYSTEM TEST (buzzer OFF)")
    print("=" * 70)

    alerter = AlertSystem(buzzer_enabled=False, cooldown_seconds=0)

    # Kịch bản FusionEngine (giống test_fusion_engine)
    def fusion(score, risk, nihss=0, rules=None):
        return {'fused_score': score, 'risk_level': risk,
                'nihss_total': nihss, 'triggered_rules': rules or [],
                'recommendation': ''}

    # ===== NHÓM 1: quyết định MỨC alert (cooldown = 0 để không chặn nhau) =====
    scenarios = [
        ("Binh thuong (score 10) -> KHONG alert",
         fusion(10, 'NORMAL'), False),
        ("MONITOR (score 40) -> KHONG alert, chi log",
         fusion(40, 'MONITOR'), False),
        ("WARNING (score 60) -> WARNING",
         fusion(60, 'WARNING', 6), True),
        ("EMERGENCY (score 85, R1 2/3 FAST) -> EMERGENCY",
         fusion(85, 'EMERGENCY', 9, ["R1: 2/3 dau hieu FAST"]),
         True),
        ("score 95 risk NORMAL -> van EMERGENCY (score > 80)",
         fusion(95, 'NORMAL'), True),
    ]

    passed = 0
    total = 0
    for i, (name, fus, expect) in enumerate(scenarios, 1):
        r = alerter.process_fusion_result(fus)
        ok = r['triggered'] == expect
        passed += ok
        total += 1
        status = 'PASS' if ok else 'FAIL'
        print(f"{i}. [{status}] {name} -> triggered={r['triggered']} "
              f"level={r['level']} ({r['reason']})")

    # ===== NHÓM 2: cooldown (bật cooldown 60s) =====
    alerter.cooldown_seconds = 60
    r = alerter.process_fusion_result(fusion(90, 'EMERGENCY'))
    ok = not r['triggered'] and 'cooldown' in r['reason']
    passed += ok
    total += 1
    print(f"6. [{'PASS' if ok else 'FAIL'}] Cooldown: EMERGENCY lien tiep "
          f"-> bi chan -> triggered={r['triggered']} ({r['reason']})")

    # ===== NHÓM 3: send_alert thủ công (tắt cooldown lại) =====
    alerter.cooldown_seconds = 0
    r = alerter.send_alert(99, "Test manual")
    ok = r['triggered'] and r['level'] == 'EMERGENCY'
    passed += ok
    total += 1
    print(f"7. [{'PASS' if ok else 'FAIL'}] "
          f"send_alert(99) manual -> {r['level']}")

    # ===== NHÓM 4: app feed cho người thân =====
    feed = alerter.get_recent_alerts(5)
    ok = (len(feed) >= 1 and '115' in feed[0]['text']
          and feed[0]['level'] == 'EMERGENCY')
    passed += ok
    total += 1
    print(f"8. [{'PASS' if ok else 'FAIL'}] App feed co tin nhan "
          f"EMERGENCY + huong dan goi 115: '{feed[0]['text'][:60]}...'")

    print(f"\nKet qua: {passed}/{total} PASS")
    print(f"Stats: {alerter.get_stats()}")
    return passed == total


if __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ok = test_alert_system()
    sys.exit(0 if ok else 1)
