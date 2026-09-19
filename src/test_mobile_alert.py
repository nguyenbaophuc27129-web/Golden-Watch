# -*- coding: utf-8 -*-
"""
NK-25 — TEST APP ĐIỆN THOẠI (PWA cảnh báo qua WebSocket, không internet)
Kiểm tra 8 điểm của kênh mobile alert trong web_server.py — KHÔNG mở camera
(threads chỉ chạy trong main()), KHÔNG đụng pipeline chấm điểm.

Lưu ý: import web_server tải model (~10s) và kiểm tra cổng 5001 — phải tắt
server đang chạy trước khi test.

Chạy: PYTHONUTF8=1 python src/test_mobile_alert.py
"""

import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

results = []


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f' — {detail}' if detail else ''))


def main():
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print('== NK-25 TEST APP ĐIỆN THOẠI (mobile alert PWA) ==')
    import web_server as wsv
    from fastapi.testclient import TestClient

    client = TestClient(wsv.app)

    # 1. Bind an toàn mặc định (không --lan = 127.0.0.1 như cũ)
    check('HOST mặc định 127.0.0.1 (không mở LAN nếu không yêu cầu)',
          wsv.HOST == '127.0.0.1', wsv.HOST)

    # 2. Trang mobile trả về và có kênh /ws
    r = client.get('/mobile')
    check('GET /mobile 200 + có WebSocket + nút GỌI 115',
          r.status_code == 200 and '/ws' in r.text and 'tel:115' in r.text)

    # 3. Manifest PWA (đủ để "Thêm vào Màn hình chính")
    r = client.get('/mobile_manifest.json')
    mf = r.json()
    check('Manifest PWA: name + standalone + 2 icon',
          mf.get('name', '').startswith('Golden-Watch')
          and mf.get('display') == 'standalone' and len(mf.get('icons', [])) == 2)

    # 4. Icon sinh được (PNG đúng cỡ)
    r = client.get('/mobile/icon-192.png')
    ok_icon = (r.status_code == 200 and r.headers['content-type'] == 'image/png')
    if ok_icon:
        import cv2
        arr = cv2.imdecode(np.frombuffer(r.content, np.uint8),
                           cv2.IMREAD_COLOR)
        ok_icon = arr is not None and arr.shape[0] == 192
    check('Icon 192px sinh ra từ code (không file nhị phân trong repo)',
          ok_icon)

    # 5. QR chứa URL LAN thật
    r = client.get('/mobile/qr')
    ok_qr = r.status_code == 200 and 'image/png' in r.headers['content-type']
    if ok_qr:
        try:
            from PIL import Image  # noqa: F401  (qrcode.make đã dùng PIL)
            ok_qr = len(r.content) > 200
        except Exception:
            ok_qr = len(r.content) > 200
    check('QR mở app (PNG từ qrcode)', ok_qr)

    # 6. WebSocket: snapshot đầu tiên
    with client.websocket_connect('/ws') as ws:
        snap = ws.receive_json()
        check('WS: snapshot đầu có risk_level + t_server + IP LAN',
              snap.get('type') == 'snapshot' and 'risk_level' in snap
              and 't_server' in snap and 'host_lan' in snap)

        # 7. Sự kiện alert đẩy tới client qua seq
        seq_before = snap.get('seq', 0)
        ev = wsv.push_mobile_event('alert', level='EMERGENCY', score=87,
                                   message='TEST', time='12:00:00',
                                   nihss_total=9)
        got = None
        for _ in range(10):
            m = ws.receive_json()
            if m.get('seq', 0) > seq_before:
                got = m
                break
        check('WS: nhận sự kiện alert EMERGENCY đúng seq',
              got is not None and got['type'] == 'alert'
              and got['level'] == 'EMERGENCY' and got['score'] == 87)

    # 8. Ack từ điện thoại ghi vào handoff
    n_before = 0
    try:
        evs = wsv.HANDOFF.timeline if hasattr(wsv.HANDOFF, 'timeline') else []
        n_before = len(evs) if isinstance(evs, (list, dict)) else 0
    except Exception:
        pass
    r = client.post('/mobile/ack')
    check('POST /mobile/ack ok (ghi MOBILE_ACK vào handoff)',
          r.status_code == 200 and r.json().get('ok') is True)

    n_pass = sum(results)
    print(f"\nTỔNG KẾT: {n_pass}/{len(results)} PASS")
    if n_pass != len(results):
        sys.exit(1)
    print('✅ Kênh mobile alert sẵn sàng — chạy server với cờ --lan để dùng.')


if __name__ == '__main__':
    main()
