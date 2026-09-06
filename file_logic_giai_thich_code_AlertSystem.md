# FILE LOGIC GIẢI THÍCH — ALERT SYSTEM (src/alerts/alert_system.py)
> Tài liệu học cho phỏng vấn KHKT 2026 — tạo 06/09/2026
> Code: `src/alerts/alert_system.py` | Test: chạy `venv/Scripts/python.exe src/alerts/alert_system.py` → 7/7 PASS

---

## 1. ALERT SYSTEM LÀ GÌ? VAI TRÒ TRONG PSCS?

```
[5 Modules] → [FusionEngine] → [AlertSystem] → User + Người thân
   prob%        fused_score       3 kênh báo động
               + risk_level
```

- **Vị trí:** tầng CUỐI của pipeline. FusionEngine đưa ra *quyết định* (risk level),
  AlertSystem đưa ra *hành động* (báo cho ai, bằng gì, khi nào).
- **Vấn đề AlertSystem giải quyết:** risk EMERGENCY xuất hiện lúc 3h sáng —
  ai nghe thấy? → cần đa kênh (còi + log + tin nhắn Zalo cho người thân).

---

## 2. LOGIC CHÍNH (đọc code theo thứ tự này)

### 2.1. Input: 2 cách gọi
| Cách | Hàm | Khi nào dùng |
|---|---|---|
| Auto | `process_fusion_result(fusion_result)` | Dashboard/main loop gọi mỗi chu kỳ fusion |
| Manual | `send_alert(score, message, level)` | Test, module khác tự báo (vd radar phát hiện ngã) |

### 2.2. Quyết định mức (đồng bộ FusionEngine)
```
score >= 80  HOẶC risk == EMERGENCY  →  EMERGENCY  (lịch Ngày 5: "trigger khi score >80")
score >= 50  HOẶC risk == WARNING    →  WARNING
còn lại (NORMAL/MONITOR)             →  chỉ ghi log INFO, KHÔNG báo động
```
**Điểm cần nói:** tại sao kiểm tra cả score LẪN risk_level? Vì 2 lớp có thể
không đồng bộ: R1 rule của fusion nâng score lên 75 (EMERGENCY), nhưng nếu
main tự tính score=95 với risk=NORMAL (config sai) → AlertSystem vẫn chặn
được nhờ ngưỡng score 80. **Phòng thủ 2 lớp (defense in depth).**

### 2.3. 3 kênh phát
| Kênh | Code | Chi tiết |
|---|---|---|
| Buzzer | `_buzzer()` | `winsound.Beep(freq, ms)` — Windows only. Pattern: WARNING 2 beep 880Hz, EMERGENCY 4 beep xen kẽ 1200/900Hz (tần số cao = nguy cấp hơn) |
| Log | `_log_event()` | Console print + file JSONL `logs/alerts/alerts_YYYYMMDD.jsonl` (1 event 1 dòng — dễ import pandas sau) |
| Zalo | `_zalo()` | **STUB** — chưa gửi thật. Chỉ EMERGENCY mới gửi. TODO: POST `openapi.zalo.me/v2.0/oa/message` |

**Điểm cần nói:** JSONL chứ không JSON array? → Append-only, không cần đọc/ghi
lại cả file, không sợ crash giữa chừng làm hỏng file. Chuẩn log production.

### 2.4. Cooldown — chống báo động spam
```
cùng MỨC alert trong cooldown_seconds (mặc định 60s) → chỉ phát 1 lần
  self._last_alert_time[level] lưu thời điểm alert cuối của MỖI mức
```
- **Tại sao per-level?** WARNING lúc 10:00 không được phép chặn EMERGENCY lúc 10:01
  (nguy cấp hơn phải luôn xuyên qua).
- Số lần bị chặn đếm vào `suppressed_count` (Dashboard hiển thị).
- **Câu hỏi hay bị hỏi:** cooldown có nguy hiểm không? → Có, nếu người bệnh
  xấu đi trong 60s thì mất 1 alert. Đó là lý do EMERGENCY cooldown nên ngắn
  hơn WARNING (future: cooldown riêng theo mức).

### 2.5. Output chuẩn của mọi đường đi
```python
{'triggered': bool, 'level': str|None, 'reason': str, 'event': dict?}
```
→ Caller (Dashboard) luôn biết alert có phát thật hay bị chặn, vì sao.

---

## 3. SỐ LIỆU FLASH CARD (học thuộc)

| Số liệu | Giá trị |
|---|---|
| Ngưỡng EMERGENCY (score) | **>= 80** (theo lịch Ngày 5) |
| Ngưỡng WARNING (score) | **>= 50** |
| Cooldown mặc định | **60 giây** (per-level) |
| Pattern WARNING | 2 beep **880 Hz** × 300ms |
| Pattern EMERGENCY | 4 beep xen kẽ **1200/900 Hz** × 250ms |
| Log format | **JSONL** — `logs/alerts/alerts_YYYYMMDD.jsonl` |
| Test coverage | **7/7 PASS** (5 mức + 1 cooldown + 1 manual) |
| Zalo | Stub — chỉ EMERGENCY, cần access token OA |

---

## 4. Q&A PHỎNG VẤN

**Q1: Tại sao không đưa AlertSystem vào luôn FusionEngine?**
A: Tách biệt trách nhiệm (separation of concerns). Fusion = THUẦN TOÁN
(đầu vào module scores → đầu ra risk level, dễ unit test không side-effect).
Alert = SIDE EFFECT (buzzer, file, network). Tách cho phép: test fusion không
kêu còi; đổi kênh báo động không đụng thuật toán; sau này thêm SMS/phone
call chỉ mở rộng AlertSystem.

**Q2: Nếu process crash giữa lúc phát EMERGENCY thì sao?**
A: Log JSONL được ghi TRƯỚC khi trả về (append ngay) nên event không mất.
Nhược điểm thừa nhận: buzzer + Zalo chưa có acknowledgment/retry — đây là
hạn chế v1, hướng phát triển: hàng đợi retry + watchdog process.

**Q3: Tại sao cooldown lưu theo level mà không theo message?**
A: Mục tiêu là chống SPAM cùng mức nguy cơ, không phải dedup nội dung.
Theo message sẽ phức tạp (hash so chuỗi) và nguy hiểm: cùng mức nhưng message
khác → spam liên tục. Level là ranh giới an toàn đúng nghĩa.

**Q4: winsound có phải lựa chọn tốt không?**
A: Không cross-platform (Windows only). v1 chọn nó vì demo chạy Windows +
không cần cài thêm gì. Code đã guard import (`WINSOUND_AVAILABLE`) — trên
Linux fallback in text, không crash. Production nhúng (RPi) sẽ dùng GPIO buzzer.

**Q5: score=95 nhưng risk_level=NORMAL — báo hay không?**
A: BÁO (EMERGENCY). Test kịch bản 5 chứng minh. Lý do: ngưỡng 80 là lưới
an toàn độc lập với config risk level phía trên — không bao giờ tin 1 lớp
duy nhất (defensive programming).

---

## 5. CODE CẦN THUỘC (5 dòng lõi)

```python
# 1. Quyết định mức — 2 lớp bảo vệ
if risk == 'EMERGENCY' or score >= self.alert_threshold:   # 80
    level = 'EMERGENCY'
elif risk == 'WARNING' or score >= 50:
    level = 'WARNING'

# 2. Cooldown per-level
if now - self._last_alert_time.get(level, 0) < self.cooldown_seconds:
    return {'triggered': False, ...}          # chặn, đếm suppressed

# 3. JSONL append — crash-safe
with open(self.log_file, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')
```
