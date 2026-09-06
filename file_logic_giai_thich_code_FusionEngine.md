# 📚 FILE LOGIC GIẢI THÍCH CODE - FUSION ENGINE
## PSCS v8.0 | Tài liệu học phỏng vấn | Ngày tạo: 04/09/2026

> **Mục đích:** Hiểu SÂU logic Fusion Engine để trả lời phỏng vấn không bị rớt.
> Học thuộc: công thức, con số, lý do thiết kế, và 2 ví dụ tính tay.

---

## 1. FUSION ENGINE LÀ GÌ? VỊ TRÍ TRONG HỆ THỐNG?

```
┌────────────────────────────────────────────────────────────────┐
│                     KIẾN TRÚC TỔNG HỆ THỐNG                     │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MODULE 1 (Face)  ──► {'score', status, nihss_item_4}          │
│  MODULE 2 (Speech)──► {'speech_prob', status, nihss_score}     │
│  MODULE 3 (Arm)   ──► {'arm_prob', status, nihss_score}        │
│  MODULE 4 (Gait)  ──► {'gait_prob', status, nihss_score}       │
│  MODULE 5 (Radar) ──► {'fall_prob', status, nihss_score}       │
│         │                                                        │
│         ▼                                                        │
│  ╔══════════════════════════════╗                               │
│  ║      FUSION ENGINE           ║                               │
│  ║  1. Chuẩn hóa (adapter)      ║                               │
│  ║  2. Weighted average         ║                               │
│  ║  3. Luật FAST (R1, R2)       ║                               │
│  ║  4. NIHSS ước tính           ║                               │
│  ╚══════════════════════════════╝                               │
│         │                                                        │
│         ▼                                                        │
│  {fused_score, risk_level, nihss_total, recommendation}        │
│         │                                                        │
│         ▼                                                        │
│  ALERT SYSTEM ──► DEFENSE ENGINE (Layer 2-4) ──► DASHBOARD     │
└────────────────────────────────────────────────────────────────┘
```

**Câu nói ngắn khi phỏng vấn:**
> "Fusion Engine là bộ não quyết định: nhận 5 kết quả từ 5 module, tính 1 điểm nguy cơ 0-100 bằng trung bình có trọng số, rồi NÂNG CẤP theo luật lâm sàng FAST, đồng thời ước tính điểm NIHSS để phân loại độ nặng."

---

## 2. BẢNG FORMAT INPUT (CẦN THUỘC)

| Module | Key xác suất | NIHSS item | Status hợp lệ |
|--------|-------------|------------|----------------|
| Face | `score` | item 4 (Facial Palsy, 0-4) | NORMAL/WARNING/DANGER |
| Speech | `speech_prob` | item 10 (Dysarthria, 0-3) | + NO_SPEECH |
| Arm | `arm_prob` | item 5 (Motor Arm, 0-4) | + NO_PERSON |
| Gait | `gait_prob` | item 6 (Motor Leg, 0-4) | + NO_DATA |
| Radar | `fall_prob` | — (không có trong NIHSS) | + NO_SIGNAL |

**Điểm mấu chốt:** mỗi module đặt tên xác suất KHÁC NHAU → cần **adapter** chuẩn hóa về `{'prob', 'status', 'nihss', 'valid'}`.

---

## 3. LOGIC 5 BƯỚC + CODE QUAN TRỌNG

### BƯỚC 1: Adapter — chuẩn hóa input

```python
prob_keys = {
    'face': 'score', 'speech': 'speech_prob', 'arm': 'arm_prob',
    'gait': 'gait_prob', 'radar': 'fall_prob'
}
pk = prob_keys.get(key, 'prob')
prob = result.get(pk)
status = result.get('status', 'ERROR')
valid = status not in self.INVALID_STATUSES
```

**Tại sao cần `valid`?** Vì `NO_FACE` (không thấy mặt) ≠ `NORMAL` (mặt bình thường).
Nếu đếm NO_FACE là 0 điểm → hệ thống tự đánh lừa rằng "không có gì sai".

---

### BƯỚC 2: Weighted Average + Renormalization ⭐ (CÔNG THỨC CHÍNH)

```python
total_w = sum(self.weights.get(k, 0.15) for k in used)
fused_score = sum(self.weights.get(k, 0.15) * v['prob'] for k, v in used.items()) / total_w
```

**Công thức toán học (viết lên bảng được):**

```
             Σ (wᵢ × probᵢ)          0.20×prob_face + 0.30×prob_arm + ...
fused =  ─────────────────────  =  ─────────────────────────────────────────
              Σ wᵢ                     w_face + w_arm + ... (chỉ module HỢP LỆ)
```

**VÌ SAO CHIA TỔNG TRỌNG SỐ (renormalize)?**
- Khi chỉ có 3/5 module hoạt động, tổng trọng số chỉ còn 0.65
- Nếu KHÔNG chia → điểm bị "pha loãng" giả tạo: 4 module bình thường + 1 module nguy → điểm thấp bất thường
- Nếu chia → module nguy giữ đúng "sức nặng" tương đối → công bằng ở mọi số module

**Ví dụ:** chỉ face (70) + speech (15) hoạt động:
- Không renormalize: (0.2×70 + 0.2×15) = 17.0 → sai!
- Có renormalize: 17.0 / (0.2+0.2) = **42.5** → đúng xu hướng

---

### BƯỚC 3: Trọng số — CON SỐ VÀ LÝ DO ⭐⭐ (HỎI NHẤT LÀ ĐÂY)

```python
DEFAULT_WEIGHTS = {'face': 0.20, 'speech': 0.20, 'arm': 0.30,
                   'gait': 0.15, 'radar': 0.15}
```

| Module | Trọng số | Lý do (trả lời phỏng vấn) |
|--------|----------|---------------------------|
| **Arm 0.30** | Cao nhất | 1) NIHSS item 5 (Motor Arm) tương quan mạnh nhất với tổn thương 1 bên; 2) ML accuracy cao nhất 99.5%; 3) dấu hiệu FAST: "Arm" |
| Face 0.20 | Trung bình | Dấu hiệu FAST cốt lõi, ML 93.75%, nhưng dễ nhiễu (ngáp, cười lệch, ánh sáng) |
| Speech 0.20 | Trung bình | Dấu hiệu FAST, ML 83.07%, nhưng phụ thuộc model tiếng Việt |
| Gait 0.15 | Thấp | **Kém đặc hiệu**: đi khập khiễng do nhiều nguyên nhân khác (viêm khớp, chân tê...), không riêng đột quỵ |
| Radar 0.15 | Thấp | Ngã chỉ là proxy (gián tiếp), chưa có validation lâm sàng |

**Câu trả lời mẫu nếu bị hỏi "trọng số có khoa học không?":**
> "Đây là trọng số hand-crafted dựa trên 3 căn cứ: (1) vị trí item trong thang NIHSS — motor arm chiếm item 5 và tương quan tổn thương khu vực vận động; (2) độ chính xác ML thực tế của từng module; (3) độ đặc hiệu lâm sàng — gait kém đặc hiệu nên trọng số thấp. Trong tương lai, khi có dữ liệu thật gắn nhãn, tôi có thể học lại trọng số bằng logistic regression — hướng phát triển đã ghi rõ."

---

### BƯỚC 4: Lớp luật lâm sàng FAST (R1, R2) ⭐⭐

```python
# R1: >= 2/3 dấu hiệu FAST bất thường (prob >= 50) -> EMERGENCY
fast_signs = ['face', 'arm', 'speech']
abnormal_fast = [k for k in fast_signs if k in used and used[k]['prob'] >= 50]
if len(abnormal_fast) >= 2:
    fused_score = max(fused_score, 75.0)   # nâng lên EMERGENCY

# R2: 1 module rất nặng (prob >= 80) -> ít nhất WARNING
for k, v in used.items():
    if v['prob'] >= 80:
        fused_score = max(fused_score, 55.0)
        break
```

**TẠI SAO CẦN LUẬT KHI ĐÃ CÓ WEIGHTED AVERAGE?** (câu hỏi hay nhất)

> "Vì weighted average có thể bị **pha loãng**: face = 70 và arm = 65 nhưng speech + gait bình thường → điểm trung bình chỉ ~46 (MONITOR), trong khi lâm sàng 2/3 dấu hiệu FAST đồng thời là chỉ định cấp cứu. Luật R1 đảm bảo tổ hợp dấu hiệu được đánh giá đúng — đây chính là automation của nguyên tắc FAST mà y học đã dùng 20 năm."

**TẠI SAO R2 CHỈ NÂNG LÊN WARNING (55) KHÔNG PHẢI EMERGENCY?**

> "Một dấu hiệu đơn lẻ ≥80 có thể là nhiễu: ngáp làm méo mặt, mệt làm chậm nói. Tách kiểu này chỉ định KHÔNG được cấp cứu ngay mà WARNING + chờ **Defense Layer 3 (Temporal)** xác nhận tính liên tục. Còn ≥2 dấu hiệu FAST đồng thời thì xác suất trùng nhiễu rất thấp → cấp cứu."

---

### BƯỚC 5: NIHSS ước tính + phân loại độ nặng

```python
nihss_total = sum(nihss_items.values())   # item4 + item5 + item6 + item10

def _classify_severity(nihss_total):
    if nihss_total >= 14: return 'SEVERE'
    if nihss_total >= 6:  return 'MODERATE'
    return 'MILD'
```

**BẢNG NIHSS (thuộc lòng):**

| Item | Module | Max điểm |
|------|--------|----------|
| 4 - Facial Palsy | Face | 4 |
| 5 - Motor Arm | Arm | 4 |
| 6 - Motor Leg | Gait | 4 |
| 10 - Dysarthria | Speech | 3 |
| **Tổng tối đa (4/15 items)** | | **15** |

**Phân loại:** MILD 0-5 | MODERATE 6-13 | SEVERE 14-15

**Câu trả lời mẫu "NIHSS này có chính xác không?":**
> "Đây là NIHSS ƯỚC TÍNH sơ bộ từ 4/15 items của thang NIHSS, mục đích triage trước viện — giúp bệnh viện chuẩn bị trước. Nó KHÔNG THAY THẾ đánh giá NIHSS đầy đủ của bác sĩ (15 items + chuyên môn). Hệ thống luôn hiển thị rõ đây là ước tính 4 items."

---

## 4. BẢNG RISK LEVEL + RECOMMENDATION (THUỘC LÒNG)

| fused_score | risk_level | recommendation |
|-------------|-----------|----------------|
| < 30 | NORMAL | Tiếp tục theo dõi bình thường |
| 30-49 | MONITOR | Yêu cầu làm lại bài test |
| 50-69 | WARNING | Báo người thân kiểm tra - liên hệ bác sĩ |
| ≥ 70 | EMERGENCY | **GỌI 115 NGAY** - có dấu hiệu đột quỵ |

**Tại sao 4 mức thay vì 3?** — Mức MONITOR tạo "vùng đệm" (hysteresis) giảm báo động giật: điểm 35 không nhảy từ NORMAL thẳng sang WARNING.

---

## 5. VÍ DỤ TÍNH TAY ⭐ (BẮT BUỘC THUỘC)

**Input:** face=70, speech=15, arm=65, gait=20 (radar chưa có)

```
Bước 1 - Weighted average (tổng trọng số = 0.20+0.20+0.30+0.15 = 0.85):
  (0.20×70 + 0.20×15 + 0.30×65 + 0.15×20) / 0.85
  = (14.0 + 3.0 + 19.5 + 3.0) / 0.85
  = 39.5 / 0.85 = 46.5  → MONITOR (30-49)

Bước 2 - Luật FAST:
  Dấu hiệu >= 50: face (70 ✓), arm (65 ✓)  → 2/3 FAST
  R1 kích hoạt: fused = max(46.5, 75) = 75.0

KẾT QUẢ: fused=75.0 → EMERGENCY → "GỌI 115 NGAY"
```

**Đây chính là kịch bản test "FAST+ face + arm" — nếu không có luật R1, hệ thống đã bỏ sót ca cấp cứu (46.5 = MONITOR)!**

---

## 6. KẾT QUẢ TEST THẬT (7 KỊCH BẢN - 04/09)

| Kịch bản | fused | Risk | Ghi chú |
|----------|-------|------|---------|
| Tất cả bình thường | 3.4 | NORMAL | không luật nào |
| Chỉ face 85 | 55.0 | WARNING | R2 kích hoạt |
| face 70 + arm 65 | 75.0 | EMERGENCY | **R1 cứu ca này** |
| 3/3 FAST + dysarthria | 82.8 | EMERGENCY | R1 + R2, NIHSS 11 |
| Thiếu module (còn 2/5) | 9.0 | NORMAL | renormalize OK |
| Module lỗi NO_FACE/NO_SPEECH | 40.0 | MONITOR | skip + renormalize |
| Không module nào | 0.0 | NORMAL | không crash |

---

## 7. CÂU HỎI PHỎNG VẤN THƯỜNG GẶP (10 CÂU)

**Q1: Tại sao dùng weighted average mà không train 1 model fusion (stacking)?**
> "3 lý do: (1) **Giải thích được** — y khoa bắt buộc biết vì sao báo động, weighted average chỉ ra được từng module đóng góp bao nhiêu; (2) **Thiếu data gắn nhãn** cho ca đột quỵ thật để train model fusion, model nhỏ sẽ overfit; (3) **Tách bạch lỗi** — khi có sự cố tôi biết module nào sai, model đen hộp thì không. Khi có data thật, hướng phát triển là Bayesian fusion."

**Q2: Trọng số lấy ở đâu ra?** → Xem mục 3, bảng lý do 3 căn cứ.

**Q3: Module bị lỗi thì sao — có dừng hệ thống không?**
> "Không. Module lỗi bị đánh dấu invalid và LOẠI KHỎI fusion, trọng số được renormalize về tổng 1 để các module còn lại vẫn hoạt động đúng tỷ lệ. Hệ thống chỉ dừng khi KHÔNG còn module nào — lúc đó trả risk NORMAL kèm danh sách module lỗi để dashboard hiển thị."

**Q4: Tại sao NO_FACE ≠ NORMAL?**
> "Vì 'không thấy mặt' là thiếu dữ liệu, không phải bằng chứng bình thường. Đếm nó là 0 sẽ che giấu sự cố camera. Đó là lý do có tập INVALID_STATUSES riêng."

**Q5: Ngưỡng 30/50/70 lấy ở đâu?**
> "Chuẩn hóa theo thang xác suất: <30 khả năng thấp (khớp ngưỡng NORMAL của từng module con), 30-50 cần theo dõi, ≥50 rõ ràng bất thường, ≥70 tổ hợp nhiều bằng chứng. Đây là tham số có thể tune lại trên dữ liệu thật."

**Q6: Trend dùng làm gì?**
> "Lưu lịch sử 100 lần fuse gần nhất, hàm get_trend() so sánh điểm đầu-cuối cửa sổ 5 lần: chênh >10 là WORSENING. Đây là tiền đề cho Defense Layer 3 (Temporal Analysis) — đột quỵ tiến triển LIÊN TỤC, còn nhiễu thì tạm thời."

**Q7: Nếu face=40, arm=45, speech=50 thì sao?**
> "Weighted average ≈ (8+13.5+10+15)/0.85... (với gait 20) ≈ 45 MONITOR. R1 không kích hoạt (cần ≥50), nhưng speech=50 đạt ngưỡng → hệ thống vẫn ở MONITOR, yêu cầu làm lại test. Đây là vùng xám có chủ đích — Defense Layer 3 sẽ xác nhận tính liên tục."

**Q8: Radar không có NIHSS item — xử lý thế nào?**
> "Đúng, ngã không có trong NIHSS. Radar đóng góp vào fused_score (nguy cơ tổng) nhưng KHÔNG vào nihss_total — nihss_items chỉ nhận face/arm/gait/speech."

**Q9: Điểm yếu lớn nhất của Fusion v1?** (thành thật = điểm cộng)
> "(1) Trọng số hand-crafted, chưa học từ dữ liệu thật; (2) fusion tĩnh từng thời điểm — chưa có bộ nhớ dài hạn (đang có history, sẽ dùng ở Layer 3); (3) ngưỡng luật FAST (50/80) chọn bằng kinh nghiệm, chưa được tối ưu ROC như Module 4."

**Q10: Nếu 2 module trả kết quả mâu thuẫn (face 90, arm 5)?**
> "R2 nâng lên WARNING (face ≥80), nhờ độ ưu tiên 'an toàn là trên hết'. Sau đó Layer 3 temporal + khả năng user làm lại test sẽ xác nhận. Fusion không tự 'bỏ phiếu' làm mờ đi — mỗi module giữ nguyên thông tin trong output để debug."

---

## 8. CODE CẦN THUỘC (TỐI THIỂU 8 DÒNG)

```python
# 1. Công thức weighted average có renormalize
total_w = sum(self.weights[k] for k in used)
fused_score = sum(self.weights[k] * v['prob'] for k, v in used.items()) / total_w

# 2. Luật FAST R1
abnormal_fast = [k for k in ['face','arm','speech'] if k in used and used[k]['prob'] >= 50]
if len(abnormal_fast) >= 2:
    fused_score = max(fused_score, 75.0)

# 3. NIHSS ước tính
nihss_total = sum(nihss_items.values())   # max 15 (4 items)
```

---

## 9. SỐ LIỆU NHANH (FLASH CARD)

- 5 module → 1 điểm | fused_score: 0-100
- Trọng số: arm **0.30** | face/speech **0.20** | gait/radar **0.15** (tổng 1.00)
- Luật R1: ≥**2**/3 FAST, prob ≥**50** → max(fused, **75**)
- Luật R2: 1 module prob ≥**80** → max(fused, **55**)
- Risk: NORMAL <**30** | MONITOR 30-**49** | WARNING 50-**69** | EMERGENCY ≥**70**
- NIHSS 4 items max **15**: MILD 0-**5** | MODERATE **6**-13 | SEVERE ≥**14**
- Trend: cửa sổ **5** lần fuse, chênh >**10** = WORSENING
- File: `src/fusion/fusion_engine.py` | Test: `venv/Scripts/python.exe src/fusion/fusion_engine.py`

---

**Người tạo:** PSCS Team (với Claude Code)
**Liên kết:** `LOI_SO_MODULE.md` (SYS-01) | Lịch Ngày 5
