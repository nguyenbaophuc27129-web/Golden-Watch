# -*- coding: utf-8 -*-
"""Vẽ bộ biểu đồ cho NHAT_KY_DU_AN.md — TOÀN BỘ từ số LIỆU THẬT trong test_results/.

Nguồn dữ liệu (không số nào bịa):
- leak_check_blocksize.json            -> 02 BlockSweep
- face_v3_multiseed_*/summary.json     -> 03 đa seed
- speech_speaker_loso_*/summary.json   -> 04 LOSO speech + mic
- abstention_*/summary.json            -> 05 abstention
- early_warning_*/summary.json         -> 06 cảnh báo sớm
- _speech_full_run.log                 -> 07 loss train + tốc độ trích
- metrics_pack_*/metrics_summary.json  -> 08 bộ số 07/09
- số ngày 30/08 (LOI_SO_MODULE ghi lại) -> 09 khủng hoảng gait
- dòng thời gian dự án (phụ lục A)      -> 10 timeline

Output: test_results/nhat_ky_charts/*.png
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TR = ROOT / "test_results"
OUT = TR / "nhat_ky_charts"
OUT.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 130
DONE = []


def save(fig, name):
    p = OUT / name
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    DONE.append(name)


def load(rel):
    return json.loads((TR / rel).read_text(encoding="utf-8"))


# ---------- 01: Hành trình "số quá đẹp đã tự hủy" ----------
try:
    stories = [
        ("Face 26/08\n(no-split)", 93.75, 94.0, "93.75% acc\n-> AUC 0.94±0.01"),
        ("Gait v2 29/08\n(cửa sổ ngẫu nhiên)", 96.88, 87.9, "96.88% acc\n-> AUC 0.879 LOSO"),
        ("Speech 28/08\n(session×mic)", 99.2, 62.0, "acc 0.992\n-> AUC 0.620 LOSO"),
        ("Face v3.1 09/09\n(HistGB cây)", 99.96, 94.0, "AUC 0.9996\n-> LogReg 0.91–0.94"),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    xs = range(len(stories))
    w = 0.38
    before = [s[1] for s in stories]
    after = [s[2] for s in stories]
    ax.bar([x - w / 2 for x in xs], before, w, label="SỐ BAN ĐẦU (không tách người/block)", color="#d9534f")
    ax.bar([x + w / 2 for x in xs], after, w, label="SỐ KIỂM ĐỊNH ĐÚNG (protocol khóa)", color="#5cb85c")
    for x, b, a, note in zip(xs, before, after, [s[3] for s in stories]):
        ax.text(x - w / 2, b + 0.8, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(x + w / 2, a + 0.8, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(x, 113, note, ha="center", fontsize=7.2, color="#333")
    ax.set_ylim(0, 124)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([s[0] for s in stories], fontsize=8)
    ax.set_ylabel("Giá trị (%)")
    ax.set_title("4 lần tự bóc leakage của chính mình — số trước vs số đúng\n(trước = accuracy split sai; sau = AUC theo block/LOSO/đa seed)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "01_leakage_before_after.png")
except Exception as e:
    print("01 FAIL:", e)

# ---------- 02: BlockSweep ----------
try:
    data = load("leak_check_blocksize.json")
    blocks = [d["block"] for d in data]
    lr = [d["auc_logreg"] for d in data]
    gb = [d["auc_histgb"] for d in data]
    nb = [d["n_blocks"] for d in data]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(blocks, gb, "o-", color="#d9534f", label="HistGB (cây) — ≈1.0 MỌI cỡ block = leakage")
    ax.plot(blocks, lr, "s-", color="#0275d8", label="LogReg — giảm dần 0.94→0.91 (bình thường)")
    for b, a, n in zip(blocks, gb, nb):
        ax.annotate(f"{a:.4f}\n({n} block)", (b, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color="#d9534f")
    for b, a in zip(blocks, lr):
        ax.annotate(f"{a:.4f}", (b, a), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7.5, color="#0275d8")
    ax.set_xscale("log")
    ax.set_xticks(blocks)
    ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("Cỡ block (số mẫu liên tiếp) — thang log")
    ax.set_ylabel("AUC block test")
    ax.set_title("GIAO THỨC BLOCKSWEEP (10/09): chỉ báo leakage\nHistGB 0.9996 (09/09) bị chứng minh là học theo block dữ liệu", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "02_blocksweep.png")
except Exception as e:
    print("02 FAIL:", e)

# ---------- 03: Face v3 đa seed ----------
try:
    s = load("face_v3_multiseed_20260910_215359/summary.json")
    runs = s["runs"]
    seeds = [r["seed"] for r in runs]
    aucs = [r["auc_test"] for r in runs]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.axhspan(s["auc_mean"] - s["auc_sd"], s["auc_mean"] + s["auc_sd"], alpha=0.18, color="#5cb85c", label=f"±1SD [{s['auc_min']:.4f} – {s['auc_max']:.4f}]")
    ax.axhline(s["auc_mean"], color="#5cb85c", lw=1.5, label=f"mean {s['auc_mean']:.4f} ± {s['auc_sd']:.4f}")
    ax.plot(seeds, aucs, "o", color="#0275d8", ms=9)
    for sd, a in zip(seeds, aucs):
        ax.annotate(f"{a:.4f}", (sd, a), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.5)
    ax.set_ylim(0.915, 0.968)
    ax.set_xlabel("Seed")
    ax.set_ylabel("AUC test (block held-out)")
    ax.set_title("FACE v3 ĐA SEED (NK-04, 10/09): 10 seed → AUC ổn định\nCông bố chính thức: AUC 0.94 ± 0.01 (10 seed)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, "03_face_multiseed.png")
except Exception as e:
    print("03 FAIL:", e)

# ---------- 04: Speech LOSO + mic ----------
try:
    s = load("speech_speaker_loso_20260910_214834/summary.json")
    o = load("speech_opensmile_loso_20260910_221102/summary.json")
    items = [
        ("LogReg 48ft\nLOSO 15 người", s["p1"]["logreg"]["auc"], "#0275d8"),
        ("MLP 48ft\nLOSO", s["p1"]["mlp"]["auc"], "#5bc0de"),
        ("headMic\nriêng", s["p2"]["headMic_logreg"]["auc"], "#f0ad4e"),
        ("arrayMic\nriêng", s["p2"]["arrayMic_logreg"]["auc"], "#f0ad4e"),
        ("array→head\ncross-mic", s["p3"]["train_array_test_head_logreg"]["auc"], "#999"),
        ("head→array\ncross-mic", s["p3"]["train_head_test_array_logreg"]["auc"], "#d9534f"),
        ("openSMILE 88ft\nLOSO", o["opensmile_logreg"]["auc"], "#5cb85c"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    xs = range(len(items))
    ax.bar(xs, [i[1] for i in items], 0.62, color=[i[2] for i in items])
    ax.axhline(0.5, color="black", ls="--", lw=1, label="0.5 = đoán ngẫu nhiên")
    ax.axhline(0.992, color="#d9534f", ls=":", lw=1.2, label="0.992 (số CŨ session-level — ĐÃ HỦY)")
    for x, (name, v, _) in zip(xs, items):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([i[0] for i in items], fontsize=7.6)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("AUC")
    ax.set_title("SPEECH THẬT THEO NGƯỜI (NK-03 + NK-06, 10/09): n=1,100 file, 15 người TORGO\nhead→array 0.415 < 0.5 = đổi micro là gãy — hạn chế được đo và ghi nhận", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "04_speech_loso_mic.png")
except Exception as e:
    print("04 FAIL:", e)

# ---------- 05: Abstention ----------
try:
    s = load("abstention_20260912_122517/summary.json")
    cov = [1.0, 0.9, 0.8, 0.7]
    f = s["modules"]["face"]["selective"]
    sp = s["modules"]["speech"]["selective"]
    f_sens = [f["base"]["sens"]] + [f["steps"][k]["kept"]["sens"] for k in ("abstain_10", "abstain_20", "abstain_30")]
    f_fpr = [f["base"]["fpr"]] + [f["steps"][k]["kept"]["fpr"] for k in ("abstain_10", "abstain_20", "abstain_30")]
    s_sens = [sp["base"]["sens"]] + [sp["steps"][k]["kept"]["sens"] for k in ("abstain_10", "abstain_20", "abstain_30")]
    s_fpr = [sp["base"]["fpr"]] + [sp["steps"][k]["kept"]["fpr"] for k in ("abstain_10", "abstain_20", "abstain_30")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True)
    for ax, sens, fpr, title, unc in (
        (axes[0], f_sens, f_fpr, f"FACE — unc_error_auc {f['unc_error_auc']:.3f} (tốt)", "#0275d8"),
        (axes[1], s_sens, s_fpr, f"SPEECH — unc_error_auc {sp['unc_error_auc']:.3f} (≈random: hạn chế thật)", "#d9534f"),
    ):
        ax.plot([c * 100 for c in cov], sens, "o-", color="#5cb85c", label="sens %")
        ax.plot([c * 100 for c in cov], fpr, "s-", color=title.split("—")[1].strip().startswith("unc") and "#d9534f" or "#d9534f", label="FPR %")
        for c, se, fp in zip(cov, sens, fpr):
            ax.annotate(f"{se:.1f}", (c * 100, se), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5)
            ax.annotate(f"{fp:.1f}", (c * 100, fp), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7.5)
        ax.set_xlabel("Coverage (%) — nhường X% mẫu bất định nhất")
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Giá trị (%)")
    fig.suptitle("TRỤ CỘT A — ABSTENTION (NK-10, 12/09): nhường 20% khó nhất, face FPR 9.5→5.3 & sens 84→91.1", fontsize=10.5)
    save(fig, "05_abstention.png")
except Exception as e:
    print("05 FAIL:", e)

# ---------- 06: Early warning ----------
try:
    s = load("early_warning_20260912_124947/summary.json")
    lat = s["latency"]
    scen = ["ramp_+40/15ph", "ramp_+40/30ph", "ramp_+40/60ph", "ramp_+18/30ph_DUOI_NGUONG"]
    labels = ["+40/15ph", "+40/30ph", "+40/60ph", "+18/30ph\n(dưới ngưỡng 50)"]
    old = [lat[k]["old"]["median_min"] if lat[k]["old"] else 0 for k in scen]
    new = [lat[k]["new"]["median_min"] for k in scen]
    fired = [lat[k]["old_ever_fired"] + " → " + lat[k]["new_ever_fired"] for k in scen]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    xs = range(len(scen))
    w = 0.38
    ax.bar([x - w / 2 for x in xs], old, w, label="OLD (chỉ alert khi vượt ngưỡng)", color="#999")
    ax.bar([x + w / 2 for x in xs], new, w, label="NEW (cộng WATCH DETERIORATING)", color="#5cb85c")
    for x, o, n, fr in zip(xs, old, new, fired):
        if o:
            ax.text(x - w / 2, o + 0.6, f"{o:.1f}", ha="center", fontsize=8)
        else:
            ax.text(x - w / 2, 1.2, "KHÔNG BAO GIỜ\n(0/20 lần)", ha="center", fontsize=7.5, color="#d9534f")
        ax.text(x + w / 2, n + 0.6, f"{n:.1f}", ha="center", fontsize=8)
        ax.text(x, -6.5, fr, ha="center", fontsize=7.5, color="#555")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Độ trễ phát hiện (phút, median 20 lần chạy)")
    ax.set_ylim(0, 46)
    ax.set_title("TRỤ CỘT B — CẢNH BÁO SỚM EWMA+CUSUM (NK-11, 12/09, simulated):\ngiảm trễ 45–50% + bắt được drift chậm mà hệ cũ BỎ LỠ hoàn toàn (FAR 0/24h)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "06_early_warning.png")
except Exception as e:
    print("06 FAIL:", e)

# ---------- 07: Speech train 100% (loss + tốc độ) ----------
try:
    log = (TR / "_speech_full_run.log").read_text(encoding="utf-8", errors="replace")
    ep, lo = [], []
    for m in re.finditer(r"epoch (\d+)/150 loss ([\d.]+)", log):
        ep.append(int(m.group(1)))
        lo.append(float(m.group(2)))
    prog = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"\[(\d+)/17633\] ([\d.]+) file/s", log)]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    axes[0].plot(ep, lo, "o-", color="#0275d8", ms=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss train")
    axes[0].set_title("NK-12 (12/09): train 100% TORGO 17,631 file\nMLP 256-128-64, 150 epochs — loss 0.461→~0.13 (mô hình NGHIÊN CỨU)", fontsize=9.5)
    axes[0].grid(alpha=0.3)
    if prog:
        axes[1].plot([p[0] for p in prog], [p[1] for p in prog], "s-", color="#5cb85c", ms=4)
        axes[1].set_xlabel("Số file đã trích (của 17,633)")
        axes[1].set_ylabel("Tốc độ (file/s)")
        axes[1].set_title("Tốc độ trích đặc trưng: 26.9→33.4 file/s\n542 giây cho 17,631 file (2 lỗi bỏ: F01_0067/0068)", fontsize=9.5)
        axes[1].grid(alpha=0.3)
    save(fig, "07_speech_train100.png")
except Exception as e:
    print("07 FAIL:", e)

# ---------- 08: Bộ số 07/09 ----------
try:
    s = load("metrics_pack_20260907_223217/metrics_summary.json")
    rows = {r["module"]: r for r in s["results"]}
    items = [
        ("face_rules\n(rules)", rows["face_rules"]["auc_roc"], rows["face_rules"]["auc_ci95"], "#d9534f"),
        ("face_ml_logreg\n(5 ft)", rows["face_ml_logreg"]["auc_roc"], rows["face_ml_logreg"]["auc_ci95"], "#0275d8"),
        ("gait_loso_v2\n(15 subject)", rows["gait_loso_v2"]["auc_roc"], rows["gait_loso_v2"]["auc_ci95"], "#5cb85c"),
        ("gait_subject15\n(theo người)", rows["gait_loso_subject15"]["auc_roc"], rows["gait_loso_subject15"]["auc_ci95"], "#2e7d32"),
        ("speech_torgo\n(session-level!)", rows["speech_torgo"]["auc_roc"], rows["speech_torgo"]["auc_ci95"], "#f0ad4e"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    xs = range(len(items))
    ax.bar(xs, [i[1] for i in items], 0.6, color=[i[3] for i in items])
    ax.axhline(0.5, color="black", ls="--", lw=1, label="0.5 = ngẫu nhiên")
    for x, (name, v, ci, _) in zip(xs, items):
        ax.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
        ax.text(x, 0.045, f"CI95\n{ci}", ha="center", fontsize=6.8, color="white" if v > 0.75 else "black")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([i[0] for i in items], fontsize=7.8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("AUC")
    ax.set_title("BỘ SỐ KIỂM ĐỊNH 07/09 (metrics_pack_20260907_223217, seed 42)\nspeech 0.992 ghi NGUYÊN rồi về sau (10/09) tự bóc là session-level → 0.620", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "08_metrics_pack_0709.png")
except Exception as e:
    print("08 FAIL:", e)

# ---------- 09: Khủng hoảng gait 30/08 ----------
try:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].bar(["Ngưỡng cũ 0.5\n(FPR 33.3%)", "Youden 0.1\n(FPR 11.3%)"], [33.3, 11.3], 0.5, color=["#d9534f", "#5cb85c"])
    for x, v in zip(range(2), [33.3, 11.3]):
        axes[0].text(x, v + 0.7, f"{v}%", ha="center", fontsize=9)
    axes[0].set_ylabel("FPR trên người khỏe (%)")
    axes[0].set_title("30/08: gait n=10 — cảnh báo sai 1/3 lượt\n→ chốt ngưỡng Youden J=0.824", fontsize=9.5)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(["AUC ngày 30/08\n(n=10)", "AUC LOSO 07/09\n(n=162 cửa sổ)"], [0.98, 0.879], 0.5, color=["#f0ad4e", "#0275d8"])
    for x, v in zip(range(2), [0.98, 0.879]):
        axes[1].text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("AUC n=10 đẹp → phải trả giá bằng LOSO\n(số n=10 chỉ tham chiếu, không công bố)", fontsize=9.5)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("KHỦNG HOẢNG 30/08: FPR 33.3% → Youden 64% → FPR 11.3% (số ghi trong LOI_SO_MODULE)", fontsize=10.5)
    save(fig, "09_crisis_gait_3008.png")
except Exception as e:
    print("09 FAIL:", e)

# ---------- 10: Timeline đổi ý tưởng ----------
try:
    events = [
        ("03/2026", "Gait + dashboard C#", 0.5, "up"),
        ("14/05", "+ rPPG / hành vi / RAG", 1.9, "down"),
        ("19–22/06", "Stroke Detect +\nkế hoạch 90 ngày", 3.5, "up"),
        ("06/07", "FAST đa cảm biến\n+ radar phòng tắm", 5.2, "down"),
        ("28/07", "FGA v3.0 —\n5 module", 6.9, "up"),
        ("19/08", "Đổi tên GOLDEN WATCH\n(8 ý kiến phê phán)", 8.5, "down"),
        ("25/08→12/09", "ML thật + TỰ BÓC 5 LEAKAGE", 9.9, "up"),
    ]
    fig, ax = plt.subplots(figsize=(12.6, 3.9))
    ax.axhline(0, color="#333", lw=1.4, zorder=1)
    for date, txt, x, side in events:
        up = side == "up"
        y = 0.58 if up else -0.58
        ax.plot([x, x], [0, y], color="#999", lw=0.9, zorder=2)
        ax.scatter([x], [0], s=52, color="#0275d8", zorder=3)
        ax.text(x, y + (0.1 if up else -0.1), f"{date}\n{txt}", ha="center", va="bottom" if up else "top", fontsize=7.6,
                bbox=dict(boxstyle="round,pad=0.28", fc="#f2f7fc" if up else "#fff7f2", ec="#bbb"))
    ax.set_xlim(-0.4, 10.9)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")
    ax.set_title("DÒNG THỜI GIAN 6 LẦN ĐỔI HƯỚNG — 03/2026 → 12/09/2026 (chi tiết: Phụ lục A)", fontsize=10.5)
    save(fig, "10_pivot_timeline.png")
except Exception as e:
    print("10 FAIL:", e)

print(f"OK {len(DONE)}/10 charts -> {OUT}")
for d in DONE:
    print(" -", d)
sys.exit(0)
