"""Pipeline architecture diagram -> results/figures/architecture.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BLUE, AQUA, VIOLET = "#2a78d6", "#1baf7a", "#6250d6"

fig, ax = plt.subplots(figsize=(11, 4.2))
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.05)
ax.axis("off")


def box(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=color, lw=2))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.68 - 0.3 * i, t, ha="center", va="top", fontsize=8.3, color=INK2)


def arrow(x0, x1, y, label=None):
    ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))
    if label:
        ax.text((x0 + x1) / 2, y + 0.12, label, ha="center", va="bottom", fontsize=7.6, color=MUTED)


y, h = 1.95, 1.95
box(0.1, y, 1.55, h, "Speech input", ["Hindi or Malayalam", "16 kHz mono", "(FLEURS / any file)"], INK2)
box(2.1, y, 1.85, h, "Language ID", ["stock whisper-medium", "<|hi|> vs <|ml|> only", "97.5% accurate"], VIOLET)
box(4.4, y, 2.15, h, "ASR", ["Baseline: whisper-small (HF)", "Improved: fine-tuned Whisper", "on faster-whisper", "(CTranslate2, fp16)"], BLUE)
box(7.0, y, 1.9, h, "Language model", ["Qwen3-4B-Instruct", "4-bit NF4, greedy", "same language/script"], AQUA)
box(9.3, y, 1.6, h, "Text response", ["restatement", "(understanding)", "reply", "(assistant answer)"], INK2)
arrow(1.65, 2.1, y + h / 2)
arrow(3.95, 4.4, y + h / 2)
arrow(6.55, 7.0, y + h / 2)
arrow(8.9, 9.3, y + h / 2)

ax.add_patch(FancyBboxPatch((0.1, 0.15), 10.8, 1.45, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc="#f3f2ee", ec="#e1e0d9", lw=1))
ax.text(5.5, 1.42, "Evaluation infrastructure (our own code)", ha="center", va="top", fontsize=10.5,
        fontweight="bold", color=INK)
ev = [("ASR", "WER / CER (Indic-safe\nnormalisation), loop rate"),
      ("Responses", "LLM judge rubric,\nlexical check, LaBSE, CER"),
      ("Native speaker", "blind ratings: quality,\nrelevance, understandability"),
      ("Latency", "per stage, TTFT,\nreal-time factor"),
      ("Statistics", "bootstrap 95% CIs,\npaired tests, oracle")]
for i, (t, d) in enumerate(ev):
    cx = 1.2 + i * 2.15
    ax.text(cx, 1.0, t, ha="center", va="top", fontsize=8.8, fontweight="bold", color=INK2)
    ax.text(cx, 0.72, d, ha="center", va="top", fontsize=7.6, color=INK2, linespacing=1.25)

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "results" / "figures" / "architecture.png"
fig.savefig(out, dpi=200)
print(out)
