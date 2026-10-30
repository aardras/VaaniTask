"""Tables + figures across runs -> results/summary.md, results/figures/*.png

    python scripts/make_report.py
Expects results/<run>/{scored.jsonl,metrics.csv} for whichever runs exist.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics.latency import add_rtf  # noqa: E402
from src.stats import paired_diff_ci  # noqa: E402

RES = Path("results")
FIG = RES / "figures"
# Fixed categorical order (validated: dataviz validator, light surface). Color follows the run.
RUNS = {"baseline": ("Baseline (whisper-small, HF)", "#2a78d6"),
        "adapted_hf": ("Adapted ASR (HF)", "#eb6834"),
        "improved": ("Adapted ASR + faster-whisper", "#1baf7a"),
        "ml_medium_control": ("Size control (whisper-medium)", "#eda100")}
LANG = {"hi_in": "Hindi", "ml_in": "Malayalam"}
SNR_ORDER = ["clean", "10", "0"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def load(run: str):
    p = RES / run
    if not (p / "scored.jsonl").exists():
        return None, None
    return (pd.read_json(p / "scored.jsonl", lines=True, dtype={"snr": str}),
            pd.read_csv(p / "metrics.csv", dtype={"snr": str}))


def pct(v, lo=None, hi=None):
    if v is None or pd.isna(v):
        return "—"  # metric not computed for this run (e.g. --asr-only)
    return f"{100 * v:.1f}" + (f" [{100 * lo:.1f}, {100 * hi:.1f}]" if lo is not None else "")


def condition_table(m: pd.DataFrame) -> str:
    rows = ["| Lang | SNR | n | WER % | CER % | Degenerate % | Response CER % | Off-script % | Correct % "
            "| Halluc. (judge) % | Halluc. (lexical) % | chrF++ | LaBSE (vs EN) | ASR RTF | E2E p50 s |",
            "|" + "---|" * 15]
    for r in m[m.lang != "ALL"].itertuples():
        rows.append(f"| {LANG.get(r.lang, r.lang)} | {r.snr} | {r.n} | {pct(r.wer, r.wer_lo, r.wer_hi)} "
                    f"| {pct(r.cer, r.cer_lo, r.cer_hi)} | {pct(r.degenerate)} | {pct(r.resp_cer, r.resp_cer_lo, r.resp_cer_hi)} "
                    f"| {pct(r.off_script)} | {pct(r.judge_correct, r.judge_correct_lo, r.judge_correct_hi)} "
                    f"| {pct(r.judge_halluc, r.judge_halluc_lo, r.judge_halluc_hi)} | {pct(r.heur_halluc)} "
                    f"| {r.chrf:.1f} | {r.labse:.3f} | {r.asr_rtf_mean:.3f} | {r.e2e_s_p50:.2f} |")
    return "\n".join(rows)


def run_table(metrics: dict) -> str:
    rows = ["| Run | Lang | WER % | CER % | Resp. CER % | Correct % | Halluc. % | ASR RTF | ASR s (mean) | LLM s (mean) | E2E RTF |",
            "|" + "---|" * 11]
    for run, m in metrics.items():
        for r in m[(m.snr == "ALL") & (m.lang != "ALL")].itertuples():
            rows.append(f"| {RUNS.get(run, (run,))[0]} | {LANG[r.lang]} | {pct(r.wer)} | {pct(r.cer)} | {pct(getattr(r, 'resp_cer', None))} "
                        f"| {pct(getattr(r, 'judge_correct', None))} | {pct(getattr(r, 'judge_halluc', None))} | {r.asr_rtf_mean:.3f} "
                        f"| {r.asr_s_mean:.2f} | {r.llm_s_mean:.2f} | {r.e2e_rtf_mean:.3f} |")
    return "\n".join(rows)


def paired_table(base: pd.DataFrame, other: pd.DataFrame, name: str) -> str:
    """Same items in both runs; improved - baseline, with paired-bootstrap 95% CI and p."""
    j = base.merge(other, on="item_id", suffixes=("_a", "_b"))
    rows = [f"**{name} − Baseline** (paired bootstrap, 2000 resamples; negative = fewer errors)\n",
            "| Lang | ΔWER pts | ΔCER pts | ΔCorrect pts | ΔHalluc. pts | ΔASR RTF |", "|" + "---|" * 6]
    for lang, g in j.groupby("lang_a"):
        t = g[~(g.is_warmup_a | g.is_warmup_b)]
        cells = []
        for a, b, den in [("word_errors_a", "word_errors_b", g.ref_words_a),
                          ("char_errors_a", "char_errors_b", g.ref_chars_a),
                          ("judge_correct_a", "judge_correct_b", None),
                          ("judge_halluc_a", "judge_halluc_b", None)]:
            if a not in g or b not in g:  # judge columns are absent in --asr-only runs
                cells.append("—")
                continue
            d = paired_diff_ci(g[a], g[b], den)
            cells.append(f"{100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] p={d['p']:.3f}")
        d = paired_diff_ci(t.asr_s_a / t.duration_s_a, t.asr_s_b / t.duration_s_b)
        cells.append(f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
        rows.append(f"| {LANG[lang]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def attribution(base: pd.DataFrame, oracle: pd.DataFrame) -> str:
    """Of baseline failures, how many disappear when the LLM gets the gold transcript?"""
    o = oracle[["utt_id", "judge_label", "judge_halluc"]].rename(
        columns={"judge_label": "o_label", "judge_halluc": "o_halluc"})
    j = base.merge(o, on="utt_id")
    rows = ["| Lang | SNR | Correct % (ASR) | Correct % (oracle) | Halluc. % (ASR) | Halluc. % (oracle) "
            "| Failures caused by ASR % | Halluc. caused by ASR % |", "|" + "---|" * 8]
    for (lang, snr), g in sorted(j.groupby(["lang", "snr"]), key=lambda x: (x[0][0], SNR_ORDER.index(x[0][1]))):
        fail = g[~g.judge_correct]
        hal = g[g.judge_halluc]
        asr_fail = (fail.o_label == "correct").mean() if len(fail) else np.nan
        asr_hal = (~hal.o_halluc).mean() if len(hal) else np.nan
        rows.append(f"| {LANG[lang]} | {snr} | {pct(g.judge_correct.mean())} | {pct((g.o_label == 'correct').mean())} "
                    f"| {pct(g.judge_halluc.mean())} | {pct(g.o_halluc.mean())} | {pct(asr_fail)} | {pct(asr_hal)} |")
    return "\n".join(rows)


def fig_cer_vs_snr(metrics: dict):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), sharey=False)
    handles = {}
    for ax, lang in zip(axes, LANG):
        placed = []  # end-label y positions, to skip labels that would collide
        for run, m in metrics.items():
            g = m[(m.lang == lang) & (m.snr != "ALL")].set_index("snr").reindex(SNR_ORDER).dropna(subset=["cer"])
            if g.empty:
                continue
            x = [SNR_ORDER.index(s) for s in g.index]
            (h,) = ax.plot(x, 100 * g.cer, color=RUNS[run][1], lw=2, marker="o", ms=7,
                           mec=SURF, mew=2, label=RUNS[run][0])
            handles[run] = h
            ax.fill_between(x, 100 * g.cer_lo, 100 * g.cer_hi, color=RUNS[run][1], alpha=0.12, lw=0)
            end = 100 * g.cer.iloc[-1]
            if all(abs(end - p) > 8 for p in placed):
                ax.annotate(f"{end:.0f}", (x[-1], end), xytext=(6, 0),
                            textcoords="offset points", va="center", color=INK2, fontsize=9)
                placed.append(end)
        ax.set_xticks(range(3), ["clean", "10 dB", "0 dB"])
        ax.set_title(LANG[lang], loc="left")
        ax.set_ylabel("CER % (95% CI)")
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.2, 2.35)
    # One figure-level legend so runs that appear in only one panel (the size control) are listed.
    fig.legend(list(handles.values()), [RUNS[r][0] for r in handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.93), ncol=2, fontsize=8.5)
    fig.suptitle("Character error rate rises with noise", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(FIG / "cer_vs_snr.png", dpi=160)
    plt.close(fig)


def fig_bars(metrics: dict, col: str, title: str, fname: str):
    runs = list(metrics)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.8 / len(runs)
    for i, run in enumerate(runs):
        m = metrics[run]
        if col not in m:
            continue
        for k, lang in enumerate(LANG):
            r = m[(m.lang == lang) & (m.snr == "ALL")]
            if r.empty:
                continue
            v, lo, hi = (100 * r[c].iloc[0] for c in [col, f"{col}_lo", f"{col}_hi"])
            x = k + (i - (len(runs) - 1) / 2) * w
            ax.bar(x, v, w * 0.92, color=RUNS[run][1], label=RUNS[run][0] if k == 0 else None)
            ax.errorbar(x, v, [[v - lo], [hi - v]], color=INK2, lw=1, capsize=0)
            ax.annotate(f"{v:.0f}", (x, hi), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(LANG)), list(LANG.values()))
    ax.set_ylabel("% of responses (95% CI)")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


def fig_latency(scored: dict):
    labels, asr, llm = [], [], []
    for run, df in scored.items():
        d = add_rtf(df[~df.is_warmup])
        for lang in LANG:
            g = d[d.lang == lang]
            if len(g):
                labels.append(f"{RUNS[run][0]} · {LANG[lang]}")
                asr.append(g.asr_s.mean())
                llm.append(g.llm_s.mean())
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(labels) + 1.2))
    ax.barh(y, asr, 0.6, color="#256abf", label="ASR")
    ax.barh(y, llm, 0.6, left=np.array(asr) + 0.02, color="#86b6ef", label="LLM")  # 2px-style gap
    for yi, a, l in zip(y, asr, llm):
        text = f"{a:.2f} + {l:.2f} s" if l > 0 else f"{a:.2f} s (ASR only)"
        ax.annotate(text, (a + l + 0.02, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlabel("Mean seconds per utterance (warm-ups excluded)")
    ax.set_title("Where the time goes: ASR vs LLM", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=max(a + l for a, l in zip(asr, llm)) * 1.3)
    fig.tight_layout()
    fig.savefig(FIG / "latency_breakdown.png", dpi=160)
    plt.close(fig)


REPLY_RUNS = {"baseline_reply": ("Baseline ASR", "#2a78d6"),
              "improved_reply": ("Improved ASR", "#1baf7a"),
              "baseline_oracle_reply": ("Gold transcript (oracle)", "#6250d6")}  # validated trio
REPLY_COLS = [("acceptable", "Acceptable"), ("on_topic", "On-topic"), ("rj_misattribution", "Misattribution"),
              ("rj_factual_halluc", "Factual halluc."), ("rj_clarification", "Asks to repeat"),
              ("off_script", "Off-script")]


def reply_table(reply: dict) -> str:
    rows = ["| Run | Lang | SNR | n | ASR CER % | " + " | ".join(f"{h} %" for _, h in REPLY_COLS)
            + " | LaBSE | LLM p50 s |", "|" + "---|" * (7 + len(REPLY_COLS))]
    for run, m in reply.items():
        for r in m[m.lang != "ALL"].itertuples():
            cells = [pct(getattr(r, c), getattr(r, f"{c}_lo"), getattr(r, f"{c}_hi")) if c == "acceptable"
                     else pct(getattr(r, c)) for c, _ in REPLY_COLS]
            llm = getattr(r, "llm_s_p50", float("nan"))
            rows.append(f"| {REPLY_RUNS[run][0]} | {LANG.get(r.lang, r.lang)} | {r.snr} | {r.n} | {pct(r.cer)} | "
                        + " | ".join(cells) + f" | {r.labse_vs_said:.3f} | {llm:.2f} |")
    return "\n".join(rows)


def reply_paired(a: pd.DataFrame, b: pd.DataFrame) -> str:
    j = a.merge(b, on="item_id", suffixes=("_a", "_b"))
    rows = ["**Improved − Baseline, reply mode** (paired bootstrap on the same items; positive = more often)\n",
            "| Lang | SNR | " + " | ".join(f"Δ{h} pts" for _, h in REPLY_COLS[:5]) + " |", "|" + "---|" * 7]
    for (lang, snr), g in sorted(j.groupby(["lang_a", "snr_a"]), key=lambda x: (x[0][0], SNR_ORDER.index(x[0][1]))):
        cells = []
        for c, _ in REPLY_COLS[:5]:
            d = paired_diff_ci(g[f"{c}_a"].astype(float), g[f"{c}_b"].astype(float))
            cells.append(f"{100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] p={d['p']:.3f}")
        rows.append(f"| {LANG[lang]} | {snr} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def fig_reply(reply: dict):
    conds = [(l, s) for l in LANG for s in ["clean", "0"]]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.8 / len(reply)
    for i, (run, m) in enumerate(reply.items()):
        for k, (lang, snr) in enumerate(conds):
            r = m[(m.lang == lang) & (m.snr == snr)]
            if r.empty:
                continue
            v, lo, hi = (100 * r[c].iloc[0] for c in ["acceptable", "acceptable_lo", "acceptable_hi"])
            x = k + (i - (len(reply) - 1) / 2) * w
            ax.bar(x, v, w * 0.92, color=REPLY_RUNS[run][1], label=REPLY_RUNS[run][0] if k == 0 else None)
            ax.errorbar(x, v, [[v - lo], [hi - v]], color=INK2, lw=1, capsize=0)
            ax.annotate(f"{v:.0f}", (x, hi), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(conds)), [f"{LANG[l]}\n{'clean' if s == 'clean' else '0 dB'}" for l, s in conds])
    ax.set_ylabel("% acceptable replies (95% CI)")
    ax.set_title("Reply mode: acceptable replies (on-topic, faithful, factual, right language)", loc="left")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "reply_acceptable.png", dpi=160)
    plt.close(fig)


def lid_table() -> str:
    p = RES / "lid" / "metrics.csv"
    if not p.exists():
        return ""
    m = pd.read_csv(p, dtype={"snr": str})
    rows = ["| Lang | SNR | n | Accuracy % | Mean p(true lang) | LID s (p50) |", "|" + "---|" * 6]
    for r in m.itertuples():
        rows.append(f"| {LANG.get(r.lang, r.lang)} | {r.snr} | {r.n} | {pct(r.accuracy, r.acc_lo, r.acc_hi)} "
                    f"| {r.mean_p_true:.3f} | {r.lid_s_p50:.3f} |")
    return "\n".join(rows)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    scored, metrics = {}, {}
    for run in RUNS:
        s, m = load(run)
        if s is not None:
            scored[run], metrics[run] = s, m
    oracle, _ = load("baseline_oracle")
    out = ["# Results summary (auto-generated by scripts/make_report.py)\n"]
    if "baseline" in metrics:
        out += ["## Baseline by condition\n", condition_table(metrics["baseline"]), ""]
    out += ["## All runs (pooled over SNR)\n", run_table(metrics), ""]
    if oracle is not None and "baseline" in scored:
        out += ["## Error attribution: ASR vs LLM (oracle-transcript ablation)\n",
                attribution(scored["baseline"], oracle), ""]
    for run in ["adapted_hf", "improved", "ml_medium_control"]:
        if run in scored and "baseline" in scored:
            out += [paired_table(scored["baseline"], scored[run], RUNS[run][0]), ""]
    if "ml_medium_control" in scored and "improved" in scored:
        out += [paired_table(scored["ml_medium_control"], scored["improved"],
                             "Adapted + faster-whisper").replace("− Baseline", "− Size control (whisper-medium)"), ""]
    reply_scored, reply_metrics = {}, {}
    for run in REPLY_RUNS:
        s, m = load(run)
        if s is not None:
            reply_scored[run], reply_metrics[run] = s, m
    if reply_metrics:
        out += ["## Reply mode (grounded rubric; 40 utterances per language x clean / 0 dB)\n",
                reply_table(reply_metrics), ""]
        if {"baseline_reply", "improved_reply"} <= set(reply_scored):
            out += [reply_paired(reply_scored["baseline_reply"], reply_scored["improved_reply"]), ""]
        fig_reply(reply_metrics)
    if lid_table():
        out += ["## Automatic language identification (stock whisper-medium, hi vs ml)\n", lid_table(), ""]
    fig_cer_vs_snr(metrics)
    fig_bars(metrics, "judge_correct", "Responses judged correct", "correctness.png")
    fig_bars(metrics, "judge_halluc", "Responses with hallucinated content", "hallucination.png")
    fig_latency(scored)
    (RES / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print((RES / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
