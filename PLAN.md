# Plan: Indic Speech → ASR → LLM → Text Response, with evaluation & ablation

## Context
Assignment (30 marks): build a reproducible Indic speech pipeline from open-source models, write our **own** evaluation infrastructure, evaluate on ≥100 examples, report ASR WER/CER, response correctness + hallucination rate, latency/RTF, performance across ≥2 conditions, and one research extension (find a measurable bottleneck, fix it, compare to baseline). Project folder `C:\Users\AARDRA\Desktop\vaaniTask` is empty. Decisions made: **Hindi + Malayalam**, run **locally on RTX 3060 (6 GB)**, Python 3.12.

## Task & data design (the key decision)
To score "response correctness" objectively we need gold answers, not just transcripts. Use **Google FLEURS** (`google/fleurs`, `hi_in` and `ml_in` test splits): every utterance has a native-script transcript **and** a parallel English gold sentence (FLoRes).

- **Task given to the LLM:** faithful **restatement in the same language and script** (Hindi→Hindi, Malayalam→Malayalam), fixing ASR errors without adding content. *(Changed from English output at the user's request.)* Responses are scored against the native transcript (surface: chrF++, response CER) and the English gold (meaning: cross-lingual LaBSE); the judge sees both. Response CER vs ASR CER also shows whether the LLM repairs ASR errors.
- **Sample:** 120 utterances per language (fixed seed, stratified by speaker gender), saved as a manifest CSV so runs are reproducible.
- **Conditions (≥2 required, we give two axes):**
  1. Linguistic: Hindi (Indo-Aryan) vs Malayalam (Dravidian, agglutinative, much lower-resource for Whisper).
  2. Acoustic: clean vs noisy — babble noise made by mixing other FLEURS utterances + white noise at **SNR 10 dB and 0 dB** (deterministic seed).
  → 240 utterances × 3 acoustic levels = 720 pipeline runs.

## Models (fit in 6 GB, run sequentially so VRAM isn't shared)
- **ASR baseline:** `openai/whisper-small` (multilingual) via HF `transformers`, language forced.
- **LLM:** `Qwen/Qwen3-4B-Instruct-2507` (119 languages incl. Malayalam; Qwen2.5-3B is weak at writing Malayalam) via HF `transformers` + bitsandbytes 4-bit (NF4), greedy decoding; TTFT timed at the first generated *token* (TextIteratorStreamer only emits at word boundaries, which misreports space-free output). Token cap = 30 tok/s × audio duration + 32 (refs peak at 22.3 tok/s), so loops can't run on and long answers aren't truncated. *(Changed from Ollama: the Ollama binary is not installed — only a stale `~/.ollama` folder — and a pure-pip stack is more reproducible.)*
- **Judge (eval only, separate pass):** `Qwen/Qwen2.5-7B-Instruct` 4-bit via transformers, run after ASR/LLM are unloaded.

## Project layout
```
vaaniTask/
  requirements.txt, README.md, REPORT.md
  configs/baseline.yaml, configs/improved.yaml   # model ids, langs, SNRs, seeds
  src/data.py        # load FLEURS (parquet branch – datasets>=3 dropped loading scripts), sample manifest, noise mixing @ SNR
  src/asr.py         # ASR wrapper interface: transcribe(audio, lang) -> text, timings
  src/llm.py         # transformers 4-bit LLM wrapper, prompt template, streaming to capture TTFT
  src/pipeline.py    # runs speech→ASR→LLM per item, records per-stage timing, writes results JSONL
  src/metrics/asr_metrics.py      # WER/CER (jiwer) with Indic normalisation
  src/metrics/response_metrics.py # chrF++ / LaBSE, off-script rate, LLM-judge, number/runaway hallucination check
  src/metrics/latency.py          # RTF, p50/p90, TTFT, tokens/s
  src/stats.py       # bootstrap 95% CIs, paired bootstrap for baseline vs improved
  scripts/prepare_data.py, run_pipeline.py, evaluate.py, make_report.py
  results/<run_name>/{predictions.jsonl, metrics.csv, figures/*.png}
```

## Evaluation definitions (our own infrastructure)
**ASR WER/CER** — reference = FLEURS raw transcription. Normalise both sides identically: Unicode NFC, `indic-nlp-library` script normaliser (nukta/chillu variants for Malayalam), strip punctuation incl. danda `।`, collapse whitespace. **Do not** use Whisper's `BasicTextNormalizer` (it deletes Indic combining marks and silently inflates/deflates scores — will note this in report). CER is primary for Malayalam (agglutination makes WER harsh).

**Response correctness** (per item, three signals):
- Automatic: chrF++ and response CER vs the native transcript; LaBSE cosine vs the English gold (cross-lingual); off-script rate (<80% of letters in the target script).
- LLM-judge rubric (JSON output): `correct` (all key content preserved), `partial`, `incorrect`.
- Correctness rate = % `correct`; also report partial-credit score (1/0.5/0).

**Hallucination rate** — % of responses containing ≥1 claim not supported by the *spoken content*. Two detectors:
- Deterministic: numbers / named entities in response absent from gold (digits normalised, Devanagari/Malayalam numerals mapped).
- Judge label `has_unsupported_content: true/false` with the offending span.
- Also separate **ASR-induced** vs **LLM-induced** hallucination via the oracle ablation below.

**Judge validation:** hand-label 40 random items myself; report agreement (Cohen's κ) with the judge so the metric is trustworthy.

**Latency** — `time.perf_counter` around each stage, `torch.cuda.synchronize()`, 3 warm-up items excluded. Report ASR time, LLM TTFT, LLM total, end-to-end; **RTF = processing time / audio duration** for ASR and end-to-end; p50/p90/mean.

All metrics with bootstrap 95% CIs, broken down by language × SNR.

## Ablations
1. **Oracle transcript:** feed gold transcript straight to the LLM → upper bound; gap = error due to ASR. This is how we *identify* the bottleneck.
2. **Noise levels:** clean / 10 dB / 0 dB.
3. **Language:** hi vs ml.
4. **Baseline vs improved** (research extension).

## Research extension (bottleneck → fix)
Expected bottleneck (to be confirmed by ablation 1): **Malayalam ASR** — whisper-small gives very high WER on ml, and the oracle gap will show it dominates response errors. Latency is a secondary bottleneck (HF Whisper RTF).

Improvement: **swap in language-adapted ASR** served via **faster-whisper (CTranslate2, fp16)**. Chosen checkpoints (after reading model cards):
- Hindi: `vasista22/whisper-hindi-small` — same size as baseline; trained on GramVaani/ULCA/Shrutilipi + FLEURS **train+dev** (FLEURS splits are sentence-disjoint, so no test leakage, but in-domain advantage — stated in report).
- Malayalam: `vrclc/Whisper-medium-Malayalam` — trained on IMaSC/MSC/OpenSLR/Festvox, **no FLEURS** (true held-out). `thennal/whisper-medium-ml` rejected: lists FLEURS in training data. No credible small Malayalam checkpoint exists, so a **size control** run (`openai/whisper-medium` on ml) separates adaptation gain from model-size gain.

Configs: `baseline.yaml`, `adapted_hf.yaml` (adapted weights, HF backend), `improved.yaml` (adapted + faster-whisper), `ml_medium_control.yaml`. One change targeting accuracy; the CTranslate2 backend also cuts RTF. To keep it a clean single-variable comparison, report two rows: (a) adapted model on HF backend, (b) + faster-whisper — so accuracy and speed gains are attributed separately.

Compare baseline vs improved on identical 720 items: ΔWER/CER, Δcorrectness, Δhallucination, ΔRTF, with paired-bootstrap significance.
(Fallback if IndicConformer/NeMo is preferred: NeMo is painful on Windows, so only via WSL — not planned.)

## Deliverables
- Working code + `README.md` (setup, one-command repro: `prepare_data → run_pipeline --config … → evaluate → make_report`).
- `results/` with per-item predictions and aggregated CSVs.
- `REPORT.md`: tables (ASR, correctness, hallucination, latency by condition), plots (WER vs SNR per language, latency breakdown bar, baseline vs improved), error analysis with ~5 qualitative examples, limitations (FLEURS read speech ≠ conversational; judge bias).

## Implementation order
1. venv + requirements (torch cu121, transformers, faster-whisper, jiwer, indic-nlp-library, sacrebleu, sentence-transformers, bitsandbytes, accelerate, pandas, matplotlib).
2. `prepare_data.py` → manifests + noisy audio.
3. ASR module + WER/CER metrics; sanity-check on 10 items.
4. LLM module + pipeline runner with timing.
5. Response metrics + judge + hand-label 40 for κ.
6. Full baseline run, oracle ablation → confirm bottleneck.
7. Improved ASR run, comparison, report.

## Verification
- Unit-check metrics: WER/CER on hand-crafted strings with known answers (incl. Malayalam chillu cases); normaliser idempotency.
- Smoke run: `run_pipeline.py --limit 5` end-to-end, inspect JSONL.
- Sanity: oracle-transcript WER = 0; clean WER < noisy WER; FLEURS-reported whisper-small Hindi WER ballpark matches ours.
- Full run reproducible: re-running with same config/seed gives identical WER and judge labels (temperature 0).


## Decisions log (implementation)
- **Judge memory:** bitsandbytes keeps `lm_head` in fp16 by default; for Qwen2.5-7B (152k vocab) that is ~1.1 GB and pushes peak VRAM to 5.67 GiB, spilling into Windows shared memory. Quantizing `lm_head` too (`quantize_lm_head: true`): 5.10 GiB, 15.9 → 1.6 s/item, identical labels on a 5-item check. Not applicable to Qwen3-4B (tied embeddings; peaks at 2.76 GiB anyway — its 12.4 tok/s is plain bnb NF4 decode speed).
- **LLM stage only where responses matter:** baseline, baseline_oracle, improved. `adapted_hf` and `ml_medium_control` are `--asr-only` (they attribute ASR accuracy/speed and model size; the LLM at ~12 tok/s would add ~5 h).
- **Degenerate ASR output** (repetition loops / runaway length) is its own metric; whisper-small loops on most Malayalam and Hindi 0 dB items.
- **Lexical hallucination = new numbers only.** Runaway length (`heur_runaway`) is reported separately: it is the LLM copying an ASR loop (correctness failure), not invented content.
- **Timing hygiene:** timed runs execute alone; smoke-test latency was ~4x inflated by concurrent downloads, so smoke latency numbers are not used.
- **Malayalam sample is 100% female** (FLEURS ml test speakers), so gender is not a usable condition.
