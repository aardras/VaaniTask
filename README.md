# vaaniTask: Indic speech → ASR → LLM → text response, with evaluation

A reproducible pipeline that takes **Hindi or Malayalam speech** and returns a **text response in the same language and script**, built from open-source pretrained models, with our own evaluation infrastructure:

- ASR WER/CER
- response correctness and hallucination under written criteria
- latency and real-time factor
- results across 2 languages × 3 noise levels
- a baseline-vs-improved research extension with ablations
- a blind native-speaker evaluation

| Document | Contents |
|---|---|
| [REPORT.md](REPORT.md) | Full technical report: methods, every result with 95% CIs, limitations |
| `VaaniTask2_Aardra.docx` / `.pdf` | Submission report (10 pages) |
| [results/summary.md](results/summary.md) | Auto-generated tables for every run |
| [PLAN.md](PLAN.md) | Design log: every decision and why it changed |

---

## Results at a glance

Baseline = stock `openai/whisper-small`. Improved = language-adapted Whisper on faster-whisper. Both feed the same LLM. All differences shown are significant (paired bootstrap, p ≤ 0.002).

| Measure | Baseline | Improved |
|---|---|---|
| ASR CER, Hindi (all noise levels) | 101.2% | **34.8%** |
| ASR CER, Malayalam (all noise levels) | 139.0% | **27.9%** |
| Restatement judged correct, Hindi / Malayalam | 18.9% / 6.4% | **44.4% / 67.2%** |
| Reply judged acceptable, clean Hindi / Malayalam | 32.5% / 7.5% | **77.5% / 57.5%** |
| Native speaker (blind, n = 48): good response / on-topic | 12% / 12% | **42% / 46%** |
| ASR real-time factor (mean), Hindi / Malayalam | 0.239 / 0.234 | **0.042 / 0.169** |
| End-to-end time per utterance (p50), Hindi / Malayalam | 10.98 s / 28.35 s | **8.25 s / 16.97 s** |
| Language-ID accuracy (720 items) | — | **97.5%** (100% on clean speech) |

**ASR CER by condition (n = 120 each):**

| | Hindi clean | Hindi 10 dB | Hindi 0 dB | Malayalam clean | Malayalam 10 dB | Malayalam 0 dB |
|---|---|---|---|---|---|---|
| Baseline | 45.9% | 75.4% | 182.3% | 160.9% | 121.7% | 134.5% |
| Improved | 5.8% | 16.2% | 82.3% | 12.8% | 18.2% | 52.8% |

- **Bottleneck:** with the gold transcript, 99.2% of LLM restatements are correct, so the baseline's errors come almost entirely from ASR.
- **Size control:** stock whisper-medium only reaches 119.0% Malayalam CER, so the gain comes from language adaptation, not model size.
- **Engine:** faster-whisper is 2.7–3.3× faster than the HF engine on the same weights.
- **What's left:** the LLM now dominates latency (85–95% of end-to-end time), and 0 dB speech remains hard.

These are the numbers to compare against when reproducing. Quality metrics should match closely, because decoding is greedy and sampling uses fixed seeds. Latency depends on hardware.

---

## 1. Setup

**Tested on:** Windows 11, Python 3.12.4, NVIDIA RTX 3060 Laptop (6 GB VRAM), CUDA 12.1, 16 GB RAM. About 30 GB of disk for models and 1.4 GB for data.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt   # exact versions used (or requirements.txt for ranges)
```

**Hugging Face access:**
1. `vrclc/Whisper-medium-Malayalam` is **gated**. Accept its terms on its model page while logged in.
2. Log in from your own terminal, so the token is never pasted into a shared log:
   ```powershell
   .\.venv\Scripts\huggingface-cli.exe login
   ```
3. Models download on first use: Qwen3-4B (~8 GB), Qwen2.5-7B (~15 GB), whisper-small/medium, LaBSE, and the two fine-tuned checkpoints.

**Check the evaluation code** (19 known-answer tests):
```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

### Version pins that matter
| Pin | Why |
|---|---|
| `torch==2.5.1+cu121` | Every run was timed on it. Don't upgrade mid-project, or latencies stop being comparable. |
| `transformers<5` (4.57.6) | transformers 5.x needs torch ≥ 2.6. |
| `datasets<4`, `sentence-transformers<5` | Newer versions need huggingface_hub 1.x, which conflicts with transformers 4.x. |
| `ctranslate2==4.8.4` | Has the Whisper `max_length/2` behaviour that `src/asr.py` corrects for. |

---

## 2. Try it on one recording

```powershell
.\.venv\Scripts\python.exe scripts/demo.py my_recording.wav               # auto-detects Hindi vs Malayalam, prints transcript + reply
.\.venv\Scripts\python.exe scripts/demo.py my_recording.wav --task both    # reply + restatement
.\.venv\Scripts\python.exe scripts/demo.py my_recording.wav --lang ml_in   # skip language detection
```
- **Input:** any format librosa reads. It's resampled to 16 kHz mono, and only the first 30 s are transcribed.
- **Output:** the detected language with its probability, the transcript and ASR real-time factor, then the response with time to first token and total time.
- **Speed:** models load one at a time, so a first run takes ~1–2 minutes including model loading.

---

## 3. Reproduce every result

Run **one job at a time**, and keep the machine otherwise idle:
- Close heavy apps.
- Disable sleep: `powercfg /change standby-timeout-ac 0`.
- Stay on AC power.

The HF Whisper decode is bound by CPU overhead, so concurrent jobs inflate latency. Every stage saves results item by item and **resumes** where it stopped, so re-running a command after an interruption is always safe.

### Step 1: data (~15 min, downloads FLEURS)
```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py --config configs/baseline.yaml
```
This writes `data/manifest.csv`: 720 rows, 240 utterances × {clean, 10 dB, 0 dB}. Always run it with `baseline.yaml`, because that config defines the shared manifest.

### Step 2: restatement experiments (~10 h total)
```bash
bash scripts/run_queue.sh 2>&1 | tee logs/queue.log
```
The queue runs these steps in order and stops at the first failure; re-running resumes.

| Step | What it does | Time on RTX 3060 |
|---|---|---|
| `baseline` | whisper-small (HF) ASR + Qwen3-4B restatement, 720 items | ~1 h ASR + ~3.5 h LLM |
| `eval_baseline` | metrics + LaBSE + 7B judge | ~30 min |
| `oracle` | gold transcript → LLM, 240 items | ~45 min |
| `adapted_hf` | fine-tuned ASR on the HF engine, ASR only | ~50 min |
| `improved` | fine-tuned ASR on faster-whisper + LLM | ~15 min ASR + ~3 h LLM |
| `ml_medium_control` | stock whisper-medium, Malayalam, ASR only | ~50 min |
| `report` | `results/summary.md` + `results/figures/` | seconds |

Each `eval_*` step writes `results/<run>/scored.jsonl` (per item) and `metrics.csv` (aggregates with CIs).

### Step 3: language identification (~3 min)
```powershell
.\.venv\Scripts\python.exe scripts/eval_lid.py      # -> results/lid/metrics.csv, lid.jsonl
```

### Step 4: reply mode (~2.5 h)
```bash
bash scripts/run_reply_queue.sh 2>&1 | tee logs/reply_queue.log
```
- **Subset:** 40 utterances per language × {clean, 0 dB}, 160 items, fixed seed. The oracle uses 80 items.
- **Runs:** baseline, improved and oracle.
- **ASR is reused** from the step 2 runs, so only the LLM and judge run.

### Step 5: native-speaker evaluation (optional, ~40 min of rating)
```powershell
.\.venv\Scripts\python.exe scripts/human_eval.py export    # results/human_eval/ratings.csv + GUIDE.md + key.csv
#   rate in Excel following GUIDE.md, then Save As "CSV UTF-8"
.\.venv\Scripts\python.exe scripts/human_eval.py analyze   # results/human_eval/results.md
```
- **Sample:** 48 blind, paired replies, with system identity hidden. Don't open `key.csv` while rating.
- **Output:** scores per system and condition, paired differences, and Cohen's κ against the LLM judge.

### Step 6: tables and figures
```powershell
.\.venv\Scripts\python.exe scripts/make_report.py          # results/summary.md + results/figures/*.png
```

### Re-scoring without re-running
`scripts/evaluate.py` and `scripts/evaluate_reply.py` can be re-run at any time. Judge verdicts are cached, and the reply-judge cache is keyed by a hash of the judge prompt, so editing the rubric re-judges automatically but never reuses stale verdicts.

---

## 4. Implementation details

Everything that affects the numbers.

### Data and conditions (`src/data.py`)
| Setting | Value |
|---|---|
| Dataset | FLEURS **test** split, `hi_in` and `ml_in`, raw TSV + audio from the `google/fleurs` HF repo |
| English gold | `en_us` transcript of the same FLoRes sentence id |
| Sample | 120 distinct sentences per language, seed 1234, gender-stratified (Hindi 60/60; Malayalam test speakers are all female) |
| Noise | Babble (sum of 4 other utterances from the same language, unused in the sample) + 30% white noise, unit RMS |
| SNR | clean, 10 dB, 0 dB. The same noise sample is used at both levels, so only loudness changes. Measured: 10.00 / 0.00 dB |

### Models and decoding
| Component | Model | Settings |
|---|---|---|
| Language ID (`src/lid.py`) | `openai/whisper-medium` | One decoder step after `<|startoftranscript|>`; softmax over `<|hi|>`, `<|ml|>` only |
| ASR baseline | `openai/whisper-small` | HF transformers, fp16, greedy (`num_beams=1`), language token forced, `<|notimestamps|>`, 440 new tokens max, one 30 s window |
| ASR improved, Hindi | `vasista22/whisper-hindi-small@e760aba89d31418e3c99068d1711911bbbd489b5` | `refs/pr/5`: HF's safetensors conversion, identical to `main` except for the added `model.safetensors`. Its outdated generation config is replaced by `openai/whisper-small`'s. |
| ASR improved, Malayalam | `vrclc/Whisper-medium-Malayalam` | Never trained on FLEURS |
| Improved engine | faster-whisper / CTranslate2 fp16 | CTranslate2 `generate()` called directly, `max_length = 2 × 440`; same greedy / no-timestamps / 30 s window as HF |
| LLM | `Qwen/Qwen3-4B-Instruct-2507` | bitsandbytes 4-bit NF4, fp16 compute, greedy. Restatement cap: 30 tokens per second of audio + 32 (max 768). Reply cap: 320 tokens. |
| LLM judge | `Qwen/Qwen2.5-7B-Instruct` | 4-bit NF4 **including `lm_head`** (needed to fit 6 GB), greedy |
| Meaning similarity | `sentence-transformers/LaBSE` | Cosine of normalised embeddings |

The prompts are in `src/llm.py` (restatement and reply) and `src/metrics/response_metrics.py` (both judges).

### Metrics (`src/metrics/`, `src/stats.py`)
| Metric | Definition |
|---|---|
| WER / CER | Corpus-level (errors summed ÷ reference length). Both sides: Unicode NFC → indic-nlp normaliser (nuktas kept, chillu forms unified) → punctuation incl. `।` stripped → whitespace collapsed. Whisper's `BasicTextNormalizer` is **not** used (it deletes Indic vowel signs). |
| Degenerate ASR output | >50% repeated words, ≥20 chars from ≤3 distinct characters, or >2× reference length |
| Restatement correctness | Judge: correct / partial / incorrect against the native transcript + English meaning |
| Restatement hallucination | Judge flag (a specific detail not said); lexical cross-check (new numbers; Devanagari/Malayalam digits mapped) |
| Reply rubric | Relevance (on/partial/off-topic), misattribution (quote-or-false), factual hallucination (correct added knowledge is fine), asks-to-repeat; **acceptable** = on-topic ∧ no misattribution ∧ no factual hallucination ∧ in-script |
| Off-script | < 80% of letters in the target Unicode block |
| Latency | `perf_counter` + `torch.cuda.synchronize()`; time to first token measured at the first generated **token**; 3 warm-ups per language × SNR excluded; RTF = processing time ÷ audio duration |
| Statistics | Utterance-level bootstrap, 2,000 resamples, 95% CI; paired bootstrap for baseline vs improved |
| Judge validation | Cohen's κ, human vs judge, overall and per language |

### Seeds
| What | Seed |
|---|---|
| Data sample and noise | 1234 |
| Reply subset | 1234 (config `seed`) |
| Human-eval sample | 7 |
| Bootstrap | 0 |

Decoding is greedy everywhere, and reruns produced byte-identical outputs.

---

## 5. Layout

| Path | Purpose |
|---|---|
| `src/data.py` | FLEURS download, stratified sampling, noise mixing at a target SNR |
| `src/asr.py` | `HFWhisperASR` and `FasterWhisperASR` behind one `transcribe()`; pinned revisions (`repo@commit`) |
| `src/lid.py` | Spoken language ID restricted to Hindi/Malayalam |
| `src/llm.py` | 4-bit LLM wrapper, restatement and reply prompts, token-level TTFT, token caps |
| `src/pipeline.py` | Two-stage resumable runner; oracle mode; ASR reuse for reply mode |
| `src/metrics/asr_metrics.py` | Indic normalisation, WER/CER, degenerate-output detection |
| `src/metrics/response_metrics.py` | chrF++, LaBSE, off-script, lexical hallucination, both judge rubrics and parsers, Cohen's κ |
| `src/metrics/latency.py`, `src/stats.py` | Latency/RTF summaries; bootstrap CIs and paired tests |
| `configs/*.yaml` | One config per run: `baseline`, `adapted_hf`, `improved`, `ml_medium_control` |
| `scripts/run_queue.sh`, `scripts/run_reply_queue.sh` | Full experiment queues (sequential, resumable) |
| `scripts/evaluate.py`, `scripts/evaluate_reply.py` | Scoring for restatement / reply runs |
| `scripts/eval_lid.py`, `scripts/human_eval.py`, `scripts/label_sample.py` | Language-ID evaluation; blind human evaluation; judge hand-label sheets |
| `scripts/make_report.py`, `scripts/demo.py` | Tables and figures; single-recording demo |
| `report/` | Builds the .docx report (`node build_docx.js`) and the architecture figure; its `node_modules/` and `pylib/` are build tools only |
| `tests/test_metrics.py` | 19 known-answer tests for the evaluation code |
| `results/<run>/` | `asr.jsonl`, `llm.jsonl`, `predictions.jsonl`, `scored.jsonl`, `metrics.csv` per run |
