// Builds Report_Indic_Speech_Pipeline.docx from the verified results (numbers mirror REPORT.md).
//   cd report && node build_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, Footer, PageNumber, LevelFormat,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "results", "figures");
const OUT = path.join(ROOT, "Report_Indic_Speech_Pipeline.docx");
const W = 9026; // A4 content width in DXA with 1" margins
const FONT = "Calibri";

// ---------- helpers ----------
const run = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 22, bold: o.bold, italics: o.italics, color: o.color });

// "**bold** plain" mini-markup -> runs
function rich(text, o = {}) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((s) =>
    s.startsWith("**") ? run(s.slice(2, -2), { ...o, bold: true }) : run(s, o));
}
const P = (text, o = {}) => new Paragraph({ children: rich(text, o), spacing: { after: 120, line: 276 }, alignment: o.align });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [run(text, { size: 30, bold: true, color: "1F3864" })], spacing: { before: 320, after: 140 }, keepNext: true });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [run(text, { size: 25, bold: true, color: "2E5597" })], spacing: { before: 220, after: 100 }, keepNext: true });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: rich(text), spacing: { after: 70, line: 264 } });
const N = (text) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: rich(text), spacing: { after: 70, line: 264 } });
const caption = (text) => new Paragraph({ children: [run(text, { size: 18, italics: true, color: "555555" })], spacing: { before: 60, after: 200 }, alignment: AlignmentType.CENTER });

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, width, o = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA }, borders,
    shading: o.header ? { type: ShadingType.CLEAR, color: "auto", fill: "DCE6F2" } : (o.fill ? { type: ShadingType.CLEAR, color: "auto", fill: o.fill } : undefined),
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ children: rich(String(text), { size: 19, bold: o.header }), alignment: o.align, keepNext: true })],  // keep tables on one page
  });
}
// widths: relative weights; scaled to the full content width (sums exactly to W)
function table(headers, rows, weights, o = {}) {
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((w / total) * W));
  widths[widths.length - 1] += W - widths.reduce((a, b) => a + b, 0);
  const numAlign = (i) => (i > 0 && o.rightAlignNumbers ? AlignmentType.CENTER : undefined);
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((h, i) => cell(h, widths[i], { header: true, align: numAlign(i) })) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, widths[i], { align: numAlign(i), fill: (o.highlight || []).includes(ri) ? "EEF6F0" : undefined })) })),
    ],
  });
}
const gap = () => new Paragraph({ children: [], spacing: { after: 80 } });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(name, cap, widthPx = 600) {
  const f = path.join(FIG, name);
  const { w, h } = pngSize(f);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(f), transformation: { width: widthPx, height: Math.round((widthPx * h) / w) },
        altText: { title: cap, description: cap, name } })] }),
    caption(cap),
  ];
}

// ---------- content ----------
const children = [];

// Title block
children.push(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 },
    children: [run("Indic Speech-to-Response Pipeline", { size: 44, bold: true, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [run("Evaluation and an ASR-bottleneck improvement for Hindi and Malayalam", { size: 28, color: "2E5597" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 },
    children: [run("Speech Input → Speech Encoder / ASR → Language Model → Text Response", { size: 22, italics: true, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 },
    children: [run("30 October 2026", { size: 22, color: "555555" })] }),
);

// 1. Summary
children.push(H1("1. Summary"));
children.push(P("We built a reproducible pipeline that takes **Hindi or Malayalam speech** and returns a **text response in the same language and script**, using open-source pretrained models, and wrote our own evaluation infrastructure around it. The system was evaluated on **720 test items** (240 utterances × 3 acoustic conditions), plus **400 assistant replies**, **720 language-ID decisions** and **48 blind native-speaker ratings**."));
children.push(table(
  ["Measure", "Baseline", "Improved", "Change"],
  [
    ["ASR CER, Hindi (all conditions)", "101.2%", "34.8%", "−66.4 pts"],
    ["ASR CER, Malayalam (all conditions)", "139.0%", "27.9%", "−111.1 pts"],
    ["Restatement judged correct, Hindi / Malayalam", "18.9% / 6.4%", "44.4% / 67.2%", "+25.6 / +60.8 pts"],
    ["Reply judged acceptable, clean Hindi / Malayalam", "32.5% / 7.5%", "77.5% / 57.5%", "+45.0 / +50.0 pts"],
    ["Native speaker: good response / on-topic", "12% / 12%", "42% / 46%", "+29 / +33 pts"],
    ["ASR real-time factor, Hindi / Malayalam (mean)", "0.239 / 0.234", "0.042 / 0.169", "5.7× / 1.4× faster"],
  ], [34, 18, 18, 18], { rightAlignNumbers: true }));
children.push(gap());
children.push(P("All baseline-vs-improved differences above are significant (paired bootstrap, p ≤ 0.002). The research extension identified **ASR as the measurable bottleneck** (with a perfect transcript, 99.2% of responses are correct) and replaced it with **language-adapted Whisper served on faster-whisper**, improving both accuracy and speed."));

// 2. Models used
children.push(H1("2. Models used"));
children.push(P("All models are open-source and used as pretrained checkpoints; none were trained or fine-tuned by us."));
children.push(table(
  ["Role", "Model", "How it is used"],
  [
    ["Language identification", "openai/whisper-medium", "One decoder step; probability over the <|hi|> and <|ml|> tokens only"],
    ["ASR, baseline (both languages)", "openai/whisper-small (244M)", "Hugging Face transformers, fp16, greedy, language forced"],
    ["ASR, improved, Hindi", "vasista22/whisper-hindi-small (244M)", "Fine-tuned on Hindi corpora; safetensors revision pinned (e760aba8); faster-whisper"],
    ["ASR, improved, Malayalam", "vrclc/Whisper-medium-Malayalam (769M)", "Fine-tuned on Malayalam corpora, never saw FLEURS; faster-whisper"],
    ["ASR, size control (ablation)", "openai/whisper-medium (769M)", "Stock model, to separate fine-tuning gains from model-size gains"],
    ["Language model", "Qwen/Qwen3-4B-Instruct-2507", "bitsandbytes 4-bit NF4, greedy (deterministic)"],
    ["LLM judge (evaluation only)", "Qwen/Qwen2.5-7B-Instruct", "4-bit, including the output layer, to fit 6 GB"],
    ["Meaning similarity (evaluation only)", "sentence-transformers/LaBSE", "Cross-lingual sentence embeddings"],
  ], [24, 30, 46]));
children.push(gap());
children.push(P("**Hardware:** a laptop with an NVIDIA RTX 3060 (6 GB VRAM), Windows 11. Models are loaded one at a time; PyTorch 2.5.1 + CUDA 12.1 throughout, so every run was timed on the same software stack."));

// 3. Dataset
children.push(H1("3. Dataset"));
children.push(B("**Source:** Google **FLEURS** test split, Hindi (hi_in) and Malayalam (ml_in). Each recording has a human native-script transcript; through the FLoRes benchmark each sentence also has a professional **English translation**, used as the meaning reference."));
children.push(B("**Sample:** 120 distinct sentences per language, fixed random seed, gender-stratified for Hindi (60/60). The Malayalam test speakers are all female."));
children.push(B("**Acoustic conditions:** clean, **10 dB** and **0 dB** SNR. The noise is babble from 4 other speakers plus white noise, mixed at exactly the target SNR (verified: 10.00 dB and 0.00 dB). Each utterance gets the same noise at both levels, so only the loudness changes."));
children.push(B("**Evaluation sets:** 720 items for ASR and restatement; 160 items per system for reply mode (40 utterances per language at clean and 0 dB) plus 80 with gold transcripts; 720 for language ID; 48 replies for native-speaker rating."));

// 4. Architecture
children.push(H1("4. Architecture"));
children.push(...figure("architecture.png", "Figure 1. Pipeline and evaluation infrastructure."));
children.push(B("**Language ID** picks Hindi or Malayalam from the audio and routes it to the matching ASR model."));
children.push(B("**ASR** transcribes the audio with greedy decoding, the language token forced and a single 30-second window. Baseline and improved runs differ only in model weights and inference engine."));
children.push(B("**Language model** turns the transcript into a response in the same language and script, in one of two modes: **restatement** or **reply** (Section 6)."));
children.push(B("**Runner:** ASR runs on all items, then the LLM, because both don't fit in 6 GB together. Every result is saved item by item, so interrupted runs resume where they stopped. A single-file demo (scripts/demo.py) runs the full chain on any recording."));

// 5. Baseline vs improved
children.push(H1("5. Baseline vs improved"));
children.push(table(
  ["", "Baseline", "Improved"],
  [
    ["Hindi ASR", "openai/whisper-small", "vasista22/whisper-hindi-small (fine-tuned, same size)"],
    ["Malayalam ASR", "openai/whisper-small", "vrclc/Whisper-medium-Malayalam (fine-tuned)"],
    ["ASR engine", "Hugging Face transformers", "faster-whisper (CTranslate2), with a decoding-limit fix"],
    ["Language model", "Qwen3-4B-Instruct, 4-bit", "Same (unchanged, so the comparison isolates ASR)"],
    ["Language selection", "Given", "Automatic language ID (stock whisper-medium)"],
  ], [20, 32, 48]));
children.push(gap());
children.push(H2("What is in the improved system"));
children.push(N("**Language-adapted ASR.** Whisper checkpoints fine-tuned on Hindi and Malayalam speech replace the stock multilingual model, which mostly failed on Malayalam: repetition loops, English translations instead of transcripts, or Hindi-script output."));
children.push(N("**A faster inference engine.** faster-whisper (CTranslate2) runs the same weights about 3× faster. We found and fixed a bug in its decoder: CTranslate2 4.8.4 returns only half the requested number of tokens, which cut long Malayalam transcripts in half. After the fix, the accuracy of the two engines matches."));
children.push(N("**Automatic language identification**, so a user doesn't need to say which language they are speaking (97.5% accurate; 100% on clean speech)."));

// 6. Why reply + restatement
children.push(H1("6. Why both a reply and a restatement"));
children.push(P("The language model runs in two modes, and both are evaluated, because a spoken assistant must get two different things right:"));
children.push(B("**Restatement (did it understand?)** The model restates faithfully what the user said, in clean Hindi or Malayalam, repairing misrecognised words without adding anything. This has a **gold answer** (what was actually said), so understanding can be measured objectively: correctness against two references, character error rate, and an oracle comparison that proves where errors come from."));
children.push(B("**Reply (did it respond well?)** The model answers the user as an assistant would, in 1–3 sentences. This is the user-facing response the brief asks for. A reply has no single gold answer, so it is measured with a grounded rubric and with native-speaker ratings."));
children.push(P("Neither mode is enough on its own. A reply metric alone can't tell \"misheard the user\" apart from \"responded badly\"; the restatement can. The restatement alone doesn't show whether the user gets a useful answer; the reply does. Both modes use the same ASR transcripts, so their results can be compared directly."));

// 7. Results against the brief
children.push(H1("7. Results against the requirements"));
children.push(P("Every metric is reported by language and acoustic condition with **bootstrap 95% confidence intervals** (2,000 resamples of whole utterances). Baseline-vs-improved comparisons use **paired** tests on the same items."));

children.push(H2("7.1 ASR word and character error rate"));
children.push(P("Both sides are normalised identically before scoring: Unicode NFC, an Indic script normaliser, punctuation including the danda (।) removed. Whisper's built-in English normaliser deletes Indic vowel signs and was deliberately **not** used. **CER is the primary metric for Malayalam**, which is agglutinative: one wrong suffix makes a whole long word wrong, so WER exaggerates the damage. Rates above 100% come from repetition loops, which insert far more text than the reference contains."));
children.push(table(
  ["Language", "Condition", "WER baseline", "WER improved", "CER baseline", "CER improved", "Loops baseline", "Loops improved"],
  [
    ["Hindi", "clean", "79.4%", "14.2%", "45.9%", "5.8%", "7.5%", "0.0%"],
    ["Hindi", "10 dB", "105.8%", "27.6%", "75.4%", "16.2%", "15.8%", "0.0%"],
    ["Hindi", "0 dB", "182.8%", "95.8%", "182.3%", "82.3%", "63.3%", "10.0%"],
    ["Malayalam", "clean", "280.3%", "36.9%", "160.9%", "12.8%", "41.7%", "0.8%"],
    ["Malayalam", "10 dB", "158.7%", "49.7%", "121.7%", "18.2%", "52.5%", "0.0%"],
    ["Malayalam", "0 dB", "159.8%", "91.1%", "134.5%", "52.8%", "70.8%", "7.5%"],
  ], [14, 11, 12, 12, 12, 12, 13, 14], { rightAlignNumbers: true }));
children.push(caption("Table: ASR error rates (n = 120 per row). \"Loops\" = share of transcripts that are repetition loops or runaway output."));
children.push(...figure("cer_vs_snr.png", "Figure 2. Character error rate by noise level, with 95% CIs. The size-control line is stock whisper-medium."));

children.push(H2("7.2 Response correctness and hallucination: criteria"));
children.push(P("Each criterion has a written definition, applied by an LLM judge (Qwen2.5-7B) and, for replies, checked against a native speaker."));
children.push(table(
  ["Mode", "Criterion", "Definition"],
  [
    ["Restatement", "Correct", "In the right language and conveys every key element of what was said (who, what, entities, numbers, places, times); paraphrase allowed"],
    ["Restatement", "Partial / incorrect", "Main idea kept but a key element missing or wrong / main idea wrong, unrelated, empty or wrong language"],
    ["Restatement", "Hallucination", "States a specific detail (name, number, event, place) that was not said. Omissions and garbled words are not hallucination"],
    ["Restatement", "Lexical check", "Judge-free cross-check: numbers in the response that appear in neither reference"],
    ["Reply", "Relevance", "On-topic: engages with what was actually said / partial / off-topic"],
    ["Reply", "Misattribution", "Says or implies the user said something they didn't, changing the meaning (paraphrase is fine)"],
    ["Reply", "Factual hallucination", "Asserts a specific claim that is false or contradicts what was said (correct background knowledge is fine)"],
    ["Reply", "Acceptable", "On-topic, no misattribution, no factual hallucination, and in the right script"],
    ["Both", "Off-script", "Less than 80% of letters in the target script (e.g. an English answer)"],
  ], [15, 20, 65]));
children.push(gap());
children.push(P("Reply criteria are judged against **what the user actually said** (the gold transcript and its English translation). The judge never sees the ASR transcript, so a reply built on a misrecognition counts as a failure."));

children.push(H2("7.2 (continued) Response correctness and hallucination: results"));
children.push(table(
  ["Language", "Condition", "Restatement correct (base → improved)", "Restatement hallucination (base → improved)", "Reply acceptable (base → improved)"],
  [
    ["Hindi", "clean", "37.5% → 80.0%", "10.8% → 0.8%", "32.5% → 77.5%"],
    ["Hindi", "10 dB", "16.7% → 50.0%", "19.2% → 9.2%", "not run"],
    ["Hindi", "0 dB", "2.5% → 3.3%", "4.2% → 39.2%", "10.0% → 12.5%"],
    ["Malayalam", "clean", "7.5% → 93.3%", "11.7% → 0.0%", "7.5% → 57.5%"],
    ["Malayalam", "10 dB", "6.7% → 79.2%", "15.0% → 6.7%", "not run"],
    ["Malayalam", "0 dB", "5.0% → 29.2%", "5.0% → 23.3%", "2.5% → 22.5%"],
  ], [13, 11, 25, 27, 24], { rightAlignNumbers: true }));
children.push(caption("Table: restatement n = 120 per row; reply n = 40 per row (the reply subset covers clean and 0 dB)."));
children.push(B("On clean and 10 dB speech, correct responses rise sharply and hallucination falls to ≤ 1% on clean audio."));
children.push(B("**At 0 dB, better ASR increases hallucination** (Hindi 4.2% → 39.2%). The baseline's 0 dB transcripts were loops that the LLM copied back (wrong, but not invented). The improved ASR gives partially correct transcripts, and the LLM, asked to repair misrecognised words, fills the gaps with invented content."));
children.push(B("In reply mode, the improved system on clean Hindi reaches the **gold-transcript ceiling** (77.5%, the same as with a perfect transcript). The remaining errors come from the language model: even with a perfect transcript it asks the user to repeat in 22.5% of Hindi replies."));
children.push(...figure("reply_acceptable.png", "Figure 3. Acceptable replies by condition, with the gold-transcript (oracle) ceiling."));

children.push(H2("7.3 Processing latency and real-time factor"));
children.push(P("Each stage is timed separately (GPU synchronised before each clock read; 3 warm-up items excluded per group). **Real-time factor (RTF) = processing time ÷ audio duration**; below 1 is faster than real time. Medians (p50) are shown."));
children.push(table(
  ["System", "Language", "ASR (s)", "LLM first token (s)", "LLM total (s)", "End-to-end (s)", "End-to-end RTF", "ASR RTF (mean)"],
  [
    ["Baseline", "Hindi", "1.58", "0.19", "9.39", "10.98", "1.01", "0.239"],
    ["Baseline", "Malayalam", "4.88", "0.21", "23.64", "28.35", "2.45", "0.234"],
    ["Improved", "Hindi", "0.42", "0.18", "7.84", "8.25", "0.82", "0.042"],
    ["Improved", "Malayalam", "2.21", "0.22", "14.72", "16.97", "1.25", "0.169"],
  ], [12, 13, 10, 13, 12, 13, 13, 14], { rightAlignNumbers: true }));
children.push(caption("Table: restatement mode, n = 351 timed items per language. Language ID adds about 0.10 s (RTF 0.008)."));
children.push(B("Speech recognition is now fast: RTF 0.042 for Hindi and 0.169 for Malayalam."));
children.push(B("**The language model is now the latency bottleneck** (85–95% of the time per utterance). The first token arrives in ~0.2 s, but 4-bit decoding runs at ~12 tokens per second on this GPU, and Malayalam text needs ~14 tokens per second of speech against ~10 for Hindi."));
children.push(...figure("latency_breakdown.png", "Figure 4. Mean time per utterance, split into ASR and LLM."));

children.push(H2("7.4 Performance across linguistic and acoustic conditions"));
children.push(table(
  ["Condition", "Finding"],
  [
    ["Linguistic: Hindi vs Malayalam", "Stock Whisper is far worse on Malayalam (CER 139% vs 101%) and mostly cannot transcribe it. After adaptation Malayalam is on par with or better than Hindi (CER 27.9% vs 34.8%). Malayalam remains costlier for the LLM (more tokens per sentence) and harder for the LLM judge."],
    ["Acoustic: clean → 10 dB", "Moderate degradation: improved CER goes 5.8% → 16.2% (Hindi) and 12.8% → 18.2% (Malayalam); restatement correctness stays useful (50% and 79%)."],
    ["Acoustic: 0 dB", "The hardest condition for every system: improved CER 82.3% (Hindi) and 52.8% (Malayalam); correct responses collapse, and hallucination rises as the LLM over-repairs partial transcripts. Language ID also drops to 86.7% for Malayalam."],
  ], [26, 74]));

// 8. Evaluation strategy
children.push(H1("8. Evaluation strategy: native-speaker judge vs LLM judge"));
children.push(P("Automatic LLM judges are convenient but untested on Indian languages. Because I am a fluent native speaker of **both Hindi and Malayalam**, I rated a sample of replies myself and used those ratings both as a **human evaluation** and as a **check on the LLM judge**."));
children.push(H2("Method"));
children.push(B("**Blind and paired:** 24 utterances (6 per language × clean / 0 dB), each with its baseline and its improved reply = 48 replies, shuffled, with no indication of which system wrote which."));
children.push(B("**Criteria:** response quality (is this a good, correct response? non-answers such as \"I didn't understand, please repeat\" count as incorrect), relevance (on-topic / partial / off-topic), and understandability to a native reader (1–5)."));
children.push(B("**Agreement:** Cohen's κ between my ratings and the LLM judge, overall and per language."));
children.push(H2("Results"));
children.push(table(
  ["", "Baseline", "Improved", "Paired difference"],
  [
    ["Good responses", "12%", "42%", "+29 pts, p < 0.001"],
    ["On-topic", "12%", "46%", "+33 pts, p = 0.002"],
    ["Understandability (1–5)", "4.29", "3.79", "−0.50, not significant"],
  ], [30, 18, 18, 34], { rightAlignNumbers: true }));
children.push(gap());
children.push(table(
  ["Agreement with the LLM judge (Cohen's κ)", "All", "Hindi", "Malayalam"],
  [
    ["Relevance", "0.52", "0.53", "0.50"],
    ["Response quality (human \"correct\" vs judge \"acceptable\")", "0.46", "0.57", "0.25"],
  ], [52, 16, 16, 16], { rightAlignNumbers: true }));
children.push(gap());
children.push(B("**The human ratings independently confirm the improvement**, so the result doesn't rest on the LLM judge."));
children.push(B("**The LLM judge is only a moderate proxy for a native speaker** (κ 0.46–0.52), and is **much weaker on Malayalam** (κ = 0.25 on response quality vs 0.57 for Hindi). A model judge misses quality problems that a native speaker sees; for example, it accepted a reply containing invented scientific terms."));
children.push(B("**Understandability** drops where the improved system engages with very noisy Malayalam: it copies garbled non-words from the transcript into its reply. The baseline's replies there were fluent but useless requests to repeat."));
children.push(H2("Why this strengthens the evaluation"));
children.push(P("Adding a native-speaker judge is an **improvement to the evaluation methodology** (separate from the system improvement in Section 9). It gives a ground truth that the LLM judge cannot provide for Indian languages, measures how far the automatic judge can be trusted in each language, and adds a criterion (understandability to a native reader) that an automatic judge cannot assess reliably. Limitations: a single annotator, who is also the project author (the ratings were blind to the system), and 6 items per condition."));

// 9. Research extension
children.push(H1("9. Research extension: bottleneck and improvement"));
children.push(H2("Identifying the bottleneck"));
children.push(P("An **oracle ablation** fed the gold transcript straight to the language model. With a perfect transcript, **99.2%** of restatements are judged correct in both languages, and **99–100%** of the baseline's failures and **100%** of its hallucinations disappear. Errors therefore come almost entirely from **ASR**, which is a measurable bottleneck: baseline CER is 101% (Hindi) and 139% (Malayalam), with 29–55% of transcripts being repetition loops."));
children.push(H2("The improvement and the comparison"));
children.push(P("We replaced the stock ASR with **language-adapted Whisper checkpoints served on faster-whisper**, keeping everything else fixed, and isolated each effect with ablations:"));
children.push(table(
  ["Run (only the ASR changes)", "Hindi CER", "Malayalam CER", "Hindi ASR RTF", "Malayalam ASR RTF"],
  [
    ["Baseline: whisper-small, HF engine", "101.2%", "139.0%", "0.239", "0.234"],
    ["Size control: stock whisper-medium, HF engine", "—", "119.0%", "—", "0.663"],
    ["Adapted ASR, HF engine", "32.3%", "26.7%", "0.137", "0.462"],
    ["Improved: adapted ASR + faster-whisper", "34.8%", "27.9%", "0.042", "0.169"],
  ], [40, 14, 16, 14, 16], { rightAlignNumbers: true, highlight: [3] }));
children.push(gap());
children.push(B("**Accuracy comes from adaptation, not size.** Scaling stock Whisper from small to medium cuts Malayalam CER by only 20 points and nearly triples its time; the adapted model cuts a further 91 points."));
children.push(B("**Speed comes from the engine.** On the same adapted weights, faster-whisper is 3.3× (Hindi) and 2.7× (Malayalam) faster, with CER within 1–2.5 points."));
children.push(B("**Effect on responses:** restatement correctness +25.6 pts (Hindi) and +60.8 pts (Malayalam); reply acceptability +45 to +50 pts on clean speech; native-speaker good responses +29 pts. All p ≤ 0.002."));
children.push(H2("What the improvement revealed"));
children.push(B("The **new bottleneck is the language model**: it now dominates latency, and its reply behaviour (needless requests to repeat, occasional misstatements) sets the quality ceiling."));
children.push(B("**Very noisy speech (0 dB)** remains unsolved and exposes a hallucination risk when the LLM repairs partial transcripts."));

// 10. Deliverables
children.push(H1("10. Deliverables"));
children.push(table(
  ["Deliverable", "Where"],
  [
    ["Working code", "src/ (pipeline, ASR, LLM, language ID, metrics, statistics), scripts/ (data preparation, runs, evaluation, report, demo), configs/ (one file per run); 19 unit tests for the evaluation code"],
    ["Evaluation results", "results/<run>/ per-item outputs and scores for every run; results/summary.md (all tables with confidence intervals); results/figures/; results/human_eval/ (blind ratings and analysis); results/lid/"],
    ["Ablation analysis", "Oracle (ASR vs LLM attribution), size control (adaptation vs model size), engine comparison (HF vs faster-whisper), noise levels, languages, restatement vs reply"],
    ["Full technical report", "REPORT.md (methods, all results, limitations); README.md (setup and one-command reproduction)"],
  ], [24, 76]));

// 11. Limitations
children.push(H1("11. Limitations"));
children.push(B("Human evaluation uses one annotator (the project author, rating blind) and 48 replies; a second annotator would allow inter-annotator agreement to be measured."));
children.push(B("The LLM judge's hallucination labels were not separately rated by a human, and they probably under-count hallucination."));
children.push(B("Whisper's decoder holds at most ~200 Malayalam characters per 30-second window, so the longest Malayalam sentences cannot be fully transcribed by any Whisper model; audio longer than 30 s is truncated."));
children.push(B("FLEURS is read speech; conversational or code-mixed speech would be harder. The Hindi adapted model was trained on FLEURS train+dev (no test overlap). Multi-step instruction following was not evaluated."));

// 12. Conclusion
children.push(H1("12. Conclusion"));
children.push(P("The pipeline meets every requirement of the brief: speech in Hindi or Malayalam goes through ASR and a language model to a text response in the same language, evaluated with our own infrastructure on 720 items across two linguistic and three acoustic conditions. ASR WER/CER, response correctness and hallucination under written criteria, latency and real-time factor are all reported with confidence intervals."));
children.push(P("The research extension found a clear, measurable bottleneck, **speech recognition**, and fixed it with language-adapted Whisper on faster-whisper: Malayalam CER fell from 139% to 28%, correct responses rose by 26–61 points, and ASR became up to 5.7× faster. Ablations show the accuracy gain comes from language adaptation rather than model size, and the speed gain from the inference engine. Blind native-speaker ratings confirm the improvement independently and show that an LLM judge is only a moderate stand-in for a native speaker, especially in Malayalam. The next bottleneck is the language model itself: its latency, and its reply behaviour on noisy input."));

const doc = new Document({
  creator: "vaaniTask",
  title: "Indic Speech-to-Response Pipeline",
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], font: FONT, size: 18, color: "777777" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
