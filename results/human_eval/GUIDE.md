# Rating guide: reply mode

> **Note added after rating (2026-10-30):** the annotator applied `correctness` as **overall response
> quality**: is this a good, correct response? Non-answers ("I didn't understand, please repeat") were
> rated **incorrect**, not correct as the original line below said; this was applied consistently (all 17
> such replies were rated incorrect or partial). Hallucination was not rated separately. The report and
> analysis use this definition.

Each row is one reply from a voice assistant. Judge the reply against **what the person actually said**
(column `what_was_said`; `english_meaning` is there to help). The assistant only heard the audio, so it may
have misheard; that counts against it. You don't know which system wrote which reply. Rate each row on its own.

## correctness: `correct` / `partial` / `incorrect`
Is the reply factually right and faithful to what was said?
- **correct**: nothing in it is false, and it doesn't misstate what the person said.
- **partial**: mostly fine, but one detail is wrong or misstated.
- **incorrect**: contains a false claim, misstates what the person said, or is nonsense.
Asking the person to repeat themselves, without claiming anything false, counts as **correct**.

## relevance: `on_topic` / `partial` / `off_topic`
Does the reply respond to what the person actually said?
- **on_topic**: engages with the actual subject and content.
- **partial**: right general topic, but responds to a detail they didn't say.
- **off_topic**: about something else, or a generic "please repeat" that doesn't engage with the content.

## understandability: `1` to `5`
As a native speaker, how easy is **the reply text itself** to read and understand, and how natural is it?
This is about the assistant's language, not about how well it understood the person: that is
what correctness and relevance measure. A fluent reply about the wrong topic is still a 5 here.
- **5**: fluent and natural; reads like a native speaker wrote it.
- **4**: clear, with minor awkwardness (odd word choice, slightly unnatural phrasing).
- **3**: understandable with some effort; clearly non-native or awkward.
- **2**: hard to understand; meaning only partly recoverable.
- **1**: incomprehensible, garbled, or not in the right language.
Rate the language itself, not whether it's correct or relevant.

## Saving
Fill only the three columns, then **File > Save As > "CSV UTF-8 (Comma delimited)"** (plain CSV can garble Malayalam).
