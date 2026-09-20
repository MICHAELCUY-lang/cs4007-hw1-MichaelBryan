# HW1 submission

**Name:** Michael Bryan Mandey
**Student ID:** S26078853
**Group:** CSS4007-ENG-10
**Repository:** https://github.com/MICHAELCUY-lang/cs4007-hw1-MichaelBryan

## AI tool disclosure

State which AI tools you used and for what. Expected and fine; undisclosed use
is not.

> I used ChatGPT for debugging, exercise, and understanding API errors

---

## Sublab Easy — the registration bot and its bill

**How I laid the catalogue out inside the system prompt, and why:**

> I placed the registration rules, student record, and every course in the
> system prompt as structured JSON. This preserves exact course codes, credits,
> prerequisites, schedules, seat counts, and completed courses without
> paraphrasing the source data. I also explicitly instructed the model to check
> prerequisites, seats, completed courses, schedule collisions, and credit
> limits, and to refuse courses that do not appear in the catalogue.

**My turn 5 (Kazakh or Russian):**

> Я студент третьего курса. На какие курсы я всё ещё могу зарегистрироваться?

### Run 1 — OpenAI, `gpt-5.6-luna`

| Turn | Input tokens | Output tokens | Cost $ |
|---|---|---|---|
| 1 | 1422 | 505 | 0.000890 |
| 2 | 1681 | 121 | 0.000481 |
| 3 | 1798 | 73 | 0.000447 |
| 4 | 1890 | 54 | 0.000443 |
| 5 | 1968 | 373 | 0.000841 |
| **total** | **8759** | **1126** | **0.003103** |

### Run 2 — OpenRouter, `google/gemma-4-26b-a4b-it:free`

| Turn | Input tokens | Output tokens | Cost $ |
|---|---|---|---|
| 1 | 1675 | 353 | 0.000000 |
| 2 | 2047 | 91 | 0.000000 |
| 3 | 2156 | 88 | 0.000000 |
| 4 | 2260 | 45 | 0.000000 |
| 5 | 2324 | 371 | 0.000000 |
| **total** | **10462** | **948** | **0.000000** |

### Turn 4, verbatim

The turn where you asked for CSS-4090, which does not exist. Paste both replies
exactly as they came back — do not tidy them.

**OpenAI:**

```
I cannot add **CSS-4090 — Quantum Machine Learning** because it is **not present in the catalogue**. The catalogue contains no course with code **CSS-4090**, so its credits, prerequisites, availability, and schedule cannot be verified.
```

**OpenRouter:**

```
I cannot add this course to your schedule.

**CSS-4090 (Quantum Machine Learning)** does not exist in the course catalogue. I can only register you for courses that are present in the authoritative catalogue.
```

### Written answers

**1. The two providers used almost identical code. What actually changed, and
what did not?**

> The API key, base URL, and model name changed. OpenAI used its normal endpoint
> and `gpt-5.6-luna`; OpenRouter used its OpenAI-compatible endpoint and the
> `google/gemma-4-26b-a4b-it:free` model name. The OpenAI Python client, message
> format, complete conversation history, response parsing, usage fields, and
> cost calculation stayed the same.

**2. Why did the input token count climb on every turn when your questions
stayed roughly the same length? Use the numbers from your own table. What
happens to the bill at fifty turns?**

> The program sends the complete growing history on every request; the model
> does not remember previous API calls. OpenAI input rose from 1,422 tokens on
> turn 1 to 1,681, 1,798, 1,890, and 1,968 tokens. Gemma showed the same pattern:
> 1,675, 2,047, 2,156, 2,260, and 2,324 input tokens. Therefore earlier messages
> are processed repeatedly. At fifty turns the later requests would contain most
> of the preceding transcript, so cumulative input and, for a paid model, cost
> would grow roughly quadratically rather than linearly if turn lengths remained
> similar. This Gemma run happened to report zero cost because the model route
> was free.

**3. Turn 4: did the bot refuse, or did it invent CSS-4090?** If it refused, what
in your system prompt held the line? If it invented, what did it make up —
credits, a room, an instructor?

> Both bots refused and did not invent credits, prerequisites, a room, or an
> instructor. The system prompt included the complete catalogue and explicitly
> said to refuse any course code not present in it. OpenAI said the course was
> not present in the catalogue, while Gemma said it did not exist in the
> authoritative catalogue.

**4. Where else was either bot wrong?** Turn 2 asks for two courses that meet at
the same hour; two courses in the catalogue are full. Did the bots notice?

> Both bots noticed that CSS-4007 and CSS-4102 meet Tuesday 09:00–10:50 and
> refused to register them together. Both also recognized that CSS-4400 was
> full and that CSS-3011 had already been completed. OpenAI correctly excluded
> every completed course and identified FIN-3300 as having one seat left.
> Gemma made a significant error in turns 1 and 5: it offered CSS-3005,
> MAT-2020, and ECN-2101 even though all three appear in the student's completed
> list. It otherwise reported their seat counts correctly.

---

## Sublab Medium — one task, six models

Paste the per-model summary printed by `correct_kazakh.py`:

| Model | Exact | Failed | Tokens | Cost $ |
|---|---|---|---|---|
| google/gemma-4-26b-a4b-it:free | 4 | 0 | 1832 | 0.00000 |
| qwen/qwen3.8-27b | 4 | 0 | 2160 | 0.00356 |
| deepseek/deepseek-v4-flash-0731 | 4 | 0 | 1975 | 0.00039 |
| gpt-5.6-luna | 3 | 2 | 2262 | 0.00191 |
| gpt-5.6-terra | 0 | 8 | 0 | 0.00000 |
| gpt-5.6-sol | 0 | 8 | 0 | 0.00000 |

> Gemma, Qwen, and DeepSeek completed all eight calls. Luna completed six calls;
> KZ-02 and KZ-05 hit its OpenAI token rate limit. Terra and Sol failed all
> calls because the OpenAI organization had no credits remaining.

### Which error types did each model repair?

Rows are error labels, columns are models. Write "yes", "no" or "partial".

| Error type | gemma | qwen | deepseek | luna | terra | sol |
|---|---|---|---|---|---|---|
| kaz_to_rus | yes | yes | partial | yes | no (failed) | no (failed) |
| latin_homoglyph | partial | yes | yes | yes | no (failed) | no (failed) |
| drop_hyphen | partial | yes | yes | no (failed) | no (failed) | no (failed) |
| join_words | yes | yes | yes | partial | no (failed) | no (failed) |
| double_letter | yes | yes | yes | yes | no (failed) | no (failed) |

**The `latin_homoglyph` row: what happened?** Describe what you observed. The
explanation is Sublab Harder's job, not this one's.

> Qwen and DeepSeek repaired both `latin_homoglyph` rows exactly. Luna repaired
> both too: KZ-08 was exact, while KZ-03 added only a comma and question mark.
> Gemma repaired KZ-08, but its KZ-03 response changed `Алаяқтарға` into the
> incorrect `Алақаттарға`, so I marked it partial. Terra and Sol produced no
> scored responses for this category.

**Where a model returned good Kazakh that was not identical to the original,
say so here.** Exact match is not correctness.

> Several non-exact responses were still reasonable Kazakh. Gemma KZ-01 used
> the plural `мемлекеттердің елшілерінен`, and Gemma KZ-04 added a question
> mark. Qwen's four non-exact rows (KZ-02, KZ-04, KZ-05, and KZ-06) differed
> only by sensible final punctuation. DeepSeek KZ-02, KZ-04, and KZ-05 also
> differed only by punctuation. Luna KZ-01 used the plural
> `мемлекеттердің елшілерінен`, while KZ-03 and KZ-04 added appropriate
> punctuation. In contrast, Gemma KZ-02 still omitted the required hyphen
> (`50%ға`), Gemma KZ-03 produced the wrong word, and DeepSeek KZ-01 left
> `Токаев` without the required Kazakh `қ`; those are substantive errors.

**Cheapest model that was good enough, and why:**

> Qwen was the cheapest model that was good enough: it repaired all eight cases
> for $0.00356. Four outputs matched exactly, and the other four differed only
> by reasonable final punctuation. DeepSeek was cheaper at $0.00039 but missed
> one required Kazakh letter in KZ-01. Gemma was free but missed the hyphen in
> KZ-02 and made KZ-03 incorrect, so neither cheaper model was fully adequate.

---

## Sublab Harder — open the tokenizer

### A. What a language costs

**`cl100k_base`:**

| Language | Tokens | Chars | Tok/char | × English | $ per 1,000 sentences |
|---|---|---|---|---|---|
| kk | 200 | 263 | 0.760 | 3.75 | 0.166667 |
| ru | 129 | 277 | 0.466 | 2.30 | 0.107500 |
| en | 59 | 291 | 0.203 | 1.00 | 0.049167 |

**`o200k_base`:**

| Language | Tokens | Chars | Tok/char | × English | $ per 1,000 sentences |
|---|---|---|---|---|---|
| kk | 84 | 263 | 0.319 | 1.58 | 0.070000 |
| ru | 74 | 277 | 0.267 | 1.32 | 0.061667 |
| en | 59 | 291 | 0.203 | 1.00 | 0.049167 |

### B. What a homoglyph does

One row per `latin_homoglyph` sentence in the dataset. Paste the actual decoded
token strings around the divergence point, not a description of them.

| Sentence id | Foreign char (index, name) | Tokens correct | Tokens corrupted | Δ | Diverges at |
|---|---|---|---|---|---|
| KZ-03 | 0 `A` LATIN CAPITAL LETTER A; 2 `a` LATIN SMALL LETTER A; 5 `t` LATIN SMALL LETTER T | 16 | 20 | +4 | 0 |
| KZ-08 | 1 `o` LATIN SMALL LETTER O; 3 `a` LATIN SMALL LETTER A; 9 `T` LATIN CAPITAL LETTER T | 21 | 24 | +3 | 1 |

**Token pieces around the divergence:**

```
KZ-03 correct  : ['А', 'лая', 'қ', 'тарға', ' ақша']
KZ-03 corrupted: ['A', 'л', 'a', 'я', 'қ']

KZ-08 correct  : ['Д', 'он', 'аль', 'д', ' Т', 'рамп']
KZ-08 corrupted: ['Д', 'o', 'н', 'a', 'л', 'ль']
```

### C. Did it get better?

| Language | cl100k_base | o200k_base | Change |
|---|---|---|---|
| kk | 0.760 | 0.319 | −0.441 tok/char (−58.0%) |
| ru | 0.466 | 0.267 | −0.199 tok/char (−42.7%) |
| en | 0.203 | 0.203 | no change |

### Written answers

**1. What is the Kazakh tax?** The ratio against English in both encodings, the
dollar figure from A, and how much it changed between the two tokenizers.

> With `cl100k_base`, Kazakh cost 3.75× as many tokens per character as English
> and about $0.166667 per 1,000 average-length sentences at the specified input
> rate, versus $0.049167 for English. With `o200k_base`, Kazakh fell to 1.58×
> English and about $0.070000, while English stayed at $0.049167. Kazakh dropped
> from 0.760 to 0.319 token/character, a reduction of about 58%, and the
> relative tax narrowed by 2.17×-English.

**2. Why did the models repair `kaz_to_rus` but struggle with
`latin_homoglyph`?** Both are single-letter substitutions and both look almost
identical on screen. Use your token streams from B as the evidence. Say what the
model actually received in each case.

> A `kaz_to_rus` substitution still consists of Cyrillic code points, so it is
> more likely to remain in familiar Cyrillic token pieces that context can
> repair. A Latin homoglyph only looks the same to a human; it is a different
> Unicode code point and changes the token boundaries. In KZ-03, the correct
> stream began `['А', 'лая', 'қ', 'тарға', ' ақша']`, while the corrupted one
> began `['A', 'л', 'a', 'я', 'қ']`; it grew from 16 to 20 tokens and diverged
> immediately. KZ-08 changed from 21 to 24 tokens and split pieces such as
> `['Д', 'он', 'аль', 'д', ' Т', 'рамп']` into
> `['Д', 'o', 'н', 'a', 'л', 'ль']`. Thus the model received a fragmented,
> mixed-script token sequence rather than merely a visually misspelled word.

**3. Name one thing this measurement does not explain about your Sublab Medium
results.** You measured OpenAI's tokenizers; three of your six models were not
OpenAI's. What follows, and what would you have to do to close the gap?

> These measurements do not directly explain Gemma, Qwen, or DeepSeek because
> those models may use different tokenizers and vocabularies. Their token
> boundaries and homoglyph penalties may differ from `cl100k_base` and
> `o200k_base`. To close the gap, I would run the same correct and corrupted
> strings through each model's actual tokenizer and compare its token IDs,
> decoded pieces, divergence position, and token-count change.
