# Natural-Language Video Investigator: design and evaluation plan

Prepared 14 November 2026 for Yesaullah Sheikh (team lead) and Murtaza Jafri, AI Engineering (AI3007).
Source proposal: `project_proposal_natural_language_video_investigator.pdf` in this folder.

This plan fixes the models, data, evaluation protocol, compute budget and work units before any code is written. Build from it section by section. Where it differs from the proposal, section 12 lists the difference.

## How to read this plan

**Decisions already made by the team (14 November 2026):** the GPU is a desktop or SFF RTX 4000 Ada; the GPU machine runs Windows only; all videos and queries are in English; the public benchmark is Charades-STA.

**Evidence labels.** Every number about a model, dataset or library carries one of three labels.

| Label | Meaning |
|---|---|
| Documented | Read on 14 November 2026 from the source listed in section 13. |
| Estimated | Derived by arithmetic from documented figures. The basis is stated. |
| To measure | Not known yet. Week 1 measures it against the stated pass threshold. |

No throughput or memory figure in this plan was measured on the project GPU. The machine this plan was written on has only Intel UHD 620 graphics, so `nvidia-smi` could not be run. Section 2 makes that the first check.

**Where each "done when" item is answered.**

| Requirement | Section |
|---|---|
| Models chosen with VRAM and speed evidence | 3 |
| Public benchmark slice chosen, with reason | 4.2 |
| Labelled-set protocol: unanswerable queries, calibration and test split | 4.1, 4.3, 4.4 |
| Baselines and ablations fixed | 5 |
| Metrics and uncertainty reporting fixed | 6 |
| Compute budget per stage | 7 |
| Implementation units per member, with acceptance checks | 8 |
| Go or no-go test for the agentic extension | 9 |
| Remaining choices, each with a recommendation | 11 |

**Terms used throughout.**

| Term | Meaning |
|---|---|
| Team set | The 30 videos the team collects and labels (section 4.1). |
| Charades slice | The fixed sample of Charades-STA videos and queries (section 4.2). |
| Answerable query | A query with at least one moment in the video that satisfies it. |
| Unanswerable query | A query that no moment in the video satisfies. Also called a negative. |
| Dev, calibration, test | Three disjoint groups of videos. Dev is for all tuning, calibration only for the abstention threshold, test for the single final run. |
| IoU | Temporal intersection over union between a predicted window and a labelled window. |
| Rejection accuracy | Share of unanswerable queries on which the system abstains. |
| False-answer rate | Share of unanswerable queries on which the system returns a result. Equals 1 minus rejection accuracy. |
| False-abstain rate | Share of answerable queries on which the system abstains. |

---

## 1. Decision summary

| Item | Decision | Status |
|---|---|---|
| GPU memory | Design to 20 GB, not the proposal's 24 GB. Peak use capped at 17 GiB. | Confirm with `nvidia-smi` in Week 1 |
| Speech model | faster-whisper, `turbo` (Whisper large-v3-turbo), float16 | Recommended; one Week 1 accuracy check |
| Frame encoder | `google/siglip2-so400m-patch16-384`, float16 | Recommended; speed gate in Week 1 |
| Captioning model | `Qwen/Qwen3-VL-4B-Instruct` as default, `Qwen/Qwen3.5-4B` as challenger | Decided by the Week 1 bake-off rule in 3.3 |
| Text embedder | `Qwen/Qwen3-Embedding-0.6B` | Fixed |
| Reranker (optional) | `Qwen/Qwen3-VL-Reranker-2B` over the top 10 candidates | Kept only if it passes the dev rule in 3.5 |
| Public benchmark | Charades-STA: 60 test videos, plus 60 train videos for dev and calibration | Fixed |
| Team set | 30 videos, 360 queries, one third unanswerable, split by video 10 / 8 / 12 | Fixed |
| Abstention threshold | Chosen on calibration videos only, by a rule fixed in 4.4 | Fixed |
| Test set custody | Yesaullah holds the test labels and runs the single final evaluation | Fixed |
| Agentic extension | Two gates: entry at end of Week 5, keep at end of Week 6 | Fixed |

---

## 2. Hardware

**What is known.** NVIDIA's datasheets list 20 GB of GDDR6 for both the full-size RTX 4000 Ada Generation and the SFF version (documented). The SFF card has lower memory bandwidth, 280 GB/s against 360 GB/s (documented), so expect it to be slower. The proposal's "24 GB" is wrong and must be corrected in the final report. The laptop GPU of the same name has 12 GB (documented, third-party specification pages); the team confirmed the project card is not that one.

**Design rule.** Peak GPU memory, read from `nvidia-smi`, must stay at or below 17 GiB (17,408 MiB) in every stage. That leaves about 3 GiB for the Windows desktop, memory fragmentation and measurement error. If a monitor is attached to the card, record idle memory use first and subtract it.

**Loading rule.** Indexing loads one model at a time: speech model, then frame encoder, then captioning model, each unloaded before the next. Peak memory is therefore the largest single stage, not the sum. At query time the frame encoder, text embedder and optional reranker stay loaded together.

**First check (unit Y1 in section 8).** Run `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv` on the GPU machine and save the output to `results/env.json`. Pass: the name contains "RTX 4000", `memory.total` is between 19,500 and 20,480 MiB, and `torch.cuda.is_available()` returns true. If `memory.total` is outside that range, stop and revisit section 3 before building anything.

---

## 3. Models

All recommended models are Apache-2.0 or MIT licensed, ungated, and load through libraries that publish Windows wheels (documented). Weight sizes are estimated as parameter count times 2 bytes for 16-bit weights, in GiB. Parameter counts are documented on each model's Hugging Face page.

### 3.1 Speech: faster-whisper `turbo`

| Option | Evidence | Verdict |
|---|---|---|
| **A. `turbo` (large-v3-turbo), float16** | 809M parameters; same model as large-v3 with the decoder cut from 32 layers to 4 (documented, OpenAI model card). A community benchmark on an RTX 2080 Ti reports 19.2 s and 2,537 MB peak, against 52.0 s and 4,521 MB for large-v3 on the same audio (documented, single community source, audio length not stated). | **Recommended** |
| B. `large-v3`, float16 | The official faster-whisper benchmark gives large-v2, the same size class: 13 minutes of audio in 63 s at 4,525 MB, or 17 s at 6,090 MB with `batch_size=8`, on an RTX 3070 Ti (documented). | Fallback if A fails the accuracy check |
| C. `distil-large-v3` | English only; 26.1 s and 2,409 MB in the same community benchmark (documented). | Not chosen: slower than A there, no advantage |

Reasoning: memory is not a constraint here, because the speech model runs alone and every option is under 7 GB. The whole project transcribes about 210 minutes of audio, so speed barely matters either. The choice rests on accuracy, and the only evidence that A matches B is one community table. Hence the check below.

- **Settings.** `WhisperModel("turbo", device="cuda", compute_type="float16")` wrapped in `BatchedInferencePipeline` with `batch_size=8`, `word_timestamps=True`, `language="en"`. The alias `turbo` resolves to the repository `mobiuslabsgmbh/faster-whisper-large-v3-turbo` (documented in `faster_whisper/utils.py`). Record the downloaded revision hash in `meta.json`.
- **To measure in Week 1.** (1) Speed at least 20 times real time on a 10-minute dev lecture. (2) Word error rate against a 5-minute transcript corrected by hand: use A if its error rate is within 2 percentage points of B's, otherwise use B.
- **Windows setup risk (documented).** GPU inference needs the cuBLAS and cuDNN 9 libraries for CUDA 12. The README's pip method for these is Linux only. On Windows, install them from NVIDIA or use the library archive the README links, and put the DLLs on `PATH`. If this costs more than half a day, switch to `openai/whisper-large-v3-turbo` through the Transformers pipeline, which uses PyTorch's bundled CUDA libraries (estimated to be slower; measure before accepting).

### 3.2 Frame encoder: SigLIP 2

| Option | Parameters | Weights at float16 | Verdict |
|---|---|---|---|
| **A. `google/siglip2-so400m-patch16-384`** | 1.14B | 2.1 GiB (estimated) | **Recommended** |
| B. `google/siglip2-base-patch16-224` | 0.38B | 0.7 GiB (estimated) | Fallback if A is too slow |
| C. `facebook/PE-Core-L14-336` | not listed | not estimated | Not chosen: needs the separate `perception-encoder` library, not Transformers (documented) |

Reasoning: both SigLIP 2 options fit easily, so A is preferred for quality and B is the speed fallback. B processes 196 image patches per frame against 576 for A, so expect B to be about three times faster (estimated from patch count). SigLIP has one property this project needs: it is trained with a sigmoid loss, so `torch.sigmoid(logits_per_image)` gives an independent match probability for each frame and query pair (documented in the Transformers SigLIP 2 guide). That is an absolute score, which abstention can use. Softmax-trained CLIP models only give scores relative to the other candidates.

- **Settings.** Frames sampled at 1 per second, embeddings L2-normalised. Query text must be lowercased and tokenised with `padding="max_length", max_length=64`, because the model was trained that way (documented). Getting this wrong silently lowers accuracy, so unit Y5 tests it.
- **To measure in Week 1.** At least 20 frames per second and peak memory at most 6 GiB at batch size 32. If speed fails, use B.

### 3.3 Captioning model

The captioning model describes each 3-second window in one sentence. It dominates indexing time, so this choice matters most.

| Option | Parameters | Weights at 16-bit | Fits under 17 GiB? | Notes (all documented unless marked) |
|---|---|---|---|---|
| **A. `Qwen/Qwen3-VL-4B-Instruct`** | 4.44B | 8.3 GiB | Yes | Released October 2025. Plain instruct model with no thinking mode. Loads with `Qwen3VLForConditionalGeneration`. The card describes multi-image and video input. |
| **B. `Qwen/Qwen3.5-4B`** | 4.66B | 8.7 GiB | Yes | Released February 2026. Its card reports 76.9 on VideoMME without subtitles, above the 73.3 it reports for the much larger Qwen3-VL-30B-A3B. Thinking mode is on by default and must be disabled. Native tool calling. New hybrid architecture; the card recommends Linux serving engines for throughput and calls the Transformers path suitable for testing and moderate load. |
| C. `Qwen/Qwen3-VL-8B-Instruct` | 8.77B | 16.3 GiB | No at 16-bit | Weights alone leave under 1 GiB of the 17 GiB cap. The FP8 version cannot be loaded by Transformers and needs vLLM or SGLang, which do not run on native Windows. Usable only with 4-bit loading through bitsandbytes, with unknown speed. |
| D. `google/gemma-4-E4B-it` | 8.00B | 14.9 GiB | Marginal | Under the cap but with about 2 GiB to spare for activations. Not shortlisted. |
| E. `Qwen/Qwen3.5-9B` | 9.65B | 18.0 GiB | No | Over the cap. |
| F. `Qwen/Qwen3-VL-2B-Instruct` | 2.13B | 4.0 GiB | Yes | Speed fallback only. |

**Recommendation.** Start with A and test B against it. B has the stronger published video scores, but its speed through Transformers on Windows is unknown and its default thinking mode is an easy mistake. A is the lower-risk default. Do not judge by impression; apply this rule, fixed now.

**Week 1 bake-off rule.** Run A and B on the same 200 windows from two dev videos, using the settings below.

1. Discard any model that takes more than 1.0 s per window on average, exceeds 17 GiB peak, or produces empty or truncated captions in more than 2% of windows.
2. Both members rate 50 randomly chosen captions per model, with the model name hidden, as "correctly describes what happens" or not.
3. Choose B only if it passes step 1 and its count of correct captions exceeds A's by at least 5 of 50, averaged over the two raters. Otherwise choose A.
4. If both fail step 1 on speed, apply the fallback ladder in section 7 and re-run.

**Caption settings.**

| Setting | Value |
|---|---|
| Window | 3 s long, new window every 1.5 s (40 windows per video minute) |
| Input | 6 frames per window (2 per second), longest side 448 px, passed as images in one user turn. FFmpeg does all decoding, so no video-decoding library is needed. |
| Prompt | "These frames are consecutive moments from a video, in order. In one sentence, describe what happens: who or what is present, what they do, and what changes. Mention any readable on-screen text." |
| Decoding | Greedy (`do_sample=False`), `max_new_tokens=48`. The model cards suggest sampling; greedy is used so that captions are reproducible. |
| Batch | 8 windows per call to start; raise if memory allows |
| Thinking | For option B, pass `enable_thinking=False` in the chat template |

Record the number of visual tokens per window during the bake-off. If it exceeds 400, lower the frame size before lowering the frame count.

### 3.4 Text embedder: `Qwen/Qwen3-Embedding-0.6B`

Used for semantic search over transcript passages and captions. 0.60B parameters, 1.1 GiB at 16-bit (estimated), Apache-2.0, 1,024-dimension output, loads through sentence-transformers (documented). Queries are encoded with `prompt_name="query"`; passages are encoded without a prompt (documented).

| Alternative | Why not |
|---|---|
| `BAAI/bge-m3` (MIT) | Acceptable substitute. Its strength is multilingual text, which this project does not need. |
| `google/embeddinggemma-300m` | Gated behind manual approval and under the Gemma licence (documented). |

### 3.5 Optional reranker: `Qwen/Qwen3-VL-Reranker-2B`

| Option | Evidence | Verdict |
|---|---|---|
| **A. `Qwen/Qwen3-VL-Reranker-2B`** | 2.13B parameters, 4.0 GiB at 16-bit (estimated). Takes a text query and a video or image document and returns a relevance score, mapped to 0–1 with a sigmoid. Loads as a sentence-transformers `CrossEncoder`. Apache-2.0 (all documented). | **Recommended** |
| B. Reuse the captioning model with a yes/no prompt | No extra download, but needs custom code to read token probabilities. | Fallback |
| C. `BAAI/bge-reranker-v2-m3` over caption and transcript text | Cheapest, but never looks at the video, so it cannot correct a wrong caption. | Not chosen |

- **Use.** Re-score the top 10 candidate windows per query, 4 frames per window.
- **Keep rule, decided on dev only.** Keep the reranker if both hold on the team-set dev queries: recall at rank 1 (IoU at least 0.5) rises by at least 3 percentage points with a paired bootstrap 95% interval that excludes zero, and median added latency is at most 3 s per query. Otherwise drop it and report the dev result.
- **Second use.** The reranker's score for the top window is an absolute relevance score, so it is also tested as an abstention feature (abstainer A3 in section 4.4).

### 3.6 Agent model (only if the extension goes ahead): `Qwen/Qwen3.5-4B`

Its card documents native tool calling. It is loaded only in agent mode. With the frame encoder (2.1 GiB) and text embedder (1.1 GiB) loaded, the total is about 11.9 GiB of weights (estimated), so the reranker (4.0 GiB) must be unloaded in agent mode to stay under 17 GiB. If the bake-off already chose option B for captioning, this is the same model and needs no extra download.

### 3.7 Peak memory by stage

| Stage | Models loaded | Weights (estimated) | Gate (to measure) |
|---|---|---|---|
| Transcribe | Whisper turbo | 1.5 GiB | Peak at most 7 GiB |
| Embed frames | SigLIP 2 so400m | 2.1 GiB | Peak at most 6 GiB |
| Caption | Captioning model | 8.3 to 8.7 GiB | Peak at most 17 GiB |
| Query | SigLIP 2, text embedder, reranker | 7.2 GiB | Peak at most 12 GiB |
| Query, agent mode | SigLIP 2, text embedder, agent model | 11.9 GiB | Peak at most 17 GiB |

### 3.8 Libraries and versions

Versions are the latest on PyPI on 14 November 2026 (documented). Pin them exactly in `requirements.txt`.

| Library | Version | Note |
|---|---|---|
| Python | 3.12 | Transformers 5.23 and PyTorch 2.15 both need 3.11 or newer |
| torch | 2.15.0 | Install the CUDA build from the PyTorch index; Windows wheels exist |
| transformers | 5.23.0 | See the risk below |
| sentence-transformers | 6.3.0 | |
| faster-whisper / ctranslate2 | 1.2.1 / 4.9.0 | ctranslate2 publishes Windows wheels |
| faiss-cpu | 1.16.0 | The GPU builds of FAISS publish Linux wheels only. CPU is enough: each video has under 2,000 vectors, so exact search takes milliseconds (estimated). |
| bm25s | 0.3.16 | Replaces `rank-bm25`, whose last release was February 2022 |
| fastapi / pydantic | 0.162.7 / 2.14.3 | |
| streamlit / plotly | 1.67.0 / 7.1.0 | |
| scikit-learn | latest stable | Logistic regression for the confidence model. Version not checked; pin whatever installs in Week 1. |
| FFmpeg | system binary, called with `subprocess` | Replaces `ffmpeg-python`, whose last release was July 2019 |
| vLLM | not used | Linux only |

**Version risk.** The Qwen3-VL and reranker cards were written for Transformers 4.57, and the current release is 5.23, a major version later. Unit Y1 loads every chosen model once under 5.23. If one fails, pin Transformers to the newest version that loads all of them and record it.

**Model pins.** Record the revision hash of every downloaded model in `meta.json`. Hashes on 14 November 2026: Qwen3-VL-4B-Instruct `ebb281ec`, Qwen3.5-4B `851bf6e8`, siglip2-so400m-patch16-384 `dd658faa`, Qwen3-Embedding-0.6B `97b0c614`, Qwen3-VL-Reranker-2B `4bd860ac` (documented, first 8 characters).

---

## 4. Data and evaluation protocol

The claim "the system says no confident match instead of guessing" is only as good as this section. Three rules carry it:

1. Videos are split, not queries, so no test video is ever seen during tuning.
2. Every choice is made on dev videos, except the abstention threshold, which is chosen on calibration videos.
3. Test labels are read once, by one script, after the configuration is frozen.

### 4.1 Team set

**Videos.** 30 English videos of 3 to 8 minutes each, about 150 minutes in total. Ten are speech-heavy (lectures, talks), ten are mixed (how-to videos, vlogs), ten are action-heavy with little speech (sport, driving, cooking without narration). Use Creative Commons or self-recorded videos so they can be shown in the demo, and record each URL and licence in `data/tl/videos.csv`.

**Split.** Assign videos to splits with a seeded shuffle within each category.

| Split | Videos | Answerable | Unanswerable | Total | Used for |
|---|---|---|---|---|---|
| Dev | 10 | 80 | 40 | 120 | All tuning and every keep-or-drop decision |
| Calibration | 8 | 64 | 32 | 96 | The abstention threshold only |
| Test | 12 | 96 | 48 | 144 | One final run |
| **Total** | **30** | **240** | **120** | **360** | |

**Query quotas.** Each video gets 8 answerable and 4 unanswerable queries.

- *Answerable, by type.* Within each split, each of the four types makes up at least 20% of answerable queries: **spoken** (the answer is only in the speech), **visual** (visible in a single frame), **temporal** (an action or event that needs several frames), **mixed** (needs speech and picture together). Videos without speech get more visual and temporal queries; lectures get more spoken ones.
- *Multi-event queries.* A query of the form "X and then Y" is tagged `multi_event=true`. Each dev video has two, giving 20 in dev, which the agent gate in section 9 needs. Each calibration and test video has at least one.
- *Unanswerable, by kind.* Per video: two **near-miss** queries (about people, objects or topics that do appear, but an event or statement that does not occur), one **swap** (an answerable query taken from another video in the same category and confirmed absent here), one **out-of-domain** (unrelated to the video). Near-miss queries are the hard case and are half of all negatives on purpose. If negatives were all unrelated, rejection would look easy and the claim would be inflated.

**Writing rules.**

1. The author writes queries by watching the video, with no access to any system output for it: no index, captions, system transcript or search results.
2. All calibration and test queries are written and frozen before the pipeline first runs on any calibration or test video. Dev queries are also written before the dev videos are indexed. A dev query added later is allowed but must be marked `late=true`.
3. For each answerable query, the author records one or more gold windows as start and end times in seconds, the query type, and whether it is multi-event.
4. Each member writes the queries for half of the videos in every split.

**Verification.**

- *Unanswerable queries, 100%.* The other member watches the video and confirms that nothing satisfies the query. If they disagree, the query is rewritten or dropped and replaced.
- *Answerable queries, 25% sample per split.* The other member labels the windows independently without seeing the first labels. Report the mean IoU between the two annotators and the share of pairs with IoU at least 0.5. This is the human agreement ceiling and goes in the report beside the system's numbers. Also report Cohen's kappa for the query-type tag on the same sample.

**Label format.** One JSON object per line in `data/tl/queries_{dev,cal,test}.jsonl`:

```json
{"qid": "tl-0042", "video_id": "tl-v07", "query": "when does the lecturer write the update rule on the board",
 "answerable": true, "gold_windows": [[212.0, 231.5]], "qtype": "mixed", "multi_event": false,
 "neg_type": null, "author": "YS", "verifier": "MJ"}
```

For unanswerable queries, `answerable` is false, `gold_windows` is empty and `neg_type` is `near_miss`, `swap` or `ood`.

**Estimated labelling effort:** about 39 person-hours, shared across Weeks 1 and 2. Basis: 360 queries at 4 minutes each, 120 negative checks at 3 minutes, 60 double labels at 3 minutes, and 360 Charades negative checks at 1 minute (section 4.2). If labelling slips, cut videos to 24 (8 / 6 / 10) and keep 12 queries per video. Do not cut the unanswerable share.

### 4.2 Public benchmark slice: Charades-STA

**Why Charades-STA and not QVHighlights.** Figures below were computed from the annotation files listed in section 13 or read from the download servers (documented). Charades-STA video lengths come from the `duration` field of the Moment of Untruth copy of the test annotations.

| | Charades-STA | QVHighlights |
|---|---|---|
| Raw video download | 16.3 GB at 480p | 143.7 GB |
| Scorable labels | Test split: 3,720 queries, 1,334 videos | Validation only: 1,550 queries, 1,519 videos. Test labels are withheld. |
| Video length | Mean 29.5 s | Mean 149.6 s |
| Video to caption for about 150 queries | About 27 minutes | About 367 minutes |
| Speech | None useful | Yes |
| Gold moments per query | One, mean length 8.0 s | 1.8 on average; 34% of queries have several |

Charades-STA is chosen because:

- every baseline and ablation can be run on it within the compute budget;
- its test labels are public;
- its queries are all short actions, which is exactly what the caption path exists to find;
- published zero-shot results and a published set of unanswerable queries both exist for it.

What it cannot show: it has no useful speech and its videos are 30 seconds long, so it says nothing about the transcript path, spoken-query routing or search in long videos. Those claims rest on the team set alone, and the report must say so.

**Licence (documented).** Charades is under a non-commercial licence from the Allen Institute for AI. It forbids redistributing the data and allows publishing only short segments or stills in an academic publication. So: never commit Charades videos or frames to the repository, and use team-set videos for the demo.

**Unanswerable queries.** The paper "Moment of Untruth" (Flanagan, Damen and Wray, WACV 2025) publishes negative queries for Charades-STA in its GitHub repository (documented):

| File | Content |
|---|---|
| `charades_id_neg_test.jsonl` | 3,720 in-domain negatives across all 1,334 test videos. Each is a real Charades query reassigned to a different video. |
| `charades_ood_neg_test.jsonl` | Out-of-domain negatives, for example wildlife scenes, written by a language model. The file covers 931 of the test videos. |
| Matching `_train` files | The same two kinds for the training split. |

Two cautions.

- The in-domain negatives were assigned automatically, using sentence similarity, and were not checked by people (documented in the paper, section 3.3). Many Charades videos show the same actions, such as opening a door, so some "negatives" will in fact occur in their assigned video. Both members must check every sampled in-domain negative against its video. Drop the ones that occur, draw replacements from the same file with the same seed, and report how many were dropped.
- The repository has no licence file. Use the files for evaluation, cite the paper, and do not redistribute them. If that is a concern, regenerate in-domain negatives with the same reassignment method.

**Slice.** Sample by video, never by query, with seed `20261114`.

| Split | Source | Videos | Answerable | Negatives | Used for |
|---|---|---|---|---|---|
| Dev | Charades-STA train | 30 | About 70 | 60 in-domain, 30 out-of-domain | Tuning on Charades-style video |
| Calibration | Charades-STA train, different videos | 30 | About 70 | 60 in-domain, 30 out-of-domain | The Charades threshold only |
| Test | Charades-STA test | 60 | About 167 | 120 in-domain, 60 out-of-domain | One final run |

- Every query of a sampled video is included. The train split averages 2.32 queries per video and the test split 2.79 (documented), hence the "about" figures. Record the exact counts.
- Each video gets two verified in-domain negatives and one out-of-domain negative. If a video has no out-of-domain row, assign one at random from the same file with the seed.
- Train and test videos are disjoint in Charades-STA, so dev and calibration cannot leak into test.
- The answerable test count, about 167, is inside the proposal's range of 100 to 200.

**Comparability.** Results on 60 sampled test videos are not directly comparable with published results on all 3,720 test queries. Report slice results with confidence intervals and quote published figures as context only, in a separate table with that caveat.

Published full-test results for retrieval (documented; VTG-GPT paper, Table 3):

| Method | Setting | R@1, IoU 0.5 | R@1, IoU 0.7 | Mean IoU |
|---|---|---|---|---|
| Moment-DETR | Supervised | 52.07 | 30.59 | 45.54 |
| VTG-GPT (frame captions matched to the query as text) | Zero-shot | 43.68 | 25.94 | 39.81 |
| Luo et al. | Zero-shot | 42.93 | 20.13 | 37.92 |
| UniVTG | Zero-shot | 25.22 | 10.03 | 27.12 |

Published results for rejection (documented; Moment of Untruth, Tables 1 and 7 and abstract):

| Method | Rejection accuracy, in-domain | Rejection accuracy, out-of-domain |
|---|---|---|
| Threshold on the saliency score of a model not trained on negatives (three models) | 9.8% to 15.5% | 6.3% to 23.0% |
| SVM on UniVTG saliency scores | 35.9% | 50.4% |
| UniVTG-NA, trained with negatives | About 98% averaged over datasets and negative kinds | |

Two readings. VTG-GPT is the closest published design to this project's caption path, so it sets a realistic expectation for retrieval. And rejecting Charades negatives by thresholding an untrained score has worked badly in published work, so a modest rejection result here would be normal and should be reported plainly.

**Stretch option, not in the plan of record.** All 1,334 test videos total about 11 hours of video, or roughly 26,000 caption windows. At the 1.0 s gate that is about 7.3 GPU-hours (estimated). Running it would make the retrieval numbers directly comparable. Consider it only after the Week 7 freeze.

### 4.3 What may be decided where

| Decision | Dev | Calibration | Test |
|---|---|---|---|
| Model choices, caption prompt, frame rate, window length and stride | Yes | No | No |
| Transcript passage size, BM25 and embedding mix | Yes | No | No |
| Score standardisation statistics, fusion weights, router rules | Yes | No | No |
| Confidence features and confidence model weights | Yes | No | No |
| Keep or drop: reranker, agent | Yes | No | No |
| Abstention threshold, one per dataset and per system variant | No | Yes | No |
| Anything at all | | | No |

Rules that follow from the table:

- Every baseline and ablation gets its own threshold, chosen on calibration by the same rule, so comparisons are fair.
- The team set and the Charades slice have separate thresholds, each from its own calibration split. As a secondary robustness result, also report the Charades test with the team-set threshold.
- There are two freezes. The **indexing freeze** at the end of Week 3 fixes everything that shapes the index: models, caption prompt, frame rate, window and stride. Calibration videos are indexed only after it, in Week 4. The **final freeze** in Week 7 fixes everything else, and test videos are indexed only then. Looking at captions of calibration or test videos and then changing the prompt would be leakage.
- If an indexing setting has to change after Week 3, re-index the calibration videos and recompute every threshold before the final freeze.

**Custody.** The proposal gives Murtaza both confidence calibration and the evaluation datasets. One person tuning the threshold while holding the test labels is the exact risk this plan removes. So:

- Yesaullah is the test custodian. Test label files live outside the shared working tree, in a private folder or a separate private repository.
- Murtaza does all tuning on dev and chooses thresholds on calibration. He does not open test label files.
- Only `vi/eval/run_final.py` reads test labels. It refuses to run unless the configuration hash matches `FROZEN.json`, and it appends the time, user and hash to `results/test_access.log`.
- One final run is allowed. A re-run is permitted only to fix a crash that changes no scoring code, and every re-run is logged and disclosed in the report.

**Freeze procedure (Week 7, day 1).**

1. Write `FROZEN.json` with the git commit, the SHA-256 of `config.yaml`, of each label file and of `thresholds.json`, and the list of system variants to run.
2. Tag the commit `v-freeze`.
3. Index the test videos under these exact settings.
4. Yesaullah runs `run_final.py` once.

The list of variants in the final run is fixed at the freeze, so no variant can be added after seeing test results.

### 4.4 Confidence score and threshold

**The trap to avoid.** If scores are rescaled per query, for example min-max normalised so the best candidate always scores 1.0, the top score carries no information about whether anything matched. Abstention would then be impossible, however the threshold is set. Ranking may use any normalisation, but the confidence score must be built from absolute, un-rescaled quantities.

**Features**, computed for the top-ranked window of each query:

| Feature | Definition |
|---|---|
| `f_frame` | Highest SigLIP sigmoid probability between the query and any frame in the window |
| `f_caption` | Highest raw cosine similarity between the query and any caption in the window |
| `f_transcript` | Highest raw cosine similarity between the query and any transcript passage overlapping the window; 0 if none |
| `f_bm25` | Highest raw BM25 score in the window, divided by the number of query terms |
| `f_margin` | Raw fused score of the top window minus that of the second window |
| `f_agree` | Number of search paths, 0 to 3, whose own top 5 overlaps the top window |
| `f_rerank` | Reranker sigmoid score of the top window (only if the reranker is kept) |

**Confidence model.** Logistic regression with L2 penalty (`C=1.0`), fitted on dev queries. The label is 1 when the query is answerable and the top window has IoU at least 0.5 with a gold window, and 0 otherwise. So confidence estimates the probability that the top result is correct, and every unanswerable query is a 0 by definition. One model is fitted per dataset.

**Abstainers compared.**

| ID | Rule |
|---|---|
| A0 | Never abstain |
| A1 | Threshold on one number: the raw fused score of the top window |
| A2 | Threshold on the logistic-regression confidence, without `f_rerank`. This is the main system. |
| A3 | A2 plus `f_rerank` (only if the reranker is kept) |

**Threshold rule, fixed in advance.** On the calibration split, choose the lowest threshold at which the system answers at most 20% of the unanswerable queries. In words: of the questions that have no answer, at most about one in five gets an answer anyway. Then report what that threshold costs in false abstentions.

- This rule fixes the false-answer side because that is the promise the product makes to the user.
- The team-set calibration split has 32 unanswerable queries, so the threshold can only be placed in steps of about 3 percentage points, and the test false-answer rate will not land exactly on 20%. Report the test rate with its interval. Do not adjust the threshold afterwards.
- The alternative rule, maximising balanced accuracy on calibration, is computed and reported as a secondary result (see section 11).

"Calibrated" in the proposal therefore means "threshold chosen on held-out calibration videos". It does not mean the probabilities are well calibrated. A five-bin reliability table on the test set is reported as description only.

---

## 5. Baselines and ablations

### 5.1 Retrieval systems

All systems share the same index and return up to 5 windows per query.

| ID | System | Purpose |
|---|---|---|
| B0 | Random window, with length drawn from dev gold-window lengths; mean of 1,000 draws | Chance level |
| B1 | Transcript only: BM25 plus embeddings | Proposal baseline |
| B2 | Frames only: SigLIP 2 | Proposal baseline |
| B3 | Captions only: BM25 plus embeddings | Proposal baseline; tests the temporal claim |
| B4 | Equal-weight fusion of the three paths, weights never tuned | Proposal's fixed-weight baseline |
| B4b | Fusion with one global weight vector tuned on dev | Separates the value of tuning from the value of routing |
| B5 | Routed fusion: rule-based router selects per-type weights tuned on dev | Main system |
| B6 | Oracle routing: uses the gold query type | Upper bound on what routing can add; team set only |
| B7 | B5 plus reranker | Only if the reranker is kept |
| B8 | Agent | Only if it passes both gates in section 9 |

On the Charades slice the transcript is expected to be empty. Run Whisper anyway, report the share of videos with any transcript, and mark B1 as not applicable if that share is under 10%.

**Shared mechanics, so that the builder does not improvise.**

- *Text paths.* Transcript passages are consecutive Whisper segments merged until the passage is at least 10 s long, with a new passage starting at every segment. Passages and captions are each searched by BM25 and by embedding similarity.
- *Timeline.* Each path's score is projected onto a 1-second timeline: the score of a second is the highest score of any unit covering it.
- *Standardisation.* Each path's raw score is standardised with a mean and standard deviation computed once on dev and then frozen. Scores are never rescaled per query (section 4.4).
- *Fusion.* The fused score of a second is the weighted sum of the standardised path scores.
- *Windows.* Take local maxima of the fused timeline as seeds. Grow each window outward while the score stays above a fraction `alpha` of the peak (default 0.7, tuned on dev). Remove windows that overlap a higher-scoring one with IoU above 0.5. Return the top 5.
- *Router.* Keyword rules assign one of spoken, visual, temporal or mixed. The rules are written from dev queries only, and the rule file is frozen with the configuration.

### 5.2 Ablations

| ID | What changes | Where it is run |
|---|---|---|
| X1 | Text search by BM25 only, embeddings only, or both | Dev for selection; all three in the final test run |
| X2 | Fusion by weighted sum of standardised scores, or by reciprocal rank fusion | Dev; both in the final test run |
| X3 | Merging neighbouring hits into windows, on or off | Dev; both in the final test run |
| X4 | Abstainers A0 to A3 | Calibration for thresholds; all in the final test run |
| X5 | Rejection accuracy by negative kind: near-miss, swap, out-of-domain; in-domain and out-of-domain on Charades | Final test run |
| X6 | Caption stride 1.5 s against 3 s | Dev only |
| X7 | Frame rate 1 against 2 per second | Dev only |

X6 and X7 change the index, so they run on dev only, before the indexing freeze, and are reported as dev results labelled as selection data. They are not test evidence. X6 needs no new captions: the 3 s stride is every second window of the 1.5 s index. X7 needs the dev frames embedded again at 2 per second.

### 5.3 Primary comparisons, declared now

Only these three comparisons support headline claims. Everything else is secondary and reported as description.

| # | Claim | Comparison | Where | Planned power |
|---|---|---|---|---|
| C1 | Abstention is honest | Rejection accuracy and false-abstain rate of A2 at the frozen threshold; and A2 against A1 on area under the risk-coverage curve | Both test sets | Adequate on the Charades slice; coarse on the team set |
| C2 | Captions find actions that single frames miss | B3 against B2 on R@1 at IoU 0.5 | Charades test (about 167 queries, all actions) | Can resolve differences of roughly 10 points or more (estimated) |
| C3 | Routing helps | B5 against B4b on R@1 at IoU 0.5 | Team-set test (96 answerable) | Underpowered: only differences of roughly 15 points or more can be resolved (estimated) |

"Inconclusive" is an acceptable reported outcome for any of these. For C3 it is the likely one unless routing helps a great deal. The report must not turn an inconclusive result into a positive one.

---

## 6. Metrics and uncertainty

### 6.1 Metrics

| Group | Metric | Definition |
|---|---|---|
| Retrieval | R@1 and R@5 at IoU 0.5 (primary); also 0.3 and 0.7 | Share of answerable queries with a returned window in the top K whose IoU with a gold window reaches the level. With several gold windows, use the best match. |
| | MRR | Mean of 1/rank of the first window with IoU at least 0.5 within the top 5; 0 if none |
| | Mean IoU | Mean IoU of the top window, over answerable queries |
| | Start-time error | Median and interquartile range of the absolute difference between predicted and gold start, for top windows that overlap a gold window |
| Abstention | Rejection accuracy | Share of unanswerable queries on which the system abstains |
| | False-abstain rate | Share of answerable queries on which the system abstains |
| | Balanced accuracy | Mean of rejection accuracy and (1 minus false-abstain rate). This replaces the proposal's "abstention accuracy". |
| | Risk-coverage curve and its area | Sort all queries by confidence. At each threshold, coverage is the share answered, and risk is the share of answered queries that are wrong: an unanswerable query that was answered, or an answerable one whose top window has IoU below 0.5. Lower area is better. |
| End to end | Abstention-aware R@1 | As R@1, but an abstention on an answerable query counts as a miss |
| Routing | Accuracy, macro-F1, confusion matrix | Predicted against gold query type; team set only |
| Cost | Indexing seconds per video minute, by stage; peak GPU memory by stage; query latency p50 and p95 | Latency measured warm, over all test queries |

Three reporting rules.

- Report retrieval metrics twice: with abstention off, which is comparable with published work, and abstention-aware, which is what a user experiences.
- Plain accuracy over answerable and unanswerable queries together is not a headline number. It depends on the share of unanswerable queries in the set, which here is fixed by design at one third on the team set and about one half on the Charades slice.
- The risk-coverage area depends on the same mix. State the mix beside every such number and never compare areas across the two datasets.

### 6.2 Uncertainty

- **Rates** (rejection accuracy, false-abstain rate, R@K, routing accuracy): Wilson 95% interval, with the count shown.
- **Every headline number and every comparison**: bootstrap that resamples videos with replacement and takes all queries of each sampled video, 10,000 resamples, fixed seed, 95% percentile interval. Queries from the same video are not independent, so resampling queries directly would give intervals that are too narrow.
- **Comparisons between two systems**: paired. Both systems are scored on the same resampled videos, and the difference is reported with its interval. A difference is called supported only if the interval excludes zero.
- **Small subsets**: per-type results on the team set have about 24 queries each. They are shown with intervals and labelled descriptive. No claim rests on them.
- **Determinism**: all models decode greedily and indexes are cached, so evaluation is repeatable given an index. The index is built once. Variation from rebuilding it is not measured, and the report says so.

**Planned precision.** Wilson 95% half-widths at the planned counts (estimated). Video-level bootstrap intervals will be somewhat wider.

| Quantity | Count | Half-width if the rate is 80% (rejection) or 50% (recall) |
|---|---|---|
| Team-set test, rejection accuracy | 48 | ±11.3 points |
| Team-set test, R@1 | 96 | ±9.8 points |
| Team-set test, R@1 for one query type | 24 | ±18.6 points |
| Charades test, rejection accuracy, all negatives | 180 | ±5.8 points |
| Charades test, rejection accuracy, in-domain only | 120 | ±7.1 points |
| Charades test, R@1 | About 167 | ±7.5 points |

Consequences:

- The team set alone supports only a coarse abstention claim, about ±11 points. The Charades slice carries the tighter one.
- The team-set test has 12 videos. A video-level bootstrap over 12 clusters is itself approximate, and the report must say this.

---

## 7. Compute budget

**Volumes.** Team set 150 minutes; Charades slice 120 videos at about 30 s, so 60 minutes. Total 210 minutes of video. At one window every 1.5 s that is 8,400 caption windows (6,000 plus 2,400), and 12,600 frames at 1 per second.

| Stage | Volume | Rate | Time per full pass | Basis |
|---|---|---|---|---|
| Decode, sample frames (CPU) | 210 min of video | To measure | Under 10 min | Estimated |
| Transcribe | 210 min of audio | Gate: at least 20 times real time | At most 10.5 min | Documented: large-v2 ran 13 min of audio in 17 s (46 times real time) on an RTX 3070 Ti at batch 8; turbo is a smaller model |
| Embed frames | 12,600 frames | Gate: at least 20 frames/s | At most 10.5 min | To measure |
| **Caption** | **8,400 windows** | **Gate: at most 1.0 s/window** | **At most 2.3 h** | To measure; see the note below |
| Embed text | About 9,500 short texts | To measure | Under 2 min | Estimated |
| Query, no reranker | Per query | Gate: p95 at most 1.0 s | | Estimated: two short text encodings plus search over under 2,000 vectors |
| Query, with reranker | 10 windows per query | Gate: median at most 4 s | | To measure |

**Basis for the caption gate.** No documented throughput exists for these models on this card. The only reference found is a third-party listing of about 53 tokens per second for an 8B model at 4-bit on an RTX 4000 Ada, which was not verified at its source. A caption of about 40 tokens would then take under 1 s to generate unbatched, plus image encoding. The 1.0 s gate is therefore plausible but unproven, which is why the bake-off in 3.3 runs in Week 1, before anything depends on it.

**GPU-hours for the whole project (estimated).**

| Activity | GPU-hours |
|---|---|
| Week 1 benchmarks and bake-off | 2 |
| Captioning dev videos: 65 min of video, 2,600 windows, 43 min per pass, 5 passes | 3.6 |
| Final captioning pass over all 210 min | 2.3 |
| Transcription and frame embedding, 3 passes | 1 |
| Reranker on dev, calibration and test | 1.5 |
| Evaluation runs without reranker | Under 0.5 |
| **Core total** | **About 11** |
| Agent, if built: dev runs plus one final run over 491 test queries capped at 60 s each | Up to 12 more |

**Disk (estimated): about 70 GB free.** Charades archive 16.3 GB plus the same again extracted (delete everything but the 120 sampled videos afterwards), model weights about 30 GB, caches about 2 GB.

**Fallback ladder if captioning misses the 1.0 s gate.** Apply in order and stop at the first that passes.

1. Raise the batch size until memory reaches 15 GiB.
2. Reduce to 4 frames per window and a longest side of 336 px.
3. Change the stride to 3 s, which halves the window count.
4. Switch to `Qwen/Qwen3-VL-2B-Instruct`.
5. If still over, accept up to 2.0 s per window and cut the dev captioning passes from 5 to 3.

Record which step was used. It is a reportable limitation.

---

## 8. Implementation units

Week 1 is assumed to start on Monday 16 November 2026. Shift all dates together if the course calendar differs.

### 8.1 Contract both members build against

**Repository layout.**

```
vi/
  config.py  schema.py
  ingest/   ffmpeg.py  asr.py  frames.py  captions.py  build_index.py
  search/   lexical.py  dense.py  router.py  fusion.py  windows.py  confidence.py  rerank.py  pipeline.py
  eval/     metrics.py  bootstrap.py  validate_labels.py  run_eval.py  calibrate.py  run_final.py
  api/      main.py
  ui/       app.py
  agent/    loop.py  tools.py
scripts/    gpu_check.py  bench_models.py  make_stub_index.py  sample_cs_slice.py
data/       tl/  cs/
index/      <video_id>/
results/
tests/
```

**Index, one folder per video (`index/<video_id>/`).** Search always runs inside one video.

| File | Content |
|---|---|
| `index.sqlite` | Tables: `video(video_id, path, duration_s, sha256, dataset, split)`; `transcript_units(unit_id, start_s, end_s, text)`; `frames(frame_id, t_s, jpg_path)`; `caption_units(unit_id, start_s, end_s, text)` |
| `frames.npy`, `transcript.npy`, `captions.npy` | float32 arrays, L2-normalised; row number equals `frame_id` or `unit_id` |
| `meta.json` | Model IDs and revision hashes, library versions, configuration hash, per-stage seconds and peak memory |

**Search interface.** `search(video_id: str, query: str, cfg) -> SearchResult`

```json
{"abstained": false, "confidence": 0.81, "route": "temporal",
 "results": [{"start_s": 41.0, "end_s": 47.5, "score": 2.31,
              "path_scores": {"transcript": 0.0, "frame": 1.2, "caption": 2.9},
              "evidence": {"frames": ["f0041.jpg"], "transcript": [], "captions": ["a person opens the door and walks in"]}}],
 "timings_ms": {"encode": 35, "search": 4, "rerank": 0}}
```

When `abstained` is true, `results` still holds the candidates so the evaluation can score retrieval with abstention off. The interface hides them and shows "no confident match".

### 8.2 Yesaullah: indexing, GPU, API, test custody

| ID | Week | Unit | Acceptance check |
|---|---|---|---|
| Y1 | 1 | Environment and GPU check | `python scripts/gpu_check.py` writes `results/env.json`. Pass: section 2 conditions hold, and every chosen model loads and produces one output under the pinned versions. |
| Y2 | 1 | Model benchmarks and bake-off | `python scripts/bench_models.py` writes `results/bench.json` with speed and peak memory for each model. Pass: every gate in section 3 is met or a fallback is recorded; the captioning choice follows the rule in 3.3; the Whisper choice follows 3.1. |
| Y3 | 2 | FFmpeg ingest with cache | `python -m vi.ingest.build_index --video X --stages decode`. Pass: a second run makes no FFmpeg call and finishes in under 5 s. A file with no audio track, a corrupt file and a zero-length file each produce a clear error and do not crash. |
| Y4 | 2 | Transcription | Pass: at least 20 times real time; word error rate at most 15% on the 5-minute hand-corrected reference; segment times never decrease and never exceed the video duration. |
| Y5 | 2 | Frame embeddings | Pass: at least 20 frames/s; every vector norm within 0.001 of 1; a unit test fails if query text is not lowercased and padded to 64 tokens; for 20 hand-written descriptions of dev frames, the described frame is among the 5 highest-scoring frames of its video in at least 14 cases. |
| Y6 | 2–3 | Captions | Pass: at most 1.0 s per window, or a recorded fallback; peak at most 17 GiB; no empty captions; under 2% reach the token limit; at least 40 of 50 random dev captions rated correct by both members. |
| Y7 | 3 | Full index build, dev videos only; then the indexing freeze | `python -m vi.ingest.build_index --split dev`. Pass: all 10 team-set dev and 30 Charades dev videos indexed; `meta.json` complete; a rebuild from cache gives byte-identical `.npy` files; ablations X6 and X7 run and the indexing settings recorded in `config.yaml` as frozen. |
| Y7b | 4 | Index calibration videos | `python -m vi.ingest.build_index --split cal`. Pass: 8 team-set and 30 Charades calibration videos indexed under the frozen indexing settings, in time for M6. |
| Y8 | 4 | FastAPI service | Endpoints `POST /videos`, `GET /videos/{id}/status`, `POST /search`. Pass: `pytest tests/test_api.py` passes against the stub index; p95 search latency at most 1.0 s on dev queries, warm, without reranker. |
| Y9 | 5 | Reranker experiment | Pass: `results/rerank_dev.json` holds the paired comparison and the keep-or-drop decision under the rule in 3.5. |
| Y10 | 6 | Agent, only if Gate A passes; otherwise GPU profiling and the cost section of the report | See section 9. |
| Y11 | 7 | Freeze and final run | Pass: `FROZEN.json` written, commit tagged, `run_final.py` run once, `test_access.log` has exactly one entry. |
| Y12 | 7–8 | Reliability tests and clean-environment reproduction | Pass: in a fresh virtual environment, the documented commands reproduce the final metrics from the cached index to 3 decimal places. |

### 8.3 Murtaza: retrieval, calibration, evaluation code, interface

| ID | Week | Unit | Acceptance check |
|---|---|---|---|
| M1 | 1 | Schema, stub index, metrics library | `python scripts/make_stub_index.py` creates 2 synthetic videos in the contract format. `pytest tests/test_metrics.py` passes, including: IoU of [0, 10] and [5, 15] equals 1/3; Wilson interval for 40 of 50 equals [0.670, 0.888] to 3 decimals; the bootstrap interval of a constant metric has zero width; the same seed gives the same interval. |
| M2 | 1 | Data manifests and Charades sampling | `python scripts/sample_cs_slice.py --seed 20261114`. Pass: a second run produces a file with the same SHA-256; an assertion confirms no video appears in two splits, for both datasets. |
| M3 | 2–3 | Evaluation harness | `python -m vi.eval.run_eval --split dev --systems B0` runs end to end on the stub index and writes `results/dev_B0.json` with every metric in section 6 and both interval types. The harness must exist before any real search code, so that every later unit is measured the same way. |
| M4 | 3 | Three search paths | `run_eval --split dev --systems B0,B1,B2,B3`. Pass on team-set dev: each of B1, B2 and B3 beats B0 on R@1 at IoU 0.5 on its own query type (spoken, visual, temporal). If one does not, stop and investigate before fusion. |
| M5 | 4 | Router, fusion, windows | `run_eval --split dev --systems B4,B4b,B5,B6`. Pass: results written with routing accuracy and confusion matrix. The main system is whichever of B4b and B5 scores higher on dev R@1 at IoU 0.5. If B4b wins, B5 is still run on test and reported. |
| M6 | 4 | Confidence model and thresholds | `python -m vi.eval.calibrate` fits on dev, picks thresholds on calibration and writes `thresholds.json`. Pass: a unit test confirms the script raises an error if given any test path; the calibration false-answer rate is at most 0.20 for every variant; a unit test confirms that halving every raw similarity score of a query lowers its confidence. A score rescaled per query would not change, so this proves the confidence uses raw scores. |
| M7 | 5 | Streamlit interface | Pass, all 8 checked by hand on a dev video: upload and index with progress shown; text query returns ranked windows; clicking a result plays from its start time; each result shows frames, transcript excerpt and caption; an unanswerable query shows "no confident match"; the route and path scores are visible; the dashboard shows the dev results table with intervals; a second upload of the same file uses the cache. |
| M8 | 5 | Dev and calibration report | `python -m vi.eval.run_eval --split dev,cal --systems all --report`. Pass: `results/dev_report.md` is generated with every table that the final report will contain. |
| M9 | 6 | Error analysis on dev | Pass: 30 dev failures each assigned to one category (speech error, caption error, frame miss, fusion error, window boundary, router error, label problem), with counts. |
| M10 | 7 | Final tables and plots from the frozen results | Pass: every number in the report is produced by a script from `results/final_*.json`. No number is typed by hand. |

### 8.4 Shared

| ID | Week | Unit | Acceptance check |
|---|---|---|---|
| S1 | 1–2 | Collect the 30 team-set videos and write all queries, following 4.1 | `python -m vi.eval.validate_labels`. Pass: counts per split match the table in 4.1; each type is at least 20% of answerable queries in each split; 20 multi-event queries in dev; every unanswerable query has a verifier different from its author; 25% of answerable queries are double-labelled; `MANIFEST.json` holds a SHA-256 for every label file. |
| S2 | 2 | Verify Charades in-domain negatives | Pass: every sampled negative checked by both members; dropped ones replaced; the number dropped is recorded in `data/cs/negatives_audit.csv`. |
| S3 | 8 | Report, setup instructions, demo on pre-indexed team-set videos, recorded backup | Pass: the report contains every item in section 12 and the limitations named in sections 4.2, 6.2 and 7. |

### 8.5 Changes to the proposal's schedule

| Proposal | This plan | Reason |
|---|---|---|
| Full evaluation in Week 7 | Evaluation harness in Weeks 2–3; dev and calibration results complete by end of Week 5; only the single test run in Week 7 | The agent decision at the end of Week 5 needs evaluation results, and every unit needs the same measuring stick |
| Murtaza labels queries alone in Weeks 1–2 | Both members label | About 39 person-hours of labelling is too much for one person alongside M1 to M3 |
| Murtaza owns calibration and the evaluation datasets | Yesaullah holds the test labels and runs the final evaluation | Section 4.3 |

---

## 9. Agentic extension: go or no-go

The agent plans each search with the tools `search_transcript`, `search_frames`, `search_captions`, `resample_frames` and `inspect_clip`. It targets multi-event queries. Both gates use dev data only.

### Gate A: entry, checked on Sunday 20 December 2026 (end of Week 5)

Start the agent only if every item passes.

| # | Check | Pass |
|---|---|---|
| 1 | Dev and calibration evaluation | `results/dev_report.md` exists, with B0 to B5 on both datasets, and `thresholds.json` exists |
| 2 | Interface | All 8 checks of M7 pass |
| 3 | Service | `pytest tests/test_api.py` passes |
| 4 | Labels | All label files frozen with hashes, including 20 multi-event dev queries |
| 5 | Tool-calling smoke test | The agent model returns a tool call that parses and matches the tool schema on at least 18 of 20 scripted prompts, with median turn time at most 5 s and peak memory at most 17 GiB with the query-time models loaded |
| 6 | Room to improve | The best fixed pipeline answers at most 12 of the 20 multi-event dev queries correctly (top window IoU at least 0.5). If it already answers more, the agent has too little to gain. |

If any item fails: no agent. Week 6 goes to error analysis and reliability, and the report states which item failed.

### Gate B: keep, checked on Sunday 27 December 2026 (end of Week 6)

Keep the agent only if every item passes on dev.

| # | Check | Pass |
|---|---|---|
| 1 | Benefit on its target | On the 20 multi-event dev queries, the agent answers at least 4 more correctly than the best fixed pipeline, net of any it breaks |
| 2 | No harm elsewhere | On a fixed random sample of 30 of the other answerable dev queries, the agent answers at most 2 fewer correctly |
| 3 | Still honest | On the 40 unanswerable dev queries, the agent answers at most 2 more than the fixed pipeline does. An agent that keeps searching until it finds something would break the project's main claim. |
| 4 | Cost | At most 8 tool calls and 60 s per query, enforced in code with fallback to the fixed pipeline at the cap; median at most 30 s; every step logged |

With 20 queries this gate is a go or no-go heuristic, not statistical evidence. A net gain of 4 of 20 could occur by chance.

- **If kept.** The agent enters the frozen configuration as system B8 and its test results are reported with intervals, whatever they show.
- **If dropped.** The report gives the dev numbers and the failed check, as the proposal promised.

Design reference: TFVTG (Zheng et al., arXiv 2408.16219) splits a query into sub-events with a language model, localises each one, and then checks their order. It is a useful pattern for `inspect_clip`.

---

## 10. Main risks

| Risk | Sign | Response |
|---|---|---|
| Captioning too slow | Bake-off over 1.0 s per window | Fallback ladder in section 7 |
| A model fails to load under Transformers 5.23 | Y1 fails | Pin an older Transformers version that loads all models |
| Whisper GPU libraries will not install on Windows | Over half a day spent on DLL setup | Use the Transformers Whisper pipeline (3.1) |
| Labelling runs late | S1 not passed by end of Week 2 | Cut to 24 videos; keep the unanswerable share |
| Rejection accuracy is low | Dev rejection under 50% at the 20% rule | Report it plainly. Add `f_rerank` (A3) if the reranker is kept. Do not relax the threshold rule or drop near-miss negatives. |
| Many Charades negatives turn out answerable | Over 30% dropped in S2 | Still feasible: replace from the same file and report the drop rate |
| Test labels seen early | More than one entry in `test_access.log` | Disclose in the report; treat the affected numbers as dev-quality |

---

## 11. Choices still open, with recommendations

| Choice | Options | Recommendation | Decided by |
|---|---|---|---|
| Captioning model | Qwen3-VL-4B-Instruct, or Qwen3.5-4B | Qwen3-VL-4B-Instruct unless the challenger wins under the rule in 3.3 | Week 1 bake-off |
| Whisper size | turbo, or large-v3 | turbo unless its word error rate is more than 2 points worse | Week 1 check in 3.1 |
| Frame encoder size | so400m at 384 px, or base at 224 px | so400m unless it runs under 20 frames/s | Week 1 gate in 3.2 |
| Threshold rule | Cap the false-answer rate at 20%, or maximise balanced accuracy | Cap at 20%; it states the promise to the user directly. Report the other as secondary. | Fixed now; tell the supervisor if you prefer the alternative before Week 4 |
| False-answer cap | 10%, 20% or 30% | 20%. With 32 calibration negatives, 10% rests on about 3 queries. | Fixed now |
| Fusion method | Weighted sum of standardised scores, or reciprocal rank fusion | Weighted sum; it keeps absolute scores. Rank fusion is ablation X2. | Dev |
| Full Charades-STA test run | Yes or no | No, unless Week 7 finishes early; about 7.3 extra GPU-hours | Team, after the freeze |

---

## 12. Differences from the proposal, to state in the final report

1. GPU memory is 20 GB, not 24 GB.
2. "Abstention accuracy" is replaced by rejection accuracy, false-abstain rate, balanced accuracy and the risk-coverage curve, because one accuracy figure depends on the share of unanswerable queries.
3. The abstention threshold is chosen on calibration videos that are separate from both dev and test.
4. The test labels are held by Yesaullah, not by the member who tunes the threshold.
5. The evaluation harness is built in Weeks 2–3, not Week 7.
6. FFmpeg is called directly, not through `ffmpeg-python`. FAISS runs on CPU. BM25 uses `bm25s`.
7. The public benchmark is Charades-STA. It cannot test the transcript path, spoken-query routing or long videos.
8. Unanswerable queries on Charades come from a published set (Flanagan et al., 2025), checked by hand.
9. Benchmark-slice results are not directly comparable with published full-test results.

---

## 13. Sources

All read on 14 November 2026.

**Hardware**
- NVIDIA RTX 4000 Ada Generation datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-4000-datasheet.pdf
- NVIDIA RTX 4000 SFF Ada Generation datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-4000-sff-datasheet.pdf
- Laptop variant, third-party specification page: https://www.notebookcheck.net/NVIDIA-RTX-4000-Ada-Generation-Laptop-GPU-Benchmarks-and-Specs.744889.0.html
- Third-party throughput listing, not verified at source: https://specpicks.com/benchmarks/nvidia-rtx-4000-ada-generation

**Speech**
- faster-whisper README, benchmarks and GPU requirements: https://github.com/SYSTRAN/faster-whisper
- Model alias table: https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/utils.py
- Community turbo benchmark (RTX 2080 Ti): https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2/discussions/3
- Whisper large-v3-turbo model card: https://huggingface.co/openai/whisper-large-v3-turbo

**Vision and text models** (licence, parameter count and revision from each page and the Hugging Face model API)
- https://huggingface.co/google/siglip2-so400m-patch16-384 and https://huggingface.co/google/siglip2-base-patch16-224
- Transformers SigLIP 2 guide: https://huggingface.co/docs/transformers/model_doc/siglip2
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct, https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct, https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8 (states that Transformers cannot load FP8 weights)
- https://huggingface.co/Qwen/Qwen3.5-4B and https://huggingface.co/Qwen/Qwen3.5-9B
- https://huggingface.co/google/gemma-4-E4B-it
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B, https://huggingface.co/BAAI/bge-m3, https://huggingface.co/google/embeddinggemma-300m
- https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B, https://huggingface.co/BAAI/bge-reranker-v2-m3
- https://huggingface.co/facebook/PE-Core-L14-336

**Libraries**
- PyPI JSON API, `https://pypi.org/pypi/<package>/json`, for every version, release date and wheel platform in 3.8

**Datasets**
- QVHighlights annotations, data description and licence: https://github.com/jayleicn/moment_detr (statistics computed from `data/highlight_{train,val,test}_release.jsonl`)
- QVHighlights video archive, size from the HTTP header: https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz
- Charades 480p archive, size from the HTTP header: https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip
- Charades licence text, as mirrored at https://huggingface.co/datasets/HuggingFaceM4/charades ; dataset page https://prior.allenai.org/projects/charades
- Charades-STA test annotations (statistics computed from `charades_sta_test.txt`): https://huggingface.co/datasets/jwnt4/charades-sta-test
- Negative queries, files and counts: https://github.com/keflanagan/MomentofUntruth

**Papers**
- Flanagan, Damen, Wray. Moment of Untruth: Dealing with Negative Queries in Video Moment Retrieval. WACV 2025. https://arxiv.org/abs/2502.08544
- Xu, Sun, Xie, Zhai, Du. VTG-GPT: Tuning-Free Zero-Shot Video Temporal Grounding with GPT. https://arxiv.org/abs/2403.02076
- Zheng, Cai, Chen et al. Training-free Video Temporal Grounding using Large-scale Pre-trained Models. https://arxiv.org/abs/2408.16219
- Lei et al. QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries. NeurIPS 2021. https://arxiv.org/abs/2107.09609
- Tschannen et al. SigLIP 2. https://arxiv.org/abs/2502.14786
- Qwen3-VL Technical Report. https://arxiv.org/abs/2511.21631
