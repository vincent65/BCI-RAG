# Confusion-Set Guided Retrieval for LLM-Constrained Brain-to-Text Decoding

This repository is a course-project workspace built on top of public Brain-to-Text Benchmark code and artifacts. The main purpose of this repo is **not** to introduce a new upstream neural decoder. Instead, it focuses on: confusion-aware reranking, retrieval, and analysis on top of an inherited speech-BCI decoding pipeline.

## What is inherited vs. what is new here?

### Inherited / adapted components
The base decoder, training configs, and much of the decoding infrastructure were cloned or adapted from prior work:

- Stanford Brain-to-Text / `speechBCI` code and language-model decoding pipeline:
  <https://github.com/fwillett/speechBCI>
- PyTorch `neural_seq_decoder` implementation:
  <https://github.com/cffan/neural_seq_decoder>
- Brain-to-Text Benchmark '24:
  <https://eval.ai/web/challenges/challenge-page/2099/overview>
- Underlying dataset paper:
  [Willett et al. (Nature, 2023)](https://www.nature.com/articles/s41586-023-06377-x)

The inherited parts include:

- `conf/config_1.yaml` and `conf/config_2.yaml`
- the GRU/CTC decoder and training loop
- the 5-gram WFST decoding pipeline
- the original evaluation setup and much of the benchmark-oriented infrastructure

### Main contribution in this repo
The main work in this repository is:

- extracting confusion spans from N-best outputs
- converting competing spans into phoneme space
- gating retrieval using score margins and phoneme distance
- building a confusion-aware BM25 retrieval corpus
- retrieval-augmented LLM reranking
- candidate expansion and multi-prompt ensembling ablations

## Repository purpose
This repo is organized around a simple question:

> Given a fixed inherited decoder, can we improve the final sentence output by reranking ambiguous N-best candidates more intelligently?

The Section 2 experiments treat the upstream neural decoder as fixed infrastructure and study the language-side decision layer instead.

## Requirements
We used [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for training and evaluation experiments.  
`environment.yml` can be used to recreate the environment.

## Repository layout

### Core code
- `neural_decoder/`
  - decoder utilities, checkpoint helpers, language-model rescoring, and Section 2 logic
- `neural_decoder_trainer.py`
  - inherited / adapted training entrypoint for the GRU decoder
- `eval_competition.py`
  - main evaluation script, including Section 2 modes
- `eval_competition_client.py`
  - decode-server client evaluation path
- `modal_inference.py`
  - remote GPU offload helpers

### Section 2 code
- `neural_decoder/section2_utils.py`
  - confusion extraction, phoneme conversion, retrieval utilities
- `neural_decoder/section2_reranking.py`
  - reranking modes and prompt variants
- `scripts/build_confusion_corpus.py`
  - builds the confusion-aware retrieval corpus
- `scripts/evaluate_section2_runs.py`
  - aggregates saved Section 2 runs into a report

### Data and external assets
- `competitionData/`
  - benchmark `.mat` files if present locally
- `data/ptDecoder_ctc`
  - prepared dataset pickle used by the decoder
- `external/lang_test/`
  - 5-gram WFST assets such as `TLG.fst` and `words.txt`


## Expected sibling layout
The code still expects the Stanford repository to be cloned as a sibling directory if you want to build and use the original decoder runtime:

```text
<workspace>/
  speechBCI_2024/
  speechBCI/
```

Only the decoder runtime from `speechBCI` is required for the WFST path used by `eval_competition.py`.

## Quick setup
1. Create the conda environment:
   `conda env create -f environment.yml`
2. Activate it:
   `conda activate speech-BCI`
3. Build and install the Stanford decoder runtime:
   `cd ../speechBCI/LanguageModelDecoder/runtime/server/x86`
   `python setup.py install`
4. Put the prepared dataset pickle at `data/ptDecoder_ctc`, or override with `--dataPath`
5. Put the 5-gram decoding graph at `external/lang_test`
6. Supply the model checkpoint with `--modelPath`

## External assets
To reproduce the full pipeline you still need artifacts not bundled directly in this repo:

1. the prepared train/test/competition dataset pickle
2. the 5-gram WFST assets (`TLG.fst`, `words.txt`, and optionally `G.fst` / `G_no_prune.fst`)
3. the pretrained checkpoint(s) for the inherited upstream decoder

The Stanford repository README links to the Dryad release that contains the language-model artifacts and formatted competition data.

## Running evaluation

### Basic evaluation
```bash
python eval_competition.py \
  --modelPath /path/to/modelWeights \
  --dataPath ./data/ptDecoder_ctc \
  --lmDir ./external/lang_test \
  --MODEL_CACHE_DIR ./models/opt-6.7b \
  --outputDir ./eval_output/run_name
```

Pass `--disable8bit` if your environment does not support 8-bit model loading.

### Section 2 modes
The main Section 2 ablations are controlled by `--section2Mode`:

- `ngram_top1`
- `llm_rescore_no_rag`
- `confusion_rag_phoneme`
- `confusion_rag_retrieval`
- `confusion_rag_expand`
- `confusion_rag_full`
- `multi_prompt_ensemble`

Example:

```bash
python eval_competition.py \
  --modelPath /path/to/modelWeights \
  --dataPath ./data/ptDecoder_ctc \
  --lmDir ./external/lang_test \
  --retrievalCorpusPath ./derived/confusion_corpus.jsonl \
  --section2Mode confusion_rag_retrieval \
  --outputDir ./eval_output/section2_confusion_rag_retrieval
```

## Modal GPU offload
`eval_competition.py` can automatically offload GPU-heavy steps to Modal when local CUDA is unavailable:

- GRU forward inference
- local OPT rescoring / prompt scoring

The following still stay local:

- dataset loading
- 5-gram WFST decoding
- retrieval and confusion analysis
- metrics and output writing

One-time setup:

```bash
modal token set --token-id <your-token-id> --token-secret <your-token-secret>
modal deploy modal_inference.py
```

Optional flags:

- `--modalAppName`
- `--modalGpu`

## Training
If you need to retrain the inherited upstream decoder, use:

```bash
python neural_decoder_trainer.py
```

The main inherited configs are:

- `conf/config_1.yaml`
- `conf/config_2.yaml`

These configs are preserved mainly so the Section 2 experiments can be run on top of the same upstream model family.

