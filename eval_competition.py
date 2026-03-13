import argparse
import json
import os
import pickle
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from neural_decoder.dataset import SpeechDataset
from neural_decoder.lm_utils import (
    build_lm_decoder,
    build_modal_rescorer,
    build_openai_rescorer,
    build_opt,
    compute_cer_wer,
    is_modal_backend,
    is_openai_backend,
    lm_decode,
    rearrange_speech_logits,
    rank_nbest_by_decoder,
    rank_nbest_with_gpt2,
)
from neural_decoder.section2_reranking import apply_section2_mode
from neural_decoder.section2_utils import (
    PhonemeConverter,
    build_analysis_record,
    build_retriever,
)
from neural_decoder_trainer import loadModel


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "data" / "ptDecoder_ctc"
DEFAULT_LM_DIR = SCRIPT_DIR / "external" / "lang_test"
DEFAULT_MODEL_CACHE_DIR = SCRIPT_DIR / "models" / "opt-6.7b"
DEFAULT_MODAL_APP_NAME = "speechbci-gpu-offload"
DEFAULT_MODAL_GPU = "A10G"
OPT67B_MODAL_GPU = "A100-40GB"
COMPETITION_DAY_INDICES = [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20]

parser = argparse.ArgumentParser(description="Run 5-gram + OPT rescoring on a dataset split.")
parser.add_argument("--modelPath", type=str, required=True, help="Path to model weights")
parser.add_argument(
    "--dataPath",
    type=str,
    default=str(DEFAULT_DATA_PATH),
    help="Path to the parsed dataset pickle.",
)
parser.add_argument(
    "--MODEL_CACHE_DIR",
    dest="model_cache_dir",
    type=str,
    default=str(DEFAULT_MODEL_CACHE_DIR),
    help="Directory for the cached OPT model weights.",
)
parser.add_argument(
    "--modalAppName",
    type=str,
    default=DEFAULT_MODAL_APP_NAME,
    help="Modal app name used for automatic GPU offload on CPU-only hosts.",
)
parser.add_argument(
    "--modalGpu",
    type=str,
    default=DEFAULT_MODAL_GPU,
    help="Modal GPU type used for automatic GPU offload.",
)
parser.add_argument(
    "--lmDir",
    type=str,
    default=str(DEFAULT_LM_DIR),
    help="Directory containing TLG.fst, words.txt, and optional G.fst files.",
)
parser.add_argument(
    "--llmModelName",
    type=str,
    default="facebook/opt-6.7b",
    help="LLM model name used for rescoring, either local HF or OpenAI API.",
)
parser.add_argument(
    "--llmBackend",
    type=str,
    choices=["local", "openai"],
    default="local",
    help="Whether to use a local Hugging Face model or the OpenAI API for rescoring.",
)
parser.add_argument(
    "--outputDir",
    type=str,
    default=str(SCRIPT_DIR / "eval_output"),
    help="Directory to save evaluation results.",
)
parser.add_argument(
    "--disable8bit",
    action="store_true",
    help="Disable 8-bit loading for the OPT model.",
)
parser.add_argument(
    "--partition",
    type=str,
    choices=["train", "test", "competition"],
    default="competition",
    help="Which dataset split to decode and evaluate.",
)
parser.add_argument(
    "--inferenceBatchSize",
    type=int,
    default=1,
    help="Batch size for the neural decoder forward pass.",
)
parser.add_argument(
    "--dataloaderWorkers",
    type=int,
    default=0,
    help="Number of DataLoader worker processes for split iteration.",
)
parser.add_argument(
    "--torchThreads",
    type=int,
    default=0,
    help="Set PyTorch intra-op CPU threads. Use 0 to keep the current default.",
)
parser.add_argument(
    "--torchInteropThreads",
    type=int,
    default=0,
    help="Set PyTorch inter-op CPU threads. Use 0 to keep the current default.",
)
parser.add_argument(
    "--section2Mode",
    type=str,
    choices=[
        "ngram_top1",
        "llm_rescore_no_rag",
        "confusion_rag_phoneme",
        "confusion_rag_retrieval",
        "confusion_rag_expand",
        "confusion_rag_full",
        "multi_prompt_ensemble",
    ],
    default="llm_rescore_no_rag",
    help="Which decoding / reranking variant to run.",
)
parser.add_argument(
    "--analysisTopK",
    type=int,
    default=10,
    help="How many ranked candidates to save per utterance in the analysis artifact.",
)
parser.add_argument(
    "--confusionTopK",
    type=int,
    default=5,
    help="How many top candidates to compare when extracting confusion spans.",
)
parser.add_argument(
    "--retrievalCorpusPath",
    type=str,
    default="",
    help="Optional JSONL confusion-aware corpus for BM25 retrieval.",
)
parser.add_argument(
    "--retrievalTopK",
    type=int,
    default=5,
    help="How many retrieved snippets to attach for retrieval-enabled variants.",
)
parser.add_argument(
    "--phonemeDistanceThreshold",
    type=int,
    default=3,
    help="Maximum phoneme edit distance for retrieval gating.",
)
parser.add_argument(
    "--scoreMarginThreshold",
    type=float,
    default=5.0,
    help="Maximum score gap between the top two candidates for retrieval gating.",
)
parser.add_argument(
    "--maxExpansions",
    type=int,
    default=3,
    help="Maximum number of retrieval-driven expanded candidates to add.",
)
parser.add_argument(
    "--openaiWorkers",
    type=int,
    default=8,
    help="Number of concurrent remote reranking requests to issue for OpenAI or Modal backends.",
)
parser.add_argument(
    "--progressEvery",
    type=int,
    default=25,
    help="Print progress and refresh checkpoint files every N completed utterances.",
)
input_args = parser.parse_args()


def validate_path(path_str, description, required_files=None):
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")

    required_files = required_files or []
    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"{description} is missing required files: {missing_str}. Checked {path}"
        )
    return path


def normalize_transcription(transcript):
    transcript = transcript.strip()
    transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
    transcript = transcript.replace("--", "").lower()
    return transcript


def collate_inference_batch(batch):
    X, y, X_lens, y_lens, days = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
    )


def format_duration(seconds):
    if seconds == float("inf"):
        return "unknown"
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def compute_total_utterances(loaded_data, partition_name):
    return sum(len(session["transcriptions"]) for session in loaded_data[partition_name])


def write_progress_file(
    progress_path,
    partition_name,
    mode_name,
    submitted_count,
    completed_count,
    total_count,
    rerank_start_time,
    fivegram_total_s,
):
    elapsed = max(time.time() - rerank_start_time, 0.0)
    rate = (completed_count / elapsed) if elapsed > 0 else 0.0
    remaining = max(total_count - completed_count, 0)
    eta_s = (remaining / rate) if rate > 0 else float("inf")
    with open(progress_path, "w") as handle:
        handle.write(f"partition: {partition_name}\n")
        handle.write(f"mode: {mode_name}\n")
        handle.write(f"submitted_utterances: {submitted_count}\n")
        handle.write(f"completed_utterances: {completed_count}\n")
        handle.write(f"total_utterances: {total_count}\n")
        handle.write(f"fivegram_seconds_total: {fivegram_total_s}\n")
        handle.write(f"fivegram_seconds_per_sample: {fivegram_total_s / max(submitted_count, 1)}\n")
        handle.write(f"rerank_elapsed_seconds: {elapsed}\n")
        handle.write(f"rerank_wall_seconds_per_completed: {elapsed / max(completed_count, 1)}\n")
        handle.write(f"estimated_time_remaining: {format_duration(eta_s)}\n")


def rerank_single_utterance(
    utterance_index,
    partition_name,
    reference_text,
    raw_nbest,
    llm,
    llm_tokenizer,
    section2_mode,
    analysis_top_k,
    confusion_top_k,
    acoustic_scale,
    phoneme_converter,
    retriever,
    retrieval_top_k,
    phoneme_distance_threshold,
    score_margin_threshold,
    max_expansions,
    length_penalty=0.0,
    llm_weight=0.5,
):
    try:
        if llm is not None and not is_openai_backend(llm):
            ranked_candidates = rank_nbest_with_gpt2(
                llm,
                llm_tokenizer,
                raw_nbest,
                acoustic_scale,
                length_penalty=length_penalty,
                alpha=llm_weight,
            )
        else:
            ranked_candidates = rank_nbest_by_decoder(raw_nbest, acoustic_scale=acoustic_scale)

        ranked_candidates = ranked_candidates[:analysis_top_k]
        include_confusion = section2_mode != "llm_rescore_no_rag"
        analysis_record = build_analysis_record(
            utterance_index=utterance_index,
            partition=partition_name,
            reference=reference_text,
            ranked_candidates=ranked_candidates,
            raw_nbest=raw_nbest,
            phoneme_converter=phoneme_converter,
            confusion_top_k=confusion_top_k,
            include_confusion=include_confusion,
        )
        mode_result = apply_section2_mode(
            section2_mode,
            ranked_candidates=ranked_candidates,
            raw_nbest=raw_nbest,
            confusion_spans=analysis_record["confusion_spans"],
            score_margin=analysis_record["score_margin"],
            model=llm,
            tokenizer=llm_tokenizer,
            retriever=retriever,
            retrieval_top_k=retrieval_top_k,
            phoneme_distance_threshold=phoneme_distance_threshold,
            score_margin_threshold=score_margin_threshold,
            length_penalty=length_penalty,
            max_expansions=max_expansions,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to rerank utterance {utterance_index} for partition '{partition_name}'"
        ) from exc

    selected_text = mode_result["selected_text"]
    confidence = 0.0
    if ranked_candidates and selected_text == ranked_candidates[0]["text"]:
        confidence = float(ranked_candidates[0]["confidence"])

    analysis_record["selected_text"] = selected_text
    analysis_record["selected_mode"] = section2_mode
    analysis_record["triggered"] = mode_result.get("triggered", False)
    analysis_record["retrieved_docs"] = mode_result.get("retrieved_docs", [])
    analysis_record["expanded_candidates"] = mode_result.get("expanded_candidates", [])
    analysis_record["variant_outputs"] = mode_result.get("variant_outputs", {})
    if "votes" in mode_result:
        analysis_record["votes"] = mode_result["votes"]

    return {
        "utterance_index": utterance_index,
        "reference": reference_text,
        "selected_text": selected_text,
        "confidence": confidence,
        "analysis_record": analysis_record,
    }


def flush_ready_results(
    completed_results,
    next_write_index,
    decoded_transcriptions,
    reference_transcriptions,
    confidences,
    analysis_records,
    decoded_handle,
    reference_handle,
    analysis_handle,
):
    flushed_count = 0
    while next_write_index in completed_results:
        result = completed_results.pop(next_write_index)
        decoded_transcriptions.append(result["selected_text"])
        reference_transcriptions.append(result["reference"])
        confidences.append(result["confidence"])
        analysis_records.append(result["analysis_record"])
        decoded_handle.write(result["selected_text"] + "\n")
        reference_handle.write(result["reference"] + "\n")
        analysis_handle.write(json.dumps(result["analysis_record"]) + "\n")
        next_write_index += 1
        flushed_count += 1

    if flushed_count > 0:
        decoded_handle.flush()
        reference_handle.flush()
        analysis_handle.flush()

    return next_write_index, flushed_count


def resolve_modal_llm_gpu(args, local_cuda_available):
    if local_cuda_available or args.llmBackend == "openai":
        return args.modalGpu
    if (
        args.modalGpu == DEFAULT_MODAL_GPU
        and args.disable8bit
        and args.llmModelName == "facebook/opt-6.7b"
    ):
        return OPT67B_MODAL_GPU
    return args.modalGpu


if input_args.torchThreads > 0:
    torch.set_num_threads(input_args.torchThreads)
if input_args.torchInteropThreads > 0:
    torch.set_num_interop_threads(input_args.torchInteropThreads)

print(f"torch threads: {torch.get_num_threads()}")
print(f"torch interop threads: {torch.get_num_interop_threads()}")


model_path = validate_path(input_args.modelPath, "Model checkpoint")
data_path = validate_path(input_args.dataPath, "Dataset pickle")
lm_dir = validate_path(input_args.lmDir, "Language-model directory", ["TLG.fst", "words.txt"])
output_dir = Path(input_args.outputDir).expanduser().resolve()
output_dir.mkdir(parents=True, exist_ok=True)

with open(data_path, "rb") as handle:
    loadedData = pickle.load(handle)

if input_args.partition not in loadedData:
    raise KeyError(
        f"The dataset pickle does not contain the '{input_args.partition}' split. "
        "Run notebooks/formatCompetitionData.ipynb or provide the correctly prepared pickle."
    )

local_cuda_available = torch.cuda.is_available()
device = "cuda" if local_cuda_available else "cpu"
use_modal_gru = not local_cuda_available
modal_llm_gpu = resolve_modal_llm_gpu(input_args, local_cuda_available)
if use_modal_gru:
    from modal_inference import ModalGRUProxy

    print(
        "Local CUDA is unavailable; offloading GRU inference to Modal.",
        flush=True,
    )
    model = ModalGRUProxy(
        str(model_path),
        app_name=input_args.modalAppName,
        gpu=input_args.modalGpu,
    )
else:
    model = loadModel(str(model_path), device=device)

model.eval()
partition = input_args.partition

if partition == "competition":
    if len(loadedData[partition]) != len(COMPETITION_DAY_INDICES):
        raise ValueError(
            "Competition split length does not match the expected day-index mapping. "
            f"Expected {len(COMPETITION_DAY_INDICES)} sessions, got {len(loadedData[partition])}."
        )
    partition_day_indices = COMPETITION_DAY_INDICES
else:
    partition_day_indices = list(range(len(loadedData[partition])))
ngramDecoder = build_lm_decoder(
    str(lm_dir), acoustic_scale=0.5, nbest=100, beam=18
)

# LM decoding hyperparameters
acoustic_scale = 0.5
blank_penalty = np.log(7)
llm_weight = 0.5

need_llm = input_args.section2Mode != "ngram_top1"
llm = None
llm_tokenizer = None
resolved_llm_backend = input_args.llmBackend
if need_llm:
    if input_args.llmBackend == "openai":
        openai_model_name = (
            "gpt-5-2"
            if input_args.llmModelName == "facebook/opt-6.7b"
            else input_args.llmModelName
        )
        llm = build_openai_rescorer(model_name=openai_model_name)
    elif not local_cuda_available:
        if modal_llm_gpu != input_args.modalGpu:
            print(
                f"Local CUDA is unavailable; upgrading Modal OPT GPU from "
                f"{input_args.modalGpu} to {modal_llm_gpu} for {input_args.llmModelName} "
                f"with --disable8bit.",
                flush=True,
            )
        print(
            "Local CUDA is unavailable; offloading OPT scoring to Modal.",
            flush=True,
        )
        llm = build_modal_rescorer(
            model_name=input_args.llmModelName,
            app_name=input_args.modalAppName,
            gpu=modal_llm_gpu,
            load_in_8bit=not input_args.disable8bit,
        )
        resolved_llm_backend = "modal"
    else:
        llm, llm_tokenizer = build_opt(
            model_name=input_args.llmModelName,
            cache_dir=input_args.model_cache_dir,
            device="auto",
            load_in_8bit=not input_args.disable8bit,
        )

retriever = None
if input_args.retrievalCorpusPath:
    retrieval_corpus_path = validate_path(
        input_args.retrievalCorpusPath, "Retrieval corpus JSONL"
    )
    retriever = build_retriever(retrieval_corpus_path)

total_utterances = compute_total_utterances(loadedData, partition)
decoded_path = output_dir / f"{partition}_decoded.txt"
reference_path = output_dir / f"{partition}_reference.txt"
summary_path = output_dir / f"{partition}_metrics.txt"
analysis_path = output_dir / f"{partition}_analysis.jsonl"
progress_path = output_dir / f"{partition}_progress.txt"

decodedTranscriptions = []
reference_transcriptions = []
confidences = []
analysis_records = []
completed_results = {}
submitted_count = 0
completed_count = 0
next_write_index = 0
last_progress_report = 0
fivegram_total_s = 0.0
rerank_start_t = time.time()
use_parallel_rerank = (
    need_llm
    and input_args.openaiWorkers > 1
    and (is_openai_backend(llm) or is_modal_backend(llm))
)
max_pending_futures = max(input_args.openaiWorkers * 4, input_args.openaiWorkers)

phoneme_converter = None
if input_args.section2Mode != "llm_rescore_no_rag":
    phoneme_converter = PhonemeConverter()

print(
    f"Processing {total_utterances} utterances with mode={input_args.section2Mode} "
    f"backend={resolved_llm_backend}",
    flush=True,
)
if use_parallel_rerank:
    print(
        f"Using {input_args.openaiWorkers} concurrent rerank workers for {resolved_llm_backend}.",
        flush=True,
    )

with open(decoded_path, "w") as decoded_handle, open(reference_path, "w") as reference_handle, open(
    analysis_path, "w"
) as analysis_handle:
    write_progress_file(
        progress_path,
        partition,
        input_args.section2Mode,
        submitted_count,
        completed_count,
        total_utterances,
        rerank_start_t,
        fivegram_total_s,
    )

    executor = None
    pending_futures = {}
    if use_parallel_rerank:
        executor = ThreadPoolExecutor(max_workers=input_args.openaiWorkers)

    try:
        for i, testDayIdx in enumerate(partition_day_indices):
            test_ds = SpeechDataset([loadedData[partition][i]])
            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=input_args.inferenceBatchSize,
                shuffle=False,
                num_workers=input_args.dataloaderWorkers,
                pin_memory=(device == "cuda"),
                collate_fn=collate_inference_batch,
            )
            for j, (X, y, X_len, y_len, _) in enumerate(test_loader):
                batch_size = X.shape[0]
                X, X_len, dayIdx = (
                    X.to(device),
                    X_len.to(device),
                    torch.full((batch_size,), testDayIdx, dtype=torch.int64).to(device),
                )
                pred = model.forward(X, dayIdx)
                adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

                for iterIdx in range(pred.shape[0]):
                    transcript_idx = j * input_args.inferenceBatchSize + iterIdx
                    transcript = normalize_transcription(
                        loadedData[partition][i]["transcriptions"][transcript_idx]
                    )

                    logits = pred[iterIdx].cpu().detach().numpy()
                    logit_len = adjustedLens[iterIdx].cpu().detach().item()
                    logits = logits[:logit_len]
                    logits = np.concatenate(
                        [logits[:, 1:], logits[:, 0:1]], axis=-1
                    )  # Blank is last token
                    logits = rearrange_speech_logits(logits[None, :, :], has_sil=True)

                    decode_start_t = time.time()
                    raw_nbest = lm_decode(
                        ngramDecoder,
                        logits[0],
                        blank_penalty=blank_penalty,
                        return_nbest=True,
                        rescore=True,
                    )
                    fivegram_total_s += time.time() - decode_start_t

                    job_args = (
                        submitted_count,
                        partition,
                        transcript,
                        raw_nbest,
                        llm,
                        llm_tokenizer,
                        input_args.section2Mode,
                        input_args.analysisTopK,
                        input_args.confusionTopK,
                        acoustic_scale,
                        phoneme_converter,
                        retriever,
                        input_args.retrievalTopK,
                        input_args.phonemeDistanceThreshold,
                        input_args.scoreMarginThreshold,
                        input_args.maxExpansions,
                    )
                    job_kwargs = {
                        "length_penalty": 0.0,
                        "llm_weight": llm_weight,
                    }

                    if executor is not None:
                        future = executor.submit(rerank_single_utterance, *job_args, **job_kwargs)
                        pending_futures[future] = submitted_count
                        while len(pending_futures) >= max_pending_futures:
                            done, _ = wait(
                                set(pending_futures.keys()),
                                return_when=FIRST_COMPLETED,
                            )
                            for future in done:
                                result = future.result()
                                completed_results[result["utterance_index"]] = result
                                pending_futures.pop(future, None)
                            next_write_index, flushed_count = flush_ready_results(
                                completed_results,
                                next_write_index,
                                decodedTranscriptions,
                                reference_transcriptions,
                                confidences,
                                analysis_records,
                                decoded_handle,
                                reference_handle,
                                analysis_handle,
                            )
                            completed_count += flushed_count
                            if (
                                completed_count == total_utterances
                                or completed_count - last_progress_report >= input_args.progressEvery
                            ):
                                elapsed = max(time.time() - rerank_start_t, 1e-6)
                                rate = completed_count / elapsed
                                eta_s = (
                                    (total_utterances - completed_count) / rate if rate > 0 else float("inf")
                                )
                                print(
                                    f"[progress] completed {completed_count}/{total_utterances} utterances "
                                    f"({completed_count / total_utterances:.1%}) | "
                                    f"5gram {fivegram_total_s / max(submitted_count + 1, 1):.3f}s/sample | "
                                    f"rerank wall {elapsed / max(completed_count, 1):.2f}s/utt | "
                                    f"eta {format_duration(eta_s)}",
                                    flush=True,
                                )
                                write_progress_file(
                                    progress_path,
                                    partition,
                                    input_args.section2Mode,
                                    submitted_count + 1,
                                    completed_count,
                                    total_utterances,
                                    rerank_start_t,
                                    fivegram_total_s,
                                )
                                last_progress_report = completed_count
                    else:
                        result = rerank_single_utterance(*job_args, **job_kwargs)
                        completed_results[result["utterance_index"]] = result
                        next_write_index, flushed_count = flush_ready_results(
                            completed_results,
                            next_write_index,
                            decodedTranscriptions,
                            reference_transcriptions,
                            confidences,
                            analysis_records,
                            decoded_handle,
                            reference_handle,
                            analysis_handle,
                        )
                        completed_count += flushed_count
                        if (
                            completed_count == total_utterances
                            or completed_count - last_progress_report >= input_args.progressEvery
                        ):
                            elapsed = max(time.time() - rerank_start_t, 1e-6)
                            rate = completed_count / elapsed
                            eta_s = (
                                (total_utterances - completed_count) / rate if rate > 0 else float("inf")
                            )
                            print(
                                f"[progress] completed {completed_count}/{total_utterances} utterances "
                                f"({completed_count / total_utterances:.1%}) | "
                                f"5gram {fivegram_total_s / max(submitted_count + 1, 1):.3f}s/sample | "
                                f"rerank wall {elapsed / max(completed_count, 1):.2f}s/utt | "
                                f"eta {format_duration(eta_s)}",
                                flush=True,
                            )
                            write_progress_file(
                                progress_path,
                                partition,
                                input_args.section2Mode,
                                submitted_count + 1,
                                completed_count,
                                total_utterances,
                                rerank_start_t,
                                fivegram_total_s,
                            )
                            last_progress_report = completed_count

                    submitted_count += 1

        while pending_futures:
            done, _ = wait(set(pending_futures.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                completed_results[result["utterance_index"]] = result
                pending_futures.pop(future, None)
            next_write_index, flushed_count = flush_ready_results(
                completed_results,
                next_write_index,
                decodedTranscriptions,
                reference_transcriptions,
                confidences,
                analysis_records,
                decoded_handle,
                reference_handle,
                analysis_handle,
            )
            completed_count += flushed_count
            if (
                completed_count == total_utterances
                or completed_count - last_progress_report >= input_args.progressEvery
            ):
                elapsed = max(time.time() - rerank_start_t, 1e-6)
                rate = completed_count / elapsed
                eta_s = (
                    (total_utterances - completed_count) / rate if rate > 0 else float("inf")
                )
                print(
                    f"[progress] completed {completed_count}/{total_utterances} utterances "
                    f"({completed_count / total_utterances:.1%}) | "
                    f"5gram {fivegram_total_s / max(submitted_count, 1):.3f}s/sample | "
                    f"rerank wall {elapsed / max(completed_count, 1):.2f}s/utt | "
                    f"eta {format_duration(eta_s)}",
                    flush=True,
                )
                write_progress_file(
                    progress_path,
                    partition,
                    input_args.section2Mode,
                    submitted_count,
                    completed_count,
                    total_utterances,
                    rerank_start_t,
                    fivegram_total_s,
                )
                last_progress_report = completed_count
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

fivegram_time_per_sample = fivegram_total_s / max(total_utterances, 1)
llm_time_per_sample = (time.time() - rerank_start_t) / max(total_utterances, 1)
print(f"5gram decoding took {fivegram_time_per_sample} seconds per sample")
print(f"Section 2 reranking took {llm_time_per_sample} seconds per sample")

cer, wer = compute_cer_wer(
    decodedTranscriptions,
    reference_transcriptions,
    output_type="speech_sil",
    return_ci=True,
)

llm_out = {
    "mode": input_args.section2Mode,
    "cer": cer,
    "wer": wer,
    "decoded_transcripts": decodedTranscriptions,
    "confidences": confidences,
    "analysis_records": analysis_records,
}

print(f"Partition: {partition}")
print(f"CER: {llm_out['cer']}")
print(f"WER: {llm_out['wer']}")

with open(output_dir / "llm_out", "wb") as handle:
    pickle.dump(llm_out, handle)

with open(summary_path, "w") as handle:
    handle.write(f"partition: {partition}\n")
    handle.write(f"mode: {input_args.section2Mode}\n")
    handle.write(f"cer: {llm_out['cer']}\n")
    handle.write(f"wer: {llm_out['wer']}\n")
    handle.write(f"fivegram_seconds_per_sample: {fivegram_time_per_sample}\n")
    handle.write(f"llm_seconds_per_sample: {llm_time_per_sample}\n")
    handle.write(f"analysis_path: {analysis_path}\n")
    handle.write(f"progress_path: {progress_path}\n")

write_progress_file(
    progress_path,
    partition,
    input_args.section2Mode,
    submitted_count,
    completed_count,
    total_utterances,
    rerank_start_t,
    fivegram_total_s,
)

if partition == "competition":
    with open(output_dir / "5gramLLMCompetitionSubmission.txt", "w") as handle:
        for transcription in decodedTranscriptions:
            handle.write(transcription + "\n")
