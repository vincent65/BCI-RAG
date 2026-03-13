import json
import os
import threading
import time as time_module
from pathlib import Path
from time import time

import numpy as np
import requests
from tqdm.auto import trange


def wer(reference, hypothesis):
    """Compute edit distance for token lists."""
    distances = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=np.uint16)
    distances[:, 0] = np.arange(len(reference) + 1)
    distances[0, :] = np.arange(len(hypothesis) + 1)

    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                distances[i, j] = distances[i - 1, j - 1]
            else:
                substitution = distances[i - 1, j - 1] + 1
                insertion = distances[i, j - 1] + 1
                deletion = distances[i - 1, j] + 1
                distances[i, j] = min(substitution, insertion, deletion)

    return int(distances[len(reference), len(hypothesis)])


def build_opt(
    model_name="facebook/opt-6.7b",
    cache_dir=None,
    device="auto",
    load_in_8bit=False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device,
        load_in_8bit=load_in_8bit,
    )
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_modal_rescorer(
    model_name="facebook/opt-6.7b",
    app_name=None,
    gpu="A10G",
    load_in_8bit=False,
):
    from modal_inference import DEFAULT_MODAL_APP_NAME, ModalOPTProxy

    return ModalOPTProxy(
        model_name=model_name,
        app_name=app_name or DEFAULT_MODAL_APP_NAME,
        gpu=gpu,
        load_in_8bit=load_in_8bit,
    )


def _read_api_key_from_dotenv(dotenv_path=None, env_var="OPENAI_API_KEY"):
    candidate_paths = []
    if dotenv_path:
        candidate_paths.append(Path(dotenv_path).expanduser().resolve())
    candidate_paths.append(Path.cwd() / ".env")
    candidate_paths.append(Path(__file__).resolve().parents[1] / ".env")

    for path in candidate_paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == env_var:
                value = value.strip().strip("'").strip('"')
                if value:
                    return value
    return None


def _normalize_openai_model_name(model_name):
    aliases = {
        "gpt-5-2": "gpt-5.2",
    }
    return aliases.get(model_name, model_name)


def build_openai_rescorer(
    model_name="gpt-5-2",
    api_key=None,
    dotenv_path=None,
    api_base="https://api.openai.com/v1/responses",
    timeout_s=60.0,
    max_retries=3,
):
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or _read_api_key_from_dotenv(
        dotenv_path=dotenv_path
    )
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found in the environment or .env file."
        )

    return {
        "provider": "openai",
        "model_name": _normalize_openai_model_name(model_name),
        "api_key": api_key,
        "api_base": api_base,
        "timeout_s": timeout_s,
        "max_retries": max_retries,
    }


def is_openai_backend(model):
    return isinstance(model, dict) and model.get("provider") == "openai"


def is_modal_backend(model):
    return getattr(model, "provider", None) == "modal"


def _get_openai_session(model):
    thread_local = model.get("_thread_local")
    if thread_local is None:
        thread_local = threading.local()
        model["_thread_local"] = thread_local

    session = getattr(thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        thread_local.session = session
    return session


def _get_model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Could not infer the model device for scoring.") from exc


def _extract_openai_text(payload):
    if isinstance(payload, dict):
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = payload.get("output", [])
        if isinstance(output, list):
            text_chunks = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                for content in item.get("content", []):
                    if not isinstance(content, dict):
                        continue
                    text_value = content.get("text")
                    if isinstance(text_value, str):
                        text_chunks.append(text_value)
            if text_chunks:
                return "".join(text_chunks).strip()
    return ""


def _openai_post_json(model, body):
    headers = {
        "Authorization": f"Bearer {model['api_key']}",
        "Content-Type": "application/json",
    }
    session = _get_openai_session(model)
    last_error = None
    for attempt in range(model["max_retries"]):
        try:
            response = session.post(
                model["api_base"],
                headers=headers,
                json=body,
                timeout=model["timeout_s"],
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < model["max_retries"]:
                time_module.sleep(min(2 ** attempt, 5))
    raise RuntimeError(f"OpenAI API request failed: {last_error}") from last_error


def choose_option_with_openai(model, prompt, options):
    if len(options) == 0:
        return -1, ""

    option_lines = [f"{idx}. {option}" for idx, option in enumerate(options, start=1)]
    response_prompt = (
        f"{prompt}\n\n"
        "Valid options:\n"
        f"{chr(10).join(option_lines)}\n\n"
        "Return only the best option number."
    )
    body = {
        "model": model["model_name"],
        "input": response_prompt,
        "max_output_tokens": 16,
        "temperature": 0,
    }
    payload = _openai_post_json(model, body)
    response_text = _extract_openai_text(payload)
    for token in response_text.replace(".", " ").split():
        if token.isdigit():
            choice = int(token) - 1
            if 0 <= choice < len(options):
                return choice, response_text
    raise ValueError(f"Could not parse OpenAI option selection from: {response_text!r}")


def rescore_with_gpt2(model, tokenizer, hypotheses, length_penalty):
    if is_openai_backend(model):
        raise ValueError(
            "OpenAI-backed rescoring does not support raw sentence log-prob scoring. "
            "Use prompt-based option selection instead."
        )
    if is_modal_backend(model):
        return model.rescore(hypotheses, length_penalty=length_penalty)
    import torch

    inputs = tokenizer(hypotheses, return_tensors="pt", padding=True)
    model_device = _get_model_device(model)
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        log_probs = torch.nn.functional.log_softmax(outputs["logits"].float(), -1).cpu().numpy()

    attention_mask = inputs["attention_mask"].cpu().numpy()
    input_ids = inputs["input_ids"].cpu().numpy()

    new_lm_scores = []
    batch_size = log_probs.shape[0]
    for i in range(batch_size):
        n_tokens = int(np.sum(attention_mask[i]))
        new_lm_score = 0.0
        for j in range(1, n_tokens):
            new_lm_score += log_probs[i, j - 1, input_ids[i, j]]
        new_lm_scores.append(new_lm_score - n_tokens * length_penalty)

    return new_lm_scores


def normalize_speech_hypothesis(hypothesis):
    hypothesis = hypothesis.strip()
    if len(hypothesis) == 0:
        return ""
    hypothesis = hypothesis.replace(">", "")
    hypothesis = hypothesis.replace("  ", " ")
    hypothesis = hypothesis.replace(" ,", ",")
    hypothesis = hypothesis.replace(" .", ".")
    hypothesis = hypothesis.replace(" ?", "?")
    return hypothesis.strip()


def _softmax_confidence(total_scores, index):
    shifted_scores = total_scores - np.max(total_scores)
    probs = np.exp(shifted_scores)
    return float(probs[index] / np.sum(probs))


def rank_nbest_with_gpt2(
    model,
    tokenizer,
    nbest,
    acoustic_scale,
    length_penalty,
    alpha,
):
    hypotheses = []
    acoustic_scores = []
    old_lm_scores = []
    original_indices = []
    for idx, out in enumerate(nbest):
        hyp = normalize_speech_hypothesis(out[0])
        if len(hyp) == 0:
            continue
        hypotheses.append(hyp)
        acoustic_scores.append(float(out[1]))
        old_lm_scores.append(float(out[2]))
        original_indices.append(idx)

    if len(hypotheses) == 0:
        return []

    acoustic_scores = np.array(acoustic_scores)
    old_lm_scores = np.array(old_lm_scores)
    new_lm_scores = np.array(
        rescore_with_gpt2(model, tokenizer, hypotheses, length_penalty)
    )
    total_scores = (
        alpha * new_lm_scores
        + (1 - alpha) * old_lm_scores
        + acoustic_scale * acoustic_scores
    )
    confidence_scores = [
        _softmax_confidence(total_scores, idx) for idx in range(len(total_scores))
    ]

    ranked = []
    for idx, text in enumerate(hypotheses):
        ranked.append(
            {
                "text": text,
                "acoustic_score": float(acoustic_scores[idx]),
                "old_lm_score": float(old_lm_scores[idx]),
                "new_lm_score": float(new_lm_scores[idx]),
                "total_score": float(total_scores[idx]),
                "confidence": float(confidence_scores[idx]),
                "source_nbest_index": int(original_indices[idx]),
            }
        )
    ranked.sort(key=lambda item: item["total_score"], reverse=True)
    return ranked


def rank_nbest_by_decoder(nbest, acoustic_scale=0.5):
    ranked = []
    for idx, out in enumerate(nbest):
        hyp = normalize_speech_hypothesis(out[0])
        if len(hyp) == 0:
            continue
        acoustic_score = float(out[1])
        old_lm_score = float(out[2])
        total_score = float(old_lm_score + acoustic_scale * acoustic_score)
        ranked.append(
            {
                "text": hyp,
                "acoustic_score": acoustic_score,
                "old_lm_score": old_lm_score,
                "new_lm_score": 0.0,
                "total_score": total_score,
                "confidence": 0.0,
                "source_nbest_index": int(idx),
            }
        )

    if not ranked:
        return []

    ranked.sort(key=lambda item: item["total_score"], reverse=True)
    score_array = np.array([item["total_score"] for item in ranked])
    for idx, item in enumerate(ranked):
        item["confidence"] = _softmax_confidence(score_array, idx)
    return ranked


def score_prompted_completions(
    model,
    tokenizer,
    prompt,
    completions,
    length_penalty=0.0,
):
    if len(completions) == 0:
        return []

    if is_openai_backend(model):
        choice_idx, _ = choose_option_with_openai(model, prompt, completions)
        return [1.0 if idx == choice_idx else 0.0 for idx in range(len(completions))]
    if is_modal_backend(model):
        return model.score_prompted(
            prompt,
            completions,
            length_penalty=length_penalty,
        )

    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_texts = [prompt + completion for completion in completions]
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True)
    model_device = _get_model_device(model)
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        log_probs = torch.nn.functional.log_softmax(outputs["logits"].float(), -1).cpu().numpy()

    attention_mask = inputs["attention_mask"].cpu().numpy()
    input_ids = inputs["input_ids"].cpu().numpy()
    prompt_len = len(prompt_ids)

    scores = []
    for i in range(log_probs.shape[0]):
        n_tokens = int(np.sum(attention_mask[i]))
        start_idx = min(prompt_len, n_tokens - 1)
        completion_score = 0.0
        for j in range(start_idx, n_tokens):
            completion_score += log_probs[i, j - 1, input_ids[i, j]]
        completion_len = max(n_tokens - start_idx, 0)
        scores.append(float(completion_score - completion_len * length_penalty))
    return scores


def gpt2_lm_decode(
    model,
    tokenizer,
    nbest,
    acoustic_scale,
    length_penalty,
    alpha,
    return_confidence=False,
):
    ranked = rank_nbest_with_gpt2(
        model,
        tokenizer,
        nbest,
        acoustic_scale,
        length_penalty,
        alpha,
    )
    if len(ranked) == 0:
        return "" if not return_confidence else ("", 0.0)

    best_hyp = ranked[0]["text"]
    if not return_confidence:
        return best_hyp

    return best_hyp, float(ranked[0]["confidence"])


def cer_with_gpt2_decoder(
    model,
    tokenizer,
    nbest_outputs,
    acoustic_scale,
    inference_out,
    output_type="handwriting",
    return_ci=False,
    length_penalty=0.0,
    alpha=1.0,
):
    decoded_sentences = []
    confidences = []
    ranked_candidates = []
    for i in trange(len(nbest_outputs)):
        ranked = rank_nbest_with_gpt2(
            model,
            tokenizer,
            nbest_outputs[i],
            acoustic_scale,
            length_penalty,
            alpha,
        )
        if len(ranked) == 0:
            decoded_sentences.append("")
            confidences.append(0.0)
            ranked_candidates.append([])
        else:
            decoded_sentences.append(ranked[0]["text"])
            confidences.append(ranked[0]["confidence"])
            ranked_candidates.append(ranked)

    if output_type == "handwriting":
        true_sentences = _extract_true_sentences(inference_out)
    elif output_type in {"speech", "speech_sil"}:
        true_sentences = _extract_transcriptions(inference_out)
    else:
        raise ValueError(f"Unsupported output type: {output_type}")

    processed_true_sentences = []
    for true_sent in true_sentences:
        if output_type == "handwriting":
            true_sent = true_sent.replace(">", " ")
            true_sent = true_sent.replace("~", ".")
            true_sent = true_sent.replace("#", "")
        if output_type in {"speech", "speech_sil"}:
            true_sent = true_sent.strip()
        processed_true_sentences.append(true_sent)

    cer, wer_score = _cer_and_wer(
        decoded_sentences, processed_true_sentences, output_type, return_ci
    )
    return {
        "cer": cer,
        "wer": wer_score,
        "decoded_transcripts": decoded_sentences,
        "confidences": confidences,
        "ranked_candidates": ranked_candidates,
    }


def build_lm_decoder(
    model_path,
    max_active=7000,
    min_active=200,
    beam=17.0,
    lattice_beam=8.0,
    acoustic_scale=1.5,
    ctc_blank_skip_threshold=1.0,
    length_penalty=0.0,
    nbest=1,
):
    try:
        import lm_decoder
    except ImportError as exc:
        raise ImportError(
            "Could not import 'lm_decoder'. Build and install it from "
            "speechBCI/LanguageModelDecoder/runtime/server/x86 first."
        ) from exc

    decode_opts = lm_decoder.DecodeOptions(
        max_active,
        min_active,
        beam,
        lattice_beam,
        acoustic_scale,
        ctc_blank_skip_threshold,
        length_penalty,
        nbest,
    )

    tlg_path = os.path.join(model_path, "TLG.fst")
    words_path = os.path.join(model_path, "words.txt")
    g_path = os.path.join(model_path, "G.fst")
    rescore_g_path = os.path.join(model_path, "G_no_prune.fst")
    if not os.path.exists(rescore_g_path):
        rescore_g_path = ""
        g_path = ""
    if not os.path.exists(tlg_path):
        raise ValueError(f"TLG file not found at {tlg_path}")
    if not os.path.exists(words_path):
        raise ValueError(f"words file not found at {words_path}")

    decode_resource = lm_decoder.DecodeResource(
        tlg_path,
        g_path,
        rescore_g_path,
        words_path,
        "",
    )
    return lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)


def lm_decode(
    decoder,
    logits,
    return_nbest=False,
    rescore=False,
    blank_penalty=0.0,
    log_priors=None,
):
    try:
        import lm_decoder
    except ImportError as exc:
        raise ImportError(
            "Could not import 'lm_decoder'. Build and install it from "
            "speechBCI/LanguageModelDecoder/runtime/server/x86 first."
        ) from exc

    assert len(logits.shape) == 2
    if log_priors is None:
        log_priors = np.zeros([1, logits.shape[1]])

    lm_decoder.DecodeNumpy(decoder, logits, log_priors, blank_penalty)
    decoder.FinishDecoding()
    if rescore:
        decoder.Rescore()

    if not return_nbest:
        decoded = "" if len(decoder.result()) == 0 else decoder.result()[0].sentence
    else:
        decoded = [
            (result.sentence, result.ac_score, result.lm_score)
            for result in decoder.result()
        ]

    decoder.Reset()
    return decoded


def nbest_with_lm_decoder(
    decoder,
    inference_out,
    include_space_symbol=True,
    output_type="handwriting",
    rescore=False,
    blank_penalty=0.0,
):
    logits = inference_out["logits"]
    logit_lengths = inference_out["logitLengths"]
    if output_type == "handwriting":
        logits = rearrange_handwriting_logits(logits, include_space_symbol)
    elif output_type in {"speech", "speech_sil"}:
        logits = rearrange_speech_logits(logits, has_sil=(output_type == "speech_sil"))

    nbest_outputs = []
    for i in range(len(logits)):
        nbest = lm_decode(
            decoder,
            logits[i, : logit_lengths[i]],
            return_nbest=True,
            blank_penalty=blank_penalty,
            rescore=rescore,
        )
        nbest_outputs.append(nbest)
    return nbest_outputs


def cer_with_lm_decoder(
    decoder,
    inference_out,
    include_space_symbol=True,
    output_type="handwriting",
    return_ci=False,
    rescore=False,
    blank_penalty=0.0,
    log_priors=None,
):
    logits = inference_out["logits"]
    if output_type == "handwriting":
        logits = rearrange_handwriting_logits(logits, include_space_symbol)
        true_sentences = _extract_true_sentences(inference_out)
    elif output_type in {"speech", "speech_sil"}:
        logits = rearrange_speech_logits(logits, has_sil=("speech_sil" == output_type))
        true_sentences = _extract_transcriptions(inference_out)
    else:
        raise ValueError(f"Unsupported output type: {output_type}")

    decoded_sentences = []
    decode_time = []
    for i in trange(len(inference_out["logits"])):
        logit_len = inference_out["logitLengths"][i]
        start = time()
        decoded = lm_decode(
            decoder,
            logits[i, :logit_len],
            rescore=rescore,
            blank_penalty=blank_penalty,
            log_priors=log_priors,
        )

        if output_type == "handwriting":
            decoded = decoded.replace(" ", "" if include_space_symbol else ">")
            decoded = decoded.replace(".", "~")
        elif output_type in {"speech", "speech_sil"}:
            decoded = decoded.strip()

        decode_time.append((time() - start) * 1000)
        decoded_sentences.append(decoded)

    cer, wer_score = _cer_and_wer(decoded_sentences, true_sentences, output_type, return_ci)
    return {
        "cer": cer,
        "wer": wer_score,
        "decoded_transcripts": decoded_sentences,
        "true_transcripts": true_sentences,
        "decode_time": decode_time,
    }


def rearrange_handwriting_logits(logits, include_space_symbol=True):
    char_range = list(range(0, 26))
    symbol_range = [26, 27, 30, 29, 28] if include_space_symbol else [27, 30, 29, 28]
    return logits[:, :, [31] + symbol_range + char_range]


def rearrange_speech_logits(logits, has_sil=False):
    if not has_sil:
        return np.concatenate([logits[:, :, -1:], logits[:, :, :-1]], axis=-1)
    return np.concatenate([logits[:, :, -1:], logits[:, :, -2:-1], logits[:, :, :-2]], axis=-1)


def _cer_and_wer(decoded_sentences, true_sentences, output_type="handwriting", return_ci=False):
    all_char_err = []
    all_char = []
    all_word_err = []
    all_word = []
    for decoded, truth in zip(decoded_sentences, true_sentences):
        n_char_err = wer([c for c in truth], [c for c in decoded])
        if output_type == "handwriting":
            true_words = truth.replace(">", " > ").split(" ")
            decoded_words = decoded.replace(">", " > ").split(" ")
        elif output_type in {"speech", "speech_sil"}:
            true_words = truth.split(" ")
            decoded_words = decoded.split(" ")
        else:
            raise ValueError(f"Unsupported output type: {output_type}")

        n_word_err = wer(true_words, decoded_words)
        all_char_err.append(n_char_err)
        all_word_err.append(n_word_err)
        all_char.append(len(truth))
        all_word.append(len(true_words))

    cer = np.sum(all_char_err) / np.sum(all_char)
    wer_score = np.sum(all_word_err) / np.sum(all_word)
    if not return_ci:
        return cer, wer_score

    all_char = np.array(all_char)
    all_char_err = np.array(all_char_err)
    all_word = np.array(all_word)
    all_word_err = np.array(all_word_err)

    n_resamples = 10000
    resampled_cer = np.zeros([n_resamples])
    resampled_wer = np.zeros([n_resamples])
    for n in range(n_resamples):
        resample_idx = np.random.randint(0, all_char.shape[0], [all_char.shape[0]])
        resampled_cer[n] = np.sum(all_char_err[resample_idx]) / np.sum(all_char[resample_idx])
        resampled_wer[n] = np.sum(all_word_err[resample_idx]) / np.sum(all_word[resample_idx])

    cer_ci = np.percentile(resampled_cer, [2.5, 97.5])
    wer_ci = np.percentile(resampled_wer, [2.5, 97.5])
    return (cer, cer_ci[0], cer_ci[1]), (wer_score, wer_ci[0], wer_ci[1])


def compute_cer_wer(decoded_sentences, true_sentences, output_type="speech_sil", return_ci=False):
    return _cer_and_wer(decoded_sentences, true_sentences, output_type, return_ci)


def _extract_transcriptions(inference_out):
    transcriptions = []
    for transcription in inference_out["transcriptions"]:
        end_idx = np.argwhere(transcription == 0)[0, 0]
        transcriptions.append("".join(chr(transcription[c]) for c in range(end_idx)))
    return transcriptions


def _extract_true_sentences(inference_out):
    char_marks = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        ">",
        ",",
        "'",
        "~",
        "?",
    ]

    true_sentences = []
    for true_seq in inference_out["trueSeqs"]:
        end_idx = np.argwhere(true_seq == -1)[0, 0]
        true_sentences.append("".join(char_marks[true_seq[c]] for c in range(end_idx)))
    return true_sentences
