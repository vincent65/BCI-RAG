#!/usr/bin/env python3
import argparse
from pathlib import Path

from neural_decoder.lm_utils import (
    build_openai_rescorer,
    build_opt,
    choose_option_with_openai,
    compute_cer_wer,
    is_openai_backend,
    rescore_with_gpt2,
)


def load_sentences(path):
    with open(path, "r") as handle:
        return [line.rstrip("\n") for line in handle]


def write_sentences(path, sentences):
    with open(path, "w") as handle:
        for sentence in sentences:
            handle.write(sentence + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct the two-model sentence-level LLM ensemble."
    )
    parser.add_argument("--model1File", type=Path, required=True, help="Decoded text from model 1.")
    parser.add_argument("--model2File", type=Path, required=True, help="Decoded text from model 2.")
    parser.add_argument("--outputFile", type=Path, required=True, help="Where to write the ensembled output.")
    parser.add_argument("--targetFile", type=Path, default=None, help="Optional reference file for WER/CER.")
    parser.add_argument(
        "--modelName",
        type=str,
        default="facebook/opt-6.7b",
        help="Causal LM used to score each model's candidate sentence.",
    )
    parser.add_argument(
        "--llmBackend",
        type=str,
        choices=["local", "openai"],
        default="local",
        help="Whether to use a local Hugging Face model or the OpenAI API.",
    )
    parser.add_argument(
        "--modelCacheDir",
        type=str,
        default="models/opt-6.7b",
        help="Cache directory for the scoring model.",
    )
    parser.add_argument(
        "--disable8bit",
        action="store_true",
        help="Disable 8-bit model loading for the scoring model.",
    )
    args = parser.parse_args()

    model1_sentences = load_sentences(args.model1File)
    model2_sentences = load_sentences(args.model2File)
    if len(model1_sentences) != len(model2_sentences):
        raise ValueError("The two decoded files must contain the same number of lines.")

    decoded = []
    picks = []
    if args.llmBackend == "openai":
        openai_model_name = (
            "gpt-5-2" if args.modelName == "facebook/opt-6.7b" else args.modelName
        )
        llm = build_openai_rescorer(model_name=openai_model_name)
        tokenizer = None
    else:
        llm, tokenizer = build_opt(
            model_name=args.modelName,
            cache_dir=args.modelCacheDir,
            device="auto",
            load_in_8bit=not args.disable8bit,
        )

    if is_openai_backend(llm):
        for idx, (sent1, sent2) in enumerate(zip(model1_sentences, model2_sentences)):
            prompt = (
                "Choose the sentence that is more plausible English for a speech BCI decode.\n"
                "Return only the best option number."
            )
            choice_idx, raw_response = choose_option_with_openai(llm, prompt, [sent1, sent2])
            picked = 1 if choice_idx == 0 else 2
            decoded.append(sent1 if picked == 1 else sent2)
            picks.append({"index": idx, "picked": picked, "raw_response": raw_response})
    else:
        model1_scores = rescore_with_gpt2(llm, tokenizer, model1_sentences, length_penalty=0.0)
        model2_scores = rescore_with_gpt2(llm, tokenizer, model2_sentences, length_penalty=0.0)
        for idx, (sent1, sent2, score1, score2) in enumerate(
            zip(model1_sentences, model2_sentences, model1_scores, model2_scores)
        ):
            if score1 >= score2:
                decoded.append(sent1)
                picks.append({"index": idx, "picked": 1, "score1": float(score1), "score2": float(score2)})
            else:
                decoded.append(sent2)
                picks.append({"index": idx, "picked": 2, "score1": float(score1), "score2": float(score2)})

    output_path = args.outputFile.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_sentences(output_path, decoded)
    print(f"Wrote ensemble output to {output_path}")

    if args.targetFile is not None:
        targets = load_sentences(args.targetFile)
        cer, wer = compute_cer_wer(decoded, targets, output_type="speech_sil", return_ci=False)
        print(f"Model 1 CER/WER not recomputed here; ensemble CER={cer:.6f} WER={wer:.6f}")


if __name__ == "__main__":
    main()
