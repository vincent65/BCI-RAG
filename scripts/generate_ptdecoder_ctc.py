#!/usr/bin/env python3
import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import scipy.io
from g2p_en import G2p


SESSION_NAMES = [
    "t12.2022.04.28",
    "t12.2022.05.26",
    "t12.2022.06.21",
    "t12.2022.07.21",
    "t12.2022.08.13",
    "t12.2022.05.05",
    "t12.2022.06.02",
    "t12.2022.06.23",
    "t12.2022.07.27",
    "t12.2022.08.18",
    "t12.2022.05.17",
    "t12.2022.06.07",
    "t12.2022.06.28",
    "t12.2022.07.29",
    "t12.2022.08.23",
    "t12.2022.05.19",
    "t12.2022.06.14",
    "t12.2022.07.05",
    "t12.2022.08.02",
    "t12.2022.08.25",
    "t12.2022.05.24",
    "t12.2022.06.16",
    "t12.2022.07.14",
    "t12.2022.08.11",
]
SESSION_NAMES.sort()

PHONE_DEF = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
PHONE_DEF_SIL = PHONE_DEF + ["SIL"]
MAX_SEQ_LEN = 500


def phone_to_id(phone):
    return PHONE_DEF_SIL.index(phone)


def load_features_and_normalize(session_path):
    data = scipy.io.loadmat(session_path)

    input_features = []
    transcriptions = []
    frame_lens = []
    n_trials = data["sentenceText"].shape[0]

    for trial_idx in range(n_trials):
        features = np.concatenate(
            [
                data["tx1"][0, trial_idx][:, 0:128],
                data["spikePow"][0, trial_idx][:, 0:128],
            ],
            axis=1,
        )

        input_features.append(features)
        transcriptions.append(data["sentenceText"][trial_idx].strip())
        frame_lens.append(features.shape[0])

    block_nums = np.squeeze(data["blockIdx"])
    block_list = np.unique(block_nums)
    blocks = []
    for block in block_list:
        sent_idx = np.argwhere(block_nums == block)[:, 0].astype(np.int32)
        blocks.append(sent_idx)

    # Match the original notebook's blockwise z-scoring.
    for block in blocks:
        feats = np.concatenate(input_features[block[0] : (block[-1] + 1)], axis=0)
        feats_mean = np.mean(feats, axis=0, keepdims=True)
        feats_std = np.std(feats, axis=0, keepdims=True)
        for idx in block:
            input_features[idx] = (input_features[idx] - feats_mean) / (feats_std + 1e-8)

    return {
        "inputFeatures": input_features,
        "transcriptions": transcriptions,
        "frameLens": frame_lens,
    }


def get_dataset(file_name, g2p):
    session_data = load_features_and_normalize(file_name)

    all_dat = []
    true_sentences = []
    seq_elements = []

    for idx in range(len(session_data["inputFeatures"])):
        all_dat.append(session_data["inputFeatures"][idx])
        true_sentences.append(session_data["transcriptions"][idx])

        transcription = str(session_data["transcriptions"][idx]).strip()
        transcription = re.sub(r"[^a-zA-Z\- ']", "", transcription)
        transcription = transcription.replace("--", "").lower()

        phonemes = []
        for phone in g2p(transcription):
            if phone == " ":
                phonemes.append("SIL")
            phone = re.sub(r"[0-9]", "", phone)
            if re.match(r"[A-Z]+", phone):
                phonemes.append(phone)

        phonemes.append("SIL")

        seq_len = len(phonemes)
        seq_class_ids = np.zeros([MAX_SEQ_LEN], dtype=np.int32)
        seq_class_ids[0:seq_len] = [phone_to_id(phone) + 1 for phone in phonemes]
        seq_elements.append(seq_class_ids)

    new_dataset = {
        "sentenceDat": all_dat,
        "transcriptions": true_sentences,
        "phonemes": seq_elements,
    }

    time_series_lens = []
    phone_lens = []
    for idx in range(len(new_dataset["sentenceDat"])):
        time_series_lens.append(new_dataset["sentenceDat"][idx].shape[0])
        zero_idx = np.argwhere(new_dataset["phonemes"][idx] == 0)
        phone_lens.append(zero_idx[0, 0])

    new_dataset["timeSeriesLens"] = np.array(time_series_lens)
    new_dataset["phoneLens"] = np.array(phone_lens)
    new_dataset["phonePerTime"] = (
        new_dataset["phoneLens"].astype(np.float32)
        / new_dataset["timeSeriesLens"].astype(np.float32)
    )
    return new_dataset


def build_split_dataset(data_dir, split_name, g2p):
    datasets = []
    split_dir = data_dir / split_name
    for day_idx, session_name in enumerate(SESSION_NAMES):
        session_path = split_dir / f"{session_name}.mat"
        if not session_path.exists():
            if split_name == "competitionHoldOut":
                continue
            raise FileNotFoundError(f"Missing expected file: {session_path}")

        print(f"[{split_name}] {day_idx + 1}/{len(SESSION_NAMES)} {session_name}")
        datasets.append(get_dataset(session_path, g2p))
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Generate the ptDecoder_ctc pickle from competitionData .mat files."
    )
    parser.add_argument(
        "--competitionDataDir",
        type=Path,
        default=Path("competitionData"),
        help="Directory containing train/, test/, and competitionHoldOut/ .mat files.",
    )
    parser.add_argument(
        "--outputPath",
        type=Path,
        default=Path("data/ptDecoder_ctc"),
        help="Path to write the generated pickle.",
    )
    args = parser.parse_args()

    data_dir = args.competitionDataDir.expanduser().resolve()
    output_path = args.outputPath.expanduser().resolve()

    for split_name in ("train", "test", "competitionHoldOut"):
        split_dir = data_dir / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Initializing grapheme-to-phoneme model...")
    g2p = G2p()

    train_datasets = build_split_dataset(data_dir, "train", g2p)
    test_datasets = build_split_dataset(data_dir, "test", g2p)
    competition_datasets = build_split_dataset(data_dir, "competitionHoldOut", g2p)

    all_datasets = {
        "train": train_datasets,
        "test": test_datasets,
        "competition": competition_datasets,
    }

    with open(output_path, "wb") as handle:
        pickle.dump(all_datasets, handle)

    print()
    print(f"Wrote dataset pickle to: {output_path}")
    print(
        "Split sizes: "
        f"train={len(train_datasets)}, "
        f"test={len(test_datasets)}, "
        f"competition={len(competition_datasets)}"
    )


if __name__ == "__main__":
    main()
