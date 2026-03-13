import argparse
import pickle
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "ptDecoder_ctc"
DEFAULT_LM_DIR = REPO_ROOT / "external" / "lang_test"
DEFAULT_MODEL_CACHE_DIR = REPO_ROOT / "models" / "opt-6.7b"
DEFAULT_SPEECH_BCI_ROOT = REPO_ROOT.parent / "speechBCI"


def status_line(ok, label, detail):
    prefix = "[ok]" if ok else "[missing]"
    print(f"{prefix} {label}: {detail}")


def check_path(path, label, required_files=None):
    required_files = required_files or []
    path = Path(path).expanduser().resolve()
    if not path.exists():
        status_line(False, label, path)
        return False, path

    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        status_line(False, label, f"{path} (missing: {', '.join(missing)})")
        return False, path

    status_line(True, label, path)
    return True, path


def check_dataset(path):
    ok, dataset_path = check_path(path, "dataset pickle")
    if not ok:
        return False

    try:
        with open(dataset_path, "rb") as handle:
            loaded = pickle.load(handle)
    except Exception as exc:
        status_line(False, "dataset contents", f"{dataset_path} ({exc})")
        return False

    missing_splits = [split for split in ("train", "test", "competition") if split not in loaded]
    if missing_splits:
        status_line(False, "dataset contents", f"missing splits: {', '.join(missing_splits)}")
        return False

    status_line(
        True,
        "dataset contents",
        f"train={len(loaded['train'])}, test={len(loaded['test'])}, competition={len(loaded['competition'])}",
    )
    return True


def check_lm_decoder_import():
    try:
        import lm_decoder  # noqa: F401
    except Exception as exc:
        status_line(False, "lm_decoder Python module", exc)
        return False

    status_line(True, "lm_decoder Python module", "import succeeded")
    return True


def check_optional_directory(path, label):
    path = Path(path).expanduser().resolve()
    if path.exists():
        status_line(True, label, path)
    else:
        status_line(True, label, f"{path} (will be created on first model download)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Check whether speechBCI_2024 is ready to run.")
    parser.add_argument("--datasetPath", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--lmDir", type=str, default=str(DEFAULT_LM_DIR))
    parser.add_argument("--modelPath", type=str, default="")
    parser.add_argument("--modelCacheDir", type=str, default=str(DEFAULT_MODEL_CACHE_DIR))
    parser.add_argument("--speechBciRoot", type=str, default=str(DEFAULT_SPEECH_BCI_ROOT))
    args = parser.parse_args()

    print(f"Repo root: {REPO_ROOT}")
    print()

    all_ok = True
    dataset_ok = check_dataset(args.datasetPath)
    all_ok = all_ok and dataset_ok

    lm_dir_ok, _ = check_path(args.lmDir, "5-gram LM directory", ["TLG.fst", "words.txt"])
    all_ok = all_ok and lm_dir_ok

    if args.modelPath:
        model_ok, _ = check_path(args.modelPath, "model checkpoint")
        all_ok = all_ok and model_ok
    else:
        status_line(False, "model checkpoint", "not provided")
        all_ok = False

    check_optional_directory(args.modelCacheDir, "OPT cache directory")

    speech_bci_ok, speech_bci_root = check_path(args.speechBciRoot, "sibling speechBCI repo")
    if speech_bci_ok:
        runtime_dir = speech_bci_root / "LanguageModelDecoder" / "runtime" / "server" / "x86"
        check_path(runtime_dir, "speechBCI LM runtime source", ["setup.py", "CMakeLists.txt"])

    lm_decoder_ok = check_lm_decoder_import()
    all_ok = all_ok and lm_decoder_ok

    print()
    if all_ok:
        print("Environment looks ready for evaluation.")
        sys.exit(0)

    print("Environment is not fully ready yet.")
    if speech_bci_ok and not lm_decoder_ok:
        runtime_dir = speech_bci_root / "LanguageModelDecoder" / "runtime" / "server" / "x86"
        print("To build lm_decoder in the active environment:")
        print(f"  cd {runtime_dir}")
        print("  python setup.py install")
    print("If the dataset pickle is missing, prepare it with notebooks/formatCompetitionData.ipynb.")
    print("If the 5-gram graph is missing, download/build the Stanford language model assets into external/lang_test.")
    sys.exit(1)


if __name__ == "__main__":
    main()
