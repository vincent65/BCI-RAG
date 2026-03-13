from pathlib import Path
import pickle

import torch

from .model import GRUDecoder


def _load_adjacent_args(model_weight_path):
    model_weight_path = Path(model_weight_path).expanduser().resolve()
    candidate_paths = [
        model_weight_path.with_name("args"),
        model_weight_path.parent / "args",
        model_weight_path.parent.parent / "args",
    ]
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        with open(candidate, "rb") as handle:
            loaded = pickle.load(handle)
        if isinstance(loaded, dict):
            return loaded
    return None


def _normalize_model_args(raw_args):
    if raw_args is None:
        raise KeyError("Model checkpoint is missing hyperparameters and no adjacent args file was found.")

    canonical = {
        "neural_dim": raw_args.get("neural_dim", raw_args.get("nInputFeatures")),
        "n_classes": raw_args.get("n_classes", raw_args.get("nClasses")),
        "hidden_dim": raw_args.get("hidden_dim", raw_args.get("nUnits")),
        "layer_dim": raw_args.get("layer_dim", raw_args.get("nLayers")),
        "dropout": raw_args.get("dropout"),
        "strideLen": raw_args.get("strideLen"),
        "kernelLen": raw_args.get("kernelLen"),
        "gaussianSmoothWidth": raw_args.get("gaussianSmoothWidth"),
        "whiteNoiseSD": raw_args.get("whiteNoiseSD", 0.0),
        "constantOffsetSD": raw_args.get("constantOffsetSD", 0.0),
        "bidirectional": raw_args.get("bidirectional"),
        "l2_decay": raw_args.get("l2_decay", 0.0),
        "lrStart": raw_args.get("lrStart", raw_args.get("lr", 0.0)),
        "lrEnd": raw_args.get("lrEnd", 0.0),
        "momentum": raw_args.get("momentum", 0.0),
        "nesterov": raw_args.get("nesterov", False),
        "gamma": raw_args.get("gamma", 1.0),
        "stepSize": raw_args.get("stepSize", 1),
        "nSteps": raw_args.get("nSteps", raw_args.get("nBatch", 1)),
        "output_dir": raw_args.get("output_dir", raw_args.get("outputDir", "")),
    }

    missing = [key for key, value in canonical.items() if value is None]
    if missing:
        raise KeyError(f"Model hyperparameters are missing required keys: {', '.join(missing)}")
    return canonical


def load_checkpoint_bundle(model_weight_path, map_location="cpu"):
    model_weight_path = Path(model_weight_path).expanduser().resolve()
    checkpoint = torch.load(str(model_weight_path), map_location=map_location)
    raw_args = checkpoint.get("hyper_parameters")
    if raw_args is None:
        raw_args = _load_adjacent_args(model_weight_path)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint at {model_weight_path} does not contain a 'state_dict'.")
    return {
        "checkpoint_path": str(model_weight_path),
        "hyper_parameters": _normalize_model_args(raw_args),
        "state_dict": state_dict,
    }


def load_model_config(model_weight_path):
    bundle = load_checkpoint_bundle(model_weight_path, map_location="cpu")
    return bundle["hyper_parameters"]


def load_model_from_checkpoint(model_weight_path, nInputLayers=24, device="cuda"):
    bundle = load_checkpoint_bundle(model_weight_path, map_location=device)
    args = bundle["hyper_parameters"]
    model = GRUDecoder(
        neural_dim=args["neural_dim"],
        n_classes=args["n_classes"],
        hidden_dim=args["hidden_dim"],
        layer_dim=args["layer_dim"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        whiteNoiseSD=args["whiteNoiseSD"],
        constantOffsetSD=args["constantOffsetSD"],
        bidirectional=args["bidirectional"],
        l2_decay=args["l2_decay"],
        lrStart=args["lrStart"],
        lrEnd=args["lrEnd"],
        momentum=args["momentum"],
        nesterov=args["nesterov"],
        gamma=args["gamma"],
        stepSize=args["stepSize"],
        nBatch=args["nSteps"],
        output_dir=args["output_dir"],
    ).to(device)
    model.load_state_dict(bundle["state_dict"])
    return model
