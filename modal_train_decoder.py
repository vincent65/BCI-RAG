import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

import modal
import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_NAME = "config_2"
DEFAULT_GPU = "A100-40GB"
DEFAULT_TIMEOUT_SECONDS = 24 * 60 * 60

DEFAULT_MODAL_APP_NAME = "speechbci-decoder-training"
DEFAULT_DATA_VOLUME_NAME = "speechbci-training-data"
DEFAULT_OUTPUT_VOLUME_NAME = "speechbci-training-output"
DATA_MOUNT_PATH = "/training-data"
OUTPUT_MOUNT_PATH = "/training-output"

modal_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "edit-distance==1.0.6",
        "numpy==1.26.4",
        "pyyaml==6.0.1",
        "pytorch-lightning==2.2.4",
        "tensorboard==2.16.2",
        "torch==2.2.2",
    )
    .add_local_python_source("neural_decoder")
)

data_volume = modal.Volume.from_name(
    DEFAULT_DATA_VOLUME_NAME,
    create_if_missing=True,
)
output_volume = modal.Volume.from_name(
    DEFAULT_OUTPUT_VOLUME_NAME,
    create_if_missing=True,
)

app = modal.App(
    name=DEFAULT_MODAL_APP_NAME,
    image=modal_image,
    volumes={
        DATA_MOUNT_PATH: data_volume,
        OUTPUT_MOUNT_PATH: output_volume,
    },
)


def _load_training_config(config_name):
    config_path = REPO_ROOT / "conf" / f"{config_name}.yaml"
    with open(config_path, "r") as handle:
        loaded = yaml.safe_load(handle)

    # Keep only the flat training arguments used by the trainer/model code.
    loaded.pop("hydra", None)
    loaded.pop("defaults", None)
    loaded.pop("wandb", None)
    return loaded


def _normalize_training_types(args):
    int_keys = [
        "seed",
        "batchSize",
        "nSteps",
        "nUnits",
        "nLayers",
        "nInputFeatures",
        "nClasses",
        "strideLen",
        "kernelLen",
        "stepSize",
        "numWorkers",
        "devices",
    ]
    float_keys = [
        "seqLen",
        "lrStart",
        "lrEnd",
        "l2_decay",
        "whiteNoiseSD",
        "constantOffsetSD",
        "gaussianSmoothWidth",
        "dropout",
        "momentum",
        "gamma",
    ]
    bool_keys = ["bidirectional", "nesterov"]

    normalized = dict(args)
    for key in int_keys:
        normalized[key] = int(normalized[key])
    for key in float_keys:
        normalized[key] = float(normalized[key])
    for key in bool_keys:
        value = normalized[key]
        if isinstance(value, str):
            normalized[key] = value.lower() == "true"
        else:
            normalized[key] = bool(value)
    return normalized


def _build_training_args(
    config_name,
    remote_dataset_path,
    run_name,
    batch_size=128,
    num_workers=4,
    precision="16-mixed",
):
    args = _load_training_config(config_name)
    args["batchSize"] = batch_size
    args["numWorkers"] = num_workers
    args["precision"] = precision
    args["devices"] = 1
    args["accelerator"] = "gpu"
    args["datasetPath"] = remote_dataset_path
    args["outputDir"] = f"{OUTPUT_MOUNT_PATH}/{config_name}/{run_name}"
    return _normalize_training_types(args)


def _upload_dataset(local_dataset_path):
    local_dataset_path = Path(local_dataset_path).expanduser().resolve()
    if not local_dataset_path.exists():
        raise FileNotFoundError(f"Dataset pickle not found: {local_dataset_path}")

    remote_relpath = f"/{local_dataset_path.name}"
    try:
        with data_volume.batch_upload() as batch:
            batch.put_file(str(local_dataset_path), remote_relpath)
    except FileExistsError:
        pass

    return f"{DATA_MOUNT_PATH}/{local_dataset_path.name}"


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SECONDS,
)
def train_decoder_remote(training_args):
    import pickle
    import time
    from pathlib import Path

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint

    from neural_decoder.dataset import SpeechDataModule
    from neural_decoder.model import GRUDecoder

    torch.set_float32_matmul_precision("medium")

    pl.seed_everything(training_args["seed"], workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_args["seed"])
    torch.backends.cudnn.deterministic = True

    output_dir = Path(training_args["outputDir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args", "wb") as handle:
        pickle.dump(training_args, handle)

    with open(training_args["datasetPath"], "rb") as handle:
        loaded_data = pickle.load(handle)

    data_module = SpeechDataModule(
        loaded_data,
        training_args["batchSize"],
        training_args["numWorkers"],
    )

    model = GRUDecoder(
        neural_dim=training_args["nInputFeatures"],
        n_classes=training_args["nClasses"],
        hidden_dim=training_args["nUnits"],
        layer_dim=training_args["nLayers"],
        nDays=len(loaded_data["train"]),
        dropout=training_args["dropout"],
        strideLen=training_args["strideLen"],
        kernelLen=training_args["kernelLen"],
        gaussianSmoothWidth=training_args["gaussianSmoothWidth"],
        whiteNoiseSD=training_args["whiteNoiseSD"],
        constantOffsetSD=training_args["constantOffsetSD"],
        bidirectional=training_args["bidirectional"],
        l2_decay=training_args["l2_decay"],
        lrStart=training_args["lrStart"],
        lrEnd=training_args["lrEnd"],
        momentum=training_args["momentum"],
        nesterov=training_args["nesterov"],
        gamma=training_args["gamma"],
        stepSize=training_args["stepSize"],
        nBatch=training_args["nSteps"],
        output_dir=str(output_dir),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="modelWeights",
        monitor="val/ser",
        mode="min",
        save_top_k=1,
        every_n_train_steps=None,
    )
    checkpoint_callback.FILE_EXTENSION = ""

    trainer = pl.Trainer(
        logger=False,
        min_steps=1,
        max_steps=training_args["nSteps"],
        accelerator=training_args["accelerator"],
        devices=training_args["devices"],
        precision=training_args["precision"],
        num_nodes=1,
        log_every_n_steps=1,
        val_check_interval=100,
        check_val_every_n_epoch=None,
        callbacks=[checkpoint_callback],
    )

    started_at = time.time()
    trainer.fit(model, data_module)
    elapsed_seconds = time.time() - started_at

    best_model_path = checkpoint_callback.best_model_path or str(output_dir / "modelWeights")
    result = {
        "run_dir": str(output_dir),
        "volume_run_dir": str(Path("/") / output_dir.relative_to(OUTPUT_MOUNT_PATH)),
        "checkpoint_path": best_model_path,
        "best_score": (
            float(checkpoint_callback.best_model_score.cpu().item())
            if checkpoint_callback.best_model_score is not None
            else None
        ),
        "elapsed_seconds": elapsed_seconds,
    }
    with open(output_dir / "modal_train_result.json", "w") as handle:
        json.dump(result, handle, indent=2)

    output_volume.commit()
    return result


@app.local_entrypoint()
def train_and_fetch(
    config_name: str = DEFAULT_CONFIG_NAME,
    dataset_path: str = str(REPO_ROOT / "data" / "ptDecoder_ctc"),
    run_name: str = "",
    batch_size: int = 128,
    num_workers: int = 4,
    precision: str = "16-mixed",
    local_download_root: str = str(REPO_ROOT / "modal_runs"),
):
    if not run_name:
        run_name = f"{config_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    remote_dataset_path = _upload_dataset(dataset_path)
    training_args = _build_training_args(
        config_name=config_name,
        remote_dataset_path=remote_dataset_path,
        run_name=run_name,
        batch_size=batch_size,
        num_workers=num_workers,
        precision=precision,
    )

    result = train_decoder_remote.remote(training_args)

    local_run_dir = Path(local_download_root).expanduser().resolve() / config_name / run_name
    local_run_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "--force",
            DEFAULT_OUTPUT_VOLUME_NAME,
            result["volume_run_dir"],
            str(local_run_dir),
        ],
        check=True,
    )

    downloaded_checkpoint = local_run_dir / "modelWeights"
    print(json.dumps(result, indent=2))
    print(f"Downloaded run to: {local_run_dir}")
    print(f"Checkpoint path: {downloaded_checkpoint}")
