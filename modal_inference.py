import hashlib
from pathlib import Path

import modal
import numpy as np
import torch

DEFAULT_MODAL_APP_NAME = "speechbci-gpu-offload"
DEFAULT_MODAL_GPU = "A10G"
DEFAULT_CHECKPOINT_VOLUME_NAME = "speechbci-model-checkpoints"
CHECKPOINT_MOUNT_PATH = "/checkpoints"

modal_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate==0.30.1",
        "bitsandbytes==0.43.1",
        "edit-distance==1.0.6",
        "numpy==1.26.4",
        "pytorch-lightning==2.2.4",
        "requests==2.32.2",
        "torch==2.2.2",
        "transformers==4.41.1",
    )
    .add_local_python_source("neural_decoder")
)

checkpoint_volume = modal.Volume.from_name(
    DEFAULT_CHECKPOINT_VOLUME_NAME,
    create_if_missing=True,
)

app = modal.App(
    name=DEFAULT_MODAL_APP_NAME,
    image=modal_image,
    volumes={CHECKPOINT_MOUNT_PATH: checkpoint_volume},
)


def _checkpoint_digest(model_path):
    hasher = hashlib.sha256()
    with open(model_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _find_adjacent_args(model_path):
    model_path = Path(model_path).expanduser().resolve()
    candidate_paths = [
        model_path.with_name("args"),
        model_path.parent / "args",
        model_path.parent.parent / "args",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None


def upload_checkpoint_to_modal(model_path, volume_name=DEFAULT_CHECKPOINT_VOLUME_NAME):
    model_path = Path(model_path).expanduser().resolve()
    digest = _checkpoint_digest(model_path)
    remote_dir = f"{digest[:16]}_{model_path.name}_bundle"
    remote_relpath = f"{remote_dir}/{model_path.name}"
    args_path = _find_adjacent_args(model_path)
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    try:
        with volume.batch_upload() as batch:
            batch.put_file(str(model_path), f"/{remote_relpath}")
            if args_path is not None:
                batch.put_file(str(args_path), f"/{remote_dir}/args")
    except FileExistsError:
        pass
    return remote_relpath


def _resolve_remote_class(app_name, class_name, gpu):
    remote_class = modal.Cls.from_name(app_name, class_name)
    if gpu:
        remote_class = remote_class.with_options(gpu=gpu)
    return remote_class


class ModalGRUProxy:
    provider = "modal"

    def __init__(
        self,
        model_path,
        app_name=DEFAULT_MODAL_APP_NAME,
        gpu=DEFAULT_MODAL_GPU,
        checkpoint_volume_name=DEFAULT_CHECKPOINT_VOLUME_NAME,
    ):
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.checkpoint_relpath = upload_checkpoint_to_modal(
            self.model_path,
            volume_name=checkpoint_volume_name,
        )
        remote_class = _resolve_remote_class(app_name, "RemoteGRUDecoder", gpu=gpu)
        self._remote = remote_class(checkpoint_relpath=self.checkpoint_relpath)
        metadata = self._remote.metadata.remote()
        self.kernelLen = int(metadata["kernelLen"])
        self.strideLen = int(metadata["strideLen"])

    def eval(self):
        return self

    def forward(self, X, dayIdx):
        prediction = self._remote.predict.remote(
            np.asarray(X.detach().cpu().numpy(), dtype=np.float32),
            np.asarray(dayIdx.detach().cpu().numpy(), dtype=np.int64),
        )
        return torch.from_numpy(np.asarray(prediction, dtype=np.float32))


class ModalOPTProxy:
    provider = "modal"

    def __init__(
        self,
        model_name,
        app_name=DEFAULT_MODAL_APP_NAME,
        gpu=DEFAULT_MODAL_GPU,
        load_in_8bit=False,
    ):
        self.model_name = model_name
        remote_class = _resolve_remote_class(app_name, "RemoteOPTScorer", gpu=gpu)
        self._remote = remote_class(
            model_name=model_name,
            load_in_8bit=load_in_8bit,
        )

    def rescore(self, hypotheses, length_penalty=0.0):
        return self._remote.rescore.remote(hypotheses, length_penalty=length_penalty)

    def score_prompted(self, prompt, completions, length_penalty=0.0):
        return self._remote.score_prompted.remote(
            prompt,
            completions,
            length_penalty=length_penalty,
        )


@app.cls(
    gpu=DEFAULT_MODAL_GPU,
    timeout=900,
    scaledown_window=300,
)
class RemoteGRUDecoder:
    checkpoint_relpath: str = modal.parameter()

    @modal.enter()
    def load(self):
        from neural_decoder.checkpoint_utils import (
            load_model_config,
            load_model_from_checkpoint,
        )

        checkpoint_path = f"{CHECKPOINT_MOUNT_PATH}/{self.checkpoint_relpath}"
        self.model = load_model_from_checkpoint(checkpoint_path, device="cuda")
        self.model.eval()
        self.config = load_model_config(checkpoint_path)

    @modal.method()
    def metadata(self):
        return {
            "kernelLen": int(self.config["kernelLen"]),
            "strideLen": int(self.config["strideLen"]),
        }

    @modal.method()
    def predict(self, X, dayIdx):
        with torch.no_grad():
            inputs = torch.as_tensor(X, dtype=torch.float32, device="cuda")
            day_indices = torch.as_tensor(dayIdx, dtype=torch.int64, device="cuda")
            predictions = self.model.forward(inputs, day_indices)
        return predictions.detach().float().cpu().numpy()


@app.cls(
    gpu=DEFAULT_MODAL_GPU,
    timeout=900,
    scaledown_window=300,
)
class RemoteOPTScorer:
    model_name: str = modal.parameter(default="facebook/opt-6.7b")
    load_in_8bit: bool = modal.parameter(default=False)

    @modal.enter()
    def load(self):
        from neural_decoder.lm_utils import build_opt

        self.model, self.tokenizer = build_opt(
            model_name=self.model_name,
            cache_dir=None,
            device="auto",
            load_in_8bit=self.load_in_8bit,
        )

    @modal.method()
    def rescore(self, hypotheses, length_penalty=0.0):
        from neural_decoder.lm_utils import rescore_with_gpt2

        return rescore_with_gpt2(
            self.model,
            self.tokenizer,
            hypotheses,
            length_penalty=length_penalty,
        )

    @modal.method()
    def score_prompted(self, prompt, completions, length_penalty=0.0):
        from neural_decoder.lm_utils import score_prompted_completions

        return score_prompted_completions(
            self.model,
            self.tokenizer,
            prompt,
            completions,
            length_penalty=length_penalty,
        )
