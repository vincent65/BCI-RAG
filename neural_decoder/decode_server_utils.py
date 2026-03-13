import argparse
from multiprocessing.connection import Client, Listener
from pathlib import Path

import numpy as np

from neural_decoder.lm_utils import build_lm_decoder, lm_decode


DEFAULT_DECODE_SERVER_HOST = "127.0.0.1"
DEFAULT_DECODE_SERVER_PORT = 50555
DEFAULT_DECODE_SERVER_AUTHKEY = "speechbci-decode"


def validate_lm_dir(path_str):
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Language-model directory not found: {path}")
    missing = [name for name in ["TLG.fst", "words.txt"] if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Language-model directory is missing required files: {', '.join(missing)}. Checked {path}"
        )
    return path


def add_decode_server_args(parser):
    parser.add_argument(
        "--decodeServerHost",
        type=str,
        default=DEFAULT_DECODE_SERVER_HOST,
        help="Hostname for the persistent local decode server.",
    )
    parser.add_argument(
        "--decodeServerPort",
        type=int,
        default=DEFAULT_DECODE_SERVER_PORT,
        help="Port for the persistent local decode server.",
    )
    parser.add_argument(
        "--decodeServerAuthkey",
        type=str,
        default=DEFAULT_DECODE_SERVER_AUTHKEY,
        help="Auth key for the persistent local decode server connection.",
    )
    return parser


class DecodeServerClient:
    def __init__(self, host=DEFAULT_DECODE_SERVER_HOST, port=DEFAULT_DECODE_SERVER_PORT, authkey=DEFAULT_DECODE_SERVER_AUTHKEY):
        self.host = host
        self.port = int(port)
        self.authkey = authkey.encode("utf-8")
        self._conn = Client((self.host, self.port), authkey=self.authkey)

    def health(self):
        self._conn.send({"command": "health"})
        return self._conn.recv()

    def decode_nbest(self, logits, blank_penalty=0.0, rescore=True):
        logits = np.asarray(logits, dtype=np.float32)
        self._conn.send(
            {
                "command": "decode_nbest",
                "logits": logits,
                "blank_penalty": float(blank_penalty),
                "rescore": bool(rescore),
            }
        )
        return self._conn.recv()

    def shutdown(self):
        self._conn.send({"command": "shutdown"})
        return self._conn.recv()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def serve_decoder(
    lm_dir,
    host=DEFAULT_DECODE_SERVER_HOST,
    port=DEFAULT_DECODE_SERVER_PORT,
    authkey=DEFAULT_DECODE_SERVER_AUTHKEY,
    acoustic_scale=0.5,
    nbest=100,
    beam=18,
):
    lm_dir = validate_lm_dir(lm_dir)
    decoder = build_lm_decoder(
        str(lm_dir),
        acoustic_scale=acoustic_scale,
        nbest=nbest,
        beam=beam,
    )
    listener = Listener((host, int(port)), authkey=authkey.encode("utf-8"))
    print(
        f"Decode server ready on {host}:{port} using lmDir={lm_dir}",
        flush=True,
    )
    try:
        while True:
            conn = listener.accept()
            try:
                while True:
                    request = conn.recv()
                    command = request.get("command")
                    if command == "health":
                        conn.send(
                            {
                                "status": "ok",
                                "lm_dir": str(lm_dir),
                                "host": host,
                                "port": int(port),
                            }
                        )
                    elif command == "decode_nbest":
                        logits = np.asarray(request["logits"], dtype=np.float32)
                        result = lm_decode(
                            decoder,
                            logits,
                            blank_penalty=float(request.get("blank_penalty", 0.0)),
                            return_nbest=True,
                            rescore=bool(request.get("rescore", True)),
                        )
                        conn.send(result)
                    elif command == "shutdown":
                        conn.send({"status": "shutting_down"})
                        return
                    else:
                        raise ValueError(f"Unknown decode server command: {command}")
            except EOFError:
                pass
            finally:
                conn.close()
    finally:
        listener.close()
