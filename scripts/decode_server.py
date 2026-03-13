#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_decoder.decode_server_utils import (
    DEFAULT_DECODE_SERVER_AUTHKEY,
    DEFAULT_DECODE_SERVER_HOST,
    DEFAULT_DECODE_SERVER_PORT,
    serve_decoder,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run a persistent local decode server that keeps WFST assets loaded in RAM."
    )
    parser.add_argument(
        "--lmDir",
        type=str,
        required=True,
        help="Directory containing TLG.fst, words.txt, and optional G.fst files.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_DECODE_SERVER_HOST,
        help="Host interface to bind for local decode requests.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_DECODE_SERVER_PORT,
        help="Port to bind for local decode requests.",
    )
    parser.add_argument(
        "--authkey",
        type=str,
        default=DEFAULT_DECODE_SERVER_AUTHKEY,
        help="Auth key required by decode clients.",
    )
    parser.add_argument(
        "--acousticScale",
        type=float,
        default=0.5,
        help="Acoustic scale for the WFST decoder.",
    )
    parser.add_argument(
        "--nbest",
        type=int,
        default=100,
        help="How many N-best entries to return.",
    )
    parser.add_argument(
        "--beam",
        type=float,
        default=18.0,
        help="Beam for the WFST decoder.",
    )
    args = parser.parse_args()

    serve_decoder(
        lm_dir=args.lmDir,
        host=args.host,
        port=args.port,
        authkey=args.authkey,
        acoustic_scale=args.acousticScale,
        nbest=args.nbest,
        beam=args.beam,
    )


if __name__ == "__main__":
    main()
