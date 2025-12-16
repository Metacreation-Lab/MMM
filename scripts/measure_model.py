import json
from pathlib import Path
from typing import Any, Tuple
from symusic import Score
from miditok import MusicTokenizer, TokSequence
from miditok.constants import SCORE_LOADING_EXCEPTION

from mmm import generate, InferenceConfig
from transformers.trainer_utils import set_seed
from utils.baselines import baselines

from tqdm import tqdm
import torch
import numpy as np
import time


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Benchmark token generation speed.")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num", type=int, default=16, help="Number of passes")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--gen_tokens", type=int, default=64)
    parser.add_argument(
        "--len",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048],
        help="List of input lengths",
    )

    args = parser.parse_args()

    if args.model not in baselines:
        raise ValueError(f"Model {args.model} not found in baselines.")
    baseline = baselines[args.model]

    # Load model
    print(f"Loading model '{args.model}'...")
    model = baseline.create_model(args.ckpt)
    model.eval()

    print("\nModel Name ::", args.model)
    print("Model Checkpoint ::", args.ckpt)
    print("Generation Config ::", baseline.generation_config)
    print(f"Benchmarking {args.num} passes per input length...\n")

    times = {}

    total_iters = len(args.len) * args.num
    pbar = tqdm(total=total_iters, desc="Benchmarking", ncols=100)

    for input_len in args.len:

        # Generate random input sequence ONCE per length
        input_seq = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(1, input_len),
            dtype=torch.long,
            device=model.device,
        )

        total_time_per_token = []
        num_generated = args.gen_tokens

        for _ in range(args.num):

            start = time.perf_counter()

            output = model.generate(
                input_seq,
                max_new_tokens=num_generated,
                do_sample=False,
            )

            end = time.perf_counter()

            actual_generated = output.shape[1] - input_len
            if actual_generated != num_generated:
                print(
                    f"WARNING: Expected {num_generated}, got {actual_generated} tokens."
                )

            elapsed = end - start
            time_per_token = elapsed / actual_generated
            total_time_per_token.append(time_per_token)

            pbar.update(1)

        times[input_len] = np.array(total_time_per_token)

    pbar.close()
    print("\n===== Benchmark Summary =====")

    for input_len in args.len:
        arr = times[input_len]
        mean_s = arr.mean()
        var_s = arr.var()
        tps = 1.0 / mean_s

        print(
            f"Input {input_len:4d} tokens | "
            f"{tps:8.2f} tokens/sec | "
            f"mean time/token: {mean_s:.6f}s | var: {var_s:.6e}"
        )
