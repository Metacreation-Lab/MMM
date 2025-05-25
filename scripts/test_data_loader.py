#!/usr/bin/python3 python

"""Train the MMM model."""

from utils.baselines import baselines
from utils.classes import Baseline

import random
import numpy as np

from tqdm import tqdm

NUM_SAMPLES = 1000
VALID_TOKEN_BASE = [
    "Loop",
    "Start",
    "End",
    "Delta",
    "AC",
    "Position"
]

def valid_token(
    token,
    valid_tokens
) -> bool:
    for valid_token in valid_tokens:
        if valid_token in token:
            return True
    return False

def test_data_loader(
    baseline: Baseline,
    version: str = None,
    num_samples: int = NUM_SAMPLES
) -> None:

    if version:
        baseline.version = version
    subsets = baseline.create_data_subsets()
    eval_dataset = subsets["validation"]

    samples = [random.randint(0, len(eval_dataset) - 1) for _ in range(num_samples)]

    #print(baseline.tokenizer.base_tokenizer._tpb_to_time_array)
    success = 0
    for i, idx in tqdm(enumerate(samples), total=num_samples):
        sample = eval_dataset[idx]
        input_ids = sample["input_ids"]
        if input_ids is not None:
            success += 1
            input_ids = input_ids.numpy()
            tokens = [
                baseline.tokenizer[tok_id] for tok_id in input_ids
            ]
            print(f"TOKEN SEQUENCE {i} - {idx} :: START")
            for i,token in enumerate(tokens):
                #if valid_token(token, VALID_TOKEN_BASE):
                #    print(f"   {token}")
                print(f"     index - {i} - {token}")
            print(f"TOKEN SEQUENCE {i} - {idx} :: END")
    print(f"Success percentage: {success/num_samples}")

if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction

    # Parse arguments for training params / model size
    parser = ArgumentParser(description="Model training script")
    parser.add_argument("--model", type=str, default="MMM_mistral")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)

    args = vars(parser.parse_args())

    # Identify model to train and tweak its training configuration
    baseline = baselines[args.pop("model")]

    # Testing DatasetMMM
    test_data_loader(
        baseline,
        version=args.pop("version"),
        num_samples=args.pop("samples")
    )
