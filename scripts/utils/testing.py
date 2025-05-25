"""Training functions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch.cuda as cuda
from torch import Tensor, argmax, device, randint
from torch.backends.mps import is_available as mps_available
from torch.utils.data import DataLoader
print(" PyTorch imported")
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint, set_seed
print(" Transformers imported")
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch.utils.data import Dataset

    from .classes import Baseline


def select_device(use_cuda: bool = True, use_mps: bool = True) -> device:
    r"""
    Return the available ``torch.device`` with a priority on cuda.

    :param use_cuda: will run on nvidia GPU if available. (default: ``True``)
    :param use_mps: will run on MPS device if available. (default: ``True``)
    :return: ``cpu``, ``cuda:0`` or ``mps`` ``torch.device`` object.
    """
    if cuda.is_available() and use_cuda:
        return device("cuda:0")
    if mps_available() and use_mps:
        return device("mps")
    return device("cpu")


def print_cuda_memory() -> None:
    """Print the total and free memory of the cuda device."""
    free_mem, global_mem = cuda.mem_get_info(0)
    print(f"Total: {round(global_mem / 1024 ** 3, 1)} GB")
    print(f"Free: {round(free_mem / 1024 ** 3, 1)} GB")

def test_forward(
    baseline: Baseline,
) -> None:
    """
    Test forward passes with model.

    :param baseline: baseline to train.
    """
    print("Test Forward")
    # Create model
    model = baseline.create_model()
    print("Model Created")
    # Load data
    set_seed(baseline.seed)  # set before loading checkpoint
    subsets = baseline.create_data_subsets()
    print("Subset created")
    collator = baseline.create_data_collator()
    print("Collator created")
    print("Running Tests")
    for n_b in [1,3,8,16]:
        try:
            print(' n_b', n_b)
            n_tok = 4123
            test_input = randint(0, 10, (n_b, n_tok))
            test_attention_mask = (test_input != 0).int()
            print(" test_batch", {k: v.shape for k, v in test_batch.items()})
            test_batch = {
                'input_ids': test_input,
                'label':  test_input,
                'attention_mask': test_attention_mask
            }
            test_output = model(**test_batch)
            print(' test_output', {k: v.shape for k, v in test_output.items()})
        except Exception as e:
            print(e)


    dataloader = DataLoader(
        subsets["test"], batch_size=64, collate_fn=collator
    )

    for batch in tqdm(dataloader, desc="iterating over dataloader"):
        print("final", {k: v.shape for k, v in batch.items()})
        num_items_in_batch = (batch["labels"].ne(-100)).sum()
        print(f"Num items in batch: {num_items_in_batch}")
        output = model(**batch)
