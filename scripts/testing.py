"""Training functions."""

from __future__ import annotations
from utils.classes import Baseline
from torch.utils.data import DataLoader
from tqdm import tqdm

def test_dataloader(
    baseline: Baseline,
    workers: int
) -> None:
    """
    Test forward passes with model.

    :param baseline: baseline to train.
    """
    subsets = baseline.create_data_subsets()
    collator = baseline.create_data_collator(pad_on_left=False)

    dataloader = DataLoader(
        subsets["test"], batch_size=32, collate_fn=collator, num_workers=workers
    )

    total_batches = 0
    total_expected = 0
    total_actual = 0


    with tqdm(dataloader, desc="iterating over dataloader", dynamic_ncols=True) as pbar:
        for batch in pbar:
            total_batches += 1

            # Expected batch size (from dataloader batch_size)
            expected_bs = dataloader.batch_size

            # Get actual batch size from the first tensor (they should all have same batch dim)
            first_key = next(iter(batch))
            actual_bs = batch[first_key].shape[0]

            total_expected += expected_bs
            total_actual += actual_bs

            # Compute average lost samples
            avg_lost = 100*(total_expected - total_actual) / total_expected

            # Update tqdm description
            pbar.set_description(f"avg lost {avg_lost:.5f}%")

if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction
    from utils.baselines import baselines

    # Parse arguments for training params / model size
    parser = ArgumentParser(description="Test for dataloader")
    parser.add_argument("--model", type=str, default="MMM_gpt2")
    parser.add_argument("--workers", type=int, default=4)


    args = parser.parse_args()

    baseline = baselines[args.model]
    test_dataloader(baseline, args.workers)