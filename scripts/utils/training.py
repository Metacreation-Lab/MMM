"""Training functions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch.cuda as cuda
from torch import Tensor, argmax, device, load
from torch.backends.mps import is_available as mps_available
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint, set_seed

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


def is_training_done(run_path: Path) -> bool:
    """
    Tells if a model has already been trained in the run_path directory,.

    :param run_path: model training directory
    :return: if model has already been fully trained
    """
    return run_path.exists() and (run_path / "train_results.json").is_file()


def is_testing_done(run_path: Path) -> bool:
    """
    Tells if a model has already been trained in the run_path directory,.

    :param run_path: model training directory
    :return: if model has already been fully trained
    """
    return run_path.exists() and (run_path / "test_results.json").is_file()


def get_checkpoints_paths(path: Path) -> list[Path]:
    """
    Return the paths of all the model checkpoints in a directory.

    :param path: path to the directory to list the checkpoints.
    :return: paths of all the model checkpoints in ``path``.
    """
    _re_checkpoint = re.compile(r"^" + "checkpoint" + r"-(\d+)$")
    return [
        child_path
        for child_path in path.iterdir()
        if child_path.is_dir()
        and _re_checkpoint.search(str(child_path.name)) is not None
    ]


def get_first_checkpoint(path: Path) -> Path:
    """
    Return the path of the first checkpoint in a directory.

    :param path: path to the directory to get the first checkpoint.
    :return: path of the first checkpoint in ``path``
    """
    checkpoints = get_checkpoints_paths(path)
    return min(checkpoints, key=lambda x: int(x.name.split("-")[1]))


def preprocess_logits(logits: Tensor, _: Tensor = None) -> Tensor:
    """
    Preprocess logits before accumulating them during evaluation.

    This allows to significantly reduce the memory usage and make the training
    tractable in cases where it is not possible to store all the logits in memory.

    :param logits: logits produced a model.
    :param _: unused here, placeholder for expected tokens.
    """
    return argmax(logits, dim=-1)  # long dtype


def whole_training_process(
    baseline: Baseline,
    compute_metrics: Callable | None = None,
    resume_from_last_checkpoint: bool = True,
    do_test: bool = True,
    pad_on_left: bool = False
) -> None:
    """
    Complete training of a model, including testing it when training is finished.

    :param baseline: baseline to train.
    :param compute_metrics: method to compute the metrics to test the model.
    :param resume_from_last_checkpoint: whether to resume training the last checkpoint
        if available. (default: ``True``)
    :param do_test: whether to test the model after training it. (default: ``True``)
    """
    # Check training / testing are not already done
    if is_training_done(baseline.run_path) and (
        not do_test or (do_test and is_testing_done(baseline.run_path))
    ):
        return

    # Create model
    model = baseline.create_model()
    # n = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load data
    set_seed(baseline.seed)  # set before loading checkpoint
    subsets = baseline.create_data_subsets()
    collator = baseline.create_data_collator(pad_on_left=pad_on_left)
    """from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        subsets["train"], batch_size=30, collate_fn=collator, num_workers=8
    )
    for x in tqdm(dataloader, desc="iterating over dataloader"):
        t = 0"""

    # Train model if not already done
    # We use the Seq2SeqTrainer anyway as it subclasses Trainer and handles
    # "predict with gen".
    if resume_from_last_checkpoint and baseline.run_path.exists():
        baseline.training_config_kwargs["resume_from_checkpoint"] = get_last_checkpoint(
            str(baseline.run_path)
        )

    training_config = Seq2SeqTrainingArguments(**baseline.training_config_kwargs)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_config,
        data_collator=collator,
        train_dataset=subsets["train"],
        eval_dataset=subsets["validation"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
    )
    if not is_training_done(baseline.run_path):
        print("Training arguments:")
        print(" Seed", trainer.args.seed)
        print(" Workers", trainer.args.dataloader_num_workers)

        train_model(trainer)
    elif do_test and not is_testing_done(baseline.run_path):
        model = model.from_pretrained(baseline.run_path, device_map="auto")
        trainer.model = model
        test_model(trainer, dataset_test=subsets["test"])


def train_model(trainer: Trainer) -> None:
    r"""
    Train a model and save the metrics.

    :param trainer: initialized Trainer
    """
    train_result = trainer.train(
        resume_from_checkpoint=trainer.args.resume_from_checkpoint
    )
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


def test_model(trainer: Trainer, dataset_test: Dataset = None) -> None:
    r"""
    Test a model and save the metrics.

    :param trainer: initialized Trainer
    :param dataset_test: dataset for test / inference data.
    """
    test_results = trainer.predict(dataset_test)
    trainer.log_metrics("test", test_results.metrics)
    trainer.save_metrics("test", test_results.metrics)
