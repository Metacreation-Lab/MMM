"""Main classes implementations."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import miditok

from .utils import path_data_directory_local_fs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.utils.data import Dataset
    from transformers import (
        DataCollator,
        GenerationConfig,
        PretrainedConfig,
        PreTrainedModel,
    )


@dataclass
class DataConfig:
    """
    Configuration of the data.

    :param subsets_names: name of the subsets of the dataset to use.
    :param data_augmentation_offsets: offsets of pitch, velocity and note duration to
        use to augment the original dataset.
    :param max_seq_len: maximum length that a token sequence should have to be used to
        train the model.
    """

    subsets_names: Sequence[str]
    data_augmentation_offsets: tuple[int, int, int]
    max_seq_len: int


@dataclass
class TokenizationConfig:
    """
    Tokenizer configuration including its class and vocabulary size.

    :param tokenization: name of the tokenizer's class.
    :param tokenizer_config: tokenizer configuration.
    :param vocab_size: size of the tokenizer's vocabulary.
    """

    tokenization: str
    tokenizer_config: miditok.TokenizerConfig
    vocab_size: int = None


@dataclass
class Baseline(ABC):
    """
    Base baseline class.

    :param dataset: name of the dataset to use.
    :param seed: seed to use before performing random operation.
    :param tokenization_config: configuration of the tokenizer.
    :param model_config: configuration of the model.
    :param training_config_kwargs: keyword arguments used to create the
        ``transformers.Trainer`` configuration.
    :param data_config: configuration of the data, including valid/test split and token
        sequence lengths.
    :param generation_config: ``transformers.GenerationConfig`` to use to generate
        tokens when evaluating/testing the model.
    """

    name: str
    dataset: str
    seed: int
    tokenization_config: TokenizationConfig
    model_config: PretrainedConfig
    training_config_kwargs: dict[str, Any]
    data_config: DataConfig
    generation_config: GenerationConfig = None

    def __post_init__(self) -> None:
        """Post init method creating the tokenizer and tweaking configs."""
        # Create tokenizer
        self.tokenizer = self.create_tokenizer()

        # Tweak configs parameters
        self.training_config_kwargs["output_dir"] = str(self.run_path)
        self.training_config_kwargs["logging_dir"] = str(self.run_path / "logs")
        self.model_config.pad_token_id = self.pad_token_id
        self.model_config.bos_token_id = self.bos_token_id
        self.model_config.eos_token_id = self.eos_token_id
        self.generation_config.pad_token_id = self.pad_token_id

    def create_tokenizer(self) -> miditok.MusicTokenizer:
        """
        Create the tokenizer of the baseline.

        :return: tokenizer of the baseline.
        """
        try:
            tokenizer = getattr(miditok, self.tokenization)(params=self.tokenizer_path)
        except FileNotFoundError:
            tokenizer = getattr(miditok, self.tokenization)(
                self.tokenization_config.tokenizer_config
            )
        return tokenizer

    @property
    def tokenization(self) -> str:
        """
        Return the name of the tokenizer class of the baseline.

        :return: name of the tokenization of the baseline.
        """
        return self.tokenization_config.tokenization

    @property
    def tokenizer_path(self) -> Path:
        """
        Return the path of the tokenizer's configuration file.

        :return: path of the tokenizer's configuration file.
        """
        return self.run_path.parent / "tokenizer_with_acs.json"

    @property
    def dataset_path(self) -> Path:
        """
        Return the path of the dataset.

        :return: path of the dataset.
        """
        return path_data_directory_local_fs() / self.dataset

    @property
    def run_path(self) -> Path:
        """
        Return the path where the model, tokenizer and other results are saved.

        :return: path where files of the baseline runs are saved.
        """
        return Path("runs", self.name)

    def __return_special_token(self, tok: str) -> int:
        return self.tokenizer[tok]

    @property
    def pad_token_id(self) -> int:
        """
        Return the padding token id of the tokenizer of the baseline.

        :return: padding token id of the tokenizer of the baseline.
        """
        return self.tokenizer["PAD_None"]

    @property
    def bos_token_id(self) -> int:
        """
        Return the "BOS" token id of the tokenizer of the baseline.

        :return: "BOS" token id of the tokenizer of the baseline.
        """
        return self.tokenizer["BOS_None"]

    @property
    def eos_token_id(self) -> int:
        """
        Return the "EOS" token id of the tokenizer of the baseline.

        :return: "EOS" token id of the tokenizer of the baseline.
        """
        return self.tokenizer["EOS_None"]

    @property
    def special_tokens(self) -> list[str]:
        """
        Return the special tokens of the tokenizer of the baseline.

        :return: special tokens of the tokenizer of the baseline.
        """
        return self.tokenizer.special_tokens

    @property
    def special_tokens_ids(self) -> list[int]:
        """
        Return the ids of the special tokens of the tokenizer of the baseline.

        :return: ids of the special tokens of the tokenizer of the baseline.
        """
        return self.tokenizer.special_tokens_ids

    def create_model(self, pretrained: str | None = None) -> PreTrainedModel:
        """
        Create the model of the baseline.

        :param pretrained: path of the model to load. If ``None`` is given, the model is
            created untrained.
        """
        raise NotImplementedError

    def create_dataset(self, *args, **kwargs) -> Dataset:  # noqa:ANN002
        """
        Create a ``pytorch.utils.data.Dataset`` to use to train/test a model.

        :return: the whole dataset.
        """
        raise NotImplementedError

    def create_data_collator(self) -> DataCollator:
        """Create a data collator to use with a ``pytorch.utils.data.DataLoader``."""
        raise NotImplementedError

    def create_data_subsets(self, *args, **kwargs) -> dict[str, Dataset]:  # noqa:ANN002
        """
        Create the ``pytorch.utils.data.Dataset`` train/valid/test subsets.

        :return: the train/valid/test subsets as ``pytorch.utils.data.Dataset`` objects.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Return the string representation of the baseline.

        :return: string representation of the baseline.
        """
        return self.name
