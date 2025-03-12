#!/usr/bin/python3 python

"""Script training the tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from miditok import MusicTokenizer, TokSequence
from miditok.constants import SCORE_LOADING_EXCEPTION
from symusic import Score

if TYPE_CHECKING:
    from datasets import Dataset


class TokTrainingIterator:
    r"""
    An iterable class to be used when training a tokenizer.

    It loads music files (MIDI, abc) and tokenize them on the fly, to be used with the
    Hugging Face tokenizers library to build a vocabulary with BPE, Unigram or WordPiece
    models.

    :param tokenizer: tokenizer to use for training.
    :param dataset: hugging face dataset to iterate from.
    """

    def __init__(
        self,
        tokenizer: MusicTokenizer,
        dataset: Dataset,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.__iter_count = 0

    def tokenize_sample(self, idx: int) -> list[str]:
        """
        Load a music file and convert it to its byte representation.

        :param idx: index of the data sample to load/tokenize.
        :return: the byte representation of the file.
        """
        # Load and tokenize file
        try:
            score = Score.from_midi(self.dataset[idx]["music"])
        except SCORE_LOADING_EXCEPTION:
            return []

        # Preprocess first to already have the appropriate tracks idx in case of deletes
        score = self.tokenizer.preprocess_score(score)

        # Tokenize the file
        # Need to specify `encode_ids=False` as it might be already pretrained
        # For MMM, we make sure to have sequences separated per track
        kwargs = {}
        # can't use isinstance because of circular import
        if type(self.tokenizer).__name__ == "MMM":
            kwargs["concatenate_track_sequences"] = False
        tokseq = self.tokenizer(
            score,
            encode_ids=False,
            no_preprocess_score=True,
            **kwargs,
        )

        # Split ids if requested
        if self.tokenizer.config.encode_ids_split in ["bar", "beat"]:
            if isinstance(tokseq, TokSequence):
                tokseq = [tokseq]

            new_seqs = []
            for seq in tokseq:
                if self.tokenizer.config.encode_ids_split == "bar":
                    new_seqs += seq.split_per_bars()
                else:
                    new_seqs += seq.split_per_beats()
            tokseq = [seq for seq in new_seqs if len(seq) > 0]

        # Convert ids to bytes for training
        if isinstance(tokseq, TokSequence):
            token_ids = tokseq.ids
        else:
            token_ids = [seq.ids for seq in tokseq]
        bytes_ = self.tokenizer._ids_to_bytes(token_ids, as_one_str=True)
        if isinstance(bytes_, str):
            bytes_ = [bytes_]

        return bytes_

    def __len__(self) -> int:
        """
        Return the number of files in the training corpus.

        :return: number of files in the training corpus.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list[str]:
        """
        Convert the ``idx``th file to its byte representation.

        :param idx: idx of the file to convert.
        :return: byte representation of the file.
        """
        return self.tokenize_sample(idx)

    def __iter__(self) -> TokTrainingIterator:
        return self

    def __next__(self) -> list[str]:
        if self.__iter_count >= len(self):
            self.__iter_count = 0
            raise StopIteration

        self.__iter_count += 1
        return self[self.__iter_count - 1]

    def __str__(self) -> str:
        """
        Return the ``str`` representation of the iterator.

        :return: string description.
        """
        return f"{self.tokenizer} - {len(self)} files"


if __name__ == "__main__":
    from transformers.trainer_utils import set_seed
    from utils.baselines import mmm_mistral
    from utils.constants import TRAINING_TOKENIZER_MAX_NUM_FILES

    set_seed(mmm_mistral.seed)

    # Train the tokenizer
    dataset_ = mmm_mistral.create_dataset_from_parquet()["train"]
    dataset_.shuffle()
    dataset_ = mmm_mistral.preprocess_dataset(dataset_).select(
        list(range(TRAINING_TOKENIZER_MAX_NUM_FILES))
    )
    iterator = TokTrainingIterator(mmm_mistral.tokenizer, dataset_)
    mmm_mistral.tokenizer.train(
        vocab_size=mmm_mistral.tokenization_config.vocab_size,
        iterator=iterator,
    )
    mmm_mistral.tokenizer.save(mmm_mistral.tokenizer_path)
