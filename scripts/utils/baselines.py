"""Lists the Experiment baselines and training."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from datasets import Dataset, load_dataset
from miditok import TokenizerConfig
from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.pytorch_data import DataCollator
from miditok.utils import get_bars_ticks
from symusic import Score
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    GPT2Config,
    LongT5Config,
    MistralConfig,
)

from mmm.data_loading import DatasetMMM

from .classes import Baseline, DataConfig, TokenizationConfig
from .constants import (
    ACS_RANDOM_RATIO_RANGE,
    BARS_IDX_RANDOM_RATIO_RANGE,
    BATCH_SIZE_PER_DEVICE_TRAIN,
    BATCH_SIZE_PER_DEVICE_VALID,
    BF16,
    BF16_EVAL,
    DATA_AUGMENTATION_OFFSETS,
    DATALOADER_NUM_WORKERS,
    DATALOADER_PERSISTENT_WORKERS,
    DDP_BUCKET_CAP_MB,
    DDP_FIND_UNUSED_PARAMETERS,
    DEEPSPEED,
    DISABLE_TQDM,
    EMBEDDING_SIZE,
    EMBEDDING_SIZE_SMALL,
    EPSILON_CUTOFF,
    ETA_CUTOFF,
    EVAL_ACCUMULATION_STEPS,
    EVAL_STRATEGY,
    FEEDFORWARD_SIZE,
    FEEDFORWARD_SIZE_SMALL,
    FP16,
    FP16_EVAL,
    FULL_DETERMINISM,
    GRAD_ACC_STEPS,
    GRADIENT_CHECKPOINTING,
    GRADIENT_CLIP_NORM,
    HALF_PRECISION_BACKEND,
    HUB_PRIVATE_REPO,
    HUB_STRATEGY,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    LOAD_BEST_MODEL_AT_END,
    LOG_LEVEL,
    LOG_STEPS_INTVL,
    LOGGING_STRATEGY,
    LR_SCHEDULER,
    MAX_POSITION_EMBEDDINGS,
    MAX_POSITION_EMBEDDINGS_SMALL,
    MAX_SEQ_LEN,
    MIN_NUM_BARS_FILE_VALID,
    MIN_NUM_NOTES_FILE_VALID,
    NEFTUNE_NOISE_ALPHA,
    NUM_ATTENTION_HEADS,
    NUM_ATTENTION_HEADS_SMALL,
    NUM_BEAMS,
    NUM_INFERENCES_EVAL,
    NUM_KEY_VALUE_HEADS,
    NUM_KEY_VALUE_HEADS_SMALL,
    NUM_LAYERS,
    NUM_LAYERS_SEQ2SEQ_DECODER,
    NUM_LAYERS_SEQ2SEQ_ENCODER,
    NUM_LAYERS_SMALL,
    NUM_TRAIN_EPOCHS,
    PUSH_TO_HF_HUB,
    RATIO_BAR_INFILLING,
    RATIOS_RANGE_BAR_INFILLING_DURATION,
    REPETITION_PENALTY,
    REPORT_TO,
    SAVE_SAFETENSOR,
    SAVE_STEPS,
    SAVE_STRATEGY,
    SAVE_TOTAL_LIMIT,
    SEED,
    SLIDING_WINDOWS,
    SLIDING_WINDOWS_SMALL,
    TEMPERATURE_SAMPLING,
    TOKENIZER_PARAMS,
    TOP_K,
    TOP_P,
    TORCH_COMPILE,
    TORCH_COMPILE_BACKEND,
    TORCH_COMPILE_MODE,
    TRACKS_IDX_RANDOM_RATIO_RANGE,
    TRACKS_SELECTION_RANDOM_RATIO_RANGE,
    TRAINING_STEPS,
    USE_CUDA,
    USE_MPS,
    VALID_DELAY,
    VOCAB_SIZE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel


attn_implem = "flash_attention_2" if "flash_attn" in sys.modules else None
dtype = torch.bfloat16 if BF16 else torch.float16 if FP16 else torch.float32


def is_score_valid(
    score: Score | Path | bytes, min_num_bars: int, min_num_notes: int
) -> bool:
    """
    Check if a ``symusic.Score`` is valid, contains the minimum required number of bars.

    :param score: ``symusic.Score`` to inspect or path to a MIDI file.
    :param min_num_bars: minimum number of bars the score should contain.
    :param min_num_notes: minimum number of notes that score should contain.
    :return: boolean indicating if ``score`` is valid.
    """
    if isinstance(score, Path):
        try:
            score = Score(score)
        except SCORE_LOADING_EXCEPTION:
            return False
    elif isinstance(score, bytes):
        try:
            score = Score.from_midi(score)
        except SCORE_LOADING_EXCEPTION:
            return False

    return (
        score.start() >= 0
        and len(get_bars_ticks(score)) >= min_num_bars
        and score.note_num() > min_num_notes
    )


class MMM(Baseline):
    """MMM model baseline."""

    seq2seq: bool = False

    def create_dataset(self) -> Dataset:
        """
        Create a ``pytorch.utils.data.Dataset`` to use to train/test a model.

        :return the ``Dataset``.
        """
        """return load_dataset(
            str(self.dataset_path), self.data_config.subset_name, trust_remote_code=True
        )"""
        # Trick required here because of potential permission error caused by the
        # hugging face datasets library when extracting webdataset tar files written
        # without user write permission.
        while True:
            try:
                return load_dataset(
                    str(self.dataset_path),
                    "all-instruments-with-drums",
                    subsets=self.data_config.subsets_names,
                    trust_remote_code=True,
                )
            except PermissionError:
                path = os.getenv("SLURM_TMPDIR")
                if path is not None:
                    path = Path(path, ".hf_cache")
                else:
                    path = Path(os.getenv("HOME"), ".cache", "huggingface")
                path = str(path / "datasets")
                os.system(f"chmod -R 777 {path}")  # noqa:S605

    def create_dataset_from_parquet(self) -> dict:
        """
        Create datasets for train, validation, and test sets from Parquet files.

        :return: A dictionary containing the train, validation, and test datasets.
        """
        dataset_path = Path("../data/GigaMIDI")

        try:
            # Load the datasets using load_dataset
            return load_dataset(
                "parquet",
                data_files={
                    "train": str(dataset_path /
                                 "all-instruments-with-drums/train.parquet"),
                    "validation": str(dataset_path /
                                      "all-instruments-with-drums/validation.parquet"),
                    "test": str(dataset_path /
                                "all-instruments-with-drums/test.parquet"),
                },
            )
        except PermissionError:
            # Handle potential permission errors
            path = os.getenv("SLURM_TMPDIR")
            if path is not None:
                path = Path(path, ".hf_cache")
            else:
                path = Path(os.getenv("HOME"), ".cache", "huggingface")
            path = str(path / "datasets")
            os.system(f"chmod -R 777 {path}")  # noqa:S605

            # Retry loading the datasets after fixing permissions
            return self.create_dataset()

    def create_data_subsets(self) -> dict[str, DatasetMMM]:
        """
        Create the train/validation/test subsets to train the model.

        :return: data subsets.
        """
        dataset = self.create_dataset_from_parquet()
        return {
            subset_name: DatasetMMM(
                self.preprocess_dataset(subset),
                self.tokenizer,
                self.data_config.max_seq_len,
                TRACKS_SELECTION_RANDOM_RATIO_RANGE,
                self.data_config.data_augmentation_offsets,
                RATIO_BAR_INFILLING,
                RATIOS_RANGE_BAR_INFILLING_DURATION,
                ac_random_ratio_range=ACS_RANDOM_RATIO_RANGE,
                ac_tracks_random_ratio_range=TRACKS_IDX_RANDOM_RATIO_RANGE,
                ac_bars_random_ratio_range=BARS_IDX_RANDOM_RATIO_RANGE,
                seq2seq=self.seq2seq,
            )
            for subset_name, subset in dataset.items()
        }

    @staticmethod
    def preprocess_dataset(dataset: Dataset) -> Dataset:
        """
        Process the dataset after being loaded.

        Makes sure every entry can be loaded as a ``symusic.Score`` and that the score
        is valid.

        :param dataset: ``datasets.Dataset`` to process.
        """
        print(dataset[0])
        return dataset.filter(
            lambda ex: is_score_valid(
                ex["music"], MIN_NUM_BARS_FILE_VALID, MIN_NUM_NOTES_FILE_VALID
            )
        )

    def create_data_collator(self, pad_on_left: bool = False) -> DataCollator:
        """Create a data collator to use with a ``pytorch.utils.data.DataLoader``."""
        return DataCollator(
            self.pad_token_id,
            pad_on_left=pad_on_left,
            shift_labels=self.seq2seq,
        )

    def create_model(self, pretrained: str | None = None) -> PreTrainedModel:
        """
        Create the model of the baseline.

        :param pretrained: path of the model to load. If ``None`` is given, the model is
            created untrained. (default: ``None``)
        """
        kwargs = {"attn_implementation": attn_implem, "torch_dtype": dtype}
        auto_model_class = (
            AutoModelForSeq2SeqLM if self.seq2seq else AutoModelForCausalLM
        )
        if pretrained is not None:
            model = auto_model_class.from_pretrained(pretrained, **kwargs)
        else:
            model = auto_model_class.from_config(self.model_config, **kwargs)
        # model = BetterTransformer.transform(model, keep_original_model=False)
        model.generation_config = self.generation_config
        return model


# Create config objects
# https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
training_config_kwargs = {
    "output_dir": "",  # overridden by Baseline class
    "overwrite_output_dir": False,
    "do_train": True,
    "do_eval": True,
    "do_predict": False,
    "eval_strategy": EVAL_STRATEGY,
    "per_device_train_batch_size": BATCH_SIZE_PER_DEVICE_TRAIN,
    "per_device_eval_batch_size": BATCH_SIZE_PER_DEVICE_VALID,
    "gradient_accumulation_steps": GRAD_ACC_STEPS,
    "eval_accumulation_steps": EVAL_ACCUMULATION_STEPS,
    "eval_delay": VALID_DELAY,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "max_grad_norm": GRADIENT_CLIP_NORM,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "max_steps": TRAINING_STEPS,
    "lr_scheduler_type": LR_SCHEDULER,
    "warmup_ratio": WARMUP_RATIO,
    "log_level": LOG_LEVEL,
    "logging_strategy": LOGGING_STRATEGY,
    "logging_steps": LOG_STEPS_INTVL,
    "save_strategy": SAVE_STRATEGY,
    "save_steps": SAVE_STEPS,
    "save_total_limit": SAVE_TOTAL_LIMIT,
    "save_safetensors": SAVE_SAFETENSOR,
    "use_cpu": not (USE_CUDA or USE_MPS),
    "seed": SEED,
    "data_seed": SEED,
    "bf16": BF16,
    "bf16_full_eval": BF16_EVAL,
    "fp16": FP16,
    "fp16_full_eval": FP16_EVAL,
    "half_precision_backend": HALF_PRECISION_BACKEND,
    "local_rank": int(os.getenv("LOCAL_RANK", -1)),  # for DDP
    "dataloader_num_workers": DATALOADER_NUM_WORKERS,
    "disable_tqdm": DISABLE_TQDM,
    "load_best_model_at_end": LOAD_BEST_MODEL_AT_END,
    "label_smoothing_factor": LABEL_SMOOTHING,
    "optim": "adamw_torch",
    "report_to": REPORT_TO,
    "ddp_find_unused_parameters": DDP_FIND_UNUSED_PARAMETERS,
    "ddp_bucket_cap_mb": DDP_BUCKET_CAP_MB,
    "dataloader_persistent_workers": DATALOADER_PERSISTENT_WORKERS,
    "push_to_hub": PUSH_TO_HF_HUB,
    "deepspeed": DEEPSPEED,  # set with argparse
    "hub_strategy": HUB_STRATEGY,
    "hub_private_repo": HUB_PRIVATE_REPO,
    "gradient_checkpointing": GRADIENT_CHECKPOINTING,
    "full_determinism": FULL_DETERMINISM,
    "use_mps_device": USE_MPS,
    "torch_compile": TORCH_COMPILE,
    "torch_compile_backend": TORCH_COMPILE_BACKEND,
    "torch_compile_mode": TORCH_COMPILE_MODE,
    "neftune_noise_alpha": NEFTUNE_NOISE_ALPHA,
    "predict_with_generate": False,
}
data_config = DataConfig(
    ["no-drums", "all-instruments-with-drums"], DATA_AUGMENTATION_OFFSETS, MAX_SEQ_LEN
)
tok_config = TokenizationConfig(
    "MMM", TokenizerConfig(**deepcopy(TOKENIZER_PARAMS)), VOCAB_SIZE
)
mistral_config = MistralConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=EMBEDDING_SIZE,
    intermediate_size=FEEDFORWARD_SIZE,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    num_key_value_heads=NUM_KEY_VALUE_HEADS,
    max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    sliding_window=SLIDING_WINDOWS,
    attn_implementation=attn_implem,
    torch_dtype=dtype,
)

# Vocab size is the size of base vocabulary of the tokenizer
mistral_small_config = MistralConfig(
    hidden_size=EMBEDDING_SIZE_SMALL,
    intermediate_size=FEEDFORWARD_SIZE_SMALL,
    num_hidden_layers=NUM_LAYERS_SMALL,
    num_attention_heads=NUM_ATTENTION_HEADS_SMALL,
    num_key_value_heads=NUM_KEY_VALUE_HEADS_SMALL,
    max_position_embeddings=MAX_POSITION_EMBEDDINGS_SMALL,
    sliding_window=SLIDING_WINDOWS_SMALL,
    attn_implementation=attn_implem,
    torch_dtype=dtype,
)
gpt2_config = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=MAX_POSITION_EMBEDDINGS,
    n_embd=EMBEDDING_SIZE,
    n_layer=NUM_LAYERS,
    n_head=NUM_ATTENTION_HEADS,
    n_inner=FEEDFORWARD_SIZE,
    attn_implementation=attn_implem,
    torch_dtype=dtype,
)

t5_config = LongT5Config(
    vocab_size=VOCAB_SIZE,
    d_model=EMBEDDING_SIZE,
    d_kv=EMBEDDING_SIZE // NUM_ATTENTION_HEADS,
    d_ff=FEEDFORWARD_SIZE,
    num_layers=NUM_LAYERS_SEQ2SEQ_ENCODER,
    num_decoder_layers=NUM_LAYERS_SEQ2SEQ_DECODER,
    num_heads=NUM_ATTENTION_HEADS,
    local_radius=SLIDING_WINDOWS,
    decoder_start_token_id=0,  # padding token
    # attn_implementation=attn_implem,  # not implemented for T5Long
    torch_dtype=dtype,
)
generation_config = GenerationConfig(
    max_new_tokens=NUM_INFERENCES_EVAL,
    num_beams=NUM_BEAMS,
    do_sample=True,
    temperature=TEMPERATURE_SAMPLING,
    top_k=TOP_K,
    top_p=TOP_P,
    epsilon_cutoff=EPSILON_CUTOFF,
    eta_cutoff=ETA_CUTOFF,
    repetition_penalty=REPETITION_PENALTY,
    use_cache=True,
)

# exp -> Model size, baseline -> pretrained + finetune
mmm_mistral = MMM(
    "MMM_mistral",
    "GigaMIDI",
    SEED,
    deepcopy(tok_config),
    deepcopy(mistral_config),
    deepcopy(training_config_kwargs),
    deepcopy(data_config),
    deepcopy(generation_config),
)

mmm_gpt2 = MMM(
    "MMM_gpt2",
    "GigaMIDI",
    SEED,
    deepcopy(tok_config),
    deepcopy(gpt2_config),
    deepcopy(training_config_kwargs),
    deepcopy(data_config),
    deepcopy(generation_config),
)

mmm_small_gpt2 = MMM(
    "MMM_11M_mistral",
    "GigaMIDI",
    SEED,
    deepcopy(tok_config),
    deepcopy(mistral_small_config),
    deepcopy(training_config_kwargs),
    deepcopy(data_config),
    deepcopy(generation_config),
)

mmm_t5 = MMM(
    "MMM_t5",
    "GigaMIDI",
    SEED,
    deepcopy(tok_config),
    deepcopy(t5_config),
    deepcopy(training_config_kwargs),
    deepcopy(data_config),
    deepcopy(generation_config),
)
mmm_t5.seq2seq = True

baselines = {baseline.name: baseline for baseline in [mmm_mistral, mmm_t5, mmm_gpt2]}
