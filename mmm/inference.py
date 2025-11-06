"""Inference method for the MMM model."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
from miditok import MMM, TokSequence
from symusic import Score
from torch import LongTensor
from transformers import LogitsProcessorList

from .logits_processor import TrackLogitsProcessor, InfillLogitsProcessor

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from .config import InferenceConfig


def generate(
    model: object,
    tokenizer: MMM,
    inference_config: InferenceConfig,
    score_or_path: Score | Path | str,
    generate_kwargs: Mapping | None = None,
) -> Score:
    """
    Use the model to generate new music content.

    The method allows to infill specific bars or generate new tracks.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param inference_config: InferenceConfig
    :param score_or_path: ``symusic.Score`` or path of the music file to infill.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    score = (
        Score(score_or_path) if not isinstance(score_or_path, Score) else score_or_path
    )

    logits_processor = StopLogitsProcessor(
        tokenizer.vocab["Bar_None"], tokenizer.vocab["FillBar_End"], tokenizer
    )

    # Infill bars
    if inference_config.infilling:
        for track, bars in inference_config.bars_to_generate.items():
            score = generate_infilling(
                model, tokenizer, track, bars, score, generate_kwargs
            )

    # Generate new tracks
    if inference_config.autoregressive:
        for track in inference_config.new_tracks:
            score = generate_new_track(
                model, tokenizer, track, score, generate_kwargs
            )

    return score


def generate_new_track(
    model: object,
    tokenizer: MMM,
    track: tuple[int, list[str]],
    score: Score,
    generate_kwargs: Mapping | None = None,
    num_context: int = 8,
    max_len: int = 2048
) -> Score:
    """
    Generate a new track for a Score, using only the last `num_context` bars of
    existing tracks as context, but keeping the full, unclipped score in the result.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param track: (program, [attribute controls])
    :param score: symusic.Score
    :param generate_kwargs: args for model.generate
    :param num_context: number of bars to keep from each track as context
    :param max_len: maximum model input length (attention window)
    :return: new symusic.Score with generated track appended
    """
    import warnings
    import numpy as np
    from torch import LongTensor

    if not generate_kwargs:
        generate_kwargs = {}
    else:
        generate_kwargs["generation_config"].eos_token_id = tokenizer.vocab[
            "Track_End"
        ]

    # --- Encode all tracks (preserve original copy)
    all_tracks_seq = tokenizer.encode(score, concatenate_track_sequences=False)
    full_tracks_seq = [seq.copy() for seq in all_tracks_seq]

    # --- Create truncated context copy (bar-aware)
    truncated_tracks = []
    bar_token_id = tokenizer.vocab["Bar_None"]
    max_length = 0
    for seq in all_tracks_seq:
        bar_idxs = np.where(np.array(seq.ids) == bar_token_id)[0]
        if bars_idxs > max_length:
            max_length = bars_idxs
        if len(bar_idxs) > num_context:
            keep_start = bar_idxs[-num_context]
            seq.ids = seq.ids[keep_start:]
            seq.tokens = seq.tokens[keep_start:]
        truncated_tracks.append(seq)
    num_bars_to_generate = min(max_length, num_context)

    # --- Concatenate reduced tracks to form model input
    input_seq = sum(truncated_tracks, start=tokenizer.empty_toksequence())

    # --- Add new track header
    input_seq.ids.append(tokenizer.vocab["Track_Start"])
    input_seq.tokens.append("Track_Start")
    input_seq.ids.append(tokenizer.vocab[f"Program_{track[0]}"])
    input_seq.tokens.append(f"Program_{track[0]}")

    # --- Attribute controls
    num_attr = len(track[1])
    for control in track[1]:
        input_seq.ids.append(tokenizer.vocab[control])
        input_seq.tokens.append(control)

    # --- Truncate if too long for model
    if len(input_seq.ids) > max_len:
        input_seq.ids = input_seq.ids[-max_len:]
        input_seq.tokens = input_seq.tokens[-max_len:]

    # --- Generate

    logits_processor = TrackLogitsProcessor(
            tokenizer.vocab["Track_Start"],
            tokenizer.vocab["Track_End"],
            tokenizer.vocab["Bar_None"],
            tokenizer.vocab["EOS_None"],
            num_bars_to_generate
        )
    logit_processor_list = LogitsProcessorList()
    logit_processor_list.append(logits_processor)

    output_ids = model.generate(
        LongTensor([input_seq.ids]), 
        logits_processor=logit_processor_list
        **generate_kwargs
    )
    output_seq = TokSequence(ids=output_ids[0].tolist(), are_ids_encoded=True)

    # --- Clean up (remove controls)
    input_len = len(input_seq) - num_attr
    output_seq = output_seq[:2] + output_seq[2 + num_attr:]

    tokenizer.decode_token_ids(output_seq)
    output_seq.tokens = tokenizer._ids_to_tokens(output_seq.ids)

    # --- Ensure <TRACK_END> closure
    if output_seq.tokens[-1] != "Track_End":
        warnings.warn(
            "Track generation did not terminate with <TRACK_END>; appending manually.",
            stacklevel=2,
        )
        output_seq.ids.append(tokenizer.vocab["Track_End"])
        output_seq.tokens.append("Track_End")

    # --- Append generated track to the full, uncut score
    full_tracks_seq.append(output_seq)

    # --- Reconstruct complete score
    return tokenizer.base_tokenizer._tokens_to_score(full_tracks_seq)

def generate_infilling(
    model: object,
    tokenizer: MMM,
    track_idx: int,
    infill: list[tuple[int, int, list[str]]],
    score: Score,
    generate_kwargs: Mapping | None = None,
    num_context: int = 8,
    max_len: int = 2048
) -> Score:
    """
    Generate a new portion of a ``symusic.Score``.

    The portion to infill will be generated with the model and added to the score
    inplace for the selected tracks. Notes originally present in the portion to
    infill will be removed.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param score: ``symusic.Score`` to generate a new track from.
    :param inference_config: InferenceConfig
    :param logits_processor: ``transformers.LogitsProcessor`` used to stop
        generation when the right number of bars is generated.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    if not generate_kwargs:
        generate_kwargs = {}
    else:
        generate_kwargs["generation_config"].eos_token_id = tokenizer.vocab[
            "FillBar_End"
        ]

    for start_bar_idx, end_bar_idx, attribute_controls in infill:
        # token_start_idx and token_end_idx are the indices of start
        # and end of infilling, when the toksequence is NOT BPE encoded

        tokens = tokenizer.encode(score, concatenate_track_sequences=False)

        conditioning_dict = {}

        toksequence_to_infill = TokSequence(are_ids_encoded=False)

        bars_ticks = tokens[track_idx]._ticks_bars
        num_bars = len(bars_ticks)
        
        bar_tick_start = bars_ticks[start_bar_idx]
        bar_tick_end = bars_ticks[end_bar_idx]

        times = np.array([event.time for event in tokens[track_idx].events])

        token_idx_start = np.nonzero(times >= bar_tick_start)[0]
        token_idx_start = token_idx_start[0]

        token_idx_end = np.nonzero(times >= bar_tick_end)[0]
        token_idx_end = token_idx_end[0]

        context_start_bar = max(start_bar_idx - num_context,0)
        context_end_bar = min(end_bar_idx + num_context, num_bars)

        # Context
        context_token_start_idx = np.nonzero(
            times >= bars_ticks[context_start_bar]
        )[0][0]
        context_token_end_idx = np.nonzero(
            times >= bars_ticks[context_end_bar]
        )[0][0]

        conditioning_dict[track_idx] = (context_token_start_idx, context_token_end_idx)

        # Decode BPE tokens: this is necessary to put <INFILL_BAR> tokens
        # at the right place
        tokenizer.decode_token_ids(tokens[track_idx])

        seq_before = (
            tokens[track_idx][:2]
            + tokens[track_idx][context_token_start_idx:token_idx_start]
        )
        for _ in range(end_bar_idx - start_bar_idx):
            seq_before.ids.append(tokenizer.vocab["Infill_Bar"])
            seq_before.tokens.append("Infill_Bar")
        seq_after = tokens[track_idx][token_idx_end:context_token_end_idx]
        toksequence_to_infill += seq_before + seq_after
        toksequence_to_infill.ids.append(tokenizer.vocab["Track_End"])
        toksequence_to_infill.tokens.append("Track_End")

        # Encode into BPE tokens
        tokenizer.encode_token_ids(toksequence_to_infill)

        input_seq = TokSequence(are_ids_encoded=True)
        for i in range(len(tokens)):
            if i == track_idx:
                input_seq += toksequence_to_infill
                continue
            times = np.array([event.time for event in tokens[i].events])
            context_token_start_idx = np.nonzero(
                times >= bars_ticks[context_start_bar]
            )[0][0]
            context_token_end_idx = np.nonzero(
                times >= bars_ticks[context_end_bar]
            )[0][0]
            conditioning_dict[i] = (context_token_start_idx, context_token_end_idx)
            input_seq += (
                tokens[i][:2]
                + tokens[i][context_token_start_idx:context_token_end_idx]
                + tokens[i][-1:]
            )

        input_seq.ids.append(tokenizer.vocab["FillBar_Start"])
        input_seq.tokens.append("FillBar_Start")

        for control in attribute_controls:
            input_seq.ids.append(tokenizer.vocab[control])
            input_seq.tokens.append(control)

        if len(input_seq.ids) > max_len:
            input_seq.ids = input_seq.ids[-max_len:]
            input_seq.tokens = input_seq.tokens[-max_len:]

        logits_processor = InfillLogitsProcessor(
            tokenizer.vocab["FillBar_Start"],
            tokenizer.vocab["FillBar_End"],
            tokenizer.vocab["Bar_None"],
            tokenizer.vocab["EOS_None"],
            start_bar_idx - end_bar_idx
        )
        logit_processor_list = LogitsProcessorList()
        logit_processor_list.append(logits_processor)

        output_ids = model.generate(
            LongTensor([input_seq.ids]),
            logits_processor=logit_processor_list
            **generate_kwargs,
        )[0].numpy()

        fill_start_idx = np.where(output_ids == tokenizer.vocab["FillBar_Start"])[0][0]

        # Here we isolate the generated tokens doing some filtering. In particular,
        # the model may generate some tokens before the first Bar_None token
        generated_tokens = TokSequence(are_ids_encoded=True)
        generated_tokens.ids = output_ids[
            fill_start_idx + len(attribute_controls) + 1 : -1
        ].tolist()
        # decode_token_ids doesn't support numpy arrays for ids list
        tokenizer.decode_token_ids(generated_tokens)
        first_bar_none_token_idxs = np.where(
            np.array(generated_tokens.ids) == tokenizer.vocab["Bar_None"]
        )[0][0]

        generated_tokens.ids = generated_tokens.ids[bar_none_token_idxs[0]:]

        tokenizer.decode_token_ids(tokens[track_idx])
        tokens[track_idx].ids[token_idx_start:token_idx_end] = generated_tokens.ids
        tokens[track_idx].tokens = tokenizer._ids_to_tokens(tokens[track_idx].ids)

        tokenizer.base_tokenizer._tokens_to_score(tokens)

    return score
