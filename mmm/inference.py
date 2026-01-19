"""Inference method for the MMM model."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING
from dataclasses import replace
import os

DEBUG = int(os.environ.get("MMM_DEBUG", 0)) == 1
if DEBUG:
    print("MMM debug active")

import numpy as np
from miditok import MMM, TokSequence
from symusic import Score
from torch import LongTensor
from transformers import LogitsProcessorList

from .logits_processor import (
    TrackLogitsProcessor, 
    InfillLogitsProcessor
)
from .utils import InferenceTimer, PerfCounterTimer, ProcessTimeTimer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from .config import InferenceConfig

def pretty_print_tokens(tokens):
    indent_track = "  "
    indent_bar = "    "
    indent_evt = "      "

    track_idx = -1
    bar_idx = -1

    def print_bar_header(label):
        print(f"{indent_bar}{label}")

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # -------- Track handling --------
        if tok == "Track_Start":
            track_idx += 1
            bar_idx = -1
            infill_bar_count = 0
            print(f"\nTrack {track_idx}:")
            i += 1
            continue

        if tok == "Track_End":
            print(f"{indent_track}<End Track>")
            i += 1
            continue

        # -------- Bar handling --------
        if tok == "Bar_None":
            bar_idx += 1
            print_bar_header(f"Bar {bar_idx}:")
            i += 1
            continue

        if tok == "Infill_Bar":
            bar_idx += 1
            print_bar_header(f"Infill_Bar {infill_bar_count}:")
            i += 1
            continue

        # -------- FillBar handling --------
        if tok == "FillBar_Start":
            print(f"\n<Start Infill>")
            i += 1
            continue

        if tok == "FillBar_End":
            print(f"{indent_track}<End Infill>")
            i += 1
            continue

        # -------- EOS --------
        if tok == "EOS_None":
            print("\n<EOS>")
            i += 1
            continue

        print(f"{indent_evt}{tok}")

        i += 1

def generate(
    model: object,
    tokenizer: MMM,
    inference_config: InferenceConfig,
    score_or_path: Score | Path | str,
    generate_kwargs: Mapping | None = None,
    seq2seq: bool = False
) -> Mapping:
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

    timer = InferenceTimer([
        PerfCounterTimer(),
        ProcessTimeTimer()
    ])

    timer.start_preprocessing()

    score = (
        Score(score_or_path) if not isinstance(score_or_path, Score) else score_or_path
    )

    # Infill bars
    if inference_config.infilling:
        for track, bars in inference_config.bars_to_generate.items():
            score = generate_infilling(
                model, 
                tokenizer, 
                track, 
                bars, 
                score, 
                generate_kwargs, 
                num_context = inference_config.context_length,
                timer = timer,
                seq2seq = seq2seq         
            )

    # Generate new tracks
    if inference_config.autoregressive:
        for track in inference_config.new_tracks:
            score = generate_new_track(
                model, 
                tokenizer, 
                track, 
                score, 
                generate_kwargs, 
                num_context = inference_config.context_length,
                timer = timer,
                seq2seq = seq2seq
            )

    timer.end_postprocessing()

    return {
        "score": score,
        "time_metrics": timer.report()
    }


def generate_new_track(
    model: object,
    tokenizer: MMM,
    track: tuple[int, list[str]],
    score: Score,
    generate_kwargs: Mapping | None = None,
    num_context: int = 8,
    max_len: int = 2048,
    timer: InferenceTimer | None = None,
    seq2seq: bool = False
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
            "EOS_None"
        ]

    # --- Encode all tracks (preserve original copy)
    all_tracks_seq = tokenizer.encode(score, concatenate_track_sequences=False)
    full_tracks_seq = [replace(seq) for seq in all_tracks_seq]

    # --- Create truncated context copy (bar-aware)
    truncated_tracks = []
    bar_token_id = tokenizer.vocab["Bar_None"]
    max_length = 0
    for seq in all_tracks_seq:
        bar_idxs = np.where(np.array(seq.ids) == bar_token_id)[0]
        if len(bar_idxs) > max_length:
            max_length = len(bar_idxs)
        if len(bar_idxs) > num_context:
            keep_start = bar_idxs[-num_context]
            seq.ids = seq.ids[keep_start:]
            seq.tokens = seq.tokens[keep_start:]
        truncated_tracks.append(seq)
    num_bars_to_generate = min(max_length, num_context)

    # --- Concatenate reduced tracks to form model input
    input_seq = sum(truncated_tracks, start=TokSequence())

    new_seq = TokSequence()

    # --- Add new track header
    input_seq.ids.append(tokenizer.vocab["Infill_Track"])
    input_seq.tokens.append("Infill_Track")
    new_seq.ids.append(tokenizer.vocab["Track_Start"])
    new_seq.tokens.append("Track_Start")
    new_seq.ids.append(tokenizer.vocab[f"Program_{track[0]}"])
    new_seq.tokens.append(f"Program_{track[0]}")

    # --- Attribute controls
    num_attr = len(track[1])
    for control in track[1]:
        new_seq.ids.append(tokenizer.vocab[control])
        new_seq.tokens.append(control)

    # --- Truncate if too long for model
    if len(input_seq.ids) + len(new_seq.ids) > max_len:
        input_seq.ids = input_seq.ids[-max_len:]
        input_seq.tokens = input_seq.tokens[-max_len:]

    if not seq2seq:
        input_seq += new_seq

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

    if timer:
        timer.end_preprocessing()
        timer.start_inference()

    if seq2seq:

        output_ids = model.generate(
            LongTensor([input_seq.ids]), 
            decoder_input_ids=new_seq.ids,
            logits_processor=logit_processor_list,
            **generate_kwargs
        )[0].tolist()

    else:

        output_ids = model.generate(
            LongTensor([input_seq.ids]), 
            logits_processor=logit_processor_list,
            **generate_kwargs
        )[0].tolist()

    if timer:
        timer.end_inference()
        timer.start_postprocessing()

    if timer:
        if seq2seq:
            timer.set_num_tokens(len(input_seq.ids), len(output_ids) - len(new_seq.ids))
        else:
            timer.set_num_tokens(len(input_seq.ids), len(output_ids) - len(input_seq.ids))

    output_seq = TokSequence(ids=output_ids, are_ids_encoded=True)

    # --- Clean up (remove controls)
    if not seq2seq:
        output_start = len(input_seq) - num_attr - 3
    else:
        output_start = 0
    output_seq = output_seq[output_start:]
    first_generated_bar = np.where(np.array(output_seq.ids) == tokenizer.vocab["Bar_None"])[0][0]
    output_seq = output_seq[:2] + output_seq[first_generated_bar:]

    tokenizer.decode_token_ids(output_seq)
    output_seq.tokens = tokenizer._ids_to_tokens(output_seq.ids)

    if output_seq.tokens[-1] == "EOS_None":
        output_seq = output_seq[:-1] 
    else:
        warnings.warn(
            "Model generation did not terminate with <EOS_NONE>; ignoring.",
            stacklevel=2,
        )


    # --- Ensure <TRACK_END> closure
    if output_seq.tokens[-1] != "Track_End":
        warnings.warn(
            "Track generation did not terminate with <TRACK_END>; appending manually.",
            stacklevel=2,
        )
        output_seq.ids.append(tokenizer.vocab["Track_End"])
        output_seq.tokens.append("Track_End")

    # --- Remove last <BAR_NONE> token if necessary
    if output_seq.tokens[-2] == "Bar_None":
        output_seq = output_seq[:-2] + output_seq[-1:]

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
    max_len: int = 2048,
    timer: InferenceTimer | None = None,
    seq2seq: bool = False
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
            "EOS_None"
        ]

    for start_bar_idx, end_bar_idx, attribute_controls in infill:
        # token_start_idx and token_end_idx are the indices of start
        # and end of infilling, when the toksequence is NOT BPE encoded

        tokens = tokenizer.encode(score, concatenate_track_sequences=False)

        toksequence_to_infill = TokSequence(are_ids_encoded=False)

        times = np.array([event.time for event in tokens[track_idx].events])

        bars_ticks = tokens[track_idx]._ticks_bars
        num_bars = len(bars_ticks)
        if DEBUG:
            print(f"Num bars: {num_bars}")
        
        assert start_bar_idx >= 0, f"Invalid infilling start bar index : {start_bar_idx}"
        assert end_bar_idx <= num_bars, f"Invalid infilling end bar index : {end_bar_idx} (must be leq than {num_bars})"
        
        bar_tick_start = bars_ticks[start_bar_idx]

        token_idx_start = np.nonzero(times >= bar_tick_start)[0]
        token_idx_start = token_idx_start[0]

        context_start_bar = max(start_bar_idx - num_context,0)

        # Context
        context_token_start_idx = np.nonzero(
            times >= bars_ticks[context_start_bar]
        )[0][0]

        # If the first bar is in the context, we remove track_start + program
        # These will be added seperately
        if context_start_bar == 0:
            context_token_start_idx += 2

        if end_bar_idx < num_bars:
            bar_tick_end = bars_ticks[end_bar_idx]

            token_idx_end = np.nonzero(times >= bar_tick_end)[0][0]

            context_end_bar = min(end_bar_idx + num_context, num_bars)

            if context_end_bar < num_bars:
                context_token_end_idx = np.nonzero(
                    times >= bars_ticks[context_end_bar]
                )[0][0]
            else:
                context_token_end_idx = len(tokens[track_idx]) - 1
        else:
            context_end_bar = end_bar_idx
            context_token_end_idx = len(tokens[track_idx].tokens)
            token_idx_end = context_token_end_idx

        if DEBUG:
            print(f"Context: bar [{context_start_bar},{context_end_bar}(")

        # Decode BPE tokens: this is necessary to put <INFILL_BAR> tokens
        # at the right place
        tokenizer.decode_token_ids(tokens[track_idx])

        track_start_tokens = tokens[track_idx][:2]

        seq_before = (
            track_start_tokens
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
            if context_start_bar == 0:
                context_token_start_idx += 2
            if context_end_bar < num_bars:
                context_token_end_idx = np.nonzero(
                    times >= bars_ticks[context_end_bar]
                )[0][0]
            else:
                context_token_end_idx = len(tokens[i]) - 1
            track_seq = (
                tokens[i][:2]
                + tokens[i][context_token_start_idx:context_token_end_idx]
                + tokens[i][-1:]
            )
            input_seq += track_seq

        new_seq = TokSequence()

        new_seq.ids.append(tokenizer.vocab["FillBar_Start"])
        new_seq.tokens.append("FillBar_Start")

        new_seq.ids.append(tokenizer.vocab["Bar_None"])
        new_seq.tokens.append("Bar_None")

        for control in attribute_controls:
            new_seq.ids.append(tokenizer.vocab[control])
            new_seq.tokens.append(control)

        if len(input_seq.ids) + len(new_seq.ids) > max_len:
            input_seq.ids = input_seq.ids[-max_len:]
            input_seq.tokens = input_seq.tokens[-max_len:]

        if not seq2seq:
            input_seq += new_seq

        num_bars_to_generate = end_bar_idx - start_bar_idx

        logits_processor = InfillLogitsProcessor(
            tokenizer.vocab["FillBar_Start"],
            tokenizer.vocab["FillBar_End"],
            tokenizer.vocab["Bar_None"],
            tokenizer.vocab["EOS_None"],
            num_bars_to_generate
        )
        logit_processor_list = LogitsProcessorList()
        logit_processor_list.append(logits_processor)

        if timer:
            timer.end_preprocessing()
            timer.start_inference()

        if seq2seq:

            output_ids = model.generate(
                LongTensor([input_seq.ids]),
                decoder_input_ids=new_seq.ids,
                logits_processor=logit_processor_list,
                **generate_kwargs,
            )[0].numpy()

        else:

            output_ids = model.generate(
                LongTensor([input_seq.ids]),
                logits_processor=logit_processor_list,
                **generate_kwargs,
            )[0].numpy()

        if timer:
            timer.end_inference()
            timer.start_postprocessing()

            if seq2seq:
                timer.set_num_tokens(len(input_seq.ids), len(output_ids) - len(new_seq.ids))
            else:
                timer.set_num_tokens(len(input_seq.ids), len(output_ids) - len(input_seq.ids))

        if DEBUG:
            pretty_print_tokens(tokenizer._ids_to_tokens(output_ids))

        if output_ids[-1] == tokenizer.vocab["EOS_None"]:
            output_ids = output_ids[:-1] 
        else:
            warnings.warn(
                "Model generation did not terminate with <EOS_NONE>; ignoring.",
                stacklevel=2,
            )

        if output_ids[-1] == tokenizer.vocab["FillBar_End"]:
            output_ids = output_ids[:-1] 
        else:
            warnings.warn(
                "Model generation did not terminate with <FILLBAR_END>; ignoring.",
                stacklevel=2,
            )

        # Remove last bar if necesssary
        if output_ids[-1] == tokenizer.vocab["Bar_None"]:
            output_ids = output_ids[:-1] 

        fill_start_idx = np.where(output_ids == tokenizer.vocab["FillBar_Start"])[0][0]

        first_bar_start = np.where(output_ids[fill_start_idx:] == tokenizer.vocab["Bar_None"])[0][0]

        output_ids = output_ids[fill_start_idx + first_bar_start:]

        bar_none_id = tokenizer.vocab["Bar_None"]
        num_generated_bars = np.sum(np.array(output_ids) == bar_none_id)

        if num_generated_bars != num_bars_to_generate:
            warnings.warn(
                "Model generation did not produce enough bars; completing with empty bars.",
                stacklevel=2,
            )
            output_ids += [bar_none_id] * (num_bars_to_generate - num_generated_bars)

        # Here we isolate the generated tokens doing some filtering. In particular,
        # the model may generate some tokens before the first Bar_None token
        generated_tokens = TokSequence(are_ids_encoded=True)
        #generated_tokens.ids = output_ids[
        #    fill_start_idx + len(attribute_controls) + 1 : -1
        #].tolist()
        generated_tokens.ids = output_ids.tolist()
        # decode_token_ids doesn't support numpy arrays for ids list
        tokenizer.decode_token_ids(generated_tokens)

        tokenizer.decode_token_ids(tokens[track_idx])
        tokens[track_idx].ids[token_idx_start:token_idx_end] = generated_tokens.ids
        tokens[track_idx].tokens = tokenizer._ids_to_tokens(tokens[track_idx].ids)

        score = tokenizer.base_tokenizer._tokens_to_score(tokens)

    return score
