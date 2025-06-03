"""Inference method for the MMM model."""

from __future__ import annotations

import re
import time
import warnings
import logging
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from miditok import MMM, TokSequence
from miditok.pytorch_data import DataCollator
from symusic import Score
from torch import LongTensor, no_grad, inference_mode
from transformers import LogitsProcessorList

from .logits_processor import InfillLogitsProcessor, TrackLogitsProcessor

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
    input_tokens: TokSequence | list[TokSequence] = None,
    device: str = None
) -> tuple[Score, dict]:
    """
    Use the model to generate a batch of new music content.

    The method allows to infill specific bars or generate new tracks.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param inference_config: InferenceConfig
    :param score_or_path: ``symusic.Score`` or path of the music file to infill.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object and metadata (eg. loops).
    """
    score = (
        Score(score_or_path) if not isinstance(score_or_path, Score) else score_or_path
    )

    # Infill bars
    if inference_config.infilling:
        logits_processor = InfillLogitsProcessor(
            tokenizer.vocab["Bar_None"], tokenizer.vocab["FillBar_End"], tokenizer
        )
        score, metadata = generate_infilling(
            model, tokenizer, inference_config, logits_processor,
            generate_kwargs, deepcopy(input_tokens)
        )

    # Generate new tracks
    if inference_config.autoregressive:
        logits_processor = TrackLogitsProcessor(
            tokenizer.vocab["Track_Start"], tokenizer.vocab["Bar_None"], tokenizer.vocab["Track_End"]
        )
        metadata = {}
        for track in inference_config.new_tracks:
            logging.debug("Generating Track")
            logging.debug(track[-1])
            score, metadata = generate_new_track(model, tokenizer, track, score, metadata, logits_processor, generate_kwargs, device)
            logging.debug(metadata)

    return score, metadata


def generate_batch(
    model: object,
    tokenizer: MMM,
    collator: DataCollator,
    inputs: list[dict[str, InferenceConfig | Score]],
    generate_kwargs: Mapping | None = None,
    device: str = None,
    remove_first_track: bool = True, #Tokenization on an empty score creates emtpy track, that we must remove
) -> tuple[Score, dict]:
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
    :return: the infilled ``symusic.Score`` object and metadata (eg. loops).
    """

    # Infill bars
    if all(i["config"].infilling for i in inputs):
        raise NotImplementedError("Batch sampling doesn't support infilling")

    # Generate new tracks
    elif all(i["config"].autoregressive for i in inputs):
        logits_processor = TrackLogitsProcessor(
            tokenizer.vocab["Track_Start"], tokenizer.vocab["Bar_None"], tokenizer.vocab["Track_End"]
        )

        batch_size = len(inputs)

        # Track current state for each item in batch
        current_scores = [i["score"] for i in inputs]          # list of mutable score strings
        current_metadatas = len(inputs) * [{}]
        configs = [i["config"] for i in inputs]                   # list of config objects
        max_tracks = max(len(cfg.new_tracks) for cfg in configs)

        for step in range(max_tracks):
            step_batch = []
            step_indices = []

            for i in range(batch_size):
                cfg = configs[i]
                if step < len(cfg.new_tracks):
                    # Prepare (score, metadata, track) for this step
                    step_batch.append((current_scores[i], current_metadatas[i], cfg.new_tracks[step]))
                    step_indices.append(i)

            if not step_batch:
                break  # All tracks completed

            # Run batch generation for this step
            step_results = generate_new_track_batch(
                model, tokenizer, collator, step_batch, logits_processor, generate_kwargs, device, (step == 0 & remove_first_track)
            )

            # Update scores and metadatas in-place
            for idx, (new_score, new_metadata) in zip(step_indices, step_results):
                current_scores[idx] = new_score
                current_metadatas[idx] = new_metadata

        return [
            {
                "score": current_scores[i],
                "metadata": current_metadatas[i]
            } 
            for i in range(batch_size)
        ]

    else:
        raise ValueError("All configurations must have the same sampling types")

    #return score, metadata


def generate_new_track(
    model: object,
    tokenizer: MMM,
    track: tuple[int, list[str]],
    score: Score,
    metadata: dict,
    logits_processor: TrackLogitsProcessor | None = None,
    generate_kwargs: Mapping | None = None,
    device: str = None
) -> tuple[Score, dict]:
    """
    Generate a new track of a given Score.

    The new track will be added to the score.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param track: tuple containing the program of the track and a list of Track
        Attribute Controls.
    :param score: symusic.Score
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    if not generate_kwargs:
        generate_kwargs = {}
        max_len = 100000
    else:
        generate_kwargs["generation_config"].eos_token_id = tokenizer.vocab[
            "Track_End"
        ]
        max_len = generate_kwargs["generation_config"].max_length

    # In this case, the prompt is a toksequence containing all the tracks
    metadata["tpq"] = score.ticks_per_quarter
    input_seq = tokenizer.encode(score, metadata=metadata)

    # Add <TRACK_START> and <PROGRAM> tokens
    input_seq.ids.append(tokenizer.vocab["Track_Start"])
    input_seq.tokens.append("Track_Start")
    input_seq.ids.append(tokenizer.vocab[f"Program_{track[0]}"])
    input_seq.tokens.append(f"Program_{track[0]}")

    # Add attribute control tokens
    for control in track[1]:
        input_seq.ids.append(tokenizer.vocab[control])
        input_seq.tokens.append(control)
    control_len = len(track[1])

    if len(input_seq) >= max_len - 1024:
        len_trim = len(input_seq) + 1024 - max_len
        input_seq_trimmed = input_seq[len_trim:]
    else:
        len_trim = 0
        input_seq_trimmed = input_seq

    logit_processor_list = LogitsProcessorList()
    logit_processor_list.append(logits_processor)

    input_tensor = LongTensor([input_seq_trimmed.ids])
    if device:
        input_tensor = input_tensor.to(device=device)

    with no_grad():
        output_ids = model.generate(
            input_tensor, 
            logits_processor=logit_processor_list,
            **generate_kwargs
        )
    output_seq = TokSequence(ids=output_ids[0].tolist(), are_ids_encoded=True)

    # Remove attribute controls from the sequence
    output_seq = (
        input_seq[:-control_len] + output_seq[len(input_seq) - len_trim + control_len:]
    )

    # Decode BPE ids before getting the associated tokens
    tokenizer.decode_token_ids(output_seq)

    pad_id = tokenizer.vocab["PAD_None"]
    eos_id = tokenizer.vocab["Track_End"]

    while output_seq.ids and (output_seq.ids[-1] == pad_id or output_seq.ids[-1] == eos_id):
        output_seq.ids.pop()
    output_seq.ids.append(eos_id)

    output_seq.tokens = tokenizer._ids_to_tokens(output_seq.ids)

    # It is expected to have a <TRACK_END> token at the end of the sequence.
    if output_seq.tokens[-1] != "Track_End":
        warnings.warn(
            "Track generation failed: the model failed to predict a <TRACK_END> token",
            stacklevel=2,
        )
        output_seq.ids.append(tokenizer.vocab["Track_End"])
        output_seq.tokens.append("Track_End")

    result, metadata = tokenizer._tokens_to_score(output_seq)
    #print(metadata)
    return result, metadata

def generate_new_track_batch(
    model: object,
    tokenizer: MMM,
    collator: DataCollator,
    scores_tracks: list[tuple[Score, dict, tuple[int, list[str]]]],
    logits_processor: TrackLogitsProcessor | None = None,
    generate_kwargs: Mapping | None = None,
    device: str = None,
    remove_first_track: bool = False
) -> tuple[Score, dict]:
    """
    Generate a new track of a given Score.

    The new track will be added to the score.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param track: tuple containing the program of the track and a list of Track
        Attribute Controls.
    :param score: symusic.Score
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    if generate_kwargs is None:
        generate_kwargs = {}
        max_len = 100000
    else:
        generate_kwargs["generation_config"].eos_token_id = tokenizer.vocab["Track_End"]
        max_len = generate_kwargs["generation_config"].max_length

    input_seqs = []
    input_ids_batch = []

    for score, metadata, track in scores_tracks:
        program, controls = track

        # Build input sequence for this track
        input_seq = tokenizer.encode(score, metadata=metadata)

        if remove_first_track:
            # Empty track includes 6 tokens
            if len(input_seq.ids) < 7:
                input_seq.ids = []
                input_seq.tokens = []
            else:
                input_seq.ids = input_seq.ids[6:]
                input_seq.tokens = input_seq.tokens[6:]

        # Append special tokens
        input_seq.ids.append(tokenizer.vocab["Track_Start"])
        input_seq.tokens.append("Track_Start")
        input_seq.ids.append(tokenizer.vocab[f"Program_{program}"])
        input_seq.tokens.append(f"Program_{program}")

        for control in controls:
            input_seq.ids.append(tokenizer.vocab[control])
            input_seq.tokens.append(control)

        if len(input_seq) >= max_len - 1024:
            len_trim = len(input_seq) + 1024 - max_len
            input_seq_trimmed = input_seq[len_trim:]
            #input_seq = input_seq[:max_len - 1024]
        else:
            len_trim = 0
            input_seq_trimmed = input_seq

        input_seqs.append((input_seq, len(controls), len_trim))  # Save original seq + control len
        input_ids_batch.append(LongTensor(input_seq_trimmed.ids))


    input_name = collator.inputs_kwarg_name
    batch = [{
        input_name:input_ids
    } for input_ids in input_ids_batch]

    padded_batch = collator(batch)[input_name]

    input_tensor = padded_batch
    if device:
        input_tensor = padded_batch.to(device=device)

    # Use shared logit processor for all sequences
    logit_processor_list = LogitsProcessorList()
    logit_processor_list.append(logits_processor)

    with inference_mode():
        output_ids_batch = model.generate(
            input_tensor,
            logits_processor=logit_processor_list,
            **generate_kwargs
        )

    results = []
    for output_ids, (input_seq, control_len, len_trim) in zip(output_ids_batch, input_seqs):
        output_seq = TokSequence(ids=output_ids.tolist(), are_ids_encoded=True)


        output_seq = (
            input_seq[:-control_len] + output_seq[len(input_seq) - len_trim + control_len:]
        )

        # Decode BPE ids before getting the associated tokens
        tokenizer.decode_token_ids(output_seq)

        pad_id = tokenizer.vocab["PAD_None"]
        eos_id = tokenizer.vocab["Track_End"]

        while output_seq.ids and (output_seq.ids[-1] == pad_id or output_seq.ids[-1] == eos_id):
            output_seq.ids.pop()
        output_seq.ids.append(eos_id)

        output_seq.tokens = tokenizer._ids_to_tokens(output_seq.ids)

        # Ensure <TRACK_END>
        if output_seq.tokens[-1] != "Track_End":
            warnings.warn(
                "Track generation failed: missing <TRACK_END>",
                stacklevel=2,
            )
            output_seq.ids.append(tokenizer.vocab["Track_End"])
            output_seq.tokens.append("Track_End")

        result, metadata = tokenizer._tokens_to_score(output_seq)
        results.append((result, metadata))

    return results  # List of (result, metadata)

def generate_infilling(
    model: object,
    tokenizer: MMM,
    inference_config: InferenceConfig,
    logits_processor: InfillLogitsProcessor | None = None,
    generate_kwargs: Mapping | None = None,
    input_tokens: TokSequence | list[TokSequence]  = None,
    device: str = None
) -> tuple[Score, dict]:
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

    tracks_to_infill = inference_config.bars_to_generate.keys()

    for track_to_infill in tracks_to_infill:
        infill_bars(
            model,
            tokenizer,
            track_to_infill,
            inference_config,
            input_tokens,
            logits_processor,
            generate_kwargs,
            device
        )

    # Here we use the base tokenizer because output_tokens is a list of TokSequences

    start_time = time.time()
    result, metadata = tokenizer.base_tokenizer._tokens_to_score(input_tokens)
    end_time = time.time()
    print(
        "[INFO::generate_infilling] Time spent for converting tokens to score: ",
        end_time - start_time,
    )
    return result, metadata


def infill_bars(
    model: object,
    tokenizer: MMM,
    track_idx: int,
    inference_config: InferenceConfig,
    tokens: list[TokSequence],
    logits_processor: InfillLogitsProcessor | None = None,
    generate_kwargs: Mapping | None = None,
    device: str = None
) -> None:
    """
    Infill bars for the ''track_idx'' track.

    The tokens are replaced inplace.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param track_idx: index of the track to infill
    :param inference_config: contains information about which tracks and bars to
        generate.
    :param tokens: TokSequence of the track to be infilled
    :param logits_processor: ``transformers.LogitsProcessor`` used to stop generation
        when the right number of bars is generated.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    """
    if not generate_kwargs:
        generate_kwargs = {}

    # For each set of bars to infill in the track, we generate new content
    # (We may have, in the same track, non-adjacent sequences of bars. For
    # each sequence, we do a generation step).
    for subset_bars_to_infill in inference_config.bars_to_generate[track_idx]:
        # token_start_idx and token_end_idx are the indices of start
        # and end of infilling, when the toksequence is NOT BPE encoded
        start_time = time.time()

        input_seq, token_start_idx, token_end_idx = _adapt_prompt_for_infilling(
            tokenizer, track_idx, tokens, subset_bars_to_infill,
            inference_config.context_length
        )

        end_time = time.time()
        print(
            "[INFO::infill_bars] Time spent for creating input sequence: ",
            end_time - start_time,
        )

        logits_processor.n_bars_to_infill = (
            subset_bars_to_infill[1] - subset_bars_to_infill[0]
        )
        logits_processor.n_attribute_controls = len(subset_bars_to_infill[2])
        logits_processor.infill_type = subset_bars_to_infill[3]
        logit_processor_list = LogitsProcessorList()
        logit_processor_list.append(logits_processor)

        start_time = time.time()

        input_tensor = LongTensor([input_seq.ids])
        if device:
            input_tensor = input_tensor.to(device=device)

        output_ids = model.generate(
            input_tensor,
            logits_processor=logit_processor_list,
            **generate_kwargs,
        )[0].numpy()

        end_time = time.time()
        generation_time = end_time - start_time
        print("[INFO::infill_bars] Time spent for generation: ", generation_time)
        print("Time spent in logits processor ", logits_processor.total_time)

        #with open("output.txt", "a") as f:
        #    f.write(f"{number}\n")

        start_time = time.time()

        if subset_bars_to_infill[3] == "bar":
            fill_start_idx = np.where(output_ids == tokenizer.vocab["FillBar_Start"])[0][0]
        elif subset_bars_to_infill[3] == "track":
            fill_start_idx = np.where(output_ids == tokenizer.vocab["Infill_Track"])[0][0]

        # Here we isolate the generated tokens doing some filtering. In particular,
        # the model may generate some tokens before the first Bar_None token
        generated_tokens = TokSequence(are_ids_encoded=True)
        generated_tokens.ids = output_ids[
            fill_start_idx + len(subset_bars_to_infill[2]) + 1 : -1
        ].tolist()
        # decode_token_ids doesn't support numpy arrays for ids list
        tokenizer.decode_token_ids(generated_tokens)
        bar_none_token_idxs = np.where(
            np.array(generated_tokens.ids) == tokenizer.vocab["Bar_None"]
        )[0]
        # bar_none_token_idxs[-1] because we must exclude the last BarNone token,
        # which is used by the logits processor to stop generation
        generated_tokens.ids = generated_tokens.ids[
            bar_none_token_idxs[0] : bar_none_token_idxs[-1]
        ]

        tokens[track_idx].ids[token_start_idx:token_end_idx] = generated_tokens.ids
        tokens[track_idx].tokens = tokenizer._ids_to_tokens(tokens[track_idx].ids)

        end_time = time.time()
        print(
            "[INFO::infill_bars] Time spend for reconstructing the sequence: ",
            end_time - start_time,
        )


def _adapt_prompt_for_infilling(
    tokenizer: MMM,
    track_idx: int,
    tokens: list[TokSequence],
    subset_bars_to_infill: tuple[int, int, list[str]],
    context_length: int
) -> TokSequence:
    """
    Construct the prompt for bar infilling.

    Constructs the prompt to be used as model's input. The sequence should have the
    "BAR_FILL" format:
    ``<TRACK_START>...<TRACK_END>...<TRACKS_START>...<INFILL_BAR>...<INFILL_BAR>...
    <TRACK_END>...<TRACK_START>...<TRACK_END><START_FILL>``
    We have as many <FILL_IN> tokens as the number of bars we want to infill.

    :param tokenizer: MMM tokenizer
    :param track_idx: index of the track to infill
    :param tokens: TokSequence of the track to be infilled
    :param subset_bars_to_infill: contains the indexes of the first and last bar to
        infill, plus a list of attribute controls
    """
    toksequence_to_infill: TokSequence = TokSequence(are_ids_encoded=False)

    # Decode BPE tokens: this is necessary to put <INFILL_BAR> tokens
    # at the right place
    tokenizer.decode_token_ids(tokens)

    start_bar_idx = subset_bars_to_infill[0]
    end_bar_idx = subset_bars_to_infill[1]

    bars_ticks = tokens[track_idx]._ticks_bars
    num_bars = len(tokens[track_idx]._ticks_bars)

    #### Infilling tokens ####
    times = np.array([event.time for event in tokens[track_idx].events])

    token_idx_start = np.nonzero(times >= bars_ticks[start_bar_idx])[0][0]

    # In this case, infilling is done until the end of the track
    if end_bar_idx >= num_bars:
        token_idx_end = len(tokens[track_idx]) - 1
    else:
        token_idx_end = np.nonzero(times >= bars_ticks[end_bar_idx])[0][0]

    if subset_bars_to_infill[3] == "bar":
        ##### Infilling Context#####
        
        # If the number of bars available as context in the left
        # is less than context length
        if start_bar_idx - context_length < 0:
            context_token_start_idx = 2
        else:
            context_token_start_idx = np.nonzero(
                times >= bars_ticks[start_bar_idx - context_length]
            )[0][0]
        
        # Take right context
        if end_bar_idx < num_bars:
            # If the number of bars available as context in the right
            # is less than context length
            if end_bar_idx + context_length > num_bars - 1:
                context_token_end_idx = len(tokens[track_idx]) - 1
            else:
                context_token_end_idx = np.nonzero(
                    times >= bars_ticks[end_bar_idx + context_length]
                )[0][0]
        # Only left context (aka END INFILLING)
        else:
            context_token_end_idx = -1
        
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
        

    output_toksequence = TokSequence(are_ids_encoded=False)

    ###### Context for other tracks #####
    for i in range(len(tokens)):
        if subset_bars_to_infill[3] == "bar":
            if i == track_idx:
                output_toksequence += toksequence_to_infill
                continue
    
            #with open("model_prompt_tokens.txt", "w") as file:
            #    for token in output_toksequence.tokens:
            #        file.write(token + "\n")
    
            times = np.array([event.time for event in tokens[i].events])
            if start_bar_idx - context_length < 0:
                context_token_start_idx = 0
            else:
                context_token_start_idx = np.nonzero(
                    times >= bars_ticks[start_bar_idx - context_length]
                )[0][0]
            if end_bar_idx + context_length >= num_bars - 1:
                context_token_end_idx = len(tokens[i]) - 1
            else:
                # TODO: number of bars of some tracks computed through
                #   miditok is not always right, meaning that the list
                #   of tokens after the context may be out of the allowed
                #   range
                context_token_end_idx = np.nonzero(
                    times >= bars_ticks[end_bar_idx + context_length]
                )[0]
                # In that case, we just take the last token as the end of context
                if len(context_token_end_idx) == 0:
                    context_token_end_idx = len(tokens[i]) - 1
                else:
                    context_token_end_idx = context_token_end_idx[0]
    
            # Add the section to the context only if it is not empty
            sliced_tokens = tokens[i][context_token_start_idx:context_token_end_idx]
    
            pattern = r"Pitch"
            if any(re.match(pattern, token) for token in sliced_tokens.tokens):
                output_toksequence += (
                    tokens[i][:2]
                    + sliced_tokens
                    + tokens[i][-1:]
                )
        elif subset_bars_to_infill[3] == "track":
            if i == track_idx:
                infill_program_id = tokens[i][1]
                infill_program_token = tokens[i].tokens[1]
                continue

            times = np.array([event.time for event in tokens[i].events])

            #### Infilling tokens ####

            token_idx_start_ = np.nonzero(times >= bars_ticks[start_bar_idx])[0][0]

            # In this case, infilling is done until the end of the track
            if end_bar_idx >= num_bars:
                token_idx_end_ = len(tokens[i]) - 1
            else:
                token_idx_end_ = np.nonzero(times >= bars_ticks[end_bar_idx])[0][0]

            # Add the section to the context only if it is not empty
            sliced_tokens = tokens[i][token_idx_start_:token_idx_end_]

            pattern = r"Pitch"
            if any(re.match(pattern, token) for token in sliced_tokens.tokens):
                output_toksequence += (
                        tokens[i][:2]
                        + sliced_tokens
                        + tokens[i][-1:]
                )
    
    if subset_bars_to_infill[3] == "bar":
        output_toksequence.ids.append(tokenizer.vocab["FillBar_Start"])
        output_toksequence.tokens.append("FillBar_Start")
    elif subset_bars_to_infill[3] == "track":
        output_toksequence.ids.append(tokenizer.vocab["Infill_Track"])
        output_toksequence.tokens.append("Infill_Track")
        output_toksequence.ids.append(infill_program_id)
        output_toksequence.tokens.append(infill_program_token)
            
            

    attribute_controls = subset_bars_to_infill[2]
    for control in attribute_controls:
        output_toksequence.ids.append(tokenizer.vocab[control])
        output_toksequence.tokens.append(control)

    # Encode into BPE tokens
    tokenizer.encode_token_ids(output_toksequence)

    #Just for debugging purposes

    with open("model_prompt_tokens.txt", "w") as file:
        bar_n = 0
        track_n = 0
        for token in output_toksequence.tokens:
            if token == "Track_End":
                bar_n = 0
                track_n += 1
            if token == "Bar_None":
                file.write(f"TrackNumber:{track_n} BarNumber:{bar_n} " + token + "\n")
                bar_n += 1
            else:
                file.write(token + "\n")

    return output_toksequence, token_idx_start, token_idx_end