"""Tests for MMM inference."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pytest
import symusic
from miditok import MMM
from symusic import Score
from transformers import GenerationConfig, MistralForCausalLM, AutoModelForCausalLM

from mmm import InferenceConfig, generate
from scripts.utils.constants import (
    EPSILON_CUTOFF,
    ETA_CUTOFF,
    MAX_LENGTH,
    MAX_NEW_TOKENS,
    NUM_BEAMS,
    REPETITION_PENALTY,
    TEMPERATURE_SAMPLING,
    TOP_K,
    TOP_P,
)

from .utils_tests import MIDI_PATH

# Definition of variables

#
# CONTEXT CONSTRUCTION
#
# TRACK 1 : []...[CONTEXT_SIZE BARS][INFILLING CONTEXT][CONTEXT_SIZE BARS]...[]
# ...
# TRACK
# TO INFILL: []...[CONTEXT_SIZE BARS][REGION_TO_INFILL][CONTEXT_SIZE BARS]...[]
# ...
# TRACK n : []...[CONTEXT_SIZE BARS][INFILLING CONTEXT][CONTEXT_SIZE BARS]...[]
# MusIAC uses 6 bars as context size!
CONTEXT_SIZE = 0

# Number of random infilling to perform per MIDI file.
NUM_INFILLINGS_PER_MIDI_FILE = 50
NUM_GENERATIONS_PER_INFILLING = 1

# Number of bars to infill in a track
NUM_BARS_TO_INFILL = 8

DRUM_GENERATION = False

MODEL_PATH = Path("runs/models/MISTRAL_123000")
MIDI_OUTPUT_FOLDER = (Path(__file__).parent
        / "tests_output"
        / "FINAL_TESTS"
        / "mistral"
        / f"temp{TEMPERATURE_SAMPLING}"
          f"_rep{REPETITION_PENALTY}"
          f"_topK{TOP_K}_topP{TOP_P}"
          f"num_bars_infill{NUM_BARS_TO_INFILL}_context_{CONTEXT_SIZE}")

@pytest.mark.parametrize(
    "tokenizer",
    [MMM(params=Path(__file__).parent.parent / "runs" / "tokenizer.json")],
    # "tokenizer", [MMM(config)]
)
@pytest.mark.parametrize("input_midi_path", MIDI_PATH)
# @pytest.mark.parametrize("context_size", CONTEXT_SIZE)
# pytest.mark.skip(reason="This is a generation test! Skipping...")
def test_generate(tokenizer: MMM, input_midi_path: str | Path):
    print(f"[INFO::test_generate] Testing MIDI file: {input_midi_path} ")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    print(model.config)
    # Creating generation config
    gen_config = GenerationConfig(
        num_beams=NUM_BEAMS,
        temperature=TEMPERATURE_SAMPLING,
        repetition_penalty=REPETITION_PENALTY,
        top_k=TOP_K,
        top_p=TOP_P,
        epsilon_cutoff=EPSILON_CUTOFF,
        eta_cutoff=ETA_CUTOFF,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=MAX_LENGTH,
        do_sample = True
    )

    # Get number of tracks and number of bars of the MIDI track
    score = symusic.Score(input_midi_path)
    tokens = tokenizer.encode(score, concatenate_track_sequences=False)

    num_tracks = len(tokens)
    print(f"[INFO::test_generate] Number of tracks: {num_tracks} ")

    output_folder_path = (
        MIDI_OUTPUT_FOLDER
        / f"test_{input_midi_path.name!s}"
    )

    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Write gen_config to JSON file
    gen_config_dict = vars(gen_config)

    i = 0
    # To be added to JSON file
    infillings = []
    while i < NUM_INFILLINGS_PER_MIDI_FILE:

        # Select random track index to infill
        track_idx = random.randint(0, num_tracks-1)
        # TODO: just find the drum track index and use that, no
        # need to do this garbage
        if DRUM_GENERATION:
            if tokens[track_idx].tokens[1] != "Program_-1":
                continue
        else:
            # If not generating drums, skip until we sample
            # a non drum track index
            if tokens[track_idx].tokens[1] == "Program_-1":
                continue
        #if len(score.tracks[track_idx].notes) == 0:
        #    continue

        bars_ticks = tokens[track_idx]._ticks_bars
        num_bars = len(bars_ticks)
        #bar_idx_infill_start = 88

        # Select random portion of the track to infill
        bar_idx_infill_start = random.randint(
            CONTEXT_SIZE // 4, (num_bars - CONTEXT_SIZE - NUM_BARS_TO_INFILL - 1) // 4
        ) * 4

        # Compute stuff to discard infillings when we have no context!
        bar_left_context_start = bars_ticks[
            bar_idx_infill_start - CONTEXT_SIZE
        ]
        bar_infilling_start = bars_ticks[bar_idx_infill_start]
        bar_infilling_end = bars_ticks[bar_idx_infill_start + CONTEXT_SIZE]
        bar_right_context_end = bars_ticks[
            bar_idx_infill_start + NUM_BARS_TO_INFILL + CONTEXT_SIZE
        ]

        times = np.array([event.time for event in tokens[track_idx].events])
        types = np.array([event.type_ for event in tokens[track_idx].events])
        tokens_left_context_idxs = np.nonzero((times >= bar_left_context_start) & (times <= bar_infilling_start))[0]
        tokens_left_context_types = set(types[tokens_left_context_idxs])
        tokens_infilling = np.nonzero((times >= bar_infilling_start) & (times <= bar_infilling_end))[0]
        tokens_right_context_idxs = np.nonzero((times >= bar_infilling_end) & (times <= bar_right_context_end))[0]
        tokens_right_context_types = set(types[tokens_right_context_idxs])
        if "Pitch" not in tokens_left_context_types or "Pitch" not in tokens_right_context_types:
            print(
                f"[WARNING::test_generate] Ignoring infilling of bars "
                f"{bar_idx_infill_start} - "
                f"{bar_idx_infill_start + NUM_BARS_TO_INFILL} on track {track_idx}"
                " because we have no context around the infilling region"
            )
            continue

        if len(tokens_infilling) == 0:
            print(
                f"[WARNING::test_generate] Infilling region"
                f"{bar_idx_infill_start} - "
                f"{bar_idx_infill_start + NUM_BARS_TO_INFILL} on track {track_idx}"
                "has no notes!"
            )

        inference_config = InferenceConfig(
            CONTEXT_SIZE,
            {
                track_idx: [
                    (
                        bar_idx_infill_start,
                        bar_idx_infill_start + NUM_BARS_TO_INFILL,
                        [],
                    )
                ],
            },
            [],
        )

        entry = {
            "name": str(input_midi_path.name),
            "track_idx": track_idx,
            "start_bar_idx": bar_idx_infill_start,
            "end_bar_idx": bar_idx_infill_start + NUM_BARS_TO_INFILL,
        }

        j = 0
        successful = True
        while j < NUM_GENERATIONS_PER_INFILLING:
            print(
                f"[INFO::test_generate] Generation #{j} for track {track_idx} "
                f"(with {num_bars} bars) on bars "
                f"{bar_idx_infill_start} -{bar_idx_infill_start + NUM_BARS_TO_INFILL}"
            )

            start_time = time.time()
            try:
                _ = generate(
                    model,
                    tokenizer,
                    inference_config,
                    input_midi_path,
                    {"generation_config": gen_config},
                    input_tokens=tokens
                )
            except Exception as e:
                print(f"An error occurred during generation: {e}")
                successful = False
                break


            end_time = time.time()

            print(f"[INFO::test_generate] ...Done in {end_time - start_time} seconds")

            _.dump_midi(
                output_folder_path / f"{input_midi_path.stem}_track{track_idx}_"
                f"infill_bars{bar_idx_infill_start}_{bar_idx_infill_start+NUM_BARS_TO_INFILL}"
                f"_context_{CONTEXT_SIZE}"
                f"_generationtime_{end_time - start_time}_{i}{j}.mid"
            )
            infillings.append(entry)

            j += 1

        if successful:
            i += 1

    json_data = {"generation_config": gen_config_dict, "infillings": infillings}

    json_string = json.dumps(json_data, indent=4)
    output_json = Path(output_folder_path) / "generation_config.json"
    with output_json.open("w") as file:
        file.write(json_string)
