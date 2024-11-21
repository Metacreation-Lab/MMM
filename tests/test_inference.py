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
from transformers import GenerationConfig, MistralForCausalLM

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
CONTEXT_SIZE = 40

# Number of random infilling to perform per MIDI file.
NUM_INFILLINGS_PER_TRACK = 10
NUM_GENERATIONS_PER_INFILLING = 1

# Number of bars to infill in a track
NUM_BARS_TO_INFILL = 20
"""
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1},
    "num_velocities": 24,
    "special_tokens": [
        "PAD",
        "BOS",
        "EOS",
        "Infill_Bar",  # Indicates a bar to be filled in a seq
        "Infill_Track",  # Used in seq2seq to instruct the decoder to gen a new track
        "FillBar_Start",  # Start of the portion to infill (containing n bars)
        "FillBar_End",  # Ends the portion to infill
    ],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_pitch_intervals": False,  # cannot be used as extracting tokens in data loading
    "use_programs": True,
    "num_tempos": 48,
    "tempo_range": (50, 200),
    "programs": list(range(-1, 127)),
    "base_tokenizer": "REMI",
}
config = TokenizerConfig(**TOKENIZER_PARAMS)
"""


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

    # Creating model
    model = MistralForCausalLM.from_pretrained(
        #Path(__file__).parent.parent / "models" / "checkpoint-87000",
        "C:\\Users\\rizzo\\Desktop\\TESI\\dependencies_test\\MMM\\models\\checkpoint-87000",
        use_safetensors=True,
    )

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
    )

    # Get number of tracks and number of bars of the MIDI track
    score = symusic.Score(input_midi_path)
    tokens = tokenizer.encode(score, concatenate_track_sequences=False)

    num_tracks = len(tokens)
    print(f"[INFO::test_generate] Number of tracks: {num_tracks} ")

    output_folder_path = (
        Path(__file__).parent
        / "tests_output"
        / "87k"
        / "TEST_DEBUG"
        / f"test_{input_midi_path.name!s}"
    )

    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Write gen_config to JSON file
    gen_config_dict = vars(gen_config)

    i = 0
    # To be added to JSON file
    infillings = []
    while i < NUM_INFILLINGS_PER_TRACK:
        track_idx = random.randint(0, num_tracks - 1)

        bars_ticks = tokens[track_idx]._ticks_bars
        num_bars = len(bars_ticks)

        bar_idx_infill_start = random.randint(
            CONTEXT_SIZE, num_bars - CONTEXT_SIZE - NUM_BARS_TO_INFILL - 1
        )

        bar_tick_end = bars_ticks[
            bar_idx_infill_start + NUM_BARS_TO_INFILL + CONTEXT_SIZE
        ]

        times = np.array([event.time for event in tokens[track_idx].events])
        token_idx_end = np.nonzero(times >= bar_tick_end)[0]

        if len(token_idx_end) == 0:
            print(
                f"[WARNING::test_generate] Ignoring infilling of bars "
                f"{bar_idx_infill_start} - "
                f"{bar_idx_infill_start + NUM_BARS_TO_INFILL} on track {track_idx}"
            )
            continue

        inference_config = InferenceConfig(
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
        while j < NUM_GENERATIONS_PER_INFILLING:
            print(
                f"[INFO::test_generate] Generation #{j} for track {track_idx} "
                f"(with {num_bars} bars) on bars "
                f"{bar_idx_infill_start} -{bar_idx_infill_start + NUM_BARS_TO_INFILL}"
            )

            start_time = time.time()

            _ = generate(
                model,
                tokenizer,
                inference_config,
                input_midi_path,
                {"generation_config": gen_config},
            )

            end_time = time.time()

            print(f"[INFO::test_generate] ...Done in {end_time - start_time} seconds")

            _.dump_midi(
                output_folder_path / f"track{track_idx}_"
                f"infill_bars{bar_idx_infill_start}_{bar_idx_infill_start+NUM_BARS_TO_INFILL}"
                f"_generationtime_{end_time - start_time}.midi.mid"
            )

            j += 1

        infillings.append(entry)

        i += 1

    json_data = {"generation_config": gen_config_dict, "infillings": infillings}

    json_string = json.dumps(json_data, indent=4)
    output_json = Path(output_folder_path) / "generation_config.json"
    with output_json.open("w") as file:
        file.write(json_string)
