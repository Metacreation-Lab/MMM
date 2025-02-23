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

from .utils_tests import MIDI_PATHS

MODEL_PATH = Path("runs/models/MISTRAL_123000")
MIDI_OUTPUT_FOLDER = (Path(__file__).parent
        / f"temp{TEMPERATURE_SAMPLING}"
          f"_rep{REPETITION_PENALTY}"
          f"_topK{TOP_K}_topP{TOP_P}"
          f"num_bars_infill{NUM_BARS_TO_INFILL}_context{CONTEXT_SIZE}")

def test_generate(tokenizer: MMM, model, gen_config, input_midi_path: str | Path):

    MIDI_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Get number of tracks and number of bars of the MIDI track
    score = symusic.Score(input_midi_path)
    tokens = tokenizer.encode(score, concatenate_track_sequences=False)

    gen_config_dict = vars(gen_config)

    # Select random track index to infill
    track_idx = random.randint(0, num_tracks-1)

    if DRUM_GENERATION:
        while tokens[track_idx].tokens[1] != "Program_-1":
            track_idx = random.randint(0, num_tracks-1)
            continue
    else:
        # If not generating drums, skip until we sample
        # a non drum track index
        while tokens[track_idx].tokens[1] == "Program_-1":
            track_idx = random.randint(0, num_tracks-1)
            continue

    bars_ticks = tokens[track_idx]._ticks_bars
    num_bars = len(bars_ticks)


    if END_INFILLING:
        bar_idx_infill_start = num_bars - NUM_BARS_TO_INFILL
    else:
        bar_idx_infill_start = random.randint(
            CONTEXT_SIZE // 4, (num_bars - CONTEXT_SIZE - NUM_BARS_TO_INFILL) // 4
        ) * 4
    
    # Compute stuff to discard infillings when we have no context!
    bar_left_context_start = bars_ticks[
        bar_idx_infill_start - CONTEXT_SIZE
    ]
    bar_infilling_start = bars_ticks[bar_idx_infill_start]
    bar_infilling_end = bars_ticks[bar_idx_infill_start + CONTEXT_SIZE]

    if not END_INFILLING:
        bar_right_context_end = bars_ticks[
            bar_idx_infill_start + NUM_BARS_TO_INFILL + CONTEXT_SIZE
        ]

    times = np.array([event.time for event in tokens[track_idx].events])
    types = np.array([event.type_ for event in tokens[track_idx].events])
    tokens_left_context_idxs = np.nonzero((times >= bar_left_context_start) & (times <= bar_infilling_start))[0]
    tokens_left_context_types = set(types[tokens_left_context_idxs])
    tokens_infilling = np.nonzero((times >= bar_infilling_start) & (times <= bar_infilling_end))[0]
    if not END_INFILLING:
        tokens_right_context_idxs = np.nonzero((times >= bar_infilling_end) & (times <= bar_right_context_end))[0]
        tokens_right_context_types = set(types[tokens_right_context_idxs])
    
    if END_INFILLING:
        if "Pitch" not in tokens_right_context_types:
            print(
                f"[WARNING::test_generate] Ignoring infilling of bars "
                f"{bar_idx_infill_start} - "
                f"{bar_idx_infill_start + NUM_BARS_TO_INFILL} on track {track_idx}"
                " because we have no context around the infilling region"
            )
            return False
    elif "Pitch" not in tokens_left_context_types or "Pitch" not in tokens_right_context_types:
        print(
            f"[WARNING::test_generate] Ignoring infilling of bars "
            f"{bar_idx_infill_start} - "
            f"{bar_idx_infill_start + NUM_BARS_TO_INFILL} on track {track_idx}"
            " because we have no context around the infilling region"
        )
        return False

    if len(tokens_infilling) == 0:
        print(
            f"[WARNING::test_generate] Infilling region"
            f"{bar_idx_infill_start} - "
            f"{bar_idx_infill_start + NUM_BARS_TO_INFILL} on track {track_idx}"
            "has no notes!"
        )
        return False

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
        return False

    _.dump_midi(
            output_folder_path / f"{input_midi_path.stem}_track{track_idx}_"
            f"infill_bars{bar_idx_infill_start}_{bar_idx_infill_start+NUM_BARS_TO_INFILL}"
            f"_context_{CONTEXT_SIZE}"
            f"_generationtime_{end_time - start_time}.mid"
        )

CONTEXT_SIZE = None

NUM_BARS_TO_INFILL = None

DRUM_GENERATION = False

END_INFILLING = False

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Generate MIDI sequences with specified parameters.")
    parser.add_argument("--num_bars_infilling", type=int, required=True, help="Number of bars for infilling")
    parser.add_argument("--context", type=int, required=True, help="Context length")
    parser.add_argument("--num_generations", type=int, required=True, help="Number of generations")
    parser.add_argument("--drums", type=lambda x: x.lower() in ['true', '1', 'yes'], required=True, help="Boolean flag for drums (True/False)")
    parser.add_argument("--end_infilling", type=lambda x: x.lower() in ['true', '1', 'yes'], required=True, help="Boolean flag for infilling end")

    # Parse arguments
    args = parser.parse_args()

    NUM_BARS_TO_INFILL = args.num_bars_infilling
    CONTEXT_SIZE = args.context
    DRUM_GENERATION = args.drums
    END_INFILLING = args.end_infilling

    tokenizer = MMM(params=Path(__file__).parent.parent / "runs" / "tokenizer.json")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

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

    i = 0
    while i < args.num_generations:
        midi_file = random.choice(MIDI_PATHS)
        if test_generate(tokenizer, model, gen_config, midi_file):
            i += 1