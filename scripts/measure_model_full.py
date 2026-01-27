import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from collections import defaultdict

from symusic import Score
from miditok import MusicTokenizer
from miditok.utils import get_bars_ticks
from mmm import generate, InferenceConfig
from transformers.trainer_utils import set_seed

from utils.baselines import baselines

TIMERS = ["PerfCounterTimer", "ProcessTimeTimer"]

def stats(x):
    x = np.asarray(x)
    return {
        "mean": float(x.mean()),
        "var": float(x.var()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max())
    }

def extract_metrics(configs):
    metrics = defaultdict(list)

    for cfg in configs:
        input_len = cfg["input_length"]
        gen_len = cfg["generated_tokens_length"]

        for t in TIMERS:
            pre = cfg[f"{t}.preprocessing"]
            inf = cfg[f"{t}.inference"]
            post = cfg[f"{t}.postprocessing"]

            total = pre + inf + post
            tps = gen_len / inf
            tps_tot = gen_len / total
            spt = inf / gen_len
            tps_norm = tps / input_len
            pre_per_input = pre / input_len

            metrics[f"{t}.inference_time"].append(inf)
            metrics[f"{t}.total_time"].append(total)
            metrics[f"{t}.tokens_per_sec"].append(tps)
            metrics[f"{t}.tokens_per_sec_total"].append(tps_tot)
            metrics[f"{t}.sec_per_token"].append(spt)
            metrics[f"{t}.tokens_per_sec_per_input"].append(tps_norm)
            metrics[f"{t}.pre_sec_per_input"].append(pre_per_input)
            metrics[f"{t}.inference_fraction"].append(inf / total)

    return metrics

# -----------------------------------------------------------
# Generate a random inference configuration for a given MIDI
# -----------------------------------------------------------

def random_inference_config(score: Score) -> InferenceConfig:
    """
    Randomly generate either:
    - bar-infilling config, or
    - new-track generation config
    based on the structure of the score.
    """

    num_tracks = len(score.tracks)

    # If only 1 track → must do bar infilling
    do_bar = num_tracks == 1 or random.random() < 0.5

    bars_ticks = get_bars_ticks(score)
    n_bars = len(bars_ticks)

    # ---- BAR INFILLING ----
    if do_bar:
        track_idx = random.randrange(num_tracks)

        # pick a random bar section
        if n_bars <= 1:
            # fallback: generate 1 bar
            bar_start = 0
            bar_end = 1
        else:
            section_len = min(max(1, int(random.random() * n_bars)), 8)
            bar_start = random.randrange(0, max(1, n_bars - section_len - 1))
            bar_end = bar_start + section_len

        bars_to_generate = {
            track_idx: [
                (bar_start, bar_end, [])
            ]
        }

        config = InferenceConfig(
            context_length=4,
            bars_to_generate=bars_to_generate,
            new_tracks=[],
        )
        return config

    else:

        # ---- NEW TRACK GENERATION ----
        new_tracks = [
            (1, [])
        ]

    config = InferenceConfig(
        context_length=4,
        bars_to_generate=[],
        new_tracks=new_tracks
    )
    return config


# -----------------------------------------------------------
# Run inference on a list of MIDI files
# -----------------------------------------------------------

def run_inference_on_list(
    model,
    tokenizer,
    midi_files: List[Path],
    num_passes: int = 1
) -> List[Dict[str, Any]]:
    """
    For each MIDI file:
    - Loads score
    - Generates random config
    - Calls generate()
    - Extracts time metrics
    """

    results = []

    for midi_path in midi_files:
        print(f"\n--- Processing {midi_path.name} ---")

        # Load MIDI → Score
        try:
            score = Score(Path(midi_path))
        except Exception as e:
            print(f"  ERROR loading MIDI: {e}")
            continue

        for i in range(num_passes):

            # generate random inference config
            config = random_inference_config(score)

            # call model
            try:
                out = generate(
                    model,
                    tokenizer,
                    config,
                    midi_path,
                    {
                        "generation_config": model.generation_config
                    }
                )
            except Exception as e:
                print(f"Failed generation: {e}")
                continue

            # extract time metric
            tm = out.get("time_metrics", None)

            results.append(tm)

    return extract_metrics(results)


# -----------------------------------------------------------
# Main execution
# -----------------------------------------------------------

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_passes", type=int, default = 1)
    parser.add_argument("--midi_list", type=str, required=True,
                        help="Text file containing one MIDI path per line")
    parser.add_argument("--output", type=str, default="inference_results.json")

    args = parser.parse_args()

    # load model
    if args.model not in baselines:
        raise ValueError(f"Model {args.model} not in baselines.")

    baseline = baselines[args.model]
    model = baseline.create_model(args.ckpt)
    tokenizer = baseline.tokenizer

    print("Loaded model:", args.model)
    print("Checkpoint:", args.ckpt)

    # read list of MIDI files
    midi_files = []
    with open(args.midi_list, "r") as f:
        for line in f:
            p = Path(line.strip())
            if p.exists():
                midi_files.append(p)
            else:
                print(f"Warning: MIDI not found ({p})")

    print(f"\nFound {len(midi_files)} MIDI files.")

    # run all
    results = run_inference_on_list(model, tokenizer, midi_files, num_passes=args.num_passes)
    summary = {k: stats(v) for k, v in results.items()}

    # save JSON summary
    out_json = Path(args.output)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {args.output}")
