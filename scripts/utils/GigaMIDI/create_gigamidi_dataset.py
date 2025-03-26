#!/usr/bin/python3 python

"""Script to create the WebDataset for the GigaMIDI dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import datasets
from miditok.constants import MIDI_FILES_EXTENSIONS, SCORE_LOADING_EXCEPTION
from symusic import Score
from symusic.core import TextMetaSecond
from tqdm import tqdm

MAX_NUM_ENTRIES_PER_SHARD = 50000
SUBSET_PATHS = {
    "all-instruments-with-drums": "drums+music",
    "drums-only": "drums",
    "no-drums": "music",
}
SPLITS = ["test", "train", "validation"]


# Specify the types of values of each column of the dataset
# This is required for the loading method (from_generator)
DATASET_FEATURES = {
    "md5": datasets.Value("string"),
    "music": datasets.Value("binary"),
    "instrument_category": datasets.Value("string"),
    "sid_matches": datasets.Sequence(
        {"sid": datasets.Value("string"), "score": datasets.Value("float16")}
    ),
    "mbid_matches": datasets.Sequence(
        {
            "sid": datasets.Value("string"),
            "mbids": datasets.Sequence(datasets.Value("string")),
        }
    ),
    "artist_scraped": datasets.Value("string"),
    "title_scraped": datasets.Value("string"),
    "genres_curated": datasets.Value("string"),
    "genres_scraped": datasets.Sequence(datasets.Value("string")),
    "genres_discogs": datasets.Sequence(
        {"genre": datasets.Value("string"), "count": datasets.Value("int16")}
    ),
    "genres_tagtraum": datasets.Sequence(
        {"genre": datasets.Value("string"), "count": datasets.Value("int16")}
    ),
    "genres_lastfm": datasets.Sequence(
        {"genre": datasets.Value("string"), "count": datasets.Value("int16")}
    ),
    "median_metric_depth": datasets.Sequence(datasets.Value("int16")),
    "loops": datasets.Sequence(
        {
            "track_idx": datasets.Value("uint16"),
            "start_tick": datasets.Value("uint32"),
            "end_tick": datasets.Value("uint32"),
        }
    ),
}


def path_data_directory_local_fs() -> Path:
    """
    Return the path to the root data directory on the local file system.

    :return: path to the root data directory.
    """
    return Path.home() / "git" / "data"


def create_parquet_gigamidi(main_data_dir_path: Path, dataset_version: str) -> None:
    """
    Create the WebDataset shard for the GigaMIDI dataset.

    :param main_data_dir_path: path of the directory containing the datasets.
    :param dataset_version: version of the dataset, used as subset name.
    """
    dataset_path = main_data_dir_path / "GigaMIDI_original"

    # Load metadata
    md5_sid_matches_scores = {}
    with (
        main_data_dir_path / "MMD_METADATA" / "MMD_audio_matches.tsv"
    ).open() as matches_file:
        matches_file.seek(0)
        next(matches_file)  # first line skipped
        for line in tqdm(matches_file, desc="Reading MMD match file"):
            midi_md5, score, audio_sid = line.split()
            if midi_md5 not in md5_sid_matches_scores:
                md5_sid_matches_scores[midi_md5] = []
            md5_sid_matches_scores[midi_md5].append((audio_sid, float(score)))
    sid_to_mbid = json.load(
        (main_data_dir_path / "MMD_METADATA" / "MMD_sid_to_mbid.json").open()
    )

    md5_genres = {}
    with (
        main_data_dir_path / "MMD_METADATA" / "MMD_audio_matched_genre.jsonl"
    ).open() as file:
        for row in tqdm(file, desc="Reading genres MMD metadata"):
            entry = json.loads(row)
            md5 = entry.pop("md5")
            md5_genres[md5] = entry
    md5_genres_scraped = {}
    with (
        main_data_dir_path / "MMD_METADATA" / "MMD_scraped_genre.jsonl"
    ).open() as file:
        for row in tqdm(file, desc="Reading scraped genres MMD metadata"):
            entry = json.loads(row)
            genres = []
            for genre_list in entry["genre"]:
                genres += genre_list
            md5_genres_scraped[entry["md5"]] = genres
    md5_genres_curated = {}
    with (dataset_path / "Curated_Style_Data_MD5_GigaMIDI.csv").open() as file:
        reader = csv.reader(file)
        next(reader)  # skipping header
        for row in tqdm(reader, desc="Reading curated genres MMD metadata"):
            md5_genres_curated[row[-1]] = row[-2]
    md5_artist_title_scraped = {}
    with (
        main_data_dir_path / "MMD_METADATA" / "MMD_scraped_title_artist.jsonl"
    ).open() as file:
        for row in tqdm(file, desc="Reading scraped titles/artists MMD metadata"):
            entry = json.loads(row)
            md5_artist_title_scraped[entry["md5"]] = entry["title_artist"][0]
    md5_expressive = {}
    with (
        dataset_path / "Expressive_Performance_Detection_NOMML_gigamidi_tismir.csv"
    ).open() as file:
        file.seek(0)
        next(file)  # skipping first row (header)
        for row in tqdm(file, desc="Reading expressiveness metadata"):
            parts = row.split(",")
            md5 = parts[0].split("/")[-1].split(".")[0]
            if md5 not in md5_expressive:
                md5_expressive[md5] = []
            md5_expressive[md5].append(int(parts[5]))
    md5_loop = {}
    with (
        dataset_path
        / "GigaMIDI-combined-non-expressive-loop-data"
        / "GigaMIDI-combined-non-expressive-loop-dataset.csv"
    ).open() as file:
        file.seek(0)
        next(file)  # skipping first row (header)
        for row in tqdm(file, desc="Reading loops metadata"):
            parts = row.split(",")
            md5 = parts[0].split("/")[-1].split(".")[0]
            if md5 not in md5_loop:
                md5_loop[md5] = []
            md5_loop[md5].append((parts[1], *parts[4:6]))

    def gen(split_: str) -> dict:
        for subset_name, subset_path_name in SUBSET_PATHS.items():
            subset_path = dataset_path / split_ / subset_path_name
            midi_files_paths = []
            for ext in MIDI_FILES_EXTENSIONS:
                midi_files_paths += subset_path.glob(f"*{ext}")
            for file_path in midi_files_paths:
                md5 = file_path.stem
                with file_path.open("rb") as file:
                    midi_file_bytes = file.read()

                # Get metadata if existing
                metadata_ = {}

                sid_matches = md5_sid_matches_scores.get(md5)
                if sid_matches:
                    metadata_["sid_matches"] = sid_matches
                    metadata_["mbid_matches"] = []
                    for sid, _ in sid_matches:
                        mbids = sid_to_mbid.get(sid, None)
                        if mbids:
                            metadata_["mbid_matches"].append([sid, mbids])

                title_artist = md5_artist_title_scraped.get(md5)
                if title_artist:
                    (
                        metadata_["title_scraped"],
                        metadata_["artist_scraped"],
                    ) = title_artist
                genres_scraped = md5_genres_scraped.get(md5)
                if genres_scraped:
                    metadata_["genres_scraped"] = genres_scraped
                genres = md5_genres.get(md5)
                if genres:
                    for key, val in genres.items():
                        metadata_[f"genres_{key.split('_')[1]}"] = val
                interpreted_scores = md5_expressive.get(md5)
                if interpreted_scores:
                    metadata_["median_metric_depth"] = interpreted_scores
                loops = md5_loop.get(md5)
                if loops:
                    try:
                        score = Score(file_path, ttype="second")
                        score.markers = []  # delete all existing ones for simplicity
                        for track_id, start_sec, end_sec in loops:
                            score.markers.append(
                                TextMetaSecond(
                                    float(start_sec), f"#LOOP_{int(track_id)}_on"
                                )
                            )
                            score.markers.append(
                                TextMetaSecond(
                                    float(end_sec), f"#LOOP_{int(track_id)}_off"
                                )
                            )
                        score = score.to(ttype="tick")
                        loops_ticks = []
                        loops_on = {}  # {track_id: start_tick}
                        for marker in score.markers:
                            _, track_id, state = marker.text.split("_")
                            if state == "on":
                                loops_on[int(track_id)] = marker.time
                            else:
                                loops_ticks.append(
                                    (
                                        int(track_id),
                                        loops_on.pop(int(track_id)),
                                        marker.time,
                                    )
                                )
                        metadata_["loops"] = loops_ticks
                    except SCORE_LOADING_EXCEPTION:
                        pass  # no loops saved for corrupted files

                yield {
                    "md5": md5,
                    "music": midi_file_bytes,
                    "instrument_category": subset_name,
                    "sid_matches": [
                        {"sid": sid, "score": score}
                        for sid, score in metadata_.get("sid_matches", [])
                    ],
                    "mbid_matches": [
                        {"sid": sid, "mbids": mbids}
                        for sid, mbids in metadata_.get("mbid_matches", [])
                    ],
                    "artist_scraped": metadata_.get("artist_scraped"),
                    "title_scraped": metadata_.get("title_scraped"),
                    "genres_curated": md5_genres_curated.get(md5),
                    "genres_scraped": metadata_.get("genres_scraped"),
                    "genres_discogs": [
                        {"genre": genre, "count": count}
                        for genre, count in metadata_.get("genres_discogs", {}).items()
                    ],
                    "genres_tagtraum": [
                        {"genre": genre, "count": count}
                        for genre, count in metadata_.get("genres_tagtraum", {}).items()
                    ],
                    "genres_lastfm": [
                        {"genre": genre, "count": count}
                        for genre, count in metadata_.get("genres_lastfm", {}).items()
                    ],
                    "median_metric_depth": metadata_.get("median_metric_depth"),
                    "loops": [
                        {
                            "track_idx": track_idx,
                            "start_tick": start_tick,
                            "end_tick": end_tick,
                        }
                        for track_idx, start_tick, end_tick in metadata_.get(
                            "loops", []
                        )
                    ],
                }

    # Loading the dataset from the generator above
    for split in SPLITS:
        dataset = datasets.Dataset.from_generator(
            gen,
            features=datasets.Features(DATASET_FEATURES),
            gen_kwargs={"split_": split},
            split=split,
        )

        # Convert the Dataset object to parquet and upload it to the Hugging Face hub
        dataset.to_parquet(
            path_data_directory_local_fs()
            / "GigaMIDI"
            / dataset_version
            / f"{split}.parquet"
        )


def load_dataset_from_generator(
    dataset_path: Path, num_files_limit: int = 100
) -> datasets.Dataset:
    """
    Load the dataset.

    :param dataset_path: path of the directory containing the datasets.
    :param num_files_limit: maximum number of entries/files to retain.
    :return: dataset.
    """
    files_paths = list(dataset_path.glob("**/*.mid"))[:num_files_limit]
    return datasets.Dataset.from_dict({"music": [str(path_) for path_ in files_paths]})


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Dataset creation script")
    parser.add_argument(
        "--hf-repo-name", type=str, required=False, default="Metacreation/GigaMIDI"
    )
    parser.add_argument("--hf-token", type=str, required=False, default=None)
    args = vars(parser.parse_args())
    DATASET_VERSION = "v1.0.0"

    """
    # Example of how to write back midi files from the parquet files
    dataset = datasets.load_dataset(args["hf_repo_name"], token=args["hf_token"])
    for split_name, split_subset in dataset.items():
        split_dir_path = Path(split_name)
        split_dir_path.mkdir(exist_ok=True)
        for row in split_subset:
            midi_bytes = row["music"]
            with (split_dir_path / f"row['md5'].mid").open("wb") as file:
                file.write(midi_bytes)"""

    # Load all the raw data (MIDI files, csv and json metadata)
    create_parquet_gigamidi(path_data_directory_local_fs(), DATASET_VERSION)

    """from datasets import load_dataset

    # dataset_ = load_dataset(
    #    str(path_data_directory_local_fs() / "GigaMIDI"),
    #    "no-drums",
    #    subsets=["no-drums", "all-instruments-with-drums"],
    #    trust_remote_code=True,
    # )
    dataset_ = load_dataset(
        args["hf_repo_name"],
        "no-drums",
        token=args["hf_token"],
    )
    data = dataset_["train"]
    for i in range(7):
        sample = data[i]
        score = Score.from_midi(sample["music"])
        f = 0

    import requests

    headers = {"Authorization": f"Bearer {args['hf_token']}"}
    API_URL = "https://datasets-server.huggingface.co/is-valid?dataset=Metacreation/GigaMIDI"


    def query():
        response = requests.get(API_URL, headers=headers)
        return response.json()


    data = query()
    t = 0"""
