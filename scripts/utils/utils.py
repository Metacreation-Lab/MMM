"""Utils module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from symusic import Score, Track
    from symusic.core import TextMetaTickList


def path_data_directory_local_fs() -> Path:
    """
    Return the path to the root data directory on the local file system.

    :return: path to the root data directory.
    """
    #return Path(os.getenv("SLURM_TMPDIR", os.getenv("SCRATCH")), "data")
    return Path(os.getenv("SCRATCH"), "data")

def symusic_track_to_struct(track: Track) -> dict[str, np.ndarray]:
    """
    Convert a Symusic Track into a SOA dictionary.

    :param track: track to convert.
    :return: SOA.
    """
    return {
        "notes": track.notes.numpy(),
        "controls": track.controls.numpy(),
        "pitch_bends": track.pitch_bends.numpy(),
    }


def symusic_score_to_struct(score: Score) -> dict:
    """
    Convert a Symusic Score into a SOA dictionary.

    :param score: score to convert.
    :return: SOA.
    """
    return {
        "time_signatures": score.time_signatures.numpy(),
        "key_signatures": score.key_signatures.numpy(),
        "tempos": score.tempos.numpy(),
        "markers": _symusic_text_soa(score.markers),
        "lyrics": _symusic_text_soa(score.lyrics),
        "tracks": [symusic_track_to_struct(track) for track in score.tracks],
    }


def _symusic_text_soa(markers: TextMetaTickList) -> dict[str, list[str] | np.ndarray]:
    times, texts = [], []
    for marker in markers:
        times.append(marker.time)
        texts.append(marker.text)
    return {"time": np.array(times), "text": texts}
