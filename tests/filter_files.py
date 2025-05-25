"""
Removes non valid MIDI files.

Removes non valid MIDI files. A file is considered valid if it's not empty and
if the last note's ticks value is bigger than last bar's ticks value.
Some MIDI files tokenized with miditok contain an inconsistent number of bars
compared to the notes within the file. These files are filtered out.
"""

from pathlib import Path

from miditok.utils.utils import get_bars_ticks
from symusic import Score

from tests.utils_tests import MIDI_PATHS


def check_file(midi_file: Path):
    score = Score(midi_file)
    bars_ticks = get_bars_ticks(score, only_notes_onsets=True)
    for track in score.tracks:
        if len(track.notes) == 0:
            return False
        if track.notes.numpy()["time"][-1] < bars_ticks[-1]:
            return False
    return True

if __name__ == "__main__":
    path = Path("valid_files.txt")  # Create a Path object
    with path.open("w") as file:
        #while i < args.num_generations:
        #    midi_file = random.choice(MIDI_PATHS)
        i = 0
        for midi_file in MIDI_PATHS:
            if check_file(midi_file):
                file.write(f"{midi_file.name}\n")
                i += 1
            else:
                print(f"midi file {midi_file} is not valid")
        file.write(f"number of files: {i}\n")
