from pathlib import Path

from miditok import MMM

from dummy_dataset import DummyDataset
from scripts.utils.constants import TRACKS_SELECTION_RANDOM_RATIO_RANGE, RATIO_BAR_INFILLING, \
    RATIOS_RANGE_BAR_INFILLING_DURATION, ACS_RANDOM_RATIO_RANGE, TRACKS_IDX_RANDOM_RATIO_RANGE, \
    BARS_IDX_RANDOM_RATIO_RANGE, MAX_SEQ_LEN, DATA_AUGMENTATION_OFFSETS

if __name__ == "__main__":

    tokenizer = MMM(params=Path(__file__).parent.parent / "runs" / "tokenizer.json")

    dataset = DummyDataset(
            tokenizer,
            MAX_SEQ_LEN,
            TRACKS_SELECTION_RANDOM_RATIO_RANGE,
            DATA_AUGMENTATION_OFFSETS,
            RATIO_BAR_INFILLING,
            RATIOS_RANGE_BAR_INFILLING_DURATION,
            ac_random_ratio_range=ACS_RANDOM_RATIO_RANGE,
            ac_tracks_random_ratio_range=TRACKS_IDX_RANDOM_RATIO_RANGE,
            ac_bars_random_ratio_range=BARS_IDX_RANDOM_RATIO_RANGE,
            seq2seq=False
    )

    dataset.debug_training_sequences()

