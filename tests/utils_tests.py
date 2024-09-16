"""Utils for tests."""

from pathlib import Path

import torch
from miditok import MMM
from symusic import Score
from torch import LongTensor

import numpy as np

HERE = Path(__file__).parent
MIDI_PATHS = list((HERE / "midis").glob("**/*.mid"))
MIDI_PATH = HERE / "midis" / "I Gotta Feeling.mid"


class DummyModel:
    """Dummy model used to test generation code."""

    def __init__(self, tokenizer: MMM) -> None:
        self.tokenizer = tokenizer
        self._bar_infilling_token_id = self.tokenizer["Infill_Bar"]
        self._bar_none_token_id = self.tokenizer["Bar_None"]
        score = Score(MIDI_PATH)
        # TODO extract the desired portion from the score tokenize it and save attribute

        kwargs={}
        if type(self.tokenizer).__name__ == "MMM":
            kwargs["concatenate_track_sequences"] = False

        tokens = tokenizer(score, **kwargs)

        tokenizer.decode_token_ids(tokens[0])
        bar_none_tokens = np.where(np.array(tokens[0].ids) == self._bar_none_token_id)[0]
        # Get 4 bars of content for the infilling part
        self.infill_generated_tokens = tokens[0].ids[bar_none_tokens[0]:bar_none_tokens[4]]
        self.infill_generated_tokens.append(tokenizer.vocab["FillBar_End"])

        self.track_generated_content = tokens[0].ids


    def generate(self, token_ids: LongTensor) -> LongTensor:
        """
        Generate method.

        :param token_ids: token ids.
        :return: supposedly generated token ids.
        """
        # If we have to infill bars
        if torch.any(token_ids == self._bar_infilling_token_id):
            generated_tokens = self.infill_generated_tokens
            token_ids = torch.cat(
                (
                    token_ids,
                    LongTensor(generated_tokens).reshape(1, len(generated_tokens)),
                ),
                dim=1,
            )

        # otherwise, we are generating a new track
        else:
            generated_tokens = self.track_generated_content
            token_ids = torch.cat(
                (
                    token_ids,
                    LongTensor(generated_tokens).reshape(1, len(generated_tokens)),
                ),
                dim=1,
            )

        return token_ids
