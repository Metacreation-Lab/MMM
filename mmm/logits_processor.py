"""Definition of logits processor used for generation."""

import time

import miditok
import numpy as np
import torch
from miditok import TokSequence
from transformers import LogitsProcessor

import time
import torch
from transformers import LogitsProcessor

class TrackLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        track_start_token_id: int,
        track_end_token_id: int,
        bar_start_token_id: int,
        eos_token_id: int,
        bars_to_generate: int,
    ) -> None:
        self.track_start_token_id = track_start_token_id
        self.track_end_token_id = track_end_token_id
        self.bar_start_token_id = bar_start_token_id
        self.eos_token_id = eos_token_id
        self.bars_to_generate = bars_to_generate
        self.current_num_bars = 0
        self.total_time = 0.0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Called at each generation step to modify logits.
        Enforces:
          - No re-generation of track_start_token
          - Stop generating new bars after bars_to_generate reached
          - Force EOS after track_end_token
        """

        start_time = time.time()

        # Disable track start token always
        scores[0, self.track_start_token_id] = -1e9

        # Get last generated token
        last_token = input_ids[0, -1].item()

        # If last token was a bar start → increment counter
        if last_token == self.bar_start_token_id:
            self.current_num_bars += 1

        # If reached bar limit → disallow further bar start tokens
        if self.current_num_bars == self.bars_to_generate + 1:
            scores[0,:] = -1e9
            scores[0, self.track_end_token_id] = 1e9
        else:
            scores[0, self.track_end_token_id] = -1e9

        # If track end token → force next token to be EOS
        if last_token == self.track_end_token_id:
            scores[0, :] = -1e9
            scores[0, self.eos_token_id] = 1e9

        self.total_time += time.time() - start_time
        return scores

import time
import torch
from transformers import LogitsProcessor

class InfillLogitsProcessor(LogitsProcessor):
    """
    Custom LogitsProcessor for bar infilling.

    Enforces:
      - Never sample FillBar_Start again.
      - Count bars and stop when num_bars_to_generate is reached.
      - Force EOS token immediately after FillBar_End.
    """

    def __init__(
        self,
        fillbar_start_token_id: int,
        fillbar_end_token_id: int,
        bar_start_token_id: int,
        eos_token_id: int,
        num_bars_to_generate: int,
    ) -> None:
        self.fillbar_start_token_id = fillbar_start_token_id
        self.fillbar_end_token_id = fillbar_end_token_id
        self.bar_start_token_id = bar_start_token_id
        self.eos_token_id = eos_token_id
        self.num_bars_to_generate = num_bars_to_generate

        self.current_num_bars = 0
        self.total_time = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Logic:
          - Mask FillBar_Start always.
          - Increment bar counter whenever Bar_Start is generated.
          - When enough bars generated, mask Bar_Start.
          - If FillBar_End generated, force EOS token next.
        """
        start_time = time.time()

        # Always prevent FillBar_Start from being generated again
        scores[0, self.fillbar_start_token_id] = -1e9

        # Get last generated token (batch size = 1)
        last_token = input_ids[0, -1].item()

        # Count bars
        if last_token == self.bar_start_token_id:
            self.current_num_bars += 1

        # When bar limit reached → mask further bar starts
        if self.current_num_bars == self.num_bars_to_generate + 1:
            scores[0, :] = -1e9
            scores[0, self.fillbar_end_token_id] = 1e9
        else:
            scores[0, self.fillbar_end_token_id] = -1e9

        # If FillBar_End generated → force EOS next
        if last_token == self.fillbar_end_token_id:
            scores[0, :] = -1e9
            scores[0, self.eos_token_id] = 1e9

        self.total_time += time.time() - start_time
        return scores

