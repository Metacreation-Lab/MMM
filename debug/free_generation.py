from pathlib import Path

import miditok
import torch
from miditok import MMM, TokSequence
from symusic import Score
from torch import LongTensor
from transformers import AutoModelForCausalLM, GenerationConfig, LogitsProcessor

from scripts.utils.constants import NUM_BEAMS, TEMPERATURE_SAMPLING, REPETITION_PENALTY, TOP_K, TOP_P, EPSILON_CUTOFF, \
    ETA_CUTOFF, MAX_NEW_TOKENS, MAX_LENGTH

MODEL_PATH = Path(__file__).parent.parent / "runs" /"models"/"MISTRAL_87000"
"""
class StopLogitsProcessor(LogitsProcessor):
    def __init__(
            self,
            eos_token_id: int,
            tokenizer: miditok.MusicTokenizer,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer

    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input
"""
if __name__ == "__main__":
    tokenizer = MMM(params=Path(__file__).parent.parent / "runs" / "tokenizer.json")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    gen_config = GenerationConfig(
        num_beams=NUM_BEAMS,
        temperature=1.0,
        repetition_penalty=REPETITION_PENALTY,
        top_k=TOP_K,
        top_p=TOP_P,
        epsilon_cutoff=EPSILON_CUTOFF,
        eta_cutoff=ETA_CUTOFF,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=MAX_LENGTH,
        do_sample = True
    )

    for i in range (20):
        output_toksequence = TokSequence(are_ids_encoded=True)

        output_toksequence.ids = model.generate(LongTensor([[tokenizer.vocab["Infill_Track"]]]), gen_config)[0].tolist()

        output_toksequence.ids[0] = tokenizer.vocab["Track_Start"]
        output_toksequence.ids[0] = tokenizer.vocab["Track_Start"]

        #output_toksequence.tokens = tokenizer._ids_to_tokens(output_toksequence.ids)

        tokenizer.decode_token_ids(output_toksequence)

        if output_toksequence.ids[-1] != tokenizer.vocab["Track_End"]:
            output_toksequence.ids.append(tokenizer.vocab["Track_End"])
            output_toksequence.tokens.append("Track_End")

        score : Score = tokenizer._tokens_to_score(output_toksequence)

        score.dump_midi(f"output_mistral_{i}.mid")

        print(f"Done generation {i}")

