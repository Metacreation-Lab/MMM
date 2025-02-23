from __future__ import annotations

from random import choice, random, sample, uniform
from typing import TYPE_CHECKING

import numpy as np
from miditok import TokSequence
from miditok.attribute_controls import BarAttributeControl
from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.data_augmentation.data_augmentation import (
    _filter_offset_tuples_to_score,
    augment_score,
)
from miditok.pytorch_data import DatasetMIDI
from miditok.utils import get_bars_ticks
from symusic import Score
from torch import LongTensor, isin

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from miditok import MMM

def concat_tokseq(sequences: list[TokSequence]) -> TokSequence:
    """
    Concatenate a sequence of :class:`miditok.TokSequence`.

    :param sequences: :class:`miditok.TokSequence`s to concatenate.
    :return: the concatenated ``sequences``.
    """
    tokseq = sequences.pop(0)
    for seq in sequences:
        tokseq += seq
    return tokseq


class DummyDataset(DatasetMIDI):
    def __init__(
        self,
        tokenizer: MMM,
        max_seq_len: int,
        ratio_random_tracks_range: tuple[float, float],
        data_augmentation_offsets: tuple[int, int, int],
        bar_fill_ratio: float,
        max_bar_infilling_length: int,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        ac_random_ratio_range: tuple[float, float] | None = None,
        ac_tracks_random_ratio_range: tuple[float, float] | None = None,
        ac_bars_random_ratio_range: tuple[float, float] | None = None,
        func_to_get_labels: Callable[
            [Score, TokSequence | list[TokSequence], Path],
            int | list[int] | LongTensor,
        ]
        | None = None,
        sample_key_name: str = "input_ids",
        decoder_key_name: str = "decoder_input_ids",
        labels_key_name: str = "labels",
        seq2seq: bool = False,
    ) -> None:
        self.ratio_random_tracks_range = ratio_random_tracks_range
        pitch_offsets = data_augmentation_offsets[0]
        self.pitch_offsets = list(range(-pitch_offsets, pitch_offsets + 1))
        velocity_offsets = data_augmentation_offsets[1]
        self.velocity_offsets = list(range(-velocity_offsets, velocity_offsets + 1))
        duration_offsets = data_augmentation_offsets[2]
        self.duration_offsets = list(range(-duration_offsets, duration_offsets + 1))
        self.bar_fill_ratio = bar_fill_ratio
        self.max_bar_infilling_length = max_bar_infilling_length
        self.ac_random_ratio_range = ac_random_ratio_range
        self.seq2seq = seq2seq

        # Infill tokens, set as attribute here to avoid to access to vocab dic
        self._infill_bar_token_id = tokenizer.vocab["Infill_Bar"]
        self._infill_bar_start_token_id = tokenizer.vocab["FillBar_Start"]
        self._infill_bar_end_token_id = tokenizer.vocab["FillBar_End"]
        self._infill_track_token_id = tokenizer.vocab["Infill_Track"]
        self._track_start_token_id = tokenizer.vocab["Track_Start"]
        self._track_end_token_id = tokenizer.vocab["Track_End"]
        self._bar_token_id = tokenizer.vocab["Bar_None"]
        self._bar_token_byte = tokenizer._vocab_base_id_to_byte[self._bar_token_id]

        # Token ids that should be masked from the "labels" entry so that the loss is
        # not computed other them. Doing so, the model will not be trained to predict
        # them.
        self._token_ids_no_loss = LongTensor(
            [
                self._infill_bar_token_id,
                self._infill_bar_start_token_id,
                *[
                    tokenizer.vocab[token]
                    for ac in tokenizer.attribute_controls
                    for token in ac.tokens
                ],
            ],
        )

        self._ac_tracks, self._ac_bars = [], []
        for i, ac in enumerate(tokenizer.attribute_controls):
            if isinstance(ac, BarAttributeControl):
                self._ac_bars.append(i)
            else:
                self._ac_tracks.append(i)

        max_seq_len -= sum([1 for t in [bos_token_id, eos_token_id] if t is not None])
        super().__init__(
            [],
            tokenizer,
            max_seq_len,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pre_tokenize=False,  # can't pre-tokenize with the random selections
            ac_tracks_random_ratio_range=ac_tracks_random_ratio_range,
            ac_bars_random_ratio_range=ac_bars_random_ratio_range,
            func_to_get_labels=func_to_get_labels,
            sample_key_name=sample_key_name,
            labels_key_name=labels_key_name,
        )
        self.decoder_key_name = decoder_key_name

    def __len__(self):
        return 0  # No data

    def __getitem__(self, idx):
        raise IndexError("This dataset is empty.")

    def debug_training_sequences(self):
        try:
            score = Score("C:\\Users\\rizzo\\Desktop\\TESI\\MMM\\tests\\midis\\I Gotta Feeling.mid")
        except SCORE_LOADING_EXCEPTION:
            item = {self.sample_key_name: None, self.labels_key_name: None}
            if self.seq2seq:
                item[self.decoder_key_name] = None
            return item

        # Tokenize the score
        try:
            tseq, decoder_input_ids = self._new_tokenize_score(score)
        except IndexError:
            item = {self.sample_key_name: None, self.labels_key_name: None}
            if self.seq2seq:
                item[self.decoder_key_name] = None
            return item
        if tseq is None:
            item = {self.sample_key_name: None, self.labels_key_name: None}
            if self.seq2seq:
                item[self.decoder_key_name] = None
            return item

        # If not one_token_stream, we only take the first track/sequence
        token_ids = tseq.ids if self.tokenizer.one_token_stream else tseq[0].ids

        item = {self.sample_key_name: LongTensor(token_ids)}

        if self.seq2seq:
            item[self.decoder_key_name] = LongTensor(decoder_input_ids.ids)
            item[self.labels_key_name] = item[self.decoder_key_name].clone()
        else:
            item[self.labels_key_name] = item[self.sample_key_name].clone()

        # Set ids of elements to discard to -100
        idx_tokens_to_discard = isin(
            item[self.labels_key_name], self._token_ids_no_loss
        )
        item[self.labels_key_name][idx_tokens_to_discard] = -100
        return item

    def _tokenize_score(
            self, score: Score
    ) -> tuple[TokSequence, TokSequence | None] | tuple[None, None]:
        # Delete unused elements in order to compute the bar ticks that we will use,
        # otherwise if unused elements are at the end of the score they can give us
        # bar ticks exceeding the tokens
        score.markers = []
        score.key_signatures = []
        for track in score.tracks:
            track.controls = []
            track.lyrics = []
            if not self.tokenizer.config.use_sustain_pedals:
                track.pedals = []
            if not self.tokenizer.config.use_pitch_bends:
                track.pitch_bends = []

        # Select k tracks (shuffled)
        num_tracks_to_keep = max(
            1, round(len(score.tracks) * uniform(*self.ratio_random_tracks_range))
        )
        bars_ticks = np.array(get_bars_ticks(score))  # always at least two bars
        tracks_idx_ok = [
            idx
            for idx in range(len(score.tracks))
            if len(score.tracks[idx].notes) > 0
               and score.tracks[idx].notes[-1].time > bars_ticks[1]
        ]  # always at least one
        # Initialize the list to store the selected indices

        selected_tracks = [] # Just for debugging purposes

        # Select random indices for the tracks
        score.tracks = [
            score.tracks[idx]
            for idx in sample(
                tracks_idx_ok, k=min(num_tracks_to_keep, len(tracks_idx_ok))
            )
            if selected_tracks.append(idx) or True  # Save each selected index
        ]

        # Remove time signatures and tempos occurring after the start of the last note
        max_note_time = 0
        for track in score.tracks:
            if len(track.notes) > 0:
                max_note_time = max(max_note_time, track.notes[-1].time)
        for ti in reversed(range(len(score.time_signatures))):
            if score.time_signatures[ti].time > max_note_time:
                del score.time_signatures[ti]
            else:
                break
        for ti in reversed(range(len(score.tempos))):
            if score.tempos[ti].time > max_note_time:
                del score.tempos[ti]
            else:
                break

        # Augment and preprocess the music.
        # We need to preprocess it here as we require it preprocessed to select the
        # bars and tracks indexes for attribute controls before tokenizing.
        #score = self.augment_and_preprocess_score(score)
        if len(score.tracks) == 0:
            return None, None

        # Set bar infilling or new track gen + their indexes
        # If there is only one track, we do bar infilling as new track gen is not
        # possible in seq2seq.
        bar_infilling = True #len(score.tracks) == 1 or random() < self.bar_fill_ratio
        track_infilling_idx = None
        bar_infilling_start_idx, bar_infilling_end_idx, infill_section_num_bars = None, None, None

        if bar_infilling:
            track_infilling_idx = choice(list(range(len(score.tracks))))
            # ac_indexes contains random bar acs only for the section to infill
            bars_ticks = np.array(get_bars_ticks(score))  # need to recompute (resample)
            bars_ticks = bars_ticks[  # remove bars ticks after end of track
                np.where(bars_ticks <= score.tracks[track_infilling_idx].notes[-1].time)
            ]

            # Generate a random number between 1 and 20
            #infill_section_num_bars = random.randint(1, self.max_bar_infilling_length)

            #if infill_section_num_bars > len(bars_ticks):
            #    infill_section_num_bars = random.randint(1, 4)

            infill_section_num_bars = max(
                1,
                round(
                    len(bars_ticks) * uniform(0.1,0.4)
                ),
            )

            # TODO: save the number of infilled bars

            if infill_section_num_bars == len(bars_ticks):
                # Special case where the portion to mask == the whole music
                bar_infilling_start_idx = 0
            else:
                bar_infilling_start_idx = choice(
                    list(range(len(bars_ticks) - infill_section_num_bars))
                )
            bar_infilling_end_idx = bar_infilling_start_idx + infill_section_num_bars

            # Get the k Bar attribute controls randomly from the ones
            # defined by the tokenizer. k = n_controls * random in (0.1,1)
            acs_idx = list(range(len(self.tokenizer.attribute_controls)))
            # Get the indeces of the bars we want to compute the attribute
            # controls on. These bars are from the infilling region only.
            ac_indexes = {
                track_infilling_idx: {
                    ac_idx: list(range(
                                bar_infilling_start_idx,
                                bar_infilling_end_idx + 1 if bar_infilling_end_idx else len(bars_ticks),
                            )
                        )
                    for ac_idx in acs_idx
                }
            }
        else:
            # ac_indexes only contains random track acs for the last track
            acs_idx = sample(
                population=self._ac_tracks,
                k=round(
                    len(self._ac_tracks) * uniform(*self.tracks_idx_random_ratio_range)
                ),
            )
            ac_indexes = {len(score.tracks) - 1: {ac_idx: True for ac_idx in acs_idx}}

        # Tokenize it
        sequences = self.tokenizer.encode(
            score,
            no_preprocess_score=True,
            attribute_controls_indexes=ac_indexes,
            concatenate_track_sequences=False,
        )

        with open('input.txt', 'w') as file:
            # Write each item in the list on a new line
            for sequence in sequences:
                file.write(str(len(sequence.ids)) + "\n")
                for item in sequence.tokens:
                    file.write(str(item) + "\n")

        # Select the chunk of tokens to proceed with based on the sequence length
        # Get the indexes of the token ids from which new bars begin
        # This can unfortunately remove tempos events as they do not begin at each bar.

        # Effective max_seq_len considering Track_Start/End tokens for the portion to
        # infill. First we check if we can keep all the tokens to condition our generation...
        max_seq_len_effective = self.max_seq_len - 2
        if bar_infilling:  # n Infill_Bar + FillBar_Start/End tokens
            max_seq_len_effective -= infill_section_num_bars + 2
        elif self.seq2seq:  # Infill_Track
            max_seq_len_effective -= 1

        # ... If not, create the window
        if sum(len(seq) for seq in sequences) > max_seq_len_effective:
            # max_seq_len_effective will be used to extract tokens within each seq
            # To extract the right number of tokens, we need to decrement it to
            # consider the Track_Start/Program/Track_End tokens, and Infill tokens
            max_seq_len_effective = self.max_seq_len - 3 * len(sequences) #(?) max_seq_len_Effective ridefined here
            if bar_infilling:  # n Infill_Bar + FillBar_Start/End tokens
                max_seq_len_effective -= infill_section_num_bars + 2

            # Fast way to get the position of bar tokens inside sequences.
            # We check if the bar token byte is inside the BPE token.
            # TODO: ask Nathan why the [:2]
            bar_tokens_idxs_per_seq = [
                [
                    i
                    for i in range(len(seq))
                    if self._bar_token_byte
                       in self.tokenizer._model.id_to_token(seq.ids[i])[:2]
                ]
                for seq in sequences
            ]
            num_bars_per_seq = max(len(seq) for seq in bar_tokens_idxs_per_seq)
            # Count the number of tokens making each bar
            bars_num_tokens_per_seq = []  # (S, B*)
            for bar_tokens_idxs, seq in zip(bar_tokens_idxs_per_seq, sequences):
                """# The first values are set to 0 to include the first tokens
                if len(bar_tokens_idxs) > 0:
                    bar_tokens_idxs[0] = 0"""
                num_tokens_bar = (
                    [
                        bar_tokens_idxs[i] - bar_tokens_idxs[i - 1]
                        for i in range(1, len(bar_tokens_idxs))
                    ]
                    if len(bar_tokens_idxs) > 1
                    else []
                )
                num_tokens_bar.append(
                    len(seq.ids) - bar_tokens_idxs[-1]
                )  # num tokens last bar
                bars_num_tokens_per_seq.append(num_tokens_bar)
            # Here we get the number of tokens
            # for each bar summed over each track.
            bars_num_tokens = [  # (B)
                sum(
                    seq[bar_num] if len(seq) > bar_num else 0
                    for seq in bars_num_tokens_per_seq
                )
                for bar_num in range(num_bars_per_seq)
            ]
            # Select the beginning (first bar) of the chunk to return
            # If bar infilling, we make sure to keep the part to infill
            if bar_infilling:
                # Computing number of cumulative tokens from the last bar to infill
                # to the beginning of the song. This is done on bars_num_tokens, so
                # this is computed along the
                # [. . . . . bar_infilling_start_idx ... bar_infilling_end_idx . . . .]
                # ^ - - - - - - - - - - - - - - - - - - - - - - - - ^
                cumsum_end = np.cumsum(
                    np.flip(np.array(bars_num_tokens[: bar_infilling_end_idx + 1]))
                )
                # Here we get the minimum bar index we can use not to exceed
                # the max_seq_len value. If the highest value of cumsum_end is
                # lower than max_seq_value, it will be 0 (i.e. we consider every
                # bar from the beginning to bar_infilling_end_idx.
                min_bar_context_start_idx = (
                        bar_infilling_end_idx
                        - len(np.nonzero(cumsum_end < max_seq_len_effective)[0])
                        + 1
                )
                if min_bar_context_start_idx >= bar_infilling_start_idx:
                    # The length of the portion to infill exceed the limit, so we reduce
                    # it so that one half is context and the other is to generate.
                    bar_infilling_start_idx = (
                            min_bar_context_start_idx + (bar_infilling_end_idx - min_bar_context_start_idx) // 2
                    )
                    max_bar_start_idx = (
                            min_bar_context_start_idx + (bar_infilling_start_idx - min_bar_context_start_idx) // 2
                    )
                    infill_section_num_bars = bar_infilling_end_idx - bar_infilling_start_idx
                else:
                    max_bar_start_idx = bar_infilling_start_idx
            # We discard the bars at the end (to maximize the chunk sequence length) and
            # the bars whose number of tokens exceed the limit.
            # TODO: Ask nathan why we don't consider the context that
            #   comes after the end of the infilling region
            else:
                min_bar_context_start_idx = 0
                cumsum_inv = np.cumsum(np.flip(np.array(bars_num_tokens)))
                max_bar_start_idx = num_bars_per_seq - len(
                    np.nonzero(cumsum_inv < max_seq_len_effective)[0]
                )

            # TODO: This process is different from what's happening in
            # Jeff's MMM. Here we randomly pick a bar index between
            # min_bar_context_start_idx and the start of the infilling part, while
            # we could maybe consider half context at left and half context
            # at right wrt the infilling zone.
            population = [
                idx
                for idx in range(min_bar_context_start_idx, max_bar_start_idx + 1)
                if bars_num_tokens[idx] < max_seq_len_effective
            ]
            bar_start_idx = choice(population)
            cumsum_start = np.cumsum(bars_num_tokens[bar_start_idx:])
            bar_end_idx = (
                num_bars_per_seq - 1
                if cumsum_start[-1] < max_seq_len_effective
                else bar_start_idx
                     + np.nonzero(cumsum_start >= max_seq_len_effective)[0][0]
                     - 1
            )
            # Decrease bar idx to infill
            if bar_infilling:
                bar_infilling_start_idx -= bar_start_idx
                bar_infilling_end_idx -= bar_start_idx
            # s_len = sum(bars_num_tokens[i] for i in range(bar_start_idx, bar_end_idx))
            for i, (seq, bar_tokens_idxs) in reversed(
                    list(enumerate(zip(sequences, bar_tokens_idxs_per_seq)))
            ):
                # If the sequence ends before the start of the portion to extract begins
                # we remove it.
                if len(bar_tokens_idxs) - 1 < bar_start_idx:
                    del sequences[i], bar_tokens_idxs_per_seq[i]
                    if self.seq2seq and len(sequences) == 1 and not bar_infilling:
                        # We need at least two sequences to generate a new track in
                        # seq2seq. If there is only one remaining left (because some
                        # were deleted when handling the token sequence length above),
                        # we skip this training sample. Resorting to bar infilling would
                        # be possible by extracting the portion extracted here from the
                        # score and recompute the bar indexes to infill.
                        return None, None
                    if track_infilling_idx and i < track_infilling_idx:
                        track_infilling_idx -= 1
                    continue
                tok_idx_start = (
                    0 if bar_tokens_idxs == 0 else bar_tokens_idxs[bar_start_idx]
                )
                tok_idx_end = (
                    bar_tokens_idxs[bar_end_idx + 1]
                    if bar_end_idx + 1 < len(bar_tokens_idxs)
                    else -1
                )
                program_id = self.tokenizer.vocab[sequences[i].tokens[1]]
                sequences[i] = TokSequence(
                    ids=seq.ids[tok_idx_start:tok_idx_end], are_ids_encoded=True
                )
                # Add Track_Start, Program and Track_End tokens
                sequences[i].ids.insert(0, self._track_start_token_id)
                sequences[i].ids.insert(1, program_id)
                # Add track_end if bar_end_idx isn't the last bar as already included
                if bar_end_idx + 1 < num_bars_per_seq:
                    sequences[i].ids.append(self._track_end_token_id)
                # For debug
                # self.tokenizer.decode_token_ids(sequences[i])
                # self.tokenizer.complete_sequence(sequences[i])
            # seq_len_final = sum(len(seq) for seq in sequences)

        # If doing bar infilling we place extract the bars and create place the
        # right infilling tokens
        # Otherwise (track infilling), there is nothing to do here. If a user wants to
        # create a new track, we'll just have to add Track_Start and Program tokens at
        # the end of the sequence and generate from here.
        decoder_input = None
        if bar_infilling:
            # Bar infilling
            # Extract token section of the bars to infill
            # We do not select tracks that end before bar_tick_start
            bar_tokens_idxs = [
                i
                for i in range(len(sequences[track_infilling_idx]))
                if self._bar_token_byte
                   in self.tokenizer._model.id_to_token(
                    sequences[track_infilling_idx].ids[i]
                )[:2]
            ]
            # token_idx_start excludes Track_Start and attribute control token
            token_idx_start = bar_tokens_idxs[bar_infilling_start_idx]
            # excluding Track_End if last bar
            token_idx_end = (
                -1
                if bar_infilling_end_idx >= len(bar_tokens_idxs) - 1
                else bar_tokens_idxs[bar_infilling_end_idx]
            )

            # Extract the tokens of the section to infill and add BarFill start/end
            seq_infill = sequences[track_infilling_idx][token_idx_start:token_idx_end]
            seq_infill.ids.insert(0, self._infill_bar_start_token_id)
            seq_infill.ids.append(self._infill_bar_end_token_id)
            # seq2seq --> decoder infill / concat seq before after
            if self.seq2seq:
                decoder_input = seq_infill
            else:
                # Adding it at the end of the list that will be flattened
                sequences.append(seq_infill)

            # Add BarInfill tokens + update sequences
            seq_before = sequences[track_infilling_idx][:token_idx_start]
            for _ in range(infill_section_num_bars):
                seq_before.ids.append(self._infill_bar_token_id)
            seq_after = sequences[track_infilling_idx][token_idx_end:]
            sequences[track_infilling_idx] = seq_before + seq_after
        # If seq2seq, the last track sequence is fed to the decoder and a `Infill_Track`
        # token is appended to the encoder input sequence
        elif self.seq2seq:
            # There are always at least two sequences
            decoder_input = sequences.pop(-1)
            sequences[-1].ids.append(self._infill_track_token_id)

        # TODO External labels: (non-)expressive, loops, genres to add to the seq

        #for sequence in sequences:

        #    self.tokenizer.decode_token_ids(sequence)


        #tokens_infilling_track = self.tokenizer._ids_to_tokens(sequences[track_infilling_idx])
        # Preprocessing token ids: reduce sequence length, add BOS/EOS tokens

        #with open('after.txt', 'w') as file:
        #    file.write(f"keeping tracks {selected_tracks}")
        #    for sequence in sequences:
        #        file.write(str(len(sequence.ids)) + "\n")
        #        # Write each item in the list on a new line
        #        for item in sequence.tokens:
        #            file.write(str(item) + "\n")

        tokseq = concat_tokseq(sequences)
        # No need to call self._preprocess_token_ids as there are no BOS/EOS tokens and
        # that the sequence le  ngth does not exceed the limit as handled above.

        self.tokenizer.decode_token_ids(tokseq)

        tokseq_tokens = self.tokenizer._ids_to_tokens(tokseq)

        with open('output.txt', 'w') as file:
            for item in tokseq_tokens:
                file.write(str(item) + "\n")

        # [DEBUG] Reconstruct MIDI file from toksequence
        tokseq_tokens = np.array(tokseq.tokens)
        indices = np.where(tokseq_tokens == "Infill_Bar")[0]
        fill_bar_start_idx = np.where(tokseq_tokens == "FillBar_Start")[0][0]
        list_seqs = []
        list_seqs.append(tokseq[:indices[0]])
        list_seqs.append(tokseq[fill_bar_start_idx+1:-1])
        list_seqs.append(tokseq[indices[-1]+1:])
        midi_tok_seq = concat_tokseq(list_seqs)

        score_ = self.tokenizer._tokens_to_score(midi_tok_seq)
        score_.dump_midi("output_reconstructed.mid")

        return tokseq, decoder_input


    def _new_tokenize_score(
            self, score: Score
    ) -> tuple[TokSequence, TokSequence | None] | tuple[None, None]:

        debug = True

        # Delete unused elements in order to compute the bar ticks that we will use,
        # otherwise if unused elements are at the end of the score they can give us
        # bar ticks exceeding the tokens
        score.markers = []
        score.key_signatures = []
        for track in score.tracks:
            track.controls = []
            track.lyrics = []
            if not self.tokenizer.config.use_sustain_pedals:
                track.pedals = []
            if not self.tokenizer.config.use_pitch_bends:
                track.pitch_bends = []

        # Select k tracks (shuffled)
        num_tracks_to_keep = max(
            1, round(len(score.tracks) * uniform(*self.ratio_random_tracks_range))
        )
        bars_ticks = np.array(get_bars_ticks(score))  # always at least two bars
        tracks_idx_ok = [
            idx
            for idx in range(len(score.tracks))
            if len(score.tracks[idx].notes) > 0
               and score.tracks[idx].notes[-1].time > bars_ticks[1]
        ]  # always at least one
        # Initialize the list to store the selected indices

        selected_tracks = [] # Just for debugging purposes

        # Select random indices for the tracks
        score.tracks = [
            score.tracks[idx]
            #for idx in sample(
            #    tracks_idx_ok, k=min(num_tracks_to_keep, len(tracks_idx_ok))
            #)
            for idx in [1,2,3,4,5,6]
            if selected_tracks.append(idx) or True  # Save each selected index
        ]

        # Remove time signatures and tempos occurring after the start of the last note
        max_note_time = 0
        for track in score.tracks:
            if len(track.notes) > 0:
                max_note_time = max(max_note_time, track.notes[-1].time)
        for ti in reversed(range(len(score.time_signatures))):
            if score.time_signatures[ti].time > max_note_time:
                del score.time_signatures[ti]
            else:
                break
        for ti in reversed(range(len(score.tempos))):
            if score.tempos[ti].time > max_note_time:
                del score.tempos[ti]
            else:
                break

        # Augment and preprocess the music.
        # We need to preprocess it here as we require it preprocessed to select the
        # bars and tracks indexes for attribute controls before tokenizing.
        # score = self.augment_and_preprocess_score(score)
        score = self.tokenizer.preprocess_score(score)
        if len(score.tracks) == 0:
            return None, None

        # Set bar infilling or new track gen + their indexes
        # If there is only one track, we do bar infilling as new track gen is not
        # possible in seq2seq.
        bar_infilling = True #len(score.tracks) == 1 or random() < self.bar_fill_ratio
        track_infilling_idx = None
        bar_infilling_start_idx, bar_infilling_end_idx, infill_section_num_bars = None, None, None

        if bar_infilling:
            track_infilling_idx = 0 #choice(list(range(len(score.tracks))))
            # ac_indexes contains random bar acs only for the section to infill
            bars_ticks = np.array(get_bars_ticks(score))  # need to recompute (resample)
            bars_ticks = bars_ticks[  # remove bars ticks after end of track
                np.where(bars_ticks <= score.tracks[track_infilling_idx].notes[-1].time)
            ]

            # Generate a random number between 1 and 20
            #infill_section_num_bars = random.randint(1, self.max_bar_infilling_length)

            #if infill_section_num_bars > len(bars_ticks):
            #    infill_section_num_bars = random.randint(1, 4)

            infill_section_num_bars = max(
                1,
                round(
                    len(bars_ticks) * uniform(0.1,0.4)
                ),
            )

            # TODO: save the number of infilled bars, just to see the distribution

            if infill_section_num_bars == len(bars_ticks):
                # Special case where the portion to mask == the whole music
                bar_infilling_start_idx = 0
            else:
                bar_infilling_start_idx = choice(
                    list(range(len(bars_ticks) - infill_section_num_bars))
                )
            # TODO: Just for the bug, remove after
            bar_infilling_start_idx = 5
            bar_infilling_end_idx = bar_infilling_start_idx + infill_section_num_bars

            acs_idx = self._ac_bars

            # If the track is a drum track, keep only note density attrbiute control
            if score.tracks[track_infilling_idx].is_drum:
                acs_idx = [
                    ac_idx
                    for ac_idx in acs_idx
                    if self.tokenizer.attribute_controls[ac_idx].__class__.__name__ == "BarNoteDensity"
                ]

            # Get indices of bars to compute attribute controls on
            # (= all the bars we want to infill)
            ac_indexes = {
                track_infilling_idx: {
                    ac_idx: list(range(
                                bar_infilling_start_idx,
                                bar_infilling_end_idx + 1 if bar_infilling_end_idx else len(bars_ticks),
                            )
                        )
                    for ac_idx in acs_idx
                }
            }
        #else: CAN'T USE THIS DUE TO MIDITOK BUG...
            # Indeces of tracks attribute controls
            #acs_idx = self._ac_tracks
            # Compute all attribute controls on all tracks
            #ac_indexes = {track_idx: {ac_idx: True for ac_idx in acs_idx}
                          #for track_idx in range(len(score.tracks))}
        else:

            whole_track_infilling_idx = 0 #choice(list(range(len(score.tracks))))

            acs_idx = self._ac_bars
            if score.tracks[whole_track_infilling_idx].is_drum:
                acs_idx = [
                    ac_idx
                    for ac_idx in acs_idx
                    if self.tokenizer.attribute_controls[ac_idx].__class__.__name__ == "BarNoteDensity"
                ]
            ac_indexes = {
                whole_track_infilling_idx: {
                    ac_idx: list(range(
                        0,
                        len(bars_ticks),
                    )
                    )
                    for ac_idx in acs_idx
                }
            }

        # Tokenize it
        sequences = self.tokenizer.encode(
            score,
            no_preprocess_score=True,
            attribute_controls_indexes=ac_indexes,
            concatenate_track_sequences=False,
        )

        if(debug):

            with open('before_window_creation.txt', 'w') as file:
                # Write each item in the list on a new line
                seq_number = 0
                for sequence in sequences:
                    file.write(str(len(sequence.ids)) + "\n")
                    bar_counter = 0
                    for item in sequence.tokens:
                        if item == "Bar_None":
                            file.write(str(item) + f"{seq_number}_{bar_counter}\n")
                            bar_counter += 1
                        else:
                            file.write(str(item) + "\n")
                    seq_number += 1
            """
            ids = sequences[track_infilling_idx].ids
            bytes = [self.tokenizer._model.id_to_token(id_) for id_ in ids]
            decoded_tokens = [
                self.tokenizer._vocab_learned_bytes_to_tokens[byte_] for byte_ in bytes
            ]
            with open('before_window_creation_bpe.txt', 'w', encoding='utf-8') as file:
                for id_, byte_, token_list in zip(ids, bytes, decoded_tokens):
                    token_str = ", ".join(token_list)  # Join the list of tokens into a single string
                    file.write(f"{id_} - {byte_} - [{token_str}]\n")
            """

        # Select the chunk of tokens to proceed with based on the sequence length
        # Get the indexes of the token ids from which new bars begin
        # This can unfortunately remove tempos events as they do not begin at each bar.

        # Effective max_seq_len considering Track_Start/End tokens for the portion to
        # infill. First we check if we can keep all the tokens to condition our generation...
        # TODO: Ask Nathan why this -2.
        max_seq_len_effective = self.max_seq_len - 2
        if bar_infilling:  # n Infill_Bar + FillBar_Start/End tokens
            max_seq_len_effective -= infill_section_num_bars + 2
        elif self.seq2seq:  # Infill_Track
            max_seq_len_effective -= 1

        # ... If not, create the window
        if sum(len(seq) for seq in sequences) > max_seq_len_effective:
            # max_seq_len_effective will be used to extract tokens within each seq
            # To extract the right number of tokens, we need to decrement it to
            # consider the Track_Start/Program/Track_End tokens, and Infill tokens
            max_seq_len_effective = self.max_seq_len - 3 * len(sequences)
            if bar_infilling:  # n Infill_Bar + FillBar_Start/End tokens
                max_seq_len_effective -= infill_section_num_bars + 2

            # Fast way to get the position of bar tokens inside sequences.
            # We check if the bar token byte is inside the BPE token.
            # TODO: ask Nathan why the [:2]
            bar_tokens_idxs_per_seq = [
                [
                    i
                    for i in range(len(seq))
                    if self._bar_token_byte
                       in self.tokenizer._model.id_to_token(seq.ids[i])[:2]
                ]
                for seq in sequences
            ]
            num_bars_per_seq = max(len(seq) for seq in bar_tokens_idxs_per_seq)
            # Count the number of tokens making each bar
            bars_num_tokens_per_seq = []  # (S, B*)
            for bar_tokens_idxs, seq in zip(bar_tokens_idxs_per_seq, sequences):
                """# The first values are set to 0 to include the first tokens
                if len(bar_tokens_idxs) > 0:
                    bar_tokens_idxs[0] = 0"""
                num_tokens_bar = (
                    [
                        bar_tokens_idxs[i] - bar_tokens_idxs[i - 1]
                        for i in range(1, len(bar_tokens_idxs))
                    ]
                    if len(bar_tokens_idxs) > 1
                    else []
                )
                num_tokens_bar.append(
                    len(seq.ids) - bar_tokens_idxs[-1]
                )  # num tokens last bar
                bars_num_tokens_per_seq.append(num_tokens_bar)
            # Here we get the number of tokens
            # for each bar summed over each track.
            bars_num_tokens = [  # (B)
                sum(
                    seq[bar_num] if len(seq) > bar_num else 0
                    for seq in bars_num_tokens_per_seq
                )
                for bar_num in range(num_bars_per_seq)
            ]
            # Select the beginning (first bar) of the chunk to return
            # If bar infilling, we make sure to keep the part to infill
            if bar_infilling:
                # Computing number of cumulative tokens from the last bar to infill
                # to the beginning of the song. This is done on bars_num_tokens, so
                # this is computed along the
                # [. . . . . bar_infilling_start_idx ... bar_infilling_end_idx . . . .]
                # ^ - - - - - - - - - - - - - - - - - - - - - - - - ^
                cumsum_end = np.cumsum(
                    np.flip(np.array(bars_num_tokens[: bar_infilling_end_idx + 1]))
                )
                # Here we get the minimum bar index we can use not to exceed
                # the max_seq_len value. If the highest value of cumsum_end is
                # lower than max_seq_value, it will be 0 (i.e. we consider every
                # bar from the beginning to bar_infilling_end_idx.
                min_bar_context_start_idx = (
                        bar_infilling_end_idx
                        - len(np.nonzero(cumsum_end < max_seq_len_effective)[0])
                        + 1
                )
                if min_bar_context_start_idx >= bar_infilling_start_idx:
                    # The length of the portion to infill exceed the limit, so we reduce
                    # it so that one half is context and the other is to generate.
                    bar_infilling_start_idx = (
                            min_bar_context_start_idx + (bar_infilling_end_idx - min_bar_context_start_idx) // 2
                    )
                    max_bar_start_idx = (
                            min_bar_context_start_idx + (bar_infilling_start_idx - min_bar_context_start_idx) // 2
                    )
                    infill_section_num_bars = bar_infilling_end_idx - bar_infilling_start_idx
                else:
                    max_bar_start_idx = bar_infilling_start_idx
            # We discard the bars at the end (to maximize the chunk sequence length) and
            # the bars whose number of tokens exceed the limit.
            # TODO: Ask nathan why we don't consider the context that
            #   comes after the end of the infilling region
            else:
                min_bar_context_start_idx = 0
                cumsum_inv = np.cumsum(np.flip(np.array(bars_num_tokens)))
                max_bar_start_idx = num_bars_per_seq - len(
                    np.nonzero(cumsum_inv < max_seq_len_effective)[0]
                )

            # TODO: This process is different from what's happening in
            # Jeff's MMM. Here we randomly pick a bar index between
            # min_bar_context_start_idx and the start of the infilling part, while
            # we could maybe consider half context at left and half context
            # at right wrt the infilling zone.
            population = [
                idx
                for idx in range(min_bar_context_start_idx, max_bar_start_idx + 1)
                if bars_num_tokens[idx] < max_seq_len_effective
            ]
            bar_start_idx = choice(population)
            cumsum_start = np.cumsum(bars_num_tokens[bar_start_idx:])
            bar_end_idx = (
                num_bars_per_seq - 1
                if cumsum_start[-1] < max_seq_len_effective
                else bar_start_idx
                     + np.nonzero(cumsum_start >= max_seq_len_effective)[0][0]
                     - 1
            )
            # Decrease bar idx to infill

            if bar_infilling:
                bar_infilling_start_idx -= bar_start_idx
                bar_infilling_end_idx -= bar_start_idx
            # s_len = sum(bars_num_tokens[i] for i in range(bar_start_idx, bar_end_idx))
            for i, (seq, bar_tokens_idxs) in reversed(
                    list(enumerate(zip(sequences, bar_tokens_idxs_per_seq)))
            ):
                # If the sequence ends before the start of the portion to extract begins
                # we remove it.
                if len(bar_tokens_idxs) - 1 < bar_start_idx:
                    del sequences[i], bar_tokens_idxs_per_seq[i]
                    if self.seq2seq and len(sequences) == 1 and not bar_infilling:
                        # We need at least two sequences to generate a new track in
                        # seq2seq. If there is only one remaining left (because some
                        # were deleted when handling the token sequence length above),
                        # we skip this training sample. Resorting to bar infilling would
                        # be possible by extracting the portion extracted here from the
                        # score and recompute the bar indexes to infill.
                        return None, None
                    if track_infilling_idx and i < track_infilling_idx:
                        track_infilling_idx -= 1
                    continue
                tok_idx_start = (
                    0 if bar_tokens_idxs == 0 else bar_tokens_idxs[bar_start_idx]
                )
                tok_idx_end = (
                    bar_tokens_idxs[bar_end_idx + 1]
                    if bar_end_idx + 1 < len(bar_tokens_idxs)
                    else -1
                )
                program_id = self.tokenizer.vocab[sequences[i].tokens[1]]

                """
                if debug and i == track_infilling_idx:
                    ids = seq.ids[tok_idx_start:tok_idx_end]
                    bytes = [self.tokenizer._model.id_to_token(id_) for id_ in ids]
                    decoded_tokens = [
                        self.tokenizer._vocab_learned_bytes_to_tokens[byte_] for byte_ in bytes
                    ]

                    with open('infilling_section.txt', 'w', encoding='utf-8') as file:
                        for id_, byte_, token_list in zip(ids, bytes, decoded_tokens):
                            token_str = ", ".join(token_list)  # Join the list of tokens into a single string
                            file.write(f"{id_} - {byte_} - [{token_str}]\n")
                """

                sequences[i] = TokSequence(
                    ids=seq.ids[tok_idx_start:tok_idx_end], are_ids_encoded=True
                )
                # Add Track_Start, Program, and Track_End tokens
                sequences[i].ids.insert(0, self._track_start_token_id)

                if not bar_infilling and i == whole_track_infilling_idx:
                        sequences[i].ids[0] = self._infill_track_token_id

                sequences[i].ids.insert(1, program_id)

                # Add track_end if bar_end_idx isn't the last bar as already included
                if bar_end_idx + 1 < num_bars_per_seq:
                    sequences[i].ids.append(self._track_end_token_id)
                # For debug
                #if(debug):
                #    self.tokenizer.decode_token_ids(sequences[i])
                    #self.tokenizer.complete_sequence(sequences[i])

            # seq_len_final = sum(len(seq) for seq in sequences)

        # -------- WINDOW CREATION FINISHED ----------- #

        with open('after_window_creation.txt', 'w') as file:
            # Write each item in the list on a new line
            seq_number = 0
            for sequence in sequences:
                file.write(str(len(sequence.ids)) + "\n")
                bar_counter = 0
                for item in sequence.tokens:
                    if item == "Bar_None":
                        file.write(str(item) + f"{seq_number}_{bar_counter}\n")
                        bar_counter += 1
                    file.write(str(item) + "\n")
                seq_number += 1

        # If doing bar infilling we place extract the bars and create place the
        # right infilling tokens
        # Otherwise (track infilling), there is nothing to do here. If a user wants to
        # create a new track, we'll just have to add Track_Start and Program tokens at
        # the end of the sequence and generate from here.
        decoder_input = None
        if bar_infilling:
            # Bar infilling
            # Extract token section of the bars to infill
            # We do not select tracks that end before bar_tick_start
            bar_tokens_idxs = [
                i
                for i in range(len(sequences[track_infilling_idx]))
                if self._bar_token_byte
                   in self.tokenizer._model.id_to_token(
                    sequences[track_infilling_idx].ids[i]
                )[:2]
            ]
            # token_idx_start excludes Track_Start and attribute control token
            token_idx_start = bar_tokens_idxs[bar_infilling_start_idx]
            # excluding Track_End if last bar
            token_idx_end = (
                -1
                if bar_infilling_end_idx >= len(bar_tokens_idxs) - 1
                else bar_tokens_idxs[bar_infilling_end_idx]
            )

            if debug:
                for sequence in sequences:
                    self.tokenizer.decode_token_ids(sequence)

                with open('before_extraction.txt', 'w') as file:
                    # Write each item in the list on a new line
                    seq_number = 0
                    for sequence in sequences:
                        file.write(str(len(sequence.ids)) + "\n")
                        bar_counter = 0
                        for item in sequence.tokens:
                            if item == "Bar_None":
                                file.write(str(item) + f"{seq_number}_{bar_counter}\n")
                                bar_counter += 1
                            file.write(str(item) + "\n")
                        seq_number += 1

            # Extract the tokens of the section to infill and add BarFill start/end
            seq_infill = sequences[track_infilling_idx][token_idx_start:token_idx_end]
            seq_infill.ids.insert(0, self._infill_bar_start_token_id)
            seq_infill.ids.append(self._infill_bar_end_token_id)
            # seq2seq --> decoder infill / concat seq before after
            if self.seq2seq:
                decoder_input = seq_infill
            else:
                # Adding it at the end of the list that will be flattened
                sequences.append(seq_infill)

            # Add BarInfill tokens + update sequences
            seq_before = sequences[track_infilling_idx][:token_idx_start]
            for _ in range(infill_section_num_bars):
                seq_before.ids.append(self._infill_bar_token_id)
            seq_after = sequences[track_infilling_idx][token_idx_end:]
            sequences[track_infilling_idx] = seq_before + seq_after
        #If we are doing track infilling
        #else:
            # We want to move one random track at the end, after putting an Infill_Track token
        # If seq2seq, the last track sequence is fed to the decoder and a `Infill_Track`
        # token is appended to the encoder input sequence
        elif self.seq2seq:
            # There are always at least two sequences
            decoder_input = sequences.pop(-1)
            sequences[-1].ids.append(self._infill_track_token_id)

        # TODO External labels: (non-)expressive, loops, genres to add to the seq

        #for sequence in sequences:

        #    self.tokenizer.decode_token_ids(sequence)


        #tokens_infilling_track = self.tokenizer._ids_to_tokens(sequences[track_infilling_idx])
        # Preprocessing token ids: reduce sequence length, add BOS/EOS tokens

        #with open('after.txt', 'w') as file:
        #    file.write(f"keeping tracks {selected_tracks}")
        #    for sequence in sequences:
        #        file.write(str(len(sequence.ids)) + "\n")
        #        # Write each item in the list on a new line
        #        for item in sequence.tokens:
        #            file.write(str(item) + "\n")

        tokseq = concat_tokseq(sequences)
        # No need to call self._preprocess_token_ids as there are no BOS/EOS tokens and
        # that the sequence le  ngth does not exceed the limit as handled above.

        self.tokenizer.decode_token_ids(tokseq)

        tokseq_tokens = self.tokenizer._ids_to_tokens(tokseq)

        with open('output.txt', 'w') as file:
            for item in tokseq_tokens:
                file.write(str(item) + "\n")

        # [DEBUG] Reconstruct MIDI file from toksequence
        tokseq_tokens = np.array(tokseq.tokens)
        indices = np.where(tokseq_tokens == "Infill_Bar")[0]
        fill_bar_start_idx = np.where(tokseq_tokens == "FillBar_Start")[0][0]
        list_seqs = []
        list_seqs.append(tokseq[:indices[0]])
        list_seqs.append(tokseq[fill_bar_start_idx+1:-1])
        list_seqs.append(tokseq[indices[-1]+1:])
        midi_tok_seq = concat_tokseq(list_seqs)

        score_ = self.tokenizer._tokens_to_score(midi_tok_seq)
        score_.dump_midi("output_reconstructed.mid")

        return tokseq, decoder_input


    def _new_tokenize_score_new_window(
            self, score: Score
    ) -> tuple[TokSequence, TokSequence | None] | tuple[None, None]:

        debug = True

        # Delete unused elements in order to compute the bar ticks that we will use,
        # otherwise if unused elements are at the end of the score they can give us
        # bar ticks exceeding the tokens
        score.markers = []
        score.key_signatures = []
        for track in score.tracks:
            track.controls = []
            track.lyrics = []
            if not self.tokenizer.config.use_sustain_pedals:
                track.pedals = []
            if not self.tokenizer.config.use_pitch_bends:
                track.pitch_bends = []

        # Select k tracks (shuffled)
        num_tracks_to_keep = max(
            1, round(len(score.tracks) * uniform(*self.ratio_random_tracks_range))
        )
        bars_ticks = np.array(get_bars_ticks(score))  # always at least two bars
        tracks_idx_ok = [
            idx
            for idx in range(len(score.tracks))
            if len(score.tracks[idx].notes) > 0
               and score.tracks[idx].notes[-1].time > bars_ticks[1]
        ]  # always at least one
        # Initialize the list to store the selected indices

        selected_tracks = [] # Just for debugging purposes

        # Select random indices for the tracks
        score.tracks = [
            score.tracks[idx]
            for idx in sample(
                tracks_idx_ok, k=min(num_tracks_to_keep, len(tracks_idx_ok))
            )
            if selected_tracks.append(idx) or True  # Save each selected index
        ]

        # Remove time signatures and tempos occurring after the start of the last note
        max_note_time = 0
        for track in score.tracks:
            if len(track.notes) > 0:
                max_note_time = max(max_note_time, track.notes[-1].time)
        for ti in reversed(range(len(score.time_signatures))):
            if score.time_signatures[ti].time > max_note_time:
                del score.time_signatures[ti]
            else:
                break
        for ti in reversed(range(len(score.tempos))):
            if score.tempos[ti].time > max_note_time:
                del score.tempos[ti]
            else:
                break

        # Augment and preprocess the music.
        # We need to preprocess it here as we require it preprocessed to select the
        # bars and tracks indexes for attribute controls before tokenizing.
        # score = self.augment_and_preprocess_score(score)
        score = self.tokenizer.preprocess_score(score)
        if len(score.tracks) == 0:
            return None, None

        # Set bar infilling or new track gen + their indexes
        # If there is only one track, we do bar infilling as new track gen is not
        # possible in seq2seq.
        bar_infilling = True #len(score.tracks) == 1 or random() < self.bar_fill_ratio
        track_infilling_idx = None
        bar_infilling_start_idx, bar_infilling_end_idx, infill_section_num_bars = None, None, None

        if bar_infilling:
            track_infilling_idx = choice(list(range(len(score.tracks))))
            # ac_indexes contains random bar acs only for the section to infill
            bars_ticks = np.array(get_bars_ticks(score))  # need to recompute (resample)
            bars_ticks = bars_ticks[  # remove bars ticks after end of track
                np.where(bars_ticks <= score.tracks[track_infilling_idx].notes[-1].time)
            ]

            # Generate a random number between 1 and 20
            #infill_section_num_bars = random.randint(1, self.max_bar_infilling_length)

            #if infill_section_num_bars > len(bars_ticks):
            #    infill_section_num_bars = random.randint(1, 4)

            infill_section_num_bars = max(
                1,
                round(
                    len(bars_ticks) * uniform(0.1,0.4)
                ),
            )

            # TODO: save the number of infilled bars, just to see the distribution

            if infill_section_num_bars == len(bars_ticks):
                # Special case where the portion to mask == the whole music
                bar_infilling_start_idx = 0
            else:
                bar_infilling_start_idx = choice(
                    list(range(len(bars_ticks) - infill_section_num_bars))
                )
            bar_infilling_end_idx = bar_infilling_start_idx + infill_section_num_bars

            # Get attribute control indices
            acs_idx = self._ac_bars

            # Get indices of bars to compute attribute controls on
            # (= all the bars we want to infill)
            ac_indexes = {
                track_infilling_idx: {
                    ac_idx: list(range(
                                bar_infilling_start_idx,
                                bar_infilling_end_idx + 1 if bar_infilling_end_idx else len(bars_ticks),
                            )
                        )
                    for ac_idx in acs_idx
                }
            }
        else:
            # Indeces of tracks attribute controls
            acs_idx = self._ac_tracks
            # Compute all attribute controls on all tracks
            ac_indexes = {track_idx: {ac_idx: True for ac_idx in acs_idx}
                          for track_idx in range(len(score.tracks))}

        # Tokenize it
        sequences = self.tokenizer.encode(
            score,
            no_preprocess_score=True,
            attribute_controls_indexes=ac_indexes,
            concatenate_track_sequences=False,
        )

        with open('input.txt', 'w') as file:
            # Write each item in the list on a new line
            seq_number = 0
            for sequence in sequences:
                file.write(str(len(sequence.ids)) + "\n")
                bar_counter = 0
                for item in sequence.tokens:
                    if item == "Bar_None":
                        file.write(str(item) + f"{seq_number}_{bar_counter}\n")
                        bar_counter += 1
                    file.write(str(item) + "\n")
                seq_number += 1

        # Select the chunk of tokens to proceed with based on the sequence length
        # Get the indexes of the token ids from which new bars begin
        # This can unfortunately remove tempos events as they do not begin at each bar.

        # Effective max_seq_len considering Track_Start/End tokens for the portion to
        # infill. First we check if we can keep all the tokens to condition our generation...
        # TODO: Ask Nathan why this -2.
        max_seq_len_effective = self.max_seq_len - 2
        if bar_infilling:  # n Infill_Bar + FillBar_Start/End tokens
            max_seq_len_effective -= infill_section_num_bars + 2
        elif self.seq2seq:  # Infill_Track
            max_seq_len_effective -= 1

        # ... If not, create the window
        if sum(len(seq) for seq in sequences) > max_seq_len_effective:
            # max_seq_len_effective will be used to extract tokens within each seq
            # To extract the right number of tokens, we need to decrement it to
            # consider the Track_Start/Program/Track_End tokens, and Infill tokens
            max_seq_len_effective = self.max_seq_len - 3 * len(sequences)
            if bar_infilling:  # n Infill_Bar + FillBar_Start/End tokens
                max_seq_len_effective -= infill_section_num_bars + 2

            # Fast way to get the position of bar tokens inside sequences.
            # We check if the bar token byte is inside the BPE token.
            # TODO: ask Nathan why the [:2]
            bar_tokens_idxs_per_seq = [
                [
                    i
                    for i in range(len(seq))
                    if self._bar_token_byte
                       in self.tokenizer._model.id_to_token(seq.ids[i])[:2]
                ]
                for seq in sequences
            ]
            num_bars_per_seq = max(len(seq) for seq in bar_tokens_idxs_per_seq)
            # Count the number of tokens making each bar
            bars_num_tokens_per_seq = []  # (S, B*)
            for bar_tokens_idxs, seq in zip(bar_tokens_idxs_per_seq, sequences):
                """# The first values are set to 0 to include the first tokens
                if len(bar_tokens_idxs) > 0:
                    bar_tokens_idxs[0] = 0"""
                num_tokens_bar = (
                    [
                        bar_tokens_idxs[i] - bar_tokens_idxs[i - 1]
                        for i in range(1, len(bar_tokens_idxs))
                    ]
                    if len(bar_tokens_idxs) > 1
                    else []
                )
                num_tokens_bar.append(
                    len(seq.ids) - bar_tokens_idxs[-1]
                )  # num tokens last bar
                bars_num_tokens_per_seq.append(num_tokens_bar)
            # Here we get the number of tokens
            # for each bar summed over each track.
            bars_num_tokens = [  # (B)
                sum(
                    seq[bar_num] if len(seq) > bar_num else 0
                    for seq in bars_num_tokens_per_seq
                )
                for bar_num in range(num_bars_per_seq)
            ]
            # Select the beginning (first bar) of the chunk to return
            # If bar infilling, we make sure to keep the part to infill
            if bar_infilling:
                # Computing number of cumulative tokens from the last bar to infill
                # to the beginning of the song. This is done on bars_num_tokens, so
                # this is computed along the
                # [. . . . . bar_infilling_start_idx ... bar_infilling_end_idx . . . .]
                # ^ - - - - - - - - - - - - - - - - - - - - - - - - ^
                cumsum_end = np.cumsum(
                    np.flip(np.array(bars_num_tokens[: bar_infilling_end_idx + 1]))
                )
                # Here we get the minimum bar index we can use not to exceed
                # the max_seq_len value. If the highest value of cumsum_end is
                # lower than max_seq_value, it will be 0 (i.e. we consider every
                # bar from the beginning to bar_infilling_end_idx.
                min_bar_context_start_idx = (
                        bar_infilling_end_idx
                        - len(np.nonzero(cumsum_end < max_seq_len_effective)[0])
                        + 1
                )
                if min_bar_context_start_idx >= bar_infilling_start_idx:
                    # The length of the portion to infill exceed the limit, so we reduce
                    # it so that one half is context and the other is to generate.
                    bar_infilling_start_idx = (
                            min_bar_context_start_idx + (bar_infilling_end_idx - min_bar_context_start_idx) // 2
                    )
                    max_bar_start_idx = (
                            min_bar_context_start_idx + (bar_infilling_start_idx - min_bar_context_start_idx) // 2
                    )
                    infill_section_num_bars = bar_infilling_end_idx - bar_infilling_start_idx
                else:
                    max_bar_start_idx = bar_infilling_start_idx
            # We discard the bars at the end (to maximize the chunk sequence length) and
            # the bars whose number of tokens exceed the limit.
            # TODO: Ask nathan why we don't consider the context that
            #   comes after the end of the infilling region
            else:
                min_bar_context_start_idx = 0
                cumsum_inv = np.cumsum(np.flip(np.array(bars_num_tokens)))
                max_bar_start_idx = num_bars_per_seq - len(
                    np.nonzero(cumsum_inv < max_seq_len_effective)[0]
                )

            # TODO: This process is different from what's happening in
            # Jeff's MMM. Here we randomly pick a bar index between
            # min_bar_context_start_idx and the start of the infilling part, while
            # we could maybe consider half context at left and half context
            # at right wrt the infilling zone.
            population = [
                idx
                for idx in range(min_bar_context_start_idx, max_bar_start_idx + 1)
                if bars_num_tokens[idx] < max_seq_len_effective
            ]
            bar_start_idx = choice(population)
            cumsum_start = np.cumsum(bars_num_tokens[bar_start_idx:])
            bar_end_idx = (
                num_bars_per_seq - 1
                if cumsum_start[-1] < max_seq_len_effective
                else bar_start_idx
                     + np.nonzero(cumsum_start >= max_seq_len_effective)[0][0]
                     - 1
            )
            # Decrease bar idx to infill
            if bar_infilling:
                bar_infilling_start_idx -= bar_start_idx
                bar_infilling_end_idx -= bar_start_idx
            # s_len = sum(bars_num_tokens[i] for i in range(bar_start_idx, bar_end_idx))
            for i, (seq, bar_tokens_idxs) in reversed(
                    list(enumerate(zip(sequences, bar_tokens_idxs_per_seq)))
            ):
                # If the sequence ends before the start of the portion to extract begins
                # we remove it.
                if len(bar_tokens_idxs) - 1 < bar_start_idx:
                    del sequences[i], bar_tokens_idxs_per_seq[i]
                    if self.seq2seq and len(sequences) == 1 and not bar_infilling:
                        # We need at least two sequences to generate a new track in
                        # seq2seq. If there is only one remaining left (because some
                        # were deleted when handling the token sequence length above),
                        # we skip this training sample. Resorting to bar infilling would
                        # be possible by extracting the portion extracted here from the
                        # score and recompute the bar indexes to infill.
                        return None, None
                    if track_infilling_idx and i < track_infilling_idx:
                        track_infilling_idx -= 1
                    continue
                tok_idx_start = (
                    0 if bar_tokens_idxs == 0 else bar_tokens_idxs[bar_start_idx]
                )
                tok_idx_end = (
                    bar_tokens_idxs[bar_end_idx + 1]
                    if bar_end_idx + 1 < len(bar_tokens_idxs)
                    else -1
                )
                program_id = self.tokenizer.vocab[sequences[i].tokens[1]]
                sequences[i] = TokSequence(
                    ids=seq.ids[tok_idx_start:tok_idx_end], are_ids_encoded=True
                )
                # Add Track_Start, Program and Track_End tokens
                sequences[i].ids.insert(0, self._track_start_token_id)
                sequences[i].ids.insert(1, program_id)
                # Add track_end if bar_end_idx isn't the last bar as already included
                if bar_end_idx + 1 < num_bars_per_seq:
                    sequences[i].ids.append(self._track_end_token_id)
                # For debug
                # self.tokenizer.decode_token_ids(sequences[i])
                # self.tokenizer.complete_sequence(sequences[i])
            # seq_len_final = sum(len(seq) for seq in sequences)

        # If doing bar infilling we place extract the bars and create place the
        # right infilling tokens
        # Otherwise (track infilling), there is nothing to do here. If a user wants to
        # create a new track, we'll just have to add Track_Start and Program tokens at
        # the end of the sequence and generate from here.
        decoder_input = None
        if bar_infilling:
            # Bar infilling
            # Extract token section of the bars to infill
            # We do not select tracks that end before bar_tick_start
            bar_tokens_idxs = [
                i
                for i in range(len(sequences[track_infilling_idx]))
                if self._bar_token_byte
                   in self.tokenizer._model.id_to_token(
                    sequences[track_infilling_idx].ids[i]
                )[:2]
            ]
            # token_idx_start excludes Track_Start and attribute control token
            token_idx_start = bar_tokens_idxs[bar_infilling_start_idx]
            # excluding Track_End if last bar
            token_idx_end = (
                -1
                if bar_infilling_end_idx >= len(bar_tokens_idxs) - 1
                else bar_tokens_idxs[bar_infilling_end_idx]
            )

            if debug:
                for sequence in sequences:
                    self.tokenizer.decode_token_ids(sequence)

                with open('before_extraction.txt', 'w') as file:
                    # Write each item in the list on a new line
                    seq_number = 0
                    for sequence in sequences:
                        file.write(str(len(sequence.ids)) + "\n")
                        bar_counter = 0
                        for item in sequence.tokens:
                            if item == "Bar_None":
                                file.write(str(item) + f"{seq_number}_{bar_counter}\n")
                                bar_counter += 1
                            file.write(str(item) + "\n")
                        seq_number += 1

            # Extract the tokens of the section to infill and add BarFill start/end
            seq_infill = sequences[track_infilling_idx][token_idx_start:token_idx_end]
            seq_infill.ids.insert(0, self._infill_bar_start_token_id)
            seq_infill.ids.append(self._infill_bar_end_token_id)
            # seq2seq --> decoder infill / concat seq before after
            if self.seq2seq:
                decoder_input = seq_infill
            else:
                # Adding it at the end of the list that will be flattened
                sequences.append(seq_infill)

            # Add BarInfill tokens + update sequences
            seq_before = sequences[track_infilling_idx][:token_idx_start]
            for _ in range(infill_section_num_bars):
                seq_before.ids.append(self._infill_bar_token_id)
            seq_after = sequences[track_infilling_idx][token_idx_end:]
            sequences[track_infilling_idx] = seq_before + seq_after
        # If seq2seq, the last track sequence is fed to the decoder and a `Infill_Track`
        # token is appended to the encoder input sequence
        elif self.seq2seq:
            # There are always at least two sequences
            decoder_input = sequences.pop(-1)
            sequences[-1].ids.append(self._infill_track_token_id)

        # TODO External labels: (non-)expressive, loops, genres to add to the seq

        #for sequence in sequences:

        #    self.tokenizer.decode_token_ids(sequence)


        #tokens_infilling_track = self.tokenizer._ids_to_tokens(sequences[track_infilling_idx])
        # Preprocessing token ids: reduce sequence length, add BOS/EOS tokens

        #with open('after.txt', 'w') as file:
        #    file.write(f"keeping tracks {selected_tracks}")
        #    for sequence in sequences:
        #        file.write(str(len(sequence.ids)) + "\n")
        #        # Write each item in the list on a new line
        #        for item in sequence.tokens:
        #            file.write(str(item) + "\n")

        tokseq = concat_tokseq(sequences)
        # No need to call self._preprocess_token_ids as there are no BOS/EOS tokens and
        # that the sequence le  ngth does not exceed the limit as handled above.

        self.tokenizer.decode_token_ids(tokseq)

        tokseq_tokens = self.tokenizer._ids_to_tokens(tokseq)

        with open('output.txt', 'w') as file:
            for item in tokseq_tokens:
                file.write(str(item) + "\n")

        # [DEBUG] Reconstruct MIDI file from toksequence
        tokseq_tokens = np.array(tokseq.tokens)
        indices = np.where(tokseq_tokens == "Infill_Bar")[0]
        fill_bar_start_idx = np.where(tokseq_tokens == "FillBar_Start")[0][0]
        list_seqs = []
        list_seqs.append(tokseq[:indices[0]])
        list_seqs.append(tokseq[fill_bar_start_idx+1:-1])
        list_seqs.append(tokseq[indices[-1]+1:])
        midi_tok_seq = concat_tokseq(list_seqs)

        score_ = self.tokenizer._tokens_to_score(midi_tok_seq)
        score_.dump_midi("output_reconstructed.mid")

        return tokseq, decoder_input