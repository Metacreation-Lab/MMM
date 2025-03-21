#!/usr/bin/python3 python

"""
Script analyzing the programs and number of tracks of short files from the dataset.

Results:
Number of files with less than 8 bars: 589546
Program -1 (Drums): 368071 (0.549%)
Program 0 (Acoustic Grand Piano): 291196 (0.434%)
Program 1 (Bright Acoustic Piano): 110 (0.000%)
Program 2 (Electric Grand Piano): 75 (0.000%)
Program 3 (Honky-tonk Piano): 43 (0.000%)
Program 4 (Electric Piano 1): 93 (0.000%)
Program 5 (Electric Piano 2): 32 (0.000%)
Program 6 (Harpsichord): 71 (0.000%)
Program 7 (Clavi): 11 (0.000%)
Program 8 (Celesta): 14 (0.000%)
Program 9 (Glockenspiel): 26 (0.000%)
Program 10 (Music Box): 48 (0.000%)
Program 11 (Vibraphone): 123 (0.000%)
Program 12 (Marimba): 54 (0.000%)
Program 13 (Xylophone): 15 (0.000%)
Program 14 (Tubular Bells): 29 (0.000%)
Program 15 (Dulcimer): 34 (0.000%)
Program 16 (Drawbar Organ): 23 (0.000%)
Program 17 (Percussive Organ): 8 (0.000%)
Program 18 (Rock Organ): 51 (0.000%)
Program 19 (Church Organ): 150 (0.000%)
Program 20 (Reed Organ): 4 (0.000%)
Program 21 (Accordion): 13 (0.000%)
Program 22 (Harmonica): 17 (0.000%)
Program 23 (Tango Accordion): 16 (0.000%)
Program 24 (Acoustic Guitar (nylon)): 189 (0.000%)
Program 25 (Acoustic Guitar (steel)): 100 (0.000%)
Program 26 (Electric Guitar (jazz)): 84 (0.000%)
Program 27 (Electric Guitar (clean)): 90 (0.000%)
Program 28 (Electric Guitar (muted)): 47 (0.000%)
Program 29 (Overdriven Guitar): 73 (0.000%)
Program 30 (Distortion Guitar): 85 (0.000%)
Program 31 (Guitar Harmonics): 6 (0.000%)
Program 32 (Acoustic Bass): 172 (0.000%)
Program 33 (Electric Bass (finger)): 154 (0.000%)
Program 34 (Electric Bass (pick)): 34 (0.000%)
Program 35 (Fretless Bass): 63 (0.000%)
Program 36 (Slap Bass 1): 23 (0.000%)
Program 37 (Slap Bass 2): 24 (0.000%)
Program 38 (Synth Bass 1): 1727 (0.003%)
Program 39 (Synth Bass 2): 28 (0.000%)
Program 40 (Violin): 40 (0.000%)
Program 41 (Viola): 22 (0.000%)
Program 42 (Cello): 24 (0.000%)
Program 43 (Contrabass): 26 (0.000%)
Program 44 (Tremolo Strings): 45 (0.000%)
Program 45 (Pizzicato Strings): 32 (0.000%)
Program 46 (Orchestral Harp): 71 (0.000%)
Program 47 (Timpani): 101 (0.000%)
Program 48 (String Ensembles 1): 388 (0.001%)
Program 49 (String Ensembles 2): 71 (0.000%)
Program 50 (SynthStrings 1): 48 (0.000%)
Program 51 (SynthStrings 2): 25 (0.000%)
Program 52 (Choir Aahs): 79 (0.000%)
Program 53 (Voice Oohs): 16 (0.000%)
Program 54 (Synth Voice): 24 (0.000%)
Program 55 (Orchestra Hit): 49 (0.000%)
Program 56 (Trumpet): 202 (0.000%)
Program 57 (Trombone): 107 (0.000%)
Program 58 (Tuba): 49 (0.000%)
Program 59 (Muted Trumpet): 27 (0.000%)
Program 60 (French Horn): 116 (0.000%)
Program 61 (Brass Section): 115 (0.000%)
Program 62 (Synth Brass 1): 42 (0.000%)
Program 63 (Synth Brass 2): 19 (0.000%)
Program 64 (Soprano Sax): 6 (0.000%)
Program 65 (Alto Sax): 28 (0.000%)
Program 66 (Tenor Sax): 30 (0.000%)
Program 67 (Baritone Sax): 12 (0.000%)
Program 68 (Oboe): 52 (0.000%)
Program 69 (English Horn): 9 (0.000%)
Program 70 (Bassoon): 32 (0.000%)
Program 71 (Clarinet): 95 (0.000%)
Program 72 (Piccolo): 29 (0.000%)
Program 73 (Flute): 92 (0.000%)
Program 74 (Recorder): 7 (0.000%)
Program 75 (Pan Flute): 28 (0.000%)
Program 76 (Blown Bottle): 9 (0.000%)
Program 77 (Shakuhachi): 12 (0.000%)
Program 78 (Whistle): 12 (0.000%)
Program 79 (Ocarina): 26 (0.000%)
Program 80 (Lead 1 (square)): 2165 (0.003%)
Program 81 (Lead 2 (sawtooth)): 2037 (0.003%)
Program 82 (Lead 3 (calliope)): 13 (0.000%)
Program 83 (Lead 4 (chiff)): 4 (0.000%)
Program 84 (Lead 5 (charang)): 9 (0.000%)
Program 85 (Lead 6 (voice)): 4 (0.000%)
Program 86 (Lead 7 (fifths)): 3 (0.000%)
Program 87 (Lead 8 (bass + lead)): 31 (0.000%)
Program 88 (Pad 1 (new age)): 40 (0.000%)
Program 89 (Pad 2 (warm)): 37 (0.000%)
Program 90 (Pad 3 (polysynth)): 19 (0.000%)
Program 91 (Pad 4 (choir)): 9 (0.000%)
Program 92 (Pad 5 (bowed)): 47 (0.000%)
Program 93 (Pad 6 (metallic)): 11 (0.000%)
Program 94 (Pad 7 (halo)): 21 (0.000%)
Program 95 (Pad 8 (sweep)): 16 (0.000%)
Program 96 (FX 1 (rain)): 3 (0.000%)
Program 97 (FX 2 (soundtrack)): 9 (0.000%)
Program 98 (FX 3 (crystal)): 10 (0.000%)
Program 99 (FX 4 (atmosphere)): 12 (0.000%)
Program 100 (FX 5 (brightness)): 9 (0.000%)
Program 101 (FX 6 (goblins)): 24 (0.000%)
Program 102 (FX 7 (echoes)): 5 (0.000%)
Program 103 (FX 8 (sci-fi)): 4 (0.000%)
Program 104 (Sitar): 16 (0.000%)
Program 105 (Banjo): 13 (0.000%)
Program 106 (Shamisen): 7 (0.000%)
Program 107 (Koto): 6 (0.000%)
Program 108 (Kalimba): 12 (0.000%)
Program 109 (Bag pipe): 1 (0.000%)
Program 110 (Fiddle): 19 (0.000%)
Program 111 (Shanai): 3 (0.000%)
Program 112 (Tinkle Bell): 15 (0.000%)
Program 113 (Agogo): 4 (0.000%)
Program 114 (Steel Drums): 7 (0.000%)
Program 115 (Woodblock): 8 (0.000%)
Program 116 (Taiko Drum): 20 (0.000%)
Program 117 (Melodic Tom): 16 (0.000%)
Program 118 (Synth Drum): 9 (0.000%)
Program 119 (Reverse Cymbal): 17 (0.000%)
Program 120 (Guitar Fret Noise, Guitar Cutting Noise): 6 (0.000%)
Program 121 (Breath Noise, Flute Key Click): 2 (0.000%)
Program 122 (Seashore, Rain, Thunder, Wind, Stream, Bubbles): 6 (0.000%)
Program 123 (Bird Tweet, Dog, Horse Gallop): 3 (0.000%)
Program 124 (Telephone Ring, Door Creaking, Door, Scratch, Wind Chime): 7 (0.000%)
Program 125 (Helicopter, Car Sounds): 4 (0.000%)
Program 126 (Applause, Laughing, Screaming, Punch, Heart Beat, Footstep): 16 (0.000%)
Program 127 (Gunshot, Machine Gun, Lasergun, Explosion): 0 (0.000%)

When reversing the condition on the file duration (to keep long files):
Number of files with less than 8 bars: 279213
Program -1 (Drums): 247301 (0.126%)
Program 0 (Acoustic Grand Piano): 243676 (0.124%)
Program 1 (Bright Acoustic Piano): 21722 (0.011%)
Program 2 (Electric Grand Piano): 7453 (0.004%)
Program 3 (Honky-tonk Piano): 4806 (0.002%)
Program 4 (Electric Piano 1): 19245 (0.010%)
Program 5 (Electric Piano 2): 13067 (0.007%)
Program 6 (Harpsichord): 15961 (0.008%)
Program 7 (Clavi): 5560 (0.003%)
Program 8 (Celesta): 3726 (0.002%)
Program 9 (Glockenspiel): 9911 (0.005%)
Program 10 (Music Box): 3361 (0.002%)
Program 11 (Vibraphone): 16293 (0.008%)
Program 12 (Marimba): 7852 (0.004%)
Program 13 (Xylophone): 3917 (0.002%)
Program 14 (Tubular Bells): 6310 (0.003%)
Program 15 (Dulcimer): 1123 (0.001%)
Program 16 (Drawbar Organ): 7152 (0.004%)
Program 17 (Percussive Organ): 7511 (0.004%)
Program 18 (Rock Organ): 14286 (0.007%)
Program 19 (Church Organ): 4997 (0.003%)
Program 20 (Reed Organ): 1365 (0.001%)
Program 21 (Accordion): 8649 (0.004%)
Program 22 (Harmonica): 7059 (0.004%)
Program 23 (Tango Accordion): 2963 (0.002%)
Program 24 (Acoustic Guitar (nylon)): 30844 (0.016%)
Program 25 (Acoustic Guitar (steel)): 62442 (0.032%)
Program 26 (Electric Guitar (jazz)): 30134 (0.015%)
Program 27 (Electric Guitar (clean)): 39305 (0.020%)
Program 28 (Electric Guitar (muted)): 23762 (0.012%)
Program 29 (Overdriven Guitar): 28155 (0.014%)
Program 30 (Distortion Guitar): 28541 (0.015%)
Program 31 (Guitar Harmonics): 2644 (0.001%)
Program 32 (Acoustic Bass): 26662 (0.014%)
Program 33 (Electric Bass (finger)): 54744 (0.028%)
Program 34 (Electric Bass (pick)): 7300 (0.004%)
Program 35 (Fretless Bass): 26724 (0.014%)
Program 36 (Slap Bass 1): 3124 (0.002%)
Program 37 (Slap Bass 2): 2557 (0.001%)
Program 38 (Synth Bass 1): 12350 (0.006%)
Program 39 (Synth Bass 2): 7469 (0.004%)
Program 40 (Violin): 14198 (0.007%)
Program 41 (Viola): 5949 (0.003%)
Program 42 (Cello): 8222 (0.004%)
Program 43 (Contrabass): 6633 (0.003%)
Program 44 (Tremolo Strings): 5769 (0.003%)
Program 45 (Pizzicato Strings): 20000 (0.010%)
Program 46 (Orchestral Harp): 12108 (0.006%)
Program 47 (Timpani): 18025 (0.009%)
Program 48 (String Ensembles 1): 87048 (0.044%)
Program 49 (String Ensembles 2): 33319 (0.017%)
Program 50 (SynthStrings 1): 26076 (0.013%)
Program 51 (SynthStrings 2): 6654 (0.003%)
Program 52 (Choir Aahs): 45370 (0.023%)
Program 53 (Voice Oohs): 18223 (0.009%)
Program 54 (Synth Voice): 10778 (0.006%)
Program 55 (Orchestra Hit): 3813 (0.002%)
Program 56 (Trumpet): 47874 (0.024%)
Program 57 (Trombone): 40421 (0.021%)
Program 58 (Tuba): 17945 (0.009%)
Program 59 (Muted Trumpet): 5584 (0.003%)
Program 60 (French Horn): 34418 (0.018%)
Program 61 (Brass Section): 20338 (0.010%)
Program 62 (Synth Brass 1): 8776 (0.004%)
Program 63 (Synth Brass 2): 3898 (0.002%)
Program 64 (Soprano Sax): 4883 (0.002%)
Program 65 (Alto Sax): 28511 (0.015%)
Program 66 (Tenor Sax): 20751 (0.011%)
Program 67 (Baritone Sax): 9189 (0.005%)
Program 68 (Oboe): 23015 (0.012%)
Program 69 (English Horn): 4493 (0.002%)
Program 70 (Bassoon): 18090 (0.009%)
Program 71 (Clarinet): 40191 (0.021%)
Program 72 (Piccolo): 10367 (0.005%)
Program 73 (Flute): 43843 (0.022%)
Program 74 (Recorder): 3815 (0.002%)
Program 75 (Pan Flute): 8700 (0.004%)
Program 76 (Blown Bottle): 1210 (0.001%)
Program 77 (Shakuhachi): 1754 (0.001%)
Program 78 (Whistle): 3024 (0.002%)
Program 79 (Ocarina): 2934 (0.001%)
Program 80 (Lead 1 (square)): 11486 (0.006%)
Program 81 (Lead 2 (sawtooth)): 16702 (0.009%)
Program 82 (Lead 3 (calliope)): 7935 (0.004%)
Program 83 (Lead 4 (chiff)): 1143 (0.001%)
Program 84 (Lead 5 (charang)): 2186 (0.001%)
Program 85 (Lead 6 (voice)): 2405 (0.001%)
Program 86 (Lead 7 (fifths)): 617 (0.000%)
Program 87 (Lead 8 (bass + lead)): 7257 (0.004%)
Program 88 (Pad 1 (new age)): 8296 (0.004%)
Program 89 (Pad 2 (warm)): 10565 (0.005%)
Program 90 (Pad 3 (polysynth)): 5853 (0.003%)
Program 91 (Pad 4 (choir)): 6063 (0.003%)
Program 92 (Pad 5 (bowed)): 2008 (0.001%)
Program 93 (Pad 6 (metallic)): 2011 (0.001%)
Program 94 (Pad 7 (halo)): 2952 (0.002%)
Program 95 (Pad 8 (sweep)): 4357 (0.002%)
Program 96 (FX 1 (rain)): 1616 (0.001%)
Program 97 (FX 2 (soundtrack)): 761 (0.000%)
Program 98 (FX 3 (crystal)): 2101 (0.001%)
Program 99 (FX 4 (atmosphere)): 4091 (0.002%)
Program 100 (FX 5 (brightness)): 6566 (0.003%)
Program 101 (FX 6 (goblins)): 1170 (0.001%)
Program 102 (FX 7 (echoes)): 2688 (0.001%)
Program 103 (FX 8 (sci-fi)): 1336 (0.001%)
Program 104 (Sitar): 1857 (0.001%)
Program 105 (Banjo): 3734 (0.002%)
Program 106 (Shamisen): 1057 (0.001%)
Program 107 (Koto): 1416 (0.001%)
Program 108 (Kalimba): 1844 (0.001%)
Program 109 (Bag pipe): 819 (0.000%)
Program 110 (Fiddle): 2041 (0.001%)
Program 111 (Shanai): 362 (0.000%)
Program 112 (Tinkle Bell): 1044 (0.001%)
Program 113 (Agogo): 552 (0.000%)
Program 114 (Steel Drums): 1569 (0.001%)
Program 115 (Woodblock): 1216 (0.001%)
Program 116 (Taiko Drum): 2245 (0.001%)
Program 117 (Melodic Tom): 1599 (0.001%)
Program 118 (Synth Drum): 3203 (0.002%)
Program 119 (Reverse Cymbal): 10701 (0.005%)
Program 120 (Guitar Fret Noise, Guitar Cutting Noise): 2697 (0.001%)
Program 121 (Breath Noise, Flute Key Click): 548 (0.000%)
Program 122 (Seashore, Rain, Thunder, Wind, Stream, Bubbles): 3429 (0.002%)
Program 123 (Bird Tweet, Dog, Horse Gallop): 659 (0.000%)
Program 124 (Telephone Ring, Door Creaking, Door, Scratch, Wind Chime): 1642 (0.001%)
Program 125 (Helicopter, Car Sounds): 1261 (0.001%)
Program 126 (Applause, Laughing, Screaming, Punch, Heart Beat, Footstep): 1438 (0.001%)
Program 127 (Gunshot, Machine Gun, Lasergun, Explosion): 0 (0.000%)
"""

if __name__ == "__main__":
    from pathlib import Path

    import numpy as np
    from matplotlib import pyplot as plt
    from miditok.constants import MIDI_INSTRUMENTS, SCORE_LOADING_EXCEPTION
    from miditok.utils import get_bars_ticks
    from symusic import Score
    from tqdm import tqdm

    from scripts.utils.baselines import mmm_mistral
    from scripts.utils.constants import (
        MIN_NUM_BARS_FILE_VALID,
    )

    NUM_HIST_BINS = 50

    # Filter non-valid files
    dataset_files_paths = mmm_mistral.dataset_files_paths
    num_tracks, programs = [], []
    for file_path in tqdm(dataset_files_paths, desc="Reading MIDI files"):
        try:
            score = Score(file_path)
        except SCORE_LOADING_EXCEPTION:
            continue
        score = mmm_mistral.tokenizer.preprocess_score(score)
        if len(get_bars_ticks(score)) < MIN_NUM_BARS_FILE_VALID:
            continue

        num_tracks.append(len(score.tracks))
        programs += [-1 if track.is_drum else track.program for track in score.tracks]

    print(
        f"Number of files with less than {MIN_NUM_BARS_FILE_VALID} bars: "
        f"{len(num_tracks)}"
    )

    programs = np.array(programs)
    for program in range(-1, 128):
        num_occurrences = len(np.where(programs == program)[0])
        ratio = num_occurrences / len(programs)
        print(
            f"Program {program} ("
            f"{'Drums' if program == -1 else MIDI_INSTRUMENTS[program]['name']}): "
            f"{num_occurrences} ({ratio:.3f}%)"
        )

    # Plotting the distributions
    fig, ax = plt.subplots()
    ax.hist(num_tracks, bins=NUM_HIST_BINS)
    ax.grid(axis="y", linestyle="--", linewidth=0.6)
    ax.set_ylabel("Count files")
    ax.set_xlabel("Number of tracks")
    fig.savefig(Path("GigaMIDI_length_bars.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
