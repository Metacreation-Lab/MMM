"""Constants file."""

from torch.backends.mps import is_available as mps_available
from torch.cuda import is_available as cuda_available
from torch.cuda import is_bf16_supported

SEED = 433
MODEL_NAME = "MMM"
HF_USERNAME = "Metacreation"

SLURM_OUT_INFO_BASE_DIR="/home/daevide/scratch/MMM/logs"

# For MMD preprocessing
MIN_NUM_BARS_FILE_VALID = 8
MIN_NUM_NOTES_FILE_VALID = 100
DATA_AUGMENTATION_OFFSETS = (6, 2, 0)  # pitch, velocity, duration
DATA_CHUNK_NUM_OVERLAP_BARS = 1


# Tokenizer params (same as MidiTok expect for new constants)
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1},
    "num_velocities": 24,
    "special_tokens": [
        "PAD",
        "BOS",
        "EOS",
        "Infill_Bar",  # Indicates a bar to be filled in a seq
        "Infill_Track",  # Used in seq2seq to instruct the decoder to gen a new track
        "FillBar_Start",  # Start of the portion to infill (containing n bars)
        "FillBar_End",  # Ends the portion to infill
    ],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_pitch_intervals": False,  # cannot be used as extracting tokens in data loading
    "use_programs": True,
    "num_tempos": 48,
    "tempo_range": (50, 200),
    "programs": list(range(-1, 127)),
    "base_tokenizer": "REMI",
    "ac_polyphony_bar": True,
    "ac_polyphony_track": True,
    "ac_polyphony_min": 1,
    "ac_polyphony_max": 6,
    "ac_pitch_class_bar": True,
    "ac_note_density_track": True,
    "ac_note_density_track_min": 0,
    "ac_note_density_track_max": 18,
    "ac_note_density_bar": True,
    "ac_note_density_bar_max": 18,
    "ac_note_duration_bar": True,
    "ac_note_duration_track": True,
    "ac_repetition_track": True,
    "ac_repetition_track_num_bins": 10,
    "ac_repetition_track_num_consec_bars": 4,
}

# TOKENIZER TRAINING PARAMS
VOCAB_SIZE = 16000
ACS_RANDOM_RATIO_RANGE = (0.05, 0.9)
TRACKS_IDX_RANDOM_RATIO_RANGE = (0.1, 1)
BARS_IDX_RANDOM_RATIO_RANGE = (0.1, 0.7)
TRAINING_TOKENIZER_MAX_NUM_FILES = 50000

# MODEL SIZE (41558528 params for Mistral, 45957632 for T5, 33408000 for GPT2)
MAX_POSITION_EMBEDDINGS = 8192
EMBEDDING_SIZE = 512
FEEDFORWARD_SIZE = EMBEDDING_SIZE * 3
NUM_LAYERS = 8
NUM_LAYERS_SEQ2SEQ_ENCODER = 6
NUM_LAYERS_SEQ2SEQ_DECODER = 6
NUM_ATTENTION_HEADS = 8
NUM_KEY_VALUE_HEADS = NUM_ATTENTION_HEADS // 2
SLIDING_WINDOWS = 384

# Parameters for training a small model
# MODEL SIZE (13693824 params for Mistral (BPE), 11082624 (NO BPE))
VOCAB_SIZE_SMALL = 4000
MAX_POSITION_EMBEDDINGS_SMALL = 4096
EMBEDDING_SIZE_SMALL = 384
FEEDFORWARD_SIZE_SMALL = EMBEDDING_SIZE * 3
NUM_LAYERS_SMALL = 6
NUM_ATTENTION_HEADS_SMALL = 6
NUM_KEY_VALUE_HEADS_SMALL = NUM_ATTENTION_HEADS // 2
SLIDING_WINDOWS_SMALL = 384

# DATA CONFIGS
MAX_SEQ_LEN = 1024 * 2
TRACKS_SELECTION_RANDOM_RATIO_RANGE = (0.4, 1)
RATIO_BAR_INFILLING = 0.75
RATIOS_RANGE_BAR_INFILLING_DURATION = (0.1, 0.4)


# TRAINING PARAMS
DROPOUT = 0.1
BATCH_SIZE_PER_DEVICE_TRAIN = 64  # multiple of 64 for A100, 8 for other GPUs (V100)
BATCH_SIZE_PER_DEVICE_VALID = 128
DATALOADER_NUM_WORKERS = 8
DATALOADER_PERSISTENT_WORKERS = False
VALID_DELAY = 500
GRAD_ACC_STEPS = 1
EVAL_STRATEGY = "steps"
EVAL_ACCUMULATION_STEPS = None  # in case of CUDA OOM during eval
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 3.0
LABEL_SMOOTHING = 0.0
NUM_TRAIN_EPOCHS = 100
TRAINING_STEPS = -1  # unused
USE_CUDA = True
USE_MPS = False
BF16 = True  # Ampere and newer
BF16_EVAL = True
FP16 = False
FP16_EVAL = False
HALF_PRECISION_BACKEND = "auto"
DEEPSPEED = None  # set with argparse
TORCH_COMPILE = True
TORCH_COMPILE_BACKEND = None  # default to "inductor"
TORCH_COMPILE_MODE = None
GRADIENT_CHECKPOINTING = False
# https://pytorch.org/docs/stable/distributed.html
# DDP_BACKEND = None
DDP_FIND_UNUSED_PARAMETERS = False
DDP_BUCKET_CAP_MB = None  # default to 25mb
FULL_DETERMINISM = True
LOG_LEVEL = "debug"
LOGGING_STRATEGY = "steps"
LOG_STEPS_INTVL = 50
SAVE_STRATEGY = "steps"
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3
SAVE_SAFETENSOR = True
LOAD_BEST_MODEL_AT_END = False
DISABLE_TQDM = True
LEARNING_RATE = 6e-5
LR_SCHEDULER = "cosine_with_restarts"
WARMUP_RATIO = 0.10
REPORT_TO = ["tensorboard"]  # logging_dir will be set within Baseline class
PUSH_TO_HF_HUB = False
HUB_STRATEGY = "every_save"
HUB_PRIVATE_REPO = True
NEFTUNE_NOISE_ALPHA = True


# TEST PARAMS (for generation)
NUM_INFERENCES_EVAL = 256  # for evaluation metrics
NUM_INFERENCES_TEST = MAX_SEQ_LEN  # for tests after training, saving the results
# we use MAX_SEQ_LEN to make levenshtein computations more faithful


# GENERATION CONFIG (for validation and tests)
NUM_BEAMS = 1
TEMPERATURE_SAMPLING = 1.0
REPETITION_PENALTY = 1.2
TOP_K = 20
TOP_P = 0.95
EPSILON_CUTOFF = None
ETA_CUTOFF = None
MAX_NEW_TOKENS = 1000
MAX_LENGTH = 99999

# DEBUG PARAMS
"""MAX_POSITION_EMBEDDINGS = 2048
EMBEDDING_SIZE = 64
FEEDFORWARD_SIZE = EMBEDDING_SIZE * 4
NUM_LAYERS = 2
NUM_LAYERS_SEQ2SEQ_ENCODER = 2
NUM_LAYERS_SEQ2SEQ_DECODER = 2
NUM_ATTENTION_HEADS = 4
NUM_KEY_VALUE_HEADS = NUM_ATTENTION_HEADS // 2
SLIDING_WINDOWS = 128
BATCH_SIZE_PER_DEVICE_TRAIN = 8
BATCH_SIZE_PER_DEVICE_VALID = 16
VALID_DELAY = 20
NUM_TRAIN_EPOCHS = 2
LOG_STEPS_INTVL = 5
MAX_SEQ_LEN = 128
MAX_NUM_FILES_NUM_TOKENS_PER_NOTE = 10"""


# in case no GPU is available
if not cuda_available():
    FP16 = FP16_EVAL = BF16 = BF16_EVAL = USE_CUDA = False
elif (BF16 or BF16_EVAL) and not is_bf16_supported():
    BF16 = BF16_EVAL = False
    FP16 = FP16_EVAL = True
if USE_CUDA or not mps_available():
    USE_MPS = False
