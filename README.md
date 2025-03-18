# MMM
Multi-track music machine implementation

## Usage example

```Python
from pathlib import Path
from mmm import MMM  # model
from miditok import MMM as MMM_T  # tokenizer
from symusic import Score  # MIDI file parsing

# Creating the model and the tokenizer
model = MMM.from_pretrained("metacreation/MMM")
tokenizer = MMM_T.from_pretrained("metacreation/MMM")

# Loading a MIDI file
score = Score(Path("to", "file.mid"))
tokens = tokenizer(score)
# TODO complete inference example
# gen_tokens = model.generate
# gen_score = tokenizer.decode(gen_tokens)
```

## Steps to reproduce

Before running these commands, make sure to load a virtual Python environment if needed.

### Install dependencies:

```bash
pip install ".[train]"
```

#### On Compute Canada:

```bash
module load python/3.11
virtualenv .venv
sbatch slurm/install_dependencies.sh
```

Flash attention might need to be installed from source (need to clone the github repository):

```bash
sbatch slurm/install_flashattention.sh
```

### Preparing the data

MMM is trained on the [GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI) dataset. On GPU clusters, the compute nodes usually can't access the internet. The dataset hence must be already downloaded before running the training itself on the nodes.

Some clusters may not have git lfs installed, thus it is easier to download the data with `huggingface_hub`: (can be installed via pip or brew)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download Metacreation/GigaMIDI --repo-type dataset
```

On Compute Canada, we save the dataset on $SCRATCH:

```bash
huggingface-cli download Metacreation/GigaMIDI --repo-type dataset --local-dir $SCRATCH/data/GigaMIDI
```

With git lfs:

```bash
git lfs install
git clone https://huggingface.co/datasets/Metacreation/GigaMIDI
```

### Training the model

#### On a Slurm cluster

It will use DeepSpeed to train the model on multiple GPUs.

```bash
sbatch --wait slurm/train_tokenizer.sh
sh scripts/train_model_loop.sh
```

#### Pure Python

```bash
python scripts/train_tokenizer.py
python scripts/train_model.py
```

## Pretrained model for inference
Download the model here:
https://drive.google.com/file/d/1Up6OxUANSUcBipPz_V9hLj6EiFTPCmiO/view?usp=sharing

Use generate.py to generate samples using the mentioned model.

## Data preprocessing

1. Filter non-valid files: corrupted or less than 8 bars;
2. Train the tokenizer on a subset of 100k files from the dataset, including Attribute Controls tokens computed for k randomly selected tracks and b randomly selected bars;
3. Split the dataset in train/valid/test subsets;
4. Split each file into chunks that make approximately 2048 tokens;
5. Augment each chunk on up/down to +-6 pitch intervals and -+2 velocities;

## Documentation

The documentation can be built with sphinx. You will need to install the required Python packages referenced in the [project file](pyproject.toml) under the "docs" category. Then run the command:

```bash
sphinx-build -M html docs docs/public
```
