# MSAI 437 Deep Learning Homeworks
1. HW1: Neural network from scratch in numpy + comparable PyTorch implementations
2. HW2: Autoencoder for generating novel emojis (PyTorch)
3. HW3: Transformer for solving elementary binary arithmetic problems specified as text strings.

## Local Installation

### Prerequisites
- Python 3.8 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- jupyter notebook server

### Setup

1. Install dependencies:
```bash
uv sync
```

2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

### Running the Projects

#### HW1
Run the jupyter notebook `hw1/chw_hw1.ipynb`

#### HW2
Run the jupyter notebook `hw2/hw2.ipynb`

#### HW3
```bash
python hw3/mathformer.py -d_model 128 -seqlen 20 -loadname weights_128 -n_layers 1 -heads 1
```

See [the hw3 readme](/hw3/README.md) for more info.

#### HW4
```bash
python hw4/starter.py
```