# Reproducing GPT-2 (124M) from Scratch

A from-scratch PyTorch reproduction of GPT-2 (124M params), following Andrej Karpathy's build series. Trained on a Lambda Labs A100 on a 10B-token subset of FineWeb-Edu.

## What's implemented

- GPT-2 architecture from scratch (`train_gpt2.py`): token + positional embeddings, multi-head causal self-attention, MLP blocks, pre-LN residual stream, weight tying between input embedding and output projection
- Distributed data-parallel training (`train_gpt2_ddp.py`) for multi-GPU runs
- FineWeb-Edu data pipeline (`fineweb.py`): download, tokenize, and shard a 10B-token subset for training

## Training setup

- Hardware: 1x A100 (Lambda Labs)
- Dataset: FineWeb-Edu, 10B-token subset
- Framework: PyTorch

## Results

_Coming soon: training/validation loss curves, final loss compared to OpenAI's released GPT-2 124M checkpoint on the same eval set, and downstream eval (HellaSwag) if run._

## Setup

```bash
pip install -r requirements.txt
python fineweb.py        # download + tokenize dataset
python train_gpt2.py      # single-GPU training
# or
torchrun --standalone --nproc_per_node=<N> train_gpt2_ddp.py   # multi-GPU
```

## Reference

Built following Andrej Karpathy's ["Let's reproduce GPT-2"](https://github.com/karpathy/build-nanogpt) series.
