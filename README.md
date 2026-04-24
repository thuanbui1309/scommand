# SCommander

Spiking Transformer with SSM-Attention Hybrid for Speech Command Recognition.

## Layout

```
.
├── scommander/        Python package (data, models, training, utils, ...)
├── scripts/           Setup + verification + anonymization
│   ├── setup.sh
│   ├── verify_env.py
│   └── anonymize_repo.sh
├── configs/           Experiment configs (OmegaConf YAML)
│   ├── base.yaml
│   ├── dataset/       Per-dataset overrides (SHD, SSC, GSC)
│   └── variant/       Per-variant overrides
├── tests/             Pytest suite (env, LIF forward, dataset shapes)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

One-button installer (creates conda env `scommander`, installs CUDA 12.8 toolkit, torch 2.7 cu128 wheels, spiking stack, mamba-ssm from source for Blackwell sm_120):

```bash
bash scripts/setup.sh
```

Flags:
- `--no-ssm` — skip mamba-ssm/causal-conv1d/cupy (CPU-friendly dev; Track C falls back to pure-PyTorch SSM)
- `--skip-cuda` — skip conda CUDA toolkit install (use system nvcc or module load)
- `--env-name=NAME` — custom conda env name
- `--python=3.10` — custom Python version (default 3.11)

Verify post-install:

```bash
conda activate scommander
python scripts/verify_env.py
pytest -q
```

## Target hardware

Production training: **RTX 5090 32GB** (Blackwell sm_120, driver ≥570, CUDA 12.8). GPU may be shared via NVIDIA MPS — effective VRAM ≈ 26GB.

Dev: any CUDA-capable GPU with `--no-ssm` flag for scaffold smoke tests.

## Datasets

SHD, SSC, and Google Speech Commands v2 (GSC). Downloaded automatically via `tonic` on first training run. Cache path: `data/tonic_cache/` (gitignored).

## License

MIT. Paper pending; double-blind anonymization via `scripts/anonymize_repo.sh`.
