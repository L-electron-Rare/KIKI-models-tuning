# KIKI-models-tuning

Fine-tuning pipeline and model registry for domain-specific LLMs.

Part of the [FineFab](https://github.com/L-electron-Rare) platform (FineFab).

## What it does

- Orchestrates full fine-tuning cycles: CPT, SFT, RLVR
- Supports LoRA, QLoRA, and Unsloth training strategies
- Manages model versioning with quality gates and evaluation benchmarks
- Publishes production-ready models to the FineFab model registry
- Provides dataset validation and preprocessing tooling

## Tech stack

Python 3.12+ | Unsloth | PEFT/LoRA | Hugging Face Transformers | Weights & Biases

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Project structure

```
src/        Training orchestration and logic
datasets/   Training data and configs
scripts/    Run, evaluate, and publish workflows
```

## Related repos

| Repo | Role |
|------|------|
| [makelife-cad](https://github.com/L-electron-Rare/makelife-cad) | CAD/EDA web platform |
| [makelife-hard](https://github.com/L-electron-Rare/makelife-hard) | Hardware design (KiCad) |
| [makelife-firmware](https://github.com/L-electron-Rare/makelife-firmware) | Embedded firmware |
| [finefab-life](https://github.com/L-electron-Rare/finefab-life) | Integration runtime and ops |

## License

MIT
