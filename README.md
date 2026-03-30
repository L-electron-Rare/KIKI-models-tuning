# KIKI-models-tuning

Pipeline FineFab pour entrainement, evaluation et publication de modeles.

## Role
- Separer clairement entrainement et inference.
- Piloter les cycles CPT -> SFT -> RLVR.
- Versionner les modeles et datasets avec controles qualite.

## Stack
- Python 3.12+
- Tooling fine-tuning (datasets, evaluation, registry)

## Structure cible
- `src/`: orchestration et logique entrainement
- `datasets/`: donnees d'entrainement
- `scripts/`: runs, evaluation, publication

## Demarrage rapide
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Roadmap immediate
- Migrer pipeline fine-tuning depuis mascarade.
- Mettre en place model registry + hot-swap contract.
- Integrer controle qualite datasets.
