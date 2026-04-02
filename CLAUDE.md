# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

KIKI-models-tuning is the ML pipeline for fine-tuning LLMs — config registry, SFT trainer, dataset builders for domain expertise (STM32, KiCad, SPICE, FreeCAD, PlatformIO, embedded).

## Commands

```bash
source .venv/bin/activate
PYTHONPATH=$PWD:$PYTHONPATH pytest tests/ -v
```

## Architecture

`src/` contains config, registry, trainer, and 10 dataset builders. Training requires GPU (KXKM-AI RTX 4090). Uses Unsloth for efficient LoRA fine-tuning.
