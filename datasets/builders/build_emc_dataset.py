#!/usr/bin/env python3
"""Build fine-tuning dataset for EMC/EMI/RF domain.

Covers: EMI filtering, PCB shielding, impedance matching, ESD protection,
        antenna design, RF layout, EMC compliance testing.

Usage:
  python build_emc_dataset.py                    # Seeds only (~4 examples)
  python build_emc_dataset.py --with-hf          # Seeds + HuggingFace datasets
  python build_emc_dataset.py --max-samples 2000 # Limit HF samples per source
"""

import argparse
import json
import os
import re

SYSTEM_PROMPT = (
    "You are an expert EMC/RF engineer specializing in electromagnetic "
    "compatibility, EMI filtering, ESD protection circuit design, "
    "RF impedance matching, antenna design, PCB layout for EMC compliance, "
    "and regulatory testing (CE, FCC). You provide practical solutions "
    "with component selection, PCB layout guidelines, and filter calculations."
)

SEED_EXAMPLES = [
    {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {
                "from": "human",
                "value": "Design an EMI input filter for a 24V/5A DC power supply to pass conducted emissions testing (CISPR 32 Class B).",
            },
            {
                "from": "gpt",
                "value": """EMI input filter for 24V/5A DC supply — CISPR 32 Class B:

**Target attenuation:** 40-60dB at 150kHz-30MHz (CISPR 32 Band A/B)

**Filter topology:** Two-stage LC π-filter (for both common-mode and differential-mode)

```
      ┌──[Fuse]──[MOV]──┬──[Cx1]──┬──[Ldm]──┬──[Cx2]──┬──[Lcm]──┬──[Cx3]──┬── DC+
DC IN │                  │         │         │         │         │         │
      │                 [Cy1]     [Cy1]     [Cy2]     [Cy2]     [Cy3]     │
      │                  │   GND   │         │   GND   │         │        │
      └──[Fuse]──────────┴─────────┴──[Ldm]──┴─────────┴──[Lcm]──┴────────┴── DC-
```

**Component values:**

1. **Common-mode choke (Lcm):** 2 × 10mH, rated ≥ 5A
   - Impedance at 150kHz: ZCM = 2π × 150k × 10m = 9.4kΩ
   - Select: Würth 744824310 (2×10mH, 5A, 28mΩ DCR)

2. **Differential-mode inductor (Ldm):** 47µH, rated ≥ 6A
   - Provides DM filtering for switching noise
   - Select: Coilcraft MSS1278-473 (47µH, 7.2A, 19mΩ)

3. **X-capacitors (Cx) — differential mode:**
   - Cx1 = Cx2 = Cx3 = 1µF, 50V, X2 rated
   - Self-resonance should be above switching frequency
   - Select: Murata DE series 1µF/50V X2

4. **Y-capacitors (Cy) — common mode to chassis ground:**
   - Cy1 = Cy2 = Cy3 = 4.7nF, 250Vac, Y2 rated
   - Limited to 4.7nF for leakage current compliance (<3.5mA)
   - Select: Murata DE series 4.7nF/250Vac Y2

5. **MOV (surge protection):** 33V varistor across input
   - Clamp voltage: ~42V at 100A (8/20µs surge)
   - Select: Littelfuse TMOV34S331 (330V AC / 33V DC, 2.5kA)

**Filter attenuation estimate:**
- DM at 150kHz: ~40dB (Ldm + Cx resonance)
- CM at 150kHz: ~50dB (Lcm + Cy)
- DM at 1MHz: ~60dB
- CM at 1MHz: ~70dB

**PCB layout rules:**
- Filter MUST be placed at PCB edge, close to connector
- Separate input and output ground planes — connect only through filter
- Cy capacitors connect to chassis/enclosure ground, NOT signal ground
- Keep traces short to minimize parasitic inductance
- Shield filter section with ground pour on both layers""",
            },
        ]
    },
    {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {
                "from": "human",
                "value": "How to design ESD protection for USB 2.0 data lines? Include TVS selection criteria.",
            },
            {
                "from": "gpt",
                "value": """ESD protection for USB 2.0 — TVS diode array selection and layout:

**Requirements:**
- IEC 61000-4-2: ±8kV contact, ±15kV air discharge
- USB 2.0: 480Mbps (High-Speed), signal swing 0-3.3V
- Must not degrade signal integrity (low capacitance)

**TVS Selection Criteria:**

| Parameter | Requirement | Why |
|-----------|-------------|-----|
| Cline | < 3pF per line | USB 2.0 HS eye diagram spec |
| Vrwm | ≥ 5V (D+/D-) | Above max signal voltage |
| Vbr | > 6V | Don't trigger during normal operation |
| Vclamp | < 15V at 8A (8/20µs) | Protect downstream IC (abs max ~5.5V for PHY) |
| Ipp | > 8A | IEC 61000-4-2 Level 4 |
| Cinter-line | < 0.1pF | Minimize crosstalk D+ ↔ D- |

**Recommended TVS arrays:**

| Part | Cline | Vclamp@8A | Package | Price |
|------|-------|-----------|---------|-------|
| USBLC6-2SC6 (ST) | 1.5pF | 10V | SOT-23-6 | $0.15 |
| TPD2E2U06 (TI) | 0.5pF | 8V | X1SON-4 | $0.20 |
| PRTR5V0U2X (Nexperia) | 0.65pF | 5.5V | SOT-363 | $0.18 |
| IP4220CZ6 (Nexperia) | 1.0pF | 9V | SOT-457 | $0.12 |

**Best choice for USB 2.0 HS: TPD2E2U06** (lowest capacitance)

**Schematic:**
```
USB Connector            TVS             USB PHY (MCU)
                    ┌──────────┐
VBUS ──────────────┤          ├──── VBUS (to regulator)
D+   ──────────────┤TPD2E2U06 ├──── D+
D-   ──────────────┤          ├──── D-
GND  ──────────────┤          ├──── GND
                    └──────────┘
```

**PCB Layout (Critical!):**

```
┌─────────────────────────────────────────┐
│  USB         [TVS]        MCU           │
│  Conn  ─D+──┤   ├──D+──  USB_DP        │
│        ─D-──┤   ├──D-──  USB_DM        │
│              │GND│                      │
│  ════════════╧═══╧══════════════════    │
│  Chassis GND / Ground plane             │
└─────────────────────────────────────────┘
```

**Layout rules:**
1. TVS MUST be first component after USB connector (before any trace length)
2. Keep trace from connector to TVS < 5mm
3. Use continuous ground plane under USB traces
4. GND pad of TVS connects to chassis ground via short, wide trace
5. 90Ω differential impedance on D+/D- (USB 2.0 spec)
6. Length-match D+ and D- traces within 2mm
7. No vias between connector and TVS
8. Add 0402 series resistors (27Ω) after TVS for impedance matching""",
            },
        ]
    },
    {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {
                "from": "human",
                "value": "Design a 50Ω impedance matching network for a 433MHz antenna on a PCB.",
            },
            {
                "from": "gpt",
                "value": """433MHz antenna impedance matching — L-network design:

**Scenario:** Match SX1278 LoRa module (50Ω output) to PCB trace antenna (Zant = 35 - j15 Ω measured)

**Step 1: Smith Chart Analysis**

Normalized impedance: Zn = Zant/Z0 = (35-j15)/50 = 0.7 - j0.3

We need to match to center of Smith chart (50+j0Ω).

**Step 2: L-Network Design (series L + shunt C)**

```
SX1278 ─── [Ls] ─── ┬ ─── Antenna (35-j15Ω)
50Ω        series L  │
                    [Cp]
                      │
                     GND
```

Calculations:
- Target: transform 35-j15 to 50+j0
- Q factor: Q = √(Rhi/Rlo - 1) = √(50/35 - 1) = 0.655

Series inductor:
- Xs = Q × Rlo + Xant = 0.655 × 35 + 15 = 37.9Ω (compensate existing -j15)
- Ls = Xs / (2π × f) = 37.9 / (2π × 433M) = 13.9nH
- → Select: **15nH** 0402 inductor (Murata LQW15AN15NJ00)

Shunt capacitor:
- Bp = Q / Rhi = 0.655 / 50 = 13.1mS
- Cp = Bp / (2π × f) = 13.1m / (2π × 433M) = 4.82pF
- → Select: **4.7pF** 0402 NP0/C0G capacitor

**Step 3: Matching Network with Pi-Pad for Tuning**

In practice, use a π-network for adjustability:
```
SX1278 ──┬── [Ls] ──┬── [Cp2] ──┬── Antenna
         │          │            │
        [Cp1]      [Cp_opt]     │
         │          │            │
        GND        GND          GND
```

Populate only needed components:
- Start with Ls and Cp1 as calculated
- Cp2 and Cp_opt are 0Ω/DNP footprints for tuning with VNA

**Step 4: PCB Layout for 433MHz**

```
     ┌─────────────────────────────────────┐
     │  IC    [match]                      │
     │  RF ──┤Ls├──┬──→ antenna trace      │
     │  pad        │    (50Ω microstrip)   │
     │            [Cp]                     │
     │             │                       │
     │  ═══GND═══GND═══GND═══════════════ │
     │  (continuous ground plane, no cuts) │
     └─────────────────────────────────────┘
```

**PCB guidelines for 433MHz:**
1. **Microstrip impedance**: for 50Ω on FR4 (εr=4.4), 1.6mm board:
   - Trace width ≈ 2.9mm on outer layer over ground plane
2. **Ground plane**: solid, unbroken under entire RF path
3. **Via stitching**: GND vias every λ/20 = 35mm around RF section
4. **Component placement**: matching network within 5mm of IC RF pin
5. **No routing under RF traces** on adjacent layers
6. **Clearance**: keep digital signals >10mm from RF traces

**Verification:**
- Use VNA (NanoVNA) to measure S11 at antenna port
- Target: S11 < -10dB at 433MHz (VSWR < 2:1)
- Bandwidth: S11 < -10dB over 433±0.5MHz for LoRa""",
            },
        ]
    },
    {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {
                "from": "human",
                "value": "What are the key PCB layout rules for EMC compliance? Give me a practical checklist.",
            },
            {
                "from": "gpt",
                "value": """PCB EMC layout checklist — practical rules for CE/FCC compliance:

**Ground & Power:**
- [ ] Solid ground plane on layer 2 (4-layer) or layer 2+3 (6-layer)
- [ ] No splits in ground plane under signal traces (creates slot antennas)
- [ ] Star ground topology for mixed analog/digital: connect planes at one point
- [ ] Decoupling caps (100nF + 10µF) within 3mm of every IC power pin
- [ ] Ferrite beads between analog and digital power rails
- [ ] Power plane on inner layer, sandwiched between ground planes

**Signal Routing:**
- [ ] Route high-speed signals on layers adjacent to ground planes
- [ ] Controlled impedance for signals >50MHz (specify in fab notes)
- [ ] No signal traces crossing ground plane splits or gaps
- [ ] Maximum 2 layer changes per high-speed signal
- [ ] Return current path: add ground vias next to signal vias
- [ ] Length-match differential pairs within 5% (USB, Ethernet, LVDS)

**Clocks & High-Speed:**
- [ ] Series termination resistor (33-47Ω) at clock source
- [ ] Guard traces or ground pour around clock traces
- [ ] No clock traces near board edges (>10mm from edge)
- [ ] Spread-spectrum clocking (SSC) for non-critical clocks
- [ ] Route clocks on inner layers (shielded by ground planes)

**I/O & Connectors:**
- [ ] All I/O at one board edge — filter partition between inside/outside
- [ ] ESD protection (TVS) as first component after connector
- [ ] Common-mode choke on all cable interfaces
- [ ] Connector shield grounded to chassis via low-impedance path
- [ ] No unfiltered signals crossing the I/O filter boundary

**Power Supply:**
- [ ] EMI filter at power input (before DC-DC converter)
- [ ] Switching node area minimized (largest noise source)
- [ ] Boot-strap and gate drive traces as short as possible
- [ ] Snubber circuits on switching nodes if needed
- [ ] Shield can over DC-DC converter section (if tight margins)

**Board Edges & Mechanical:**
- [ ] Ground pour on all layers, stitched with vias every 5mm at edges
- [ ] No traces within 2mm of board edge (prevents edge radiation)
- [ ] Mounting holes connected to chassis ground
- [ ] Keepout zone around crystal oscillator (no routing underneath)

**Layer Stack Example (4-layer, optimal EMC):**
```
Layer 1: Signals (components side)
Layer 2: GROUND (continuous, unbroken)      ← most important
Layer 3: POWER (with ground fill)
Layer 4: Signals (bottom)
```

**Quick Checks Before Fab:**
1. DRC: no unconnected copper near board edges
2. All high-speed return currents have adjacent ground plane
3. No floating copper pours (all copper connected or removed)
4. Test points accessible for EMC pre-scan with near-field probe
5. Provisions for shield cans (footprints placed even if DNP)

**Common failures and fixes:**
| Failure | Freq Range | Likely Cause | Fix |
|---------|-----------|--------------|-----|
| CE >6dB | 30-100MHz | Ground plane gap | Fill the gap |
| RE >6dB | 100-300MHz | Clock harmonics | Add spread spectrum |
| RE >6dB | 300MHz-1GHz | Cable emission | Add CM choke on cable |
| CE >6dB | 150kHz-1MHz | SMPS fundamental | Larger input filter |""",
            },
        ]
    },
]


EMC_RF_KEYWORDS = [
    "emc",
    "emi",
    "esd",
    "electromagnetic",
    "interference",
    "radiated emission",
    "conducted emission",
    "cispr",
    "fcc",
    "ce mark",
    "shielding",
    "ferrite",
    "common mode",
    "differential mode",
    "tvs",
    "varistor",
    "surge",
    "transient",
    "impedance matching",
    "smith chart",
    "s-parameter",
    "s11",
    "vswr",
    "antenna",
    "rf",
    "radio frequency",
    "microstrip",
    "stripline",
    "pcb layout",
    "ground plane",
    "via stitching",
    "decoupling",
    "filter",
    "pi filter",
    "lc filter",
    "emi filter",
    "near field",
    "far field",
    "spectrum analyzer",
]


ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def sharegpt_to_openai(sample: dict) -> dict:
    """Convert ShareGPT format (conversations/from/value) to OpenAI chat format (messages/role/content)."""
    if "messages" in sample:
        return sample
    messages = [
        {"role": ROLE_MAP.get(turn["from"], turn["from"]), "content": turn["value"]}
        for turn in sample.get("conversations", [])
    ]
    return {"messages": messages}


def build_from_huggingface(max_samples: int) -> list[dict]:
    """Download and convert EMC/RF datasets from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] pip install datasets — skipping HF download")
        return []

    samples = []

    # 1. Electronics StackExchange (filter for EMC/RF topics)
    # Columns: Id, Tags, Answer, Body, Title, CreationDate
    print("  Downloading bshada/electronics.stackexchange.com (EMC/RF filter)...")
    try:
        ds = load_dataset(
            "bshada/electronics.stackexchange.com", split="train", streaming=True
        )
        count = 0
        for row in ds:
            title = row.get("Title", "")
            body = row.get("Body", "")
            answer = row.get("Answer", "")
            tags = row.get("Tags", "")
            if not title or not answer or len(answer) < 100:
                continue
            combined = (title + " " + body + " " + answer + " " + tags).lower()
            if not any(kw in combined for kw in EMC_RF_KEYWORDS):
                continue
            question = re.sub(r"<[^>]+>", "", title + "\n" + body).strip()
            answer_clean = re.sub(r"<[^>]+>", "", answer).strip()
            if len(answer_clean) < 80:
                continue
            samples.append(
                {
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "human", "value": question[:1000]},
                        {"from": "gpt", "value": answer_clean[:4000]},
                    ]
                }
            )
            count += 1
            if count >= max_samples:
                break
        print(f"    Got {count} Electronics SE (EMC/RF) examples")
    except Exception as e:
        print(f"    Failed: {e}")

    # 2. Electrical engineering dataset (STEM-AI, EMC/RF filter)
    print("  Downloading STEM-AI-mtl/Electrical-engineering (EMC/RF filter)...")
    try:
        ds = load_dataset(
            "STEM-AI-mtl/Electrical-engineering", split="train", streaming=True
        )
        count = 0
        for row in ds:
            question = (
                row.get("question", "")
                or row.get("instruction", "")
                or row.get("title", "")
            )
            answer = (
                row.get("answer", "")
                or row.get("output", "")
                or row.get("response", "")
            )
            if not question or not answer or len(answer) < 80:
                continue
            combined = (question + " " + answer).lower()
            if not any(kw in combined for kw in EMC_RF_KEYWORDS):
                continue
            answer = re.sub(r"<[^>]+>", "", answer)
            question = re.sub(r"<[^>]+>", "", question)
            samples.append(
                {
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "human", "value": question.strip()},
                        {"from": "gpt", "value": answer.strip()},
                    ]
                }
            )
            count += 1
            if count >= max_samples // 2:
                break
        print(f"    Got {count} STEM-AI EMC/RF examples")
    except Exception as e:
        print(f"    Failed: {e}")

    # 3. EEVblog forum posts (filter for EMC/RF)
    # Columns: url, thread_title, posts (list of dicts), post_count, domain, subdomain
    print("  Downloading nick007x/eevblog-posts (EMC/RF filter)...")
    try:
        ds = load_dataset("nick007x/eevblog-posts", split="train", streaming=True)
        count = 0
        for row in ds:
            title = row.get("thread_title", "")
            posts = row.get("posts", [])
            if not title or not posts or len(posts) < 2:
                continue
            if not any(kw in title.lower() for kw in EMC_RF_KEYWORDS):
                continue
            # Extract OP and best reply
            op_text = ""
            reply_text = ""
            for post in posts:
                if isinstance(post, dict):
                    content = (
                        post.get("content", "")
                        or post.get("body", "")
                        or post.get("text", "")
                    )
                    if post.get("is_op", False) and not op_text:
                        op_text = re.sub(r"<[^>]+>", "", str(content)).strip()
                    elif (
                        not post.get("is_op", False)
                        and not reply_text
                        and len(str(content)) > 100
                    ):
                        reply_text = re.sub(r"<[^>]+>", "", str(content)).strip()
            if not reply_text or len(reply_text) < 100:
                continue
            question = f"{title.strip()}\n{op_text[:500]}" if op_text else title.strip()
            samples.append(
                {
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": reply_text[:4000]},
                    ]
                }
            )
            count += 1
            if count >= max_samples // 2:
                break
        print(f"    Got {count} EEVblog EMC/RF examples")
    except Exception as e:
        print(f"    Failed: {e}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Build EMC/RF fine-tuning dataset")
    parser.add_argument(
        "--with-hf", action="store_true", help="Include HuggingFace datasets"
    )
    parser.add_argument(
        "--max-samples", type=int, default=2000, help="Max HF samples per source"
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, "datasets", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output or os.path.join(output_dir, "emc_train.jsonl")

    all_samples = list(SEED_EXAMPLES)
    print(f"  {len(SEED_EXAMPLES)} seed examples")

    if args.with_hf:
        hf_samples = build_from_huggingface(args.max_samples)
        all_samples.extend(hf_samples)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sharegpt_to_openai(sample), ensure_ascii=False) + "\n")

    print(f"\n  Wrote {len(all_samples)} examples to {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    if not args.with_hf:
        print("\n  To enrich: python build_emc_dataset.py --with-hf --max-samples 2000")


if __name__ == "__main__":
    main()
