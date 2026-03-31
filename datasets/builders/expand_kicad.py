#!/usr/bin/env python3
"""Expand KiCad dataset with synthetic EDA examples."""

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "processed" / "kicad_train.jsonl"
SYSTEM = "Tu es un expert en conception électronique avec KiCad. Réponds de manière technique et précise."

EXAMPLES = [
    ("Comment router un bus USB 2.0 sur un PCB 2 couches ?",
     """Routing USB 2.0 sur PCB 2 couches :

**Impédance différentielle :** 90Ω ±10%

**Règles de routage :**
- Pistes D+/D- en paire différentielle serrée
- Largeur : ~0.3mm pour 90Ω sur FR4 1.6mm
- Espacement entre D+ et D- : ~0.15mm
- Longueur maximale : 150mm (Low/Full speed)
- Pas de vias sur la paire différentielle

**Dans KiCad :**
1. Créer une netclass "USB" avec les règles d'impédance
2. Utiliser le routeur différentiel (touche 'D' après sélection)
3. Configurer les contraintes dans le DRC :
   ```
   (rule "USB_differential"
     (condition "A.NetClass == 'USB'")
     (constraint diff_pair_gap (min 0.15mm))
     (constraint diff_pair_uncoupled (max 5mm)))
   ```

**Bonnes pratiques :**
- Plan de masse continu sous les pistes USB
- Résistances de terminaison 22Ω en série près du contrôleur
- Connecteur USB en bord de carte"""),

    ("Comment créer une empreinte personnalisée dans KiCad ?",
     """Création d'empreinte dans l'éditeur de footprints KiCad :

1. **Ouvrir l'éditeur** : Outils → Éditeur d'empreintes
2. **Nouvelle empreinte** : Fichier → Nouvelle empreinte
3. **Ajouter les pads** :
   - Clic droit → Ajouter un pad
   - Type : SMD, Through-hole, ou NPTH
   - Forme : Rectangle, Rond, Ovale
   - Dimensions selon la datasheet

4. **Courtyard** (zone d'exclusion) :
   - Couche F.CrtYd
   - 0.25mm autour du composant minimum

5. **Silk screen** :
   - Couche F.SilkS
   - Contour du composant
   - Pin 1 marqueur (point ou trait)

6. **Fab layer** :
   - Couche F.Fab
   - Dimensions réelles du composant
   - Texte Reference et Value

7. **Modèle 3D** (optionnel) :
   - Propriétés → Modèles 3D
   - Importer .step ou .wrl

**Vérification :**
- DRC dans l'éditeur
- Comparer avec la datasheet du fabricant
- IPC-7351 pour les standards de pads"""),

    ("Comment configurer les règles DRC pour un PCB haute fréquence ?",
     """Règles DRC pour PCB haute fréquence (>100 MHz) :

```kicad_dru
(version 1)

(rule "HF_clearance"
  (condition "A.NetClass == 'HF' || B.NetClass == 'HF'")
  (constraint clearance (min 0.3mm)))

(rule "HF_trace_width"
  (condition "A.NetClass == 'HF'")
  (constraint track_width (min 0.2mm) (max 0.4mm)))

(rule "HF_via"
  (condition "A.NetClass == 'HF'")
  (constraint via_diameter (min 0.6mm))
  (constraint hole_size (min 0.3mm)))

(rule "HF_length_match"
  (condition "A.NetClass == 'DDR'")
  (constraint skew (max 0.5mm)))

(rule "no_stub"
  (condition "A.NetClass == 'HF'")
  (constraint disallow track_stub))
```

**Bonnes pratiques HF :**
- Stitching vias autour des pistes HF (tous les λ/20)
- Plan de masse continu sous les signaux critiques
- Éviter les changements de couche (vias = inductance parasite)
- Terminaisons adaptées (série, parallèle, Thévenin)"""),

    ("Comment générer des fichiers Gerber pour la fabrication ?",
     """Export Gerber dans KiCad 8 :

1. **Fichier → Tracer** (ou Ctrl+P dans l'éditeur PCB)

2. **Couches à exporter :**
   | Couche KiCad | Fichier Gerber | Extension |
   |---|---|---|
   | F.Cu | Cuivre face avant | .gtl |
   | B.Cu | Cuivre face arrière | .gbl |
   | F.SilkS | Sérigraphie avant | .gto |
   | B.SilkS | Sérigraphie arrière | .gbo |
   | F.Mask | Masque soudure avant | .gts |
   | B.Mask | Masque soudure arrière | .gbs |
   | Edge.Cuts | Contour carte | .gm1 |

3. **Options :**
   - Format : Gerber X2 (recommandé)
   - Coordonnées : auxiliaire origin
   - Cocher "Soustraire le masque de soudure des couches silk"

4. **Fichier de perçage :**
   - Générer le fichier de perçage
   - Format : Excellon
   - Unités : mm

5. **Vérification :**
   - Ouvrir dans un viewer Gerber (KiCad GerbView, ou en ligne)
   - Vérifier l'alignement des couches
   - Uploader sur le site du fabricant (JLCPCB, PCBWay)"""),

    ("Comment faire un plan de masse avec des zones de cuivre dans KiCad ?",
     """Zones de cuivre (plan de masse) dans KiCad :

1. **Créer la zone :**
   - Sélectionner la couche (B.Cu pour un plan de masse)
   - Outil "Ajouter une zone remplie" (ou raccourci)
   - Sélectionner le net : GND
   - Dessiner le contour (souvent le contour de la carte)

2. **Propriétés de la zone :**
   ```
   Clearance : 0.3mm (distance aux autres nets)
   Min width : 0.2mm
   Pad connections : Thermal relief
   Thermal relief gap : 0.3mm
   Thermal relief width : 0.25mm
   Fill type : Solid
   ```

3. **Remplir :** Édition → Remplir toutes les zones (ou 'B')

4. **Multi-zones :**
   - Priorité : zone avec numéro plus bas = priorité plus haute
   - Keep-out zones : interdire le cuivre dans certaines régions

5. **Bonnes pratiques :**
   - Plan de masse sur la couche opposée aux signaux
   - Pas de fentes dans le plan sous les pistes signal
   - Stitching vias pour connecter les plans GND entre couches"""),

    ("Qu'est-ce que le DFM et comment l'appliquer dans KiCad ?",
     """DFM (Design For Manufacturing) dans KiCad :

**Règles JLCPCB (standard) :**
- Track width min : 0.127mm (5 mil)
- Clearance min : 0.127mm
- Via drill min : 0.3mm
- Via pad min : 0.6mm
- Annular ring min : 0.13mm
- Silk text min : 1mm × 0.15mm
- Board outline radius min : 1mm
- Trou min : 0.3mm

**Dans KiCad — Design Rules :**
```
Fichier → Configuration de la carte → Design Rules

Default clearance : 0.2mm (safe margin)
Default track width : 0.25mm
Min track width : 0.15mm
Default via size : 0.8mm
Default via drill : 0.4mm
```

**Checklist avant commande :**
- [ ] DRC PASS (0 erreurs)
- [ ] Pas de silk sur les pads
- [ ] Courtyard sans chevauchement
- [ ] Fichiers Gerber vérifiés
- [ ] BOM avec références fabricant
- [ ] Pick & Place (position file) si assemblage SMD"""),

    ("Comment utiliser les bus dans les schématiques KiCad ?",
     """Bus schématiques dans KiCad :

**Bus simple (même signal) :**
1. Dessiner un bus (outil Bus, raccourci 'B')
2. Nommer le bus : `DATA[0..7]`
3. Connecter les fils au bus avec des entrées de bus
4. Nommer chaque fil : `DATA0`, `DATA1`, etc.

**Bus hiérarchique :**
1. Créer une feuille hiérarchique
2. Ajouter des pins hiérarchiques (entrées/sorties)
3. Connecter les pins aux bus du schéma parent

**Labels de bus :**
```
{ADDR[0..15]}     → Bus d'adresse 16 bits
{DATA[0..7]}      → Bus de données 8 bits
{SPI{MOSI,MISO,SCK,CS}}  → Bus SPI nommé
```

**Power flags :**
- Toujours ajouter `PWR_FLAG` sur les nets d'alimentation
- Sinon ERC signale "Power pin not driven"

**Bonnes pratiques :**
- Un bus par groupe fonctionnel
- Noms explicites (pas BUS1, BUS2)
- Annotations cohérentes (R1xx pour analog, R2xx pour digital)"""),

    ("Comment optimiser le placement des composants sur un PCB ?",
     """Stratégie de placement PCB :

**1. Placer d'abord les contraintes mécaniques :**
- Connecteurs en bord de carte
- Trous de fixation
- Composants avec contrainte thermique (régulateurs)

**2. Grouper par fonction :**
- Alimentation ensemble (régulateur + condensateurs)
- MCU au centre avec ses découplages
- Interface USB/Ethernet en bord de carte

**3. Règles de découplage :**
- Condensateur 100nF à < 3mm de chaque pin VCC
- Condensateur bulk (10µF) près du régulateur
- Ordre : MCU pin → 100nF → via → plan GND

**4. Signal flow :**
- Entrée → traitement → sortie (flux gauche→droite ou haut→bas)
- Minimiser les croisements de signaux
- Séparer analogique et digital

**5. Thermique :**
- Composants chauds avec plan de cuivre pour dissipation
- Pas de composants sensibles près des sources de chaleur
- Thermal vias sous les exposed pads

**Dans KiCad :**
- Utiliser "Répartir" pour aligner les composants
- Verrouiller les composants placés (clic droit → Verrouiller)
- Vérifier la ratsnest régulièrement"""),

    ("Comment configurer l'impédance contrôlée pour des pistes PCB ?",
     """Impédance contrôlée dans KiCad :

**Calcul de la largeur de piste :**
Utiliser le calculateur PCB intégré : Outils → Calculateur de pistes

Pour une microstrip sur FR4 (εr=4.4, h=1.6mm) :
- 50Ω → largeur ~2.9mm (en surface)
- 75Ω → largeur ~1.4mm
- 90Ω différentiel → 2×0.3mm avec espace 0.15mm

**Formule microstrip :**
```
Z0 = (87 / sqrt(εr + 1.41)) × ln(5.98h / (0.8w + t))
```

**Formule stripline :**
```
Z0 = (60 / sqrt(εr)) × ln(4b / (0.67π × (0.8w + t)))
```

**Dans KiCad — netclasses :**
```
Netclass "50Ω":
  Track width: 2.9mm
  Differential pair gap: 0.15mm

Netclass "90Ω_diff":
  Differential pair width: 0.3mm
  Differential pair gap: 0.15mm
```

**Commande au fabricant :**
Spécifier dans les notes de fabrication :
"Controlled impedance: 50Ω ±10% on outer layers, verify with coupon"

**Vérification :**
- TDR (Time Domain Reflectometry) par le fabricant
- Mesure sur coupon d'impédance dédié"""),

    ("Comment faire une simulation SPICE depuis KiCad ?",
     """Simulation SPICE depuis KiCad (ngspice intégré) :

**1. Préparation du schéma :**
- Utiliser des composants avec modèles SPICE (bibliothèques Simulation_SPICE)
- Ajouter une source de tension/courant (V, I, VPULSE, VSIN)
- Ajouter des directives SPICE (.tran, .ac, .dc)

**2. Ajouter une directive de simulation :**
Composant "SPICE_directive" → texte :
```spice
.tran 1us 10ms    ; simulation transitoire
.ac dec 100 1Hz 1GHz  ; analyse fréquentielle
.dc V1 0 5 0.1    ; balayage DC
```

**3. Lancer la simulation :**
Outils → Simulateur → Exécuter la simulation (ngspice)

**4. Visualiser les résultats :**
- Ajouter des sondes sur les nœuds à mesurer
- Courbes : tension, courant, puissance
- FFT disponible dans le visualisateur

**5. Exemples de sondes :**
```spice
.probe V(OUT) V(IN) I(R1)
```

**Limitations ngspice/KiCad :**
- Modèles SPICE Level 3 ou BSIM requis
- Certains composants n'ont pas de modèle intégré
- Pour RF : utiliser Qucs-S ou LTspice avec export netlist"""),
]


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            existing = [json.loads(line) for line in f if line.strip()]

    new_examples = []
    for i, (q, a) in enumerate(EXAMPLES):
        new_examples.append({
            "id": f"kicad-expanded-{i:04d}",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
        })

    all_ex = existing + new_examples
    with open(OUTPUT, "w") as f:
        for ex in all_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"KiCad dataset: {len(existing)} existing + {len(new_examples)} new = {len(all_ex)} total")


if __name__ == "__main__":
    build()
