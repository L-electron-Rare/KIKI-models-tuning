#!/usr/bin/env python3
"""Expand SPICE simulation dataset."""
import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "processed" / "spice_train.jsonl"
SYSTEM = "Tu es un expert en simulation de circuits électroniques avec SPICE (LTspice, ngspice). Réponds de manière technique et précise."

EXAMPLES = [
    (
        "Comment effectuer une analyse transitoire (.tran) dans LTspice ?",
        """L'analyse transitoire simule le comportement du circuit dans le temps. Voici comment la configurer dans LTspice.

**Syntaxe de base**
```spice
.tran <step> <stop> [start] [max_step]
```

**Exemple : circuit RC**
```spice
* Charge/décharge d'un condensateur RC
V1 in 0 PULSE(0 5 0 1n 1n 500u 1m)  ; Signal carré 1 kHz, 5V
R1 in mid 1k
C1 mid 0 100n

.tran 1u 5m          ; Pas de 1 µs, durée 5 ms
.probe V(mid) V(in)  ; Sauvegarder ces nœuds
.end
```

**Paramètres expliqués**
- `step` : pas de temps pour les résultats (ne pas confondre avec le pas de calcul interne)
- `stop` : temps de fin de simulation
- `start` : temps à partir duquel les données sont sauvegardées (utile pour omettre le transitoire initial)
- `max_step` : pas de calcul maximum (important pour la précision des signaux rapides)

**Signaux de source courants**
```spice
; Pulse (signal carré/trapézoïdal)
VPULSE n1 n2 PULSE(Vlow Vhigh Tdelay Trise Tfall Ton Tperiod)

; Sinus
VSIN   n1 n2 SIN(Voffset Vampl Freq Tdelay Theta Phase)

; PWL (Piecewise Linear — tension arbitraire)
VPWL   n1 n2 PWL(0 0 1u 3.3 2u 3.3 3u 0)
```

**Contrôle de la précision**
```spice
.options abstol=1n reltol=0.001 vntol=1u
```
- `reltol` : tolérance relative (défaut 0.001) — réduire pour plus de précision
- `abstol` : tolérance absolue en courant (1 nA par défaut)
- Problème de convergence → augmenter `reltol` à 0.01 ou ajouter `.options method=gear`

**Bonnes pratiques**
- Toujours vérifier le résultat au régime établi : est-il cohérent avec le calcul analytique ?
- Utiliser `.meas` pour extraire automatiquement les métriques :
```spice
.meas tran t_rise TRIG V(out) VAL=0.5 RISE=1 TARG V(out) VAL=2.5 RISE=1
```""",
    ),
    (
        "Comment simuler la réponse en fréquence d'un filtre actif avec .ac dans ngspice ?",
        """L'analyse AC calcule le gain et la phase d'un circuit en fonction de la fréquence. C'est l'outil central pour la conception de filtres.

**Syntaxe**
```spice
.ac <type> <points> <fstart> <fstop>
```
Types : `dec` (décades), `oct` (octaves), `lin` (linéaire)

**Exemple : filtre passe-bas actif Sallen-Key 2nd ordre**
```spice
* Filtre Sallen-Key Low Pass, f0=1kHz, Q=0.707 (Butterworth)
* Fc = 1/(2*pi*R*C) avec R=10k, C=15.9nF

Vin in 0 AC 1          ; Source AC unitaire (gain = 1 = 0 dB)

; Etage Sallen-Key
R1 in  n1  10k
R2 n1  n2  10k
C1 n1  out 15.9n
C2 n2  0   15.9n

; Op-amp idéal (buffer gain=1)
Eout out 0 VCVS n2 0 1e6   ; VCVS avec gain très élevé (simule un op-amp idéal)

.ac dec 100 10 1Meg         ; 100 points/décade de 10 Hz à 1 MHz
.probe V(out)
.end
```

**Op-amp réel dans ngspice**
```spice
* Inclure le modèle SPICE de l'op-amp (ex: TL071)
.lib "tl071.lib"
X1 inp inn vcc vss out TL071
```

**Analyse des résultats**
```spice
; Dans ngspice interactif :
run
plot db(V(out))          ; Gain en dB vs fréquence (diagramme de Bode amplitude)
plot ph(V(out))          ; Phase en degrés vs fréquence
```

**Mesures automatiques**
```spice
.meas ac f3db WHEN db(V(out))=-3 FALL=1    ; Fréquence de coupure à -3 dB
.meas ac gain_dc FIND db(V(out)) AT=1Hz    ; Gain continu
.meas ac phase_margin WHEN db(V(out))=-0 FALL=1 ; Marge de phase (pour boucle fermée)
```

**Pièges courants**
- La source AC doit avoir `AC 1` (pas de tension DC, ou séparée)
- Les condensateurs de couplage bloquent le DC mais n'apparaissent pas en .ac si la valeur est grande
- Un op-amp idéal (VCVS) évite les problèmes de convergence mais ne modélise pas la bande passante réelle
- Toujours vérifier le point de fonctionnement DC (`.op`) avant une analyse AC""",
    ),
    (
        "Comment simuler une alimentation à découpage (buck converter) dans LTspice ?",
        """La simulation d'un convertisseur buck permet de valider l'ondulation, la stabilité et le rendement avant fabrication.

**Schéma de principe**
```spice
* Buck Converter 12V → 5V, 2A, 100 kHz
* MOSFET + diode de roue libre

Vin  vin  0   DC 12
; Driver simplifié : source de tension commandée
Vdrv gate 0   PULSE(0 12 0 10n 10n 3.3u 10u)  ; D=0.33 → Vout=4V

; MOSFET (modèle simplifié)
M1  sw  gate vin  vin  NMOS W=1 L=1u
.model NMOS NMOS(VTO=2 KP=10 LAMBDA=0.01)

; Diode de roue libre
D1  0   sw   DIODE
.model DIODE D(IS=1n Rs=0.01)

; Filtre LC de sortie
L1  sw  out  100u  Rser=0.05   ; 100 µH avec résistance série 50 mΩ
C1  out 0    220u  Rser=0.01   ; 220 µF avec ESR 10 mΩ

; Charge résistive
Rload out 0  2.5    ; 2.5 Ω → 5V/2A = 2.5 Ω

.tran 100n 2m 1m     ; Simuler 2 ms, afficher seulement la dernière ms (régime établi)
.probe V(out) I(L1) I(Rload)
.end
```

**Modèle comportemental plus simple (averaged model)**
```spice
* Modèle moyen du buck (idéal, sans switching)
* Utilise le ratio de duty cycle directement
Vin  vin 0    DC 12
E1   sw  0    VALUE {V(vin) * 0.417}    ; D = Vout/Vin = 5/12 = 0.417
L1   sw  out  100u  Rser=0.05
C1   out 0    220u  Rser=0.01
Rload out 0   2.5

.tran 10u 10m         ; Analyse plus rapide — pas de switching HF à simuler
.ac dec 100 1 100k    ; Réponse fréquentielle de la boucle de régulation
.end
```

**Métriques à mesurer**
```spice
.meas tran Vout_avg   AVG V(out)  FROM 1m TO 2m   ; Tension moyenne de sortie
.meas tran Vripple    PP  V(out)  FROM 1m TO 2m   ; Ondulation peak-to-peak
.meas tran Iripple    PP  I(L1)   FROM 1m TO 2m   ; Ondulation courant inductance
```

**Conseils pratiques**
- Commencer avec un modèle MOSFET idéal, puis remplacer par le modèle SPICE réel du composant
- L'ESR du condensateur de sortie est crucial pour l'ondulation : `Rser=0` donne des résultats optimistes
- Vérifier que `Vout_avg` ≈ `Vin * D` (relation fondamentale du buck)
- L'analyse `.ac` du modèle moyen permet de concevoir la compensation du régulateur de tension""",
    ),
    (
        "Comment modéliser un op-amp non idéal dans SPICE ?",
        """Modéliser correctement un op-amp dans SPICE est essentiel pour simuler des comportements comme le slew rate, la bande passante et le bruit.

**Niveau 1 : Modèle VCVS idéal (ultra-simplifié)**
```spice
* Op-amp parfait : gain infini, bande passante infinie
Eout  out 0  VCVS  inp inn  1Meg
```
Utile pour les simulations fonctionnelles rapides, sans effets réels.

**Niveau 2 : Modèle comportemental avec GBP et slew rate**
```spice
* Op-amp avec GBP=1MHz et slew rate=10V/µs
* Approximation 1 pôle

.subckt OPAMP_1P  inp inn vcc vee out
Gin   mid  0   inn inp  1m         ; Transconductance d'entrée
Rout  mid  0   1Meg                ; Résistance de sortie interne
Cout  mid  0   159p                ; Cout = 1/(2*pi*Rout*GBP) = 1/(2*pi*1M*1M) ≈ 159p
Eout  out  0   VCVS  mid 0  1     ; Buffer de sortie
.ends OPAMP_1P

X1 inp inn vcc vee out OPAMP_1P
```

**Niveau 3 : Modèle SPICE complet (fourni par le fabricant)**
```spice
* Télécharger le modèle SPICE sur le site du fabricant (Texas Instruments, STMicroelectronics...)
.lib "TLV9001.lib"

X1  inp inn  VCC  GND  out  TLV9001
VCC VCC 0  DC 3.3
```

**Paramètres importants à vérifier dans le modèle**
- `GBP` (Gain-Bandwidth Product) : fréquence à laquelle le gain = 1
- `SR` (Slew Rate) : vitesse maximale de variation de sortie (V/µs)
- `Vos` (offset de tension) : tension d'entrée différentielle pour Vout = 0
- `Ib` (courant d'entrée de polarisation) : courant absorbé par les entrées
- Rails de sortie : un op-amp rail-to-rail peut approcher Vcc/Vss ; sinon il reste ~1–2 V en deçà

**Test du slew rate dans LTspice**
```spice
* Tester le slew rate d'un op-amp en configuration buffer
Vin in 0 PULSE(0 10 0 1n 1n 50u 100u)   ; Échelon rapide 0→10V
X1  in  out  VCC  GND  out  TL071        ; Buffer (non-inverseur, gain=1)
VCC VCC 0   DC 15
GND GND 0   DC -15

.tran 50n 200u
; Slew rate mesuré = ΔVout/Δt lors du front montant
.meas tran SR DERIV V(out) AT=10u
```

**Pièges fréquents**
- Ne pas oublier les alimentations Vcc/Vee dans les sous-circuits (modèles réels)
- Le modèle `VCVS` avec gain 1e6 est non-linéaire aux saturations : ajouter des diodes de limitation
- Les modèles fabricants peuvent ne pas inclure le bruit — utiliser `.noise` pour l'ajouter manuellement""",
    ),
    (
        "Comment résoudre les problèmes de convergence dans ngspice ?",
        """Les problèmes de convergence sont courants avec des circuits contenant des composants non-linéaires. Voici une approche méthodique.

**Symptômes**
- `ERROR: no convergence in transient analysis`
- `Singular matrix` lors de l'analyse DC
- Simulation qui s'arrête prématurément

**1. Ajuster les options de simulation**
```spice
.options reltol=0.01       ; Défaut 0.001 — assouplir si problèmes de convergence
.options abstol=1u         ; Tolérance absolue courant — assouplir (défaut 1p)
.options vntol=1m          ; Tolérance tension de nœud — assouplir (défaut 1µ)
.options itl4=100          ; Itérations max par pas transitoire (défaut 10)
.options itl5=5000         ; Itérations max totales (augmenter)
.options method=gear       ; Méthode Gear (meilleure stabilité que trapézoïdale pour les raideurs)
```

**2. Réduire le pas de simulation**
```spice
; Problème : signal rapide mal capturé
.tran 100n 1m              ; Trop grossier pour un signal à 10 MHz
; Solution : réduire le pas ET ajouter un pas max
.tran 10n 1m 0 1n          ; Pas 10 ns, pas max 1 ns
```

**3. Ajouter des résistances de stabilisation**
```spice
; Nœud flottant → ajouter une résistance vers la masse
Rbleed  problematic_node  0  1Meg

; Inductance → ajouter une résistance série
L1  in  out  10u  Rser=1m   ; Évite L avec impédance nulle à DC
```

**4. Conditionner les modèles de diodes/transistors**
```spice
; Modèle de diode avec paramètres de convergence
.model MYDIODE D(IS=1n Rs=1 N=1.5 CJO=1p)
;                         ^^^ Rs non nul pour éviter singularité
```

**5. Utiliser le ramping de source**
```spice
; Au lieu d'un échelon brusque, utiliser un front lent au démarrage
Vin in 0 PULSE(0 5 0 100n 100n 500u 1m)  ; Trise=100ns (pas 0)
;                       ^^^  pas de front instantané
```

**6. Source d'aide à la convergence (.ic)**
```spice
; Forcer les conditions initiales pour aider Newton-Raphson
.ic V(out)=5 V(gate)=0    ; Conditions initiales connues
; OU sur l'instance :
C1 out 0 100n IC=5         ; Condensateur pré-chargé à 5V
```

**Démarche recommandée**
1. Vérifier les nœuds flottants (`netlist check`)
2. Remplacer les composants non-linéaires par des modèles idéaux, puis les réintroduire un par un
3. Simuler d'abord en DC (`.op`) avant de lancer `.tran`
4. Si le problème persiste avec un MOSFET, utiliser `level=1` au lieu de `level=3` pour le modèle""",
    ),
    (
        "Comment simuler le bruit dans un circuit amplificateur avec SPICE ?",
        """L'analyse du bruit en SPICE permet de calculer la densité spectrale de bruit et le rapport signal/bruit d'un circuit.

**Types de bruit modélisés**
- **Bruit thermique (Johnson-Nyquist)** : bruit blanc des résistances — `v²/Hz = 4kTR`
- **Bruit de grenaille (Shot noise)** : jonctions PN — `i²/Hz = 2qI`
- **Bruit en 1/f (Flicker noise)** : composante basse fréquence des transistors et op-amps

**Commande .noise**
```spice
.noise V(output) Vsource dec 100 1 100Meg
;      ^^^^^^^^  ^^^^^^^  ^^^ ^^^
;      Nœud de  Source    Type Points Fstart Fstop
;      sortie   de réf.   log
```

**Exemple complet : amplificateur à transistor BJT**
```spice
* Amplificateur CE (Common Emitter) avec analyse bruit
Vin  in   0   AC 1 sin(0 1m 1k)  ; 1mV @ 1kHz
Vcc  vcc  0   DC 12

; Polarisation
R1   vcc  base  100k
R2   base 0     10k
Rc   vcc  coll  4.7k
Re   emit 0     1k
Ce   emit 0     100u   ; Court-circuit AC l'émetteur

Q1   coll base emit  0  BC547

; Modèle BJT avec paramètres de bruit
.model BC547 NPN(Is=1fA Bf=200 Rb=100 Rc=10 Re=1
+             Kf=1e-11 Af=1)  ; Kf, Af = paramètres flicker noise

.ac dec 100 1 100Meg
.noise V(coll) Vin dec 100 1 100Meg

.end
```

**Interprétation des résultats ngspice**
```spice
; Après simulation :
setplot noise1          ; Sélectionner le plot bruit
plot onoise_spectrum    ; Densité bruit sortie (V/√Hz)
plot inoise_spectrum    ; Densité bruit ramenée à l'entrée (V/√Hz)

; Bruit intégré (RMS) entre 1kHz et 100kHz :
.meas noise Vnoise_rms INTEG onoise_spectrum FROM 1k TO 100k
```

**Calcul du NF (Noise Figure)**
```spice
* Le NF est calculé comme :
* NF = 10*log10(SNR_in / SNR_out) = 10*log10(inoise²/(4kT*Rs))
* ngspice le calcule automatiquement avec .noise
```

**Paramètres de bruit des modèles SPICE**
```spice
; Résistance : bruit thermique automatique (pas de paramètre à ajouter)
; Diode : IS détermine le shot noise
; BJT :
.model QNAME NPN(
+  Kf=1e-11   ; Coefficient flicker noise (A)
+  Af=1       ; Exposant fréquentiel (1 = 1/f parfait)
+  Rb=50      ; Résistance de base (contribue au bruit thermique)
+)
```

**Règles pratiques**
- Minimiser `Rb` (résistance de base) pour réduire le bruit thermique en entrée
- Le flicker noise domine en dessous du "corner frequency" = `Kf*Ic / (2q*Kf*f)` environ
- Pour les op-amps : utiliser les modèles comportementaux incluant `en` et `in` (densités de bruit spécifiées dans la datasheet)""",
    ),
    (
        "Comment utiliser les paramètres .param et .step dans LTspice pour balayer des valeurs ?",
        """`.param` et `.step` permettent d'automatiser l'exploration de l'espace de conception sans modifier le netlist manuellement.

**Déclaration de paramètres**
```spice
.param R_val = 10k          ; Valeur fixe
.param C_val = {1/(2*pi*f0*R_val)}  ; Expression calculée — f0 doit être défini
.param f0 = 1k              ; Fréquence de coupure cible
```

**Utilisation dans le netlist**
```spice
R1   in  out  {R_val}      ; Accolades pour utiliser un paramètre
C1   out 0    {C_val}
```

**Balayage linéaire avec .step**
```spice
* Faire varier la résistance de 1k à 10k par pas de 1k
.step param R_val 1k 10k 1k
.tran 1u 10m
```

**Balayage en liste**
```spice
* Balayer des valeurs E24 spécifiques
.step param R_val LIST 1k 2.2k 4.7k 10k 22k 47k
```

**Balayage décimal (oct/dec)**
```spice
* Balayer la fréquence de coupure de 100 Hz à 100 kHz sur 3 décades
.step param f0 100 100k 10  ; Problème : .step est linéaire uniquement
; Solution : utiliser une variable de puissance
.step param n 0 2 0.5       ; n = 0, 0.5, 1, 1.5, 2
.param f0 = {100 * 10^n}    ; 100, 316, 1k, 3.16k, 10k, 31.6k, 100k Hz
```

**Exemple complet : optimisation filtre RC**
```spice
* Trouver la résistance qui donne f_-3dB = 1kHz avec C = 100nF
.param R_val = 10k
.param C_fix = 100n

Vin  in  0   AC 1
R1   in  out  {R_val}
C1   out 0    {C_fix}

.ac dec 50 100 100k
.step param R_val LIST 1k 1.59k 2.2k 4.7k 10k

; Mesurer la fréquence de coupure pour chaque valeur
.meas ac f3db WHEN db(V(out))=-3 FALL=1

.end
```

**Balayage de température**
```spice
.step temp -40 125 25       ; De -40°C à +125°C par pas de 25°C
.tran 1u 1m                 ; Vérifier le comportement sur la plage de température
```

**Montée en charge : .step + Monte Carlo**
```spice
.param R_tol = 0.05         ; Tolérance 5%
.param R_val = {10k * (1 + R_tol * mc(1,1))}  ; Distribution uniforme ±5%
; Utiliser .mc dans LTspice XVII pour les analyses Monte Carlo officielles
```

**Exploitation des résultats**
- Chaque `step` produit une courbe distincte superposée dans le grapheur LTspice
- Clic droit sur la légende → "Select Steps" pour filtrer
- Les résultats `.meas` sont exportables : View → SPICE Error Log""",
    ),
    (
        "Comment simuler un filtre LC avec pertes et vérifier sa réponse impulsionnelle ?",
        """La simulation d'un filtre LC avec pertes (résistances parasites) donne des résultats réalistes proches du comportement réel du composant.

**Modèle complet d'un composant LC réel**
```spice
* Inductance réelle : résistance DC de bobinage + capacité parasite
.subckt L_REAL  p  n  L_val=10u DCR=0.5 Cpar=5p
L1   p  mid    {L_val}
R1   mid n     {DCR}        ; Résistance DC de la bobine (mesurée au RLCmètre)
C1   p   n     {Cpar}       ; Capacité parasite entre les bornes
.ends

* Condensateur réel : ESR + ESL
.subckt C_REAL  p  n  C_val=100n ESR=0.05 ESL=2n
C1   mid n     {C_val}
R1   p   mid2  {ESR}        ; Equivalent Series Resistance
L1   mid2 mid  {ESL}        ; Equivalent Series Inductance (fils de raccordement)
.ends
```

**Filtre LC passe-bas 4th ordre (Butterworth) avec pertes**
```spice
* Filtre Butterworth 4ème ordre, f0=10kHz, Zs=50Ω

Vin  in   0   AC 1
Rs   in   n1  50            ; Résistance source

; Étage 1 : L1=573µH, C1=221nF
XL1  n1   n2   L_REAL  L_val=573u  DCR=2
XC1  n2   0    C_REAL  C_val=221n  ESR=0.05

; Étage 2 : L2=238µH, C2=532nF
XL2  n2   n3   L_REAL  L_val=238u  DCR=0.8
XC2  n3   0    C_REAL  C_val=532n  ESR=0.02

Rload n3  0    50           ; Charge 50Ω

.ac dec 100 1k 1Meg         ; Réponse en fréquence
.end
```

**Réponse impulsionnelle via analyse transitoire**
```spice
* Impulsion de Dirac approchée : pulse très courte, amplitude élevée
Vin  in   0   PULSE(0 100 0 1n 1n 10n 10m)   ; 100V, durée 10 ns ≈ δ(t)
; Énergie = V * T = 100 * 10n = 1 µJ — normaliser selon les besoins

.tran 10n 500u              ; Capturer la décroissance du filtre
.probe V(n3)
.end
```

**Calcul du facteur de qualité Q**
```spice
; Pour un résonateur LC simple :
.meas ac Q PARAM {f_res / bandwidth}
.meas ac f_res WHEN db(V(out))=max(db(V(out))) RISE=1
.meas ac bw3db_low  WHEN db(V(out))=-3 RISE=1
.meas ac bw3db_high WHEN db(V(out))=-3 FALL=1
.meas ac bandwidth PARAM {bw3db_high - bw3db_low}
```

**Comparaison idéal vs réel**
```spice
* Simuler avec DCR=0, ESR=0 (idéal) puis avec les vraies valeurs
.step param DCR_val LIST 0 0.1 0.5 2
XL1  n1 n2  L_REAL  L_val=573u  DCR={DCR_val}
```

**Résultats attendus**
- Les pertes (DCR, ESR) réduisent le facteur Q et amortissent le pic de résonance
- L'ESL du condensateur crée une résonance parasite (anti-résonance) à haute fréquence
- La réponse impulsionnelle montre les oscillations amorties caractéristiques du filtre
- Un filtre Butterworth optimisé a la réponse maximalement plate sans dépassement dans la bande passante""",
    ),
]


def build() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            existing = [json.loads(line) for line in f if line.strip()]

    existing_questions = {
        m["content"]
        for entry in existing
        for m in entry.get("messages", [])
        if m.get("role") == "user"
    }

    new_examples = []
    for question, answer in EXAMPLES:
        if question in existing_questions:
            continue
        entry = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }
        new_examples.append(entry)

    all_examples = existing + new_examples

    with open(OUTPUT, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(
        f"SPICE dataset: {len(existing)} existing + {len(new_examples)} new"
        f" = {len(all_examples)} total → {OUTPUT}"
    )
    return len(all_examples)


if __name__ == "__main__":
    build()
