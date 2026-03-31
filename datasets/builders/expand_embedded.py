#!/usr/bin/env python3
"""Expand embedded systems dataset."""
import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "processed" / "embedded_train.jsonl"
SYSTEM = "Tu es un expert en systèmes embarqués. Réponds de manière technique et précise."

EXAMPLES = [
    (
        "Quelles sont les différences entre FreeRTOS et Zephyr ?",
        """FreeRTOS et Zephyr sont deux RTOS open-source très répandus, mais avec des philosophies très différentes.

**FreeRTOS**
- Noyau minimaliste (~5 000 lignes de C), facile à porter sur n'importe quel MCU
- API simple : `xTaskCreate`, `xQueueSend`, `xSemaphoreTake`, `vTaskDelay`
- Licence MIT — peut être intégré dans des produits commerciaux sans restrictions
- Supporté nativement par STM32CubeIDE, ESP-IDF, AWS IoT
- Pas de système de build intégré : on l'importe dans son propre projet (Makefile/CMake)
- Gestion mémoire via `pvPortMalloc` / `vPortFree` avec plusieurs algorithmes (heap_1..5)

**Zephyr RTOS**
- Noyau plus complet : threads, pipes, message queues, memory slabs, poll()
- Système de build West (CMake + Kconfig) avec configuration matérielle en DeviceTree
- Support natif de centaines de cartes (ARM, RISC-V, x86, Xtensa)
- Pile réseau intégrée (IPv4/6, BLE, LoRa, Thread, Wi-Fi)
- Licence Apache 2.0 — utilisable commercialement
- Courbe d'apprentissage plus raide, mais ecosystème riche

**Quand choisir quoi ?**
- Projet simple, MCU contraint (< 64 KB RAM) → **FreeRTOS**
- Produit IoT avec connectivité réseau, BLE, OTA intégrés → **Zephyr**
- Besoin de conformité certifiable (IEC 61508, FDA) → **SAFERTOS** (FreeRTOS certifié)

```c
// FreeRTOS : créer une tâche
xTaskCreate(vMyTask, "MyTask", 256, NULL, tskIDLE_PRIORITY + 1, NULL);

// Zephyr : définir un thread statique
K_THREAD_DEFINE(my_tid, 512, my_entry, NULL, NULL, NULL, 5, 0, 0);
```

**Résumé des différences clés :**
| Critère          | FreeRTOS     | Zephyr          |
|------------------|--------------|-----------------|
| Empreinte RAM    | 2–4 KB min   | 8–32 KB min     |
| Build system     | Makefile/CMake | West (Kconfig) |
| DeviceTree       | Non          | Oui             |
| Pile réseau      | Via add-ons  | Intégrée        |
| Certification    | SAFERTOS     | En cours (PSA)  |""",
    ),
    (
        "Comment optimiser la consommation d'énergie d'un microcontrôleur ?",
        """L'optimisation de la consommation passe par plusieurs niveaux complémentaires.

**1. Choisir le bon mode de veille**
La plupart des MCU modernes offrent plusieurs niveaux :
- **Sleep** : CPU arrêté, périphériques actifs (réveils fréquents, ~mA)
- **Deep Sleep / Stop** : CPU + la plupart des horloges arrêtées, RAM conservée (~µA)
- **Standby / Hibernation** : tout éteint sauf RTC/backup domain (~1–2 µA)

```c
// Exemple STM32L4 — entrer en Stop 2 (< 1 µA)
HAL_PWR_EnterSTOPMode(PWR_LOWPOWERREGULATOR_ON, PWR_STOPENTRY_WFI);
// Réveil sur RTC Alarm ou pin EXTI
```

**2. Réduire la fréquence d'horloge**
- Utiliser le MSI (Multi-Speed Internal) oscillateur à basse fréquence au lieu de PLL
- Sur STM32L4, MSI à 100 kHz consomme ~3 µA actif vs ~10 mA à 80 MHz PLL
- Activer le clock gating pour les périphériques inutilisés

```c
// Désactiver une horloge périphérique non utilisée
__HAL_RCC_SPI2_CLK_DISABLE();
__HAL_RCC_USART3_CLK_DISABLE();
```

**3. Configurer les GPIOs correctement**
- Les GPIOs en mode flottant (input sans pull) consomment du courant par commutation
- Mettre toutes les GPIO inutilisées en sortie LOW ou en entrée analogique
- Désactiver les pull-up/pull-down internes si inutiles

**4. Optimiser le code applicatif**
- Maximiser le temps en veille (event-driven vs polling)
- Regrouper les traitements pour rester éveillé le moins longtemps possible
- Technique "burst processing" : traiter rapidement, puis dormir longtemps

```c
// Pattern recommandé pour nœud IoT sur batterie
while (1) {
    Read_Sensor_Data();         // Rapide
    Transmit_Over_LoRa();       // ~100 ms
    Schedule_Next_RTC_Alarm();  // Dans 60 s
    Enter_Deep_Sleep();         // ~59.9 s de veille
}
```

**5. Dimensionner correctement les régulateurs**
- Un LDO surdimensionné gaspille de l'énergie à charge légère
- Préférer un régulateur buck/boost à haute efficacité pour alimenter depuis batterie LiPo
- Couper l'alimentation des sous-systèmes (capteurs, radio) avec un transistor PMOS ou load switch

**Résultats typiques** sur un nœud IoT optimisé :
- Actif (traitement + LoRa TX) : ~50 mW pendant 200 ms
- Veille profonde (RTC actif) : 2 µA = 6 µW
- Autonomie avec batterie 1000 mAh : > 2 ans à 1 mesure/minute""",
    ),
    (
        "Comment implémenter un bootloader OTA sur ARM Cortex-M ?",
        """Un bootloader OTA (Over-The-Air) sur ARM Cortex-M se compose de deux parties : le bootloader lui-même et l'application.

**Architecture mémoire typique (STM32F4, 512 KB flash)**
```
0x08000000 ┌─────────────────┐ Bootloader (32 KB)
           │   Bootloader    │ — démarre toujours ici après reset
0x08008000 ├─────────────────┤ Application Slot A (220 KB)
           │   App Slot A    │ — version active
0x0803F000 ├─────────────────┤ Application Slot B (220 KB)
           │   App Slot B    │ — nouvelle version téléchargée
0x08076000 ├─────────────────┤ Metadata (8 KB)
           │   Metadata      │ — version, CRC, flag "à mettre à jour"
0x08078000 └─────────────────┘
```

**Flux de mise à jour OTA**
1. L'application télécharge le nouveau firmware dans le Slot B via Wi-Fi/BLE/UART
2. Elle vérifie l'intégrité (CRC32 ou SHA-256)
3. Elle écrit le flag "update_pending" dans la zone Metadata
4. Elle reset le MCU
5. Le bootloader lit les métadonnées, copie Slot B → Slot A, efface le flag
6. Il saute dans l'application mise à jour

**Code bootloader essentiel**
```c
typedef struct {
    uint32_t magic;          // 0xB007B007
    uint32_t version;
    uint32_t crc32;
    uint32_t update_pending; // 0x01 = à appliquer
    uint32_t slot_b_size;
} FirmwareMeta;

#define META_ADDR   0x08076000UL
#define SLOT_A_ADDR 0x08008000UL
#define SLOT_B_ADDR 0x0803F000UL

void Bootloader_Run(void) {
    FirmwareMeta *meta = (FirmwareMeta *)META_ADDR;

    if (meta->magic == 0xB007B007 && meta->update_pending == 0x01) {
        // Vérifier CRC du Slot B
        if (CRC32_Verify((uint8_t *)SLOT_B_ADDR, meta->slot_b_size, meta->crc32)) {
            Flash_CopySlot(SLOT_B_ADDR, SLOT_A_ADDR, meta->slot_b_size);
            meta->update_pending = 0x00; // Effacer le flag
        }
    }

    // Sauter dans l'application (Slot A)
    Jump_To_Application(SLOT_A_ADDR);
}

// Saut vers l'application
void Jump_To_Application(uint32_t app_addr) {
    // Vecteur de reset de l'application est à app_addr + 4
    uint32_t stack_ptr = *(volatile uint32_t *)(app_addr);
    uint32_t reset_handler = *(volatile uint32_t *)(app_addr + 4);

    // Désactiver les interruptions, reconfigurer SCB
    __disable_irq();
    SCB->VTOR = app_addr; // Déplacer la table des vecteurs
    __set_MSP(stack_ptr);
    ((void (*)(void))reset_handler)(); // Sauter
}
```

**Précautions critiques**
- Compiler le bootloader et l'application avec des adresses de base différentes (linker script)
- L'application doit reconfigurer `SCB->VTOR` à son adresse de démarrage
- Implémenter un "rollback" : si l'application ne démarre pas en X secondes, revenir à l'ancienne version
- Protéger le secteur bootloader en flash (write protection bits)""",
    ),
    (
        "Comment gérer la mémoire sur un système sans MMU ?",
        """Sur un MCU sans MMU (la grande majorité des Cortex-M0/M3/M4), la gestion mémoire est entièrement sous la responsabilité du développeur.

**Types d'allocation**

**1. Allocation statique (recommandée)**
Tout est déclaré à la compilation — zero fragmentation, déterministe :
```c
static uint8_t  rx_buffer[256];    // .bss (RAM initialisée à 0)
static uint8_t  tx_buffer[256] = {0};
const  uint8_t  lut[256] PROGMEM; // .rodata → flash (si pas de MMU)
```

**2. Allocation sur la pile (stack)**
Rapide mais limitée (typiquement 2–8 KB sur MCU) :
```c
void process_packet(void) {
    uint8_t tmp[64]; // Alloué sur la pile, libéré à la fin de la fonction
    parse_header(tmp);
}
```
Risque : **stack overflow** silencieux sur MCU sans MPU actif.

**3. Allocation dynamique (à utiliser avec précaution)**
```c
// FreeRTOS heap_4 (meilleur compromis anti-fragmentation)
void *buf = pvPortMalloc(128);
if (buf == NULL) { Error_Handler(); } // Toujours vérifier
// ...
vPortFree(buf);
```
Fragmentation possible sur des allocations/libérations répétées de tailles variées.

**Stratégies pour éviter les problèmes**

**Memory pools** — allouer des blocs de taille fixe depuis un pool statique :
```c
#define POOL_SIZE 16
#define BLOCK_SIZE 64

static uint8_t pool[POOL_SIZE][BLOCK_SIZE];
static bool    pool_used[POOL_SIZE] = {false};

void *pool_alloc(void) {
    for (int i = 0; i < POOL_SIZE; i++) {
        if (!pool_used[i]) { pool_used[i] = true; return pool[i]; }
    }
    return NULL; // Pool épuisé
}

void pool_free(void *ptr) {
    int i = ((uint8_t *)ptr - &pool[0][0]) / BLOCK_SIZE;
    pool_used[i] = false;
}
```

**Détection de stack overflow**
```c
// Remplir la pile avec un motif connu au démarrage
memset((void *)STACK_START, 0xAA, STACK_SIZE);

// Vérifier périodiquement
bool Stack_Overflow_Detected(void) {
    uint8_t *canary = (uint8_t *)STACK_START;
    return (*canary != 0xAA); // Écrasé = overflow
}
```

**Bonne pratique** : activer le MPU (Memory Protection Unit) sur Cortex-M3/M4 pour détecter les accès hors zones allouées — sans MMU, le MPU reste un filet de sécurité précieux.""",
    ),
    (
        "Quand utiliser MQTT vs CoAP pour l'IoT ?",
        """MQTT et CoAP sont deux protocoles de messagerie IoT populaires mais conçus pour des contextes différents.

**MQTT (Message Queuing Telemetry Transport)**
- Protocole publish/subscribe, fonctionne sur **TCP** (port 1883 / 8883 TLS)
- Architecture centralisée autour d'un **broker** (Mosquitto, EMQX, AWS IoT Core)
- QoS 0/1/2 pour la fiabilité des messages
- Persistance des messages ("retained") et "last will" pour détecter les déconnexions
- Sessions persistantes : le broker stocke les messages si le client est hors ligne

```python
# Client Python MQTT (paho)
import paho.mqtt.client as mqtt

client = mqtt.Client()
client.connect("broker.example.com", 1883)
client.publish("sensors/temperature", "23.5", qos=1)
client.loop_forever()
```

**CoAP (Constrained Application Protocol)**
- Protocole requête/réponse REST-like, fonctionne sur **UDP** (port 5683 / 5684 DTLS)
- Conçu pour les réseaux contraints : 6LoWPAN, Thread, LoWPAN, Zigbee IP
- Messages courts (header 4 octets seulement)
- Observe : mécanisme de souscription léger similaire à pub/sub
- Support natif du multicast UDP pour la découverte de services

```c
// CoAP request minimal avec libcoap
coap_context_t *ctx = coap_new_context(NULL);
coap_session_t *session = coap_new_client_session(ctx, NULL, &dst, COAP_PROTO_UDP);
coap_pdu_t *pdu = coap_new_pdu(COAP_MESSAGE_CON, COAP_REQUEST_CODE_GET, session);
coap_add_option(pdu, COAP_OPTION_URI_PATH, 4, (uint8_t *)"temp");
coap_send(session, pdu);
```

**Tableau de comparaison**
| Critère                | MQTT           | CoAP              |
|------------------------|----------------|-------------------|
| Transport              | TCP (fiable)   | UDP (léger)       |
| Architecture           | Pub/Sub broker | Client/Serveur    |
| Overhead header        | 2 octets min   | 4 octets min      |
| RAM nécessaire         | ~10–50 KB      | ~5–10 KB          |
| Réseau contraint       | Non optimal    | Conçu pour ça     |
| Interopérabilité REST  | Non native     | Oui (HTTP proxy)  |
| Cloud IoT (AWS/Azure)  | Natif          | Via passerelle    |

**Recommandations pratiques**
- Capteurs connectés à internet via Wi-Fi/4G → **MQTT** (brokers cloud disponibles, TLS, QoS)
- Réseau maillé 6LoWPAN / Thread, nodes très contraints (< 32 KB RAM) → **CoAP**
- Applications industrielles OPC-UA ou Modbus TCP → ni l'un ni l'autre, choisir OPC-UA""",
    ),
    (
        "Comment débugger un firmware avec JTAG/SWD ?",
        """JTAG et SWD sont deux interfaces de débogage matériel pour les MCU. Voici comment les utiliser efficacement.

**JTAG vs SWD**
- **JTAG** : 4 broches (TCK, TMS, TDI, TDO) + TRST optionnel — standard IEEE 1149.1, supporte les chaînes de scan
- **SWD** : 2 broches (SWDCLK, SWDIO) — spécifique ARM, plus compact, recommandé pour les nouveaux designs

**Outils courants**
- **J-Link** (Segger) : ultra-rapide, supporte JTAG+SWD, logiciel Ozone pour le débogage graphique
- **ST-Link/V3** : intégré sur les boards Nucleo/Discovery, SWD uniquement, gratuit
- **OpenOCD** : logiciel open-source qui pilote la plupart des sondes

**Débogage avec GDB + OpenOCD**
```bash
# Terminal 1 : lancer OpenOCD
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg

# Terminal 2 : connecter GDB
arm-none-eabi-gdb build/firmware.elf
(gdb) target remote :3333
(gdb) monitor reset halt    # Reset et pause au démarrage
(gdb) load                  # Flasher le firmware
(gdb) break main            # Breakpoint sur main()
(gdb) continue
```

**Commandes GDB essentielles**
```
break fonction          # Breakpoint sur une fonction
break fichier.c:42      # Breakpoint ligne 42
watch variable          # Watchpoint : pause quand la variable change
info registers          # Afficher tous les registres ARM
x/10xw 0x20000000      # Afficher 10 mots à l'adresse RAM
backtrace               # Pile d'appels (call stack)
print $sp               # Valeur du stack pointer
info threads            # (avec FreeRTOS aware plugin) liste les tasks
```

**Techniques avancées**

**SWO (Serial Wire Output)** — trace non-intrusive via ITM :
```c
// Envoyer des données de trace via ITM port 0
void ITM_Print(const char *s) {
    while (*s) {
        while (ITM->PORT[0].u32 == 0); // Attendre que le port soit libre
        ITM->PORT[0].u8 = *s++;
    }
}
```
Visible dans Ozone, STM32CubeIDE ou `openocd` avec `tpiu config`.

**Hard fault handler** pour capturer les crashes :
```c
void HardFault_Handler(void) {
    __ASM volatile(
        "TST LR, #4\n"
        "ITE EQ\n"
        "MRSEQ R0, MSP\n"
        "MRSNE R0, PSP\n"
        "B Hard_Fault_Handler_C\n"
    );
}
void Hard_Fault_Handler_C(uint32_t *stack) {
    // stack[6] = PC au moment du crash
    volatile uint32_t pc = stack[6];
    (void)pc; // Mettre un breakpoint ici pour voir le PC
    while(1);
}
```
Flasher puis inspecter `pc` avec GDB pour localiser l'instruction fautive.""",
    ),
    (
        "Comment implémenter un ring buffer thread-safe ?",
        """Un ring buffer (circular buffer) thread-safe est essentiel pour communiquer entre une ISR et le code principal, ou entre deux tâches RTOS.

**Version lock-free (ISR ↔ main, single-producer/single-consumer)**
Cette version ne nécessite aucun mutex grâce à l'atomicité des accès à `head` et `tail` sur des architectures 32 bits :

```c
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define RB_CAPACITY 256  // Doit être une puissance de 2

typedef struct {
    uint8_t  data[RB_CAPACITY];
    volatile uint32_t head; // Écrit par le producteur uniquement
    volatile uint32_t tail; // Écrit par le consommateur uniquement
} RingBuf;

// Taille occupée
static inline uint32_t rb_size(const RingBuf *rb) {
    return (rb->head - rb->tail) & (RB_CAPACITY - 1);
}

// Espace disponible
static inline uint32_t rb_free(const RingBuf *rb) {
    return RB_CAPACITY - 1 - rb_size(rb);
}

bool rb_push(RingBuf *rb, uint8_t byte) {
    if (rb_free(rb) == 0) return false; // Plein
    rb->data[rb->head & (RB_CAPACITY - 1)] = byte;
    // Barrière mémoire : s'assurer que la donnée est écrite AVANT head
    __sync_synchronize(); // ou __DMB() sur ARM
    rb->head++;
    return true;
}

bool rb_pop(RingBuf *rb, uint8_t *out) {
    if (rb_size(rb) == 0) return false; // Vide
    *out = rb->data[rb->tail & (RB_CAPACITY - 1)];
    __sync_synchronize();
    rb->tail++;
    return true;
}

// Version burst : lire N octets d'un coup
uint32_t rb_read_bulk(RingBuf *rb, uint8_t *dst, uint32_t max) {
    uint32_t available = rb_size(rb);
    uint32_t count = (available < max) ? available : max;
    for (uint32_t i = 0; i < count; i++) {
        dst[i] = rb->data[(rb->tail + i) & (RB_CAPACITY - 1)];
    }
    __sync_synchronize();
    rb->tail += count;
    return count;
}
```

**Version multi-producteur/consommateur (avec mutex FreeRTOS)**
```c
#include "FreeRTOS.h"
#include "semphr.h"

typedef struct {
    uint8_t           data[RB_CAPACITY];
    uint32_t          head, tail;
    SemaphoreHandle_t mutex;
    SemaphoreHandle_t not_full;   // Compte les places libres
    SemaphoreHandle_t not_empty;  // Compte les éléments disponibles
} RingBufMT;

void rb_mt_init(RingBufMT *rb) {
    rb->head = rb->tail = 0;
    rb->mutex     = xSemaphoreCreateMutex();
    rb->not_full  = xSemaphoreCreateCounting(RB_CAPACITY - 1, RB_CAPACITY - 1);
    rb->not_empty = xSemaphoreCreateCounting(RB_CAPACITY - 1, 0);
}

bool rb_mt_push(RingBufMT *rb, uint8_t byte, TickType_t timeout) {
    if (xSemaphoreTake(rb->not_full, timeout) != pdTRUE) return false;
    xSemaphoreTake(rb->mutex, portMAX_DELAY);
    rb->data[rb->head++ & (RB_CAPACITY - 1)] = byte;
    xSemaphoreGive(rb->mutex);
    xSemaphoreGive(rb->not_empty);
    return true;
}
```

**Points clés**
- Taille en puissance de 2 → masquage bitwise au lieu de modulo (pas de division en ISR)
- `volatile` + barrière mémoire (`__DMB()` sur ARM) pour éviter la réorganisation des accès par le compilateur
- La version lock-free est **uniquement** valide avec un seul producteur et un seul consommateur""",
    ),
    (
        "Quelle est la différence entre bare-metal et embedded Linux ?",
        """Le choix entre bare-metal et embedded Linux est fondamental et dépend des contraintes du projet.

**Bare-metal**
Exécution directe sur le matériel, sans OS entre l'application et le MCU :
- Démarrage en quelques millisecondes (reset → code en < 10 ms)
- Empreinte mémoire minimale : quelques Ko de RAM et flash
- Déterminisme temporel maximal (interruptions en dizaines de nanosecondes)
- Contrôle total des ressources matérielles
- Cibles : Cortex-M0/M3/M4, PIC, AVR, MSP430

```c
// Bare-metal : boucle principale sans OS
int main(void) {
    SystemClock_Config();
    GPIO_Init();
    UART_Init();

    while (1) {
        if (UART_DataAvailable()) {
            uint8_t cmd = UART_ReadByte();
            Process_Command(cmd);
        }
        Update_Outputs();
    }
}
```

**Embedded Linux**
Linux complet adapté aux plateformes embarquées (Raspberry Pi, BeagleBone, i.MX8, etc.) :
- MMU obligatoire → isolation des processus, protection mémoire
- Démarrage en quelques secondes (U-Boot + kernel + rootfs)
- Accès aux drivers standard, system calls POSIX, protocoles réseau complets
- Développement applicatif en Python, C++, Node.js, etc.
- Empreinte minimale : ~32 MB RAM, ~128 MB stockage (avec Buildroot ou Yocto minimal)

```python
# Embedded Linux : lecture GPIO via sysfs ou libgpiod
import gpiod
chip = gpiod.Chip('gpiochip0')
line = chip.get_line(17)
line.request(consumer="myapp", type=gpiod.LINE_REQ_DIR_OUT)
line.set_value(1)
```

**Tableau comparatif**
| Critère                  | Bare-metal              | Embedded Linux          |
|--------------------------|-------------------------|-------------------------|
| Temps de démarrage       | < 10 ms                 | 2–30 s (kernel boot)    |
| RAM minimale             | 2–256 KB                | 32–128 MB               |
| Déterminisme temps-réel  | Oui (µs)                | Non (sauf RT-Linux)     |
| Connectivité réseau      | Via bibliothèques        | TCP/IP natif            |
| Développement            | C/C++ bare-metal        | Tout langage POSIX      |
| Mise à jour OTA          | Complexe                | Facile (swupdate/mender)|
| Sécurité (isolation)     | Aucune                  | MMU + SELinux possible  |

**Architecture hybride courante (AMP)**
- MCU Cortex-M (bare-metal) pour les entrées/sorties temps-réel
- MPU/SoC Cortex-A (embedded Linux) pour l'interface utilisateur, le cloud, le traitement
- Communication entre les deux via UART, SPI, mémoire partagée, RPMsg (OpenAMP)

Exemple : STM32MP1 qui intègre un Cortex-A7 (Linux) + Cortex-M4 (bare-metal) dans le même SoC.""",
    ),
]


def build() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Load existing seeds
    existing = []
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            existing = [json.loads(line) for line in f if line.strip()]

    # Build new examples (skip duplicates by checking user content)
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
        f"Embedded dataset: {len(existing)} existing + {len(new_examples)} new"
        f" = {len(all_examples)} total → {OUTPUT}"
    )
    return len(all_examples)


if __name__ == "__main__":
    build()
