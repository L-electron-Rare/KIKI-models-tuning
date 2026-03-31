#!/usr/bin/env python3
"""Expand STM32 dataset with synthetic domain examples."""

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "processed" / "stm32_train.jsonl"
SYSTEM = (
    "You are an expert STM32 and ARM Cortex-M embedded systems engineer. "
    "You write production-ready C firmware using STM32 HAL/LL drivers, CMSIS, FreeRTOS, "
    "and bare-metal register-level programming. You provide complete, compilable code with "
    "proper error handling, correct register addresses, and clear technical explanations. "
    "You also write ARM Thumb-2 assembly when needed."
)

EXAMPLES = [
    (
        "Configure ADC1 on STM32F4 to read PA0 with 12-bit resolution using HAL.",
        """Here is a complete ADC1 setup for PA0 on STM32F4:

```c
#include "stm32f4xx_hal.h"

ADC_HandleTypeDef hadc1;

void ADC1_Init(void) {
    // 1. Enable clocks
    __HAL_RCC_ADC1_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // 2. Configure PA0 as analog input
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin  = GPIO_PIN_0;
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // 3. Configure ADC
    hadc1.Instance                   = ADC1;
    hadc1.Init.ClockPrescaler        = ADC_CLOCK_SYNC_PCLK_DIV4;
    hadc1.Init.Resolution            = ADC_RESOLUTION_12B;
    hadc1.Init.ScanConvMode          = DISABLE;
    hadc1.Init.ContinuousConvMode    = DISABLE;
    hadc1.Init.DiscontinuousConvMode = DISABLE;
    hadc1.Init.ExternalTrigConvEdge  = ADC_EXTERNALTRIGCONVEDGE_NONE;
    hadc1.Init.DataAlign             = ADC_DATAALIGN_RIGHT;
    hadc1.Init.NbrOfConversion       = 1;
    HAL_ADC_Init(&hadc1);

    // 4. Configure channel 0
    ADC_ChannelConfTypeDef sConfig = {0};
    sConfig.Channel      = ADC_CHANNEL_0;
    sConfig.Rank         = 1;
    sConfig.SamplingTime = ADC_SAMPLETIME_84CYCLES;
    HAL_ADC_ConfigChannel(&hadc1, &sConfig);
}

uint32_t ADC1_Read(void) {
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);
    uint32_t raw = HAL_ADC_GetValue(&hadc1);
    HAL_ADC_Stop(&hadc1);
    // raw: 0..4095, voltage = raw * 3.3f / 4095.0f
    return raw;
}
```

**Key points:**
- 12-bit resolution → values 0–4095 mapping 0–3.3 V
- `ADC_SAMPLETIME_84CYCLES` gives better accuracy for high-impedance sources
- Always call `HAL_ADC_Stop` after polling to allow re-triggering""",
    ),
    (
        "How do I use DMA with SPI to drive a display on STM32?",
        """Here is SPI + DMA TX for a display (e.g., ST7735) on STM32F4:

```c
#include "stm32f4xx_hal.h"

SPI_HandleTypeDef hspi1;
DMA_HandleTypeDef hdma_spi1_tx;

volatile uint8_t spi_dma_done = 1;

void SPI1_DMA_Init(void) {
    __HAL_RCC_SPI1_CLK_ENABLE();
    __HAL_RCC_DMA2_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // PA5=SCK, PA7=MOSI
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin       = GPIO_PIN_5 | GPIO_PIN_7;
    GPIO_InitStruct.Mode      = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF5_SPI1;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // SPI master, 8-bit, CPOL=0 CPHA=0
    hspi1.Instance               = SPI1;
    hspi1.Init.Mode              = SPI_MODE_MASTER;
    hspi1.Init.Direction         = SPI_DIRECTION_2LINES;
    hspi1.Init.DataSize          = SPI_DATASIZE_8BIT;
    hspi1.Init.CLKPolarity       = SPI_POLARITY_LOW;
    hspi1.Init.CLKPhase          = SPI_PHASE_1EDGE;
    hspi1.Init.NSS               = SPI_NSS_SOFT;
    hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_4; // 21 MHz @ 84 MHz APB2
    HAL_SPI_Init(&hspi1);

    // DMA2 Stream3 Channel3 for SPI1_TX
    hdma_spi1_tx.Instance                 = DMA2_Stream3;
    hdma_spi1_tx.Init.Channel             = DMA_CHANNEL_3;
    hdma_spi1_tx.Init.Direction           = DMA_MEMORY_TO_PERIPH;
    hdma_spi1_tx.Init.PeriphInc           = DMA_PINC_DISABLE;
    hdma_spi1_tx.Init.MemInc              = DMA_MINC_ENABLE;
    hdma_spi1_tx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_spi1_tx.Init.MemDataAlignment    = DMA_MDATAALIGN_BYTE;
    hdma_spi1_tx.Init.Mode                = DMA_NORMAL;
    hdma_spi1_tx.Init.Priority            = DMA_PRIORITY_HIGH;
    HAL_DMA_Init(&hdma_spi1_tx);
    __HAL_LINKDMA(&hspi1, hdmatx, hdma_spi1_tx);

    HAL_NVIC_SetPriority(DMA2_Stream3_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(DMA2_Stream3_IRQn);
}

void DMA2_Stream3_IRQHandler(void) {
    HAL_DMA_IRQHandler(&hdma_spi1_tx);
}

void HAL_SPI_TxCpltCallback(SPI_HandleTypeDef *hspi) {
    if (hspi->Instance == SPI1) {
        HAL_GPIO_WritePin(CS_GPIO_Port, CS_Pin, GPIO_PIN_SET); // deassert CS
        spi_dma_done = 1;
    }
}

void Display_SendBuffer(uint8_t *buf, uint16_t len) {
    while (!spi_dma_done);  // wait for previous transfer
    spi_dma_done = 0;
    HAL_GPIO_WritePin(CS_GPIO_Port, CS_Pin, GPIO_PIN_RESET); // assert CS
    HAL_SPI_Transmit_DMA(&hspi1, buf, len);
    // Returns immediately; callback fires on completion
}
```

**Benefits:** CPU is free during the entire frame transfer, enabling smooth ~60 fps updates on small displays.""",
    ),
    (
        "Implement an IWDG (Independent Watchdog) on STM32F4 with a 5-second timeout.",
        """Here is IWDG configuration for a 5-second timeout on STM32F4:

```c
#include "stm32f4xx_hal.h"

IWDG_HandleTypeDef hiwdg;

// STM32F4 IWDG clock: LSI ≈ 32 kHz
// Timeout = (Reload + 1) * Prescaler / LSI_Hz
// Target: 5 s → Reload = (5 * 32000) / 256 - 1 = 624 - 1 = 623

void IWDG_Init(void) {
    hiwdg.Instance       = IWDG;
    hiwdg.Init.Prescaler = IWDG_PRESCALER_256; // 256/32000 = 8 ms per tick
    hiwdg.Init.Reload    = 623;                // 624 ticks * 8 ms = 4.992 s ≈ 5 s
    if (HAL_IWDG_Init(&hiwdg) != HAL_OK) {
        Error_Handler();
    }
}

// Call this regularly — must be called within 5 seconds or MCU resets
void IWDG_Feed(void) {
    HAL_IWDG_Refresh(&hiwdg);
}

// Example main loop
int main(void) {
    HAL_Init();
    SystemClock_Config();
    IWDG_Init();

    while (1) {
        IWDG_Feed();          // Reset the watchdog counter
        DoApplicationWork();  // Must complete within 5 s
    }
}
```

**Important notes:**
- IWDG runs on LSI (not affected by main clock failures) — ideal for detecting software hangs
- Once started, IWDG **cannot be stopped** without a reset
- `HAL_IWDG_Init()` also starts the watchdog — call it last in initialization
- LSI frequency varies ±20% across temperature; use a conservative timeout margin""",
    ),
    (
        "Configure I2C1 on STM32F4 and read a register from a sensor at address 0x68.",
        """Here is I2C1 master read for a sensor (e.g., MPU-6050) on STM32F4:

```c
#include "stm32f4xx_hal.h"

I2C_HandleTypeDef hi2c1;

#define SENSOR_ADDR  (0x68 << 1)  // HAL uses 8-bit address
#define REG_WHO_AM_I  0x75

void I2C1_Init(void) {
    __HAL_RCC_I2C1_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    // PB8=SCL, PB9=SDA (open-drain, AF4)
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin       = GPIO_PIN_8 | GPIO_PIN_9;
    GPIO_InitStruct.Mode      = GPIO_MODE_AF_OD;
    GPIO_InitStruct.Pull      = GPIO_NOPULL; // external 4.7k pull-ups required
    GPIO_InitStruct.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    hi2c1.Instance             = I2C1;
    hi2c1.Init.ClockSpeed      = 400000;          // 400 kHz Fast Mode
    hi2c1.Init.DutyCycle       = I2C_DUTYCYCLE_2;
    hi2c1.Init.OwnAddress1     = 0;
    hi2c1.Init.AddressingMode  = I2C_ADDRESSINGMODE_7BIT;
    hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c1.Init.NoStretchMode   = I2C_NOSTRETCH_DISABLE;
    HAL_I2C_Init(&hi2c1);
}

// Write one register
HAL_StatusTypeDef Sensor_WriteReg(uint8_t reg, uint8_t value) {
    uint8_t buf[2] = {reg, value};
    return HAL_I2C_Master_Transmit(&hi2c1, SENSOR_ADDR, buf, 2, 10);
}

// Read one register
uint8_t Sensor_ReadReg(uint8_t reg) {
    uint8_t data = 0;
    HAL_I2C_Master_Transmit(&hi2c1, SENSOR_ADDR, &reg, 1, 10);
    HAL_I2C_Master_Receive(&hi2c1, SENSOR_ADDR, &data, 1, 10);
    return data;
}

// Read multiple registers (e.g., 6-byte accelerometer burst)
void Sensor_ReadBurst(uint8_t start_reg, uint8_t *buf, uint8_t len) {
    HAL_I2C_Master_Transmit(&hi2c1, SENSOR_ADDR, &start_reg, 1, 10);
    HAL_I2C_Master_Receive(&hi2c1, SENSOR_ADDR, buf, len, 50);
}
```

**Usage:**
```c
I2C1_Init();
uint8_t who = Sensor_ReadReg(REG_WHO_AM_I); // Should return 0x68
```""",
    ),
    (
        "How do I implement low-power sleep modes on STM32 to save battery?",
        """STM32 offers three main low-power modes; here is how to use each:

```c
#include "stm32f4xx_hal.h"

// --- 1. SLEEP mode: CPU halted, peripherals run ---
// Wake-up: any interrupt
void Enter_Sleep(void) {
    HAL_SuspendTick();          // Prevent SysTick from waking immediately
    HAL_PWR_EnableSleepOnExit();
    HAL_PWR_EnterSLEEPMode(PWR_MAINREGULATOR_ON, PWR_SLEEPENTRY_WFI);
    HAL_ResumeTick();
}

// --- 2. STOP mode: CPU + most clocks stopped, RAM retained ---
// Wake-up: EXTI line or RTC alarm
// Current: ~20 µA (typical STM32L4)
void Enter_Stop(void) {
    __HAL_RCC_PWR_CLK_ENABLE();
    HAL_SuspendTick();
    HAL_PWR_EnterSTOPMode(PWR_LOWPOWERREGULATOR_ON, PWR_STOPENTRY_WFI);
    // After wake-up, clocks are reset to HSI — restore system clock
    SystemClock_Config();
    HAL_ResumeTick();
}

// --- 3. STANDBY mode: deepest sleep, only RTC/backup RAM retained ---
// Wake-up: WKUP pin, RTC alarm, reset
// Current: ~2 µA; RAM lost on wake-up
void Enter_Standby(void) {
    __HAL_PWR_CLEAR_FLAG(PWR_FLAG_WU);
    HAL_PWR_EnableWakeUpPin(PWR_WAKEUP_PIN1);
    HAL_PWR_EnterSTANDBYMode();
    // Code never reaches here
}
```

**Typical strategy for a battery-powered sensor node:**
```c
while (1) {
    Read_Sensors();
    Transmit_Data();         // BLE/LoRa burst
    HAL_Delay(10);           // Flush UART
    Enter_Stop();            // Sleep until RTC alarm (e.g., every 60 s)
}
```

**Checklist before entering Stop:**
1. Flush all UART/SPI FIFOs
2. Disable unused GPIOs (set to analog input = lowest leakage)
3. Stop ADC, TIM, DMA
4. Configure the wake-up source (RTC alarm or EXTI)""",
    ),
    (
        "Implement a ring buffer for UART receive on STM32 without HAL.",
        """Here is a lock-free ring buffer for UART RX using bare-metal registers on STM32F4:

```c
#include "stm32f4xx.h"
#include <stdint.h>
#include <stdbool.h>

#define RB_SIZE 256  // Must be a power of 2

typedef struct {
    volatile uint8_t  buf[RB_SIZE];
    volatile uint16_t head;  // Written by ISR
    volatile uint16_t tail;  // Read by main code
} RingBuffer;

static RingBuffer rb = {0};

// Initialize USART2 at 115200, 8N1 (APB1 = 42 MHz)
void UART2_Init(void) {
    RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;

    // PA2=TX, PA3=RX — AF7
    GPIOA->MODER   |= (2U << 4) | (2U << 6);   // Alternate function
    GPIOA->AFR[0]  |= (7U << 8) | (7U << 12);  // AF7

    USART2->BRR  = 42000000 / 115200;  // Mantissa=22, Fraction=13 → ~115200
    USART2->CR1  = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE | USART_CR1_RXNEIE;

    NVIC_SetPriority(USART2_IRQn, 5);
    NVIC_EnableIRQ(USART2_IRQn);
}

void USART2_IRQHandler(void) {
    if (USART2->SR & USART_SR_RXNE) {
        uint8_t byte = (uint8_t)USART2->DR;
        uint16_t next_head = (rb.head + 1) & (RB_SIZE - 1);
        if (next_head != rb.tail) {  // Not full
            rb.buf[rb.head] = byte;
            rb.head = next_head;
        }
        // Overflow silently dropped — add OVF counter for production
    }
}

bool RB_ReadByte(uint8_t *out) {
    if (rb.tail == rb.head) return false;  // Empty
    *out = rb.buf[rb.tail];
    rb.tail = (rb.tail + 1) & (RB_SIZE - 1);
    return true;
}

uint16_t RB_Available(void) {
    return (rb.head - rb.tail) & (RB_SIZE - 1);
}
```

**Key design decisions:**
- Power-of-2 size allows `& (SIZE-1)` masking instead of modulo (no division in ISR)
- `head`/`tail` are `volatile` — the compiler will not cache them in registers
- Single-producer (ISR), single-consumer (main) makes this lock-free without atomics""",
    ),
    (
        "How do I configure TIM2 for PWM output on STM32F4?",
        """TIM2 PWM on PA5 (TIM2_CH1) at 50 Hz for a servo on STM32F4:

```c
#include "stm32f4xx_hal.h"

TIM_HandleTypeDef htim2;

// TIM2 runs on APB1 (84 MHz on STM32F4 with default config)
// Servo: 50 Hz, pulse 1–2 ms
// Period = 84 MHz / (Prescaler+1) / (ARR+1)
// Prescaler=839 → TIM clock = 100 kHz; ARR=1999 → 50 Hz; CCR=100..200 → 1..2 ms

void TIM2_PWM_Init(void) {
    __HAL_RCC_TIM2_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // PA5 = TIM2_CH1, AF1
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin       = GPIO_PIN_5;
    GPIO_InitStruct.Mode      = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull      = GPIO_NOPULL;
    GPIO_InitStruct.Speed     = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF1_TIM2;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    htim2.Instance               = TIM2;
    htim2.Init.Prescaler         = 839;   // 84 MHz / 840 = 100 kHz
    htim2.Init.CounterMode       = TIM_COUNTERMODE_UP;
    htim2.Init.Period            = 1999;  // 100 kHz / 2000 = 50 Hz
    htim2.Init.ClockDivision     = TIM_CLOCKDIVISION_DIV1;
    htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
    HAL_TIM_PWM_Init(&htim2);

    TIM_OC_InitTypeDef sConfigOC = {0};
    sConfigOC.OCMode     = TIM_OCMODE_PWM1;
    sConfigOC.Pulse      = 150;    // 1.5 ms → center position
    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
    sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
    HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1);

    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);
}

// Set servo angle: 0° = 100 (1 ms), 90° = 150 (1.5 ms), 180° = 200 (2 ms)
void Servo_SetPulse(uint32_t pulse_ticks) {
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, pulse_ticks);
}
```""",
    ),
    (
        "How do I use the STM32 CRC peripheral for data integrity checks?",
        """STM32 hardware CRC-32 (MPEG-2 polynomial) example:

```c
#include "stm32f4xx_hal.h"

CRC_HandleTypeDef hcrc;

void CRC_Init(void) {
    __HAL_RCC_CRC_CLK_ENABLE();
    hcrc.Instance = CRC;
    HAL_CRC_Init(&hcrc);
}

// Compute CRC32 over a byte buffer (zero-padded to 32-bit words)
uint32_t CRC32_Compute(const uint8_t *data, uint32_t len) {
    // HAL_CRC_Calculate accepts uint32_t words
    // For byte arrays, handle padding
    uint32_t full_words = len / 4;
    uint32_t remainder  = len % 4;

    uint32_t crc = HAL_CRC_Calculate(&hcrc, (uint32_t *)data, full_words);

    if (remainder > 0) {
        uint32_t last_word = 0;
        for (uint32_t i = 0; i < remainder; i++) {
            last_word |= ((uint32_t)data[full_words * 4 + i]) << (24 - 8 * i);
        }
        crc = HAL_CRC_Accumulate(&hcrc, &last_word, 1);
    }

    return crc;
}

// Practical use: verify firmware block integrity
bool Verify_Block(const uint8_t *block, uint32_t len, uint32_t expected_crc) {
    uint32_t computed = CRC32_Compute(block, len);
    return computed == expected_crc;
}
```

**Notes:**
- STM32 CRC uses the **MPEG-2** polynomial (0x04C11DB7), **not** standard CRC-32b (0xEDB88320 reversed)
- Results will differ from standard tools (zlib, Python binascii) unless you configure input/output reflection on STM32H7+
- The peripheral resets automatically in `HAL_CRC_Calculate`; use `HAL_CRC_Accumulate` for streaming data""",
    ),
    (
        "Implement a FreeRTOS mutex to protect shared SPI bus access on STM32.",
        """FreeRTOS mutex for shared SPI bus between two tasks:

```c
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "stm32f4xx_hal.h"

extern SPI_HandleTypeDef hspi1;
static SemaphoreHandle_t spi_mutex = NULL;

void SPI_Bus_Init(void) {
    spi_mutex = xSemaphoreCreateMutex();
    configASSERT(spi_mutex != NULL);
}

// Thread-safe SPI transaction
bool SPI_Transfer(uint8_t *tx, uint8_t *rx, uint16_t len, TickType_t timeout) {
    if (xSemaphoreTake(spi_mutex, timeout) != pdTRUE) {
        return false;  // Timeout — bus busy
    }

    HAL_StatusTypeDef status;
    if (rx != NULL) {
        status = HAL_SPI_TransmitReceive(&hspi1, tx, rx, len, 100);
    } else {
        status = HAL_SPI_Transmit(&hspi1, tx, len, 100);
    }

    xSemaphoreGive(spi_mutex);
    return status == HAL_OK;
}

// Task A — reads display sensor
void Task_Display(void *pvParameters) {
    uint8_t cmd[4] = {0x01, 0x00, 0x00, 0x00};
    for (;;) {
        SPI_Transfer(cmd, NULL, sizeof(cmd), pdMS_TO_TICKS(50));
        vTaskDelay(pdMS_TO_TICKS(16));  // ~60 fps
    }
}

// Task B — reads SD card
void Task_SD(void *pvParameters) {
    uint8_t tx[512], rx[512];
    for (;;) {
        SPI_Transfer(tx, rx, sizeof(tx), pdMS_TO_TICKS(500));
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    SPI1_Init();
    SPI_Bus_Init();

    xTaskCreate(Task_Display, "Display", 256, NULL, 2, NULL);
    xTaskCreate(Task_SD,      "SD",      512, NULL, 1, NULL);
    vTaskStartScheduler();
}
```

**Design notes:**
- Use `xSemaphoreCreateMutex()` (not binary semaphore) to get **priority inheritance**, preventing priority inversion
- Never call `HAL_SPI_*` directly from tasks — always go through the guarded wrapper
- Set CS pin inside the guarded section if multiple devices share the bus""",
    ),
    (
        "How do I implement flash memory read/write on STM32F4 internal flash?",
        """Internal flash read/write on STM32F4 using HAL:

```c
#include "stm32f4xx_hal.h"

// STM32F405: Sector 11 starts at 0x080E0000, size 128 KB
#define FLASH_SECTOR    FLASH_SECTOR_11
#define FLASH_ADDRESS   0x080E0000UL

// Read is simple — memory-mapped, just dereference the pointer
uint32_t Flash_ReadWord(uint32_t address) {
    return *(__IO uint32_t *)address;
}

// Erase one sector before writing
HAL_StatusTypeDef Flash_EraseSector(void) {
    HAL_FLASH_Unlock();

    FLASH_EraseInitTypeDef erase_init;
    erase_init.TypeErase    = FLASH_TYPEERASE_SECTORS;
    erase_init.VoltageRange = FLASH_VOLTAGE_RANGE_3; // 2.7–3.6 V
    erase_init.Sector       = FLASH_SECTOR;
    erase_init.NbSectors    = 1;

    uint32_t sector_error = 0;
    HAL_StatusTypeDef status = HAL_FLASHEx_Erase(&erase_init, &sector_error);

    HAL_FLASH_Lock();
    return status;
}

// Write a buffer of 32-bit words
HAL_StatusTypeDef Flash_WriteBuffer(uint32_t address, const uint32_t *data, uint32_t count) {
    HAL_FLASH_Unlock();

    HAL_StatusTypeDef status = HAL_OK;
    for (uint32_t i = 0; i < count && status == HAL_OK; i++) {
        status = HAL_FLASH_Program(FLASH_TYPEPROGRAM_WORD,
                                   address + i * 4,
                                   data[i]);
    }

    HAL_FLASH_Lock();
    return status;
}
```

**Critical rules:**
1. Always **erase before write** — flash bits can only be cleared (1→0) by programming; erasing resets all bits to 1
2. **Never execute code from a sector while erasing it** — use the pragma `__attribute__((section(".ccmram")))` for the erase function if running from the same flash bank
3. Interrupts are **not disabled** automatically during erase — consider disabling them for deterministic timing
4. Flash endurance: typically **10,000 erase cycles** — use wear-leveling for frequently updated data""",
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
        f"STM32 dataset: {len(existing)} existing + {len(new_examples)} new"
        f" = {len(all_examples)} total → {OUTPUT}"
    )
    return len(all_examples)


if __name__ == "__main__":
    build()
