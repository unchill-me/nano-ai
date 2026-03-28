#!/usr/bin/env python3
"""
Script de Entrenamiento QLoRA con Unsloth
Modelo: unsloth/meta-llama-3.1-8b-bnb-4bit
Dataset: dataset.jsonl (instruction, input, output)
"""

import json
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# ==================== CONFIGURACIÓN ====================
MODEL_NAME = "unsloth/meta-llama-3.1-8b-bnb-4bit"
DATASET_PATH = "dataset.jsonl"
OUTPUT_DIR = "./lora_model"

# LoRA Config
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training Config
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

# GGUF Export
GGUF_FILENAME = "modelo_custom_m2.gguf"
QUANTIZATION_METHOD = "q4_k_m"

# ==================== FORMATO PROMPT ====================
ALPACA_PROMPT = """A continuación se presenta una instrucción que describe una tarea, junto con una entrada que proporciona contexto adicional. Escribe una respuesta que complete adecuadamente la solicitud.

### Instrucción:
{instruction}

### Entrada:
{input}

### Respuesta:
{output}"""

EOS_TOKEN = ""

def formatting_prompts_func(examples):
    """Formatea los ejemplos para el entrenamiento."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = ALPACA_PROMPT.format(
            instruction=instruction,
            input=input_text,
            output=output
        ) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}

# ==================== CARGA MODELO ====================
print("🔄 Cargando modelo en 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detection
    load_in_4bit=True,
)

print(f"🎯 Configurando LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=LORA_TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ==================== CARGA DATASET ====================
print(f"📂 Cargando dataset desde {DATASET_PATH}...")
dataset = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line.strip()))

# Convertir a formato Dataset de HuggingFace
dataset = Dataset.from_list(dataset)

# Formatear prompts
dataset = dataset.map(formatting_prompts_func, batched=True)

print(f"✅ Dataset cargado: {len(dataset)} ejemplos")

# ==================== ENTRENAMIENTO ====================
print("🏋️ Iniciando entrenamiento QLoRA...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        max_steps=-1,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

# Entrenar
trainer_stats = trainer.train()

print("✅ Entrenamiento completado!")
print(f"⏱️  Tiempo de entrenamiento: {trainer_stats.metrics['train_runtime']:.2f} segundos")

# ==================== GUARDAR ADAPTADOR LoRA ====================
print(f"💾 Guardando adaptador LoRA en {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ==================== EXPORTAR A GGUF ====================
print(f"🔄 Exportando a GGUF (quantization={QUANTIZATION_METHOD})...")
print(f"📁 Archivo de salida: {GGUF_FILENAME}")

# Guardar modelo fusionado en GGUF
model.save_pretrained_gguf(
    OUTPUT_DIR,
    tokenizer,
    quantization_method=QUANTIZATION_METHOD,
)

# Renombrar el archivo GGUF generado al nombre deseado
import os
import glob

gguf_files = glob.glob(os.path.join(OUTPUT_DIR, "*.gguf"))
if gguf_files:
    source_file = gguf_files[0]  # Toma el primer archivo GGUF encontrado
    target_file = os.path.join(OUTPUT_DIR, GGUF_FILENAME)
    
    if source_file != target_file:
        os.rename(source_file, target_file)
        print(f"✅ Archivo renombrado: {GGUF_FILENAME}")

print("\n" + "="*60)
print("🎉 PROCESO COMPLETADO EXITOSAMENTE")
print("="*60)
print(f"📍 Adaptador LoRA: {OUTPUT_DIR}/")
print(f"📍 Modelo GGUF:   {OUTPUT_DIR}/{GGUF_FILENAME}")
print(f"\n💡 Para usar en Mac M2:")
print(f"   1. Descarga: scp -r runpod-ip:{OUTPUT_DIR}/{GGUF_FILENAME} ./")
print(f"   2. Ejecuta con llama.cpp o ollama")
