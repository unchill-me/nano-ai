#!/usr/bin/env python3
"""
Dataset Creation Pipeline using Bonito
Generates instruction-tuning datasets from unstructured text for fine-tuning.

Bonito is a model from BatsResearch that generates synthetic instruction tuning
data from raw text/context. It transforms unstructured text into formatted
training examples (instruction, input, output).

Reference: https://github.com/batsresearch/bonito
"""

import argparse
import json
import os
from typing import List, Dict, Optional
from pathlib import Path

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

# Modelo Bonito por defecto
BONITO_MODEL = "BatsResearch/bonito-v1"

# Configuración de generación
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_NUM_SAMPLES = 5  # Número de ejemplos a generar por texto de entrada

# Tipos de tareas soportadas por Bonito
# Referencia: https://huggingface.co/BatsResearch/bonito-v1
SUPPORTED_TASK_TYPES = [
    "conversational",  # Genera conversaciones/diálogos
    "explanatory",  # Explicaciones y definiciones
    "summarization",  # Resúmenes
    "qa",  # Preguntas y respuestas
    "creative",  # Escritura creativa
    "technical",  # Instrucciones técnicas
    "mixed",  # Mezcla de tipos
]

DEFAULT_TASK_TYPE = "mixed"


# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================


def load_raw_texts(file_path: str) -> List[str]:
    """
    Carga textos sin procesar desde un archivo.

    Soporta formatos:
    - .jsonl: Cada línea es un JSON con campo 'text' o similar
    - .json: Array de objetos con campo 'text'
    - .txt: Un texto por línea (líneas vacías separan documentos)
    - .md: Archivos Markdown (separados por doble newline)

    Args:
        file_path: Ruta al archivo de entrada

    Returns:
        Lista de textos (strings)
    """
    texts = []
    file_ext = Path(file_path).suffix.lower()

    with open(file_path, "r", encoding="utf-8") as f:
        if file_ext == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Soporta múltiples formatos de campo
                text = data.get("text") or data.get("content") or data.get("body")
                if text:
                    texts.append(text)

        elif file_ext == ".json":
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    text = item.get("text") or item.get("content") or item.get("body")
                    if text:
                        texts.append(text)

        elif file_ext == ".txt":
            content = f.read()
            # Separa por doble newline (documentos separados por línea vacía)
            docs = content.split("\n\n")
            texts = [doc.strip() for doc in docs if doc.strip()]

        elif file_ext == ".md":
            content = f.read()
            # Separa por doble newline (documentos separados por línea vacía)
            docs = content.split("\n\n")
            texts = [doc.strip() for doc in docs if doc.strip()]

        else:
            raise ValueError(f"Formato de archivo no soportado: {file_ext}")

    return texts


def format_as_alpaca(instruction: str, input_text: str, output: str) -> Dict:
    """
    Formatea un ejemplo en formato Alpaca (compatible con train.py).

    Args:
        instruction: La instrucción/tarea
        input_text: Contexto o entrada adicional
        output: La respuesta esperada

    Returns:
        Diccionario con campos instruction, input, output
    """
    return {"instruction": instruction, "input": input_text, "output": output}


def save_dataset(examples: List[Dict], output_path: str):
    """
    Guarda el dataset en formato JSONL.

    Args:
        examples: Lista de ejemplos (diccionarios)
        output_path: Ruta del archivo de salida
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"✅ Dataset guardado: {output_path}")
    print(f"   Total de ejemplos: {len(examples)}")


# ==============================================================================
# INTEGRACIÓN CON BONITO
# ==============================================================================


def generate_with_bonito(
    raw_texts: List[str],
    model_name: str = BONITO_MODEL,
    task_type: str = DEFAULT_TASK_TYPE,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device: Optional[str] = None,
) -> List[Dict]:
    """
    Genera ejemplos de entrenamiento usando Bonito.

    Args:
        raw_texts: Lista de textos de entrada sin procesar
        model_name: Nombre del modelo Bonito a usar
        task_type: Tipo de tarea (conversational, explanatory, etc.)
        num_samples: Cuántos ejemplos generar por texto de entrada
        batch_size: Tamaño del batch para generación
        max_new_tokens: Máximo de tokens a generar
        device: Dispositivo ('cuda', 'cpu', o None para auto)

    Returns:
        Lista de ejemplos formateados (instruction, input, output)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("❌ Error: Bonito requiere transformers y torch.")
        print("   Instala: uv pip install transformers torch")
        raise

    print(f"🔄 Cargando modelo Bonito: {model_name}")

    # Detecta dispositivo automáticamente si no se especificó
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Dispositivo detectado: {device}")

    # Carga del modelo y tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configura el token de padding si no existe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Usa float16 para GPU, float32 para CPU
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
    )

    # Confirma el dispositivo donde se cargó el modelo
    actual_device = next(model.parameters()).device
    print(f"   Modelo cargado en: {actual_device}")

    print(f"✅ Modelo cargado en: {device}")
    print(f"📝 Generando {num_samples} ejemplos por texto...")
    print(f"🔧 Tipo de tarea: {task_type}")

    all_examples = []

    for idx, text in enumerate(raw_texts):
        print(f"\n📄 Procesando texto {idx + 1}/{len(raw_texts)}")
        print(f"   Longitud: {len(text)} caracteres")

        # Construye el prompt para Bonito
        # El prompt indica el tipo de tarea y el contexto
        prompt = f"""Generate {num_samples} instruction-following examples based on the following context.
Task type: {task_type}

Context:
{text}

Generate examples in this format:
Instruction: [task description]
Input: [optional context]
Output: [expected response]

Examples:"""

        # Tokenizar entrada
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=2048, truncation=True, padding=True
        ).to(device)

        # Generar
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=1,
            )

        # Decodificar
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parsear ejemplos generados
        examples = parse_bonito_output(generated_text, text)
        all_examples.extend(examples)

        print(f"   ✅ Generados: {len(examples)} ejemplos")

    return all_examples


def parse_bonito_output(generated_text: str, context: str) -> List[Dict]:
    """
    Parsea la salida de Bonito en ejemplos estructurados.

    Args:
        generated_text: Texto generado por el modelo
        context: El contexto/texto original de entrada

    Returns:
        Lista de ejemplos formateados
    """
    examples = []

    # Divide por "Instruction:" para obtener ejemplos individuales
    parts = generated_text.split("Instruction:")

    for part in parts[1:]:  # Skip primera parte (prompt)
        try:
            lines = part.strip().split("\n")

            instruction = lines[0].strip() if lines else ""
            input_text = ""
            output = ""

            # Extrae Input y Output
            current_field = None
            current_value = []

            for line in lines[1:]:
                line = line.strip()

                if line.startswith("Input:"):
                    if current_field == "output":
                        output = "\n".join(current_value).strip()
                    current_field = "input"
                    current_value = [line[6:].strip()]

                elif line.startswith("Output:"):
                    if current_field == "input":
                        input_text = "\n".join(current_value).strip()
                    current_field = "output"
                    current_value = [line[7:].strip()]

                elif current_field:
                    current_value.append(line)

            # Captura el último campo
            if current_field == "output":
                output = "\n".join(current_value).strip()
            elif current_field == "input":
                input_text = "\n".join(current_value).strip()

            # Si no hay input, usa el contexto como referencia
            if not input_text and context:
                input_text = f"Refer to the following context:\n{context[:500]}..."

            # Solo agrega si tiene instrucción y output
            if instruction and output:
                examples.append(format_as_alpaca(instruction, input_text, output))

        except Exception as e:
            print(f"   ⚠️  Error parseando ejemplo: {e}")
            continue

    return examples


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================


def create_dataset(
    input_file: str,
    output_file: str = "dataset.jsonl",
    task_type: str = DEFAULT_TASK_TYPE,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    model_name: str = BONITO_MODEL,
    device: Optional[str] = None,
):
    """
    Pipeline completo de creación de datasets.

    Args:
        input_file: Archivo con textos sin procesar
        output_file: Archivo de salida (JSONL)
        task_type: Tipo de tarea para Bonito
        num_samples: Ejemplos a generar por texto
        batch_size: Tamaño del batch
        max_new_tokens: Máximo de tokens
        model_name: Modelo Bonito a usar
        device: Dispositivo de cómputo
    """
    print("=" * 70)
    print("🚀 PIPELINE DE CREACIÓN DE DATASET CON BONITO")
    print("=" * 70)

    # Verificar archivo de entrada
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"No se encontró el archivo: {input_file}")

    # 1. Cargar textos
    print(f"\n📂 Cargando textos desde: {input_file}")
    raw_texts = load_raw_texts(input_file)
    print(f"✅ {len(raw_texts)} textos cargados")

    if not raw_texts:
        raise ValueError("No se encontraron textos en el archivo de entrada")

    # 2. Generar ejemplos con Bonito
    print("\n🤖 Generando ejemplos sintéticos con Bonito...")
    examples = generate_with_bonito(
        raw_texts=raw_texts,
        model_name=model_name,
        task_type=task_type,
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    if not examples:
        raise RuntimeError("No se generaron ejemplos. Revisa la salida del modelo.")

    # 3. Guardar dataset
    print(f"\n💾 Guardando dataset...")
    save_dataset(examples, output_file)

    # 4. Estadísticas
    print("\n" + "=" * 70)
    print("📊 ESTADÍSTICAS DEL DATASET")
    print("=" * 70)
    print(f"   Archivo de entrada:    {input_file}")
    print(f"   Archivo de salida:     {output_file}")
    print(f"   Textos procesados:     {len(raw_texts)}")
    print(f"   Ejemplos generados:    {len(examples)}")
    print(f"   Promedio por texto:    {len(examples) / len(raw_texts):.1f}")
    print(f"   Tipo de tarea:         {task_type}")
    print("=" * 70)
    print("🎉 Pipeline completado exitosamente!")
    print(f"\n💡 Ahora puedes entrenar con: python train.py")
    print("   (Asegúrate de que DATASET_PATH apunte al archivo generado)")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera datasets de instrucciones usando Bonito",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Generar dataset básico (modo mixto)
  python create_dataset.py --input raw_texts.jsonl --output dataset.jsonl

  # Generar preguntas y respuestas
  python create_dataset.py --input documentos.txt --task-type qa --num-samples 3

  # Generar explicaciones detalladas
  python create_dataset.py --input articulos.jsonl --task-type explanatory

  # Usar GPU específica
  python create_dataset.py --input datos.txt --device cuda:0

Formatos de entrada soportados:
  - .jsonl: {"text": "contenido aquí"} (una línea por ejemplo)
  - .json: [{"text": "..."}, {"text": "..."}]
  - .txt: Un documento por párrafo (separados por línea vacía)
  - .md: Archivos Markdown (separados por doble newline)
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Archivo con textos sin procesar (jsonl, json, txt, o md)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="dataset.jsonl",
        help="Archivo de salida (default: dataset.jsonl)",
    )

    parser.add_argument(
        "--task-type",
        "-t",
        type=str,
        default=DEFAULT_TASK_TYPE,
        choices=SUPPORTED_TASK_TYPES,
        help=f"Tipo de tarea para generar (default: {DEFAULT_TASK_TYPE})",
    )

    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Ejemplos a generar por texto (default: {DEFAULT_NUM_SAMPLES})",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Tamaño del batch (default: {DEFAULT_BATCH_SIZE})",
    )

    parser.add_argument(
        "--max-new-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Máximo de tokens nuevos (default: {DEFAULT_MAX_NEW_TOKENS})",
    )

    parser.add_argument(
        "--model",
        "-M",
        type=str,
        default=BONITO_MODEL,
        help=f"Modelo Bonito a usar (default: {BONITO_MODEL})",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Dispositivo de cómputo (default: auto)",
    )

    args = parser.parse_args()

    create_dataset(
        input_file=args.input,
        output_file=args.output,
        task_type=args.task_type,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        model_name=args.model,
        device=args.device,
    )
