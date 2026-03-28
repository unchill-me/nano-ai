# Guía de Conexión Remota - RunPod + Zed

## Conexión SSH a RunPod

### 1. Obtener Credenciales de RunPod
1. Crea un Pod en RunPod con:
   - GPU: RTX 3090, A100, H100 (recomendado)
   - Template: PyTorch o CUDA
   - Container Disk: 50GB mínimo
   - Volume Disk: 100GB+ para datasets

2. Una vez creado, copia:
   - **SSH Public IP**: `xxx.runpod.io`
   - **SSH Port**: `xxxxx` (ej: 39647)
   - **SSH Command**: `ssh root@xxx.runpod.io -p xxxxx`

### 2. Configurar Clave SSH en RunPod

Desde la terminal local:

```bash
# Copiar tu clave pública a RunPod
ssh-copy-id -p [PORT] root@[HOST]

# O manualmente:
cat ~/.ssh/id_rsa.pub | ssh -p [PORT] root@[HOST] "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### 3. Configurar Zed para Conexión Remota

**Opción A: SSH Remote Development**

1. Abre Zed
2. Presiona `Cmd+Shift+P` → "remote ssh: connect to host"
3. Selecciona "Configure SSH Hosts"
4. Añade a `~/.ssh/config`:

```ssh-config
Host runpod-qlora
    HostName [HOST].runpod.io
    User root
    Port [PORT]
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

5. En Zed: `Cmd+Shift+P` → "remote ssh: connect to host" → Selecciona `runpod-qlora`
6. Selecciona `/workspace` como directorio de trabajo

**Opción B: Zed Remoting (Experimental)**

```bash
# En RunPod, instala el servidor de Zed
curl -f https://zed.dev/install-remote | bash
```

### 4. Generar Dataset con Bonito (Opcional)

Si no tienes un dataset preparado, puedes generar uno sintético desde textos sin procesar usando Bonito:

**Formatos de entrada soportados:**
- `.jsonl`: `{"text": "contenido"}` (una línea por ejemplo)
- `.json`: `[{"text": "..."}, {"text": "..."}]`
- `.txt`: Un documento por párrafo (separados por línea vacía)

**Uso básico:**
```bash
# Generar dataset desde textos
python create_dataset.py --input raw_data.jsonl --output dataset.jsonl

# Generar preguntas y respuestas
python create_dataset.py --input documentos.txt --task-type qa --num-samples 3

# Generar explicaciones detalladas
python create_dataset.py --input articulos.jsonl --task-type explanatory
```

**Tipos de tarea disponibles:**
- `mixed` (default): Mezcla de todos los tipos
- `qa`: Preguntas y respuestas
- `explanatory`: Explicaciones y definiciones
- `conversational`: Diálogos y conversaciones
- `summarization`: Resúmenes
- `creative`: Escritura creativa
- `technical`: Instrucciones técnicas

**Opciones:**
```bash
--input, -i           # Archivo de entrada (requerido)
--output, -o          # Archivo de salida (default: dataset.jsonl)
--task-type, -t       # Tipo de tarea (default: mixed)
--num-samples, -n     # Ejemplos por texto (default: 5)
--batch-size, -b      # Tamaño del batch (default: 8)
--max-new-tokens, -m  # Máximo tokens (default: 1024)
--model, -M           # Modelo Bonito (default: bespokelabs/bonito-v1)
--device, -d          # Dispositivo: cpu, cuda, cuda:0, cuda:1
```

### 5. Ejecutar el Entrenamiento

Una vez conectado en Zed:

```bash
# 1. Abrir terminal integrada: Ctrl+`  o  Cmd+J

# 2. Ejecutar setup (primera vez)
bash setup.sh
source venv/bin/activate

# 3. Subir dataset.jsonl (desde otra terminal local)
scp -P [PORT] dataset.jsonl root@[HOST]:/workspace/

# 4. Ejecutar entrenamiento
python train.py

# 5. Monitorear uso de GPU (en otra terminal)
watch -n 1 nvidia-smi
```

### 6. Descargar Modelo a Mac M2

```bash
# Desde tu terminal LOCAL
scp -P [PORT] root@[HOST]:/workspace/lora_model/modelo_custom_m2.gguf ./

# O usar rsync para carpetas completas
rsync -avz -e "ssh -p [PORT]" root@[HOST]:/workspace/lora_model/ ./modelo_local/
```

### 7. Uso en Mac M2

**Con llama.cpp:**
```bash
# Instalar llama.cpp
brew install llama.cpp

# Ejecutar modelo
./llama-cli -m modelo_custom_m2.gguf -p "Tu prompt aquí"
```

**Con Ollama:**
```bash
# Crear Modelfile
cat > Modelfile << EOF
FROM ./modelo_custom_m2.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM You are a helpful assistant.
EOF

# Crear modelo
ollama create mi-modelo -f Modelfile

# Ejecutar
ollama run mi-modelo
```

## Solución de Problemas

### Error de CUDA
```bash
# Verificar CUDA
nvidia-smi
nvcc --version

# Si CUDA no está disponible:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Error de memoria OOM
```bash
# Reducir batch size o secuencia en train.py:
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 1024
GRADIENT_ACCUMULATION_STEPS = 8
```

### Desconexión de SSH
```bash
# Usar tmux o screen para mantener procesos
tmux new -s training
python train.py

# Para reconectar:
tmux attach -t training
```

## Recursos Adicionales

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [RunPod Docs](https://docs.runpod.io/)
- [Zed SSH Remote](https://zed.dev/docs/remote-development)
