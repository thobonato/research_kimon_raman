# Embedding Manipulation Tool for Research

## Features
- Load and manipulate LLM embeddings
- Compute token probabilities
- Generate text outputs from modified embeddings
- Support for GPU acceleration

## Requirements
- Python 3.6+
- PyTorch
- Transformers
- 16GB+ RAM recommended
- CUDA-capable GPU (optional)

## Installation
```bash
pip install torch transformers
```

## Quick Start
```python
import utils

# Load model (defaults to Qwen/Qwen2.5-1.5B)
model, tokenizer = utils.load_model()

# Create embeddings
embedding = utils.embed_text(model, tokenizer, "Your text here")

# Analyze token probabilities
utils.topk_probabilities(model, tokenizer, embedding, tok_position=0, k=5)

# Generate text output
text = utils.get_text_output(model, tokenizer, embedding)
```

## Core Functions
- `load_model()`: Loads model with disabled embedding layer
- `embed_text()`: Generates embeddings from input text
- `topk_probabilities()`: Shows top-k probable tokens at specified position
- `get_text_output()`: Converts embeddings back to text

## Model Output Structure
- `outputs.last_hidden_state`: Internal text representations
- `logits`: Raw token probability distributions (shape: batch_size × seq_len × vocab_size)
- `probabilities`: Softmax-normalized token probabilities
