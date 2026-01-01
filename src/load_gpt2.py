import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Loading GPT-2 small...")

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Model loaded successfully on {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Model config: {model.config}")
