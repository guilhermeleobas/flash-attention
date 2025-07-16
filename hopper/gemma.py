from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model with static cache support
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",  # or "google/gemma-7b-it"
    attn_implementation="flash_attention_3",
    torch_dtype=torch.float16,  # or bfloat16 if preferred
    device_map="auto"  # optional, for automatic GPU placement
)

# Optional: Compile the model with torch.compile (PyTorch 2.1+)
# model.forward = torch.compile(model.forward)
model = torch.compile(model)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

# Encode the prompt
inputs = tokenizer("Once upon a time", return_tensors="pt").to(model.device)

# Generate text
outputs = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
