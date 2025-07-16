import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    cache_implementation="static",   # ✅ enables sliding window cache
    torch_dtype=torch.float16
).to("cuda")

# Optional: compile for performance
model = torch.compile(model)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
inputs = tokenizer("Hello world!", return_tensors="pt").to("cuda")

output = model.generate(inputs.input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
