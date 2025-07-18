from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch

# Load the smallest model for fast training
model_name = "Qwen/Qwen2-0.5B"  # Smallest option
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_3",
    torch_dtype=torch.float16,
    device_map="auto",    
)
model = torch.compile(model, backend="aot_eager", fullgraph=True)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load AG News dataset
dataset = load_dataset("ag_news")

# Take only a tiny subset for ultra-fast training
train_dataset = dataset["train"].select(range(10))  # Just 10 samples
test_dataset = dataset["test"].select(range(5))     # Just 5 samples

# Tokenization function
def tokenize_function(examples):
    # Just use the text directly for language modeling
    return tokenizer(
        examples["text"],  # Use the text field directly
        truncation=True,
        padding=False,  # Let data collator handle padding
        max_length=128,
    )

# Tokenize datasets and remove unnecessary columns
train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])
test_tokenized = test_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])

# Minimal training arguments for fast execution
training_args = TrainingArguments(
    output_dir="./quick_test",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    max_steps=5,  # Just 5 steps to trigger backward passes
    logging_steps=1,
    save_strategy="no",
    eval_strategy="no",
    dataloader_drop_last=False,
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # False for causal LM (GPT-style)
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    data_collator=data_collator,
)

# Run training (this will execute backward passes)
print("Starting training...")
trainer.train()
print("Training completed!")

# Manual backward pass example
print("\nManual backward pass example:")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for i, batch in enumerate(trainer.get_train_dataloader()):
    if i >= 3:  # Just 3 manual backward passes
        break
    
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    print(f"Step {i+1}: Loss = {loss.item():.4f}")
    
    loss.backward()  # Your backward pass!
    optimizer.step()
    
print("Manual backward passes completed!")

# Test the model with a simple prompt
print("\n" + "="*50)
print("TESTING THE MODEL")
print("="*50)

model.eval()  # Set to evaluation mode

# Simple test prompt
test_prompt = "The latest technology news shows that"

# Tokenize the prompt
inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print(f"Input prompt: '{test_prompt}'")
print("\nGenerating response...")

# Generate response
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=30,
        do_sample=False,  # Use greedy decoding instead of sampling
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode the response
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated text: '{generated_text}'")

print("\nModel testing completed!")