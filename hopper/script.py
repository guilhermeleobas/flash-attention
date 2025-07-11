"""
Simple reproduction script for Flash Attention 3 with PHI-1
Requires: transformers, torch, flash-attn (with FA3 support)
"""

import flash_attn_3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.recompile_limit = 32

def test_flash_attention_3():
    """Test Flash Attention 3 with Microsoft Phi-1 model"""
    
    model_name = "microsoft/phi-1"
    
    # Load model with Flash Attention 3
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_3",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.forward = torch.compile(model.forward, backend="aot_eager", fullgraph=False, dynamic=False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test input
    prompt = "The future of artificial intelligence is"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Test basic forward pass
    with torch.no_grad():
        logits = model(**inputs).logits
    
    return True

test_flash_attention_3()