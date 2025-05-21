from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

def load_model(model_name, quant_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config["use_4bit"],
        bnb_4bit_quant_type=quant_config["quant_type"],
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16
    )
    return model, tokenizer

def generate_output(model, tokenizer, question, system_prompt=None):
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    else:
        messages = [{"role": "user", "content": question}]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)
