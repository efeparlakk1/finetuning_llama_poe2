import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = 'unsloth/Llama-3.2-3B-Instruct'

dataset = load_dataset("json", data_files="/home/eplinux/unsloth-ft/poe2_data.jsonl", split="train")

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True

)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def alpaca_to_text(example):
    user = example["instruction"]
    if example.get("input"):
        user = f"{example['instruction']}\n\n{example['input']}"
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": example["output"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = dataset.map(alpaca_to_text, remove_columns=dataset.column_names)

model = prepare_model_for_kbit_training(model)

peft_config= LoraConfig(

    r=16,
    lora_alpha=16*2,
    target_modules="all-linear",
    lora_dropout=.075,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=False,
    use_dora=False

)

training_args = SFTConfig(

    dataset_text_field="text",
    max_length=2048,
    packing=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    max_steps=250,
    gradient_checkpointing=True,
    lr_scheduler_type='cosine',
    warmup_steps=10,
    output_dir='./qlora-output',
    logging_steps=10,


)

trainer= SFTTrainer(

    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args

)

trainer.train()

trainer.model.save_pretrained("./qlora-final-adapter")
tokenizer.save_pretrained("./qlora-final-adapter")