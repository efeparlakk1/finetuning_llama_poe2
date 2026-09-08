import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL_ID = 'unsloth/Llama-3.2-3B-Instruct'
MAX_SEQ_LENGTH=2048
DTYPE=None
LOAD_IN_4BIT=True

dataset = load_dataset("json", data_files="/home/eplinux/unsloth-ft/poe2_data.jsonl", split="train")

model, tokenizer = FastLanguageModel.from_pretrained(

    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    dtype=DTYPE

)

dataset = dataset.map(lambda x: {
    "messages": [
        {"role": "user", "content": x["instruction"] + "\n" + x["input"]},
        {"role": "assistant", "content": x["output"]}
    ]
})

dataset = dataset.map(lambda x: {
    "text": tokenizer.apply_chat_template(
        x["messages"],
        tokenize=False
    )
})  

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=23,
    lora_dropout=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias='none',
    use_gradient_checkpointing="unsloth",
    random_state=17
)

training_args=SFTConfig(

    output_dir="./unsloth-output",
    per_device_train_batch_size=2,
    max_steps=250,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    warmup_steps=10,
    gradient_accumulation_steps=4,
    dataset_text_field='text',
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    weight_decay=0.01,
    seed=17,
    max_seq_length=MAX_SEQ_LENGTH,

)

trainer=SFTTrainer(

    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,

)

trainer.train()

model.save_pretrained("./unsloth-qlora-adapter")
tokenizer.save_pretrained("./unsloth-qlora-adapter")