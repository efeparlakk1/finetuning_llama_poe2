import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
LORA_ADAPTER_PATH = "./qlora-final-adapter"
SAVE_PATH = "./llama-3.2-3b-merged"

# 1. Ana Modeli Tam Hassasiyette (16-bit) Yükleme
# ÖNEMLİ: Merge işlemi için model 4-bit/8-bit DEĞİL, float16 veya bfloat16 yüklenmelidir!
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    return_dict=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="cpu",  # VRAM taşmasını önlemek için birleştirmeyi RAM/CPU üzerinde yapmak güvenlidir
)

# 2. LoRA Adaptörünü Yükleme
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

# 3. Adaptörü Ana Modelle Birleştirme (Merge)
print("Adaptör ana modelle birleştiriliyor...")
model = model.merge_and_unload()

# 4. Birleşmiş Modeli ve Tokenizer'ı Diske Kaydetme
model.save_pretrained(SAVE_PATH, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.save_pretrained(SAVE_PATH)

print(f"Model başarıyla {SAVE_PATH} dizinine kaydedildi.")