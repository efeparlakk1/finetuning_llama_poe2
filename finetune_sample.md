# `finetune_sample.py` — Adım adım açıklama

Bu script, **Llama 3.2 3B Instruct** modelini Path of Exile 2 verisiyle **QLoRA** (4-bit quantize + LoRA) yöntemiyle fine-tune eder. Tam modeli değil, küçük bir adaptör eğitir; VRAM tasarrufu için.

**Akış:** veri yükle → modeli 4-bit aç → sohbet formatına çevir → LoRA ekle → eğit → adaptörü kaydet.

---

## 1. Importlar — neyi, neden?

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
```

| Paket | Ne işe yarar |
|---|---|
| `torch` | GPU tensörleri, `bfloat16` tipi |
| `datasets` | `jsonl` dosyasını HuggingFace Dataset olarak yükler |
| `transformers` | Model, tokenizer, 4-bit quantize config |
| `peft` | LoRA: sadece küçük ek katmanları eğitir |
| `trl` | `SFTTrainer`: supervised fine-tuning (metin tamamlama) |

---

## 2. Model ve veri

```python
MODEL_ID = 'unsloth/Llama-3.2-3B-Instruct'

dataset = load_dataset("json", data_files="/home/eplinux/unsloth-ft/poe2_data.jsonl", split="train")
```

- **`MODEL_ID`**: Unsloth’un Llama 3.2 3B Instruct kopyası. Instruct sürümü zaten sohbet formatını bilir; ham base modele göre fine-tune daha kolay oturur.
- **`load_dataset("json", ...)`**: `poe2_data.jsonl` satır satır JSON. Alpaca tarzı alanlar beklenir: `instruction`, `input` (opsiyonel), `output`.
- **`split="train"`**: Tek dosya olduğu için hepsi eğitim seti.

---

## 3. 4-bit quantize (QLoRA’nın “Q” kısmı)

```python
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
```

3B modeli 16-bit açmak ~6 GB VRAM ister. 4-bit ile ağırlıklar ~1.5–2 GB’a iner; consumer GPU’da eğitim mümkün olur.

| Argüman | Anlamı | Neden |
|---|---|---|
| `load_in_4bit=True` | Ağırlıkları 4-bit sakla | VRAM’i düşürür |
| `bnb_4bit_compute_dtype=bfloat16` | Matematik bfloat16 | FP32’den hızlı, FP16’dan daha stabil |
| `bnb_4bit_quant_type='nf4'` | NormalFloat4 | Ağırlık dağılımına uygun; QLoRA’nın standart tipi |
| `bnb_4bit_use_double_quant=True` | Quantize sabitlerini de quantize et | Biraz daha VRAM kazancı |
| `device_map='auto'` | Katmanları GPU’ya otomatik yerleştir | Elle `cuda:0` yazmana gerek yok |

`AutoModelForCausalLM`: “sonraki token’ı tahmin et” modeli. Chat / instruction fine-tune bunun üzerine kurulur.

---

## 4. Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
```

Tokenizer metni token ID’lere çevirir. Llama’nın ayrı bir `pad` token’ı yoktur; batch’te kısa örnekleri doldurmak için `eos` kullanılır. Bunu set etmezsen padding patlar veya model pad’i “gerçek token” sanır.

---

## 5. Alpaca → chat template

```python
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
```

Ham satır kabaca şöyle:

```json
{"instruction": "Bu item ne işe yarar?", "input": "Lioneye's Glare", "output": "..."}
```

Script bunu Llama’nın beklediği sohbet string’ine çevirir (`<|start_header_id|>user<|end_header_id|>` vs.).

| Argüman | Neden |
|---|---|
| `tokenize=False` | Şimdilik string bırak; tokenizasyonu trainer yapsın |
| `add_generation_prompt=False` | Eğitimde cevap zaten var; “assistant başlasın” eki ekleme |
| `remove_columns=...` | Eski `instruction`/`input`/`output` kalsın diye karışmasın; sadece `text` kalsın |

`input` boşsa yalnızca `instruction` user mesajı olur.

---

## 6. 4-bit eğitim hazırlığı

```python
model = prepare_model_for_kbit_training(model)
```

Quantize modelde gradient’lerin doğru akması için bazı katmanları dondurur / cast eder, gradient checkpointing ile uyumlu hale getirir. Bunu atlamak 4-bit eğitimde sık kırılır.

---

## 7. LoRA (eğitilen küçük parçalar)

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=16*2,
    target_modules="all-linear",
    lora_dropout=.075,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=False,
    use_dora=False
)
```

Ana ağırlıklar donuk. Her linear katmana `A` ve `B` matrisleri eklenir; sadece onlar güncellenir. Checkpoint megabayt mertebesinde kalır.

| Argüman | Anlamı | Neden bu değer |
|---|---|---|
| `r=16` | LoRA rank (adaptör boyutu) | 8 = daha küçük/ucuz, 16 = daha fazla kapasite, 64+ genelde gereksiz |
| `lora_alpha=32` | Ölçek: etki ≈ `alpha/r` = 2 | Klasik kural: `alpha = 2 * r` |
| `target_modules="all-linear"` | Tüm linear katmanlara LoRA | Sadece `q_proj`/`v_proj` yerine daha güçlü, biraz daha VRAM |
| `lora_dropout=0.075` | Adaptör dropout | Küçük veri setinde ezberlemeyi biraz kırar |
| `bias="none"` | Bias’ları eğitme | Standart; az parametre, az risk |
| `task_type="CAUSAL_LM"` | Dil modeli (sonraki token) | PEFT doğru forward’ı seçsin |
| `use_rslora=False` | Rank-stabilized LoRA kapalı | Klasik LoRA yeterli |
| `use_dora=False` | DoRA kapalı | Daha yavaş/ağır; burada gerek yok |

---

## 8. Eğitim ayarları

```python
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
```

| Argüman | Anlamı | Neden |
|---|---|---|
| `dataset_text_field="text"` | Hangi kolonu eğit | `alpaca_to_text`’in ürettiği alan |
| `max_length=2048` | Bundan uzun örnekler kesilir | 4096 daha çok VRAM; 2048 PoE item metni için genelde yeter |
| `packing=False` | Kısa örnekleri tek sequence’e yapıştırma | Daha temiz loss; packing hız için, karışık context riski var |
| `per_device_train_batch_size=2` | GPU’da aynı anda 2 örnek | 3B + 4-bit + LoRA için makul |
| `gradient_accumulation_steps=2` | 2 adımda gradient biriktir, sonra update | Efektif batch = 2×2 = **4**. VRAM yetmezken büyük batch illüzyonu |
| `optim="paged_adamw_8bit"` | 8-bit Adam + paged bellek | Optimizer state VRAM’ini düşürür; OOM’da CPU’ya sayfalar |
| `learning_rate=2e-4` | LoRA için tipik LR | Full fine-tune’da 1e-5 civarı; LoRA daha yüksek LR ister |
| `max_steps=250` | 250 optimizer adımı | Epoch değil adım. Küçük jsonl için hızlı deneme; gerçek işte epoch veya daha uzun step |
| `gradient_checkpointing=True` | Aktivasyonları saklamak yerine yeniden hesapla | VRAM ↓, süre biraz ↑ |
| `lr_scheduler_type='cosine'` | LR yavaşça düşer | Sonda daha ince ayar |
| `warmup_steps=10` | İlk 10 adımda LR 0 → 2e-4 | Başta patlamayı önler |
| `output_dir='./qlora-output'` | Checkpoint / log | Ara kayıtlar buraya |
| `logging_steps=10` | Her 10 adımda loss yaz | İlerlemeyi görmek için |

---

## 9. Trainer, eğitim, kayıt

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args
)

trainer.train()

trainer.model.save_pretrained("./qlora-final-adapter")
tokenizer.save_pretrained("./qlora-final-adapter")
```

`SFTTrainer` her örnekte: tokenize et → model sonraki token’ı tahmin etsin → loss’u geriye yay → sadece LoRA ağırlıklarını güncelle.

Kayıt **tam model değil**, LoRA adaptörü + tokenizer. Birkaç yüz MB civarı.

Sonra `sample_to_gguf.py` bu adaptörü 16-bit base modele merge eder; GGUF’a çevirmek için o birleşik klasör kullanılır.

---

## Tek bakışta pipeline

```
poe2_data.jsonl
    → Alpaca satırları
    → Llama chat string ("text")
    → 4-bit Llama 3.2 3B + LoRA
    → 250 adım SFT
    → ./qlora-final-adapter   (sadece adaptör)
    → (sonraki script) merge → GGUF
```

---

## Kullanım

Özel CLI argümanı yok; sabitler dosyanın içinde.

```bash
python finetune_sample.py
```

Çalışmadan önce:

1. `poe2_data.jsonl` yolu doğru olsun.
2. GPU + CUDA + `bitsandbytes` kurulu olsun.
3. HuggingFace’ten model indirilebilsin (`huggingface-cli login` gerekebilir).

Çıktılar:

- `./qlora-output` — eğitim checkpoint’leri
- `./qlora-final-adapter` — son LoRA + tokenizer

---

## Hızlı “neden QLoRA?” özeti

| Yöntem | Ne eğitilir | VRAM | Sonuç |
|---|---|---|---|
| Full fine-tune | Tüm 3B ağırlık | Yüksek | Tam model kopyası |
| LoRA (16-bit) | Sadece adaptör | Orta | Adaptör |
| **QLoRA (bu script)** | Adaptör; base 4-bit | Düşük | Aynı adaptör, daha az VRAM |
