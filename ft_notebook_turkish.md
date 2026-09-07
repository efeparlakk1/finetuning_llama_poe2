# Path of Exile 2 metin SFT — Unsloth ile

Bu notebook, Llama 3.1 8B Instruct modelinin Path of Exile 2 crafting (işleme) soru-cevap verisi üzerinde **denetimli ince ayar (supervised fine-tune, SFT)** çalışmasıdır. Bir ders gibi yazılmıştır: her kod hücresinin önünde, kodun *ne yaptığını* değil, *neden var olduğunu* anlatan bir bölüm vardır.

Önceden ince ayar deneyiminiz olması gerekmez. Markdown'ı okuyun, hücreyi çalıştırın, sonra bir sonraki markdown'da az önce gördüğünüz çıktının yorumunu okuyun.

## Hangi sorunu çözüyoruz?

Önceden eğitilmiş bir LLM zaten internet metninden çok şey "bilir". Ancak *sizin* tarzınızda, *sizin* gerçeklerinizle, *sizin* göreviniz için otomatik olarak cevap vermez. İnce ayar, modele bir PoE 2 crafting senaryosu verildiğinde `poe2_data.jsonl` içindeki rehberler gibi yanıt vermesini öğretir.

Bu **RAG (retrieval) değildir**. RAG, çıkarım anında ekstra belgeleri prompt'a tıkıştırır ve ağırlıkları değiştirmez. İnce ayar **ağırlıkları değiştirir**. Unsloth dokümantasyonu bunu net söyler: ince ayar bilgi enjekte edebilir ve davranışı değiştirebilir; RAG modelin kendisini değiştiremez.

## İnsanların karıştırdığı üç "eğitim" katmanı

1. **Ön eğitim (pretraining)** — devasa bir metin yığını üzerinde bir sonraki token'ı tahmin etmek. Bir *temel (base)* model üretir. Pahalıdır. Bunu yapmıyoruz.
2. **Denetimli ince ayar (SFT)** — modele çok sayıda `(prompt, ideal cevap)` çifti gösterip cevabı üretmesini öğretmek. **Bu notebook budur.**
3. **Tercih / RL yöntemleri** (DPO, GRPO, …) — sonraki aşama. Model cevap üretir ve puanlanır. SFT çalışana kadar bunu atlayın.

## Instruct vs base (bu seçim önemlidir)

| Tür | Tipik isim | En uygun olduğu iş |
| --- | --- | --- |
| **Base** | `Llama-3.1-8B` | Devam ön eğitimi, veya özel bir şablon üzerinde sıfırdan SFT |
| **Instruct** | `Llama-3.1-8B-Instruct` | Sohbet / Soru-cevap. Kullanıcı/asistan turlarını takip etmeye zaten eğitilmiş |

Unsloth, **Instruct** bir modelle başlamanızı önerir. Instruct modeller zaten sohbet-şablonu dilini konuşur, bu yüzden **daha az veri** gerekir. Bu notebook `Meta-Llama-3.1-8B-Instruct` kullanır.

## Tam ince ayar vs LoRA vs QLoRA

Llama 3.1 8B yaklaşık 8 milyar parametreye sahiptir. Hepsini güncellemek (tam ince ayar) çok VRAM ister ve genellikle gereksizdir.

**LoRA** (Low-Rank Adaptation): orijinal ağırlıkları \(W\) dondurun. Her büyük matrisin yanına iki ince matris \(A\) ve \(B\) eğitin; etkin ağırlık şöyle olur:

\[
\hat{W} = W + \frac{\alpha}{r} AB
\]

Tipik olarak parametrelerin yaklaşık %0.5–1'ini eğitirsiniz. LoRA **tüm temel lineer katmanlara** uygulandığında kalite tam ince ayara yaklaşabilir.

**QLoRA**: aynı LoRA adaptörleri, ancak dondurulmuş temel model **4-bit** olarak saklanır. VRAM'i yaklaşık 4 kat düşürür. 16-bit LoRA'dan biraz daha yavaş / kıl payı daha az doğru. Unsloth'un **dinamik 4-bit** checkpoint'leri (`*-unsloth-bnb-4bit`) bu doğruluk farkının çoğunu kapatır.

Bu notebook **QLoRA**'dır: `load_in_4bit=True` + LoRA adaptörleri.

Unsloth'un kuralı: önce LoRA/QLoRA deneyin. Başarısız olursa, tam ince ayar kötü bir veri setini neredeyse hiçbir zaman sihirli şekilde düzeltmez.

## Unsloth sizin için ne yapıyor?

[Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), Hugging Face `transformers` + TRL üzerine kurulmuş bir eğitim yığınıdır. Şunları yapar:

- Llama kernel'larını daha hızlı ileri/geri geçiş için yamalar
- 4-bit modelleri daha az VRAM ile yükler
- Sohbet-şablonu yardımcıları sağlar (`get_chat_template`, `train_on_responses_only`)
- LoRA adaptörlerini veya llama.cpp / Ollama için GGUF dışa aktarabilir

Eğitim *tarifi* hâlâ standart SFT'dir. Unsloth bu tarifi tek bir tüketici GPU'suna sığdırır (bu makine: RTX 4070 Ti SUPER, 16 GB).

## Boru hattı (bu haritayı aklınızda tutun)

```
1. 4-bit Instruct modeli + tokenizer yükle
2. LoRA adaptörlerini tak (eğittiğimiz tek ağırlıklar)
3. Alpaca JSONL → Llama 3.1 sohbet metnine çevir
4. Eğitim / değerlendirme ayır
5. SFTTrainer, kayıp yalnızca asistan token'larında
6. Yaklaşık 2 epoch eğit
7. Test cevabı üret
8. LoRA kaydet (zorunlu). İsteğe bağlı: birleştir + GGUF
```

## Veri seti

`poe2_data.jsonl` — 260 Alpaca tarzı satır:

- `instruction` — görev ("bu omen'i açıkla")
- `input` — crafting senaryosu (taban eşya, hedef, envanter)
- `output` — altın standart crafting rehberi

Bu **küçük** bir alan veri setidir. Küçük veri, bir Instruct modelde *tarz ve biçim* için yeterlidir. Modele tüm PoE 2 wiki'sini öğretmez. Eğitimden sonra cevaplar jenerik görünüyorsa, çözüm genellikle **daha iyi / daha fazla veri**dir, daha büyük rank değil.

Bu notlarda kullanılan resmi kaynaklar:

- [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Chat Templates](https://unsloth.ai/docs/basics/chat-templates)
- [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
- [Saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)

---

## 1. 4-bit Instruct modelini yükle

### Bu hücre neden var?

Eğitim, GPU belleğinde iki nesneye ihtiyaç duyar:

1. **`model`** — sinir ağı (transformer blokları, attention, MLP'ler).
2. **`tokenizer`** — metin ↔ tamsayı dönüştürücüsü. Modeller kelime görmez. **Token ID** görürler. `"Hello"` `[9906]` olabilir. Tokenizer ayrıca **sohbet şablonunu** da sahiplenir (kullanıcı/asistan turlarının özel token'larla nasıl sarıldığı).

`FastLanguageModel.from_pretrained` Unsloth'un yükleyicisidir. Ham `AutoModelForCausalLM.from_pretrained` yerine bunu kullanın ki Unsloth kernel'ları yamalayabilsin ve 4-bit yüklemeyi doğru uygulayabilsin. Unsloth'u ağır `transformers` kullanımından **önce** import edin; ilk import `Will patch your computer...` yazdırır.

### Satır satır

**`from unsloth import FastLanguageModel`**  
Yükleme, LoRA, çıkarım modu ve (sonra) GGUF dışa aktarma için giriş noktası.

**`import torch`**  
PyTorch. CUDA cihaz sorguları için kullanırız. Unsloth tensörler ve autograd için arka planda onu kullanır.

**`max_seq_length = 2048`**  
Bir eğitim örneğindeki (prompt + cevap) maksimum token sayısı. Llama 3.1 128k bağlam yapabilir, ama 2048 Unsloth'un önerilen başlangıç noktasıdır: daha ucuz, daha hızlı, bu crafting rehberleri için yeterli. Bir satır daha uzunsa **kesilir**. Sonra kesilmiş cevaplar görürseniz bunu yükseltin (VRAM artar). Unsloth, aynı GPU'da stok Hugging Face'den daha uzun bağlam eğitebilir.

**`dtype = None`**  
Dondurulmamış / LoRA matematiği için hesaplama dtype'ı. `None` otomatik demektir:

- Ampere ve yenilerde **bfloat16** (bu 4070 Ti SUPER: evet — `Bfloat16 = TRUE` göreceksiniz)
- Eski GPU'larda **float16**

İnce ayar için `float32` seçmeyin; VRAM israf eder. bf16, mevcut olduğunda tercih edilir çünkü fp16 gibi bir kayıp-ölçekleme dansına ihtiyaç duymaz.

**`load_in_4bit = True`**  
Bu **QLoRA'dır**. Dondurulmuş 8B ağırlıklar 4-bit'te yaşar. 16-bit LoRA için bunu `False` yapın (veya `load_in_16bit=True`): daha fazla VRAM, biraz daha fazla doğruluk. Unsloth: 4-bit / 8-bit / 16-bit / tam FT'den yalnızca biri açık olmalı.

**`model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"`**  
Hugging Face repo kimliği. Unsloth adlandırma:

| Sonek | Anlamı |
| --- | --- |
| `unsloth-bnb-4bit` | Unsloth **dinamik 4-bit**. Düz bnb-4bit'ten biraz daha fazla VRAM, 16-bit doğruluğa çok daha yakın |
| `bnb-4bit` (`unsloth` yok) | Standart BitsAndBytes 4-bit |
| sonek yok | Orijinal 16-bit (veya 8-bit) ağırlıklar |

İsimdeki **Instruct**, sohbet-ayarlı demektir. 8B VRAM'de sıkışıksa yorumdaki daha küçük/hızlı seçenek `unsloth/Llama-3.2-3B-Instruct`.

**`FastLanguageModel.from_pretrained(...)`**  
Gerekirse ağırlıkları indirir, 4-bit modeli GPU'da somutlaştırır, `(model, tokenizer)` döndürür. Gördüğünüz banner GPU, CUDA yeteneği, Torch sürümü ve Flash Attention'ın açık olup olmadığını listeler. Burada `FA2 = False` sorun değil; Unsloth yine de Llama'yı yamalar.

### "2x faster free finetuning" ne demek?

Unsloth bazı PyTorch autograd yollarını el yazması geriye doğru kernel'lar ve bellek hileleriyle (gradient offload, checkpointing) değiştirir. Aynı matematik, daha az zaman ve VRAM. API'lerini kullanmanın ötesinde bunu yapılandırmazsınız.

### Bu adımda yaygın hatalar

- Yalnızca metin verisi için bir **Vision** checkpoint yüklemek (bu notebook eskiden öyleydi). Metin SFT bir metin Instruct model ister.
- `max_seq_length`'i unutmak — sonraki trainer ve loader aynı değerde anlaşmalıdır.
- 4-bit eğitip 16-bit sunmak (veya tersi) dikkatsizce. Unsloth: *eğitebildiğinizde aynı kesinlikle eğitin ve sunun.*

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None  # otomatik: Ampere+'da bf16 (bu GPU), değilse fp16
load_in_4bit = True

# Dinamik 4-bit Instruct checkpoint (düz bnb-4bit'ten daha yüksek doğruluk).
# Daha küçük/hızlı seçenek: "unsloth/Llama-3.2-3B-Instruct"
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

Örnek çıktı:

```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
...
==((====))==  Unsloth 2026.9.2: Fast Llama patching. Transformers: 5.5.0.
   \\   /|    NVIDIA GeForce RTX 4070 Ti SUPER. Num GPUs = 1. Max memory: 15.992 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.11.0+cu130. CUDA: 8.9. CUDA Toolkit: 13.0. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
...
Loading weights: 100%|██████████| 291/291 [00:03<00:00, 88.27it/s]
Unsloth: Will load unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit as a legacy tokenizer.
```

### Az önce ne görmüş olmalısınız?

Unsloth banner'ı şunları doğrular:

- **Unsloth 2026.x + Transformers** — yığın sürümleri.
- **NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB** — bir GPU.
- **CUDA 8.9, Bfloat16 = TRUE** — bu GPU bf16 kullanabilir. İyi.
- **Fast Llama patching** — kernel'lar bağlandı.
- Ağırlık yükleme ~291 parça, önbellekteyse birkaç saniye.

Uyarılar (`IProgress`, `HF_HUB_ENABLE_HF_TRANSFER`, kimlik doğrulamasız Hub) gürültülüdür ama ölümcül değildir. Bir `HF_TOKEN` yalnızca hız sınırlarına yardımcı olur.

---

## 1b. LoRA takmadan önce VRAM'i kontrol et

### Bu hücre neden var?

İnce ayar iki sıkıcı şekilde ölür: **OOM** (bellek yetersiz) ve **boş yeriniz olduğunu sanmak**. 4-bit yüklemeden sonra, LoRA, aktivasyonlar ve optimizer'dan **önce** ne kaldığını ölçün.

`torch.cuda` sürücüyle konuşur:

- **`get_device_name(0)`** — GPU 0'ın pazarlama adı.
- **`get_device_properties(0).total_memory`** — toplam VRAM, bayt cinsinden. GiB için `1024**3`'e bölün. Bu kartta ~16.0 bekleyin.
- **`mem_get_info()[0]`** — şu anki **boş** bayt (tuple `(free, total)`).

8B QLoRA yükledikten sonra 16 GB'ın yaklaşık **~9 GB boş** görmek normaldir: 4-bit temel model artı çalışma zamanı birkaç GB yedi. LoRA adaptörleri küçüktür; eğitim sırasında aç olan parçalar:

- aktivasyonlar (batch boyutu ve `max_seq_length` ile büyür)
- AdamW durumları (eğitilebilir parametrelerin 2 katı, burada `adamw_8bit` ile azaltılır)
- 128k kelime dağarcığı için logit'ler

Boş VRAM zaten ~1–2 GB olsaydı, `per_device_train_batch_size`'ı 1'e düşürür, `max_seq_length`'i kısaltır veya eğitimden **önce** 3B Instruct modele geçerdiniz, OOM'dan sonra değil.

```python
print(torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Free: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
```

Örnek çıktı:

```
NVIDIA GeForce RTX 4070 Ti SUPER
VRAM: 16.0 GB
Free: 9.0 GB
```

### VRAM çıktısını nasıl okumalısınız?

4070 Ti SUPER, toplam **16.0 GB** ve 4-bit yüklemeden **sonra** boş bellek (önceki çalışmada ~9 GB) görmelisiniz. Kalan bütçe, eğitimin içinde yaşayacağı yerdir.

---

## 2. LoRA adaptörlerini tak (eğittiğimiz tek ağırlıklar)

### Bu hücre neden var?

8B ağırlıklar 4-bit'te dondurulmuştur. Şimdi eğitseydik, yararlı hiçbir şey güncellenmezdi. `get_peft_model` **LoRA** katmanlarını enjekte eder (PEFT = Parameter-Efficient Fine-Tuning). Bundan sonra `model` bir PEFT modelidir: ileri geçiş = dondurulmuş \(W\) + \(AB\) adaptörleri.

Unsloth'un `patched 32 layers with 32 QKV ...` çıktısı, 32 Llama 3.1 bloğunun hepsinin Q/K/V, çıktı projeksiyonu ve MLP adaptörleri aldığı anlamına gelir. Llama 3.1 8B'nin 32 transformer katmanı vardır.

### Tek resimde LoRA

\(d_{\text{out}} \times d_{\text{in}}\) şeklindeki her hedef lineer katman normalde o kadar eğitilebilir ağırlığa sahip olurdu. LoRA şunu kullanır:

- \(A\): \(r \times d_{\text{in}}\)
- \(B\): \(d_{\text{out}} \times r\)

rank **`r` ≪ d** ile. `r=16` için bu, 4096×4096 bir matrise kıyasla küçüktür. Eğitimden sonra bu notebook **8,072,204,288 parametrenin 41,943,040'ının eğitilebilir (%0.52)** olduğunu bildirdi. LoRA'nın tüm noktası budur.

### Her argüman

**`r=16`** — LoRA rank. Unsloth'un önerdiği değerler: 8, 16, 32, 64, 128. Daha yüksek \(r\) = daha fazla kapasite, daha fazla VRAM, daha kolay aşırı öğrenme. 260 crafting satırı için **16 doğru varsayılandır**. Bu veri setinde 128'e zıplamak JSONL'i ezberlemenin yoludur.

**`target_modules`** — hangi lineer katmanların adaptör alacağı.

| Modül | Nerede | Rol |
| --- | --- | --- |
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention | Token'ların birbirine nasıl baktığı |
| `gate_proj`, `up_proj`, `down_proj` | MLP / FFN | Token başına ileri besleme dönüşümü |

QLoRA makalesi + Unsloth: **yedi tanesinin hepsini** hedefleyin. Yalnızca attention veya yalnızca MLP daha kötüdür. VRAM kurtarmak için modül düşürmek neredeyse hiçbir şey kurtarmaz ve kaliteye mal olur.

**`lora_alpha=16`** — güncellemeyi ölçekler: \(\alpha / r\). Unsloth: \(\alpha = r\) veya \(\alpha = 2r\) yapın. Burada \(\alpha = r = 16\), yani ölçek = 1. İnce ayar çok çekingense `lora_alpha=32` deneyin. Çok özelleşmişse, eğitimden **sonra** alpha'yı aşağı ölçeklemeyi bile önerirler.

**`lora_dropout=0`** — LoRA aktivasyonlarını rastgele sıfırlar. Unsloth `0` yolunu optimize eder (daha hızlı). Kısa SFT koşularında dropout zayıf bir düzenleyicidir. Aşırı öğrenirseniz, rank'i panikle değiştirmeden önce `0.05–0.1` deneyin.

**`bias="none"`** — bias vektörlerini eğitmeyin. Daha hızlı, daha az bellek, onları eğitmekten gerçek bir kalite kazancı yok.

**`use_gradient_checkpointing="unsloth"`** — geriye doğru geçiş için her aktivasyonu saklamayın; yeniden hesaplayın. Klasik checkpointing VRAM kurtarır; Unsloth'un ekstra yolu yaklaşık %30 daha fazla tasarruf ve uzun bağlam dostluğu iddia eder. Hız hata ayıklamıyorsanız `"unsloth"` kullanın.

**`random_state=3407`** — LoRA başlatma tohumu (ve sonraki veri seti ayrımı aynı sayıyı kullanır). Aynı tohum ⇒ karşılaştırılabilir koşular. 3407 Unsloth notebook geleneğidir, sihir değil.

**`use_rslora=False`** — rank-stabilize LoRA, \(\alpha / r\) yerine \(\alpha / \sqrt{r}\) ile ölçekler. Bazen **yüksek** rank'te yardımcı olur. r=16'da kapalı bırakın.

**`loftq_config=None`** — LoftQ, \(A,B\)'yi \(W\)'nin SVD'sinden başlatır. Doğruluğa yardımcı olabilir; başlangıçta VRAM'i sıçratır. Yeni başlayanlar: `None` bırakın.

### Yapmadığımız şeyler

- Embedding'leri veya lm_head'i eğitmiyoruz (özel token eklemezseniz — bu `get_peft_model`'den **önce** olmalıdır).
- Tam ince ayar değil (`full_finetuning=True`).
- 4-bit dondurulmuş ağırlıkları değiştirmiyoruz; yalnızca adaptörler.

Bu hücreden sonra model eğitime hazırdır. Sonra Llama 3.1 Instruct'ın beklediği **tam sohbet biçiminde** metin beslemeliyiz.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

Örnek çıktı:

```
Unsloth 2026.9.2 patched 32 layers with 32 QKV layers, 32 O layers and 32 MLP layers.
```

### Az önce ne görmüş olmalısınız?

`Unsloth ... patched 32 layers with 32 QKV layers, 32 O layers and 32 MLP layers.` Tüm 32 bloktaki tüm attention + MLP projeksiyonlarının LoRA'sı var. Bu sayı daha küçük olsaydı, `target_modules` yanlış olurdu.

---

## 3. Veri seti: Alpaca JSONL → Llama 3.1 sohbet `text`

### Bu hücre neden var?

Trainer "instruction / input / output" sütunlarını anlamaz. Çıkarımda modelin gördüğüne benzeyen **satır başına bir string** ister: özel token'lar, kullanıcı turu, asistan turu. Yanlış şablon ⇒ model üretimde asla görmeyeceği bir biçim öğrenir (veya sonra Ollama'da saçmalık alırsınız).

Unsloth'un veri setleri rehberi: `get_chat_template` uygulayın, konuşmaları `tokenizer.apply_chat_template` ile eşleyin, sonucu bir `"text"` sütununda saklayın. Bu notebook o sütunu **önceden hesaplar**. Trainer'a canlı bir `formatting_func` **geçirmez** (hata ayıklaması daha kolay: `text`'i yazdırabilirsiniz).

### Alpaca JSONL (diskte ne var)

`poe2_data.jsonl` dosyasının her satırı JSON'dur:

```json
{
  "instruction": "Provide a step-by-step PoE 2 crafting guide ...",
  "input": "Crafting Item: Omen of Connections\\n...",
  "output": "### Path of Exile 2 Crafting Overview: ..."
}
```

- **instruction** — kullanıcı niyeti / görev ifadesi.
- **input** — ekstra bağlam (eşya, hedef). Bazı veri setlerinde boş olabilir; burada senaryodur.
- **output** — modelin üretmesini istediğimiz cevap.

Bu **tek turlu** SFT'dir: bir kullanıcı mesajı, bir asistan mesajı. Çok turlu sohbet birkaç rol turunun listesi olurdu; Unsloth Alpaca satırlarını `conversation_extension` ile sahte çok turlu hale getirebilir, bunu **kullanmıyoruz**.

### Hugging Face `Dataset`

`Dataset.from_list(rows)` bellek içi bir Arrow tablosu kurar. `.map(fn)` her satıra (veya batch'e) bir fonksiyon uygular. Bu standart HF datasets iş akışıdır; Unsloth onu değiştirmez.

### `get_chat_template(tokenizer, chat_template="llama-3.1")`

Unsloth tokenizer sohbet şablonunu **değiştirir / düzeltir**. Yukarı akış şablonları bazen yanlıştır; Unsloth düzeltilmiş olanları tutar (`llama-3.1`, `chatml`, `gemma-3`, …). Şablonu her zaman model ailesine eşleştirin. Llama modelinde Gemma token'ları = bozuk eğitim.

Llama 3.1 turları şöyle görünür:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

...isteğe bağlı sistem...
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

KULLANICI METNİ<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

ASİSTAN METNİ<|eot_id|>
```

`<|eot_id|>` tur sonudur (model durmalıdır). Onsuz eğitirseniz, üretimler gevezelik eder.

### `alpaca_to_conversation`

Unsloth'un beklediği ChatML tarzı listeyi kurar: `role` / `content` (ShareGPT `from` / `value` değil). ShareGPT veriniz olsaydı önce `standardize_sharegpt` çağırırdınız.

Kullanıcı içeriği = `instruction`, veya input boş değilse `instruction + "\n\n" + input`. Bu olağan Alpaca birleştirmesidir: model hem görevi hem senaryoyu **tek** bir kullanıcı turunda görmelidir.

Asistan içeriği = `output` değişmeden.

### `formatting_prompts_func` (batch'li)

Her konuşma için:

```python
tokenizer.apply_chat_template(
    convo,
    tokenize=False,              # string döndür, id değil (trainer sonra tokenize eder)
    add_generation_prompt=False, # altın asistan cevabını dahil et; boş asistan başlığı EKLEME
)
```

**`add_generation_prompt`** eğitim vs çıkarım anahtarıdır:

| Aşama | Değer | Etki |
| --- | --- | --- |
| Eğitim | `False` | String altın asistan yanıtı + `<|eot_id|>` içerir |
| Çıkarım | `True` | String asistan başlığında biter, model yanıtı **doldurur** |

`True` ile eğitirseniz, kayıp hedeflerinden **cevapları düşürürsünüz**. `False` ile çıkarım yaparsanız, şablon asistan turunu açmayabilir.

Fonksiyon `{"text": texts}` döndürür — `text` adlı yeni sütun. Bu isim sonra `dataset_text_field="text"` ile eşleşmelidir.

### Eğitim / değerlendirme ayrımı

```python
dataset.train_test_split(test_size=0.2, seed=3407)
```

260 satır → **208 eğitim / 52 değerlendirme** (%80/%20). LoRA ile aynı tohum, ayrım tekrarlanabilir olsun diye.

Unsloth: tüm seti eğitime yakmışsanız, kaliteyi yalnızca **elle** yargılayabilirsiniz. %20 tutmak, `eval_strategy="epoch"`'un tutulmuş kayıp hesaplamasına izin verir. 52 satır küçüktür; eval kaybını bilimsel bir kıyas değil, bir eğilim olarak ele alın.

### 260 satır neden hâlâ işe yarayabilir?

Instruct modeller zaten İngilizce ve sohbet yapısını bilir. Siz **alan biçimini** (PoE 2 omen/currency rehberleri) ve biraz yerel jargonu yönlendiriyorsunuz. Bu, bir temel modele konuşmayı öğretmekten farklı bir iştir. `output` metninin kalitesi, zekice LoRA ayarlarından daha önemlidir.

Hücreyi çalıştırın, sonra çıktıdaki `conversations`'a (Python dict'leri) bakın. Sonraki hücre **işlenmiş** özel-token string'ini gösterir — kaybın gerçekten gördüğü budur.

```python
import json
from datasets import Dataset
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

rows = []
with open("poe2_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

raw = Dataset.from_list(rows)


def alpaca_to_conversation(example):
    user = example["instruction"]
    if example.get("input"):
        user = f"{example['instruction']}\n\n{example['input']}"
    return {
        "conversations": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        )
        for convo in examples["conversations"]
    ]
    return {"text": texts}


dataset = raw.map(alpaca_to_conversation)
dataset = dataset.map(formatting_prompts_func, batched=True)
split = dataset.train_test_split(test_size=0.2, seed=3407)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")
print(train_dataset[0]["conversations"])
```

Örnek çıktı:

```
Map: 100%|██████████| 260/260 [00:00<00:00, 22849.58 examples/s]
Map: 100%|██████████| 260/260 [00:00<00:00, 15518.54 examples/s]
Train: 208  Eval: 52
[{'role': 'user', 'content': 'Provide a step-by-step PoE 2 crafting guide using the provided base item and outcome goal.\n\nCrafting Item: Omen of Connections\n...'}, {'role': 'assistant', 'content': '### Path of Exile 2 Crafting Overview: Omen of Connections\n...'}]
```

### Az önce ne görmüş olmalısınız?

260 örnek üzerinde iki kez `Map`, sonra:

`Train: 208  Eval: 52`

ve `train_dataset[0]["conversations"]` iki dict'lik bir liste (`user`, `assistant`). `input` birleştirildiyse, kullanıcı `content` hem talimatı hem crafting senaryosunu içerir.

Eğitim/değerlendirme 260/0 olsaydı, ayrım başarısız olmuştu. `conversations` hâlâ `instruction` anahtarları taşıyorsa, `alpaca_to_conversation` çalışmamıştı.

---

## 3b. İşlenmiş `text`'i incele (gerçek eğitim string'i budur)

### Bu hücre neden var?

`conversations` insanlar içindir. Model `text` üzerinde eğitilir. İlk ~1200 karakteri yazdırmak şunları yakalar:

- **Çift BOS** — `<|begin_of_text|>` iki kez (Unsloth sonradan birini siler; tokenize sırasında bir log görürsünüz).
- **Eksik asistan / eksik `<|eot_id|>`** — model durmayı hiç öğrenmez.
- **Yanlış şablon** — Llama Instruct modelinde Vicuna `### Human:`.
- **Sistem önsözü** — Llama 3.1 Instruct şablonları, sistem mesajı geçmeseniz bile çoğu zaman bir bilgi-kesim tarihi / tarih sistemi turu enjekte eder. Bu beklenir.

Notebook okunabilir kalsın diye yalnızca `[:1200]` dilimliyorsunuz. Yapıyı kaydırın:

1. `<|begin_of_text|>` — dizinin başlangıcı.
2. `system` başlığı — şablon varsayılanı.
3. `user` başlığı + birleştirilmiş instruction/input'unuz.
4. `<|eot_id|>` — kullanıcı turu biter.
5. `assistant` başlığı + altın rehber.
6. `<|eot_id|>` — asistan turu biter (dur token'ı).

Bu string doğru görünüyorsa, trainer `"text"` alanına bağlanabilir. Yanlış görünüyorsa **durun** ve şablonu düzeltin; eğitmeyin.

```python
print(train_dataset[0]["text"][:1200])
```

Örnek çıktı:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 July 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Provide a step-by-step PoE 2 crafting guide using the provided base item and outcome goal.

Crafting Item: Omen of Connections
...
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Path of Exile 2 Crafting Overview: Omen of Connections
...
<|eot_id|>
```

### Az önce ne görmüş olmalısınız?

`<|start_header_id|>user/assistant<|end_header_id|>` ve crafting cevabını içeren Llama 3.1 biçimli bir transkript. "Cutting Knowledge Date" içeren sistem bloğu Instruct şablonudur, JSONL'iniz değil.

`dataset_text_field="text"` o `text` sütununu tokenize edecektir.

---

## 4. SFT trainer'ı kur (henüz eğitim yok)

### Bu hücre neden var?

**SFTTrainer** (Hugging Face **TRL**'den) standart döngüdür: string batch'leri → tokenize → ileri geçiş → kayıp → geriye geçiş → optimizer adımı. Unsloth bu trainer'ı hız için yamalar; siz yine **`SFTConfig`** ile yapılandırırsınız.

Bu hücre trainer'ı **kurar** ve sonra `train_on_responses_only` ile sarar. GPU eğitimi **çalıştırmaz**. O sonraki hücredir.

### TRL API isimleri (eski bloglardan kafanız karışmasın)

Eski notebook'lar `tokenizer=` ve `max_seq_length=`'i `SFTTrainer`'a geçirir. Güncel TRL şunu kullanır:

- `processing_class=tokenizer`
- `SFTConfig` içinde `max_length=...`
- `SFTConfig` içinde `dataset_text_field="text"`

Aynı fikir, yeni argüman isimleri.

### `DataCollatorForSeq2Seq`

Bir collator, bir batch'teki örnekleri aynı uzunluğa pad'ler. `DataCollatorForSeq2Seq` ayrıca **etiketleri** de pad'ler. Buna ihtiyacımız var çünkü `train_on_responses_only` yok sayılan konumları **`-100`** yapar. Cross-entropy `-100` indeksini yok sayar. Naif bir dil modeli collator'ı bu etiketleri bozabilir.

### Etkin batch boyutu (sayıları değiştirmeden önce bunu okuyun)

\[
\text{etkin batch} = \text{per\_device\_train\_batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{num\_GPUs}
\]

Burada: \(2 \times 4 \times 1 = 8\).

Bir **optimizer adımı** 8 örneği ortalar. Unsloth tarihsel bir hatayı düzeltti: `batch=1, accum=8` ve `batch=8, accum=1` artık eşleşir. OOM olursanız **daha küçük mikro-batch + daha fazla biriktirme** tercih edin.

Unsloth'un genel önerisi etkin batch **16**'dır (`2 × 8`). Bu notebook **8** kullanır — biraz daha gürültülü güncellemeler, 208 satır için sorun değil. Epoch başına adım = \(\lceil 208 / 8 \rceil = 26\). İki epoch ⇒ **52 adım** (eğitim loguyla eşleşir).

### `SFTConfig` argümanları

**`per_device_train_batch_size=2`** — mikro-adım başına GPU başına örnek. Birincil VRAM düğmesi. Hâlâ yedek belleğiniz varsa yükseltin; pad'leme eskiden büyük batch'leri *yavaşlatırdı*, Unsloth bu yüzden packing / padding-free yollar ekledi.

**`gradient_accumulation_steps=4`** — bir Adam adımından önce 4 mikro-batch bekle. 8 tam aktivasyonu bir anda saklamadan daha büyük bir batch simüle eder.

**`warmup_steps=10`** — 10 adım boyunca LR'yi 0'dan `learning_rate`'e doğrusal yükselt (~52'nin %20'si, olağan %5–10 sezgiselinden biraz daha zengin). İlk adımların kocaman gürültülü güncellemeler almasını engeller.

**`num_train_epochs=2`** — eğitim setini iki kez geç. Unsloth: talimat verisi için **1–3 epoch**. 208 satırda 3'ten fazla, aşırı öğrenmenin yoludur. **Aynı anda `max_steps` ayarlamayın.** `max_steps` kazanır ve hangi programı çalıştırdığınızı bilemezsiniz. Yorumdaki `max_steps=60` bir duman testidir: yorumunu kaldırın **ve** `num_train_epochs`'u yorumlayın.

**`learning_rate=2e-4`** — LoRA/QLoRA SFT için Unsloth varsayılanı (olağan bant `2e-4` ile `5e-6`). RL yöntemleri ~`5e-6` ister. Tam FT daha da düşük ister. Kayıp patlarsa LR'yi düşürün. Kayıp zar zor kıpırdarsa biraz yükseltin veya daha uzun eğitin.

**`logging_steps=1`** — her adımı logla. 52 adım için sorun değil; 10k-adımlık koşularda gürültülü.

**`eval_strategy="epoch"`** — her epoch'tan sonra 52 satırlık eval setini çalıştır. Dev eval'lerde yavaştır; burada ucuzdur. Alternatif: `eval_steps=N`.

**`output_dir="outputs_ft_cursor"`** — checkpoint'ler (`checkpoint-52`, tokenizer, adapter). Nihai `ft_lora` dışa aktarmasıyla aynı klasör değil.

**`optim="adamw_8bit"`** — 8-bit optimizer durumlarıyla AdamW (bitsandbytes). fp32 Adam'a kıyasla büyük VRAM kazancı. Unsloth notebook'ları bunu varsayılan kullanır.

**`weight_decay=0.01`** — ağırlıklara L2 tarzı ceza. Unsloth'un önerilen başlangıç değeri. Genelleşmeye yardımcı olur; 0.5'e çıkarmayın.

**`lr_scheduler_type="linear"`** — ısınmadan sonra LR doğrusal olarak 0'a düşer. `cosine` diğer yaygın seçenektir. Bu ölçekte ikisi de sorun değil.

**`seed=3407`** — dataloader karıştırma / başlatma tohumu. LoRA ve veri seti ayrımıyla eşleşir.

**`report_to="none"`** — W&B/TensorBoard'a akış yapma. Grafik isterseniz `"wandb"` yapın.

**`dataset_text_field="text"`** — hangi sütunun tokenize edileceği. Var olmalıdır (biz oluşturduk).

**`max_length=max_seq_length`** — 2048. Yüklenen model bağlamıyla eşleşmelidir.

**`packing=False`** — **packing** kısa örnekleri tek bir 2048-token akışında birleştirir. Daha hızlıdır, ama **satır sayısı küçülür** ve kayıp sayıları karşılaştırılabilir olmaz. Unsloth: `packing=False` ile yine de padding-free batching kullanırlar, yani eski yolda bedava hız bırakmıyorsunuz. Bu veri seti 260 kısa rehber — `False` tutun ki 208 örnek 208 örnek kalsın.

### `train_on_responses_only(trainer)`

Varsayılan nedensel LM kaybı **her** token üzerinde eğitir, kullanıcı sorusu dahil. Sonra model kapasitesini prompt'u kopyalamaya harcar.

QLoRA makalesi **prompt'u maskelemenin** ve yalnızca asistan token'larında eğitmenin yaklaşık %1 kazandırdığını ve sohbet için özellikle önemli olduğunu buldu. Unsloth bunu, kullanıcı (ve sistem) aralıklarındaki etiket id'lerini `-100` yaparak uygular.

Dokümanlar Llama 3.x için açık işaretleyiciler gösterir:

```python
instruction_part = "<|start_header_id|>user<|end_header_id|>\\n\\n"
response_part    = "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
```

Bu notebook `train_on_responses_only(trainer)`'ı **bu string'ler olmadan** çağırır. Güncel Unsloth onları şablondan **otomatik algılar**. Log şöyle görünmelidir:

`Auto-detected instruction_part = '<|start_header_id|>user...' and response_part = '<|start_header_id|>assistant...'`

Eğer **"All labels are -100"** / kayıp her zaman 0 görürseniz, algılama başarısız olmuş demektir — iki string'i açıkça geçirin (Unsloth sorun giderme'ye bakın).

Ayrıca **çift BOS kaldırıldı** ve 208 + 52 satırın tokenize edildiğini göreceksiniz. Bu hâlâ kurulumdur, `trainer.train()` değil.

```python
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=2,
        # max_steps=60,  # yalnızca duman testi; bunu açarsanız num_train_epochs'u kapatın
        learning_rate=2e-4,
        logging_steps=1,
        eval_strategy="epoch",
        output_dir="outputs_ft_cursor",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        dataset_text_field="text",
        max_length=max_seq_length,
        packing=False,
    ),
)

trainer = train_on_responses_only(trainer)
```

Örnek çıktı:

```
Unsloth: reducing dataset_num_proc 6 -> 4 to fit free memory (~1GB per worker). Set UNSLOTH_DATASET_NUM_PROC to override.
Unsloth: We found double BOS tokens - we shall remove one automatically.
Unsloth: Tokenizing ["text"] (num_proc=4): 100%|██████████| 208/208 [00:01<00:00, 143.43 examples/s]
Unsloth: We found double BOS tokens - we shall remove one automatically.
Unsloth: Tokenizing ["text"] (num_proc=4): 100%|██████████| 52/52 [00:01<00:00, 39.60 examples/s]
Unsloth: Auto-detected instruction_part = '<|start_header_id|>user<|end_header_id|>\n\n' and response_part = '<|start_header_id|>assistant<|end_header_id|>\n\n'
Map: 100%|██████████| 208/208 [00:00<00:00, 15588.31 examples/s]
Map: 100%|██████████| 52/52 [00:00<00:00, 13247.32 examples/s]
```

### Az önce ne görmüş olmalısınız?

- Eğitim (208) ve eval (52) için `text` tokenize ediliyor.
- Olası `reducing dataset_num_proc` (tokenize alt süreçleri vs RAM).
- Çift-BOS temizliği.
- Otomatik algılanan Llama 3 kullanıcı/asistan ayırıcıları.
- Maskelenmiş `labels` ekleyen ekstra `Map` geçişleri.

Trainer nesnesi hazır. Henüz hiçbir ağırlık güncellenmedi.

---

## 5. Eğitimi çalıştır (`trainer.train()`)

### Bu hücre neden var?

Bu gerçek GPU döngüsüdür. Her **adım**:

1. Tokenize edilmiş `text`'in sonraki mikro-batch'ini al.
2. İleri geçiş (4-bit temel + LoRA).
3. Kayıp = `labels ≠ -100` olan token'larda cross-entropy (yalnızca asistan).
4. LoRA üzerinden geriye geçiş (ve checkpoint'lenmiş aktivasyonlar).
5. Her `gradient_accumulation_steps` mikro-batch'te: **AdamW 8-bit**'i kırp/uygula, LR zamanlayıcısını adımla.
6. Periyodik olarak eval, log, checkpoint.

`trainer_stats = trainer.train()` bir `TrainOutput` döndürür (küresel adım, ortalama kayıp, zamanlama). `trainer_stats`'ı göstermek o nesneyi yazdırır.

### Unsloth eğitim banner'ını nasıl okumalısınız?

Bu notebook'taki önceki başarılı bir koşudan:

| Alan | Örnek | Anlamı |
| --- | --- | --- |
| Num examples | 208 | Eğitim satırları |
| Num Epochs | 2 | Tam geçişler |
| Total steps | 52 | \(26 \times 2\) |
| Batch size per device | 2 | Mikro-batch |
| Gradient accumulation | 4 | |
| Total batch size | 8 | Etkin batch |
| Trainable parameters | 41.9M / 8.07B (%0.52) | Yalnızca LoRA |

`smartly offload gradients` / `Double buffering` — Unsloth VRAM hileleri. Zararsız.

### "İyi" kayıp nedir?

Unsloth: birçok SFT koşusu yaklaşık **0.5–1.0** civarında biter. Bu koşunun `train_loss ≈ 0.80`'i o banttadır.

| Kayıp davranışı | Olası anlam |
| --- | --- |
| ~0.5–1.0'a doğru pürüzsüz düşüş | Sağlıklı |
| Düz / düşmüyor | LR çok düşük, etiketlerde hata, veya veri çok kolay/rastgele |
| **0**'a doğru düşüş | Aşırı öğrenme (208 satırı ezberleme) |
| Patlama / NaN | LR çok yüksek, kötü sayısal kurulum |

Eval kaybı (her epoch loglanır) eğitim kaybı düşmeye devam ederken **fırlamamalıdır** — bu aşırı öğrenme imzasıdır.

### Aşırı öğrenme vs yetersiz öğrenme (Unsloth'un pratik tavsiyesi)

**Aşırı öğrenme** (çok özelleşmiş): epoch'ları kes, `weight_decay`'i yükselt, belki `lora_dropout=0.1`, daha çeşitli veri, veya çıkarımda `lora_alpha`'yı aşağı ölçekle.

**Yetersiz öğrenme** (hâlâ jenerik): biraz daha yüksek LR veya rank, daha fazla epoch (dikkatli), daha fazla alan içi veri, güncellemelerin daha güçlü olması için daha küçük batch.

260 satırla **varsayılan risk aşırı öğrenmedir**, yetersiz öğrenme değil. İki epoch zaten doyurucu bir öğündür.

### Süre

Bu GPU'da 52 adım için ~106 saniye doğru mertebedir. 10 kat yavaşsa CPU'dasınız veya CUDA kullanılmıyordur.

Hücre bittiğinde, bellekte ince ayarlı bir adaptörünüz vardır. Sonra üretiriz, sonra kaydederiz (RAM bir checkpoint değildir).

```python
trainer_stats = trainer.train()
trainer_stats
```

Örnek çıktı:

```
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 208 | Num Epochs = 2 | Total steps = 52
O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (2 x 4 x 1) = 8
 "-____-"     Trainable parameters = 41,943,040 of 8,072,204,288 (0.52% trained)
...
TrainOutput(global_step=52, training_loss=0.7994460738801326, metrics={'train_runtime': 105.9634, 'train_samples_per_second': 3.926, 'train_steps_per_second': 0.491, 'total_flos': 4302167407902720.0, 'train_loss': 0.7994460738801326, 'epoch': 2.0})
```

### Az önce ne görmüş olmalısınız?

`TrainOutput(global_step=52, training_loss≈0.80, epoch=2.0)` artı çalışma zamanı metrikleri (`train_samples_per_second`, `total_flos`, …). Trainer `outputs_ft_cursor/checkpoint-52` altına bir checkpoint yazar.

`global_step` 60 ise, hâlâ `max_steps=60` açıktı. Kayıp 0.0 ise, yanıt maskelemesi yanlıştır.

---

## 6. Çıkarım: ince ayarlı modelle konuş

### Bu hücre neden var?

Kayıp bir PoE 2 sınavı değildir. Bir **cevabı okumalısınız**. Bu hücre:

1. Modeli eğitim modundan Unsloth'un hızlı çıkarım yoluna geçirir.
2. **Yalnızca kullanıcı** sohbeti kurar (altın asistan yok).
3. CUDA'da token üretir ve metni notebook'a akıtır.

### `FastLanguageModel.for_inference(model)`

Unsloth: `generate`'ten önce her zaman bunu çağırın. **2× çıkarım** yolunu açar (QLoRA, LoRA ve dense). Eğitime özgü kernel'lar (checkpointing, dropout) decode anında istediğiniz şey değildir.

### `messages` kurmak

```python
[{"role": "user", "content": "What is the best way to craft quarterstaff with elemental damage focus?"}]
```

Bu prompt kasıtlı olarak bir eğitim satırından **kopyalanmamıştır**. Genelleştirmenin koku testini istersiniz. Tek bir prompt bir eval seti değildir — bir sağduyu kontrolüdür.

### `apply_chat_template(..., tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)`

Eğitimin tersi:

- **`tokenize=True`** — Python string değil, id tensörleri döndür.
- **`add_generation_prompt=True`** — asistan başlığını ekle ki sonraki token'lar modelin yanıtı olsun.
- **`return_tensors="pt"`** — PyTorch tensörleri.
- **`return_dict=True`** — `input_ids` (ve attention mask) içeren dict, `**inputs` olarak açılabilir.

`.to("cuda")` her tensörü GPU'ya taşır. Bunu unutmak CPU/GPU cihaz hatalarına yol açar.

### `TextStreamer(tokenizer, skip_prompt=True)`

Token'ları üretildikçe yazdırır. `skip_prompt=True` yankılanan kullanıcı şablonunu gizler, yalnızca cevabı görürsünüz.

### `model.generate` argümanları

**`max_new_tokens=512`** — yanıt uzunluğu tavanı. Unsloth: cevaplar kesilirse bunu yükseltin (256, 1024, …). Daha uzun beklersiniz.

**`use_cache=True`** — KV önbelleği. Standart; bellek hata ayıklamıyorsanız kapatmayın.

**`temperature=1.5`** — örnekleme yumuşaklığı. `1.0` modelin ham dağılımıdır. **Daha yüksek = daha rastgele.** Unsloth sohbet demoları canlı sohbet için sıklıkla 1.5 kullanır. Bir **crafting rehberi** için tariflerin daha az kaprisli olması adına `0.3–0.7` tercih edebilirsiniz. Bu notebook Unsloth tarzı demo olarak 1.5 bırakır.

**`min_p=0.1`** — min-p örnekleme: olasılığı en üst token'ın %10'unun altında olan token'ları düşür. top-p'ye modern bir alternatif. Yüksek sıcaklıkta min-p, kuyruğun çöp üretmesini durdurur.

`max_new_tokens and max_length seem to have been set` uyarısı Transformers gürültüsüdür: tokenizer/model yapılandırması hâlâ kocaman bir `max_length` ilan ediyor olabilir (örn. 131072). **`max_new_tokens` kazanır.** Yok sayın.

### Yazdırılan cevabı nasıl yargılamalısınız?

Şunu sorun:

- Veri setindeki **rehber yapısını** kullanıyor mu (çekirdek mekanik, bağlam, adımlar)?
- PoE 2 gibi mi duruyor, yoksa jenerik RPG dolgusu mu?
- `<|eot_id|>`'de duruyor mu yoksa gevezelik mi ediyor?

Güzel bir eğitim kaybı ve işe yaramaz bir örnek, veri setinin o soruyu kapsamadığı (veya şablon cümlelerine aşırı öğrendiğiniz) anlamına gelir. Veriyi düzeltin, `r`'yi değil.

Bundan sonra adaptör, siz kaydedene kadar yalnızca bu Python sürecinde yaşar.

```python
from transformers import TextStreamer

FastLanguageModel.for_inference(model)

messages = [
    {
        "role": "user",
        "content": "What is the best way to craft quarterstaff with elemental damage focus?",
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
inputs = {key: value.to("cuda") for key, value in inputs.items()}

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=512,
    use_cache=True,
    temperature=1.5,
    min_p=0.1,
)
```

Örnek çıktı:

```
Both `max_new_tokens` (=512) and `max_length`(=131072) seem to have been set. `max_new_tokens` will take precedence.
Here is the breakdown for crafting Quarterstaff with elemental damage focus:

**Base Item Analysis:** Quarterstaff
**Type:** Bludgeoning weapon used for melee combat at range.
**Crafting Context:** Elemental damage dealers seeking a versatile, easy-to-manage reach weapon.
...
<|eot_id|>
```

### Az önce ne görmüş olmalısınız?

`<|eot_id|>`'de biten, akıtılmış crafting tarzı bir cevap. Kalite prompt'a göre değişir. "Biçimi ezberledi mi?" kontrolü için eğitim benzeri bir talimatla generate'i yeniden çalıştırın; tutulmuş bir soruyla da karşılaştırın.

---

## 7. LoRA adaptörlerini kaydet (bunu yapın; gerçek artefakt budur)

### Bu hücre neden var?

`trainer.train()` `outputs_ft_cursor/` altına checkpoint yazar, ama temiz teslimat artefaktı sonra yeniden yükleyebileceğiniz **küçük bir LoRA klasörüdür**:

```python
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
```

Bu (diğer dosyaların yanı sıra) şunu yazar:

| Dosya | Rol |
| --- | --- |
| `adapter_config.json` | Rank, alpha, hedef modüller, temel model id |
| `adapter_model.safetensors` | Eğitilmiş \(A,B\) ağırlıkları (onlarca ila yüzlerce MB, 16 GB değil) |
| tokenizer dosyaları + `chat_template.jinja` | Eğitimdekiyle aynı şekilde kodlama/çözme |

**Tam 8B modeli kaydetmediniz.** Yükleme anında hâlâ **aynı temele** ihtiyacınız vardır (`unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit` veya 16-bit Instruct orijinali) artı bu adaptör.

Yeniden yükleme kalıbı (sonraki oturum):

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ft_lora",  # adaptör dizini; Unsloth temel için adapter_config okur
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

Çıkarımda eğitimdekiyle **aynı sohbet şablonunu** kullanın. Unsloth'un 1 numaralı dışa aktarma hatası: Unsloth notebook harika görünür, Ollama sarhoş görünür — neredeyse her zaman şablon / EOS uyumsuzluğu.

Hugging Face yükleme, özel bir repo isterseniz `model.push_to_hub(...)` / `tokenizer.push_to_hub(...)`'dır; burada gerekli değil.

`lora_dir = "ft_lora"` göreli bir yoldur (proje klasörü). Yazdırma yolu doğrular. "Restored added_tokens_decoder metadata" satırı Unsloth'un tokenizer JSON'unu düzeltmesidir ki yeniden yükleme tutarlı kalsın.

```python
lora_dir = "ft_lora"
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
print(f"Saved LoRA adapters to {lora_dir}")
```

Örnek çıktı:

```
Unsloth: Restored added_tokens_decoder metadata in ft_lora/tokenizer_config.json.
Saved LoRA adapters to ft_lora
```

### Az önce ne görmüş olmalısınız?

`Saved LoRA adapters to ft_lora`. Bu klasör eğitimi sürdürmek veya Unsloth çıkarımı çalıştırmak için yeterlidir. GGUF yalnızca **diğer** çalışma zamanları içindir.

---

## 8. İsteğe bağlı: birleştir + GGUF dışa aktar (llama.cpp / Ollama)

### Bu hücre neden var?

LoRA adaptörleri [llama.cpp](https://github.com/ggml-org/llama.cpp), **Ollama** veya LM Studio'nun yüklediği şey değildir. Onlar bir **GGUF** dosyası ister: dondurulmuş ağırlıklar + LoRA'nız **birleştirilmiş**, sonra kuantize edilmiş.

```python
model.save_pretrained_gguf(
    "ft_cursor_gguf",
    tokenizer,
    quantization_method="q4_k_m",
    maximum_memory_usage=0.75,
)
```

Unsloth adaptörleri yoğun modele birleştirir, dönüştürür ve kuantize eder. **Yavaş** ve **VRAM-yoğundur**. Bu hücre `if False:` ile kapılıdır ki tam notebook "Run All" istemediğiniz bir birleştirme için takılmasın. GGUF gerektiğinde `if True:` yapın.

### Argümanlar

**`"ft_cursor_gguf"`** — `.gguf` dosya(lar)ı için çıktı dizini / önek.

**`quantization_method="q4_k_m"`** — Unsloth'un önerilen günlük kuantı: karışık 4-bit K-quant'lar, bazı attention/FFN tensörlerinde Q6_K. F16'dan daha küçük ve hızlı, eski `q4_0`'dan çok daha iyi. [GGUF dokümanlarından](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf) diğer yararlı değerler:

| Yöntem | Ne zaman |
| --- | --- |
| `q4_k_m` | Varsayılan dağıtım |
| `q5_k_m` | Ekstra kalite, daha büyük |
| `q8_0` | Ağır, fp16'ya daha yakın |
| `f16` | Arşiv / dönüştürme, kocaman |

**`maximum_memory_usage=0.75`** — kaydetme sırasında tepe GPU kullanımını VRAM'in %75'inde sınırla. Birleştirme **OOM** olursa `0.5` veya `0.4`'e düşürün.

### Dışa aktarmadan sonra

Ollama / llama.cpp'yi GGUF'a **ve** Llama 3.1 Instruct sohbet biçimlendirmesine yöneltin. Unsloth akışı iyi görünüyorsa ve GGUF bozuksa, rank'i değil şablonu düzeltin.

Ayrıca `push_to_hub_gguf("username/repo", tokenizer, quantization_method="q4_k_m")` da yapabilirsiniz.

Unsloth GGUF başarısız olursa elle yol: `save_pretrained_merged(..., save_method="merged_16bit")` sonra llama.cpp'den `convert_hf_to_gguf.py`.

`ft_lora`'daki adaptörler göndermeye razı olduğunuz cevaplar olana kadar bu hücreyi `False` bırakın.

```python
if False:
    model.save_pretrained_gguf(
        "ft_cursor_gguf",
        tokenizer,
        quantization_method="q4_k_m",
        maximum_memory_usage=0.75,
    )
```

---

## Ders özeti: aslında ne yaptınız?

Eksiksiz bir **QLoRA SFT** döngüsü çalıştırdınız:

1. Unsloth ile **dinamik 4-bit Llama 3.1 8B Instruct** yüklediniz (`max_seq_length=2048`, `dtype` otomatik-bf16).
2. Tüm attention + MLP projeksiyonlarına **LoRA r=16** taktınız (ağırlıkların ~%0.52'si).
3. **Alpaca JSONL** → Llama 3.1 **sohbet `text`** eşlediniz (canlı biçimlendirme fonksiyonu değil).
4. **%20** eval ayırdınız (208 / 52).
5. **TRL SFTTrainer** ile eğittiniz: etkin batch **8**, **2 epoch**, **52 adım**, **LR 2e-4**, kayıp **yalnızca asistan token'larında**, **packing kapalı**.
6. **`for_inference`** + sohbet şablonu **`add_generation_prompt=True`** ile ürettiniz.
7. **LoRA**'yı `ft_lora`'ya kaydettiniz. GGUF isteğe bağlı kaldı.

### Yeni bir projeye başlarken hatırlanacak fikirler

- **Veri > rank.** r=16 ve 2e-4 güçlü varsayılanlardır. Çöp JSONL r=128 ile kurtarılamaz.
- **Şablon eşleşmelidir** eğitimde, Unsloth generate'de ve Ollama'da.
- **Önce QLoRA**, en son tam FT.
- **`num_train_epochs` XOR `max_steps`**, asla ikisi birden.
- Sohbet SFT için **kullanıcı token'larını maskeleyin** (`train_on_responses_only`).
- **Adaptörleri hemen kaydedin**; çöken bir kernel bellek içi ağırlıkları düşürür.

### Mantıklı sonraki deneyler (her seferinde bir şeyi değiştirin)

- Generate `temperature`'ı 0.4'e düşürüp crafting tutarlılığını karşılaştırın.
- Gerçek nitel bir eval ekleyin: sizin puanladığınız 10 tutulmuş senaryo.
- `r`'ye dokunmadan önce `poe2_data.jsonl`'i daha zor craft'larla büyütün.
- Model hâlâ stok Llama gibi konuşuyorsa `lora_alpha=32` deneyin.
- Epoch 2'de eval kaybı kötüleştiyse 1 epoch'ta durun.

Sonraki bir notebook [Unsloth'un ince ayar rehberinden](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) saparsa, yeni API isimleri için rehbere güvenin; veri setinin sağlıklı olup olmadığı için **yazdırdığınız `text` örneğine** güvenin.
