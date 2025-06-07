# Türkçe Gemma 21B Fine-tuning Projesi

Bu proje, Google'ın güçlü **Gemma 21B** dil modelinin Türkçe becerilerini geliştirmek için bir yolculuğa çıkıyor. Amacımız, modelin Türkçe metinleri daha iyi anlamasını ve daha akıcı yanıtlar üretmesini sağlamak.

## Kullanılan Veri Setleri

Projemizde çeşitli açık kaynak veri setlerinden faydalanıyoruz. İşte projemizde kullandığımız veri setlerinin kısa bir özeti:

### 1. OSCAR (Open Super-large Crawled ALMAnaCH coRpus)

**Kaynak:** [Hugging Face Datasets: oscar-corpus/oscar](https://huggingface.co/datasets/oscar-corpus/oscar)

OSCAR, Common Crawl'dan elde edilmiş devasa, çok dilli bir metin korpusu. İçerdiği zengin Türkçe içeriğiyle bizim için önemli bir kaynak. Bu korpus, dil modellerini ve kelime temsillerini önceden eğitmek için harika bir başlangıç noktası. Her örnek genellikle basit bir metin içerir: `{"id": 0, "text": "metin örneği..."}`. Creative Commons CC0 lisansı altında yayınlanmıştır, yani kullanımı oldukça serbest.

### 2. Wikipedia (legacy-datasets/wikipedia)

**Kaynak:** [Hugging Face Datasets: legacy-datasets/wikipedia](https://huggingface.co/datasets/legacy-datasets/wikipedia)

Bu veri seti, tüm dillerdeki temizlenmiş Wikipedia makalelerini barındırıyor. Her makale, referanslar gibi istenmeyen kısımlardan arındırılarak özenle düzenlenmiş. Projemiz için tabii ki Türkçe Wikipedia makalelerini kullanacağız. Genellikle dil modelleme görevleri için tercih edilen bu veri setindeki her girdi, bir kimlik (`id`), makalenin URL'si (`url`), başlığı (`title`) ve tabii ki metin içeriği (`text`) gibi bilgileri içeriyor. Creative Commons Attribution-ShareAlike 3.0 Unported License (CC BY-SA) ve GNU Free Documentation License (GFDL) lisanslarıyla sunulmuştur.

### 3. OpenHermes-2.5

**Kaynak:** [Hugging Face Datasets: teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)

OpenHermes-2.5, genellikle GPT-4 tarafından üretilen ve çeşitli açık kaynaklardan derlenen yüksek kaliteli sentetik verilerle dolu bir veri seti. Temelde İngilizce içerikli olsa da, içindeki Türkçe konuşmaları değerlendiriyoruz. Ancak, Türkçe kısmının beklediğimiz kadar yoğun olmayabileceğini aklımızda tutmalıyız. Eğer yeterli Türkçe veri bulamazsak, farklı Türkçe instruction veri setleri keşfetmek iyi bir fikir olabilir. Veri yapısı `ShareGPT` formatında olup, konuşmalar `conversations` listesi içinde yer alır. Genellikle MIT lisansına sahip bileşenler içeriyor.

## Fine-tuning Yaklaşımımız

Bu projede, yukarıda bahsettiğimiz veri setlerinin yalnızca **Türkçe** bölümlerini kullanarak Google'ın **Gemma 21B** dil modelini ince ayar yapacağız. Bu sayede, modelin Türkçe metinleri daha iyi anlamasını ve daha akıcı yanıtlar üretmesini sağlamayı hedefliyoruz.

## Fizibilite Analizi

Bu projenin uygulanabilirliğini ve potansiyel değerini daha iyi anlamak için kısa bir fizibilite analizi yapalım:

### 1. Maliyet ve Zaman Tahmini

Runpod.io üzerinden saatlik 6.36 dolara kiraladığımız NVIDIA B200 GPU ile fine-tuning sürecinin maliyetini ve süresini tahmin edelim. Eğitim süresi, veri setlerinin büyüklüğüne, `max_seq_length`, `per_device_train_batch_size` ve `gradient_accumulation_steps` gibi parametrelere bağlı olarak değişecektir. Deneyimlerimize göre, bu boyutta bir modelin Türkçe veri setleri üzerinde birkaç epoch eğitilmesi:

*   **Tahmini Süre:** Ortalama 10 - 30 saat (veri setinin nihai boyutuna ve parametre ayarlarına göre değişebilir).
*   **Tahmini Maliyet:** 10 saat * 6.36$ = 63.60$ ile 30 saat * 6.36$ = 190.80$ arasında değişebilir.

Bu maliyet, güçlü bir modelin Türkçe yeteneklerini geliştirmek için oldukça makul sayılabilir.

### 2. Beklenen Faydalar

Bu fine-tuning projesi tamamlandığında, aşağıdaki önemli faydaları sağlamayı hedefliyoruz:

*   **Türkçe Dil Yeteneklerinde Gelişme:** Gemma 21B modelinin Türkçe metinleri anlama, analiz etme ve üretme becerileri önemli ölçüde artacak.
*   **Göreve Özgü Performans:** Özellikle Türkçe metin özetleme, çeviri, soru yanıtlama veya içerik üretimi gibi spesifik Türkçe görevlerde daha iyi performans sergileyecek.
*   **Özelleştirilmiş Çözümler İçin Temel:** Elde edilen model, belirli Türkçe uygulamalar veya projeler için güçlü bir temel oluşturacak.
*   **Kaynak Optimizasyonu:** LoRA gibi verimli yöntemler ve 4-bit nicemleme sayesinde, büyük bir modeli daha küçük bütçelerle ve daha hızlı bir şekilde eğitebilme imkanı.

### 3. Potansiyel Zorluklar

Her projenin olduğu gibi, bu fine-tuning sürecinin de bazı potansiyel zorlukları olabilir:

*   **Veri Kalitesi ve Miktarı:** Özellikle `OpenHermes-2.5` gibi İngilizce ağırlıklı veri setlerinden yeterli ve kaliteli Türkçe veri çıkarımı zorlayıcı olabilir. Kalitesiz veri, modelin performansını olumsuz etkileyebilir.
*   **Parametre Ayarları:** En iyi performansı ve hızı yakalamak için `per_device_train_batch_size`, `max_seq_length`, `lora_r` gibi eğitim parametreleriyle denemeler yapmak gerekebilir. Bu, başlangıçta biraz zaman alabilir.
*   **Model Boyutu:** Gemma 21B hala çok büyük bir model. Bazı beklenmedik bellek veya hesaplama kaynakları sorunları yaşanabilir, ancak güçlü B200 GPU bu riskleri minimize etmeye yardımcı olacaktır.

Genel olarak, bu projenin belirlenen hedeflere ulaşmak için güçlü bir potansiyele sahip olduğunu ve Runpod.io gibi platformların sunduğu donanım imkanlarıyla oldukça fizibil olduğunu düşünüyoruz.

## Sistem Gereksinimleri ve Donanım Özellikleri

Fine-tuning sürecini hızlandırmak ve verimli kılmak için güçlü bir donanıma ihtiyacımız var. Bu işlem için Runpod.io'dan kiraladığımız aşağıdaki sisteme güveniyoruz:

*   **Sağlayıcı:** Runpod.io
*   **Model:** NVIDIA B200
*   **Saatlik Maliyet:** 6.36 Dolar
*   **VRAM:** 180GB
*   **RAM:** 283GB
*   **vCPU:** 28

Bu güçlü donanım, Gemma 21B gibi büyük bir modeli Türkçe veri setlerimizle verimli bir şekilde eğitmek için harika bir temel sağlıyor.

## Gerekli Python Modülleri

Projemizin sorunsuz çalışması için bazı Python kütüphanelerine ihtiyacımız olacak. İşte o kütüphanelerin listesi ve ne işe yaradıkları:

| Modül                           | Açıklama                                                                |
| :------------------------------ | :---------------------------------------------------------------------- |
| `transformers`                  | Hugging Face'in model ve eğitim API'si                                  |
| `datasets`                      | Veri setlerini yüklemek ve üzerinde işlem yapmak için                   |
| `accelerate`                    | Çoklu GPU/TPU desteğiyle hızlı eğitim için                              |
| `peft`                          | (LoRA için) — Tam fine-tuning yerine daha verimli bir yaklaşım          |
| `bitsandbytes`                  | QLoRA/4-bit nicemleme (VRAM kullanımını azaltmak için, opsiyonel)       |
| `trl`                           | `SFTTrainer` veya `DPOTrainer` gibi özel trainer'lar                    |
| `wandb` veya `tensorboard`      | (Eğitim sürecini izlemek ve loglamak için, opsiyonel ama çok faydalı) |
| `scipy`, `numpy`, `tqdm`        | Çeşitli yardımcı araçlar                                                |
| `sentencepiece`                 | Bazı tokenizer'lar için gerekli (özellikle Gemma gibi modellerde)       |
| `einops`                        | Bazı model modülleri için                                               |
| `pyyaml`, `json`, `huggingface_hub` | Konfigürasyon dosyalarını yönetmek ve Hugging Face ile etkileşim için |

## Kurulum

Projemizi kurmak çok kolay! Gerekli tüm Python kütüphanelerini, hazırladığımız `requirements.txt` dosyasını kullanarak tek bir komutla bilgisayarınıza kurabilirsiniz:

```bash
pip install -r requirements.txt
```

Bu tek komut, ihtiyacınız olan her şeyi sizin için halledecek.

## Fine-tuning Adımları: Türkçe Gemma 21B Modelini Eğitme

Gemma 21B modelimizi Türkçe veri setlerimizle eğitime başlamak için atacağımız adımlar şunlar. Tüm eğitim mantığını `trainer.py` adlı bir Python dosyasına yerleştirdik.

### 1. `trainer.py` Dosyasını İnceleyin

Projenizin ana eğitim kodu `trainer.py` dosyasında yer almaktadır. Bu dosya, veri setlerimizi hazırlamaktan modelimizi kaydetmeye kadar her şeyi adım adım yönetiyor. Kendi ihtiyaçlarınıza göre ince ayarlar yapmak isterseniz, bu dosyayı dikkatlice gözden geçirebilirsiniz. Özellikle `model_id` değişkeninin kullandığınız Gemma modeline uygun olduğundan ve OpenHermes veri setinden beklediğiniz Türkçe içeriği gerçekten filtrelediğinden emin olmak önemli.

### 2. Eğitimi Başlatın

Runpod.io gibi çoklu GPU'ya sahip bir sistem kullanıyorsanız, `accelerate` kütüphanesi ile eğitimi başlatmak en akıllıca yol.

Eğitimi başlatmadan önce, `accelerate` için bir yapılandırma yapmamız gerekiyor. Bu sadece bir kerelik bir işlem:

```bash
accelerate config
```

Bu komut size birkaç soru soracak (kaç GPU kullanacağınız, eğitim tipi gibi). Sisteminizin özelliklerine göre doğru yanıtları verdiğinizden emin olun. Bellek yönetimi için `DeepSpeed` gibi gelişmiş seçenekleri de düşünebilirsiniz.

Yapılandırma tamamlandıktan sonra, fine-tuning işlemini aşağıdaki komutla başlatabilirsiniz:

```bash
accelerate launch trainer.py \
  --datasets oscar-corpus/oscar wikimedia/wikipedia teknium/OpenHermes-2.5 \
  --max_seq_length 2048 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 16 \
  --packing False
```

Yukarıdaki komutta, fine-tuning parametrelerini (`--max_seq_length`, `--num_train_epochs`, `--per_device_train_batch_size`, `--gradient_accumulation_steps`, `--learning_rate`, `--lora_r`, `--lora_alpha`, `--packing`) doğrudan komut satırından kontrol edebilirsiniz. İşte bazı önemli noktalar ve öneriler:

*   `--datasets`: Fine-tune etmek istediğiniz veri setlerinin Hugging Face ID'lerini boşlukla ayırarak belirtebilirsiniz. Eğer bu argümanı belirtmezseniz, kod varsayılan olarak OSCAR, Wikipedia ve OpenHermes-2.5'i kullanacaktır.
*   `--max_seq_length`: Modelin işleyeceği maksimum dizi uzunluğu. Daha uzun metinleri işlemek için bu değeri artırabilirsiniz, ancak VRAM kullanımını artıracağını unutmayın. Gemma 21B gibi bir model için 2048 veya 4096 makul başlangıç değerleridir. Unutmayın, ne kadar uzun olursa o kadar fazla VRAM kullanır.
*   `--num_train_epochs`: Eğitimin kaç dönem (epoch) süreceği. Genellikle 1 ila 3 epoch, iyi sonuçlar elde etmek için yeterli olabilir.
*   `--per_device_train_batch_size`: Her GPU üzerindeki eğitim batch boyutu. 180GB VRAM'li NVIDIA B200 gibi güçlü bir sistemde bu değeri deneyerek artırabilirsiniz. Daha büyük batch boyutları eğitimi hızlandırabilir.
*   `--gradient_accumulation_steps`: Gradyan biriktirme adımları. `per_device_train_batch_size` ile çarpılarak etkin global batch boyutunu oluşturur. Bellek kısıtlı olduğunda veya çok büyük batch boyutlarına ulaşmak istediğinizde kullanışlıdır.
*   `--learning_rate`: Öğrenme oranı. Genellikle `2e-4` veya `5e-5` gibi değerler tercih edilir. Deneme yanılma ile en iyi değeri bulabilirsiniz.
*   `--lora_r` ve `--lora_alpha`: LoRA (Low-Rank Adaptation) parametreleri. `r` (rank) ne kadar yüksek olursa, model o kadar fazla parametre öğrenir ve genellikle kalite artar, ancak bellek kullanımı da yükselir. `lora_alpha` ise `r` ile aynı veya iki katı olabilir. Bu değerleri Runpod.io sisteminizin kapasitesine göre ayarlayabilirsiniz.
*   `--packing`: `True` olarak ayarlandığında, veri setindeki kısa örnekler bir araya toplanarak daha verimli bir şekilde işlenir, bu da eğitimi hızlandırabilir. Ancak, bazı durumlarda bu, örnekler arasındaki bağlamı etkileyebilir, bu yüzden `False` bırakmak daha güvenli bir başlangıç olabilir.

`trainer.py` dosyasındaki veri setlerini ön işleme (`.map` fonksiyonu) adımında `num_proc=os.cpu_count()` ayarını ekleyerek, veri yükleme ve formatlama işlemini CPU çekirdeklerinizin tamamını kullanarak hızlandırdık. Bu, özellikle büyük veri setleriyle çalışırken önemli bir performans artışı sağlar.

Bu komut, `trainer.py` dosyasındaki eğitim sürecini ayarladığınız GPU'lar üzerinde başlatacak. Eğitim süresi, veri setinizin büyüklüğüne ve belirlediğiniz eğitim dönemi sayısına göre farklılık gösterecektir.

### 3. Modeli Kaydedin ve Kullanın

Eğitim bittiğinde, `trainer.py` betiği modelinizi ve tokenizer'ınızı otomatik olarak `fine_tuned_gemma_21b_tr` adlı klasöre kaydedecek. 

Modeli daha sonra denemek veya farklı bir uygulamaya entegre etmek isterseniz, `trainer.py` dosyasının sonunda küçük bir örnek kullanım kodu bulabilirsiniz. Bu örnek kod parçacığını kullanarak modelinizden ilk tahminleri alabilir veya kendi uygulamanıza kolayca dahil edebilirsiniz.

Bu yolda size başarılar ve keyifli bir fine-tuning deneyimi dileriz!
