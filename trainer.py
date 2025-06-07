import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 21B model with specified datasets.")
    parser.add_argument(
        "--datasets", 
        nargs='+', 
        default=["oscar-corpus/oscar", "wikimedia/wikipedia", "teknium/OpenHermes-2.5"], # Varsayılan veri setleri
        help="Fine-tune edilecek veri setlerinin Hugging Face ID'leri. Örn: --datasets oscar-corpus/oscar wikimedia/wikipedia"
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=2048, 
        help="Modelin işleyeceği maksimum dizi uzunluğu."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Eğitim dönemi sayısı."
    )
    parser.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=4, 
        help="Her GPU'daki batch boyutu. VRAM'e göre ayarlanmalı."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=8, 
        help="Gradient biriktirme adımları."
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-4, 
        help="Öğrenme oranı."
    )
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16, 
        help="LoRA rank'ı. Daha yüksek değerler daha iyi kalite ama daha fazla bellek kullanır."
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=16, 
        help="LoRA alpha değeri."
    )
    parser.add_argument(
        "--packing", 
        type=bool, 
        default=False, 
        help="Veri setini paketleyip paketlemeyeceği. True daha hızlı eğitim sağlayabilir, ancak bazı durumlarda konteksti etkileyebilir."
    )

    args = parser.parse_args()

    loaded_datasets = []

    for dataset_name in args.datasets:
        print(f"{dataset_name} veri seti yükleniyor...")
        try:
            if dataset_name == "oscar-corpus/oscar":
                dataset = load_dataset("oscar-corpus/oscar", "unshuffled_deduplicated_tr", split="train")
                print(f"OSCAR Türkçe veri seti boyutu: {len(dataset)} örnek")
                loaded_datasets.append(dataset)
            elif dataset_name == "wikimedia/wikipedia":
                try:
                    dataset = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train")
                except ValueError:
                    print("Doğrudan '20231101.tr' bulunamadı, 'tr' dil kodunu deneyin.")
                    dataset = load_dataset("wikimedia/wikipedia", language="tr", date="20231101", split="train")
                print(f"Wikipedia Türkçe veri seti boyutu: {len(dataset)} örnek")
                loaded_datasets.append(dataset)
            elif dataset_name == "teknium/OpenHermes-2.5":
                openhermes_raw = load_dataset("teknium/OpenHermes-2.5", split="train")
                openhermes_tr = openhermes_raw.filter(lambda x: any("türkçe" in msg['value'].lower() for msg in x['conversations']))
                print(f"OpenHermes-2.5 Türkçe (filtreli) veri seti boyutu: {len(openhermes_tr)} örnek")
                if len(openhermes_tr) > 100: 
                    loaded_datasets.append(openhermes_tr)
                else:
                    print("OpenHermes-2.5'ten yeterli Türkçe veri bulunamadı, bu veri seti dahil edilmedi.")
            else:
                print(f"Desteklenmeyen veri seti: {dataset_name}. Lütfen tanımlı veri setlerinden birini seçin.")
        except Exception as e:
            print(f"Hata oluştu {dataset_name} yüklenirken: {e}")

    if not loaded_datasets:
        print("Hiçbir veri seti başarıyla yüklenemedi. Lütfen geçerli veri seti ID'leri sağlayın.")
        return

    combined_dataset = concatenate_datasets(loaded_datasets)
    print(f"Toplam birleştirilmiş Türkçe veri seti boyutu: {len(combined_dataset)} örnek")

    def format_data(example):
        if 'text' in example:
            return {"text": example["text"]}
        elif 'conversations' in example:
            formatted_text = ""
            for turn in example['conversations']:
                if turn['from'] == 'human':
                    formatted_text += f"<|start_header_id|>user<|end_header_id|>{turn['value']}<|eot_id|>"
                elif turn['from'] == 'gpt':
                    formatted_text += f"<|start_header_id|>assistant<|end_header_id|>{turn['value']}<|eot_id|>"
            return {"text": formatted_text + "<|end_of_text|>"}
        return {"text": ""}

    formatted_dataset = combined_dataset.map(format_data, remove_columns=combined_dataset.column_names, num_proc=os.cpu_count())
    print("Veri seti formatlandı.")

    model_id = "google/gemma-2-15b" 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    print(f"{model_id} modeli yükleniyor...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Model ve Tokenizer başarıyla yüklendi.")

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        disable_tqdm=False,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=args.packing,
        # num_workers'ı artırmak veri yükleme hızını artırabilir. CPU çekirdek sayınıza ayarlayın.
        # data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # Gerekirse custom collator
    )

    print("Fine-tuning başlatılıyor...")
    trainer.train()
    print("Fine-tuning tamamlandı!")

    output_dir = "./fine_tuned_gemma_21b_tr"
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model kaydedildi: {output_dir}")

    print("\n\n--- Model Testi ---")
    model.eval()
    prompt = "Türkiye'nin başkenti neresidir?"
    chat = [
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**input_ids, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Soru: {prompt}")
    print(f"Cevap: {response}")

if __name__ == "__main__":
    main() 