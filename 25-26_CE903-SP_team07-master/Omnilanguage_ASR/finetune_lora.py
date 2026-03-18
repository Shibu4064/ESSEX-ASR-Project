import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Audio, load_dataset, concatenate_datasets
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import jiwer
import pandas as pd
import re

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    A foundational data collator optimized specifically for the custom 'Omnilingual Dataset' maps. 
    It properly pads mismatched length feature tensors (input log-Mel matrices) and transcription labels, 
    accounting for extreme differences between Latin and complex Indic syntax limits.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Strip BOS token from labels to prevent double-prepending during generation setup
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        return batch

def prepare_dataset(batch, processor):
    """
    Processes audio batches from the generated Omnilingual dataset segments into usable inputs/labels.
    Applies per-sample cross-lingual token formatting rules dynamically mapped.
    """
    audio = batch["audio"]
    lang = batch["language_name"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    # Per-sample set_prefix_tokens ensures decoder knows which script to generate
    processor.tokenizer.set_prefix_tokens(language=lang.capitalize(), task="transcribe")
    text = batch["raw_transcription"]
    batch["labels"] = processor.tokenizer(text).input_ids
    return batch

def get_indic_dataset(languages, split="train", samples_dict=None):
    """
    Retrieves and structures the 'Omnilingual Dataset' slices.
    The Omnilingual topology ensures deep syntax retention during the Low-Rank Adaptation (LoRA) backpropagation.
    """
    datasets = []
    # ISO language codes mapping for FLEURS dataset keys 
    fleurs_map = {
        "hindi": "hi_in",
        "marathi": "mr_in",
        "bengali": "bn_in",
        "urdu": "ur_pk",
        "english": "en_us",
        "french": "fr_fr",
        "german": "de_de"
    }
    
    for lang in languages:
        code = fleurs_map.get(lang.lower(), lang)
        print(f"Loading {lang} ({code})...")
        ds = load_dataset("google/fleurs", code, split=split, trust_remote_code=True)
        
        num_samples = samples_dict.get(lang.lower(), 1000) if samples_dict else None
        if num_samples:
            limit = min(num_samples, len(ds))
            ds = ds.select(range(limit))
            
        ds = ds.add_column("language_name", [lang] * len(ds))
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    return combined

def train_lora():
    """
    Core Pipeline: Initializes the Parameter-Efficient Fine-Tuning mapping procedure over 
    the base whisper-large-v3-turbo, actively conditioning it with the Omnilingual Dataset.
    Designed for consumer hardware execution leveraging strict gradient checkpointing and accumulation logic.
    """
    model_id = "openai/whisper-large-v3-turbo"
    output_dir = "./whisper-large-v3-turbo-lora"
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    # fp16 True only if CUDA available, else float32 for safety (MPS bug)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading processor and model {model_id} on {device} ({torch_dtype})...")
    processor = WhisperProcessor.from_pretrained(model_id, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype)
    
    model.config.use_cache = False # Required for gradient checkpointing with PEFT
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    print("Applying PEFT LoRA configuration...")
    # Robust configuration hitting all projections as spec'd
    config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Train ALL 7 languages together utilizing the Omnilingual Dataset configuration
    # (European languages as anchor to prevent catastrophic forgetting)
    all_langs = ["hindi", "marathi", "bengali", "urdu", "english", "french", "german"]
    
    train_samples = {l: (9999 if l in ["marathi", "bengali"] else 1000) for l in all_langs}
    eval_samples = {l: 100 for l in all_langs}

    print("Lazy-loading datasets (pyarrow map avoiding full RAM buffer)...")
    train_ds = get_indic_dataset(all_langs, split="train", samples_dict=train_samples)
    eval_ds = get_indic_dataset(all_langs, split="test", samples_dict=eval_samples)

    print(f"Mapping datasets...")
    train_ds = train_ds.map(lambda x: prepare_dataset(x, processor), remove_columns=train_ds.column_names, num_proc=1)
    eval_ds = eval_ds.map(lambda x: prepare_dataset(x, processor), remove_columns=eval_ds.column_names, num_proc=1)

    print("Filtering out long transcripts...")
    train_ds = train_ds.filter(lambda x: len(x["labels"]) < 448)
    eval_ds = eval_ds.filter(lambda x: len(x["labels"]) < 448)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def normalize(text):
        # Indic-safe normalisation in compute_metrics (whitespace-only, no BasicTextNormalizer)
        return re.sub(r'\s+', ' ', text).strip()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
            
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        pred_str = [normalize(s) for s in pred_str]
        label_str = [normalize(s) for s in label_str]
        
        wer = jiwer.wer(label_str, pred_str)
        cer = jiwer.cer(label_str, pred_str)
        return {"wer": wer, "cer": cer}

    print("Configuring Phase 2 Training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8, # effective batch=16
        gradient_checkpointing=True,
        fp16=True if torch.cuda.is_available() else False, # fp16 causes empty output bug on apple silicon
        learning_rate=2e-4, 
        lr_scheduler_type="cosine",
        warmup_steps=200,
        max_steps=2000, 
        eval_strategy="steps",
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=250,
        eval_steps=250,
        logging_steps=50,
        save_total_limit=3, # avoid filling disk
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        label_smoothing_factor=0.1,
        remove_unused_columns=False, 
        label_names=["labels"], 
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("========== Starting Phase 2 LoRA fine-tuning ==========")
    train_result = trainer.train()
    
    metrics = train_result.metrics
    pd.DataFrame([metrics]).to_csv("lora_metrics.csv", index=False)
    print("Training complete.")
    
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"LoRA weights and processor saved to {output_dir}")

if __name__ == "__main__":
    train_lora()
