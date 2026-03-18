import torch
from datasets import load_dataset
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import jiwer
import pandas as pd
import re
from tqdm import tqdm
from peft import PeftModel
import os

class TextNormalizer:
    """
    Custom Text Normalization Class (Phase 1 Optimization).
    Prevents the destructive basic normalizer from wiping out critical Indic diacritics and morphology.
    Latin scripts utilize standard BasicTextNormalizer, whereas Indic languages are gated behind 
    a whitespace-only strict reduction algorithm.
    """
    def __init__(self):
        self.basic_normalizer = BasicTextNormalizer()
        
    def __call__(self, text, lang):
        if lang.lower() in ["english", "french", "german"]:
            return self.basic_normalizer(text).strip()
        else:
            # Whitespace-only normalisation for Indic/Urdu (no decapitalizing or dropping diacritics)
            return re.sub(r'\s+', ' ', text).strip()

def evaluate_model(languages, base_model_id="openai/whisper-large-v3-turbo", peft_model_id="./whisper-large-v3-turbo-lora", num_samples=50):
    """
    Runs an exhaustive evaluation loop over requested datasets comparing identical inferences against
    normalized ground-truths. Merges LoRA weights automatically if found to ascertain post-training capability.
    
    Args:
        languages (dict): Top-level language configurations mapped to human-readable strings.
        base_model_id (str): HuggingFace identifier for foundation model.
        peft_model_id (str): Local path array for fine-tuned LoRA matrices.
        num_samples (int): Max ceiling of evaluation fragments pulled per language.
        
    Returns:
        pd.DataFrame: Metric distribution containing WER/CER per language.
    """
    device = "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
    # Keep torch_dtype=float32 on MPS (fp16 causes silent empty output bug on Apple Silicon)
    torch_dtype = torch.float32 if device == "mps" else (torch.float16 if torch.cuda.is_available() else torch.float32)
    
    print(f"Initializing base model {base_model_id} on {device} ({torch_dtype})...")
    
    if os.path.exists(peft_model_id):
        print(f"Loading LoRA PEFT adapter weights from {peft_model_id}...")
        processor = WhisperProcessor.from_pretrained(base_model_id, task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
        
        # Merge weights for fast inference
        model = PeftModel.from_pretrained(model, peft_model_id)
        model = model.merge_and_unload()
        model.to(device)
        model.to(torch_dtype)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            torch_dtype=torch_dtype,
            chunk_length_s=30,
        )
    else:
        print("No PEFT model found. Running zero-shot base pipeline...")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=base_model_id,
            device=device,
            torch_dtype=torch_dtype,
            chunk_length_s=30,
        )

    normalizer = TextNormalizer()
    results = []

    for lang_code, lang_name in languages.items():
        print(f"Evaluating {lang_name}...")
        try:
            ds = load_dataset("google/fleurs", lang_code, split="test", trust_remote_code=True)
            if num_samples:
                ds = ds.select(range(min(num_samples, len(ds))))
        except Exception as e:
            print(f"Could not load dataset for {lang_name}: {e}")
            continue
            
        references = []
        predictions = []

        for batch in tqdm(ds, desc=f"Processing {lang_name}"):
            audio_array = batch["audio"]["array"]
            sampling_rate = batch["audio"]["sampling_rate"]

            # Ensures decoder knows which script to generate
            try:
                pipe.tokenizer.set_prefix_tokens(language=lang_name.capitalize(), task="transcribe")
            except:
                pass
                
            # Beam search implementation (5 beams, ngram size 3) + correct language kwargs
            pred_text = pipe(
                {"array": audio_array, "sampling_rate": sampling_rate}, 
                generate_kwargs={
                    "language": lang_name, 
                    "task": "transcribe",
                    "num_beams": 5, 
                    "no_repeat_ngram_size": 3
                }
            )["text"]

            ref = batch["raw_transcription"]
            pred = pred_text
            
            ref = normalizer(ref, lang_name)
            pred = normalizer(pred, lang_name)
                
            if ref and pred:
                references.append(ref)
                predictions.append(pred)

        if references and predictions:
            wer = jiwer.wer(references, predictions)
            cer = jiwer.cer(references, predictions)
            print(f"WER for {lang_name}: {wer:.4f} | CER: {cer:.4f}")
            results.append({"Language": lang_name, "WER": wer, "CER": cer})
        else:
            print(f"Skipping {lang_name} due to empty references/predictions after normalization.")

    return pd.DataFrame(results)

if __name__ == "__main__":
    langs = {
        "hi_in": "hindi",
        "mr_in": "marathi",
        "bn_in": "bengali",
        "ur_pk": "urdu",
        "en_us": "english",
        "fr_fr": "french",
        "de_de": "german"
    }

    print("Running evaluation (Phase 1 + 3 settings)...")
    df = evaluate_model(langs, num_samples=50)
    df.to_csv("evaluation_results_finetuned.csv", index=False)
    
    print("\n--- Evaluation Results ---")
    print(df.to_markdown(index=False))
